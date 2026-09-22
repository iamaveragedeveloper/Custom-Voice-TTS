"""HiFi-GAN vocoder (generator + multi-period / multi-scale discriminators), written from scratch."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from ..config import VocoderConfig

LR = 0.1


def init_w(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d)):
        m.weight.data.normal_(0.0, 0.01)


class ResBlock(nn.Module):
    def __init__(self, ch, k, dils):
        super().__init__()
        self.c1 = nn.ModuleList(weight_norm(nn.Conv1d(ch, ch, k, 1, dilation=d, padding=(k * d - d) // 2)) for d in dils)
        self.c2 = nn.ModuleList(weight_norm(nn.Conv1d(ch, ch, k, 1, padding=(k - 1) // 2)) for _ in dils)
        self.apply(init_w)

    def forward(self, x):
        for a, b in zip(self.c1, self.c2):
            x = x + b(F.leaky_relu(a(F.leaky_relu(x, LR)), LR))
        return x


class Generator(nn.Module):
    def __init__(self, c: VocoderConfig, n_mels=80):
        super().__init__()
        self.pre = weight_norm(nn.Conv1d(n_mels, c.channels, 7, 1, 3))
        self.ups, self.res = nn.ModuleList(), nn.ModuleList()
        ch = c.channels
        for i, (u, k) in enumerate(zip(c.upsample_rates, c.upsample_kernels)):
            self.ups.append(weight_norm(nn.ConvTranspose1d(ch // 2 ** i, ch // 2 ** (i + 1), k, u, (k - u) // 2)))
            for rk, rd in zip(c.resblock_kernels, c.resblock_dilations):
                self.res.append(ResBlock(ch // 2 ** (i + 1), rk, rd))
        self.nk = len(c.resblock_kernels)
        self.post = weight_norm(nn.Conv1d(ch // 2 ** len(c.upsample_rates), 1, 7, 1, 3))
        self.apply(init_w)

    def forward(self, mel):
        x = self.pre(mel)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, LR))
            x = sum(self.res[i * self.nk + j](x) for j in range(self.nk)) / self.nk
        return torch.tanh(self.post(F.leaky_relu(x)))

    def strip_norm(self):
        for m in self.modules():
            try:
                remove_weight_norm(m)
            except ValueError:
                pass
        return self


class DiscP(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.p = period
        ch = [1, 32, 128, 256, 512]
        self.convs = nn.ModuleList([weight_norm(nn.Conv2d(ch[i], ch[i + 1], (5, 1), (3, 1), (2, 0))) for i in range(4)]
                                   + [weight_norm(nn.Conv2d(512, 512, (5, 1), 1, (2, 0)))])
        self.post = weight_norm(nn.Conv2d(512, 1, (3, 1), 1, (1, 0)))

    def forward(self, x):
        b, c, t = x.shape
        if t % self.p:
            x = F.pad(x, (0, self.p - t % self.p), "reflect")
            t = x.shape[-1]
        x = x.view(b, c, t // self.p, self.p)
        fm = []
        for cv in self.convs:
            x = F.leaky_relu(cv(x), LR)
            fm.append(x)
        x = self.post(x)
        fm.append(x)
        return x.flatten(1), fm


class DiscS(nn.Module):
    def __init__(self, spectral=False):
        super().__init__()
        n = spectral_norm if spectral else weight_norm
        cfg = [(1, 64, 15, 1, 7, 1), (64, 64, 41, 2, 20, 4), (64, 128, 41, 2, 20, 16), (128, 256, 41, 4, 20, 16),
               (256, 512, 41, 4, 20, 16), (512, 512, 41, 1, 20, 16), (512, 512, 5, 1, 2, 1)]
        self.convs = nn.ModuleList(n(nn.Conv1d(i, o, k, s, p, groups=g)) for i, o, k, s, p, g in cfg)
        self.post = n(nn.Conv1d(512, 1, 3, 1, 1))

    def forward(self, x):
        fm = []
        for cv in self.convs:
            x = F.leaky_relu(cv(x), LR)
            fm.append(x)
        x = self.post(x)
        fm.append(x)
        return x.flatten(1), fm


def _run(ds, y, yh, pool=None):
    r, g, fr, fg = [], [], [], []
    for i, d in enumerate(ds):
        if pool is not None and i:
            y, yh = pool(y), pool(yh)
        a, b = d(y), d(yh)
        r.append(a[0]); g.append(b[0]); fr.append(a[1]); fg.append(b[1])
    return r, g, fr, fg


class Discriminators(nn.Module):
    """MPD + MSD. forward(real, fake) -> (real_scores, fake_scores, real_fmaps, fake_fmaps)."""

    def __init__(self):
        super().__init__()
        self.mpd = nn.ModuleList(DiscP(p) for p in (2, 3, 5, 7, 11))
        self.msd = nn.ModuleList([DiscS(True), DiscS(), DiscS()])
        self.pool = nn.AvgPool1d(4, 2, 2)

    def forward(self, y, yh):
        a = _run(self.mpd, y, yh)
        b = _run(self.msd, y, yh, self.pool)
        return tuple(x + z for x, z in zip(a, b))


def d_loss(real, fake):
    return sum(((1 - r) ** 2).mean() + (g ** 2).mean() for r, g in zip(real, fake))


def g_loss(fake):
    return sum(((1 - g) ** 2).mean() for g in fake)


def fm_loss(fr, fg):
    return sum((a - b).abs().mean() for x, y in zip(fr, fg) for a, b in zip(x, y))


def mr_stft_loss(y, yg, cfgs=((512, 128), (1024, 256), (2048, 512))):
    """Multi-resolution STFT loss (spectral convergence + log-magnitude). Penalises the periodic
    frame-rate buzz that a single mel L1 loss lets through. y, yg: (B, T)."""
    def mag(x, n, h):
        st = torch.stft(x, n, h, n, torch.hann_window(n, device=x.device), return_complex=True)
        return torch.sqrt(st.real ** 2 + st.imag ** 2 + 1e-9)
    tot = 0.0
    for n, h in cfgs:
        a, b = mag(y, n, h), mag(yg, n, h)
        tot = tot + torch.norm(a - b) / torch.norm(a) + F.l1_loss(torch.log(a), torch.log(b))
    return tot / len(cfgs)
