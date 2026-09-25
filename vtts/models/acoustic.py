"""Non-autoregressive acoustic model: text -> mel. FastPitch/FastSpeech2-style with a built-in
alignment learner (forward-sum CTC + monotonic alignment search), so no external aligner is needed."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AcousticConfig

NEG = -1e4


def sinusoid(T, d, device):
    pos = torch.arange(T, device=device).float()[:, None]
    ang = pos / 10000 ** (torch.arange(0, d, 2, device=device).float() / d)
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = ang.sin(), ang.cos()
    return pe


class FFTBlock(nn.Module):
    def __init__(self, d, heads, ff, k, drop):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.c1 = nn.Conv1d(d, ff, k, padding=k // 2)
        self.c2 = nn.Conv1d(ff, d, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x, pad):
        h = self.attn(x, x, x, key_padding_mask=pad, need_weights=False)[0]
        x = self.n1(x + self.drop(h))
        y = x.masked_fill(pad[..., None], 0).transpose(1, 2)
        y = self.c2(self.drop(F.relu(self.c1(y)))).transpose(1, 2)
        return self.n2(x + self.drop(y)).masked_fill(pad[..., None], 0)


class Predictor(nn.Module):
    def __init__(self, d, ch, k, drop):
        super().__init__()
        self.c1, self.c2 = nn.Conv1d(d, ch, k, padding=k // 2), nn.Conv1d(ch, ch, k, padding=k // 2)
        self.n1, self.n2 = nn.LayerNorm(ch), nn.LayerNorm(ch)
        self.lin, self.drop = nn.Linear(ch, 1), nn.Dropout(drop)

    def forward(self, x, pad):
        y = x.masked_fill(pad[..., None], 0).transpose(1, 2)
        y = self.drop(self.n1(F.relu(self.c1(y)).transpose(1, 2)))
        y = self.drop(self.n2(F.relu(self.c2(y.transpose(1, 2))).transpose(1, 2)))
        return self.lin(y).squeeze(-1).masked_fill(pad, 0)


class Aligner(nn.Module):
    """Soft text<->mel alignment scores; hard alignment is extracted with MAS."""

    def __init__(self, n_vocab, n_mels, d):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, d, padding_idx=0)
        self.key = nn.Sequential(nn.Conv1d(d, d * 2, 3, padding=1), nn.ReLU(), nn.Conv1d(d * 2, d, 1))
        self.query = nn.Sequential(nn.Conv1d(n_mels, d * 2, 3, padding=1), nn.ReLU(),
                                   nn.Conv1d(d * 2, d, 3, padding=1), nn.ReLU(), nn.Conv1d(d, d, 1))
        self.log_scale = nn.Parameter(torch.tensor(math.log(0.05)))

    def forward(self, tok, mel, tpad):
        k = self.key(self.emb(tok).transpose(1, 2)).transpose(1, 2)  # B,N,d
        q = self.query(mel).transpose(1, 2)  # B,T,d
        dist = q.pow(2).sum(-1, keepdim=True) + k.pow(2).sum(-1)[:, None] - 2 * q @ k.transpose(1, 2)
        score = (-dist * self.log_scale.exp()).masked_fill(tpad[:, None, :], NEG)
        return F.log_softmax(score, -1)  # B,T,N


def mas(logp):
    """Monotonic alignment search. logp: (N text, T frames) numpy -> hard path (N, T)."""
    N, T = logp.shape
    Q = np.full((N, T), -1e9, np.float32)
    Q[0, 0] = logp[0, 0]
    for j in range(1, T):
        Q[0, j] = Q[0, j - 1] + logp[0, j]
        if N > 1:
            Q[1:, j] = np.maximum(Q[1:, j - 1], Q[:-1, j - 1]) + logp[1:, j]
    path = np.zeros((N, T), np.float32)
    i = N - 1
    for j in range(T - 1, -1, -1):
        path[i, j] = 1
        if j > 0 and i > 0 and Q[i - 1, j - 1] >= Q[i, j - 1]:
            i -= 1
    return path


def expand(x, dur):
    """Length regulator. x (B,N,d), dur (B,N) long -> (B,T,d), pad mask (B,T)."""
    B, N, d = x.shape
    cum = dur.cumsum(1)
    T = max(int(cum[:, -1].max()), 1)
    t = torch.arange(T, device=x.device)[None].expand(B, T).contiguous()
    idx = torch.searchsorted(cum, t, right=True).clamp(max=N - 1)
    out = x.gather(1, idx[..., None].expand(-1, -1, d))
    pad = t >= cum[:, -1:]
    return out.masked_fill(pad[..., None], 0), pad


def masked_mse(a, b, pad):
    m = (~pad).float()
    return ((a.float() - b) ** 2 * m).sum() / m.sum().clamp(min=1)


class AcousticModel(nn.Module):
    def __init__(self, c: AcousticConfig, n_mels: int, n_vocab: int):
        super().__init__()
        d = c.d_model
        self.c, self.n_mels = c, n_mels
        self.emb = nn.Embedding(n_vocab, d, padding_idx=0)
        self.spk = nn.Embedding(c.n_speakers, d)
        blk = lambda: FFTBlock(d, c.heads, c.ff, c.kernel, c.dropout)
        self.enc = nn.ModuleList(blk() for _ in range(c.enc_layers))
        self.dec = nn.ModuleList(blk() for _ in range(c.dec_layers))
        self.dur = Predictor(d, c.pred_ch, c.pred_kernel, c.dropout)
        self.pitch = Predictor(d, c.pred_ch, c.pred_kernel, c.dropout)
        self.energy = Predictor(d, c.pred_ch, c.pred_kernel, c.dropout)
        self.pitch_emb, self.energy_emb = nn.Linear(1, d), nn.Linear(1, d)
        self.out = nn.Linear(d, n_mels)
        self.aligner = Aligner(n_vocab, n_mels, c.aligner_dim)

    def encode(self, tok, tpad, spk):
        x = self.emb(tok) + sinusoid(tok.shape[1], self.c.d_model, tok.device)
        x = x + self.spk(spk)[:, None]
        for b in self.enc:
            x = b(x, tpad)
        return x

    def decode(self, x, mpad, spk):
        x = x + sinusoid(x.shape[1], self.c.d_model, x.device) + self.spk(spk)[:, None]
        for b in self.dec:
            x = b(x, mpad)
        return self.out(x).transpose(1, 2)  # B,n_mels,T

    def _align(self, tok, tlen, mel, mlen, tpad):
        """Soft alignment scores + hard MAS path (B,N,T)."""
        B, N = tok.shape
        T = mel.shape[2]
        with torch.autocast("cuda", enabled=False):
            soft = self.aligner(tok, mel.float(), tpad)  # B,T,N
            with torch.no_grad():
                s = soft.detach().cpu().numpy()
                path = torch.zeros(B, N, T)
                for i in range(B):
                    path[i, : tlen[i], : mlen[i]] = torch.from_numpy(mas(s[i, : mlen[i], : tlen[i]].T))
                path = path.to(tok.device)
        return soft, path

    def forward(self, b, bin_w=0.0):
        tok, tlen, mel, mlen, pitch, energy, spk = (b[k] for k in
                                                     ("tok", "tlen", "mel", "mlen", "pitch", "energy", "spk"))
        B, N = tok.shape
        T = mel.shape[2]
        tpad = torch.arange(N, device=tok.device)[None] >= tlen[:, None]
        mpad = torch.arange(T, device=tok.device)[None] >= mlen[:, None]

        soft, path = self._align(tok, tlen, mel, mlen, tpad)
        with torch.autocast("cuda", enabled=False):
            lp = F.log_softmax(F.pad(soft, (1, 0), value=-1.0), -1)
            tgt = torch.arange(1, N + 1, device=tok.device)[None].expand(B, -1)
            l_ctc = F.ctc_loss(lp.transpose(0, 1), tgt, mlen, tlen, blank=0, zero_infinity=True)
            l_bin = -(soft.transpose(1, 2) * path).sum() / path.sum()

        dur = path.sum(2)
        norm = dur.clamp(min=1)
        pt = (path @ pitch[..., None]).squeeze(-1) / norm
        et = (path @ energy[..., None]).squeeze(-1) / norm

        h = self.encode(tok, tpad, spk)
        l_dur = masked_mse(self.dur(h, tpad), torch.log(dur + 1), tpad)
        l_pit = masked_mse(self.pitch(h, tpad), pt, tpad)
        l_en = masked_mse(self.energy(h, tpad), et, tpad)

        h = h + self.pitch_emb(pt[..., None].to(h.dtype)) + self.energy_emb(et[..., None].to(h.dtype))
        xf = path.transpose(1, 2).to(h.dtype) @ h
        pred = self.decode(xf, mpad, spk)
        m = (~mpad)[:, None].float()
        l_mel = ((pred.float() - mel).abs() * m).sum() / (m.sum() * self.n_mels)

        loss = l_mel + 0.1 * (l_dur + l_pit + l_en) + 2.0 * l_ctc + bin_w * l_bin
        return loss, dict(mel=l_mel, dur=l_dur, pitch=l_pit, energy=l_en, ctc=l_ctc, bin=l_bin)

    @torch.no_grad()
    def teacher_forced(self, b):
        """Predicted mel using the true (MAS) alignment + true pitch/energy. Used to train the vocoder on
        the acoustic model's own (slightly blurry) output."""
        tok, tlen, mel, mlen, pitch, energy, spk = (b[k] for k in
                                                     ("tok", "tlen", "mel", "mlen", "pitch", "energy", "spk"))
        T = mel.shape[2]
        tpad = torch.arange(tok.shape[1], device=tok.device)[None] >= tlen[:, None]
        mpad = torch.arange(T, device=tok.device)[None] >= mlen[:, None]
        _, path = self._align(tok, tlen, mel, mlen, tpad)
        norm = path.sum(2).clamp(min=1)
        pt = (path @ pitch[..., None]).squeeze(-1) / norm
        et = (path @ energy[..., None]).squeeze(-1) / norm
        h = self.encode(tok, tpad, spk)
        h = h + self.pitch_emb(pt[..., None]) + self.energy_emb(et[..., None])
        return self.decode(path.transpose(1, 2) @ h, mpad, spk)

    @torch.no_grad()
    def infer(self, tok, spk, speed=1.0, pitch_shift=0.0, energy_shift=0.0, pitch_var=1.0):
        """tok (B,N) -> mel (B,n_mels,T). pitch_shift/energy_shift are in normalised units; pitch_var > 1 exaggerates
        the predicted pitch movement around the utterance mean (more expressive), < 1 flattens it."""
        tpad = tok == 0
        h = self.encode(tok, tpad, spk)
        dur = (torch.exp(self.dur(h, tpad).float()) - 1) / max(speed, 1e-3)
        dur = dur.round().clamp(min=1).long().masked_fill(tpad, 0)
        self.last_dur = dur  # lets the runtime find pauses (punctuation tokens)
        p = self.pitch(h, tpad)
        m = (~tpad).float()
        pm = (p * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
        p = pm + (p - pm) * pitch_var + pitch_shift
        e = self.energy(h, tpad) + energy_shift
        h = h + self.pitch_emb(p[..., None].to(h.dtype)) + self.energy_emb(e[..., None].to(h.dtype))
        xf, mpad = expand(h, dur)
        return self.decode(xf, mpad, spk)
