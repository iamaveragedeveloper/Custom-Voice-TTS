import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .audio import mel_spectrogram
from .config import AcousticConfig, AudioConfig, VocoderConfig, to_dict
from .data import AcousticDataset, WavSegments, load_meta
from .models.acoustic import AcousticModel
from .models.vocoder import Discriminators, Generator, d_loss, fm_loss, g_loss


def _dev(device):
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _load_partial(model, state):
    own = model.state_dict()
    ok = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    skipped = [k for k in state if k not in ok]
    model.load_state_dict(ok, strict=False)
    if skipped:
        print("init: skipped (shape mismatch/missing):", skipped)


def train_acoustic(data, out, steps=20000, max_frames=10000, lr=1e-3, device=None, init=None, resume=True,
                   amp=False, log_every=50, save_every=500, cfg: AcousticConfig = None):
    dev = _dev(device)
    meta = load_meta(data)
    audio = AudioConfig(**meta["audio"])
    cfg = cfg or AcousticConfig()
    cfg.n_speakers = max(len(meta["speakers"]), 1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ds = AcousticDataset(data)
    model = AcousticModel(cfg, audio.n_mels, meta["n_vocab"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=1e-6)
    warm = max(min(1000, steps // 10), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, 0.5 * (1 + math.cos(math.pi * min(s / steps, 1))) * 0.95 + 0.05))
    use_amp = amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step = 0
    last = out / "acoustic_last.pt"
    if resume and last.exists():
        ck = torch.load(last, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step = ck["step"]
        print("resumed at step", step)
    elif init:
        _load_partial(model, torch.load(init, map_location=dev)["model"])
        print("initialised from", init)
    print(f"acoustic: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, {len(ds)} utts, device={dev}")

    def save(path):
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step,
                        cfg=to_dict(cfg), audio=to_dict(audio), n_vocab=meta["n_vocab"], stats=meta["stats"],
                        phonemes=meta["phonemes"], speakers=meta["speakers"]), path)

    model.train()
    t0, acc = time.time(), {}
    while step < steps:
        for ids in ds.batches(max_frames):
            b = {k: v.to(dev) for k, v in ds.collate(ids).items()}
            bin_w = min(max((step / steps - 0.1) / 0.3, 0.0), 1.0)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                loss, parts = model(b, bin_w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            for k, v in parts.items():
                acc[k] = acc.get(k, 0) + float(v)
            if step % log_every == 0:
                print(f"step {step}/{steps} " + " ".join(f"{k}={v / log_every:.3f}" for k, v in acc.items())
                      + f" {(time.time() - t0) / log_every:.2f}s/it")
                acc, t0 = {}, time.time()
            if step % save_every == 0:
                save(last)
            if step >= steps:
                break
    save(last)
    save(out / "acoustic.pt")
    print("saved", out / "acoustic.pt")


def train_vocoder(data, out, steps=100000, batch=8, seg=8192, lr=2e-4, device=None, init=None, resume=True,
                  small=False, log_every=50, save_every=2000, workers=0, g_only_steps=0):
    dev = _dev(device)
    meta = load_meta(data)
    audio = AudioConfig(**meta["audio"])
    vc = VocoderConfig(channels=128 if small else 256)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    G, D = Generator(vc, audio.n_mels).to(dev), Discriminators().to(dev)
    og = torch.optim.AdamW(G.parameters(), lr, betas=(0.8, 0.99))
    od = torch.optim.AdamW(D.parameters(), lr, betas=(0.8, 0.99))
    step = 0
    last = out / "vocoder_last.pt"
    if resume and last.exists():
        ck = torch.load(last, map_location=dev)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"]); og.load_state_dict(ck["og"]); od.load_state_dict(ck["od"])
        step = ck["step"]
        print("resumed at step", step)
    elif init:
        ck = torch.load(init, map_location=dev)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"]) if "D" in ck else None
        print("initialised from", init)
    dl = DataLoader(WavSegments(data, audio.sr, seg), batch_size=batch, shuffle=True, drop_last=True,
                    num_workers=workers)
    print(f"vocoder: G {sum(p.numel() for p in G.parameters()) / 1e6:.1f}M params, device={dev}")

    def save(path):
        torch.save(dict(G=G.state_dict(), D=D.state_dict(), og=og.state_dict(), od=od.state_dict(), step=step,
                        cfg=to_dict(vc), audio=to_dict(audio)), path)

    t0 = time.time()
    while step < steps:
        for y in dl:
            y = y.to(dev)
            mel = mel_spectrogram(y, audio)
            yg = G(mel)
            n = min(yg.shape[-1], y.shape[-1])
            y_, yg = y[:, None, :n], yg[..., :n]
            gan = step >= g_only_steps  # generator-only warm-up: mel loss alone, ~6x cheaper per step
            ld = torch.zeros(())
            if gan:
                od.zero_grad()
                r, g, _, _ = D(y_, yg.detach())
                ld = d_loss(r, g)
                ld.backward(); od.step()

            og.zero_grad()
            l_mel = torch.nn.functional.l1_loss(mel_spectrogram(yg[:, 0], audio), mel)
            lg = 45 * l_mel
            if gan:
                r, g, fr, fg = D(y_, yg)
                lg = lg + 2 * fm_loss(fr, fg) + g_loss(g)
            lg.backward(); og.step()
            step += 1
            if step % log_every == 0:
                print(f"step {step}/{steps} mel={l_mel.item():.3f} D={ld.item():.3f} G={lg.item():.3f} "
                      f"{(time.time() - t0) / log_every:.2f}s/it")
                t0 = time.time()
            if step % save_every == 0:
                save(last)
            if step >= steps:
                break
    save(last)
    torch.save(dict(G=G.state_dict(), cfg=to_dict(vc), audio=to_dict(audio), step=step), out / "vocoder.pt")
    print("saved", out / "vocoder.pt")
