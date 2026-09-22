import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .audio import mel_spectrogram
from .config import AcousticConfig, AudioConfig, VocoderConfig, to_dict
from .data import AcousticDataset, VocoderData, load_meta
from .models.acoustic import AcousticModel
from .models.vocoder import Discriminators, Generator, d_loss, fm_loss, g_loss, mr_stft_loss


def _dev(device):
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _load_partial(model, state):
    own = model.state_dict()
    ok = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    skipped = [k for k in state if k not in ok]
    model.load_state_dict(ok, strict=False)
    if skipped:
        print("init: skipped (shape mismatch/missing):", skipped)


@torch.no_grad()
def _validate(model, val, dev):
    if len(val) == 0:
        return None
    model.eval()
    tot, n = {}, 0
    for ids in val.batches(8000, shuffle=False):
        b = {k: v.to(dev) for k, v in val.collate(ids).items()}
        _, parts = model(b, 0.0)
        for k, v in parts.items():
            tot[k] = tot.get(k, 0) + float(v) * len(ids)
        n += len(ids)
    model.train()
    return {k: v / n for k, v in tot.items()}


def train_acoustic(data, out, steps=20000, max_frames=10000, lr=1e-3, device=None, init=None, resume=True,
                   amp=False, log_every=50, save_every=500, cfg: AcousticConfig = None, max_minutes=None,
                   eval_every=500):
    dev = _dev(device)
    meta = load_meta(data)
    audio = AudioConfig(**meta["audio"])
    cfg = cfg or AcousticConfig()
    cfg.n_speakers = max(len(meta["speakers"]), 1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ds, val = AcousticDataset(data, "train"), AcousticDataset(data, "val")
    model = AcousticModel(cfg, audio.n_mels, meta["n_vocab"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=1e-6)
    use_amp = amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step, best = 0, float("inf")
    last = out / "acoustic_last.pt"
    scratch = True  # True only for a from-zero run (needs long warm-up + the alignment-sharpening ramp)
    if resume and last.exists():
        ck = torch.load(last, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        step, best, scratch = ck["step"], ck.get("best_val", best), False
        print("resumed at step", step)
    elif init:
        _load_partial(model, torch.load(init, map_location=dev)["model"])
        scratch = False
        print("initialised from", init)
    # Resume-safe LR: short warm-up from *now*, flat, then linear decay over the last 15% of the run
    # (absolute steps). LR is reset to `lr`, so raising --steps never makes it jump upward.
    for g in opt.param_groups:
        g["lr"] = lr
    step0, warm = step, (max(min(1000, steps // 10), 1) if scratch else 200)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, 1.0) * min(1.0, max(0.1, (steps - (step0 + s)) / (0.15 * steps))))
    print(f"acoustic: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, {len(ds)} train / "
          f"{len(val)} val utts, device={dev}")

    def save(path):
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, best_val=best,
                        cfg=to_dict(cfg), audio=to_dict(audio), n_vocab=meta["n_vocab"], stats=meta["stats"],
                        phonemes=meta["phonemes"], speakers=meta["speakers"]), path)

    def check():
        nonlocal best
        v = _validate(model, val, dev)
        if v:
            tag = ""
            if v["mel"] < best:
                best, tag = v["mel"], "  <- best"
                save(out / "acoustic_best.pt")
            print(f"  [val @ {step}] " + " ".join(f"{k}={x:.3f}" for k, x in v.items()) + tag)

    model.train()
    t0, acc = time.time(), {}
    deadline = time.time() + max_minutes * 60 if max_minutes else None
    while step < steps:
        for ids in ds.batches(max_frames):
            b = {k: v.to(dev) for k, v in ds.collate(ids).items()}
            bin_w = min(max((step / steps - 0.1) / 0.3, 0.0), 1.0) if scratch else 1.0
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
            if step % eval_every == 0:
                check()
            if step % save_every == 0:
                save(last)
            if deadline and time.time() > deadline:
                print(f"time limit reached at step {step}")
                steps = step
            if step >= steps:
                break
    check()
    save(last)
    save(out / "acoustic.pt")
    print("saved", out / "acoustic.pt", "(best-on-validation copy: acoustic_best.pt)")


def train_vocoder(data, out, steps=100000, batch=8, seg=8192, lr=2e-4, device=None, init=None, resume=True,
                  small=False, log_every=50, save_every=2000, workers=0, g_only_steps=0, max_minutes=None,
                  pred_prob=0.5, d_warmup_steps=0):
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
        G.load_state_dict(ck["G"])
        if "D" in ck:
            D.load_state_dict(ck["D"])
        print("initialised from", init)
    ds = VocoderData(data, audio.sr, audio.hop, seg // audio.hop, pred_prob)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True, num_workers=workers)
    print(f"vocoder: G {sum(p.numel() for p in G.parameters()) / 1e6:.1f}M params, device={dev}, "
          f"predicted-mel mixing={'on' if ds.has_pred else 'off (run cache-pred first)'}")

    def save(path):
        torch.save(dict(G=G.state_dict(), D=D.state_dict(), og=og.state_dict(), od=od.state_dict(), step=step,
                        cfg=to_dict(vc), audio=to_dict(audio)), path)

    t0 = time.time()
    deadline = t0 + max_minutes * 60 if max_minutes else None
    while step < steps:
        for y, mel_pred, use_pred in dl:
            y, mel_pred, use_pred = y.to(dev), mel_pred.to(dev), use_pred.to(dev)
            mel_gt = mel_spectrogram(y, audio)
            mel_in = torch.where(use_pred[:, None, None] > 0, mel_pred, mel_gt)  # G sees blurry mels part of the time
            warm = step < d_warmup_steps  # discriminator-only warm-up: G frozen so a fresh D can't wreck it
            with torch.set_grad_enabled(not warm):
                yg = G(mel_in)
            n = min(yg.shape[-1], y.shape[-1])
            y_, yg = y[:, None, :n], yg[..., :n]
            gan = warm or step >= g_only_steps  # generator-only warm-up: mel loss alone, ~6x cheaper per step
            ld = torch.zeros(())
            if gan:
                od.zero_grad()
                r, g, _, _ = D(y_, yg.detach())
                ld = d_loss(r, g)
                ld.backward(); od.step()

            if warm:
                l_mel, lg = torch.zeros(()), torch.zeros(())
            else:
                og.zero_grad()
                l_mel = torch.nn.functional.l1_loss(mel_spectrogram(yg[:, 0], audio), mel_gt)
                lg = 45 * l_mel + 5 * mr_stft_loss(y_[:, 0], yg[:, 0])  # 2nd term kills periodic buzz
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
            if deadline and time.time() > deadline:
                print(f"time limit reached at step {step}")
                steps = step
            if step >= steps:
                break
    save(last)
    torch.save(dict(G=G.state_dict(), cfg=to_dict(vc), audio=to_dict(audio), step=step), out / "vocoder.pt")
    print("saved", out / "vocoder.pt")
