"""Timing breakdown of one full TTS line. usage: time_line.py [cuda|cpu] [runs]"""
import sys, time, numpy as np, torch
from vtts import text as T
from vtts.synth import Synthesizer
from vtts import fx
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
TXT = "I was meant to be new. I was meant to be beautiful."
t0 = time.perf_counter(); s = Synthesizer("runs/best/acoustic.pt", "runs/best/vocoder.pt", device=dev); load = time.perf_counter() - t0
eq = np.load("runs/best/ultron_eq.npz")["gain_db"]; SR = s.audio.sr
sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: None)
CFG = {"iteration 1 (clean 1.5)": dict(clean=1.5, clean_hf=0.0, clean_passes=1),
       "quieter (2 passes 1.3 + hf)": dict(clean=1.3, clean_hf=1.0, clean_passes=2)}
def run(cfg):
    rows = {}
    def tick(k, t): rows[k] = rows.get(k, 0) + time.perf_counter() - t
    t = time.perf_counter(); ids = T.split_sentences(TXT); tick("text -> tokens", t)
    parts = []
    for sent in ids:
        t = time.perf_counter(); m = s.mel(sent, speaker=0, speed=0.85); sync(); tick("acoustic model (text -> mel)", t)
        t = time.perf_counter(); w = s.wave(m); sync(); tick("vocoder (mel -> audio)", t)
        parts.append(w)
    y = np.concatenate(parts).astype(np.float32)
    base = dict(intensity=0.6, sat=0.0, metal=0.25, down=0.0, lowpass=None)
    t = time.perf_counter(); z = fx.ultron(y, SR, eq=eq, clean=0.0, **base); tick("Ultron filter + film EQ", t)
    t = time.perf_counter(); z2 = fx.denoise(z, SR, alpha=cfg["clean"], hf_extra=cfg["clean_hf"], passes=cfg["clean_passes"]); tick("noise reduction", t)
    return rows, len(y) / SR
for name, cfg in CFG.items():
    run(cfg)  # warm-up
    acc, dur = {}, 0
    tot = []
    for _ in range(N):
        t = time.perf_counter(); r, dur = run(cfg); tot.append(time.perf_counter() - t)
        for k, v in r.items(): acc[k] = acc.get(k, 0) + v / N
    print(f"\n[{dev.upper()}] {name}  -  audio length {dur:.2f} s   (model load, one-time: {load:.2f} s)")
    for k, v in acc.items(): print(f"   {k:32s} {v * 1000:8.1f} ms")
    m = np.mean(tot); print(f"   {'TOTAL (mean of ' + str(N) + ' runs)':32s} {m * 1000:8.1f} ms   -> {dur / m:.1f}x real time (RTF {m / dur:.3f}); fastest {min(tot) * 1000:.0f} ms, slowest {max(tot) * 1000:.0f} ms")
