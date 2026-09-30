"""Correctness check: formant_shift must leave pitch (F0) essentially unchanged while moving the
spectral envelope. Also times it for the real-time budget."""
import time
import numpy as np
import torch
from scipy.signal import welch

from vtts.audio import estimate_f0, load_wav
from vtts.config import AudioConfig
from vtts.fx import formant_shift

c = AudioConfig()
y = load_wav("data/base/wavs/ultron_10min_0003.wav", c.sr)
y = y[: len(y) // c.hop * c.hop]
yt = torch.from_numpy(y)


def f0_med(y):
    f0 = estimate_f0(torch.from_numpy(y[: len(y) // c.hop * c.hop]), c)
    v = f0[f0 > 0]
    return float(v.median()) if len(v) else float("nan")


def spectral_centroid(y):
    f, p = welch(y, c.sr, nperseg=2048)
    return float((f * p).sum() / p.sum())


base_f0 = f0_med(y)
base_c = spectral_centroid(y)
print(f"original: F0 median {base_f0:.0f} Hz, spectral centroid {base_c:.0f} Hz")
for r in (0.80, 0.90, 1.0, 1.15):
    t0 = time.perf_counter()
    z = formant_shift(y, c.sr, ratio=r)
    dt = time.perf_counter() - t0
    print(f"ratio={r:.2f}: F0 median {f0_med(z):6.0f} Hz (delta {f0_med(z) - base_f0:+5.1f}), "
          f"centroid {spectral_centroid(z):6.0f} Hz (delta {spectral_centroid(z) - base_c:+6.0f}), "
          f"{dt * 1000:.0f} ms for {len(y) / c.sr:.1f}s audio")
