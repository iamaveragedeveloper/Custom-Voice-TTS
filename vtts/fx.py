"""Ultron-style voice effect chain (pure DSP, real-time capable):
pitch-down layer + short metallic comb resonance + soft saturation + small room, mixed with the dry voice."""
import numpy as np
import torch
import torchaudio.functional as AF
from scipy.signal import butter, fftconvolve, hilbert, iirnotch, sosfilt, sosfiltfilt, tf2sos


def _comb(y, sr, ms, fb):
    d = max(int(sr * ms / 1000), 1)
    out = y.copy()
    for i in range(d, len(y)):  # feedback comb -> metallic ringing; vectorised per block below for speed
        out[i] += fb * out[i - d]
    return out


def _comb_fast(y, sr, ms, fb, taps=6):
    """Feed-forward approximation of the feedback comb (fb^k taps): no Python loop."""
    d = max(int(sr * ms / 1000), 1)
    out = y.copy()
    for k in range(1, taps + 1):
        out[k * d:] += (fb ** k) * y[: len(y) - k * d]
    return out


def _room(y, sr, secs=0.35, level=0.12, seed=0):
    n = int(sr * secs)
    ir = np.random.default_rng(seed).standard_normal(n) * np.exp(-np.linspace(0, 7, n))
    ir[0] = 0
    return y + level * fftconvolve(y, ir / np.abs(ir).sum() * 3, mode="full")[: len(y)]


def ultron(y, sr, intensity=1.0, down=-3.0, metal_ms=4.5, metal=0.45, sat=2.2, room=0.12):
    """y: float32 mono. intensity 0..1.5 scales how strong the effect is."""
    y = y.astype(np.float32)
    lo = AF.pitch_shift(torch.from_numpy(y)[None], sr, down * intensity)[0].numpy()  # deeper layer
    lo = lo[: len(y)] if len(lo) >= len(y) else np.pad(lo, (0, len(y) - len(lo)))
    x = 0.55 * y + 0.75 * lo
    if metal > 0:
        x = x + metal * intensity * (_comb_fast(x, sr, metal_ms, 0.7) - x)  # metallic resonance
    if sat > 0:
        x = np.tanh(sat * intensity * x) / np.tanh(sat * intensity)  # grit (0 = off)
    x = sosfilt(butter(2, [90, 7500], "bandpass", fs=sr, output="sos"), x)
    x = _room(x, sr, level=room * intensity)
    return (x / (np.abs(x).max() + 1e-9) * 0.9).astype(np.float32)


def depulse(y, sr, hop=256, harmonics=(1, 2, 3, 4, 5, 6), q=12.0, bands=((1200, 3000), (3000, 6000), (6000, 10000))):
    """Remove the vocoder's frame-rate buzz: in each high band, notch the amplitude envelope at k * sr/hop
    (k = 1..6) and re-apply the smoothed gain. Speech dynamics (< ~60 Hz) are left untouched."""
    y = y.astype(np.float64)
    rate = sr / hop
    out = y.copy()
    for lo, hi in bands:
        hi = min(hi, sr / 2 * 0.98)
        sos = butter(4, [lo, hi], "bandpass", fs=sr, output="sos")
        band = sosfiltfilt(sos, y)
        env = np.abs(hilbert(band)) + 1e-6
        e2 = env
        for k in harmonics:
            f = rate * k
            if f < sr / 2 * 0.98:
                b, a = iirnotch(f, q, fs=sr)
                e2 = sosfiltfilt(tf2sos(b, a), e2)
        gain = np.clip(np.maximum(e2, 0.2 * env) / env, 0.3, 1.0)  # only ever reduce the pulse peaks
        out += band * (gain - 1.0)
    return out.astype(np.float32)
