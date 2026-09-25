"""Ultron-style voice effect chain (pure DSP, real-time capable).

ultron(): optional deeper pitch layer + short metallic comb resonance + optional soft saturation + small room,
then an optional *matching EQ* (build_eq) that pulls the overall tone towards a reference recording (the film clip).
"""
import numpy as np
import torch
import torchaudio.functional as AF
from scipy.signal import butter, fftconvolve, firwin2, hilbert, iirnotch, istft, sosfilt, sosfiltfilt, stft, tf2sos, welch


def _comb_fast(y, sr, ms, fb, taps=6):
    """Feed-forward approximation of a feedback comb (fb^k taps): metallic ringing, no Python loop."""
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


# ---------------------------------------------------------------- matching EQ
_GRID = np.geomspace(50, 10500, 220)  # log-frequency grid (Hz)


def _ltas_db(y, sr):
    f, p = welch(y, sr, nperseg=4096)
    return 10 * np.log10(np.interp(_GRID, f, p) + 1e-14)


def build_eq(target, sources, sr, max_boost=14.0, max_cut=-8.0, smooth_oct=1 / 3):
    """Gain curve (dB on _GRID) that moves the long-term average spectrum of `sources` (list of arrays)
    to that of `target`. Levels are normalised in the 150-3000 Hz region so only the *shape* changes."""
    tgt = _ltas_db(target, sr)
    src = np.mean([_ltas_db(s, sr) for s in sources], axis=0)
    g = tgt - src
    w = max(int(round(smooth_oct * len(_GRID) / np.log2(_GRID[-1] / _GRID[0]))), 1)
    g = np.convolve(np.pad(g, w, mode="edge"), np.ones(2 * w + 1) / (2 * w + 1), mode="valid")
    mid = (_GRID > 150) & (_GRID < 3000)
    g = np.clip(g - g[mid].mean(), max_cut, max_boost)
    return g


def apply_eq(y, sr, gain_db, taps=2049):
    """Zero-phase FIR EQ from a gain curve on _GRID."""
    f = np.concatenate([[0.0], _GRID, [sr / 2]])
    g = np.concatenate([[gain_db[0]], gain_db, [gain_db[-1]]])
    h = firwin2(taps, f / (sr / 2), 10 ** (g / 20))
    return fftconvolve(y, h, mode="same").astype(np.float32)


# ---------------------------------------------------------------- noise reduction
def denoise(y, sr, alpha=1.5, floor_db=-20.0, quiet_pct=12, n_fft=1024, hop=256):
    """Spectral-subtraction noise reduction. The noise spectrum is learned from the quietest non-silent
    frames of the clip itself (vocoder hiss/buzz), subtracted with strength `alpha`, gains are smoothed in
    time and frequency to avoid 'musical noise', and never drop below floor_db."""
    _, _, X = stft(y, sr, nperseg=n_fft, noverlap=n_fft - hop)
    mag = np.abs(X)
    e = mag.mean(0)
    live = e > 1e-5
    if live.sum() < 8:
        return y
    q = live & (e <= np.percentile(e[live], quiet_pct))
    noise = mag[:, q].mean(1, keepdims=True)
    gain = np.clip((mag - alpha * noise) / (mag + 1e-9), 10 ** (floor_db / 20), 1.0)
    k = np.array([0.25, 0.5, 0.25])  # smooth over time, then frequency
    gain = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, gain)
    gain = np.apply_along_axis(lambda c: np.convolve(c, np.ones(5) / 5, mode="same"), 0, gain)
    _, out = istft(X * gain, sr, nperseg=n_fft, noverlap=n_fft - hop)
    out = out[: len(y)]
    return np.pad(out, (0, len(y) - len(out))).astype(np.float32)


# ---------------------------------------------------------------- effect chain
def ultron(y, sr, intensity=1.0, down=-3.0, metal_ms=4.5, metal=0.45, sat=2.2, room=0.12, lowpass=7500, eq=None, follow_silence=True, clean=0.0):
    """y: float32 mono. intensity scales the effect. down=0 skips the deeper layer. lowpass=None keeps the top end.
    eq: gain curve from build_eq (applied last, so it corrects the tone of the whole chain).
    clean: noise-reduction strength (0 = off, 1.5 = moderate, 2.5 = strong)."""
    y = y.astype(np.float32)
    if down != 0:
        lo = AF.pitch_shift(torch.from_numpy(y)[None], sr, down * intensity)[0].numpy()  # deeper layer
        lo = lo[: len(y)] if len(lo) >= len(y) else np.pad(lo, (0, len(y) - len(lo)))
        x = 0.55 * y + 0.75 * lo
    else:
        x = y.copy()
    if metal > 0:
        x = x + metal * intensity * (_comb_fast(x, sr, metal_ms, 0.7) - x)  # metallic resonance
    if sat > 0:
        x = np.tanh(sat * intensity * x) / np.tanh(sat * intensity)  # grit (0 = off)
    band = [70, lowpass] if lowpass else 70
    x = sosfilt(butter(2, band, "bandpass" if lowpass else "highpass", fs=sr, output="sos"), x)
    x = _room(x, sr, level=room * intensity)
    if eq is not None:
        x = apply_eq(x, sr, eq)
    if clean > 0:  # noise reduction AFTER the EQ, so the boosted highs are cleaned too
        x = denoise(x, sr, alpha=clean)
    if follow_silence:  # digital silence in the dry voice stays silent after the chain
        hop = int(sr * 0.005)
        n = len(y) // hop
        live = (np.abs(y[: n * hop]).reshape(n, hop).max(1) > 1e-4).astype(np.float32)
        live = np.convolve(live, np.ones(5), mode="same") > 0  # +/-10 ms guard around real audio
        m = np.repeat(live.astype(np.float32), hop)
        m = np.pad(m, (0, len(x) - len(m)))[: len(x)]
        k = int(sr * 0.012) | 1
        x = x * np.convolve(m, np.hanning(k) / np.hanning(k).sum(), mode="same")
    return (x / (np.abs(x).max() + 1e-9) * 0.9).astype(np.float32)


def depulse(y, sr, hop=256, harmonics=(1, 2, 3, 4, 5, 6), q=12.0, bands=((1200, 3000), (3000, 6000), (6000, 10000))):
    """EXPERIMENTAL / ineffective: envelope-notch attempt at removing the vocoder frame-rate buzz. Kept for reference;
    the buzz is a vocoder training issue (see README)."""
    y = y.astype(np.float64)
    rate = sr / hop
    out = y.copy()
    for lo, hi in bands:
        hi = min(hi, sr / 2 * 0.98)
        band = sosfiltfilt(butter(4, [lo, hi], "bandpass", fs=sr, output="sos"), y)
        env = np.abs(hilbert(band)) + 1e-6
        e2 = env
        for k in harmonics:
            f = rate * k
            if f < sr / 2 * 0.98:
                b, a = iirnotch(f, q, fs=sr)
                e2 = sosfiltfilt(tf2sos(b, a), e2)
        gain = np.clip(np.maximum(e2, 0.2 * env) / env, 0.3, 1.0)
        out += band * (gain - 1.0)
    return out.astype(np.float32)
