import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from .config import AudioConfig

_cache = {}


def load_wav(path, sr):
    x, orig = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(1)
    if orig != sr:
        x = torchaudio.functional.resample(torch.from_numpy(x), orig, sr).numpy()
    return x


def save_wav(path, y, sr):
    sf.write(str(path), np.clip(y, -1, 1), sr, subtype="PCM_16")


def _basis(c: AudioConfig, device):
    key = (c.sr, c.n_fft, c.n_mels, c.fmin, c.fmax, str(device))
    if key not in _cache:
        fb = torchaudio.functional.melscale_fbanks(c.n_fft // 2 + 1, c.fmin, c.fmax, c.n_mels, c.sr,
                                                   norm="slaney", mel_scale="slaney")
        _cache[key] = (fb.T.contiguous().to(device), torch.hann_window(c.win).to(device))
    return _cache[key]


def mel_spectrogram(y, c: AudioConfig):
    """y: (B, T) float in [-1, 1] -> log-mel (B, n_mels, T // hop)."""
    basis, win = _basis(c, y.device)
    pad = (c.n_fft - c.hop) // 2
    y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    s = torch.stft(y, c.n_fft, c.hop, c.win, win, center=False, return_complex=True)
    mag = torch.sqrt(s.real ** 2 + s.imag ** 2 + 1e-9)
    return torch.log(torch.clamp(basis @ mag, min=1e-5))


def estimate_f0(y, c: AudioConfig, fmin=40.0, fmax=400.0, thr=0.25):
    """Autocorrelation F0 per mel frame (0 = unvoiced). y: (T,) with T % hop == 0."""
    pad = (c.n_fft - c.hop) // 2
    y = F.pad(y[None, None], (pad, pad), mode="reflect")[0, 0]
    fr = y.unfold(0, c.n_fft, c.hop)
    rms = fr.pow(2).mean(1).sqrt()
    fr = (fr - fr.mean(1, keepdim=True)) * torch.hann_window(c.n_fft)
    n = 2 * c.n_fft
    ac = torch.fft.irfft(torch.fft.rfft(fr, n=n).abs() ** 2, n=n)[:, : c.n_fft]
    ac = ac / (ac[:, :1] + 1e-8)
    lo, hi = int(c.sr / fmax), int(c.sr / fmin)
    seg = ac[:, lo - 1: hi + 1]
    is_peak = (seg[:, 1:-1] > seg[:, :-2]) & (seg[:, 1:-1] >= seg[:, 2:])  # true local maxima only
    cand = torch.where(is_peak, seg[:, 1:-1], torch.full_like(seg[:, 1:-1], -1.0))
    peak = cand.max(1).values
    # first (shortest-lag) peak within 90% of the best avoids octave-down errors; skip picks inside low-lag ramp
    idx = (cand >= 0.9 * peak[:, None]).float().argmax(1)
    f0 = c.sr / (idx + lo).float()
    return torch.where((peak > thr) & (rms > 2e-3), f0, torch.zeros_like(f0))


def griffin_lim(mel, c: AudioConfig, iters=48):
    """Vocoder-free preview of a log-mel (n_mels, T) -> waveform. Low quality, for debugging."""
    basis, _ = _basis(c, mel.device)
    lin = torch.linalg.pinv(basis) @ torch.exp(mel)
    gl = torchaudio.transforms.GriffinLim(c.n_fft, iters, c.win, c.hop, power=1.0).to(mel.device)
    return gl(lin.clamp(min=1e-5))
