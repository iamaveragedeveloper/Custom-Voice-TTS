import re
import sys

import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, hilbert, sosfilt, welch

from vtts.audio import estimate_f0, save_wav
from vtts.config import AudioConfig
from vtts.fx import ultron
from vtts.synth import Synthesizer
from faster_whisper import WhisperModel  # after torch (cuDNN clash otherwise)

TEXT = sys.argv[1] if len(sys.argv) > 1 else "I was meant to be new. I was meant to be beautiful."
REF = "I was meant to be new I was meant to be beautiful"
SR = 22050
eq = np.load("runs/best/ultron_eq.npz")["gain_db"]
s = Synthesizer("runs/best/acoustic.pt", "runs/best/vocoder.pt")
wm = WhisperModel("small", device="cpu", compute_type="int8")
hear = lambda p: " ".join(x.text.strip() for x in wm.transcribe(p, language="en", beam_size=5)[0])
norm = lambda t: re.sub(r"[^a-z ]", "", t.lower()).split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        p, d = d, np.zeros(len(h) + 1, int); d[0] = i
        for j in range(1, len(h) + 1):
            d[j] = min(p[j] + 1, d[j - 1] + 1, p[j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1] / len(r)


def hf_db(y):  # energy above 5 kHz relative to total (film clip: about -17 dB)
    f, p = welch(y, SR, nperseg=4096)
    return 10 * np.log10(p[f > 5000].sum() / p[(f > 60) & (f < 10500)].sum())


def pulse(y):
    b = sosfilt(butter(4, [1500, 10000], "bandpass", fs=SR, output="sos"), y)
    env = np.abs(hilbert(b)); env -= env.mean()
    S = np.abs(np.fft.rfft(env * np.hanning(len(env)))); fr = np.fft.rfftfreq(len(env), 1 / SR)
    return max(S[abs(fr - 86.13 * k) < 3].max() for k in (3, 4)) / S[(fr > 20) & (fr < 400)].mean()


def f0(y):
    v = estimate_f0(torch.from_numpy(y[: len(y) // 256 * 256]), AudioConfig())
    v = v[v > 0].numpy()
    return f"{np.median(v):.0f} Hz ({np.percentile(v, 5):.0f}-{np.percentile(v, 95):.0f})"


ORD = dict(intensity=0.6, sat=0.0, metal=0.25)
V = {  # name: (synth kwargs, fx kwargs)
    "0_old_presetB":        (dict(),                                ORD),
    "1_new_eq":             (dict(),                                dict(ORD, down=0.0, lowpass=None, eq=eq)),
    "2_new_eq_pitch+3":     (dict(semitones=3, pitch_var=1.5),      dict(ORD, down=0.0, lowpass=None, eq=eq)),
    "3_new_eq_pitch+5":     (dict(semitones=5, pitch_var=1.8),      dict(ORD, down=0.0, lowpass=None, eq=eq)),
}
film, _ = sf.read("raw/ultron_exp.wav", dtype="float32")
print(f"film clip reference: energy>5kHz = -17 dB (approx), pitch median 97 Hz (58-380), pulse n/a")
for n, (sk, fk) in V.items():
    y = s.tts(TEXT, speaker=0, speed=0.85, **sk)
    z = ultron(y, SR, **fk)
    path = f"outputs/line_{n}.wav"; save_wav(path, z, SR)
    print(f"{n:20s} >5kHz={hf_db(z):6.1f} dB  pitch {f0(y):22s} pulse={pulse(z):5.1f}  "
          f"WER={wer(REF, hear(path)):.2f}  heard: {hear(path)[:70]}")
