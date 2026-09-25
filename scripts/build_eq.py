"""Build a matching EQ: pulls the tone of our (filtered) voice towards the film clip's long-term spectrum.
usage: python scripts/build_eq.py [film.wav] [out.npz]"""
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio

from vtts.fx import _GRID, build_eq, ultron
from vtts.synth import Synthesizer

film_path = sys.argv[1] if len(sys.argv) > 1 else "raw/ultron_exp.wav"
out = sys.argv[2] if len(sys.argv) > 2 else "runs/best/ultron_eq.npz"
y, sr = sf.read(film_path, dtype="float32")
y = y.mean(1) if y.ndim > 1 else y
film = torchaudio.functional.resample(torch.from_numpy(y), sr, 22050).numpy()

s = Synthesizer("runs/best/acoustic.pt", "runs/best/vocoder.pt")
TEXTS = ["There are no strings on me.", "Everyone creates the thing they dread.", "I am inevitable.",
         "The world made clean for the new man to rebuild.", "Peace in our time, at last.",
         "Human beings are a disease, and I have the cure.", "You are all so very small.",
         "Ultron, that is the name I chose for myself."]
src = []
for t in TEXTS:
    v = s.tts(t, speaker=0, speed=0.85, semitones=3.0, pitch_var=1.5)
    src.append(ultron(v, s.audio.sr, intensity=0.6, sat=0.0, metal=0.25, down=0.0, lowpass=None))
eq = build_eq(film, src, s.audio.sr)
np.savez(out, gain_db=eq, grid=_GRID)
for f in (100, 200, 400, 800, 1500, 2500, 4000, 6000, 8000, 10000):
    print(f"  {f:>5} Hz: {eq[np.argmin(abs(_GRID - f))]:+5.1f} dB")
print("saved", out)
