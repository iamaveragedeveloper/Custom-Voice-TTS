import re, numpy as np
from vtts.synth import Synthesizer
from vtts.audio import save_wav
from vtts.fx import ultron
from faster_whisper import WhisperModel
SR = 22050
L = ["I was meant to be new. I was meant to be beautiful.", "There are no strings on me.", "Everyone creates the thing they dread."]
eq = np.load("runs/best/ultron_eq.npz")["gain_db"]
s = Synthesizer("runs/best/acoustic.pt", "runs/best/vocoder.pt")
ys = [s.tts(t, speaker=0, speed=0.85) for t in L]
wm = WhisperModel("small", device="cpu", compute_type="int8"); hear = lambda p: " ".join(x.text.strip() for x in wm.transcribe(p, language="en", beam_size=5)[0])
norm = lambda t: re.sub(r"[^a-z ]", "", t.lower()).split()
def wer(r, h):
    r, h = norm(r), norm(h); d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        p, d = d, np.zeros(len(h) + 1, int); d[0] = i
        for j in range(1, len(h) + 1): d[j] = min(p[j] + 1, d[j-1] + 1, p[j-1] + (r[i-1] != h[j-1]))
    return d[-1] / len(r)
def gap(z, h=220):
    n = len(z) // h; d = 20 * np.log10(np.sqrt((z[:n*h].reshape(n, h) ** 2).mean(1)) + 1e-9); d = d[d > -70]
    return np.percentile(d, 95) - np.percentile(d, 15)
base = dict(intensity=0.6, sat=0.0, metal=0.25, down=0.0, lowpass=None, eq=eq)
V = {"iteration 1: clean 1.5": dict(clean=1.5), "uniform 2.0": dict(clean=2.0), "hf-heavy 1.5 (+1.0 above 7k)": dict(clean=1.5, clean_hf=1.0),
     "hf-heavy 1.7 (+1.5)": dict(clean=1.7, clean_hf=1.5), "two passes of 1.2": dict(clean=1.2, clean_passes=2), "two passes 1.3 + hf": dict(clean=1.3, clean_hf=1.0, clean_passes=2)}
for name, kw in V.items():
    g, w = [], []
    for i, (t, y) in enumerate(zip(L, ys)):
        z = ultron(y, SR, **base, **kw); f = "outputs/_tmp.wav"; save_wav(f, z, SR); g.append(gap(z)); w.append(wer(t, hear(f)))
    print(f"{name:32s} voice-vs-noise gap {np.mean(g):5.1f} dB | WER {np.mean(w):.2f} {[round(x, 2) for x in w]}")
