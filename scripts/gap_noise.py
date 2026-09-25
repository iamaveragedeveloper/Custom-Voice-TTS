import numpy as np, sys
from vtts.synth import Synthesizer
from vtts.fx import ultron
eq = np.load("runs/best/ultron_eq.npz")["gain_db"]
s = Synthesizer("runs/best/acoustic.pt", "runs/best/vocoder.pt"); SR = 22050
y = s.tts("I was meant to be new. I was meant to be beautiful.", speaker=0, speed=0.85)
z = ultron(y, SR, intensity=0.6, sat=0.0, metal=0.25, down=0.0, lowpass=None, eq=eq)
h = SR // 100
def frames_db(x):
    n = len(x) // h; return 20 * np.log10(np.sqrt((x[:n * h].reshape(n, h) ** 2).mean(1)) + 1e-9)
for name, x in (('dry synth', y), ('final (filter+EQ)', z)):
    d = frames_db(x); print(f"{name:18s} 10ms-frame level dBFS  p5={np.percentile(d,5):6.1f} p25={np.percentile(d,25):6.1f} p50={np.percentile(d,50):6.1f} p75={np.percentile(d,75):6.1f} p95={np.percentile(d,95):6.1f}")
d = frames_db(y); hist, edges = np.histogram(d, bins=14); print('dry level histogram (dB: count):', {int(edges[i]): int(hist[i]) for i in range(len(hist))})
