import numpy as np
from vtts import text as T
from vtts.synth import Synthesizer
from vtts.fx import ultron, denoise
SR = 22050; hop = 256
eq = np.load("runs/best/ultron_eq.npz")["gain_db"]
s = Synthesizer("runs/best/acoustic.pt", "runs/best/vocoder.pt")
txt = "I was meant to be new."
m = s.mel(txt, speaker=0, speed=0.85)
tok = s._last_tok[0].cpu().numpy(); dur = s.am.last_dur[0].cpu().numpy()
y = s.wave(m); st = np.concatenate([[0], np.cumsum(dur)])
print("first tokens: ", "  ".join(f"{T.VOCAB[t]!r}:{d}fr" for t, d in list(zip(tok, dur))[:9]), "   (1 frame = 11.6 ms)")
base = dict(intensity=0.6, sat=0.0, metal=0.25, down=0.0, lowpass=None, eq=eq)
z0 = ultron(y, SR, clean=0.0, **base); z1 = ultron(y, SR, clean=1.5, **base); z2 = ultron(y, SR, clean=1.3, clean_hf=1.0, clean_passes=2, **base)
w = int(0.05 * SR)
def lv(x, i): seg = x[i*w:(i+1)*w]; return 20*np.log10(np.sqrt((seg**2).mean()) + 1e-9)
print("level of each 50 ms window in the first 0.6 s (dBFS, each version normalised to its own peak):")
for name, x in (("dry voice", y), ("filter+EQ, no cleaning", z0), ("cleaning 1.5 (iteration 1)", z1), ("cleaning 'quieter'", z2)):
    x = x / (np.abs(x).max() + 1e-9)
    print(f"  {name:28s}", " ".join(f"{lv(x, i):5.0f}" for i in range(12)))
