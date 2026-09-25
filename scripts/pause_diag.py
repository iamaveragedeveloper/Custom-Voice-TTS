import numpy as np, torch
from vtts import text as T
from vtts.synth import Synthesizer
s = Synthesizer("runs/best/acoustic.pt", "runs/best/vocoder.pt")
txt = "I was meant to be new. I was meant to be beautiful."
m = s.mel(txt, speaker=0, speed=0.85)
tok = s._last_tok[0].cpu().numpy(); dur = s.am.last_dur[0].cpu().numpy()
w = s.wave(m); hop = 256
starts = np.concatenate([[0], np.cumsum(dur)])
print("token  frames  level(dBFS)")
for i, t in enumerate(tok):
    seg = w[starts[i]*hop: starts[i+1]*hop]
    lvl = 20*np.log10(np.sqrt((seg**2).mean()) + 1e-9) if len(seg) else float('nan')
    sym = T.VOCAB[t]
    if sym in ' .,!?;:-' or dur[i] >= 12: print(f"{sym!r:6} {dur[i]:5d}   {lvl:6.1f}")
print("total frames", dur.sum(), "= %.2fs" % (dur.sum()*hop/22050))
print("sentence split ->", T.split_sentences(txt))
