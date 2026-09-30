import re
import numpy as np
from vtts.synth import Synthesizer
from vtts.audio import save_wav
from vtts.fx import ultron
from faster_whisper import WhisperModel

SR = 22050
TXT = "I was meant to be new. I was meant to be beautiful."
REF = "I was meant to be new I was meant to be beautiful"
eq = np.load("runs/best/ultron_eq.npz")["gain_db"]
s = Synthesizer("runs/best/acoustic.pt", "runs/best/vocoder.pt")
wm = WhisperModel("small", device="cpu", compute_type="int8")
hear = lambda p: " ".join(x.text.strip() for x in wm.transcribe(p, language="en", beam_size=5)[0])
norm = lambda t: re.sub(r"[^a-z ]", "", t.lower()).split()


def wer(r, h):
    r, h = norm(r), norm(h)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        p, d = d, np.zeros(len(h) + 1, int); d[0] = i
        for j in range(1, len(h) + 1):
            d[j] = min(p[j] + 1, d[j - 1] + 1, p[j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1] / len(r)


base = dict(intensity=0.6, sat=0.0, metal=0.25, down=0.0, lowpass=None, eq=eq, clean=1.0)
V = {
    "1_iteration2_baseline": dict(),
    "2_contour_fall": dict(contour_fall=0.5),
    "3_formant_deeper": dict(formant=0.85),
    "4_formant_deeper_strong": dict(formant=0.75),
    "5_ring_light": dict(ring_freq=30, ring_mix=0.25),
    "6_ring_moderate": dict(ring_freq=30, ring_mix=0.45),
    "7_all_combined": dict(contour_fall=0.5, formant=0.85, ring_freq=30, ring_mix=0.2),
}
for name, kw in V.items():
    synth_kw = {k: v for k, v in kw.items() if k == "contour_fall"}
    fx_kw = {k: v for k, v in kw.items() if k != "contour_fall"}
    y = s.tts(TXT, speaker=0, speed=0.85, **synth_kw)
    z = ultron(y, SR, **base, **fx_kw)
    f = f"outputs/newfx_{name}.wav"
    save_wav(f, z, SR)
    print(f"{name:26s} WER={wer(REF, hear(f)):.2f}  heard: {hear(f)}")

# a gentler combo, since the strong combo (7) stacked too much distortion
y = s.tts(TXT, speaker=0, speed=0.85, contour_fall=0.4)
z = ultron(y, SR, **base, formant=0.9, ring_freq=25, ring_mix=0.12)
f = "outputs/newfx_8_combined_gentle.wav"
save_wav(f, z, SR)
print(f"8_combined_gentle          WER={wer(REF, hear(f)):.2f}  heard: {hear(f)}")
