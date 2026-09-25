import re, numpy as np
from scipy.signal import butter, hilbert, sosfilt
from vtts.synth import Synthesizer
from vtts.audio import save_wav
from vtts.fx import ultron
from faster_whisper import WhisperModel
SR = 22050; TXT = "I was meant to be new. I was meant to be beautiful."; REF = "I was meant to be new I was meant to be beautiful"
eq = np.load("runs/best/ultron_eq.npz")["gain_db"]
s = Synthesizer("runs/best/acoustic.pt", "runs/best/vocoder.pt"); y = s.tts(TXT, speaker=0, speed=0.85)
wm = WhisperModel("small", device="cpu", compute_type="int8"); hear = lambda p: " ".join(x.text.strip() for x in wm.transcribe(p, language="en", beam_size=5)[0])
norm = lambda t: re.sub(r"[^a-z ]", "", t.lower()).split()
def wer(r, h):
    r, h = norm(r), norm(h); d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        p, d = d, np.zeros(len(h) + 1, int); d[0] = i
        for j in range(1, len(h) + 1): d[j] = min(p[j] + 1, d[j-1] + 1, p[j-1] + (r[i-1] != h[j-1]))
    return d[-1] / len(r)
def floor_db(z, h=220):  # level of the quietest 15% non-silent 10ms frames = the noise floor
    n = len(z) // h; d = 20 * np.log10(np.sqrt((z[:n*h].reshape(n, h) ** 2).mean(1)) + 1e-9); d = d[d > -70]
    return np.percentile(d, 15), np.percentile(d, 95)
def pulse(z):
    b = sosfilt(butter(4, [1500, 10000], "bandpass", fs=SR, output="sos"), z); e = np.abs(hilbert(b)); e -= e.mean()
    S = np.abs(np.fft.rfft(e * np.hanning(len(e)))); fr = np.fft.rfftfreq(len(e), 1 / SR)
    return max(S[abs(fr - 86.13 * k) < 3].max() for k in (3, 4)) / S[(fr > 20) & (fr < 400)].mean()
base = dict(intensity=0.6, sat=0.0, metal=0.25, down=0.0, lowpass=None, eq=eq)
for name, c in (("no cleaning", 0.0), ("clean 1.5 (moderate)", 1.5), ("clean 2.5 (strong)", 2.5)):
    z = ultron(y, SR, clean=c, **base); f = f"outputs/pure_{'off' if c == 0 else c}.wav"; save_wav(f, z, SR)
    lo, hi = floor_db(z)
    print(f"{name:22s} noise floor {lo:6.1f} dB, voice {hi:6.1f} dB -> gap {hi - lo:4.1f} dB | pulse {pulse(z):4.1f} | WER {wer(REF, hear(f)):.2f}")
