import re, sys
import numpy as np
import soundfile as sf
from scipy.signal import butter, hilbert, sosfilt
from vtts.synth import Synthesizer
from vtts.audio import save_wav
from faster_whisper import WhisperModel   # must come after torch (cuDNN clash otherwise)

SENT = ["I am Ultron, I come for peace and I want the Avengers extinction",
        "There are no strings on me",
        "Everyone creates the thing they dread"]
norm = lambda t: re.sub(r"[^a-z ]", "", t.lower()).split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        p, d = d, np.zeros(len(h) + 1, int); d[0] = i
        for j in range(1, len(h) + 1):
            d[j] = min(p[j] + 1, d[j - 1] + 1, p[j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1] / len(r)


def pulse(y, sr):  # strength of vocoder frame-rate (86.13 Hz x3/x4) modulation in the >1.5 kHz band
    b = sosfilt(butter(4, [1500, 10000], 'bandpass', fs=sr, output='sos'), y)
    env = np.abs(hilbert(b)); env -= env.mean()
    S = np.abs(np.fft.rfft(env * np.hanning(len(env)))); fr = np.fft.rfftfreq(len(env), 1 / sr)
    m = (fr > 20) & (fr < 400)
    return max(S[abs(fr - 86.13 * k) < 3].max() for k in (3, 4)) / S[m].mean()


wm = WhisperModel('small', device='cpu', compute_type='int8')
hear = lambda p: ' '.join(s.text.strip() for s in wm.transcribe(p, language='en', beam_size=5)[0])
COMBOS = {'v1_ac + v1_voc (old)': ('runs/best/acoustic.pt', 'runs/best/vocoder.pt'),
          'v1_ac + v3_voc': ('runs/best/acoustic.pt', 'runs/colab3/vocoder.pt'),
          'v3_ac(500) + v3_voc': ('runs/colab3/acoustic_best.pt', 'runs/colab3/vocoder.pt')}
for name, (a, v) in COMBOS.items():
    s = Synthesizer(a, v)
    w, pl = [], []
    for i, t in enumerate(SENT):
        y = s.tts(t, speaker=0, speed=0.85)
        f = f"outputs/ab2_{name.split(' ')[0]}_{name.split(' ')[2].replace('(', '').replace(')', '')}_{i}.wav"
        save_wav(f, y, s.audio.sr)
        w.append(wer(t, hear(f))); pl.append(pulse(y, s.audio.sr))
    print(f"{name:24s} WER={np.mean(w):.2f}  pulse={np.mean(pl):.1f}   per-sentence WER={[round(x, 2) for x in w]}")
