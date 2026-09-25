import re, numpy as np
from vtts.synth import Synthesizer
from vtts.audio import save_wav
from vtts.fx import ultron
from faster_whisper import WhisperModel
SR = 22050
eq = np.load("runs/best/ultron_eq.npz")["gain_db"]
s = Synthesizer("runs/best/acoustic.pt", "runs/best/vocoder.pt")
wm = WhisperModel("small", device="cpu", compute_type="int8")
I_LIKE = {"i", "im", "ive", "ill", "id", "eye", "aye", "hi"}
norm = lambda t: re.sub(r"[^a-z ]", "", t.lower()).split()
def wer(r, h):
    r, h = norm(r), norm(h); d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        p, d = d, np.zeros(len(h) + 1, int); d[0] = i
        for j in range(1, len(h) + 1): d[j] = min(p[j] + 1, d[j-1] + 1, p[j-1] + (r[i-1] != h[j-1]))
    return d[-1] / len(r)
def transcribe(path):
    segs, _ = wm.transcribe(path, language="en", beam_size=5, word_timestamps=True)
    words = [w for sg in segs for w in sg.words]
    return " ".join(w.word.strip() for w in words), (words[0].word.strip() if words else "")
I_LINES = ["I was meant to be new. I was meant to be beautiful.", "I am Ultron.", "I know what it is to lose.", "I am inevitable.",
           "I come for peace.", "I wanted to show you.", "I think about meteors."]
OTHER = ["There are no strings on me.", "Everyone creates the thing they dread.", "Peace in our time, at last."]
base = dict(intensity=0.6, sat=0.0, metal=0.25, down=0.0, lowpass=None, eq=eq, clean=1.5)
import vtts.synth as VS
CONFIGS = [("floors 7/11/14 (current)", dict(mono=7, diph=11, first_vowel=14, cons=2)),
           ("floors 8/12/18", dict(mono=8, diph=12, first_vowel=18, cons=2)),
           ("floors 9/13/22", dict(mono=9, diph=13, first_vowel=22, cons=3)),
           ("floors 10/14/26", dict(mono=10, diph=14, first_vowel=26, cons=3))]
for name, fl in CONFIGS:
    VS._MIN_FRAMES.update(fl); md = True
    hits, w_i, w_o, firsts = 0, [], [], []
    for t in I_LINES:
        z = ultron(s.tts(t, speaker=0, speed=0.85, min_dur=md), SR, **base); save_wav("outputs/_e.wav", z, SR)
        txt, fw = transcribe("outputs/_e.wav"); firsts.append(fw); hits += norm(fw)[:1] and norm(fw)[0] in I_LIKE; w_i.append(wer(t, txt))
    for t in OTHER:
        z = ultron(s.tts(t, speaker=0, speed=0.85, min_dur=md), SR, **base); save_wav("outputs/_e.wav", z, SR); w_o.append(wer(t, transcribe("outputs/_e.wav")[0]))
    print(f"{name}: first word heard as 'I' in {int(hits)}/{len(I_LINES)} lines | WER on 'I' lines {np.mean(w_i):.2f} | WER on other lines {np.mean(w_o):.2f} | first words heard: {firsts}")
