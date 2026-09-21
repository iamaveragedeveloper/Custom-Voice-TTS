import re, subprocess, numpy as np, soundfile as sf
from vtts.synth import Synthesizer
from vtts.audio import save_wav
from faster_whisper import WhisperModel
REF = "I am Ultron, I come for peace and I want the Avengers extinction"
norm = lambda t: re.sub(r"[^a-z ]", "", t.lower()).split()
def wer(ref, hyp):
    r, h = norm(ref), norm(hyp); d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        p, d = d, np.zeros(len(h) + 1, int); d[0] = i
        for j in range(1, len(h) + 1): d[j] = min(p[j] + 1, d[j-1] + 1, p[j-1] + (r[i-1] != h[j-1]))
    return d[-1] / len(r)
wm = WhisperModel('small', device='cpu', compute_type='int8')
def hear(path): return ' '.join(s.text.strip() for s in wm.transcribe(path, language='en', beam_size=5)[0])
A = {'ac8k': 'runs/colab/acoustic.pt', 'ac14k': 'runs/colab2/acoustic.pt'}
V = {'voc_v1': 'runs/colab/vocoder.pt', 'voc_v2': 'runs/colab2/vocoder.pt'}
for an, ap in A.items():
    for vn, vp in V.items():
        s = Synthesizer(ap, vp)
        for spk in (0,):
            y = s.tts(REF, speaker=spk, speed=0.85); f = f'outputs/ab/{an}_{vn}_spk{spk}.wav'; save_wav(f, y, s.audio.sr)
            subprocess.run(['python', 'scripts/clarify.py', f, f.replace('.wav', '_clear.wav')])
            for tag, ff in (('raw', f), ('clear', f.replace('.wav', '_clear.wav'))):
                t = hear(ff); print(f'{an:6s} {vn} {tag:5s} WER={wer(REF, t):.2f} | {t}')
