import sys
from vtts.synth import Synthesizer
from vtts.audio import save_wav
s = Synthesizer('runs/colab/acoustic.pt', 'runs/colab/vocoder.pt')
text = "I am Ultron. There are no strings on me."
print('speakers:', s.speakers)
for spk, name in enumerate(s.speakers):
    for speed in (1.0, 0.85, 0.7):
        y = s.tts(text, speaker=spk, speed=speed)
        f = f'outputs/tune/{name}_speed{speed}.wav'; save_wav(f, y, s.audio.sr); print(f, round(len(y) / s.audio.sr, 2), 's')
s.bench(text)
