import torch
from vtts.synth import Synthesizer
from vtts.audio import load_wav, mel_spectrogram
from vtts.config import AudioConfig
s = Synthesizer('runs/ultron/acoustic.pt')
sec = lambda m: round(m.shape[1] * 256 / 22050, 2)
for spk in (0, 1):
    for t in ['I am Ultron.', 'There are no strings on me.']:
        m = s.mel(t, speaker=spk)[0]
        print(spk, t, sec(m), 's  time-variation', round(float(m.std(1).mean()), 2))
c = AudioConfig()
g = mel_spectrogram(torch.from_numpy(load_wav('data/target/wavs/ultron_exp_0006.wav', 22050))[None], c)[0]
m = s.mel('Just makes me stronger.', speaker=1)[0]
print('real clip:', sec(g), 's variation', round(float(g.std(1).mean()), 2), '| synth:', sec(m), 's variation', round(float(m.std(1).mean()), 2))
