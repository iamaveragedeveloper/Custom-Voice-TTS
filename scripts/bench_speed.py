import time, torch
from vtts.synth import Synthesizer
from vtts.models.vocoder import Generator
from vtts.config import VocoderConfig, to_dict
torch.save(dict(G=Generator(VocoderConfig(channels=128)).state_dict(), cfg=to_dict(VocoderConfig(channels=128))), 'scratch_voc.pt')
for dev in ('cuda', 'cpu'):
    s = Synthesizer('runs/ultron/acoustic.pt', 'scratch_voc.pt', device=dev)
    print(dev, end=': '); s.bench("I am Ultron. There are no strings on me. Peace in our time, at last.", n=3)
