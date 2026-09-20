import time, torch, numpy as np
from vtts.data import AcousticDataset
from vtts.models import acoustic as A
from vtts.config import AcousticConfig, AudioConfig, VocoderConfig
from vtts.models.vocoder import Generator, Discriminators
ds = AcousticDataset('data/all'); dev = 'cuda'
m = A.AcousticModel(AcousticConfig(n_speakers=2), 80, ds.meta['n_vocab']).to(dev)
bs = [{k: v.to(dev) for k, v in ds.collate(i).items()} for i in ds.batches(10000)[:6]]
orig = A.mas; tm = [0.0]
def timed(x):
    t = time.time(); r = orig(x); tm[0] += time.time() - t; return r
A.mas = timed
torch.cuda.synchronize(); t = time.time()
for b in bs:
    loss, _ = m(b, 0.); loss.backward()
torch.cuda.synchronize(); tot = (time.time() - t) / 6
print(f'acoustic step {tot:.2f}s, of which MAS {tm[0] / 6:.2f}s ({100 * tm[0] / 6 / tot:.0f}%)')
G, D = Generator(VocoderConfig(channels=128), 80).to(dev), Discriminators().to(dev)
y = torch.randn(8, 1, 8192, device=dev); mel = torch.randn(8, 80, 32, device=dev)
def tt(f, n=5):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n
print(f'vocoder: G fwd+bwd {tt(lambda: G(mel).sum().backward()):.2f}s, D fwd+bwd {tt(lambda: sum(x.sum() for x in D(y, y)[0]).backward()):.2f}s')
