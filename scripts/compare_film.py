import numpy as np, soundfile as sf, torch
from scipy.signal import welch
from vtts.audio import estimate_f0
from vtts.config import AudioConfig
c = AudioConfig()
def load(f):
    y, sr = sf.read(f, dtype='float32'); y = y.mean(1) if y.ndim > 1 else y
    if sr != 22050:
        import torchaudio; y = torchaudio.functional.resample(torch.from_numpy(y), sr, 22050).numpy()
    return y / (np.abs(y).max() + 1e-9)
bands = [(60,150),(150,300),(300,600),(600,1200),(1200,2500),(2500,5000),(5000,8000),(8000,10500)]
def profile(y):
    f, p = welch(y, 22050, nperseg=4096); tot = p[(f>60)&(f<10500)].sum()
    return [round(10*np.log10(p[(f>=lo)&(f<hi)].sum()/tot+1e-12),1) for lo,hi in bands]
def f0stats(y):
    y = torch.from_numpy(y[:len(y)//256*256]); f0 = estimate_f0(y, c); v = f0[f0>0].numpy()
    return f'median {np.median(v):.0f} Hz, spread(5-95%) {np.percentile(v,5):.0f}-{np.percentile(v,95):.0f} Hz, voiced {100*len(v)/len(f0):.0f}%'
def flatness(y):  # spectral flatness in 1-5 kHz (higher = more noise/buzz-like, lower = tonal/harmonic)
    f, p = welch(y, 22050, nperseg=2048); m = (f>1000)&(f<5000); p = p[m]+1e-12
    return round(float(np.exp(np.log(p).mean())/p.mean()), 3)
S = {'FILM clip (target)': 'raw/ultron_exp.wav', 'clean 10min voice': 'raw/ultron_10min.wav',
     'our output, plain': 'outputs/final_plain.wav', 'our output + filter B': 'outputs/final_B.wav'}
print('band dB (rel. total) :', [f'{lo}-{hi}' for lo,hi in bands])
for n, f in S.items():
    y = load(f); print(f'{n:24s}', profile(y)); print(f'{"":24s} pitch: {f0stats(y)} | flatness(1-5k) {flatness(y)}')
