import numpy as np, soundfile as sf
from scipy.signal import hilbert, butter, sosfilt
def strength(y, sr, lo, hi):
    b = sosfilt(butter(4, [lo, hi], 'bandpass', fs=sr, output='sos'), y); env = np.abs(hilbert(b)); env = env - env.mean()
    S = np.abs(np.fft.rfft(env * np.hanning(len(env)))); fr = np.fft.rfftfreq(len(env), 1 / sr); m = (fr > 20) & (fr < 400)
    pk = max(S[(abs(fr - 86.13 * k) < 3)].max() for k in (3, 4)); return round(float(pk / S[m].mean()), 1)
y, sr = sf.read('outputs/peace_spk0.wav'); print('pulse strength per band (x-avg), raw TTS output:')
for lo, hi in [(150, 500), (500, 1500), (1500, 3000), (3000, 5000), (5000, 8000), (8000, 10500)]:
    print(f'  {lo:>5}-{hi:<5} Hz : {strength(y, sr, lo, hi)}')
for fc in (4000, 5000, 6000):
    yl = sosfilt(butter(4, fc, 'low', fs=sr, output='sos'), y); print(f'after low-pass {fc} Hz, band 1500-10000 pulse:', strength(yl, sr, 1500, 10000) if fc > 1600 else '-')
