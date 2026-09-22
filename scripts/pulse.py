import numpy as np, soundfile as sf
from scipy.signal import hilbert, butter, sosfilt
def modspec(f):
    y, sr = sf.read(f); env = np.abs(hilbert(sosfilt(butter(2, 1500, 'high', fs=sr, output='sos'), y)))  # hiss/buzz band envelope
    env = env - env.mean(); S = np.abs(np.fft.rfft(env * np.hanning(len(env)))); fr = np.fft.rfftfreq(len(env), 1 / sr)
    band = (fr > 20) & (fr < 400); S, fr = S[band], fr[band]
    top = np.argsort(S)[-4:][::-1]
    return [(round(float(fr[i]), 1), round(float(S[i] / S.mean()), 1)) for i in top]
for f in ['outputs/peace_spk0.wav','outputs/fx3_raw_depulsed.wav','outputs/fx2_B_no_grit.wav','outputs/fx3_B_depulsed.wav']:
    print(f.split('/')[-1], 'strongest modulation rates (Hz, x-avg):', modspec(f))
print('vocoder frame rate = 22050/256 =', round(22050 / 256, 1), 'Hz; comb 4.5 ms ->', round(1000 / 4.5, 1), 'Hz')
