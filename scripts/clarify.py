import sys, numpy as np, soundfile as sf
from scipy.signal import butter, sosfilt
y, sr = sf.read(sys.argv[1]); y = y.astype(np.float64)
hp = sosfilt(butter(2, 1800, 'high', fs=sr, output='sos'), y)   # presence band
y2 = y + 0.9 * hp                                             # ~ +5 dB above 1.8 kHz
y2 = sosfilt(butter(2, 60, 'high', fs=sr, output='sos'), y2)   # drop rumble
y2 = np.tanh(1.6 * y2 / np.abs(y2).max()) / np.tanh(1.6)      # gentle compression
y2 = y2 / np.abs(y2).max() * 0.9
sf.write(sys.argv[2], y2.astype(np.float32), sr, subtype='PCM_16')
