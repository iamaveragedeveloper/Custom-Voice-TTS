import soundfile as sf
from vtts.fx import ultron, depulse
y, sr = sf.read('outputs/peace_spk0.wav', dtype='float32')
d = depulse(y, sr); sf.write('outputs/fx3_raw_depulsed.wav', d, sr, subtype='PCM_16')
sf.write('outputs/fx3_B_depulsed.wav', ultron(d, sr, intensity=0.6, sat=0.0, metal=0.25), sr, subtype='PCM_16')
