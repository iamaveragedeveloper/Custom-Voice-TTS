import soundfile as sf
from vtts.fx import ultron
y, sr = sf.read('outputs/peace_spk0.wav', dtype='float32')
V = {'A_less_grit':   dict(intensity=0.6, sat=1.0, metal=0.30),
     'B_no_grit':     dict(intensity=0.6, sat=0.0, metal=0.25),
     'C_pitch_room':  dict(intensity=0.6, sat=0.0, metal=0.0)}
for n, kw in V.items():
    sf.write(f'outputs/fx2_{n}.wav', ultron(y, sr, **kw), sr, subtype='PCM_16'); print('wrote', n)
