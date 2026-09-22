import sys, soundfile as sf
from vtts.fx import ultron
src = sys.argv[1]; y, sr = sf.read(src, dtype='float32')
for name, kw in (('light', dict(intensity=0.6)), ('full', dict(intensity=1.0)), ('heavy', dict(intensity=1.4))):
    sf.write(f'outputs/fx_{name}.wav', ultron(y, sr, **kw), sr, subtype='PCM_16'); print('wrote', name)
