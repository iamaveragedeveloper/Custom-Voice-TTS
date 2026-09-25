import numpy as np, soundfile as sf
def line(f, h=2205):  # 0.1 s blocks
    y, sr = sf.read(f); n = len(y) // h; d = 20 * np.log10(np.sqrt((y[:n*h].reshape(n, h) ** 2).mean(1)) + 1e-9)
    ref = np.percentile(d, 95)
    # bar: '#' loud speech .. '.' quiet noise floor .. ' ' true silence, based on dB below speech level
    return ''.join(' ' if v < -70 else ('#' if v > ref - 6 else ('+' if v > ref - 12 else ('-' if v > ref - 20 else '.'))) for v in d), d
for name, f in (('before', 'outputs/line_1_new_eq.wav'), ('after ', 'outputs/pure_line.wav')):
    s, d = line(f); print(f'{name} |{s}|')
print("legend: # loud voice  + voice  - quiet  . low-level noise (12-25 dB under the voice)  ' ' silence")
