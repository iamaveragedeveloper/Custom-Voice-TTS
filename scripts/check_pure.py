import numpy as np, soundfile as sf, re
from faster_whisper import WhisperModel
def db_frames(f, h=220):
    y, sr = sf.read(f); n = len(y) // h; return sr, 20 * np.log10(np.sqrt((y[:n*h].reshape(n, h) ** 2).mean(1)) + 1e-9)
for name, f in (('before (line_1_new_eq)', 'outputs/line_1_new_eq.wav'), ('after  (pure_line)', 'outputs/pure_line.wav')):
    sr, d = db_frames(f); speech = d > np.percentile(d, 95) - 25
    q = d[~speech]
    print(f"{name:24s} length {len(d)*0.01:4.2f}s | non-speech frames: {100*(~speech).mean():4.1f}% of the clip, their loudest level {q.max() if len(q) else float('nan'):6.1f} dB, "
          f"median {np.median(q) if len(q) else float('nan'):6.1f} dB | speech median {np.median(d[speech]):5.1f} dB")
wm = WhisperModel('small', device='cpu', compute_type='int8')
print('Whisper hears:', ' '.join(s.text.strip() for s in wm.transcribe('outputs/pure_line.wav', language='en', beam_size=5)[0]))
