import soundfile as sf, numpy as np, glob
from vtts.synth import Synthesizer
from vtts.audio import save_wav
from vtts.fx import ultron
import importlib.util, sys
sys.argv = ['x']; spec = importlib.util.spec_from_file_location('m', 'scripts/pulse_ref.py')
def pulse(y, sr):
    from scipy.signal import butter, hilbert, sosfilt
    b = sosfilt(butter(4, [1500, 10000], 'bandpass', fs=sr, output='sos'), y); env = np.abs(hilbert(b)); env -= env.mean()
    S = np.abs(np.fft.rfft(env * np.hanning(len(env)))); fr = np.fft.rfftfreq(len(env), 1 / sr); m = (fr > 20) & (fr < 400)
    return max(S[abs(fr - 86.13 * k) < 3].max() for k in (3, 4)) / S[m].mean()
ref = [pulse(*sf.read(f)[:1], 22050) for f in sorted(glob.glob('data/base/wavs/*.wav'))[:12]]
print('REAL recording pulse level (clean reference): %.1f' % np.mean(ref))
s = Synthesizer('runs/best/acoustic.pt', 'runs/colab3/vocoder.pt')
T = "I am Ultron, I come for peace and I want the Avengers extinction"
y = s.tts(T, speaker=0, speed=0.85); save_wav('outputs/final_plain.wav', y, s.audio.sr)
print('new voice pulse: %.1f' % pulse(y, s.audio.sr))
sf.write('outputs/final_B.wav', ultron(y, s.audio.sr, intensity=0.6, sat=0.0, metal=0.25), s.audio.sr, subtype='PCM_16')
