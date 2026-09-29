# Voice iterations

Each iteration is a voice line you picked as good. GitHub plays the `.wav` files directly: click a file, then the play button.


## Iteration 1

- **Audio:** [`iteration_1.wav`](iteration_1.wav) (6.79 s, 22050 Hz)
- **Line:** "I was meant to be new. I was meant to be beautiful."
- **What changed:** First saved voice: clean voice (speaker 0) + Ultron filter (preset B, 0.6) + film-matched EQ (brighter, like the film clip) + moderate noise reduction.
- **Speed / quality:** GPU (GTX 1650 Ti): 122 ms for 6.5 s of speech = 53x real time. CPU only: 482 ms = 13.5x real time. Voice-vs-noise gap 21 dB. Whisper word errors on this line: 8%.
- **Reproduce:**

```bash
python -m vtts synth --text "I was meant to be new. I was meant to be beautiful." --acoustic runs/best/acoustic.pt --vocoder runs/best/vocoder.pt --speaker 0 --speed 0.85 --ultron-fx 0.6 --eq runs/best/ultron_eq.npz --clean 1.5 --out outputs/ultron.wav
```

## Iteration 2

- **Audio:** [`iteration_2.wav`](iteration_2.wav) (7.28 s, 22050 Hz)
- **Line:** "I was meant to be new. I was meant to be beautiful."
- **What changed:** Fixed the swallowed first word ('I' was ~23ms, now floored to ~160ms for the utterance's first vowel, shorter floors for other vowels/diphthongs). Lowered default noise reduction from 1.5 to 1.0 - the stronger settings measurably ate speech (Whisper word-error rose from 8% to 40-50%+) even though they looked cleaner by noise-floor numbers alone.
- **Speed / quality:** Whisper hears 'I' as the first word in 2/7 test lines (was 1/7); word errors on I-initial lines 58% (was 70%); other lines 39% (was 44%). Timing (GTX 1650 Ti): ~120ms for 6.5s of speech, ~54x real time; CPU-only ~13x real time.
- **Reproduce:**

```bash
python -m vtts synth --text "I was meant to be new. I was meant to be beautiful." --acoustic runs/best/acoustic.pt --vocoder runs/best/vocoder.pt --speaker 0 --speed 0.85 --ultron-fx 0.6 --eq runs/best/ultron_eq.npz --out outputs/ultron.wav
```
