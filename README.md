# vtts — custom local voice-cloning TTS engine

Everything is written from scratch in PyTorch and runs locally. No paid services and no third-party pretrained weights.

- **Acoustic model** (`vtts/models/acoustic.py`): non-autoregressive FastPitch-style FFT transformer with duration,
  pitch and energy predictors. It has a built-in alignment learner (forward-sum CTC plus monotonic alignment search),
  so no external forced aligner is needed. About 13M parameters.
- **Vocoder** (`vtts/models/vocoder.py`): HiFi-GAN generator with multi-period and multi-scale discriminators.
  `--small` gives a ~4x cheaper generator.
- **Runtime** (`vtts/synth.py`): fp16 on GPU, per-sentence streaming, speed and pitch-shift controls, and a `--bench`
  RTF measurement.

## Workflow

```bash
# 1. Data: one long recording or a folder. Splits on silence, transcribes locally with Whisper (optional).
python -m vtts prep --input ultron.wav --out data/ultron --whisper small
#    (no Whisper? it writes data/ultron/metadata_todo.csv; fill in the text and re-run with --metadata)

# 2. Train
python -m vtts train-acoustic --data data/ultron --out runs/ultron
python -m vtts train-vocoder  --data data/ultron --out runs/ultron --small

# 3. Speak
python -m vtts synth --text "I am inevitable." --acoustic runs/ultron/acoustic.pt --vocoder runs/ultron/vocoder.pt --bench
```

Optional local installs: `pip install faster-whisper` (auto-transcription) and `pip install g2p_en`
(`prep --phonemes`, strongly recommended for small datasets).

Tests: `python -m pytest tests`
