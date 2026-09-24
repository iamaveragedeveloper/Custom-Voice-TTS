# vtts - a from-scratch, fully local voice-cloning TTS engine (Ultron voice project)

A complete text-to-speech engine written in PyTorch from scratch: text frontend, acoustic model, vocoder, data
pipeline, trainers, real-time runtime and an "Ultron" voice effect chain. **No paid services, no third-party
pretrained weights, no cloud API.** Training can run on your own GPU or on free Colab; running the finished voice
needs neither.

The target voice is Ultron (James Spader, *Avengers: Age of Ultron*), built from two audio sources:

| Source | Length | Role |
|---|---|---|
| `raw/ultron_10min.wav` - a clean, deep, Ultron-like voice | ~10 min (65 clips) | Teaches speech and pronunciation. Speaker `ultron_clean` (id 0) |
| `raw/ultron_exp.wav` - clip from the film | ~1 min (7 clips) | The real target sound. Speaker `ultron_movie` (id 1) |

---

## Status (honest summary)

| Area | State |
|---|---|
| Engine (all components) | Done, tested (`pytest`), all custom |
| Speed | Excellent: ~82x real time on a GTX 1650 Ti, ~13x real time on CPU alone |
| Clean voice (speaker 0) | Intelligible, deep, Ultron-like. Some slurred words. Residual vocoder buzz ("pulse") |
| Movie voice (speaker 1) | Right timbre in places (e.g. the word "me"), but inconsistent - only 1 minute of source audio |
| "Ultron sound" | Best result so far = clean voice + DSP Ultron filter (preset B, strength 0.6) |
| Not solved | Studio-clean quality; needs more Ultron audio and/or more vocoder training |

Nothing here is a finished product. It is a working proof of concept with a clear path to improve.

## Quick start - make it speak

Requirements: Python 3.11, PyTorch (see [Setup](#setup)). Best models are in `runs/best/`
(v1 acoustic model + v3 vocoder).

```bash
# clean voice + your chosen Ultron filter (preset B)
python -m vtts synth --text "I am Ultron, I come for peace and I want the Avengers extinction" \
    --acoustic runs/best/acoustic.pt --vocoder runs/best/vocoder.pt --speaker 0 --speed 0.85 \
    --ultron-fx 0.6 --out outputs/ultron.wav

# plain voice, no filter: drop --ultron-fx.  Movie voice: --speaker 1.
# speed check:            add --bench
```

Options: `--speed` (0.85 sounds best; <1 = slower), `--semitones` (pitch shift), `--speaker` (0 = clean,
1 = movie), `--ultron-fx STRENGTH` (0 = off; 0.6 = preset B; 1.0 = full).

From Python:

```python
from vtts.synth import Synthesizer
from vtts.fx import ultron
s = Synthesizer("runs/best/acoustic.pt", "runs/best/vocoder.pt")
y = s.tts("There are no strings on me.", speaker=0, speed=0.85)       # numpy float32 @ 22050 Hz
y = ultron(y, s.audio.sr, intensity=0.6, sat=0.0, metal=0.25)         # preset B
for chunk in s.stream("Long text. Split per sentence. Playback can start early."):
    ...                                                                # streaming, one chunk per sentence
```

## How it works

```
text -> normalise + numbers -> ARPAbet phonemes (g2p_en) -> tokens
     -> ACOUSTIC MODEL (FastPitch-style, non-autoregressive, 14.2M params) -> 80-band mel spectrogram
     -> VOCODER (HiFi-GAN generator, "small" 0.9M params) -> 22.05 kHz waveform
     -> optional Ultron filter (pure DSP)
```

- **Text frontend** (`vtts/text.py`): lowercase/ASCII normalisation, number expansion ("23" -> "twenty three"),
  optional ARPAbet phonemes via `g2p_en` (used for this project - much better than characters on small data).
- **Acoustic model** (`vtts/models/acoustic.py`): FFT-transformer encoder/decoder with duration, pitch and energy
  predictors. Non-autoregressive, so all frames are produced in one parallel pass. It has a **built-in alignment
  learner** (forward-sum CTC + monotonic alignment search), so no external aligner (MFA etc.) is needed.
  Multi-speaker through a learned speaker embedding.
- **Vocoder** (`vtts/models/vocoder.py`): HiFi-GAN generator with multi-period + multi-scale discriminators.
  Trained with mel L1, feature matching, adversarial loss and a multi-resolution STFT loss.
- **Ultron filter** (`vtts/fx.py`): deeper pitch layer, short metallic comb resonance, optional soft saturation,
  small room. Real-time capable. Preset B = `intensity=0.6, sat=0, metal=0.25`.
- **Runtime** (`vtts/synth.py`): per-sentence streaming, speed and pitch control, benchmark, Griffin-Lim fallback
  when no vocoder is given.

## Project layout

```
vtts/                  the engine
  text.py              frontend (normalise, numbers, phonemes)
  audio.py             log-mel, F0 estimator, Griffin-Lim
  prep.py              audio -> training set (split, Whisper transcripts, features, statistics)
  data.py              datasets, held-out split, length-bucketed batches, vocoder crops
  models/acoustic.py   acoustic model + aligner + MAS + length regulator
  models/vocoder.py    HiFi-GAN generator, discriminators, losses
  train.py             acoustic + vocoder trainers (resume, time limit, validation, warm-ups)
  predcache.py         cache the acoustic model's own mels (for vocoder training)
  synth.py             inference / streaming / benchmark
  fx.py                Ultron effect chain (+ experimental depulse, see Findings)
  cli.py               python -m vtts <command>
tests/test_smoke.py    end-to-end test on synthetic audio
scripts/               merge datasets, notebook builder, benchmarks and analysis helpers
colab/                 Colab notebook + upload package (see below)
raw/                   your source recordings            (git-ignored)
data/                  prepared datasets                 (git-ignored)
runs/                  checkpoints                       (git-ignored)
outputs/               generated audio                   (git-ignored)
```

`runs/best/` holds the current best pair (`acoustic.pt`, `vocoder.pt`) plus `vocoder_v1.pt` (older vocoder).
`runs/colab*/` are downloads from earlier Colab sessions.

## Setup

```bash
pip install torch torchaudio numpy scipy soundfile        # torch with CUDA if you have an NVIDIA GPU
pip install faster-whisper g2p_en                         # free, local: transcription + phonemes
python -c "import nltk; [nltk.download(p) for p in ['averaged_perceptron_tagger','averaged_perceptron_tagger_eng','cmudict']]"
python -m pytest tests                                    # sanity check
```

## Workflow: from recordings to a voice

### 1. Prepare data

```bash
# one long recording or a folder; splits on silence, transcribes locally with Whisper
python -m vtts prep --input raw/ultron_10min.wav --out data/base   --whisper small --phonemes --speaker ultron_clean
python -m vtts prep --input raw/ultron_exp.wav   --out data/target --whisper small --phonemes --speaker ultron_movie

# merge into one 2-speaker set; ":4" = show the movie clips 4x per epoch (no duplicate files)
python -m scripts.merge data/all ultron_clean=data/base ultron_movie=data/target:4
```

**Always read the transcripts** printed by `prep` - Whisper makes small mistakes and the model learns them.
Without Whisper, `prep` writes `metadata_todo.csv` to fill by hand (`file|text[|speaker]`), then use `--metadata`.
Every speaker gets ~10% held-out clips (never trained on) for validation.

### 2. Train

```bash
python -m vtts train-acoustic --data data/all --out runs/x [--init ckpt] [--lr 3e-4] [--steps N] [--max-minutes M]
python -m vtts cache-pred     --data data/all --acoustic runs/x/acoustic_best.pt
python -m vtts train-vocoder  --data data/all --out runs/x --small [--init ckpt] [--d-warmup-steps 1500] [--max-minutes M]
```

- Both trainers **resume automatically** from `*_last.pt` in the output folder.
- `--max-minutes` stops cleanly and saves. `--steps` are *total* targets.
- The acoustic trainer prints `[val @ step]` lines and keeps `acoustic_best.pt` (lowest held-out mel loss).
- `cache-pred` stores the acoustic model's own output so the vocoder can train on what it will really receive.
- **On a GTX 16xx card do not use `--amp`** (fp16 gives NaN there). T4 and newer are fine.

### 3. Train on free Colab (recommended - about 3-5x faster than a 1650 Ti)

1. Open `colab/train_ultron_v3.ipynb` in Colab, set Runtime to **T4 GPU**.
2. Cell 1: upload `colab/ultron_project_v3.zip` (first time) - no Google Drive is used.
3. Run the cells: cache predictions, train for `MINUTES`, sample audio, download.
4. The last cell downloads `ultron_models.zip` (use on your PC) and `ultron_project_next.zip`.
5. **Next session**: upload `ultron_project_next.zip` instead, raise the `*_STEPS` numbers, run again.

Colab wipes its disk when the session ends - download before closing the tab. Free sessions can disconnect early.
The notebook rebuilds itself with `python -m scripts.make_notebook_v3`.

## Results and measurements

**Speed** (GTX 1650 Ti, small vocoder, fp32): 3.8 s of audio in 46 ms (~82x real time). CPU only: ~13x real time.

**Clarity and buzz** (Whisper word-error rate on 3 test sentences; pulse = strength of the vocoder frame-rate buzz,
real recordings measure 2.2):

| Setup | Word errors | Pulse |
|---|---|---|
| v1 acoustic (8k steps) + v1 vocoder | 52% | 14.5 |
| v1 acoustic + **v3 vocoder** (current best) | 60% (within noise) | **9.9** |
| v3 acoustic (best val, step 500) + v3 vocoder | 74% | 10.1 |

(The word-error test is small and noisy - use it to spot big changes, not small ones.)

## Findings and lessons (so we don't repeat them)

- **GTX 16xx + fp16:** cuDNN half-precision convolutions return NaN on this GPU family. Train in fp32 locally.
- **More acoustic steps on 11 minutes did not help.** A run to 14k steps sounded worse than 8k. Cause: a learning-rate
  jump on resume (now fixed) plus a data-size ceiling. The best held-out score was reached almost immediately.
- **The "pulse" noise is a vocoder artifact**, not from the filter: strong modulation at 3x and 4x the vocoder frame
  rate (22050/256 = 86.1 Hz -> 258/344 Hz), broadband above 500 Hz. Low-pass filters and an envelope-notch attempt
  (`fx.depulse`, kept but ineffective) do not remove it. The fix is more/better vocoder training (mixed real and
  predicted mels, discriminator warm-up, multi-resolution STFT loss) - this cut it from 14.5 to 9.9 in one Colab hour.
- **F0 tracking** on the deep, processed voice needed a lower range (40-400 Hz) and true peak picking.
- **Restarting a vocoder with fresh discriminators** disturbs the generator (v2 regression) - hence `--d-warmup-steps`.
- **Import order:** import torch before `faster_whisper`, otherwise a cuDNN library clash occurs.

## Known limitations

- 11.5 minutes total audio (only ~1 minute of the real film voice) is far below what a from-scratch model wants.
- The movie voice is inconsistent; the clean voice + filter is the more reliable route.
- Residual vocoder buzz; small vocoder (0.9M params). A larger generator (`--small` off) is clearer but restarts training.
- English only. Long or unusual words can slur.

## Roadmap (highest value first)

1. **More clean Ultron dialogue** (20-30+ min, no music/effects; drop in `raw/movie/`, re-run prep and merge). Biggest gain.
2. **Vocoder-only Colab sessions** (the acoustic model has plateaued on current data) until the pulse level nears ~2-4.
3. Pretrain the acoustic model and vocoder on a large free public speech dataset (e.g. LJSpeech), then fine-tune.
4. Try the full-size vocoder (256 channels) if the small one stays muddy.
5. Tune the filter against the film clip (deeper pitch, resonance, room).

## Tests and helper scripts

`python -m pytest tests` - synthetic-audio test of prep, held-out split, resume, best-checkpoint, prediction caching,
vocoder mixing and synthesis. Handy scripts (run with `PYTHONPATH=.`): `scripts/ab2.py` (Whisper word-error + pulse
comparison of checkpoints), `scripts/bench_speed.py`, `scripts/pulse2.py`, `scripts/fx_demo*.py`,
`scripts/merge.py`, `scripts/make_notebook_v3.py`.

## Data and use

The training audio is Ultron voice material you supplied; the film audio is copyrighted. This project is for personal
experimentation on your own machine. Do not use it to impersonate a real person or publish audio that could pass as
the actor's real performance.
