import math
import time

import numpy as np
import torch

from . import text as T
from .audio import griffin_lim
from .config import AcousticConfig, AudioConfig, VocoderConfig
from .models.acoustic import AcousticModel
from .models.vocoder import Generator


_PAUSE_IDS = [T.TOK[c] for c in ".,!?;:-"]
_DIPH = {"AW", "AY", "OY", "EY", "OW"}
_VOWEL = _DIPH | {"AA", "AE", "AH", "AO", "EH", "ER", "IH", "IY", "UH", "UW"}
# minimum frames (1 frame = 11.6 ms): the duration model sometimes squeezes a vowel to ~2 frames, which makes
# the word (especially a sentence-initial "I") disappear
_MIN_FRAMES = dict(mono=7, diph=11, first_vowel=14, cons=2)


class Synthesizer:
    """Loads acoustic + vocoder checkpoints; fp16 is opt-in (broken cuDNN half convs on GTX 16xx). Without a vocoder falls back to Griffin-Lim."""

    def __init__(self, acoustic, vocoder=None, device=None, fp16=False):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.amp = fp16 and self.dev.type == "cuda"
        ck = torch.load(acoustic, map_location=self.dev)
        self.audio = AudioConfig(**ck["audio"])
        self.stats, self.phonemes, self.speakers = ck["stats"], ck["phonemes"], ck["speakers"]
        self.am = AcousticModel(AcousticConfig(**ck["cfg"]), self.audio.n_mels, ck["n_vocab"]).to(self.dev)
        self.am.load_state_dict(ck["model"])
        self.am.eval()
        self.voc = None
        if vocoder:
            vk = torch.load(vocoder, map_location=self.dev)
            self.voc = Generator(VocoderConfig(**vk["cfg"]), self.audio.n_mels).to(self.dev)
            self.voc.load_state_dict(vk["G"])
            self.voc.strip_norm().eval()

    @torch.no_grad()
    def mel(self, text, speaker=0, speed=1.0, semitones=0.0, pitch_var=1.0, min_dur=True):
        tok = torch.tensor([T.encode(text, self.phonemes)], device=self.dev)
        self._last_tok = tok
        md = self._min_dur(tok[0]) if min_dur else None
        if tok.numel() == 0:
            return None
        spk = torch.tensor([speaker], device=self.dev)
        shift = semitones * math.log(2) / 12 / self.stats["pitch_std"]
        with torch.autocast(self.dev.type, dtype=torch.float16, enabled=self.amp):
            return self.am.infer(tok, spk, speed, shift, pitch_var=pitch_var, min_dur=md)

    def _min_dur(self, tok):
        """(1,N) minimum frames per token; the first vowel of the utterance gets a longer floor."""
        syms = [T.VOCAB[int(t)] for t in tok.cpu()]
        out, first_done = [], False
        for sym in syms:
            b = sym.rstrip("012")
            if b in _VOWEL:
                v = _MIN_FRAMES["diph" if b in _DIPH else "mono"]
                if not first_done:
                    v, first_done = _MIN_FRAMES["first_vowel"], True
            elif sym == " " or not sym[0].isalpha():
                v = 0
            else:
                v = _MIN_FRAMES["cons"]
            out.append(v)
        return torch.tensor([out], device=self.dev)

    def _pause_mask(self, n_samples, fade_ms=12):
        """1 where there is speech, 0 over punctuation tokens (the model's own pause regions), smooth edges."""
        tok = self._last_tok[0].cpu().numpy()
        dur = self.am.last_dur[0].cpu().numpy()
        frame_tok = np.repeat(tok, dur)
        speech = ~np.isin(frame_tok, _PAUSE_IDS)
        m = np.repeat(speech.astype(np.float32), self.audio.hop)
        m = np.pad(m, (0, max(n_samples - len(m), 0)), constant_values=1.0)[:n_samples]
        k = int(self.audio.sr * fade_ms / 1000) | 1
        return np.clip(np.convolve(m, np.hanning(k) / np.hanning(k).sum(), mode="same"), 0, 1).astype(np.float32)

    @torch.no_grad()
    def wave(self, mel):
        if self.voc is None:
            return griffin_lim(mel[0].float(), self.audio).cpu().numpy()
        with torch.autocast(self.dev.type, dtype=torch.float16, enabled=self.amp):
            return self.voc(mel)[0, 0].float().cpu().numpy()

    def stream(self, text, gap=0.3, gate=True, **kw):
        """Yield one waveform chunk per sentence so playback can start before the full text is done."""
        for s in T.split_sentences(text):
            m = self.mel(s, **kw)
            if m is not None:
                w = self.wave(m)
                if gate:
                    w = w * self._pause_mask(len(w))
                yield w
                yield np.zeros(int(gap * self.audio.sr), np.float32)

    def tts(self, text, **kw):
        chunks = list(self.stream(text, **kw))
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    def bench(self, text="The quick brown fox jumps over the lazy dog. I am inevitable.", n=5):
        self.tts(text)  # warm-up
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(n):
            y = self.tts(text)
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t) / n
        secs = len(y) / self.audio.sr
        print(f"{secs:.2f}s audio in {dt * 1000:.0f} ms -> RTF {dt / secs:.3f} ({secs / dt:.0f}x realtime)")
