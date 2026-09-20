import math
import time

import numpy as np
import torch

from . import text as T
from .audio import griffin_lim
from .config import AcousticConfig, AudioConfig, VocoderConfig
from .models.acoustic import AcousticModel
from .models.vocoder import Generator


class Synthesizer:
    """Loads acoustic + vocoder checkpoints; fp16 on GPU. Without a vocoder falls back to Griffin-Lim."""

    def __init__(self, acoustic, vocoder=None, device=None, fp16=True):
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
    def mel(self, text, speaker=0, speed=1.0, semitones=0.0):
        tok = torch.tensor([T.encode(text, self.phonemes)], device=self.dev)
        if tok.numel() == 0:
            return None
        spk = torch.tensor([speaker], device=self.dev)
        shift = semitones * math.log(2) / 12 / self.stats["pitch_std"]
        with torch.autocast(self.dev.type, dtype=torch.float16, enabled=self.amp):
            return self.am.infer(tok, spk, speed, shift)

    @torch.no_grad()
    def wave(self, mel):
        if self.voc is None:
            return griffin_lim(mel[0].float(), self.audio).cpu().numpy()
        with torch.autocast(self.dev.type, dtype=torch.float16, enabled=self.amp):
            return self.voc(mel)[0, 0].float().cpu().numpy()

    def stream(self, text, **kw):
        """Yield one waveform chunk per sentence so playback can start before the full text is done."""
        for s in T.split_sentences(text):
            m = self.mel(s, **kw)
            if m is not None:
                yield self.wave(m)
                yield np.zeros(int(0.12 * self.audio.sr), np.float32)

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
