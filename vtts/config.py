from dataclasses import dataclass, field, asdict


@dataclass
class AudioConfig:
    sr: int = 22050
    n_fft: int = 1024
    hop: int = 256
    win: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0


@dataclass
class AcousticConfig:
    d_model: int = 256
    heads: int = 2
    enc_layers: int = 4
    dec_layers: int = 4
    ff: int = 512
    kernel: int = 9
    dropout: float = 0.1
    pred_ch: int = 256
    pred_kernel: int = 3
    aligner_dim: int = 128
    n_speakers: int = 1


@dataclass
class VocoderConfig:
    channels: int = 256  # 128 = "small" preset (~4x cheaper)
    upsample_rates: list = field(default_factory=lambda: [8, 8, 2, 2])  # product must equal hop
    upsample_kernels: list = field(default_factory=lambda: [16, 16, 4, 4])
    resblock_kernels: list = field(default_factory=lambda: [3, 7, 11])
    resblock_dilations: list = field(default_factory=lambda: [[1, 3, 5]] * 3)


def to_dict(c):
    return asdict(c)
