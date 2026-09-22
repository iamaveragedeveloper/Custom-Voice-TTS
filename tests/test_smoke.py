"""End-to-end smoke test on synthetic audio: prep -> train acoustic -> train vocoder -> synth."""
import numpy as np

from vtts.audio import save_wav
from vtts.data import AcousticDataset
from vtts.predcache import cache_pred
from vtts.prep import prep, split_audio
from vtts.synth import Synthesizer
from vtts.text import normalize
from vtts.train import train_acoustic, train_vocoder

SR = 22050
WORDS = ["hello", "i am ultron", "there are no strings on me", "peace in our time", "twenty three humans"]


def fake_voice(seconds, f0):
    t = np.arange(int(SR * seconds)) / SR
    f = f0 * (1 + 0.1 * np.sin(2 * np.pi * 2 * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = sum(np.sin(k * ph) / k for k in range(1, 8)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) ** 2
    return (y / np.abs(y).max() * 0.8).astype(np.float32)


def test_normalize():
    assert normalize("I have 23 friends & 1,000 foes.") == "i have twenty three friends and one thousand foes."


def test_split():
    y = np.concatenate([fake_voice(2, 120), np.zeros(SR), fake_voice(2, 130)])
    assert len(split_audio(y, SR, max_len=3)) == 2


def test_pipeline(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    lines = []
    for i in range(12):
        save_wav(raw / f"u{i}.wav", fake_voice(1.5 + 0.1 * i, 100 + 5 * i), SR)
        lines.append(f"u{i}|{WORDS[i % len(WORDS)]}|{'a' if i < 8 else 'b'}")
    (tmp_path / "meta.csv").write_text("\n".join(lines), encoding="utf-8")
    data = tmp_path / "data"
    prep(raw, data, metadata=tmp_path / "meta.csv", repeat={"b": 3})
    tr, va = AcousticDataset(data, "train"), AcousticDataset(data, "val")
    assert len(va) >= 1 and not set(va.names) & set(tr.names)   # held-out clips never trained on
    assert sum(n.startswith("u8") for n in tr.names) in (0, 3)   # speaker b repeated 3x
    run = tmp_path / "run"
    train_acoustic(data, run, steps=6, max_frames=3000, log_every=2, save_every=3, eval_every=3)
    train_acoustic(data, run, steps=9, max_frames=3000, log_every=3, save_every=3, eval_every=3)  # resume
    cache_pred(data, run / "acoustic.pt")
    assert (data / "pred").exists() and (run / "acoustic_best.pt").exists()
    train_vocoder(data, run, steps=2, batch=2, seg=4096, small=True, log_every=1, save_every=1)
    s = Synthesizer(run / "acoustic.pt", run / "vocoder.pt")
    y = s.tts("Hello. I am Ultron.")
    assert y.ndim == 1 and len(y) > SR * 0.1 and np.isfinite(y).all()
    s2 = Synthesizer(run / "acoustic.pt")  # griffin-lim fallback
    assert np.isfinite(s2.tts("hello")).all()
