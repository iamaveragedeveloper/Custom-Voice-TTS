"""Turn raw voice recordings into a training set: split -> transcribe (optional, local) -> features."""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from . import text as T
from .audio import estimate_f0, load_wav, mel_spectrogram, save_wav
from .config import AudioConfig, to_dict


def split_audio(y, sr, min_len=1.0, max_len=12.0, min_gap=0.3, top_db=40.0):
    """Energy-based splitter: cut at silences >= min_gap, then merge pieces up to max_len."""
    hop = int(sr * 0.01)
    n = len(y) // hop
    if n == 0:
        return []
    rms = np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-10)
    db = 20 * np.log10(rms)
    voiced = db > np.percentile(db, 95) - top_db
    pieces, start, gap = [], None, 0
    for i, v in enumerate(voiced):
        if v:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap * 0.01 >= min_gap:
                pieces.append((start, i - gap + 1))
                start, gap = None, 0
    if start is not None:
        pieces.append((start, n))
    # force-cut over-long pieces at the quietest frame in the back half of the window
    max_f = int(max_len / 0.01)
    cut = []
    for a, b in pieces:
        while b - a > max_f:
            lo, hi = a + max_f // 2, a + max_f
            k = lo + int(np.argmin(db[lo:hi]))
            cut.append((a, k))
            a = k
        cut.append((a, b))
    pieces = cut
    merged = []
    for a, b in pieces:
        if merged and (b - merged[-1][0]) * 0.01 <= max_len:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    pad = int(0.08 * sr)
    out = []
    for a, b in merged:
        if (b - a) * 0.01 >= min_len:
            out.append(y[max(a * hop - pad, 0): b * hop + pad])
    return out


def _transcribe(paths, model_name):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    res = []
    for p in paths:
        segs, _ = model.transcribe(str(p), language="en", beam_size=5)
        res.append(" ".join(s.text.strip() for s in segs).strip())
        print(f"  {Path(p).name}: {res[-1]}")
    return res


def prep(input, out, metadata=None, whisper=None, phonemes=False, speaker="speaker0", max_len=12.0, repeat=None):
    cfg = AudioConfig()
    input, out = Path(input), Path(out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    (out / "feats").mkdir(exist_ok=True)
    rows = []  # (id, text, speaker)

    if metadata:  # "file|text[|speaker]" — files used as-is (only resampled)
        for r in csv.reader(open(metadata, encoding="utf-8"), delimiter="|"):
            if len(r) < 2:
                continue
            name = Path(r[0]).stem
            src = next(input.glob(name + ".*"))
            save_wav(out / "wavs" / f"{name}.wav", load_wav(src, cfg.sr), cfg.sr)
            rows.append((name, r[1], r[2] if len(r) > 2 else speaker))
    else:
        files = [input] if input.is_file() else sorted(p for p in input.rglob("*") if p.suffix.lower() in
                                                       (".wav", ".flac", ".mp3", ".ogg"))
        ids = []
        for f in files:
            for k, seg in enumerate(split_audio(load_wav(f, cfg.sr), cfg.sr, max_len=max_len)):
                seg = seg / max(np.abs(seg).max(), 1e-6) * 0.95
                name = f"{f.stem}_{k:04d}"
                save_wav(out / "wavs" / f"{name}.wav", seg, cfg.sr)
                ids.append(name)
        print(f"split into {len(ids)} segments")
        texts = _transcribe([out / "wavs" / f"{i}.wav" for i in ids], whisper) if whisper else None
        if texts is None:
            todo = out / "metadata_todo.csv"
            with open(todo, "w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh, delimiter="|")
                for i in ids:
                    w.writerow([i, ""])
            raise SystemExit(f"No transcripts. Either pass --whisper small (pip install faster-whisper) or fill in "
                             f"{todo} and re-run with --metadata. Segments are in {out / 'wavs'}.")
        rows = [(i, t, speaker) for i, t in zip(ids, texts) if t]

    spk_names = sorted({r[2] for r in rows})
    index = {}
    for name, txt, sp in rows:
        y = load_wav(out / "wavs" / f"{name}.wav", cfg.sr)
        y = y[: len(y) // cfg.hop * cfg.hop]
        yt = torch.from_numpy(y)
        mel = mel_spectrogram(yt[None], cfg)[0]
        tok = torch.tensor(T.encode(txt, phonemes), dtype=torch.long)
        if len(tok) == 0 or mel.shape[1] < len(tok):
            print("skip", name)
            continue
        torch.save(dict(tok=tok, mel=mel, f0=estimate_f0(yt, cfg), energy=mel.mean(0), spk=spk_names.index(sp)),
                   out / "feats" / f"{name}.pt")
        index[name] = dict(frames=mel.shape[1], tokens=len(tok), text=txt, spk=sp)

    # normalisation statistics, then rewrite features with normalised pitch/energy
    lf0, en = [], []
    for n in index:
        d = torch.load(out / "feats" / f"{n}.pt")
        v = d["f0"] > 0
        lf0.append(torch.log(d["f0"][v]))
        en.append(d["energy"])
    lf0, en = torch.cat(lf0), torch.cat(en)
    stats = dict(pitch_mean=lf0.mean().item(), pitch_std=lf0.std().item(),
                 energy_mean=en.mean().item(), energy_std=en.std().item())
    for n in index:
        p = out / "feats" / f"{n}.pt"
        d = torch.load(p)
        v = d["f0"] > 0
        pit = torch.zeros_like(d["f0"])
        pit[v] = (torch.log(d["f0"][v]) - stats["pitch_mean"]) / stats["pitch_std"]
        d["pitch"] = pit
        d["energy"] = (d["energy"] - stats["energy_mean"]) / stats["energy_std"]
        del d["f0"]
        torch.save(d, p)
    meta = dict(audio=to_dict(cfg), phonemes=phonemes, speakers=spk_names, stats=stats, index=index,
                n_vocab=T.N_VOCAB, repeat=repeat or {})
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    hrs = sum(v["frames"] for v in index.values()) * cfg.hop / cfg.sr / 3600
    print(f"prepared {len(index)} utterances, {hrs * 60:.1f} min, speakers={spk_names} -> {out}")
