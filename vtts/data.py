import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .audio import load_wav


def load_meta(root):
    return json.load(open(Path(root) / "meta.json"))


def split_names(meta, val_frac=0.1, min_for_val=5):
    """Held-out clips per speaker (evenly spaced). Returns (train_names, val_names), both unique."""
    by_spk = {}
    for n in sorted(meta["index"]):
        by_spk.setdefault(meta["index"][n].get("spk", n.split("_")[0]), []).append(n)
    train, val = [], []
    for names in by_spk.values():
        k = max(1, round(len(names) * val_frac)) if len(names) >= min_for_val else 0
        v = {names[int((i + 0.5) * len(names) / k)] for i in range(k)} if k else set()
        val += sorted(v)
        train += [n for n in names if n not in v]
    return train, val


class AcousticDataset(Dataset):
    """split: 'train' (held-out removed, speakers repeated per meta['repeat']), 'val', or 'all' (no repeats)."""

    def __init__(self, root, split="train"):
        self.root = Path(root)
        self.meta = load_meta(root)
        train, val = split_names(self.meta)
        if split == "val":
            self.names = val
        elif split == "all":
            self.names = sorted(self.meta["index"])
        else:
            rep = self.meta.get("repeat", {})
            self.names = [n for n in train for _ in range(rep.get(self.meta["index"][n].get("spk"), 1))]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        return torch.load(self.root / "feats" / f"{self.names[i]}.pt")

    def batches(self, max_frames=10000, max_items=32, shuffle=True):
        """Length-bucketed batches capped by total padded frames (keeps VRAM flat)."""
        idx = sorted(range(len(self)), key=lambda i: self.meta["index"][self.names[i]]["frames"])
        out, cur, mx = [], [], 0
        for i in idx:
            f = self.meta["index"][self.names[i]]["frames"]
            if cur and (max(mx, f) * (len(cur) + 1) > max_frames or len(cur) >= max_items):
                out.append(cur)
                cur, mx = [], 0
            cur.append(i)
            mx = max(mx, f)
        if cur:
            out.append(cur)
        if shuffle:
            random.shuffle(out)
        return out

    def collate(self, ids):
        items = [self[i] for i in ids]
        B = len(items)
        N = max(len(d["tok"]) for d in items)
        T = max(d["mel"].shape[1] for d in items)
        n_mels = items[0]["mel"].shape[0]
        b = dict(tok=torch.zeros(B, N, dtype=torch.long), tlen=torch.zeros(B, dtype=torch.long),
                 mel=torch.zeros(B, n_mels, T), mlen=torch.zeros(B, dtype=torch.long),
                 pitch=torch.zeros(B, T), energy=torch.zeros(B, T), spk=torch.zeros(B, dtype=torch.long))
        for i, d in enumerate(items):
            n, t = len(d["tok"]), d["mel"].shape[1]
            b["tok"][i, :n], b["tlen"][i] = d["tok"], n
            b["mel"][i, :, :t], b["mlen"][i] = d["mel"], t
            b["pitch"][i, :t], b["energy"][i, :t], b["spk"][i] = d["pitch"], d["energy"], d["spk"]
        return b


class VocoderData(Dataset):
    """Aligned (waveform crop, mel crop) pairs. If data/pred/<name>.pt exists (acoustic-model output, see
    `cache-pred`), the vocoder is fed those blurry mels with probability pred_prob so it learns to
    clean up what the acoustic model actually produces."""

    def __init__(self, root, sr, hop=256, seg_frames=32, pred_prob=0.5):
        self.root, self.sr, self.hop, self.sf = Path(root), sr, hop, seg_frames
        self.names = sorted(load_meta(root)["index"])
        self.has_pred = (self.root / "pred").exists()
        self.pred_prob = pred_prob if self.has_pred else 0.0
        self.cache = {}

    def __len__(self):
        return len(self.names) * 8

    def __getitem__(self, i):
        n = self.names[i % len(self.names)]
        if n not in self.cache:
            y = torch.from_numpy(load_wav(self.root / "wavs" / f"{n}.wav", self.sr))
            mp = torch.load(self.root / "pred" / f"{n}.pt").float() if self.has_pred else None
            self.cache[n] = (y[: len(y) // self.hop * self.hop], mp)
        y, mp = self.cache[n]
        seg = self.sf * self.hop
        n_frames = len(y) // self.hop
        if n_frames <= self.sf:
            return torch.nn.functional.pad(y, (0, seg - len(y))), torch.zeros(mp.shape[0] if mp is not None else 80, self.sf), torch.tensor(0.0)
        s = random.randint(0, n_frames - self.sf)
        use_pred = mp is not None and random.random() < self.pred_prob
        mel = mp[:, s: s + self.sf] if use_pred else torch.zeros(mp.shape[0] if mp is not None else 80, self.sf)
        return y[s * self.hop: (s + self.sf) * self.hop], mel, torch.tensor(1.0 if use_pred else 0.0)
