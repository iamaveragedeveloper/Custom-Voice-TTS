import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .audio import load_wav


def load_meta(root):
    return json.load(open(Path(root) / "meta.json"))


class AcousticDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.meta = load_meta(root)
        self.names = list(self.meta["index"])

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


class WavSegments(Dataset):
    """Random fixed-length waveform crops for vocoder training."""

    def __init__(self, root, sr, seg=8192):
        self.files = sorted((Path(root) / "wavs").glob("*.wav"))
        self.sr, self.seg = sr, seg
        self.cache = {}

    def __len__(self):
        return len(self.files) * 8  # several crops per file per epoch

    def __getitem__(self, i):
        f = self.files[i % len(self.files)]
        if f not in self.cache:
            self.cache[f] = torch.from_numpy(load_wav(f, self.sr))
        y = self.cache[f]
        if len(y) <= self.seg:
            y = torch.nn.functional.pad(y, (0, self.seg - len(y) + 1))
        s = random.randint(0, len(y) - self.seg - 1)
        return y[s: s + self.seg]
