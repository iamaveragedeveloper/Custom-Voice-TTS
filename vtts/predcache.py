"""Cache acoustic-model (teacher-forced) mels for every utterance -> data/pred/<name>.pt"""
from pathlib import Path

import torch

from .config import AcousticConfig
from .data import AcousticDataset
from .models.acoustic import AcousticModel


def cache_pred(data, acoustic, device=None):
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(acoustic, map_location=dev)
    ds = AcousticDataset(data, "all")
    model = AcousticModel(AcousticConfig(**ck["cfg"]), ck["audio"]["n_mels"], ck["n_vocab"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    out = Path(data) / "pred"
    out.mkdir(exist_ok=True)
    n = 0
    for ids in ds.batches(8000, shuffle=False):
        b = {k: v.to(dev) for k, v in ds.collate(ids).items()}
        pred = model.teacher_forced(b).cpu()
        for j, i in enumerate(ids):
            torch.save(pred[j, :, : int(b["mlen"][j])].half(), out / f"{ds.names[i]}.pt")
            n += 1
    print(f"cached predicted mels for {n} utterances -> {out}")
