"""Rebuild data/<name>/feats/ from wavs/ + meta.json (used on Colab: the uploaded zip ships wavs only, to
keep it small - feats/ (mel spectrograms etc.) are cheap to recompute and don't need Whisper, since the
transcripts are already in meta.json).

usage: python scripts/regen_feats.py [data/all]
"""
import csv
import json
import sys
from pathlib import Path

from vtts.prep import prep

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/all")
if (root / "feats").exists():
    print(f"{root}/feats already present, nothing to do")
    raise SystemExit(0)

meta = json.load(open(root / "meta.json"))
csv_path = root / "_regen_metadata.csv"
with open(csv_path, "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh, delimiter="|")
    for name, v in meta["index"].items():
        w.writerow([name, v["text"], v.get("spk", "speaker0")])

prep(root / "wavs", root, metadata=csv_path, phonemes=meta["phonemes"], repeat=meta.get("repeat", {}))
csv_path.unlink()
print(f"rebuilt {root}/feats from {root}/wavs + meta.json")
