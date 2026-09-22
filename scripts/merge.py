"""Merge prepped datasets into one multi-speaker set.
usage: merge.py OUT name=dir[:repeat] ...   (repeat = how often that speaker's clips are seen per epoch)"""
import json, shutil, sys
from pathlib import Path
from vtts.prep import prep

out = Path(sys.argv[1]); src = out.parent / (out.name + "_src"); shutil.rmtree(src, ignore_errors=True); src.mkdir(parents=True)
rows, phon, repeat = [], None, {}
for arg in sys.argv[2:]:
    spk, rest = arg.split("=", 1); d, _, rep = rest.partition(":")
    repeat[spk] = int(rep or 1)
    meta = json.load(open(Path(d) / "meta.json")); phon = meta["phonemes"]
    for n, v in meta["index"].items():
        nm = f"{spk}_{n}"
        shutil.copy(Path(d) / "wavs" / f"{n}.wav", src / f"{nm}.wav")
        rows.append(f"{nm}|{v['text']}|{spk}")
(src / "metadata.csv").write_text("\n".join(rows), encoding="utf-8")
shutil.rmtree(out, ignore_errors=True)
prep(src, out, metadata=src / "metadata.csv", phonemes=phon, repeat=repeat)
