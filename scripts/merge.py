"""Merge prepped datasets into one multi-speaker set. usage: merge.py OUT name=dir[:repeat] ..."""
import json, shutil, sys
from pathlib import Path
from vtts.prep import prep

out = Path(sys.argv[1]); src = out.parent / (out.name + "_src"); shutil.rmtree(src, ignore_errors=True); src.mkdir(parents=True)
rows, phon = [], None
for arg in sys.argv[2:]:
    spk, rest = arg.split("=", 1); d, _, rep = rest.partition(":")
    meta = json.load(open(Path(d) / "meta.json")); phon = meta["phonemes"]
    for n, v in meta["index"].items():
        for r in range(int(rep or 1)):
            nm = f"{spk}_{n}_r{r}"
            shutil.copy(Path(d) / "wavs" / f"{n}.wav", src / f"{nm}.wav")
            rows.append(f"{nm}|{v['text']}|{spk}")
(src / "metadata.csv").write_text("\n".join(rows), encoding="utf-8")
shutil.rmtree(out, ignore_errors=True)
prep(src, out, metadata=src / "metadata.csv", phonemes=phon)
