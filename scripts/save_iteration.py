"""Save a voice you liked as the next numbered iteration: iterations/iteration_N.wav + a row in iterations/README.md.

usage: python scripts/save_iteration.py outputs/pure_final.wav --text "..." --changes "..." --command "..." --stats "..."
"""
import argparse
import re
import shutil
from pathlib import Path

import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("wav")
ap.add_argument("--text", required=True, help="the line that was spoken")
ap.add_argument("--changes", required=True, help="what changed vs the previous iteration")
ap.add_argument("--command", required=True, help="command that reproduces it")
ap.add_argument("--stats", required=True, help="timing / quality numbers")
a = ap.parse_args()

root = Path("iterations")
root.mkdir(exist_ok=True)
readme = root / "README.md"
head = ("# Voice iterations\n\nEach iteration is a voice line you picked as good. GitHub plays the `.wav` files "
        "directly: click a file, then the play button.\n\n")
body = readme.read_text(encoding="utf-8") if readme.exists() else head
n = max([int(m) for m in re.findall(r"^## Iteration (\d+)", body, re.M)] + [0]) + 1
dst = root / f"iteration_{n}.wav"
shutil.copyfile(a.wav, dst)
info = sf.info(dst)
body += (f"\n## Iteration {n}\n\n"
         f"- **Audio:** [`{dst.name}`]({dst.name}) ({info.duration:.2f} s, {info.samplerate} Hz)\n"
         f"- **Line:** \"{a.text}\"\n"
         f"- **What changed:** {a.changes}\n"
         f"- **Speed / quality:** {a.stats}\n"
         f"- **Reproduce:**\n\n```bash\n{a.command}\n```\n")
readme.write_text(body, encoding="utf-8")
print(f"saved {dst} and updated {readme}")
