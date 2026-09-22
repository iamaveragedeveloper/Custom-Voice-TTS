"""Builds colab/train_ultron_v3.ipynb and colab/ultron_project_v3.zip"""
import json
import os
import zipfile

with zipfile.ZipFile('colab/ultron_project_v3.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for root in ('vtts', 'data/all'):
        for d, _, fs in os.walk(root):
            if '__pycache__' in d:
                continue
            for f in fs:
                z.write(os.path.join(d, f))
    z.write('runs/best/acoustic.pt', 'runs/ultron/acoustic_init.pt')   # v1 acoustic (step 8000, with optimizer)
    z.write('runs/best/vocoder.pt', 'runs/ultron/vocoder_init.pt')     # v1 vocoder generator
print('zip MB', round(os.path.getsize('colab/ultron_project_v3.zip') / 1e6, 1))

md = lambda s: dict(cell_type='markdown', metadata={}, source=s.strip().splitlines(True))
code = lambda s: dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s.strip().splitlines(True))
HEAD = ("import os\nBASE = '/content/ultron'   # defined in every cell so it survives a runtime restart\n"
        "assert os.path.exists(f'{BASE}/vtts'), 'Project files are gone (runtime restarted) - run cell 1 again and upload the zip.'\n")

cells = [
    md("""# Ultron voice v3 - resumable, no Google Drive
Runtime -> Change runtime type -> **T4 GPU**. Run cells top to bottom.

**What is new vs v2:** validation on held-out clips (keeps the *best* acoustic checkpoint), resume-safe learning rate,
a vocoder that trains on the acoustic model's own output, a discriminator warm-up, and a spectral loss that targets the
periodic 'pulse' buzz.

**Resuming:** the last cell downloads `ultron_project_next.zip`. Next session upload *that* in cell 1 and just re-run.
Colab wipes its disk when the session ends - download before closing the tab."""),
    code("""
import os
BASE = '/content/ultron'
if not os.path.exists(f'{BASE}/vtts'):
    from google.colab import files
    up = files.upload()   # ultron_project_v3.zip the first time, ultron_project_next.zip later
    name = list(up)[0]
    !mkdir -p {BASE} && unzip -q -o "{name}" -d {BASE}
    !rm -f "{name}"
!ls {BASE}/runs/ultron
!pip -q install soundfile g2p_en
import nltk
for p in ['averaged_perceptron_tagger','averaged_perceptron_tagger_eng','cmudict']: nltk.download(p, quiet=True)
import torch; print(torch.cuda.get_device_name(0))
"""),
    md("""### Cell 2: cache the acoustic model's own spectrograms (about 1 minute)
Uses the best acoustic checkpoint available. Re-run at the start of every session."""),
    code(HEAD + """
ck = next(p for p in ['acoustic_best.pt', 'acoustic_last.pt', 'acoustic_init.pt'] if os.path.exists(f'{BASE}/runs/ultron/{p}'))
print('using', ck)
!cd {BASE} && PYTHONPATH={BASE} python -W ignore -m vtts cache-pred --data data/all --acoustic runs/ultron/{ck}
"""),
    md("""### Cell 3: train (stops and saves by itself after MINUTES)
`*_STEPS` are **total** targets - raise them each session. Watch the `[val @ ...]` lines: if `mel` stops
falling (or rises), the acoustic model has peaked - stop raising `ACOUSTIC_STEPS`; `acoustic_best.pt` keeps the best one."""),
    code(HEAD + """
import subprocess, time
ACOUSTIC_STEPS = 12000     # total steps counted from now (starts from the v1 weights)
VOCODER_STEPS  = 20000     # total
MINUTES = 60
R = f'{BASE}/runs/ultron'
env = f'cd {BASE} && PYTHONPATH={BASE}'
a_init = '--init runs/ultron/acoustic_init.pt' if not os.path.exists(f'{R}/acoustic_last.pt') else ''
v_init = '--init runs/ultron/vocoder_init.pt' if not os.path.exists(f'{R}/vocoder_last.pt') else ''
d_warm = 1500 if v_init else 0
ac = subprocess.Popen(f'{env} python -u -m vtts train-acoustic --data data/all --out runs/ultron --steps {ACOUSTIC_STEPS} --lr 3e-4 --amp {a_init} --max-minutes {MINUTES} > ac.log 2>&1', shell=True)
vo = subprocess.Popen(f'{env} python -u -m vtts train-vocoder --data data/all --out runs/ultron --steps {VOCODER_STEPS} --small {v_init} --d-warmup-steps {d_warm} --max-minutes {MINUTES} > voc.log 2>&1', shell=True)
while ac.poll() is None or vo.poll() is None:
    time.sleep(120)
    print('ACOUSTIC:', subprocess.getoutput(f'grep -E "val @|^step" {BASE}/ac.log | tail -n 2')[:260])
    print('VOCODER :', subprocess.getoutput(f'tail -n 1 {BASE}/voc.log')[:150])
print('done', ac.returncode, vo.returncode)
"""),
    md("If any log line shows `nan`, tell me before continuing."),
    code(HEAD + """
import sys; sys.path.insert(0, BASE)
from vtts.synth import Synthesizer
from IPython.display import Audio, display
ac = f'{BASE}/runs/ultron/acoustic_best.pt'
s = Synthesizer(ac if os.path.exists(ac) else f'{BASE}/runs/ultron/acoustic.pt', f'{BASE}/runs/ultron/vocoder.pt')
text = "I am Ultron, I come for peace and I want the Avengers extinction"
for spk, name in enumerate(s.speakers):
    y = s.tts(text, speaker=spk, speed=0.85); print(name); display(Audio(y, rate=s.audio.sr))
"""),
    md("### Last cell: download both (do this before closing!)\n"
       "`ultron_models.zip` = the files you use on your PC. `ultron_project_next.zip` = upload it next session to continue."),
    code(HEAD + """
!cd {BASE}/runs/ultron && cp -n acoustic.pt acoustic_best.pt; zip -q -j /content/ultron_models.zip acoustic_best.pt vocoder.pt
!cd {BASE} && zip -q -r /content/ultron_project_next.zip vtts data/all runs/ultron/acoustic_last.pt runs/ultron/acoustic_best.pt runs/ultron/vocoder_last.pt
from google.colab import files
files.download('/content/ultron_models.zip')
files.download('/content/ultron_project_next.zip')
"""),
]
nb = dict(cells=cells, metadata=dict(accelerator='GPU', colab=dict(provenance=[]),
                                     kernelspec=dict(name='python3', display_name='Python 3')), nbformat=4, nbformat_minor=0)
json.dump(nb, open('colab/train_ultron_v3.ipynb', 'w'), indent=1)
print('notebook written')
