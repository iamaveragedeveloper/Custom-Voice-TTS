import argparse


def main():
    ap = argparse.ArgumentParser(prog="vtts")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prep", help="split/transcribe/featurise a voice recording")
    p.add_argument("--input", required=True, help="audio file or folder")
    p.add_argument("--out", required=True)
    p.add_argument("--metadata", help="'file|text[|speaker]' csv; skips splitting")
    p.add_argument("--whisper", help="local faster-whisper model (tiny/base/small/medium) to auto-transcribe")
    p.add_argument("--phonemes", action="store_true", help="ARPAbet frontend (needs g2p_en)")
    p.add_argument("--speaker", default="speaker0")

    p = sub.add_parser("train-acoustic")
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--max-frames", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--init", help="checkpoint to fine-tune from")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--amp", action="store_true", help="fp16 (do NOT use on GTX 16xx cards)")
    p.add_argument("--max-minutes", type=float, help="stop and save after this long")
    p.add_argument("--eval-every", type=int, default=500)

    p = sub.add_parser("train-vocoder")
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--small", action="store_true", help="128-channel generator (~4x faster)")
    p.add_argument("--init"); p.add_argument("--no-resume", action="store_true")
    p.add_argument("--g-only-steps", type=int, default=0, help="train generator on mel loss only for N steps first")
    p.add_argument("--max-minutes", type=float, help="stop and save after this long")
    p.add_argument("--d-warmup-steps", type=int, default=0, help="train only the discriminators for N steps first")
    p.add_argument("--pred-prob", type=float, default=0.5, help="share of steps fed acoustic-model mels")

    p = sub.add_parser("cache-pred", help="cache acoustic-model mels for vocoder training")
    p.add_argument("--data", required=True); p.add_argument("--acoustic", required=True)

    p = sub.add_parser("synth")
    p.add_argument("--text", required=True); p.add_argument("--acoustic", required=True)
    p.add_argument("--vocoder"); p.add_argument("--out", default="out.wav")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--semitones", type=float, default=0.0)
    p.add_argument("--pitch-var", type=float, default=1.0, help="pitch movement: >1 more expressive, <1 flatter")
    p.add_argument("--clean", type=float, default=1.5, help="noise reduction strength (1.5 moderate, 2.5 strong)")
    p.add_argument("--eq", help="matching-EQ file from scripts/build_eq.py (applied with --ultron-fx)")
    p.add_argument("--speaker", type=int, default=0)
    p.add_argument("--bench", action="store_true")
    p.add_argument("--ultron-fx", type=float, metavar="STRENGTH", help="apply the Ultron filter (0.6 = your chosen preset B)")

    a = ap.parse_args()
    if a.cmd == "prep":
        from .prep import prep
        prep(a.input, a.out, a.metadata, a.whisper, a.phonemes, a.speaker)
    elif a.cmd == "train-acoustic":
        from .train import train_acoustic
        train_acoustic(a.data, a.out, a.steps, a.max_frames, a.lr, init=a.init, resume=not a.no_resume, amp=a.amp,
                       max_minutes=a.max_minutes, eval_every=a.eval_every)
    elif a.cmd == "train-vocoder":
        from .train import train_vocoder
        train_vocoder(a.data, a.out, a.steps, a.batch, small=a.small, init=a.init, resume=not a.no_resume,
                      g_only_steps=a.g_only_steps, max_minutes=a.max_minutes, pred_prob=a.pred_prob,
                      d_warmup_steps=a.d_warmup_steps)
    elif a.cmd == "cache-pred":
        from .predcache import cache_pred
        cache_pred(a.data, a.acoustic)
    else:
        from .audio import save_wav
        from .synth import Synthesizer
        s = Synthesizer(a.acoustic, a.vocoder)
        if a.bench:
            s.bench(a.text)
        y = s.tts(a.text, speaker=a.speaker, speed=a.speed, semitones=a.semitones, pitch_var=a.pitch_var)
        if a.ultron_fx:
            from .fx import ultron
            import numpy as np
            eq = np.load(a.eq)["gain_db"] if a.eq else None
            # preset B (no grit, light metal); with an EQ the deeper layer and top-end cut are dropped
            y = ultron(y, s.audio.sr, intensity=a.ultron_fx, sat=0.0, metal=0.25, down=0.0 if eq is not None else -3.0,
                       lowpass=None if eq is not None else 7500, eq=eq, clean=a.clean)
        save_wav(a.out, y, s.audio.sr)
        print(f"wrote {a.out} ({len(y) / s.audio.sr:.2f}s)")


if __name__ == "__main__":
    main()
