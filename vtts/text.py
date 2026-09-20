"""Text frontend: normalisation, number expansion, char or ARPAbet (optional g2p_en) tokens."""
import re
import unicodedata

ARPA = ("AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH"
        ).split()
ARPA = [p + s for p in ARPA for s in ("0", "1", "2")] + ["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
        "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S",
        "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"]
ARPA = sorted(set(ARPA))
PUNCT = list(" !',-.?;:")
VOCAB = ["_"] + list("abcdefghijklmnopqrstuvwxyz") + PUNCT + ARPA
TOK = {s: i for i, s in enumerate(VOCAB)}
N_VOCAB = len(VOCAB)

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = [(10**9, "billion"), (10**6, "million"), (10**3, "thousand")]


def _say(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + _say(n % 100))
    for v, name in _SCALES:
        if n >= v:
            return _say(n // v) + " " + name + ("" if n % v == 0 else " " + _say(n % v))
    return str(n)


def _num(m):
    s = m.group(0)
    if "." in s:
        a, b = s.split(".")
        return _say(int(a)) + " point " + " ".join(_ONES[int(c)] for c in b)
    n = int(s)
    return _say(n) if n < 10**12 else " ".join(_ONES[int(c)] for c in s)


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    t = t.lower()
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)
    t = t.replace("%", " percent").replace("&", " and ").replace("$", " dollars ")
    t = re.sub(r"\d+(\.\d+)?", lambda m: " " + _num(m) + " ", t)
    t = re.sub(r"[^a-z !',\-.?;:]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


_g2p = None


def encode(text: str, phonemes: bool = False) -> list:
    global _g2p
    t = normalize(text)
    if phonemes:
        if _g2p is None:
            from g2p_en import G2p  # lazy, optional
            _g2p = G2p()
        syms = _g2p(t)
    else:
        syms = list(t)
    ids = [TOK[s] for s in syms if s in TOK]
    return ids


def split_sentences(text: str, max_chars: int = 220) -> list:
    parts = re.split(r"(?<=[.!?;:])\s+", text.strip())
    out, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) > max_chars:
            out.append(cur)
            cur = p
        else:
            cur = (cur + " " + p).strip()
    if cur:
        out.append(cur)
    return out
