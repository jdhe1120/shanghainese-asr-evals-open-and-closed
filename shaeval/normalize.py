"""Text normalisation and the Wu spelling folds used for scoring.

normalize(): NFKC, then optional digits -> Chinese numerals, then OpenCC t2s (the references are Simplified; some
models write Traditional), then drop punctuation, symbols and whitespace.

fold_pairs(): the character pairs rules/wu_folds.toml says are the same word. Folds are pairs, not a translate table:
nothing chains across entries (末=么 and 么=嘛 don't make 末=嘛), so metrics.py compares characters with them instead
of rewriting the text.

canonical(): the reference rewritten into the one spelling convention the strict score holds models to
(rules/wu_canonical.toml: 勿 for veq, 伐 only for va, 个, 我, 人 ...). One character for one character, so the folded
view can still line the canonical reference up with the corpus spelling.
"""
import re
import tomllib
import unicodedata
from functools import lru_cache
from itertools import permutations
from pathlib import Path

import opencc

RULES = Path(__file__).resolve().parent.parent / "rules" / "wu_folds.toml"
CANON = RULES.with_name("wu_canonical.toml")


@lru_cache(maxsize=1)
def _t2s() -> opencc.OpenCC:
    return opencc.OpenCC("t2s")


def fold_rules() -> list[dict]:
    with RULES.open("rb") as f:
        return tomllib.load(f)["fold"]


@lru_cache(maxsize=1)
def fold_pairs() -> frozenset[tuple[str, str]]:
    """Ordered (a, b) pairs, both directions, from every "same" entry; a "different" entry removes its pairs."""
    pairs = lambda g: set(permutations(g, 2))  # noqa: E731
    rules = fold_rules()
    same = set().union(*(pairs(r["graphs"]) for r in rules if r["decision"] == "same"))
    different = set().union(*(pairs(r["graphs"]) for r in rules if r["decision"] == "different"))
    return frozenset(same - different)


@lru_cache(maxsize=1)
def _rewrites() -> list[tuple[re.Pattern, str]]:
    with CANON.open("rb") as f:
        rules = tomllib.load(f)["rewrite"]
    for r in rules:
        assert len(r["to"]) == 1, f"canonical rule {r['pattern']!r}: `to` must be one character"
    return [(re.compile(r["pattern"]), r["to"]) for r in rules]


def canonical(text: str) -> str:
    """The reference in the canonical spelling. Run on the raw text, before normalize(): the rules look at punctuation
    for context."""
    for pat, to in _rewrites():
        out = pat.sub(to, text)
        assert len(out) == len(text), f"canonical rule {pat.pattern!r} changed the length"
        text = out
    return text


_DIGITS = "零一二三四五六七八九"
_UNITS = ["", "十", "百", "千"]


def _group(n: int) -> str:
    """0 < n < 10000 as Chinese: 12 -> 十二, 105 -> 一百零五, 2016 -> 二千零一十六."""
    out, zero = "", False
    for i, ch in enumerate(f"{n:04d}"):
        d, unit = int(ch), _UNITS[3 - i]
        if d == 0:
            zero = bool(out)
        else:
            if zero:
                out += "零"
            zero = False
            out += ("" if (d == 1 and unit == "十" and not out) else _DIGITS[d]) + unit
    return out


def _cardinal(n: int) -> str:
    if n == 0:
        return "零"
    if n >= 10**8:
        return "".join(_DIGITS[int(c)] for c in str(n))  # too big to read as a quantity
    hi, lo = divmod(n, 10000)
    out = (_group(hi) + "万") if hi else ""
    if lo:
        out += ("零" if hi and lo < 1000 else "") + _group(lo)
    return out


def _number(m: re.Match) -> str:
    whole, frac, tail = m.group("int"), m.group("frac"), m.group("tail")
    spoken_digits = lambda d: "".join(_DIGITS[int(c)] for c in d)  # noqa: E731
    # Years, long runs (phone/ID numbers) and leading zeros are read digit by digit.
    if tail == "年" or len(whole) >= 7 or (len(whole) > 1 and whole[0] == "0"):
        body = spoken_digits(whole)
    else:
        body = _cardinal(int(whole))
    return body + ("点" + spoken_digits(frac) if frac else "") + (tail or "")


def numerals_to_chinese(text: str) -> str:
    """Arabic digits -> Chinese numerals (87年 -> 八七年, 50万 -> 五十万). MagicData writes numbers in characters."""
    return re.sub(r"(?P<int>\d+)(?:\.(?P<frac>\d+))?(?P<tail>年)?", _number, text)


def normalize(text: str, numerals: bool = False) -> str:
    text = unicodedata.normalize("NFKC", text)
    if numerals:
        text = numerals_to_chinese(text)
    text = _t2s().convert(text)
    return "".join(ch for ch in text if unicodedata.category(ch)[0] not in "PSZC")
