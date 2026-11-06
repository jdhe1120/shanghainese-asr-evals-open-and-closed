"""Character error rate and per-clip diffs.

One alignment serves both: a Levenshtein table where two characters match if they are equal or, in the folded view,
a fold pair (normalize.fold_pairs). The diff is that table's backtrace, so it always agrees with the score.

The reference is first rewritten into the canonical spelling (normalize.canonical). Strict = the model must write that
spelling. Folded = the model may write any spelling of the word: a character matches if it equals or folds with the
canonical character OR the corpus's own character at that position (canonical() is one-for-one, so they line up).
"""
from .normalize import canonical as canon, fold_pairs, normalize


def _refs(reference: str, numerals: bool, canonical: bool) -> tuple[str, str]:
    """(reference the score is against, corpus spelling at the same positions)."""
    orig = normalize(reference, numerals)
    if not canonical:
        return orig, orig
    ref = normalize(canon(reference), numerals)
    return ref, (orig if len(orig) == len(ref) else ref)


def _align(ref: str, hyp: str, equiv: bool, alt: str | None = None) -> tuple[int, list[list[str]]]:
    pairs = fold_pairs() if equiv else frozenset()
    alt = alt if equiv and alt else ref
    match = lambda a, b: a == b or (a, b) in pairs  # noqa: E731
    same_at = lambda i, b: match(ref[i], b) or match(alt[i], b)  # noqa: E731
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (not same_at(i - 1, hyp[j - 1])))
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (not same_at(i - 1, hyp[j - 1])):
            ops.append(("eq" if same_at(i - 1, hyp[j - 1]) else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", hyp[j - 1]))
            j -= 1
    segs: list[list[str]] = []
    for kind, r, h in reversed(ops):  # merge runs of one kind
        if segs and segs[-1][0] == kind:
            segs[-1][1] += r
            segs[-1][2] += h
        else:
            segs.append([kind, r, h])
    return d[n][m], segs


def score(reference: str, hypothesis: str, equiv: bool = False, numerals: bool = False,
          canonical: bool = True) -> dict:
    ref, alt = _refs(reference, numerals, canonical)
    edits, _ = _align(ref, normalize(hypothesis, numerals), equiv, alt)
    return {"edits": edits, "refLen": len(ref), "cer": edits / len(ref) if ref else 0.0}


def diff(reference: str, hypothesis: str, equiv: bool = False, numerals: bool = False,
         canonical: bool = True) -> list[list[str]]:
    """Segments of [kind, refText, hypText]; kind is eq | sub | del | ins. In the folded view an "eq" segment can hold
    different characters on the two sides (a fold pair). refText is the canonical reference unless canonical=False."""
    ref, alt = _refs(reference, numerals, canonical)
    return _align(ref, normalize(hypothesis, numerals), equiv, alt)[1]


def summarize(scores: list[dict]) -> dict:
    """Mean per-utterance CER and corpus CER over a list of score() dicts."""
    if not scores:
        return {"meanCer": None, "corpusCer": None, "n": 0}
    total_ref = sum(s["refLen"] for s in scores)
    return {
        "meanCer": sum(s["cer"] for s in scores) / len(scores),
        "corpusCer": sum(s["edits"] for s in scores) / total_ref if total_ref else 0.0,
        "n": len(scores),
    }
