"""Wake-word detection for the name "Seeker" (tolerant of ASR mis-hearings)."""
from __future__ import annotations

import difflib
import re

_WORD = re.compile(r"[a-z0-9']+")
DEFAULT_CORE = "seeker"
# Observed ASR mis-hearings of a leading "Seeker".  They only count as a wake
# word when they are the very first word and a command follows, so phrases like
# "the speaker is loud" stay ignored.
DEFAULT_LEADING = ("secret",)


def _tokens(text: str) -> list[str]:
    return _WORD.findall((text or "").lower().replace("'", ""))


def _normalize_alias(alias: str) -> list[str]:
    return _tokens(alias)


def _similar(token: str, core: str, cutoff: float) -> bool:
    if token == core:
        return True
    if len(token) < max(4, len(core) - 2):
        return False
    return difflib.SequenceMatcher(None, token, core).ratio() >= cutoff


def detect(text: str, aliases: list[str] | None = None, core: str = DEFAULT_CORE,
           cutoff: float = 0.86, leading_mishears=None) -> tuple[bool, str, str | None]:
    """Return (matched, command_text, alias_used).

    command_text is the transcript with the wake phrase removed, so
    "seeker turn on the lights" -> "turn on the lights".  A bare "seeker"
    yields an empty command.
    """
    tokens = _tokens(text)
    if not tokens:
        return False, "", None

    leading = set(leading_mishears if leading_mishears is not None else DEFAULT_LEADING)

    alias_seqs = [s for s in (_normalize_alias(a) for a in (aliases or [])) if s]
    if not any(s == [core] for s in alias_seqs):
        alias_seqs.append([core])
    singles = {w for seq in alias_seqs if len(seq) == 1 for w in seq}

    # 1. Multi-word aliases anywhere ("hey seeker", "ok seeker", ...) - exact.
    for seq in sorted(alias_seqs, key=len, reverse=True):
        n = len(seq)
        if n < 2:
            continue
        for i in range(len(tokens) - n + 1):
            window = tokens[i:i + n]
            if window == seq:
                return True, " ".join(tokens[i + n:]), " ".join(window)

    # 2. Single-word aliases (exact) or a fuzzy match to the core name, but only
    # near the start of the utterance - the wake word leads, media chatter does not.
    for i, token in enumerate(tokens[:2]):
        if token in singles or _similar(token, core, cutoff):
            return True, " ".join(tokens[i + 1:]), token

    # 3. Known leading mis-hearings ("secret", "see kurt"), only at the very
    # start and only when a command follows - media chatter stays ignored.
    for n in (2, 1):
        if len(tokens) > n and " ".join(tokens[:n]) in leading:
            return True, " ".join(tokens[n:]), " ".join(tokens[:n])

    return False, "", None
