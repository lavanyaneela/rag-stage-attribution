"""Decide whether an answer is right, so attribution knows what to explain.

Correctness is containment of the golden key phrase after normalisation. The
key phrases are short factual payloads (an amount, a duration, an owner), the
paraphraser is forbidden from touching them, and the parametric bank is
forbidden from equalling them, so containment is exact for this fixture by
construction. An injected fault can still produce an accidentally correct
answer when a distractor chunk happens to state the same figure; that answer
is simply correct, because attribution explains failures, not intentions.
"""

from __future__ import annotations

import re

_KEEP = re.compile(r"[a-z0-9$.%]+")


def normalise(text: str) -> str:
    """Lowercase, keep payload characters, and strip edge periods per token.

    The edge strip exists because of a measured bug: a payload ending a
    sentence tokenised as '$25.' and failed a word boundary match against
    '$25', so a faithful answer scored wrong whenever its fact closed the
    sentence. Interior dots survive, which '$0.62' depends on.
    """
    return " ".join(token.strip(".") for token in _KEEP.findall(text.lower()))


def is_correct(answer: str, key_phrase: str) -> bool:
    """Word boundary containment: 'IT' must not match inside 'items'."""
    return f" {normalise(key_phrase)} " in f" {normalise(answer)} "
