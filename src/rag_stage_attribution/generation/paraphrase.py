"""Seeded paraphrase that never touches a protected token.

The stand-in's faithful answers are paraphrases of corpus sentences, and the
whole measurement depends on two properties holding at once: the wording must
drift (so lexical support inference is genuinely noisy, as it is against a
real model) and the factual payload must not (so a faithful answer is never
scored wrong by accident). Protection is therefore structural: any token
containing a digit or a currency sign, and every word of the caller's protect
phrase, passes through unchanged no matter what the synonym table says.
"""

from __future__ import annotations

import random
import re

# Corpus vocabulary to everyday vocabulary. One direction only; the table is
# applied to corpus sentences, never to queries.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "requires": ("needs", "calls for"),
    "require": ("need", "call for"),
    "required": ("needed", "mandatory"),
    "approval": ("sign off", "a green light"),
    "approved": ("signed off", "cleared"),
    "submitted": ("turned in", "filed"),
    "reimbursed": ("paid back", "covered"),
    "reimbursable": ("claimable", "coverable"),
    "retained": ("kept", "held"),
    "deleted": ("removed", "purged"),
    "automatically": ("on its own", "without anyone acting"),
    "receive": ("get", "are given"),
    "receives": ("gets", "is given"),
    "employee": ("staff member", "person"),
    "employees": ("staff", "people"),
    "within": ("inside", "in under"),
    "must": ("has to", "needs to"),
    "may": ("can", "is allowed to"),
    "begins": ("starts", "kicks off"),
    "reported": ("flagged", "raised"),
    "notification": ("notice", "an alert"),
    "mandatory": ("compulsory", "non optional"),
    "purchase": ("buy", "order"),
    "vendor": ("supplier", "provider"),
    "request": ("ask", "ticket"),
    "requests": ("asks", "tickets"),
    "capped": ("limited", "topped out"),
    "expire": ("lapse", "run out"),
    "expires": ("lapses", "runs out"),
}

_TOKEN = re.compile(r"\S+")


def _protected(token: str, protect_words: frozenset[str]) -> bool:
    bare = token.strip(".,;:()").lower()
    return any(ch.isdigit() for ch in token) or "$" in token or bare in protect_words


def paraphrase(sentence: str, rng: random.Random, protect: str = "") -> str:
    """Rewrite a sentence with seeded synonym drift, payload intact."""
    protect_words = frozenset(w.strip(".,;:").lower() for w in protect.split())

    def swap(match: re.Match[str]) -> str:
        token = match.group(0)
        if _protected(token, protect_words):
            return token
        stripped = token.strip(".,;:()")
        trailing = token[len(stripped):] if token.endswith((".", ",", ";", ":")) else ""
        key = stripped.lower()
        if key in SYNONYMS and rng.random() < 0.8:
            choice = rng.choice(SYNONYMS[key])
            if stripped[0].isupper():
                choice = choice[0].upper() + choice[1:]
            return choice + trailing
        return token

    out = _TOKEN.sub(swap, sentence)
    # A seeded front rotation drifts sentence openings without moving facts.
    openers = ("", "Per the policy, ", "As written, ", "To be specific, ")
    opener = rng.choice(openers)
    if opener:
        if out and out[0].isupper() and out[1:2].islower():
            out = out[0].lower() + out[1:]
        out = opener + out
    return out
