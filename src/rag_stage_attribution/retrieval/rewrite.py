"""Deterministic query rewrite: expand the acronyms the corpus spells out.

This node exists as an instrumentation point as much as a feature. The trace
records the query before and after, so a retrieval failure can be pinned to
the rewrite when the rewrite caused it. The table is committed data, and the
embedding fixture stores vectors for the rewritten form, which is why the
rewrite must be a pure function of the query text: the fixture was captured
through this exact function, and a drifted rewrite would silently look up the
wrong vector. The fixture manifest pins this table's hash for that reason.
"""

from __future__ import annotations

import hashlib
import json
import re

# Expansion keeps the original token and appends the corpus vocabulary,
# because dropping the acronym loses signal when the corpus uses it too.
REWRITE_TABLE: dict[str, str] = {
    "PTO": "PTO paid time off",
    "VPN": "VPN virtual private network remote access",
    "SEV1": "SEV1 severity one incident",
    "SEV2": "SEV2 severity two incident",
    "SEV3": "SEV3 severity three incident",
    "MFA": "MFA multi factor authentication",
    "RTO": "RTO recovery time objective",
    "RPO": "RPO recovery point objective",
    "PO": "PO purchase order",
}

_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(REWRITE_TABLE, key=len, reverse=True)) + r")\b"
)


def rewrite_query(query: str) -> str:
    return _PATTERN.sub(lambda m: REWRITE_TABLE[m.group(1)], query)


def rewrite_table_sha() -> str:
    """Stable hash of the table, pinned by the embedding fixture manifest."""
    blob = json.dumps(REWRITE_TABLE, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
