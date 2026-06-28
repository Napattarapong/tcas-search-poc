"""Citation validator.

A sentence survives only if it ends with a citation token of the form
[src=table#id] or [src=chunk#id], optionally with ,field=col_name.

Recognized tables: admission_rounds, programs, universities,
cutoff_scores, requirements, chunks.
"""
from __future__ import annotations
import re

_CITATION_RE = re.compile(
    r"\[src=(admission_rounds|programs|universities|cutoff_scores|requirements|chunks?)#(\d+)(?:,field=(\w+))?\]\s*\.?\s*$"
)


def validate_prose(prose: str) -> str:
    """Drop sentences that lack a citation. Return remaining joined by newlines."""
    if not prose:
        return ""
    sentences = [s.strip() for s in re.split(r"(?<=[。!?])\s+|\n+", prose) if s.strip()]
    kept = [s for s in sentences if _CITATION_RE.search(s)]
    return "\n".join(kept)
