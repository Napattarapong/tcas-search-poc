"""Compose Thai prose from SQL rows and/or vector chunks. Validates citations.

Signature: compose_prose(question, rows=None, chunks=None) -> (prose, sources)
  - `rows` is a list of dicts (each must have 'table' and 'id' keys).
  - `chunks` is a list of dicts (each must have 'id' and 'source_document_id').
  - `sources` is a list of {"type": "row"|"chunk", "id": int, ...} that were
    actually referenced by surviving sentences.
"""
from __future__ import annotations
import json
import re
from src.llm import chat, LLMError
from src.validator import validate_prose

_CITE_RE = re.compile(
    r"\[src=(admission_rounds|programs|universities|cutoff_scores|requirements|chunks?)#(\d+)(?:,field=(\w+))?\]"
)

COMPOSE_PROMPT = """You are a careful Thai-language assistant. Write 1-3 short Thai sentences summarizing the data below for the user's question. Every sentence MUST end with a citation token `[src=table#id]` (for DB rows) or `[src=chunk#id]` (for text chunks), optionally with `,field=col_name`. Do NOT add any fact that is not present in the sources. If a source has no useful data, omit it. Begin directly with the Thai sentence — no preamble, no "Answer:".

Question: {question}

DB rows (JSON):
{rows_json}

Text chunks (JSON):
{chunks_json}
"""


def compose_prose(
    question: str,
    rows: list[dict] | None,
    chunks: list[dict] | None,
) -> tuple[str, list[dict]]:
    """Call the LLM, validate citations, return (prose, sources_used)."""
    if not (rows or chunks):
        return "", []
    rows_json = json.dumps(rows or [], ensure_ascii=False, indent=2)
    chunks_json = json.dumps(chunks or [], ensure_ascii=False, indent=2)
    prompt = COMPOSE_PROMPT.format(
        question=question, rows_json=rows_json, chunks_json=chunks_json
    )
    try:
        raw = chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
    except LLMError as e:
        raise LLMError(f"compose_prose: LLM call failed: {e}") from e

    prose = validate_prose(raw)

    # Extract cited ids actually used
    row_ids_by_table: dict[str, set[int]] = {}
    chunk_ids: set[int] = set()
    for m in _CITE_RE.finditer(prose):
        kind, raw_id = m.group(1), int(m.group(2))
        # The regex accepts both `chunk` and `chunks` for the chunk kind.
        if kind in ("chunk", "chunks"):
            chunk_ids.add(raw_id)
        else:
            row_ids_by_table.setdefault(kind, set()).add(raw_id)

    sources: list[dict] = []
    if rows:
        for r in rows:
            tbl = r.get("table")
            rid = r.get("id")
            if tbl and rid is not None and rid in row_ids_by_table.get(tbl, set()):
                sources.append({"type": "row", "table": tbl, "id": rid, **r})
    if chunks:
        for c in chunks:
            if c["id"] in chunk_ids:
                sources.append({"type": "chunk", **c})
    return prose, sources
