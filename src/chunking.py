"""Thai-aware paragraph chunking for vector indexing.

Strategy: split by Thai sentence boundaries (pythainlp), accumulate
sentences into a buffer until the token budget (rough word count) is
reached, then emit a chunk. Carry the last `overlap` words into the
next chunk for context continuity.
"""
from __future__ import annotations
from typing import List


def _split_sentences(text: str) -> List[str]:
    """Split a Thai string into sentences. Falls back to whitespace + '।' if pythainlp fails."""
    text = text.strip()
    if not text:
        return []
    try:
        from pythainlp.tokenize import sent_tokenize
        sents = sent_tokenize(text, engine="whitespace+newline")
        return [s.strip() for s in sents if s.strip()]
    except Exception:
        # Fallback: split on common Thai/English sentence terminators
        import re
        parts = re.split(r"(?<=[\.\!\?।])\s+", text)
        return [p.strip() for p in parts if p.strip()]


def _approx_tokens(text: str) -> int:
    """Rough token count: split on whitespace. Thai has no spaces, so
    approximate 1 token per ~4 characters of Thai text."""
    if not text.strip():
        return 0
    # Mixed heuristic: count whitespace-separated words + Thai char groups
    import re
    en_words = len(re.findall(r"\S+", text))
    thai_chars = len(re.findall(r"[฀-๿]", text))
    return en_words + max(1, thai_chars // 4)


def chunk_markdown(text: str, max_tokens: int = 300, overlap: int = 50) -> List[str]:
    """Chunk `text` into pieces of ~`max_tokens` with `overlap` words carryover."""
    sents = _split_sentences(text)
    if not sents:
        return []

    chunks: List[str] = []
    buf: List[str] = []
    buf_tokens = 0

    def _emit():
        nonlocal buf, buf_tokens
        if buf:
            chunks.append(" ".join(buf).strip())
        # carry overlap (last `overlap` approx-words) into next chunk
        if overlap > 0 and buf:
            tail = " ".join(buf)
            tail_words = tail.split()
            carry = " ".join(tail_words[-overlap:]) if len(tail_words) > overlap else tail
            buf = [carry]
            buf_tokens = _approx_tokens(carry)
        else:
            buf = []
            buf_tokens = 0

    for s in sents:
        t = _approx_tokens(s)
        if buf_tokens + t > max_tokens and buf:
            _emit()
        buf.append(s)
        buf_tokens += t

    if buf:
        chunks.append(" ".join(buf).strip())

    return chunks