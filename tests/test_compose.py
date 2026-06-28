"""Composer must produce prose that cites ONLY ids present in the provided sources."""
from __future__ import annotations
from unittest.mock import patch
from src.compose import compose_prose

def _fake_chat(messages, temperature=0.0, response_format=None):
    # Echo a known-good answer citing the chunk we provided.
    return "นี่คือคำตอบทดสอบ [src=chunk#1]"


def test_compose_with_chunks_only():
    chunks = [{"id": 1, "source_document_id": 1, "text": "ทุนเรียนดี มีเงื่อนไข GPA 3.5"}]
    with patch("src.compose.chat", side_effect=_fake_chat):
        prose, sources = compose_prose("ทุนอะไรบ้าง", rows=None, chunks=chunks)
    assert "[src=chunk#1]" in prose
    assert any(s["type"] == "chunk" and s["id"] == 1 for s in sources)


def test_compose_with_rows_and_chunks():
    rows = [{"id": 42, "table": "admission_rounds", "seats": 60}]
    chunks = [{"id": 1, "source_document_id": 1, "text": "x"}]

    def _chat(messages, temperature=0.0, response_format=None):
        return "รับ 60 คน [src=admission_rounds#42,field=seats]\nเงื่อนไข [src=chunk#1]"

    with patch("src.compose.chat", side_effect=_chat):
        prose, sources = compose_prose("วิศวะจุฬารับกี่คน", rows=rows, chunks=chunks)
    assert "[src=admission_rounds#42,field=seats]" in prose
    assert "[src=chunk#1]" in prose
    kinds = {s["type"] for s in sources}
    assert kinds == {"row", "chunk"}


def test_compose_with_no_sources_returns_empty():
    prose, sources = compose_prose("ไม่มีข้อมูล", rows=None, chunks=None)
    assert prose == ""
    assert sources == []
