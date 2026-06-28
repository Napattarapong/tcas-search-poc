"""Thai-aware paragraph chunking with token budget + overlap."""
from __future__ import annotations
from src.chunking import chunk_markdown

def test_short_text_yields_single_chunk():
    text = "นี่คือข้อความสั้น ๆ ที่ไม่ควรถูกแบ่ง"
    chunks = chunk_markdown(text, max_tokens=300, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_long_text_yields_multiple_chunks_with_overlap():
    # Build a long Thai paragraph
    sentences = [f"ประโยคที่ {i} ของข้อความทดสอบระบบแบ่งส่วน" for i in range(50)]
    text = " ".join(sentences)
    chunks = chunk_markdown(text, max_tokens=60, overlap=10)
    assert len(chunks) >= 2
    # Each chunk must be non-empty
    assert all(c.strip() for c in chunks)
    # Adjacent chunks must share some overlap text
    assert any(
        set(a.split()).intersection(b.split())
        for a, b in zip(chunks, chunks[1:])
    )

def test_empty_text_returns_empty_list():
    assert chunk_markdown("", max_tokens=300, overlap=50) == []
    assert chunk_markdown("   \n\n  ", max_tokens=300, overlap=50) == []