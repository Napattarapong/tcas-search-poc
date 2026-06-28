"""Citation validator: every surviving sentence ends with [src=table#id] or [src=chunk#id]."""
from __future__ import annotations
import pytest
from src.validator import validate_prose

def test_keeps_sentence_with_row_citation():
    prose = "วิศวะจุฬารับ 60 คน [src=admission_rounds#42]"
    assert validate_prose(prose).strip() == prose

def test_keeps_sentence_with_chunk_citation():
    prose = "ทุนนี้มีเงื่อนไขคือต้องมี GPA ไม่ต่ำกว่า 3.5 [src=chunk#17]"
    assert validate_prose(prose).strip() == prose

def test_drops_sentence_without_citation():
    prose = "ประโยคนี้ไม่มีการอ้างอิง [src=admission_rounds#1]\nประโยคนี้ไม่มี citation"
    out = validate_prose(prose)
    assert "[src=admission_rounds#1]" in out
    assert "ประโยคนี้ไม่มี citation" not in out

def test_field_qualifier_accepted():
    prose = "วันปิดรับสมัครคือ 1 เม.ย. [src=admission_rounds#42,field=apply_close]"
    assert validate_prose(prose).strip() == prose
