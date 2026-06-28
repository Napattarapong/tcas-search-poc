"""Keyword-based router picks structured vs free-text path."""
from __future__ import annotations
import pytest
from src.router import route, FREE_TEXT_KEYWORDS

@pytest.mark.parametrize("q", [
    "ทุนเรียนดีของมหิดลมีอะไรบ้าง",
    "คุณสมบัติผู้สมัครวิศวะจุฬารอบ 1",
    "เอกสารที่ต้องใช้สมัครคืออะไร",
    "เงื่อนไขการรับทุน",
    "ข้อกำหนดของคณะ",
])
def test_free_text_keywords_route_to_free(q):
    assert route(q) == "free"

@pytest.mark.parametrize("q", [
    "วิศวะจุฬารอบ 1 ปี 2569 รับกี่คน",
    "ค่าเทอมคณะวิทยาศาสตร์ มหิดลเท่าไหร่",
    "GPA ขั้นต่ำเท่าไหร่",
    "คะแนน TGAT",
])
def test_structured_questions_route_to_structured(q):
    assert route(q) == "structured"

def test_keyword_list_not_empty():
    assert len(FREE_TEXT_KEYWORDS) >= 5