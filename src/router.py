"""Keyword-based router: free-text vs structured path."""
from __future__ import annotations

FREE_TEXT_KEYWORDS = [
    "ทุน",
    "คุณสมบัติ",
    "เงื่อนไข",
    "ข้อกำหนด",
    "เอกสาร",
    "สมัครยังไง",
    "เตรียมตัว",
    "เหมาะกับ",
    "แนะนำ",
    "ขอบเขต",
    "ลักษณะ",
]


def route(question: str) -> str:
    """Return 'free' if the question looks like free-text (qualitative),
    else 'structured' (filters, counts, scores, dates)."""
    if any(kw in question for kw in FREE_TEXT_KEYWORDS):
        return "free"
    return "structured"