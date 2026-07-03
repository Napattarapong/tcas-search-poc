"""Thai free-form input pipeline: normalize -> word-segment (LLM-free, local).

Stage 1 of the input pipeline. Uses PyThaiNLP 'newmm' (dictionary maximal
matching) with a TCAS domain dictionary so multi-word terms like ค่าน้ำหนัก /
รหัสวิชา / คณิตศาสตร์ประยุกต์ stay whole. The domain dictionary is seeded from
the component labels already extracted (data/ml/component_map*.json) plus a core
hand list, so it grows with the corpus.

Usage:
    from pipeline.thai_tokenize import tokenize, normalize
    tokens = tokenize("ค่าน้ำหนักร้อยละ 30 รหัสวิชา 61")
"""
import json
import os
import re
import unicodedata

from pythainlp.tokenize import Trie, word_tokenize
from pythainlp.corpus import thai_words

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# core TCAS/admission terms that must segment as single tokens
CORE_DOMAIN = [
    "ค่าน้ำหนัก", "ร้อยละ", "รหัสวิชา", "จำนวนรับ", "จำนวนรับตามแผน", "รอบที่",
    "รับตรง", "รับตรงร่วมกัน", "แฟ้มสะสมผลงาน", "ความถนัด", "ความถนัดทั่วไป",
    "ไม่น้อยกว่า", "สาขาวิชา", "คุณสมบัติผู้สมัคร", "เกณฑ์ขั้นต่ำ", "สัดส่วน",
    "Adj. T-SCORE", "คะแนนรวม", "นักเรียน", "คัดเลือก", "ร่วมกัน", "เทียบวุฒิ",
    "คณิตศาสตร์ประยุกต์", "วิทยาศาสตร์ประยุกต์", "สังคมศึกษา", "ภาษาอังกฤษ",
    "ภาษาไทย", "วิศวกรรมศาสตร์", "คณะวิศวกรรมศาสตร์", "หลักสูตร",
]


def _seed_from_maps():
    """Add surface names from the extracted component maps to the dictionary."""
    words = set()
    for name in ("component_map.json", "component_map_llm.json"):
        p = os.path.join(ROOT, "data", "ml", name)
        if os.path.exists(p):
            for key in json.load(open(p, encoding="utf-8")):
                # key = "category||code||name"
                parts = key.split("||")
                if len(parts) == 3 and parts[2]:
                    words.add(parts[2].strip())
    return words


_TRIE = None


def _trie():
    global _TRIE
    if _TRIE is None:
        words = set(thai_words()) | set(CORE_DOMAIN) | _seed_from_maps()
        _TRIE = Trie(list(words))
    return _TRIE


def normalize(text):
    """Unicode-normalize and collapse whitespace/PyMuPDF spacing artifacts."""
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text, engine="newmm"):
    """Normalize + word-segment Thai free-form text -> list of tokens."""
    return word_tokenize(normalize(text), engine=engine, custom_dict=_trie())


if __name__ == "__main__":
    import sys
    t = " ".join(sys.argv[1:]) or "ค่าน้ำหนักร้อยละ 30 ของวิชาคณิตศาสตร์ประยุกต์ 1 รหัสวิชา 61"
    print(" | ".join(tokenize(t)))
