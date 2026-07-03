#!/usr/bin/env python3
"""
Let the LLM group all distinct admission-score component labels (across the 3
universities) into a fixed canonical vocabulary — so synonyms like
GPAX / เกรดเฉลี่ย / GPA / ผลการเรียนเฉลี่ยสะสม collapse to one concept.

Reads every weighted_components/weighted_subjects label, dedupes them, asks
glm-5.1 to assign each a canonical concept from a fixed vocabulary, and writes
data/ml/component_map.json  {(category,code,name_th): canonical}.

Then score_topics.py uses this map instead of a hand-coded canon().
"""
import glob
import json
import os
import sys
from collections import Counter
from typing import List

from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(ROOT, "data", "extracted")
OUT = os.path.join(ROOT, "data", "ml", "component_map.json")

VOCAB = [
    "GPAX",
    "TGAT", "TGAT1", "TGAT2", "TGAT3",
    "TPAT2", "TPAT3", "TPAT4", "TPAT5",
    "61", "62", "63", "64", "65", "66", "70",
    "81", "82", "83", "84", "85", "86", "87", "88", "89",
    "AL1_aggregate", "AL2_aggregate", "bestof_language",
    "Interview", "MMI", "Portfolio", "ENG_test", "fine_arts", "law", "other",
]

PROMPT = (
    "You are normalizing Thai university admission-score component labels into a "
    "FIXED canonical vocabulary. Each input line is: <idx> | category | code | thai_name\n"
    "Rules:\n"
    "- GPAX / GPA / เกรดเฉลี่ย / ผลการเรียนเฉลี่ย / grade-average → 'GPAX'. "
    "Subject-specific GPAX (e.g. 'GPA วิชาเคมี', 'ภาษาอังกฤษ (GPAX ม.4-6)') → the matching subject code.\n"
    "- TGAT / ความถนัดทั่วไป / code 90 → 'TGAT'; if specifically TGAT1/91→'TGAT1', TGAT2/92→'TGAT2', TGAT3/93→'TGAT3'.\n"
    "- TPAT by number: code 21/TPAT1→'TPAT2' if ทัศนศิลป์, 20→'TPAT2', 30/TPAT3→'TPAT3', 40/TPAT4→'TPAT4', 50/TPAT5→'TPAT5'. "
    "If just 'TPAT' with no number, use 'TPAT3'.\n"
    "- A-Level subject by code: 61→'61', 62→'62', 63,64,65,66,70,81..89 likewise (the code IS the canonical).\n"
    "- 'A-Level ชุดที่ 1' (sci bundle 61,64,65,66,70,81,82) → 'AL1_aggregate'; 'ชุดที่ 2' (arts bundle) → 'AL2_aggregate'.\n"
    "- 'best of' language lists like 62/83/84/85/86/87/88/89 → 'bestof_language'.\n"
    "- สัมภาษณ์/Interview → 'Interview'; MMI/Multiple Mini → 'MMI'; แฟ้มสะสม/Portfolio → 'Portfolio'; "
    "IELTS/TOEFL/CU-TEP/TU-GET/ENG → 'ENG_test'; FA1/FA3/FA4/วาดเส้น/ทัศนศิลป์ → 'fine_arts'; "
    "นิติศาสตร์/Law → 'law'.\n"
    "- Anything else → 'other'.\n"
    "Return ONE canonical per idx, choosing ONLY from this vocabulary:\n"
    f"{', '.join(VOCAB)}\n\n"
)


class Assign(BaseModel):
    idx: int
    canonical: str = Field(description="one value from the vocabulary only")


class AssignList(BaseModel):
    items: List[Assign] = Field(default_factory=list)


def collect_labels():
    labels = Counter()
    for f in glob.glob(os.path.join(EXT, "**", "*.json"), recursive=True):
        d = json.load(open(f, encoding="utf-8"))
        for p in d.get("programs", []):
            for c in (p.get("weighted_components") or p.get("weighted_subjects") or []):
                cat = (c.get("category") or "").strip()
                code = (c.get("test_or_subject_code") or c.get("subject_code") or "").strip()
                name = (c.get("subject_name_th") or "").strip()
                labels[(cat, code, name)] += 1
    return labels


def main():
    labels = collect_labels()
    items = list(labels.items())  # ((cat,code,name), count)
    print(f"[*] {len(items)} distinct labels", file=sys.stderr)

    llm = ChatAnthropic(model="glm-5.1", temperature=0, max_tokens=8192, timeout=180)
    struct = llm.with_structured_output(AssignList)

    BATCH = 70
    idx_of = {}
    mapping = {}  # (cat,code,name) -> canonical
    for b in range(0, len(items), BATCH):
        batch = items[b:b + BATCH]
        lines = []
        for i, ((cat, code, name), _n) in enumerate(batch):
            idx = b + i
            idx_of[idx] = (cat, code, name)
            lines.append(f"{idx} | {cat} | {code} | {name}")
        res = struct.invoke(PROMPT + "\n".join(lines))
        for a in res.items:
            key = idx_of.get(a.idx)
            if key:
                mapping[key] = a.canonical
        print(f"  [batch {b//BATCH+1}] {len(res.items)} assigned", file=sys.stderr, flush=True)

    # report grouping quality
    by_canon = {}
    for key, canon in mapping.items():
        by_canon.setdefault(canon, []).append(key)
    print(f"\n[*] {len(mapping)} labels -> {len(by_canon)} canonical concepts", file=sys.stderr)
    for canon in sorted(by_canon, key=lambda c: -len(by_canon[c])):
        ex = ", ".join(f"{k[2] or k[1] or k[0]}" for k in by_canon[canon][:3])
        print(f"  {canon:16s} ({len(by_canon[canon]):3d}): {ex[:60]}", file=sys.stderr)

    json.dump({f"{k[0]}||{k[1]}||{k[2]}": v for k, v in mapping.items()},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[+] wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
