#!/usr/bin/env python3
"""
Materialize the combined search table (docs/SYSTEM_DESIGN.md schema) from the
per-university extracted JSONs. One row per program, written as JSONL to
data/search/programs.jsonl. Components are normalized through the unsupervised
component_map so synonyms collapse to canonical subject codes.

Run:  python build_search_table.py
"""
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(ROOT, "data", "extracted")
OUT_DIR = os.path.join(ROOT, "data", "search")
OUT = os.path.join(OUT_DIR, "programs.jsonl")
MAP_PATH = os.path.join(ROOT, "data", "ml", "component_map.json")

UID_NAME = {"001": "Chulalongkorn", "004": "Chiang Mai", "005": "Thammasat"}
# also load all university id->name from the scrape manifest (52 unis)
_EC = os.path.join(ROOT, "data", "entrance_conditions.json")
if os.path.exists(_EC):
    try:
        for _u in json.load(open(_EC, encoding="utf-8")):
            UID_NAME.setdefault(_u["university_id"], _u.get("university_name_en", _u["university_id"]))
    except Exception:
        pass


def canon(cat, code, name):
    return MAP.get(f"{cat}||{code}||{name}") or (code if (code or "").isdigit() else None)


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def normalize_program(p, uni, uid, rnd, source="?"):
    name_th = p.get("program_name_th") or p.get("faculty_major_th") or ""
    if "subject_codes" in p:                       # RB criteria/index output
        subject_codes = sorted({c for c in (p["subject_codes"] or []) if c.isdigit()})
        weights = {}
    else:                                          # matrix / LLM components
        comps = p.get("weighted_components") or p.get("weighted_subjects") or []
        subject_codes, weights = set(), {}
        for c in comps:
            raw = (c.get("test_or_subject_code") or c.get("subject_code") or "").strip()
            w = num(c.get("weight_percent"))
            for code in {x for x in re.split(r"[,/ +]+", raw) if x.isdigit()}:
                if 61 <= int(code) <= 89:
                    subject_codes.add(code)
                if w is not None:
                    weights[code] = weights.get(code, 0) + w
        subject_codes = sorted(subject_codes)
    return {
        "university": uni,
        "university_id": uid,
        "round": rnd,
        "tcas_code": p.get("tcas_code") or p.get("program_code"),
        "faculty_th": p.get("faculty_major_th") or p.get("faculty_th") or name_th,
        "program_name_th": name_th,
        "program_name_en": p.get("major_name_en") or "",
        "seats": p.get("seats"),
        "min_gpax": num(p.get("min_gpax")) if p.get("min_gpax") is not None else num(p.get("gpax_min")),
        "subject_codes": subject_codes,
        "topic": None,
        "source": source,
        "weights": weights,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, seen = [], set()
    for f in sorted(glob.glob(os.path.join(EXT, "**", "*.json"), recursive=True)):
        base = os.path.basename(f)
        if base.endswith("_admission.json"):   # legacy duplicate of 001_R3.json
            continue
        d = json.load(open(f, encoding="utf-8"))
        uid = base[:3]
        uni = UID_NAME.get(uid, "?")
        rnd = d.get("round") or base[4:6]
        for p in d.get("programs", []):
            row = normalize_program(p, uni, uid, rnd, d.get("archetype", "?"))
            key = (uid, rnd, row["tcas_code"] or row["program_name_th"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    with open(OUT, "w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    with_subj = sum(1 for r in rows if r["subject_codes"])
    print(f"[+] {len(rows)} programs -> {OUT}")
    print(f"    {with_subj} with subject_codes; "
          f"{sum(1 for r in rows if r['program_name_en'])} with English name")


if __name__ == "__main__":
    main()
