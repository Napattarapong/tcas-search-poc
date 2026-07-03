#!/usr/bin/env python3
"""
Rule-based extractors for ALL archetypes (LLM-free) + archetype detector + runner.
  matrix    fixed-column table rows        (Chula R3)
  criteria  TCAS CODE headers + regex      (CMU/Thammasat-style booklets)
  index     code|name|seats table          (program index)
  list      numbered admission items        (Chula R1/R2)

Run across every converted markdown:
    python -m pipeline.rb_all            # writes data/extracted/<id>_<name>/*.json
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline.extract_rb import parse_matrix  # noqa: E402

THAI = "฀-๿"
TEXT = os.path.join(ROOT, "data", "text")
EXT = os.path.join(ROOT, "data", "extracted")
EC = os.path.join(ROOT, "data", "entrance_conditions.json")


def _sq(s):
    """Remove spurious spaces PyMuPDF inserts inside Thai words."""
    return re.sub(rf"(?<=[{THAI}])\s+(?=[{THAI}])", "", s)


# ---------- criteria ----------
_TCAS = re.compile(r"TCAS\s*CODE[:\s]*([0-9A-Za-z]+)")
_NAME = re.compile(r"(สาขาวิชา[^|\n]*?)\s*TCAS")
_PLAN = re.compile(r"จำนวนรับตามแผน[^\d]*(\d+)")
_VCODE = re.compile(r"รหัสวิชา[^\d]*(\d{2})")


def parse_criteria(md):
    lines = [_sq(l) for l in md.splitlines()]
    progs = []
    for i, line in enumerate(lines):
        m = _TCAS.search(line)
        if not m:
            continue
        code = m.group(1)
        nm = _NAME.search(line)
        seats = int(p.group(1)) if (p := _PLAN.search(line)) else None
        codes = set()
        for j in range(i, min(i + 40, len(lines))):
            if j > i and _TCAS.search(lines[j]):
                break
            for c in _VCODE.findall(lines[j]):
                if 61 <= int(c) <= 89:
                    codes.add(c)
        progs.append({"tcas_code": code, "program_name_th": (nm.group(1).strip(" |") if nm else ""),
                      "seats": seats, "subject_codes": sorted(codes)})
    return progs


# ---------- index (code|name|seats table) ----------
def parse_index(md):
    progs = []
    for row in md.splitlines():
        if not row.startswith("|") or "---" in row:
            continue
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        code = next((c for c in cells if re.fullmatch(r"[0-9A-Za-z]{8,}", c)), None)
        seats = next((int(c) for c in cells if c.isdigit() and 1 <= int(c) <= 5000), None)
        name = next((c for c in cells if re.search(THAI, c) and len(c) > 3), None)
        if code and (name or seats is not None):
            progs.append({"tcas_code": code, "program_name_th": name or "",
                          "seats": seats, "subject_codes": []})
    return progs


# ---------- list (numbered admission items) ----------
def parse_list(md):
    items = []
    for row in md.splitlines():
        m = re.search(r"(\d{1,3})\.\s*(การรับสมัคร|โครงการ|รับ)", _sq(row))
        if m:
            items.append({"item_no": m.group(1), "text": row.strip("| ").strip()[:300]})
    return items


# ---------- archetype detector ----------
def detect_archetype(md):
    if _TCAS.search(md):
        return "criteria"
    if "ค่าน้ำหนัก" in md or re.search(r"\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|", md):
        return "matrix"
    if re.search(r"\d{1,3}\.\s*(การรับสมัคร|โครงการ)", _sq(md)):
        return "list"
    if md.count("|") > 50:
        return "index"
    return "unknown"


def _uni_name_map():
    out = {}
    if os.path.exists(EC):
        for u in json.load(open(EC, encoding="utf-8")):
            out[u["university_id"]] = u.get("university_name_en", u["university_id"])
    return out


def best_parse(md):
    """Run every parser, keep the archetype with the most records (robust)."""
    res = {"criteria": parse_criteria(md), "matrix": parse_matrix(md),
           "index": parse_index(md), "list": parse_list(md)}
    arch = max(res, key=lambda k: len(res[k]))
    key = "items" if arch == "list" else "programs"
    return arch, res[arch], key


def run_all():
    names = _uni_name_map()
    totals = {}
    for md_path in sorted(glob.glob(os.path.join(TEXT, "*.md"))):
        base = os.path.basename(md_path)
        if base.startswith("_"):
            continue
        uid, rk = base[:3], base[4:6]
        md = open(md_path, encoding="utf-8").read()
        if len(md) < 800:                      # scanned / empty -> skip (needs OCR)
            print(f"[{uid} {rk}] skipped (scanned/empty)", flush=True)
            continue
        arch, recs, key = best_parse(md)
        totals[arch] = totals.get(arch, 0) + len(recs)
        folder = os.path.join(EXT, f"{uid}_{(names.get(uid,uid)).replace(' ','_')}")
        os.makedirs(folder, exist_ok=True)
        out = {"university_id": uid, "round": rk, "archetype": arch,
               "count": len(recs), key: recs}
        json.dump(out, open(os.path.join(folder, base.replace(".md", ".json")), "w"),
                  ensure_ascii=False, indent=2)
        print(f"[{uid} {rk}] {arch:9s} {len(recs):4d}", flush=True)
    print(f"\n[+] totals: {totals}", flush=True)


if __name__ == "__main__":
    run_all()
