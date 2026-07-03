#!/usr/bin/env python3
"""
Rule-based (LLM-FREE) extraction for the 'matrix' archetype (Chula Round 3).

Parses the markdown table directly: each program is one row; columns are mapped
by fixed position (the same layout the LLM was told). No LLM, no API cost.

Usage:
    python extract_rb.py matrix <md_path>            # prints JSON to stdout
    python extract_rb.py validate <md_path> <llm_json>  # compare to LLM output
"""
import json
import re
import sys


def _num(s):
    s = (s or "").strip()
    if not s or s == "-":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _code_name(cell):
    """'61 / คณิตศาสตร์ประยุกต์ 1' -> ('61', 'คณิตศาสตร์ประยุกต์ 1')"""
    cell = (cell or "").strip()
    if not cell or cell == "-":
        return None, None
    if "/" in cell:
        code, name = cell.split("/", 1)
        return code.strip(), name.strip()
    return None, cell


# fixed column indices for the Chula R3 matrix (27 cols, 0-26)
COL = {
    "code": 1, "name": 2, "min_gpax": 3, "gpax_w": 4,
    "tgat_w": 5, "tgat_min": 6, "tgat1_w": 7, "tgat1_min": 8,
    "tgat2_w": 9, "tgat2_min": 10, "tpat_subj": 11, "tpat_w": 12,
    "tpat_min": 15,
    "alevel1": 16, "alevel1_w": 17, "alevel2": 18, "alevel2_w": 19,
    "alevel3": 20, "alevel3_w": 21, "alevel4": 22, "alevel4_w": 23,
    "alevel_min": 24, "min_total": 25, "seats": 26,
}


def parse_matrix(md):
    progs = []
    for row in md.splitlines():
        if not row.startswith("|") or "---" in row:
            continue
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) < 26:
            continue
        code = cells[COL["code"]]
        if not (code.isdigit() and len(code) == 3):
            continue  # not a program data row
        comps = []
        if _num(cells[COL["gpax_w"]]) is not None:
            comps.append({"category": "GPAX", "weight_percent": _num(cells[COL["gpax_w"]])})
        for cat, wi in [("TGAT", "tgat_w"), ("TGAT1", "tgat1_w"), ("TGAT2", "tgat2_w")]:
            w = _num(cells[COL[wi]])
            if w is not None:
                comps.append({"category": cat, "weight_percent": w})
        tps = cells[COL["tpat_subj"]]
        tpw = _num(cells[COL["tpat_w"]])
        if tps and tps != "-" or tpw is not None:
            comps.append({"category": "TPAT", "test_or_subject_code": tps if tps != "-" else None,
                          "weight_percent": tpw})
        for ki in ("alevel1", "alevel2", "alevel3", "alevel4"):
            c, n = _code_name(cells[COL[ki]])
            w = _num(cells[COL[ki + "_w"]])
            if c or n or w is not None:
                comps.append({"category": "A-Level", "test_or_subject_code": c,
                              "subject_name_th": n, "weight_percent": w})
        progs.append({
            "program_code": code,
            "faculty_major_th": cells[COL["name"]],
            "min_gpax": _num(cells[COL["min_gpax"]]),
            "weighted_components": comps,
            "min_total_score": _num(cells[COL["min_total"]]),
            "seats": int(cells[COL["seats"]]) if cells[COL["seats"]].isdigit() else None,
        })
    return progs


def validate(md_path, llm_json):
    md = open(md_path, encoding="utf-8").read()
    rb = {p["program_code"]: p for p in parse_matrix(md)}
    llm = {p["program_code"]: p for p in json.load(open(llm_json))["programs"]}
    print(f"rule-based programs: {len(rb)} | LLM programs: {len(llm)}")
    print(f"codes only in RB : {sorted(set(rb) - set(llm))[:10]}")
    print(f"codes only in LLM: {sorted(set(llm) - set(rb))[:10]}")
    # compare seats + seats agreement
    agree_seats = agree_code = 0
    common = set(rb) & set(llm)
    for c in common:
        if (rb[c].get("seats") or 0) == (llm[c].get("seats") or 0):
            agree_seats += 1
        rb_codes = {x.get("test_or_subject_code") for x in rb[c]["weighted_components"] if x.get("test_or_subject_code")}
        llm_codes = {x.get("test_or_subject_code") for x in llm[c]["weighted_components"] if x.get("test_or_subject_code") or x.get("category") == "TPAT"}
        # normalize llm alevel codes too
        llm_all = {x.get("test_or_subject_code") for x in llm[c]["weighted_components"]} | {
            x.get("subject_code") for x in llm[c]["weighted_components"]}
        if rb_codes.issubset(llm_all) or not rb_codes:
            agree_code += 1
    print(f"common codes: {len(common)} | seats agree: {agree_seat_check(agree_seats, common)}")
    print(f"sample RB row 001: {json.dumps(rb.get('001'), ensure_ascii=False)[:200]}")


def agree_seat_check(a, c):
    return f"{a}/{len(c)} ({100*a/len(c):.0f}%)" if c else "n/a"


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "matrix":
        print(json.dumps(parse_matrix(open(sys.argv[2], encoding="utf-8").read()), ensure_ascii=False, indent=2))
    elif mode == "validate":
        validate(sys.argv[2], sys.argv[3])
