"""Matrix extractor — Chula Round 3 (Admission): a compact weight matrix.

Each program is one table row with fixed columns (GPAX/TGAT/TPAT/A-Level weights,
min thresholds, seats). Flat schema (nested lists return empty under
with_structured_output), extracted per page (each page carries its own header),
then normalized into weighted_components.
"""
import json
import sys
from typing import List, Optional

from pydantic import BaseModel, Field

from .. import llm
from ..convert import chunk_pages

COLUMN_HINT = (
    "Column order of each data row: [blank] | program_code | faculty/major(Thai) | "
    "min_gpax | gpax_weight | tgat_weight | tgat_min | tgat1_weight | tgat1_min | "
    "tgat2_weight | tgat2_min | tpat_subject | tpat_weight | _ | _ | tpat_min_per_subject | "
    "alevel1(code/name) | alevel1_weight | alevel2 | alevel2_weight | alevel3 | alevel3_weight | "
    "alevel4 | alevel4_weight | alevel_min_per_subject | min_total_score | seats. "
    "A '-' or empty cell = null. Extract EVERY program row. Only set a weight/code when real."
)


class ProgramCriteria(BaseModel):
    program_code: str = Field(description="3-digit program code")
    faculty_major_th: str = Field(description="Thai faculty and major/branch")
    min_gpax: Optional[float] = Field(default=None)
    gpax_weight: Optional[float] = Field(default=None)
    tgat_weight: Optional[float] = Field(default=None)
    tgat1_weight: Optional[float] = Field(default=None)
    tgat2_weight: Optional[float] = Field(default=None)
    tpat_subject: Optional[str] = Field(default=None)
    tpat_weight: Optional[float] = Field(default=None)
    tpat_min_per_subject: Optional[float] = Field(default=None)
    alevel1_code: Optional[str] = Field(default=None)
    alevel1_name_th: Optional[str] = Field(default=None)
    alevel1_weight: Optional[float] = Field(default=None)
    alevel2_code: Optional[str] = Field(default=None)
    alevel2_name_th: Optional[str] = Field(default=None)
    alevel2_weight: Optional[float] = Field(default=None)
    alevel3_code: Optional[str] = Field(default=None)
    alevel3_name_th: Optional[str] = Field(default=None)
    alevel3_weight: Optional[float] = Field(default=None)
    alevel4_code: Optional[str] = Field(default=None)
    alevel4_name_th: Optional[str] = Field(default=None)
    alevel4_weight: Optional[float] = Field(default=None)
    alevel_min_per_subject: Optional[float] = Field(default=None)
    min_total_score: Optional[float] = Field(default=None)
    seats: Optional[int] = Field(default=None)


class ProgramList(BaseModel):
    items: List[ProgramCriteria] = Field(default_factory=list)


def normalize(p):
    comps = []
    for cat, code, name, w in [
        ("GPAX", None, None, p.get("gpax_weight")),
        ("TGAT", None, None, p.get("tgat_weight")),
        ("TGAT1", None, None, p.get("tgat1_weight")),
        ("TGAT2", None, None, p.get("tgat2_weight")),
        ("TPAT", p.get("tpat_subject"), None, p.get("tpat_weight")),
        ("A-Level", p.get("alevel1_code"), p.get("alevel1_name_th"), p.get("alevel1_weight")),
        ("A-Level", p.get("alevel2_code"), p.get("alevel2_name_th"), p.get("alevel2_weight")),
        ("A-Level", p.get("alevel3_code"), p.get("alevel3_name_th"), p.get("alevel3_weight")),
        ("A-Level", p.get("alevel4_code"), p.get("alevel4_name_th"), p.get("alevel4_weight")),
    ]:
        if w is None and not code and not name:
            continue
        comps.append({
            "category": cat, "test_or_subject_code": code, "subject_name_th": name,
            "weight_percent": w,
            "min_threshold_percent": p.get("tpat_min_per_subject") if cat == "TPAT"
            else (p.get("alevel_min_per_subject") if cat == "A-Level" else None),
        })
    return {
        "program_code": p["program_code"], "faculty_major_th": p["faculty_major_th"],
        "min_gpax": p.get("min_gpax"), "weighted_components": comps,
        "min_total_score": p.get("min_total_score"), "seats": p.get("seats"),
    }


def run(t):
    md = open(t["md_path"], encoding="utf-8").read()
    chunks = chunk_pages(md, 1)
    instr_fn = lambda ch: ("Thai university TCAS Round-3 admission criteria TABLE (markdown). "
                           + COLUMN_HINT + "\n\n" + ch)
    flat = llm.extract_chunks(ProgramList, instr_fn, chunks, label=f"{t['university_id']}{t['round']}")
    progs = [normalize(p.model_dump()) for p in flat]
    out = {
        "university": t["university_name"], "university_id": t["university_id"],
        "round": t["round"], "archetype": "matrix",
        "program_count": len(progs), "programs": progs,
    }
    json.dump(out, open(t["out_json"], "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[matrix] {t['university_id']} {t['round']}: {len(progs)} programs -> {t['out_json']}",
          file=sys.stderr, flush=True)
    return out
