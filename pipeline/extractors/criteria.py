"""Criteria extractor — CMU (R1-R4) & Thammasat (R3) criteria booklets.

Each program is 1-2 table rows whose criteria cell embeds required subjects,
weights (ค่าน้ำหนักร้อยละ X) and minimum thresholds (Adj. T-SCORE >= Y).
Large docs -> batch N pages per call with a FLAT schema, then merge.
"""
import json
import sys
from typing import List, Optional

from pydantic import BaseModel, Field

from .. import llm
from ..convert import chunk_pages

PAGES_PER_CHUNK = 8

INSTR = (
    "Part of a Thai university TCAS admission criteria booklet. Each program is a header row "
    "(สาขาวิชา… / TCAS CODE: … / จำนวนรับตามแผน … คน) followed by project rows (รหัสโครงการ … | "
    "รอบ … | <seats> | <criteria text>). The criteria text contains คุณสมบัติผู้สมัคร, "
    "เกณฑ์ขั้นต่ำ (e.g. Adj. T-SCORE >= 45), and สัดส่วน/ค่าน้ำหนักร้อยละ X (weights).\n"
    "For EVERY program/project row extract: program name, TCAS code, faculty, seats, project "
    "code, round label, GPAX min, and up to 6 weighted subjects (code, Thai name, weight %, "
    "min threshold). Omit unused slots. Keep Thai as-is. Brief other criteria in other_conditions_th.\n\n"
)


class ProgramCriteria(BaseModel):
    program_name_th: str
    tcas_code: Optional[str] = None
    faculty_th: Optional[str] = None
    seats: Optional[int] = None
    project_code: Optional[str] = None
    round_label: Optional[str] = None
    gpax_min: Optional[str] = None
    subj1_code: Optional[str] = None
    subj1_name_th: Optional[str] = None
    subj1_weight: Optional[str] = None
    subj1_min: Optional[str] = None
    subj2_code: Optional[str] = None
    subj2_name_th: Optional[str] = None
    subj2_weight: Optional[str] = None
    subj2_min: Optional[str] = None
    subj3_code: Optional[str] = None
    subj3_name_th: Optional[str] = None
    subj3_weight: Optional[str] = None
    subj3_min: Optional[str] = None
    subj4_code: Optional[str] = None
    subj4_name_th: Optional[str] = None
    subj4_weight: Optional[str] = None
    subj4_min: Optional[str] = None
    subj5_code: Optional[str] = None
    subj5_name_th: Optional[str] = None
    subj5_weight: Optional[str] = None
    subj5_min: Optional[str] = None
    subj6_code: Optional[str] = None
    subj6_name_th: Optional[str] = None
    subj6_weight: Optional[str] = None
    subj6_min: Optional[str] = None
    other_conditions_th: Optional[str] = None


class ProgramList(BaseModel):
    items: List[ProgramCriteria] = Field(default_factory=list)


def normalize(p):
    subs = []
    for i in range(1, 7):
        code = p.get(f"subj{i}_code"); name = p.get(f"subj{i}_name_th")
        w = p.get(f"subj{i}_weight"); mn = p.get(f"subj{i}_min")
        if not (code or name or w):
            continue
        subs.append({"subject_code": code, "subject_name_th": name,
                     "weight_percent": w, "min_threshold": mn})
    return {
        "program_name_th": p["program_name_th"], "tcas_code": p.get("tcas_code"),
        "faculty_th": p.get("faculty_th"), "seats": p.get("seats"),
        "project_code": p.get("project_code"), "round_label": p.get("round_label"),
        "gpax_min": p.get("gpax_min"), "weighted_subjects": subs,
        "other_conditions_th": p.get("other_conditions_th"),
    }


def run(t):
    md = open(t["md_path"], encoding="utf-8").read()
    chunks = chunk_pages(md, PAGES_PER_CHUNK)
    instr_fn = lambda ch: INSTR + ch
    flat = llm.extract_chunks(ProgramList, instr_fn, chunks, label=f"{t['university_id']}{t['round']}")
    progs = [normalize(p.model_dump()) for p in flat]
    out = {
        "university": t["university_name"], "university_id": t["university_id"],
        "round": t["round"], "archetype": "criteria",
        "program_count": len(progs), "programs": progs,
    }
    json.dump(out, open(t["out_json"], "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[criteria] {t['university_id']} {t['round']}: {len(progs)} programs -> {t['out_json']}",
          file=sys.stderr, flush=True)
    return out
