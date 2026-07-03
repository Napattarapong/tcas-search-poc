"""List extractor — Chula Round 1 (Portfolio) & Round 2 (Quota).

These docs are a numbered list of admission projects/quotas, grouped by faculty.
Each item reads like:
  "N. การรับสมัครคัดเลือกนักเรียน<who> เข้าศึกษาใน<program/faculty> • สอบถาม...โทร.<phone> อีเมล:<email>"
No seats / weights / subjects — it's qualitative. We split the run-on text into
flat fields. Per-page extraction (small docs, keeps faculty context).
"""
import json
import sys
from typing import List, Optional

from pydantic import BaseModel, Field

from .. import llm
from ..convert import chunk_pages

INSTR = (
    "Below is a page from a Thai university TCAS Round-1 (Portfolio) / Round-2 (Quota) "
    "admission list. It is a NUMBERED list of admission projects/quotas, grouped under "
    "faculty headers (คณะ…). Each item starts with a number and a sentence shaped like "
    "'การรับสมัครคัดเลือกนักเรียน<WHO> เข้าศึกษาใน<PROGRAM/FACULTY> • สอบถาม…โทร.<phone> อีเมล:<email>'.\n"
    "For EVERY numbered item, extract: item_no, the faculty it belongs to, target_group_th "
    "(who is eligible / the project description), target_program_th (the program/faculty entered), "
    "and contact_th (phone + email). Keep Thai text as-is. Skip non-item rows (headers/footers).\n\n"
)


class AdmissionItem(BaseModel):
    item_no: str = Field(description="the item number, e.g. '1'")
    faculty_th: Optional[str] = Field(default=None, description="Thai faculty grouping (คณะ…)")
    target_group_th: Optional[str] = Field(default=None, description="who is eligible / project description")
    target_program_th: Optional[str] = Field(default=None, description="program / faculty entered")
    contact_th: Optional[str] = Field(default=None, description="phone and email if present")


class ItemList(BaseModel):
    items: List[AdmissionItem] = Field(default_factory=list)


def run(t):
    md = open(t["md_path"], encoding="utf-8").read()
    chunks = chunk_pages(md, 1)  # per page
    instr_fn = lambda ch: INSTR + ch
    items = llm.extract_chunks(ItemList, instr_fn, chunks, label=f"{t['university_id']}{t['round']}")
    items = [i.model_dump() for i in items]  # pydantic -> dict for JSON
    out = {
        "university": t["university_name"], "university_id": t["university_id"],
        "round": t["round"], "round_label": t["round_label"],
        "archetype": "list", "item_count": len(items), "items": items,
    }
    json.dump(out, open(t["out_json"], "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[list] {t['university_id']} {t['round']}: {len(items)} items -> {t['out_json']}",
          file=sys.stderr, flush=True)
    return out
