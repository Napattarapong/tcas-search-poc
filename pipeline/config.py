"""Config for the flexible TCAS extraction pipeline.

Derives the list of (university, round, archetype, paths) targets from
data/entrance_conditions.json (written by scrape.py) + an explicit
(university_id, round) -> archetype map. Adding a new round/university is
just a row in ARCHETYPE_MAP (and a new extractor only if it's a new archetype).
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
EC_PATH = os.path.join(DATA, "entrance_conditions.json")
TEXT_DIR = os.path.join(DATA, "text")
EXTRACTED_DIR = os.path.join(DATA, "extracted")

# file_path_* key -> short round label
ROUND_KEY = {
    "file_path_1": "R1", "file_path_2": "R2", "file_path_3": "R3",
    "file_path_4": "R4", "file_path_handicap": "handicap",
}

# (university_id, round_label) -> archetype:  list | matrix | criteria | narrative
ARCHETYPE_MAP = {
    ("001", "R1"): "list",       # Chula Portfolio  -> project list
    ("001", "R2"): "list",       # Chula Quota      -> quota list
    ("001", "R3"): "matrix",     # Chula Admission  -> weight matrix
    ("004", "R1"): "criteria",   # Chiang Mai       -> criteria booklet
    ("004", "R2"): "criteria",
    ("004", "R3"): "criteria",
    ("004", "R4"): "criteria",
    ("004", "handicap"): "narrative",
    ("005", "R1"): "criteria",   # Thammasat (verify on first run)
    ("005", "R3"): "criteria",
    ("005", "handicap"): "narrative",
}

GLM_MODEL = "glm-5.1"


def targets(only=None):
    """Yield target dicts for every downloaded PDF whose archetype is known.

    only: optional set of (university_id, round) tuples to restrict to.
    Skips 'narrative' (handicap) unless explicitly requested later.
    """
    ec = json.load(open(EC_PATH, encoding="utf-8"))
    for u in ec:
        uid = u["university_id"]
        for p in u["entrance_condition_pdfs"]:
            if p.get("status") != "ok":
                continue
            rk = ROUND_KEY.get(p["round_key"], p["round_key"])
            arch = ARCHETYPE_MAP.get((uid, rk))
            if arch is None:
                continue
            if only and (uid, rk) not in only:
                continue
            yield {
                "university_id": uid,
                "university_name": u.get("university_name_en", uid),
                "round": rk,
                "round_label": p.get("round_label", rk),
                "archetype": arch,
                "pdf_path": p["saved_to"],
                "md_path": os.path.join(TEXT_DIR, f"{uid}_{rk}.md"),
                "out_folder": os.path.join(EXTRACTED_DIR, f"{uid}_{u.get('university_name_en', uid).replace(' ', '_')}"),
                "out_json": os.path.join(EXTRACTED_DIR, f"{uid}_{u.get('university_name_en', uid).replace(' ', '_')}",
                                         f"{uid}_{rk}.json"),
            }
