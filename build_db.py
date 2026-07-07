#!/usr/bin/env python3
"""Build the relational SQLite DB (data/search/tcas.db) from the extracted data.

Tables: university, subject, program, program_subject (weights/required junction).
Run:  python build_db.py"""
import glob
import gzip
import json
import os
import sqlite3
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(ROOT, "data", "search", "tcas.db")
EXT = os.path.join(ROOT, "data", "extracted")
EC = os.path.join(ROOT, "data", "entrance_conditions.json")

EN = {"61": "Math1", "62": "Math2", "63": "Stat", "64": "Physics", "65": "Chem",
      "66": "Bio", "67": "Sci", "70": "Social", "71": "Geo", "72": "History",
      "73": "Religion", "74": "Citizen", "75": "Econ", "80": "Design&Tech",
      "81": "Thai", "82": "English", "83": "French", "84": "German", "85": "Japanese",
      "86": "Korean", "87": "Chinese", "88": "Pali", "89": "Spanish",
      "GPAX": "GPAX", "TGAT": "TGAT", "TGAT1": "TGAT1", "TGAT2": "TGAT2", "TGAT3": "TGAT3",
      "TPAT2": "TPAT2", "TPAT3": "TPAT3", "TPAT4": "TPAT4", "TPAT5": "TPAT5"}

SCHEMA = """
DROP TABLE IF EXISTS program_subject;
DROP TABLE IF EXISTS program;
DROP TABLE IF EXISTS subject;
DROP TABLE IF EXISTS university;
CREATE TABLE university(id TEXT PRIMARY KEY, name_en TEXT, name_th TEXT);
CREATE TABLE subject(code TEXT PRIMARY KEY, name_th TEXT, name_en TEXT, category TEXT);
CREATE TABLE program(
  id INTEGER PRIMARY KEY,
  university_id TEXT REFERENCES university(id),
  round TEXT, tcas_code TEXT, program_name_th TEXT, program_name_en TEXT,
  faculty_th TEXT, seats INT, min_gpax REAL, min_total REAL, topic TEXT, source TEXT);
CREATE TABLE program_subject(
  program_id INT REFERENCES program(id), subject_code TEXT REFERENCES subject(code),
  weight_percent REAL, min_threshold REAL,
  PRIMARY KEY(program_id, subject_code));
CREATE INDEX idx_program_uni ON program(university_id);
CREATE INDEX idx_program_round ON program(round);
CREATE INDEX idx_ps_subject ON program_subject(subject_code);
CREATE INDEX idx_ps_program ON program_subject(program_id);
"""


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def category(c):
    c = (c or "").strip()
    if c == "GPAX":
        return "GPAX"
    if c.startswith("TGAT") or c in ("90", "91", "92", "93"):
        return "TGAT"
    if c.startswith("TPAT") or c in ("21", "30", "40", "50"):
        return "TPAT"
    if c.isdigit() and 61 <= int(c) <= 89:
        return "A-Level"
    return "other"


def fetch_courses():
    try:
        raw = urllib.request.urlopen(urllib.request.Request(
            "https://my-tcas.s3.ap-southeast-1.amazonaws.com/mytcas/courses.json",
            headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read()
        try:
            return json.loads(raw.decode())
        except UnicodeDecodeError:
            return json.loads(gzip.decompress(raw).decode())
    except Exception:
        return []


def main():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.executescript(SCHEMA)

    cat = fetch_courses()
    uni_th = {}
    for p in cat:
        uid = p.get("university_id")
        if uid and p.get("university_name_th"):
            uni_th.setdefault(uid, p["university_name_th"])

    # university
    ec = json.load(open(EC, encoding="utf-8"))
    uid_name = {u["university_id"]: u.get("university_name_en", u["university_id"]) for u in ec}
    for uid, name in uid_name.items():
        cur.execute("INSERT INTO university VALUES(?,?,?)", (uid, name, uni_th.get(uid)))
    cat_keys = set(EN)
    subj_th = {}

    # program + program_subject
    seen = set()
    for f in sorted(glob.glob(os.path.join(EXT, "**", "*.json"), recursive=True)):
        base = os.path.basename(f)
        if base.endswith("_admission.json"):
            continue
        d = json.load(open(f, encoding="utf-8"))
        uid = base[:3]
        rnd = d.get("round") or base[4:6]
        source = d.get("archetype", "?")
        for p in d.get("programs") or []:
            name_th = p.get("program_name_th") or p.get("faculty_major_th") or ""
            tcas = p.get("tcas_code") or p.get("program_code")
            key = (uid, rnd, tcas or name_th)
            if key in seen:
                continue
            seen.add(key)
            cur.execute(
                "INSERT INTO program(university_id,round,tcas_code,program_name_th,"
                "program_name_en,faculty_th,seats,min_gpax,min_total,topic,source) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (uid, rnd, tcas, name_th, p.get("program_name_en") or "",
                 p.get("faculty_th") or p.get("faculty_major_th") or name_th,
                 p.get("seats"), num(p.get("min_gpax")) if p.get("min_gpax") is not None else num(p.get("gpax_min")),
                 num(p.get("min_total_score")), p.get("topic"), source))
            pid = cur.lastrowid
            comps = p.get("weighted_components") or p.get("weighted_subjects") or []
            for c in comps:
                code = (c.get("test_or_subject_code") or c.get("subject_code") or "").strip()
                if not code:
                    continue
                subj_th.setdefault(code, (c.get("subject_name_th") or "").strip() or None)
                cat_keys.add(code)
                cur.execute("INSERT OR IGNORE INTO program_subject VALUES(?,?,?,?)",
                            (pid, code, num(c.get("weight_percent")), num(c.get("min_threshold"))))
            for code in (p.get("subject_codes") or []):
                cat_keys.add(code)
                cur.execute("INSERT OR IGNORE INTO program_subject VALUES(?,?,?,?)", (pid, code, None, None))

    # subject catalog
    for c in sorted(cat_keys, key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x)):
        cur.execute("INSERT OR IGNORE INTO subject VALUES(?,?,?,?)",
                    (c, subj_th.get(c), EN.get(c), category(c)))

    con.commit()
    for t in ("university", "subject", "program", "program_subject"):
        n = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:16s} {n}")
    print(f"[+] DB -> {DB}")


if __name__ == "__main__":
    main()
