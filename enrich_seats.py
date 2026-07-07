#!/usr/bin/env python3
"""Enrich missing seats in extracted programs from the mytcas catalog (courses.json),
delete junk/legacy files, then we rebuild the table.

Match order: tcas_code (exact) -> normalized program_name_th (exact)."""
import glob
import gzip
import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(ROOT, "data", "extracted")
JUNK = {"009_handicap", "024_R2", "199_handicap", "001_R3_admission"}  # files to delete (no .json)


def num(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def norm(s):
    return re.sub(r"\s+", "", (s or "")).lower()


def main():
    raw = urllib_request()
    try:
        d = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        d = json.loads(gzip.decompress(raw).decode("utf-8"))

    by_code, by_name = {}, {}
    for p in d:
        s = num(p.get("number_acceptance_mko2"))
        if p.get("program_id"):
            by_code.setdefault(p["program_id"], s)
        nm = norm(p.get("program_name_th"))
        if nm:
            by_name.setdefault(nm, s)

    filled = total_missing = 0
    for f in glob.glob(os.path.join(EXT, "**", "*.json"), recursive=True):
        d2 = json.load(open(f, encoding="utf-8"))
        progs = d2.get("programs") or []
        changed = False
        for p in progs:
            if p.get("seats") is None:
                total_missing += 1
                s = by_code.get(p.get("tcas_code")) or by_name.get(norm(p.get("program_name_th")))
                if s:
                    p["seats"] = s
                    filled += 1
                    changed = True
        if changed:
            json.dump(d2, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[*] seats filled: {filled}/{total_missing} missing")

    # delete junk / legacy dup
    removed = 0
    for f in glob.glob(os.path.join(EXT, "**", "*.json"), recursive=True):
        if os.path.basename(f)[:-5] in JUNK:
            os.remove(f)
            removed += 1
    for d2 in glob.glob(os.path.join(EXT, "*/")):
        if not os.listdir(d2):
            os.rmdir(d2)
    print(f"[*] junk/legacy files deleted: {removed}")


def urllib_request():
    import urllib.request
    return urllib.request.urlopen(urllib.request.Request(
        "https://my-tcas.s3.ap-southeast-1.amazonaws.com/mytcas/courses.json",
        headers={"User-Agent": "Mozilla/5.0"})).read()


if __name__ == "__main__":
    main()
