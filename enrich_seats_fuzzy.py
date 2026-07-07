#!/usr/bin/env python3
"""Fuzzy-fill missing seats, CONSTRAINED to the same university (avoids cross-uni
mismatches). Matches each no-seats program against its own university's catalog
entries (courses.json) at rapidfuzz WRatio >= 90."""
import glob
import gzip
import json
import os
import re
from collections import defaultdict

from rapidfuzz import fuzz, process

ROOT = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(ROOT, "data", "extracted")
THR = 90


def num(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def norm(s):
    return re.sub(r"\s+", "", (s or "")).lower()


def main():
    import urllib.request
    raw = urllib.request.urlopen(urllib.request.Request(
        "https://my-tcas.s3.ap-southeast-1.amazonaws.com/mytcas/courses.json",
        headers={"User-Agent": "Mozilla/5.0"})).read()
    try:
        cat = json.loads(raw.decode())
    except UnicodeDecodeError:
        cat = json.loads(gzip.decompress(raw).decode())

    # catalog grouped by university: uid -> list of (norm_name, seats)
    by_uni = defaultdict(list)
    for p in cat:
        s = num(p.get("number_acceptance_mko2"))
        nm = norm(p.get("program_name_th"))
        if nm and s is not None:
            by_uni[p.get("university_id")].append((nm, s))

    filled = total = 0
    for f in glob.glob(os.path.join(EXT, "**", "*.json"), recursive=True):
        uid = os.path.basename(f)[:3]
        unis = by_uni.get(uid, [])
        if not unis:
            continue
        names = [n for n, _ in unis]
        seats_of = {n: s for n, s in unis}
        d = json.load(open(f, encoding="utf-8"))
        changed = False
        for p in d.get("programs") or []:
            if p.get("seats") is not None:
                continue
            total += 1
            nm = norm(p.get("program_name_th"))
            if not nm:
                continue
            best = process.extractOne(nm, names, scorer=fuzz.WRatio)
            if best and best[1] >= THR:
                p["seats"] = seats_of[best[0]]
                filled += 1
                changed = True
        if changed:
            json.dump(d, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[*] fuzzy seats filled (same-uni, WRatio>={THR}): {filled}/{total}")


if __name__ == "__main__":
    main()
