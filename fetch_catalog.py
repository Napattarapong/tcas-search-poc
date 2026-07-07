#!/usr/bin/env python3
"""Recover universities whose PDFs are announcements/templates/links (164, 023, 911)
by pulling their program catalog from the mytcas S3 courses.json."""
import gzip
import json
import os
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(ROOT, "data", "extracted")
TARGETS = ["164", "023", "911"]


def num(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def main():
    raw = urllib.request.urlopen(urllib.request.Request(
        "https://my-tcas.s3.ap-southeast-1.amazonaws.com/mytcas/courses.json",
        headers={"User-Agent": "Mozilla/5.0"})).read()
    try:
        d = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        d = json.loads(gzip.decompress(raw).decode("utf-8"))

    for uid in TARGETS:
        ps = [p for p in d if p.get("university_id") == uid]
        if not ps:
            print(f"[!] {uid}: none in catalog")
            continue
        name = ps[0].get("university_name_en") or uid
        programs = [{
            "tcas_code": p.get("program_id"),
            "program_name_th": p.get("program_name_th"),
            "faculty_th": p.get("faculty_name_th"),
            "seats": num(p.get("number_acceptance_mko2")) or num(p.get("major_acceptance_number")),
            "subject_codes": [],
        } for p in ps]
        folder = os.path.join(EXT, f"{uid}_{name.replace(' ', '_')}")
        os.makedirs(folder, exist_ok=True)
        out = {"university_id": uid, "university": name, "round": "catalog",
               "archetype": "catalog", "count": len(programs), "programs": programs}
        json.dump(out, open(os.path.join(folder, f"{uid}_catalog.json"), "w"),
                  ensure_ascii=False, indent=2)
        print(f"[+] {uid} {name}: {len(programs)} programs (catalog)")


if __name__ == "__main__":
    main()
