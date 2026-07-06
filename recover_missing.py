#!/usr/bin/env python3
"""Recover the last few missing universities.
1. Force render+OCR the garbled/scanned ones (016 text layer is scrambled; 023/146/221 empty).
2. LLM-extract (criteria) all missing docs that now have usable text.
"""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from pipeline.ocr import ocr_pdf  # noqa: E402
from pipeline.extractors import criteria  # noqa: E402
from pipeline.rb_all import _uni_name_map  # noqa: E402

FORCE_OCR = ["016", "023", "146", "221"]        # garbled / scanned
MISSING = ["016", "023", "146", "221", "103", "144", "911"]
ROUND_N = {"R1": "1", "R2": "2", "R3": "3", "R4": "4", "handicap": "handicap"}
names = _uni_name_map()


def safe(s):
    return (s or "").replace(" ", "_")


def main():
    # 1. force-OCR garbled/scanned
    for uid in FORCE_OCR:
        for md in sorted(glob.glob(f"data/text/{uid}_*.md")):
            rk = os.path.basename(md)[4:6]
            pdfs = (glob.glob(f"data/pdfs/{uid}_*/file_path_{ROUND_N.get(rk)}__*.pdf")
                    + glob.glob(f"data/pdfs/{uid}_*/file_path_{rk}__*.pdf"))
            if not pdfs:
                continue
            try:
                ocr_pdf(pdfs[0], md, dpi=250)
                print(f"[ocr] {uid} {rk} -> {os.path.getsize(md)} chars", flush=True)
            except Exception as e:
                print(f"[ocr ERR] {uid} {rk}: {e}", flush=True)
    # 2. LLM extract anything missing with usable text
    for uid in MISSING:
        for md in sorted(glob.glob(f"data/text/{uid}_*.md")):
            if os.path.getsize(md) < 800:
                print(f"[skip] {md} still empty", flush=True)
                continue
            base = os.path.basename(md)[:-3]
            rk = base[4:6]
            folder = os.path.join(ROOT, "data/extracted", f"{uid}_{safe(names.get(uid,uid))}")
            os.makedirs(folder, exist_ok=True)
            t = {"university_id": uid, "university_name": names.get(uid, uid), "round": rk,
                 "round_label": rk, "md_path": md,
                 "out_json": os.path.join(folder, base + ".json")}
            try:
                criteria.run(t)
            except Exception as e:
                print(f"[ERR] {base}: {e}", flush=True)
    print("[+] recovery done", flush=True)


if __name__ == "__main__":
    main()
