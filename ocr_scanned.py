#!/usr/bin/env python3
"""OCR every converted markdown that came back near-empty (scanned image PDFs),
via OCR.space (language=tha). Overwrites the md with real text so rb_all can parse it.

Run:  OCRSPACE_API_KEY=... python ocr_scanned.py
"""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from pipeline.ocr import ocr_pdf  # noqa: E402

ROUND_N = {"R1": "1", "R2": "2", "R3": "3", "R4": "4", "handicap": "handicap"}


def main():
    targets = []
    for md in sorted(glob.glob("data/text/[0-9]*.md")):
        if os.path.getsize(md) >= 800:
            continue  # already has text
        base = os.path.basename(md)[:-3]          # <id>_<round>
        uid, rk = base[:3], base[4:6]
        n = ROUND_N.get(rk)
        if not n:
            continue
        pdfs = glob.glob(f"data/pdfs/{uid}_*/file_path_{n}__*.pdf") \
            + glob.glob(f"data/pdfs/{uid}_*/file_path_{rk}__*.pdf")
        if not pdfs:
            print(f"[skip] {base}: PDF not found", flush=True)
            continue
        targets.append((base, pdfs[0], md))
    print(f"[*] {len(targets)} scanned docs to OCR", flush=True)
    for base, pdf, md in targets:
        print(f"[ocr] {base}", flush=True)
        try:
            ocr_pdf(pdf, md)
        except Exception as e:
            print(f"[ERR] {base}: {e}", flush=True)
    print("[+] done", flush=True)


if __name__ == "__main__":
    main()
