#!/usr/bin/env python3
"""Convert every downloaded PDF to markdown (PyMuPDF tables). Idempotent.
Writes data/text/<id>_<round>.md and prints an inventory (incl. scanned/empty)."""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.convert import pdf_to_markdown  # noqa: E402

ROUND = {"file_path_1": "R1", "file_path_2": "R2", "file_path_3": "R3",
         "file_path_4": "R4", "file_path_handicap": "handicap"}


def round_of(fname):
    m = re.match(r"(file_path_\w+?)__", fname)
    return ROUND.get(m.group(1)) if m else None


def main():
    os.makedirs("data/text", exist_ok=True)
    pdfs = glob.glob("data/pdfs/*/*.pdf")
    print(f"[*] {len(pdfs)} PDFs to convert", flush=True)
    done = skip = fail = scanned = 0
    inv = []
    for pdf in sorted(pdfs):
        uid = os.path.basename(os.path.dirname(pdf))[:3]
        rk = round_of(os.path.basename(pdf))
        if not rk:
            continue
        out = f"data/text/{uid}_{rk}.md"
        if os.path.exists(out):
            skip += 1
            continue
        try:
            md = pdf_to_markdown(pdf)
            open(out, "w", encoding="utf-8").write(md)
            done += 1
            is_scan = len(md) < 800       # near-empty => likely scanned images
            if is_scan:
                scanned += 1
            inv.append((uid, rk, len(md), is_scan, os.path.basename(pdf)[:40]))
            print(f"[{uid} {rk:8s}] {len(md):7d} chars{'  (SCANNED?)' if is_scan else ''}", flush=True)
        except Exception as e:
            fail += 1
            print(f"[ERR] {pdf}: {e}", flush=True)
    print(f"\n[+] converted {done}, cached {skip}, failed {fail}, suspected-scanned {scanned}", flush=True)
    # write inventory csv
    with open("data/text/_inventory.csv", "w", encoding="utf-8") as f:
        f.write("uni,round,chars,scanned,file\n")
        for uid, rk, n, sc, fn in sorted(inv):
            f.write(f"{uid},{rk},{n},{int(sc)},{fn}\n")
    print("[+] inventory -> data/text/_inventory.csv", flush=True)


if __name__ == "__main__":
    main()
