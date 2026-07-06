#!/usr/bin/env python3
"""OCR 016/023/146 (OCR.space couldn't read them) with Tesseract + Thai,
rendering pages at 300 DPI. Overwrites data/text/<id>_<round>.md."""
import glob
import os
import re
import subprocess
import tempfile

import fitz

UNIS = ["016", "023", "146"]
ROUND = {"file_path_1": "R1", "file_path_2": "R2", "file_path_3": "R3",
         "file_path_4": "R4", "file_path_handicap": "handicap"}


def round_of(fname):
    m = re.match(r"(file_path_\w+?)__", fname)
    return ROUND.get(m.group(1)) if m else None


def main():
    for uid in UNIS:
        for pdf in sorted(glob.glob(f"data/pdfs/{uid}_*/file_path_*__*.pdf")):
            rk = round_of(os.path.basename(pdf))
            if not rk:
                continue
            md = f"data/text/{uid}_{rk}.md"
            doc = fitz.open(pdf)
            pages = []
            for i, page in enumerate(doc, 1):
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                    img = tf.name
                page.get_pixmap(dpi=300).save(img)
                r = subprocess.run(["tesseract", img, "stdout", "-l", "tha"],
                                   capture_output=True, text=True, timeout=120)
                os.remove(img)
                pages.append(f"<!-- page {i} -->\n" + r.stdout.strip())
                print(f"  [{uid} {rk} p{i}/{doc.page_count}] {len(r.stdout)} chars", flush=True)
            doc.close()
            open(md, "w", encoding="utf-8").write("\n\n".join(pages))
            print(f"[tess] {uid} {rk} -> {os.path.getsize(md)} chars", flush=True)
    print("[+] tesseract done", flush=True)


if __name__ == "__main__":
    main()
