"""OCR scanned PDFs via OCR.space (Thai), rendering pages to images first.

For PDFs with no text layer (e.g. CMU R4), render each page to a JPEG and send
to https://api.ocr.space/parse/image (language=tha, engine 2). Writes markdown
to the target's md_path, then the normal criteria extractor can run.

Usage:
    OCRSPACE_API_KEY=... python -m pipeline.ocr 004:R4
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

import fitz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline import config as C  # noqa: E402

API = "https://api.ocr.space/parse/image"
KEY = os.environ.get("OCRSPACE_API_KEY", "")  # set OCRSPACE_API_KEY in your env


def _post(img_bytes, lang="tha", engine="2", is_table="true"):
    b = "----ocr" + uuid.uuid4().hex

    def field(n, v):
        return f"--{b}\r\nContent-Disposition: form-data; name=\"{n}\"\r\n\r\n{v}\r\n".encode()

    body = (field("apikey", KEY) + field("language", lang) + field("OCREngine", engine)
            + field("isTable", is_table))
    body += (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"p.jpg\"\r\n"
             "Content-Type: image/jpeg\r\n\r\n").encode()
    body += img_bytes + b"\r\n" + f"--{b}--\r\n".encode()
    req = urllib.request.Request(API, data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def ocr_image(img_bytes, lang="tha", retries=3):
    for attempt in range(retries):
        try:
            d = _post(img_bytes, lang=lang)
            if d.get("IsErroredOnProcessing"):
                # rate limit / transient -> backoff and retry
                time.sleep(2 * (attempt + 1))
                continue
            return d.get("ParsedResults", [{}])[0].get("ParsedText", "")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            time.sleep(2 * (attempt + 1))
    return ""


def ocr_pdf(pdf_path, md_path, dpi=200, lang="tha"):
    doc = fitz.open(pdf_path)
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    pages = []
    for i, page in enumerate(doc, 1):
        img = page.get_pixmap(dpi=dpi).tobytes("jpg")
        txt = ocr_image(img, lang=lang).strip()
        pages.append(f"<!-- page {i} -->\n{txt}")
        print(f"  [ocr {i}/{doc.page_count}] {len(txt)} chars", file=sys.stderr, flush=True)
        time.sleep(0.3)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(pages))
    print(f"[ocr] wrote {md_path} ({sum(len(p) for p in pages)} chars)", file=sys.stderr, flush=True)


def main():
    only = None
    if len(sys.argv) > 1:
        only = set()
        for part in sys.argv[1].split(","):
            uid, rk = part.split(":")
            only.add((uid.strip(), rk.strip()))
    for t in C.targets(only=only):
        if not t["pdf_path"]:
            continue
        print(f"=== OCR {t['university_id']} {t['round']} ===", file=sys.stderr, flush=True)
        ocr_pdf(t["pdf_path"], t["md_path"])


if __name__ == "__main__":
    main()
