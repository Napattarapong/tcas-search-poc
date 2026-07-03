#!/usr/bin/env python3
"""
Download entrance-condition PDFs for ALL universities from the mytcas S3 bucket
(https://my-tcas.s3.ap-southeast-1.amazonaws.com/mytcas/universities.json).

Recreates the download step of the pipeline. Idempotent: skips files already on disk.

Usage:
    python scrape_all.py            # all universities that publish PDFs
    python scrape_all.py 001 004    # specific university_ids
"""
import gzip
import json
import os
import sys
import urllib.request
from urllib.parse import quote, urlsplit

API = "https://my-tcas.s3.ap-southeast-1.amazonaws.com/mytcas"
UNIS_URL = f"{API}/universities.json"
ROOT = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(ROOT, "data", "pdfs")
EC_PATH = os.path.join(ROOT, "data", "entrance_conditions.json")
UA = {"User-Agent": "Mozilla/5.0"}
ROUND_KEYS = ["file_path_1", "file_path_2", "file_path_3", "file_path_4", "file_path_handicap"]


def fetch(url):
    req = urllib.request.Request(encode_url(url), headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def encode_url(url):
    p = urlsplit(url)
    return p._replace(path=quote(p.path, safe="/")).geturl()


def safe_name(s):
    return (s.replace("/", "-").replace("\\", "-").replace('"', "")
            .replace(" ", "_").strip() or "x")


def load_unis():
    raw = fetch(UNIS_URL)
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(gzip.decompress(raw).decode("utf-8"))


def main(targets):
    os.makedirs(PDF_DIR, exist_ok=True)
    unis = load_unis()
    if not targets:
        targets = [u["university_id"] for u in unis
                   if any(u.get(k) for k in ROUND_KEYS)]
    print(f"[*] {len(targets)} universities to download", flush=True)
    results, n_files, n_skip, n_fail = [], 0, 0, 0
    for uid in targets:
        u = next((x for x in unis if x["university_id"] == uid), None)
        if not u:
            print(f"[!] {uid} not found", flush=True)
            continue
        uni_dir = os.path.join(PDF_DIR, f"{uid}_{safe_name(u.get('university_name_en', uid))}")
        os.makedirs(uni_dir, exist_ok=True)
        pdfs = []
        for key in ROUND_KEYS:
            url = u.get(key)
            if not url:
                continue
            base = os.path.basename(url.split("?")[0]) or f"{key}.pdf"
            fname = f"{key}__{safe_name(base)}"
            if not fname.lower().endswith(".pdf"):
                fname += ".pdf"
            dest = os.path.join(uni_dir, fname)
            if os.path.exists(dest):
                n_skip += 1
                status, size = "skip", os.path.getsize(dest)
            else:
                try:
                    data = fetch(url)
                    with open(dest, "wb") as f:
                        f.write(data)
                    size, status, n_files = len(data), "ok", n_files + 1
                except Exception as e:
                    size, status, n_fail = 0, f"fail:{type(e).__name__}", n_fail + 1
            pdfs.append({"round_key": key, "url": url, "path": os.path.relpath(dest, ROOT),
                         "bytes": size, "status": status})
        results.append({"university_id": uid,
                        "university_name_en": u.get("university_name_en"),
                        "pdfs": pdfs})
        got = sum(1 for p in pdfs if p["status"] in ("ok", "skip"))
        print(f"[{uid}] {u.get('university_name_en','')[:30]:30s} {got} PDFs", flush=True)
    json.dump(results, open(EC_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[+] downloaded {n_files}, skipped {n_skip} (cached), failed {n_fail}", flush=True)
    print(f"[+] manifest -> {EC_PATH}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else [])
