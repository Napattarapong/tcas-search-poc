"""Download TCAS admission PDFs for our 3 universities (จุฬาฯ, เชียงใหม่, มหิดล)."""
import gzip
import json
import subprocess
import sys
import urllib.parse
from pathlib import Path

UNIVERSITIES_URL = "https://my-tcas.s3.ap-southeast-1.amazonaws.com/mytcas/universities.json"
TARGET_IDS = {"001", "004", "006"}
OUT_DIR = Path("data/raw/tcas_pdfs")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Fetch + decompress universities.json
    print("Downloading universities.json ...")
    raw = subprocess.check_output(
        ["curl", "-sL", "--max-time", "30", UNIVERSITIES_URL], text=False
    )
    data = json.loads(gzip.decompress(raw))
    print(f"  {len(data)} universities")

    # 2) Filter to our 3
    targets = [u for u in data if u.get("university_id") in TARGET_IDS]
    print(f"  targets: {[(u['university_id'], u.get('university_name_en')) for u in targets]}")

    # 3) Download each round PDF (use Python urllib with percent-encoded URL —
    #    some filenames contain Thai spaces and other chars that curl mishandles.)
    import urllib.request
    downloaded = 0
    for u in targets:
        uid = u["university_id"]
        name = (u.get("university_name_en") or uid).replace(" ", "_")
        for round_no in (1, 2, 3, 4):
            raw_url = u.get(f"file_path_{round_no}")
            if not raw_url:
                continue
            # Split into base + filename so we can percent-encode only the filename
            parts = urllib.parse.urlsplit(raw_url)
            encoded_path = urllib.parse.quote(parts.path, safe="/")
            url = urllib.parse.urlunsplit((parts.scheme, parts.netloc, encoded_path,
                                           parts.query, parts.fragment))
            out = OUT_DIR / f"{uid}_{name}_R{round_no}.pdf"
            if out.exists() and out.stat().st_size > 1000:
                print(f"  {out.name}: already present ({out.stat().st_size} bytes)")
                continue
            print(f"  Downloading {out.name}")
            print(f"    {url}")
            try:
                with urllib.request.urlopen(url, timeout=120) as resp:
                    body = resp.read()
                if len(body) < 1000:
                    print(f"    FAILED (size={len(body)})")
                    continue
                out.write_bytes(body)
                downloaded += 1
                print(f"    OK ({len(body)} bytes)")
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}")
                if out.exists():
                    out.unlink()

    print(f"\nDownloaded {downloaded} new PDFs into {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
