"""Convert every PDF in data/raw/tcas_pdfs/ to markdown via markitdown.

Idempotent: skips PDFs whose markdown already exists with non-trivial size.
Caches by sha256 so renames don't re-convert.
"""
from __future__ import annotations
import hashlib
import subprocess
import sys
from pathlib import Path

PDF_DIR = Path("data/raw/tcas_pdfs")
MD_DIR = Path("data/cache/markdown")
MIN_OK_BYTES = 100  # markdowns smaller than this are treated as failed


def main() -> int:
    MD_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {PDF_DIR}")
        return 1

    converted = skipped = failed = 0
    for pdf in pdfs:
        sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
        out = MD_DIR / f"{sha}.md"
        if out.exists() and out.stat().st_size >= MIN_OK_BYTES:
            print(f"  [skip]  {pdf.name} → {out.name} ({out.stat().st_size:,} bytes)")
            skipped += 1
            continue
        if out.exists():
            out.unlink()  # remove zero/empty file
        print(f"  [run ]  {pdf.name} ({pdf.stat().st_size:,} B) → {sha[:12]}.md ...", end=" ", flush=True)
        try:
            subprocess.run(
                [sys.executable, "-m", "markitdown", str(pdf), "-o", str(out)],
                check=True, timeout=300,
            )
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            failed += 1
            continue
        except subprocess.CalledProcessError as e:
            print(f"FAILED (exit {e.returncode})")
            if out.exists():
                out.unlink()
            failed += 1
            continue
        size = out.stat().st_size if out.exists() else 0
        if size < MIN_OK_BYTES:
            print(f"EMPTY ({size} B)")
            failed += 1
        else:
            print(f"OK ({size:,} B)")
            converted += 1

    print(f"\nconverted={converted} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
