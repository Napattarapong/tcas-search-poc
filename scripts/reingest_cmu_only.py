"""Re-ingest ONLY the Chiang Mai PDFs (the ones the previous run didn't finish).

These are the largest sources (CMU R1=2694 chunks, R2=1852, R3=478). Their chunks
in the DB are still embedded from the old PUA-corrupted markdowns.

Run with `python -u` so we can see live progress.
"""
from __future__ import annotations
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))

from src.db import get_conn, init_db
from src.vector_search import BgeM3Embedder
from src.ingest import ingest_pdf_markdown_only

PDF_DIR = Path("data/raw/tcas_pdfs")
MD_DIR = Path("data/cache/markdown")
DB_PATH = "data/university.db"


def main() -> int:
    init_db(DB_PATH)
    embedder = BgeM3Embedder(cache_dir="data/models")

    # Only CMU (เชียงใหม่) PDFs - the ones the previous run didn't finish.
    pdfs = sorted(p for p in PDF_DIR.glob("*.pdf") if "Chiang_mai" in p.name)
    if not pdfs:
        print(f"No CMU PDFs in {PDF_DIR}")
        return 1

    print(f"Re-ingesting {len(pdfs)} CMU PDFs from cleaned markdowns\n", flush=True)

    total_chunks = 0
    t0 = time.time()
    for pdf in pdfs:
        sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
        md_path = MD_DIR / f"{sha}.md"
        if not md_path.exists():
            print(f"  SKIP  {pdf.name} (no markdown)", flush=True)
            continue

        with get_conn(DB_PATH, read_only=True) as conn:
            row = conn.execute(
                "SELECT id FROM source_documents WHERE sha256=?", (sha,)
            ).fetchone()
        if not row:
            print(f"  SKIP  {pdf.name} (no source_document row)", flush=True)
            continue
        sd_id = row[0]

        with get_conn(DB_PATH, read_only=False) as conn:
            n_deleted = conn.execute(
                "DELETE FROM chunks WHERE source_document_id=?", (sd_id,)
            ).rowcount
            conn.commit()

        print(
            f"  >>> {pdf.name}  src={sd_id}  deleted_old={n_deleted}  "
            f"md_size={md_path.stat().st_size:,}B",
            flush=True,
        )

        t1 = time.time()
        n = ingest_pdf_markdown_only(
            str(md_path), sha, DB_PATH, embedder,
            source_doc_id=sd_id, year=2569,
        )
        dt = time.time() - t1
        print(f"      done in {dt:.1f}s  -> {n} chunks", flush=True)
        total_chunks += n

    print(
        f"\nTotal new chunks: {total_chunks}  (total wall: {time.time()-t0:.1f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())