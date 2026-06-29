"""Re-chunk and re-embed all PDFs from the cleaned/structured markdowns.

The original chunks in `chunks` table were embedded from corrupted
PUA-tainted markdowns. After running clean_markdowns.py and
structure_markdowns.py, the markdowns are readable Thai Unicode with
Markdown structure. This script deletes old chunks for each PDF source
and re-runs the chunk + embed pipeline against the cleaned .md files.

Idempotent: skips sources whose markdown no longer exists.
"""
from __future__ import annotations
import hashlib
import sys
from pathlib import Path

# Make src importable when run as a script
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
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {PDF_DIR}")
        return 1

    print(f"Re-ingesting {len(pdfs)} PDFs from cleaned markdowns\n")

    total_chunks = 0
    for pdf in pdfs:
        sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
        md_path = MD_DIR / f"{sha}.md"
        if not md_path.exists():
            print(f"  SKIP  {pdf.name} (no markdown)")
            continue

        # Find source_document_id
        with get_conn(DB_PATH, read_only=True) as conn:
            row = conn.execute(
                "SELECT id FROM source_documents WHERE sha256=?", (sha,)
            ).fetchone()
        if not row:
            print(f"  SKIP  {pdf.name} (no source_document row)")
            continue
        sd_id = row[0]

        # Delete old chunks for this source (if any)
        with get_conn(DB_PATH, read_only=False) as conn:
            n_deleted = conn.execute(
                "DELETE FROM chunks WHERE source_document_id=?", (sd_id,)
            ).rowcount
            conn.commit()
        print(f"  {pdf.name[:50]:<50}  src={sd_id}  deleted={n_deleted}", end="  ")

        # Re-chunk + re-embed (also rebuilds in-memory FAISS from full DB)
        n = ingest_pdf_markdown_only(
            str(md_path), sha, DB_PATH, embedder,
            source_doc_id=sd_id, year=2569,
        )
        print(f"new_chunks={n}")
        total_chunks += n

    print(f"\nTotal chunks now in DB: {total_chunks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
