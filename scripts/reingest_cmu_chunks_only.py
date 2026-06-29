"""Re-ingest CMU PDFs to chunks ONLY — no FAISS rebuild per call.

The full reingest_clean_markdowns.py is slow because ingest_pdf_markdown_only
rebuilds the in-memory FAISS index (embedding ALL chunks) after every PDF.
This script just inserts chunks to SQLite; the FAISS index is rebuilt lazily
on first search.
"""
from __future__ import annotations
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))

from src.db import get_conn, init_db
from src.chunking import chunk_markdown

PDF_DIR = Path("data/raw/tcas_pdfs")
MD_DIR = Path("data/cache/markdown")
DB_PATH = "data/university.db"


def main() -> int:
    init_db(DB_PATH)

    pdfs = sorted(p for p in PDF_DIR.glob("*.pdf") if "Chiang_mai" in p.name)
    if not pdfs:
        print(f"No CMU PDFs in {PDF_DIR}", flush=True)
        return 1

    print(f"Re-chunking {len(pdfs)} CMU PDFs (DB only, no embedding)\n", flush=True)

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

        t1 = time.time()
        text = md_path.read_text(encoding="utf-8")
        chunks_text = chunk_markdown(text, max_tokens=300, overlap=50)

        with get_conn(DB_PATH, read_only=False) as conn:
            conn.execute("DELETE FROM chunks WHERE source_document_id=?", (sd_id,))
            for i, ctext in enumerate(chunks_text):
                conn.execute(
                    "INSERT INTO chunks(source_document_id, chunk_index, text, faiss_index_offset) "
                    "VALUES(?,?,?,?)",
                    (sd_id, i, ctext, i),
                )
            conn.commit()

        dt = time.time() - t1
        print(
            f"  {pdf.name}  src={sd_id}  md={len(text):,}B  "
            f"-> {len(chunks_text)} chunks  ({dt:.1f}s)",
            flush=True,
        )

    print(f"\nDone in {time.time()-t0:.1f}s. FAISS will rebuild on next search.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())