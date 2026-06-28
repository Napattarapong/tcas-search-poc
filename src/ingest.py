"""Ingestion pipelines. Use as `python -m src.ingest tcas|pdf`."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from src.db import init_db, get_conn


def _record_source_document(
    conn: sqlite3.Connection,
    file_path: str,
    raw_bytes: bytes,
    source_kind: str,
    university_id: int | None,
    year: int | None,
) -> int:
    """Insert source_documents row if sha256 not seen. Return its id."""
    sha = hashlib.sha256(raw_bytes).hexdigest()
    row = conn.execute(
        "SELECT id FROM source_documents WHERE sha256=?", (sha,)
    ).fetchone()
    if row:
        return row[0]
    cur = conn.execute(
        "INSERT INTO source_documents (file_path, sha256, source_kind, university_id, year) "
        "VALUES (?, ?, ?, ?, ?)",
        (file_path, sha, source_kind, university_id, year),
    )
    return cur.lastrowid


def ingest_tcas_from_path(
    json_path: Path | None,
    gz_path: Path | None,
    db_path: str,
    allowed_tcas_ids: set[str],
) -> dict:
    """Load TCAS rows for the allowed university IDs into the DB. Returns a summary dict."""
    if json_path is None and gz_path is None:
        raise ValueError("Provide either json_path or gz_path")
    if gz_path is not None:
        raw = gzip.decompress(gz_path.read_bytes())
        file_label = str(gz_path)
    else:
        raw = json_path.read_bytes()
        file_label = str(json_path)

    rows = json.loads(raw)
    init_db(db_path)
    with get_conn(db_path, read_only=False) as conn:
        sd_id = _record_source_document(
            conn, file_label, raw, "tcas_json", None, None,
        )
        n_uni = n_prog = 0
        for r in rows:
            if r.get("university_id") not in allowed_tcas_ids:
                continue
            existing = conn.execute(
                "SELECT id FROM universities WHERE tcas_id=?", (r["university_id"],)
            ).fetchone()
            if existing:
                uni_id = existing[0]
            else:
                cur = conn.execute(
                    "INSERT INTO universities (tcas_id, name_th, name_en, "
                    "university_type_th, source_document_id) VALUES (?,?,?,?,?)",
                    (
                        r["university_id"], r["university_name_th"],
                        r.get("university_name_en"), r.get("university_type_name_th"),
                        sd_id,
                    ),
                )
                uni_id = cur.lastrowid
                n_uni += 1

            cur = conn.execute(
                "INSERT OR IGNORE INTO programs ("
                "university_id, tcas_program_id, program_name_th, program_name_en,"
                "faculty_name_th, faculty_name_en, field_name_th, field_name_en,"
                "program_type_th, degree, cost, number_acceptance_mko2,"
                "major_acceptance_number, graduate_rate, employment_rate,"
                "median_salary, source_document_id"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    uni_id, r["program_id"], r["program_name_th"], r.get("program_name_en"),
                    r.get("faculty_name_th"), r.get("faculty_name_en"),
                    r.get("field_name_th"), r.get("field_name_en"),
                    r.get("program_type_name_th"),
                    "bachelor",
                    r.get("cost"), r.get("number_acceptance_mko2"),
                    r.get("major_acceptance_number"),
                    r.get("graduate_rate"), r.get("employment_rate"),
                    r.get("median_salary"), sd_id,
                ),
            )
            if cur.rowcount == 1:
                n_prog += 1
        conn.commit()
    return {"universities": n_uni, "programs": n_prog, "source_document_id": sd_id}


# --------------------------------------------------------------------------
# PDF ingest (markitdown + LLM extract)
# --------------------------------------------------------------------------

def _run_markitdown(pdf_path: Path, out_md_path: Path) -> None:
    """Run the installed markitdown CLI to convert a PDF to markdown."""
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "markitdown", str(pdf_path), "-o", str(out_md_path)],
        check=True,
    )


def _lookup_program_id(conn, tcas_program_id: str) -> int | None:
    row = conn.execute(
        "SELECT id FROM programs WHERE tcas_program_id=?", (tcas_program_id,)
    ).fetchone()
    return row[0] if row else None


def _persist_structured(db_path: str, structured: dict, source_document_id: int) -> dict:
    """Write structured extract (rounds/cutoffs/requirements) into the DB.

    Accepts the new shape from _extract_structured:
        {"rounds": [...], "cutoffs": [...], "requirements": [...]}
    Returns counts {"rounds": int, "cutoffs": int, "requirements": int}.
    """
    if not isinstance(structured, dict):
        return {"rounds": 0, "cutoffs": 0, "requirements": 0}
    n_round = n_cut = n_req = 0
    with get_conn(db_path, read_only=False) as conn:
        # Idempotency: if this source already wrote rounds, skip.
        existing_rounds = conn.execute(
            "SELECT COUNT(*) FROM admission_rounds WHERE source_document_id=?",
            (source_document_id,),
        ).fetchone()[0]
        if existing_rounds:
            return {"rounds": 0, "cutoffs": 0, "requirements": 0}

        for r in structured.get("rounds", []):
            pid = _lookup_program_id(conn, r.get("program_tcas_id", ""))
            if pid is None:
                continue
            cur = conn.execute(
                "INSERT INTO admission_rounds (program_id, year, round_no,"
                "apply_open, apply_close, exam_date, interview_date, seats,"
                "source_document_id) VALUES (?,?,?,?,?,?,?,?,?)",
                (pid, r.get("year"), r.get("round_no"),
                 r.get("apply_open"), r.get("apply_close"),
                 r.get("exam_date"), r.get("interview_date"),
                 r.get("seats"), source_document_id),
            )
            round_id = cur.lastrowid
            n_round += 1
            for c in structured.get("cutoffs", []):
                if (c.get("program_tcas_id") == r.get("program_tcas_id")
                        and c.get("year") == r.get("year")
                        and c.get("round_no") == r.get("round_no")):
                    conn.execute(
                        "INSERT INTO cutoff_scores (round_id, score_type,"
                        "min_score, max_score, gpa_min, source_document_id) "
                        "VALUES (?,?,?,?,?,?)",
                        (round_id, c.get("score_type"),
                         c.get("min_score"), c.get("max_score"),
                         c.get("gpa_min"), source_document_id),
                    )
                    n_cut += 1
            for q in structured.get("requirements", []):
                if (q.get("program_tcas_id") == r.get("program_tcas_id")
                        and q.get("year") == r.get("year")
                        and q.get("round_no") == r.get("round_no")):
                    conn.execute(
                        "INSERT INTO requirements (round_id, kind, text,"
                        "source_document_id) VALUES (?,?,?,?)",
                        (round_id, q.get("kind"), q.get("text"),
                         source_document_id),
                    )
                    n_req += 1
        conn.commit()
    return {"rounds": n_round, "cutoffs": n_cut, "requirements": n_req}


# ---- CLI ----

def cli_tcas(args) -> None:
    db_path = args.db_path
    gz_path = Path(args.gz_path) if args.gz_path else None
    json_path = Path(args.json_path) if args.json_path else None
    summary = ingest_tcas_from_path(
        json_path=json_path,
        gz_path=gz_path,
        db_path=db_path,
        allowed_tcas_ids=set(args.allowed.split(",")),
    )
    print(json.dumps(summary, ensure_ascii=False))


def cli_pdf(args) -> None:
    from src.vector_search import BgeM3Embedder
    embedder = BgeM3Embedder(cache_dir="data/models")
    summary = ingest_pdf(
        pdf_path=args.pdf,
        db_path=args.db_path,
        embedder=embedder,
        year=args.year,
    )
    if summary.get("structured", {}).get("_error"):
        print(json.dumps(summary, ensure_ascii=False), file=sys.stderr)
        sys.exit(2)
    print(json.dumps(summary, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="src.ingest")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tcas = sub.add_parser("tcas")
    p_tcas.add_argument("--db-path", default="data/university.db")
    p_tcas.add_argument("--json-path", default=None)
    p_tcas.add_argument("--gz-path", default=None)
    p_tcas.add_argument("--allowed", default="001,004,006")
    p_tcas.set_defaults(func=cli_tcas)

    p_pdf = sub.add_parser("pdf")
    p_pdf.add_argument("pdf")
    p_pdf.add_argument("--db-path", default="data/university.db")
    p_pdf.add_argument("--year", type=int, required=True)
    p_pdf.set_defaults(func=cli_pdf)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


# --------------------------------------------------------------------------
# Chunk + embed path (vector search ingest)
# --------------------------------------------------------------------------

def _extract_structured(markdown_text: str) -> dict:
    """Stub: in production, this calls the LLM to extract rounds/cutoffs/requirements.
    Tests can monkeypatch this to return {}."""
    # Import lazily to avoid circular imports
    from src.llm import chat
    prompt = (
        "Extract admission_rounds, cutoff_scores, and requirements from this markdown. "
        "Return JSON: {\"rounds\": [...], \"cutoffs\": [...], \"requirements\": [...]}.\n\n"
        f"{markdown_text}"
    )
    import json
    import re
    raw = chat(messages=[{"role": "user", "content": prompt}], temperature=0.0,
               response_format={"type": "json_object"}, max_tokens=4096)
    # Some models (e.g. Typhoon) wrap JSON in markdown fences even with
    # response_format=json_object. Strip them before parsing.
    stripped = raw.strip()
    fence_match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", stripped, re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1).strip()
    return json.loads(stripped)


def ingest_pdf_markdown_only(
    md_path: str, sha256: str, db_path: str, embedder, source_doc_id: int | None = None, year: int | None = None
) -> int:
    """Insert source_document row, chunk the markdown, embed, and insert chunks.

    Returns the number of chunks inserted. Idempotent: if a source_document with
    the given sha256 already has chunks, skip re-chunking and return the existing
    count. Does NOT run structured extraction (caller is expected to do that
    separately, possibly via the LLM).
    """
    from src.db import init_db, get_conn
    from src.chunking import chunk_markdown
    from src.vector_search import build_index

    init_db(db_path)

    with get_conn(db_path) as conn:
        if source_doc_id is None:
            existing = conn.execute(
                "SELECT id FROM source_documents WHERE sha256=?", (sha256,)
            ).fetchone()
            if existing:
                source_doc_id = existing[0]
            else:
                cur = conn.execute(
                    "INSERT INTO source_documents(file_path, sha256, source_kind, year) VALUES(?,?,?,?)",
                    (md_path, sha256, "pdf", year),
                )
                source_doc_id = cur.lastrowid
                conn.commit()

        # Idempotency: if chunks already exist for this source, skip re-chunking.
        already = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_document_id=?",
            (source_doc_id,),
        ).fetchone()[0]
        if already:
            n_chunks = already
        else:
            text = Path(md_path).read_text(encoding="utf-8")
            chunks_text = chunk_markdown(text, max_tokens=300, overlap=50)
            for i, ctext in enumerate(chunks_text):
                conn.execute(
                    "INSERT INTO chunks(source_document_id, chunk_index, text, faiss_index_offset) "
                    "VALUES(?,?,?,?)",
                    (source_doc_id, i, ctext, i),
                )
            conn.commit()
            n_chunks = len(chunks_text)

    # Rebuild in-memory FAISS index from all chunks in DB
    with get_conn(db_path, read_only=True) as conn:
        rows = conn.execute(
            "SELECT id, source_document_id, text FROM chunks ORDER BY id"
        ).fetchall()
    chunk_dicts = [
        {"id": r[0], "source_document_id": r[1], "text": r[2]} for r in rows
    ]
    index = build_index(chunk_dicts, embedder=embedder)
    return n_chunks


def ingest_pdf(pdf_path: str, db_path: str, embedder, year: int | None = None) -> dict:
    """Full PDF ingest: markitdown → structured extract + chunks + embeddings."""
    import hashlib
    from pathlib import Path
    sha256 = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()

    # 1) markitdown
    md_path = Path("data/cache/markdown") / f"{sha256}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    import sys
    subprocess.run(
        [sys.executable, "-m", "markitdown", pdf_path, "-o", str(md_path)],
        check=True,
    )

    # 2) structured extract (best-effort, may fail without API key)
    md_text = md_path.read_text(encoding="utf-8")
    try:
        structured = _extract_structured(md_text)
    except Exception as e:
        structured = {"rounds": [], "cutoffs": [], "requirements": [], "_error": str(e)}

    # 3) chunk + embed (always) — also resolves source_document_id
    init_db(db_path)
    with get_conn(db_path, read_only=False) as conn:
        existing = conn.execute(
            "SELECT id FROM source_documents WHERE sha256=?", (sha256,)
        ).fetchone()
        if existing:
            source_document_id = existing[0]
        else:
            cur = conn.execute(
                "INSERT INTO source_documents (file_path, sha256, source_kind, year) "
                "VALUES (?, ?, 'pdf', ?)",
                (str(pdf_path), sha256, year),
            )
            source_document_id = cur.lastrowid
            conn.commit()

    # Persist structured data into DB tables (best-effort)
    if structured and not structured.get("_error"):
        try:
            counts = _persist_structured(db_path, structured, source_document_id)
            structured["_db_counts"] = counts
        except Exception as e:
            structured.setdefault("_error", f"persist: {e}")

    # Chunk + embed (always)
    n_chunks = ingest_pdf_markdown_only(str(md_path), sha256, db_path, embedder,
                                        source_doc_id=source_document_id,
                                        year=year)

    return {"sha256": sha256, "markdown_path": str(md_path), "chunks": n_chunks,
            "structured": structured, "source_document_id": source_document_id}


if __name__ == "__main__":
    sys.exit(main())