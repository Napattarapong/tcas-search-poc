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
from src.llm import chat


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

# NOTE: {markdown} is sourced from untrusted PDFs. Validate LLM JSON output against
# the schema before using it (prompt injection risk — mitigation is out of scope for POC).
EXTRACT_PROMPT_TEMPLATE = """You are a structured-data extractor. From the markdown below, produce ONE JSON object with three arrays: admission_rounds, cutoff_scores, requirements.

Schema:
- admission_rounds: {{program_tcas_id: str, year: int, round_no: int, apply_open: date|null, apply_close: date|null, exam_date: date|null, interview_date: date|null, seats: int|null}}
- cutoff_scores: {{program_tcas_id: str, year: int, round_no: int, score_type: str, min_score: number|null, max_score: number|null, gpa_min: number|null}}
- requirements: {{program_tcas_id: str, year: int, round_no: int, kind: "doc"|"eligibility"|"condition", text: str}}

Use Thai year as-is (2569 = 2026 academic year).
Output ONLY JSON. No commentary.

Markdown:
---
{markdown}
"""


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


def ingest_pdf_from_path(
    pdf_path: Path,
    db_path: str,
    year: int,
) -> dict:
    """Convert PDF → markdown → LLM extract → upsert admission_rounds, cutoff_scores, requirements."""
    pdf_path = Path(pdf_path)
    raw = pdf_path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    init_db(db_path)
    with get_conn(db_path, read_only=False) as conn:
        existing = conn.execute(
            "SELECT id FROM source_documents WHERE sha256=?", (sha,)
        ).fetchone()
        if existing:
            return {"rounds": 0, "cutoffs": 0, "requirements": 0,
                    "skipped": True, "source_document_id": existing[0]}

        cur = conn.execute(
            "INSERT INTO source_documents (file_path, sha256, source_kind, year) "
            "VALUES (?, ?, 'pdf', ?)",
            (str(pdf_path), sha, year),
        )
        sd_id = cur.lastrowid
        conn.commit()

    md_path = Path("data/cache/markdown") / f"{sha}.md"
    _run_markitdown(pdf_path, md_path)
    markdown = md_path.read_text(encoding="utf-8", errors="replace")

    prompt = EXTRACT_PROMPT_TEMPLATE.format(markdown=markdown)
    raw_response = chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    Path("data/extracted").mkdir(parents=True, exist_ok=True)
    Path(f"data/extracted/{sha}.json").write_text(raw_response, encoding="utf-8")

    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        return {"rounds": 0, "cutoffs": 0, "requirements": 0, "error": "json_decode",
                "source_document_id": sd_id}

    with get_conn(db_path, read_only=False) as conn:
        n_round = n_cut = n_req = 0
        for r in parsed.get("admission_rounds", []):
            pid = _lookup_program_id(conn, r["program_tcas_id"])
            if pid is None:
                continue
            cur = conn.execute(
                "INSERT INTO admission_rounds (program_id, year, round_no,"
                "apply_open, apply_close, exam_date, interview_date, seats,"
                "source_document_id) VALUES (?,?,?,?,?,?,?,?,?)",
                (pid, r["year"], r["round_no"], r.get("apply_open"),
                 r.get("apply_close"), r.get("exam_date"),
                 r.get("interview_date"), r.get("seats"), sd_id),
            )
            round_id = cur.lastrowid
            n_round += 1
            for c in parsed.get("cutoff_scores", []):
                if (c.get("program_tcas_id") == r["program_tcas_id"]
                        and c.get("year") == r["year"]
                        and c.get("round_no") == r["round_no"]):
                    conn.execute(
                        "INSERT INTO cutoff_scores (round_id, score_type,"
                        "min_score, max_score, gpa_min, source_document_id) "
                        "VALUES (?,?,?,?,?,?)",
                        (round_id, c["score_type"], c.get("min_score"),
                         c.get("max_score"), c.get("gpa_min"), sd_id),
                    )
                    n_cut += 1
            for q in parsed.get("requirements", []):
                if (q.get("program_tcas_id") == r["program_tcas_id"]
                        and q.get("year") == r["year"]
                        and q.get("round_no") == r["round_no"]):
                    conn.execute(
                        "INSERT INTO requirements (round_id, kind, text,"
                        "source_document_id) VALUES (?,?,?,?)",
                        (round_id, q["kind"], q["text"], sd_id),
                    )
                    n_req += 1
        conn.commit()
    return {"rounds": n_round, "cutoffs": n_cut, "requirements": n_req,
            "source_document_id": sd_id}


def cli_pdf(args) -> None:
    summary = ingest_pdf_from_path(
        pdf_path=Path(args.pdf),
        db_path=args.db_path,
        year=args.year,
    )
    if "error" in summary:
        print(json.dumps(summary, ensure_ascii=False), file=sys.stderr)
        sys.exit(2)
    print(json.dumps(summary, ensure_ascii=False))


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


if __name__ == "__main__":
    sys.exit(main())