"""Tests for PDF ingest. Uses a fake LLM and a real markitdown run on a tiny PDF."""
import json
import subprocess as _real_subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.db import init_db, get_conn
from src.ingest import ingest_pdf
from src.vector_search import FakeEmbedder

# Capture the real subprocess.run BEFORE any patches are applied.
_REAL_RUN = _real_subprocess.run

# Shape returned by _extract_structured: {"rounds": [...], "cutoffs": [...], "requirements": [...]}
LLM_STRUCTURED = {
    "rounds": [{
        "program_tcas_id": "P001",
        "year": 2569,
        "round_no": 1,
        "apply_open": "2026-01-15",
        "apply_close": "2026-02-15",
        "exam_date": "2026-03-01",
        "interview_date": None,
        "seats": 50,
    }],
    "cutoffs": [{
        "program_tcas_id": "P001",
        "year": 2569,
        "round_no": 1,
        "score_type": "gpa",
        "min_score": None,
        "max_score": None,
        "gpa_min": 3.00,
    }],
    "requirements": [
        {"program_tcas_id": "P001", "year": 2569, "round_no": 1,
         "kind": "doc", "text": "transcript"},
        {"program_tcas_id": "P001", "year": 2569, "round_no": 1,
         "kind": "doc", "text": "ID card"},
    ],
}


def _seed_program(tmp_db_path):
    init_db(tmp_db_path)
    from src.ingest import ingest_tcas_from_path
    ingest_tcas_from_path(
        json_path=Path("tests/fixtures/tcas_sample.json"),
        gz_path=None,
        db_path=tmp_db_path,
        allowed_tcas_ids={"001", "004"},
    )


def fake_markitdown(cmd, **kwargs):
    """Mock for src.ingest.subprocess.run that does the real PDF→markdown via
    `python -m markitdown` (avoiding recursion with the patched subprocess.run).

    Accepts either the bare `["markitdown", pdf, "-o", md]` shape or the
    `["python", "-m", "markitdown", pdf, "-o", md]` shape.
    """
    if cmd and cmd[0] == "markitdown":
        _, pdf_path, _, md_path_str = cmd[:4]
    else:
        # [python, -m, markitdown, pdf, -o, md]
        _, _, _, pdf_path, _, md_path_str = cmd[:6]
    md_path = Path(md_path_str)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    real = _REAL_RUN(
        [sys.executable, "-m", "markitdown", str(pdf_path), "-o", str(md_path)],
        check=True,
        capture_output=True,
    )
    return real


def test_ingest_pdf_creates_round_cutoff_and_requirements(tmp_db_path):
    _seed_program(tmp_db_path)
    with patch("src.ingest._extract_structured", return_value=LLM_STRUCTURED), \
         patch("src.ingest.subprocess.run", side_effect=fake_markitdown):
        summary = ingest_pdf(
            pdf_path="tests/fixtures/sample.pdf",
            db_path=tmp_db_path,
            embedder=FakeEmbedder(dim=16),
            year=2569,
        )
    assert summary["structured"] == LLM_STRUCTURED
    assert summary["chunks"] >= 1
    assert summary["sha256"]
    with get_conn(tmp_db_path, read_only=True) as conn:
        n_rounds = conn.execute("SELECT COUNT(*) FROM admission_rounds").fetchone()[0]
        n_cut = conn.execute("SELECT COUNT(*) FROM cutoff_scores").fetchone()[0]
        n_req = conn.execute("SELECT COUNT(*) FROM requirements").fetchone()[0]
        apply_close = conn.execute(
            "SELECT apply_close FROM admission_rounds WHERE year=?", (2569,)
        ).fetchone()[0]
    assert n_rounds == 1 and n_cut == 1 and n_req == 2
    assert apply_close == "2026-02-15"


def test_ingest_pdf_is_idempotent(tmp_db_path):
    _seed_program(tmp_db_path)
    with patch("src.ingest._extract_structured", return_value=LLM_STRUCTURED), \
         patch("src.ingest.subprocess.run", side_effect=fake_markitdown):
        summary1 = ingest_pdf(
            pdf_path="tests/fixtures/sample.pdf",
            db_path=tmp_db_path,
            embedder=FakeEmbedder(dim=16),
            year=2569,
        )
        summary2 = ingest_pdf(
            pdf_path="tests/fixtures/sample.pdf",
            db_path=tmp_db_path,
            embedder=FakeEmbedder(dim=16),
            year=2569,
        )
    # Second call should not duplicate rows
    assert summary1["sha256"] == summary2["sha256"]
    with get_conn(tmp_db_path, read_only=True) as conn:
        n = conn.execute("SELECT COUNT(*) FROM admission_rounds").fetchone()[0]
    assert n == 1


def test_pdf_ingest_populates_chunks_table(tmp_db, monkeypatch):
    """Even without an LLM extract, the chunks table should be populated from markdown."""
    from src.db import init_db, get_conn
    init_db(tmp_db)

    # Skip the structured-extract step by mocking it to no-op
    from src import ingest as ingest_mod
    monkeypatch.setattr(ingest_mod, "_extract_structured", lambda md: {"rounds": [], "cutoffs": [], "requirements": []})

    # Provide a fixture markdown file
    md_dir = tmp_db_path(tmp_db)
    md_path = md_dir / "announcement.md"
    md_path.write_text(
        "# ประกาศรับสมัคร\n\nนักศึกษาต้องมี GPA ไม่ต่ำกว่า 3.5 "
        "และสามารถสมัครได้ตั้งแต่วันที่ 1 มีนาคม ถึง 30 เมษายน "
        "2569 ผ่านระบบออนไลน์เท่านั้น\n",
        encoding="utf-8",
    )
    # Run only the chunk+embed portion (the LLM extract is mocked to no-op)
    ingest_mod.ingest_pdf_markdown_only(str(md_path), sha256="abc", db_path=tmp_db, embedder=FakeEmbedder(dim=16))

    with get_conn(tmp_db, read_only=True) as conn:
        n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert n >= 1


def tmp_db_path(db_path: str):
    from pathlib import Path
    return Path(db_path).parent