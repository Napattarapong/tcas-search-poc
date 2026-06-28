"""Tests for PDF ingest. Uses a fake LLM and a real markitdown run on a tiny PDF."""
import json
from pathlib import Path
from unittest.mock import patch
from src.db import init_db, get_conn
from src.ingest import ingest_pdf_from_path
from src.vector_search import FakeEmbedder

LLM_RESPONSE = json.dumps({
    "admission_rounds": [{
        "program_tcas_id": "P001",
        "year": 2569,
        "round_no": 1,
        "apply_open": "2026-01-15",
        "apply_close": "2026-02-15",
        "exam_date": "2026-03-01",
        "interview_date": None,
        "seats": 50,
    }],
    "cutoff_scores": [{
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
})


def _seed_program(tmp_db_path):
    init_db(tmp_db_path)
    from src.ingest import ingest_tcas_from_path
    ingest_tcas_from_path(
        json_path=Path("tests/fixtures/tcas_sample.json"),
        gz_path=None,
        db_path=tmp_db_path,
        allowed_tcas_ids={"001", "004"},
    )


def test_ingest_pdf_creates_round_cutoff_and_requirements(tmp_db_path):
    _seed_program(tmp_db_path)
    with patch("src.ingest.chat", return_value=LLM_RESPONSE):
        summary = ingest_pdf_from_path(
            pdf_path=Path("tests/fixtures/sample.pdf"),
            db_path=tmp_db_path,
            year=2569,
        )
    assert summary["rounds"] == 1
    assert summary["cutoffs"] == 1
    assert summary["requirements"] == 2
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
    with patch("src.ingest.chat", return_value=LLM_RESPONSE):
        summary1 = ingest_pdf_from_path(Path("tests/fixtures/sample.pdf"), tmp_db_path, 2569)
        summary2 = ingest_pdf_from_path(Path("tests/fixtures/sample.pdf"), tmp_db_path, 2569)
    assert summary2["skipped"] is True
    with get_conn(tmp_db_path, read_only=True) as conn:
        n = conn.execute("SELECT COUNT(*) FROM admission_rounds").fetchone()[0]
    assert n == 1


def test_pdf_ingest_populates_chunks_table(tmp_db, monkeypatch):
    """Even without an LLM extract, the chunks table should be populated from markdown."""
    from src.db import init_db, get_conn
    init_db(tmp_db)

    # Skip the structured-extract step by mocking the LLM
    from src import ingest as ingest_mod
    monkeypatch.setattr(ingest_mod, "_extract_structured", lambda md: {"rounds": [], "cutoffs": [], "requirements": []})

    # Provide a fixture markdown file
    md_path = tmp_db_path(tmp_db) / "announcement.md"
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
