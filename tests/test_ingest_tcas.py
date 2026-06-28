"""Tests for TCAS JSON ingestion."""
import gzip, json, hashlib
from pathlib import Path
from src.db import init_db, get_conn
from src.ingest import ingest_tcas_from_path

def test_ingest_tcas_loads_filtered_rows(tmp_db_path, tmp_path):
    fixture = Path("tests/fixtures/tcas_sample.json")
    init_db(tmp_db_path)
    summary = ingest_tcas_from_path(
        json_path=fixture,
        gz_path=None,
        db_path=tmp_db_path,
        allowed_tcas_ids={"001", "004"},
    )
    assert summary["universities"] == 2
    assert summary["programs"] == 2

    with get_conn(tmp_db_path, read_only=True) as conn:
        unis = conn.execute("SELECT tcas_id, name_th FROM universities ORDER BY tcas_id").fetchall()
        progs = conn.execute(
            "SELECT tcas_program_id, program_name_th, cost FROM programs ORDER BY tcas_program_id"
        ).fetchall()
    assert unis == [("001", "จุฬาลงกรณ์มหาวิทยาลัย"), ("004", "มหาวิทยาลัยเชียงใหม่")]
    assert progs[0] == ("P001", "วิศวกรรมคอมพิวเตอร์", 25000.0)

def test_ingest_tcas_is_idempotent(tmp_db_path):
    fixture = Path("tests/fixtures/tcas_sample.json")
    init_db(tmp_db_path)
    ingest_tcas_from_path(fixture, None, tmp_db_path, {"001", "004"})
    with get_conn(tmp_db_path, read_only=True) as conn:
        ids_before = conn.execute(
            "SELECT id, tcas_program_id FROM programs ORDER BY id"
        ).fetchall()
    summary2 = ingest_tcas_from_path(fixture, None, tmp_db_path, {"001", "004"})
    with get_conn(tmp_db_path, read_only=True) as conn:
        ids_after = conn.execute(
            "SELECT id, tcas_program_id FROM programs ORDER BY id"
        ).fetchall()
    assert ids_before == ids_after
    assert summary2["programs"] == 0

def test_ingest_tcas_records_source_document(tmp_db_path, tmp_path):
    gz = tmp_path / "tcas.json.gz"
    raw = Path("tests/fixtures/tcas_sample.json").read_bytes()
    gz.write_bytes(gzip.compress(raw))
    init_db(tmp_db_path)
    ingest_tcas_from_path(
        json_path=None,
        gz_path=gz,
        db_path=tmp_db_path,
        allowed_tcas_ids={"001", "004"},
    )
    with get_conn(tmp_db_path, read_only=True) as conn:
        sd = conn.execute(
            "SELECT file_path, sha256, source_kind FROM source_documents"
        ).fetchone()
    assert sd[0].endswith("tcas.json.gz")
    assert sd[1] == hashlib.sha256(raw).hexdigest()
    assert sd[2] == "tcas_json"