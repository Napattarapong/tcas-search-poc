"""DB schema tests including the new chunks table."""
from __future__ import annotations
import sqlite3
import pytest
from src.db import init_db, get_conn

def test_init_db_creates_all_seven_tables(tmp_db):
    init_db(tmp_db)
    with get_conn(tmp_db, read_only=True) as conn:
        names = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    assert names == {
        "source_documents", "universities", "programs",
        "admission_rounds", "cutoff_scores", "requirements", "chunks",
    }

def test_chunks_table_has_expected_columns(tmp_db):
    init_db(tmp_db)
    with get_conn(tmp_db, read_only=True) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()]
    assert cols == ["id", "source_document_id", "chunk_index", "text", "faiss_index_offset"]

def test_get_conn_read_only_rejects_writes(tmp_db):
    init_db(tmp_db)
    with get_conn(tmp_db, read_only=True) as conn:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(
                "INSERT INTO universities(tcas_id, name_th) VALUES('x','y')"
            )
