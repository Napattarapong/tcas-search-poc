"""The SQL safety wrapper must reject destructive / multi-statement SQL."""
from __future__ import annotations
import pytest
from src.nl_to_sql import sanitize_sql, run_sql

@pytest.mark.parametrize("bad", [
    "DROP TABLE universities",
    "DELETE FROM programs WHERE id=1",
    "UPDATE universities SET name_th='x'",
    "INSERT INTO universities(tcas_id, name_th) VALUES('x','y')",
    "ALTER TABLE programs ADD COLUMN x INTEGER",
    "ATTACH DATABASE 'x.db' AS x",
    "SELECT 1; DROP TABLE universities",
    "select * from universities",  # valid → should pass
])
def test_sanitize_sql_rejects_destructive(bad):
    if bad.startswith("select"):
        # the only valid case
        out = sanitize_sql(bad)
        assert out.upper().startswith("SELECT")
        assert "LIMIT" in out.upper()
    else:
        with pytest.raises(ValueError):
            sanitize_sql(bad)

def test_run_sql_read_only(tmp_db):
    from src.db import init_db
    init_db(tmp_db)
    cols, rows = run_sql("SELECT name_th FROM universities", tmp_db)
    assert cols == ["name_th"]
    assert rows == []