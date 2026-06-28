"""SQLite schema (7 tables) and connection helpers."""
from __future__ import annotations
import sqlite3
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS source_documents (
  id            INTEGER PRIMARY KEY,
  file_path     TEXT NOT NULL,
  sha256        TEXT NOT NULL UNIQUE,
  source_kind   TEXT NOT NULL,
  university_id INTEGER,
  year          INTEGER,
  ingested_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS universities (
  id                  INTEGER PRIMARY KEY,
  tcas_id             TEXT UNIQUE NOT NULL,
  name_th             TEXT NOT NULL,
  name_en             TEXT,
  university_type_th  TEXT,
  source_document_id  INTEGER REFERENCES source_documents(id)
);

CREATE TABLE IF NOT EXISTS programs (
  id                       INTEGER PRIMARY KEY,
  university_id            INTEGER NOT NULL REFERENCES universities(id),
  tcas_program_id          TEXT UNIQUE NOT NULL,
  program_name_th          TEXT NOT NULL,
  program_name_en          TEXT,
  faculty_name_th          TEXT,
  faculty_name_en          TEXT,
  field_name_th            TEXT,
  field_name_en            TEXT,
  program_type_th          TEXT,
  degree                   TEXT,
  cost                     REAL,
  number_acceptance_mko2   INTEGER,
  major_acceptance_number  INTEGER,
  graduate_rate            REAL,
  employment_rate          REAL,
  median_salary            REAL,
  source_document_id       INTEGER REFERENCES source_documents(id)
);

CREATE TABLE IF NOT EXISTS admission_rounds (
  id               INTEGER PRIMARY KEY,
  program_id       INTEGER NOT NULL REFERENCES programs(id),
  year             INTEGER NOT NULL,
  round_no         INTEGER NOT NULL,
  apply_open       DATE,
  apply_close      DATE,
  exam_date        DATE,
  interview_date   DATE,
  seats            INTEGER,
  source_document_id INTEGER REFERENCES source_documents(id),
  UNIQUE(program_id, year, round_no)
);

CREATE TABLE IF NOT EXISTS cutoff_scores (
  id               INTEGER PRIMARY KEY,
  round_id         INTEGER NOT NULL REFERENCES admission_rounds(id),
  score_type       TEXT NOT NULL,
  min_score        REAL,
  max_score        REAL,
  gpa_min          REAL,
  source_document_id INTEGER REFERENCES source_documents(id)
);

CREATE TABLE IF NOT EXISTS requirements (
  id               INTEGER PRIMARY KEY,
  round_id         INTEGER NOT NULL REFERENCES admission_rounds(id),
  kind             TEXT NOT NULL,
  text             TEXT NOT NULL,
  source_document_id INTEGER REFERENCES source_documents(id)
);

CREATE TABLE IF NOT EXISTS chunks (
  id                  INTEGER PRIMARY KEY,
  source_document_id  INTEGER NOT NULL REFERENCES source_documents(id),
  chunk_index         INTEGER NOT NULL,
  text                TEXT NOT NULL,
  faiss_index_offset  INTEGER NOT NULL,
  UNIQUE(source_document_id, chunk_index)
);
"""


def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


def get_conn(db_path: str, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    return sqlite3.connect(db_path)
