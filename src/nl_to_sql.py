"""NL→SQL: safety wrapper around generated SQL + LLM-backed generation."""
from __future__ import annotations
import json
import re

from src.db import get_conn
from src.llm import chat

_FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
    "ATTACH", "DETACH", "REPLACE", "TRUNCATE", "CREATE",
]
_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)


def sanitize_sql(sql: str) -> str:
    """Validate and lightly normalize LLM-generated SQL. Raises ValueError on unsafe input."""
    if sql is None:
        raise ValueError("empty sql")
    text = sql.strip().rstrip(";").strip()
    # Strip code fences if present
    text = re.sub(r"^```(?:sql)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    # Reject multi-statement (any further ';' inside the body)
    # Note: intentionally naive — semicolons inside string literals (e.g. Thai program names)
    # would need a proper tokenizer to allow; doing so is out of POC scope
    if ";" in text:
        raise ValueError(f"multi-statement SQL rejected: {text!r}")

    upper = text.upper().lstrip()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        raise ValueError(f"only SELECT or WITH allowed, got: {text[:40]!r}")

    for kw in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper):
            raise ValueError(f"forbidden keyword {kw} in SQL: {text!r}")

    if not _LIMIT_RE.search(text):
        text = text.rstrip() + " LIMIT 50"
    return text


def run_sql(sql: str, db_path: str) -> tuple[list[str], list[tuple]]:
    """Sanitize + execute SQL in read-only mode. Return (column_names, rows)."""
    safe = sanitize_sql(sql)
    with get_conn(db_path, read_only=True) as conn:
        cur = conn.execute(safe)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
    return cols, rows


SCHEMA_DESCRIPTION = """
Tables (SQLite):
- universities(id, tcas_id, name_th, name_en, university_type_th)
- programs(id, university_id, tcas_program_id, program_name_th, program_name_en, faculty_name_th, field_name_th, program_type_th, degree, cost, number_acceptance_mko2, major_acceptance_number, graduate_rate, employment_rate, median_salary)
- admission_rounds(id, program_id, year, round_no, apply_open, apply_close, exam_date, interview_date, seats)
- cutoff_scores(id, round_id, score_type, min_score, max_score, gpa_min)
- requirements(id, round_id, kind, text)  -- kind in ('doc','eligibility','condition')
- source_documents(id, file_path, sha256, source_kind, university_id, year)

Notes:
- Thai academic year e.g. 2569 means the year starting May 2026.
- Use LIKE '%keyword%' for Thai program/faculty names.
"""

FEW_SHOT_EXAMPLES = """
Example 1
Q: จุฬาฯ มีคณะวิศวกรรมศาสตร์ไหม
A: {"sql": "SELECT p.id, p.program_name_th, p.cost FROM programs p JOIN universities u ON p.university_id=u.id WHERE u.tcas_id='001' AND p.faculty_name_th LIKE '%วิศวกรรม%'", "intent": "list engineering programs at Chulalongkorn"}

Example 2
Q: วิศวะจุฬารอบ 1 ปี 2569 รับกี่คน
A: {"sql": "SELECT ar.seats, ar.apply_open, ar.apply_close FROM admission_rounds ar JOIN programs p ON ar.program_id=p.id JOIN universities u ON p.university_id=u.id WHERE u.tcas_id='001' AND p.program_name_th LIKE '%วิศว%' AND ar.year=2569 AND ar.round_no=1", "intent": "engineering round 1 2569 seats at Chulalongkorn"}

Example 3
Q: ค่าเล่าเรียนคณะวิทยาศาสตร์ มหิดล
A: {"sql": "SELECT p.program_name_th, p.cost FROM programs p JOIN universities u ON p.university_id=u.id WHERE u.tcas_id='006' AND p.faculty_name_th LIKE '%วิทยาศาสตร์%'", "intent": "Mahidol science tuition"}
"""


def _build_prompt(question: str) -> list[dict]:
    return [{
        "role": "user",
        "content": (
            "You translate Thai questions into ONE safe SELECT or WITH statement over the schema.\n"
            "Return ONLY JSON: {\"sql\": \"...\", \"intent\": \"...\"}.\n"
            "Rules: only SELECT or WITH; end with LIMIT 50 (or another number if appropriate); "
            "do not invent columns; do not join tables not in the schema.\n\n"
            f"{SCHEMA_DESCRIPTION}\n{FEW_SHOT_EXAMPLES}\n"
            f"Question: {question}\nAnswer:"
        ),
    }]


def generate_sql(question: str, db_path: str, max_retries: int = 3) -> tuple[str, str]:
    """Call the LLM, parse JSON, sanitize SQL, retry on rejection.

    `db_path` is reserved for per-query connection scoping; currently unused (schema is baked into the prompt).
    `max_retries` is the total number of attempts (not retries-after-first)."""
    last_err: Exception | None = None
    for _ in range(max_retries):
        raw = chat(
            messages=_build_prompt(question),
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        try:
            parsed = json.loads(raw)
            sql = parsed["sql"]
            intent = parsed.get("intent", "")
            safe = sanitize_sql(sql)
            return safe, intent
        except (ValueError, json.JSONDecodeError, KeyError) as e:
            last_err = e
            continue
    raise ValueError(f"generate_sql failed after {max_retries} retries: {last_err}")
