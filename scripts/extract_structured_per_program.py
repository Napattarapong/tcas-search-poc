"""Per-program structured extraction from clean markdowns.

Replaces the broken _extract_structured mega-prompt with a deterministic split
+ one focused LLM call per program. Each program section is delimited by the
`### สาขาวิชa...` headers added by structure_markdowns.py.

For each program section we:
  1. Find the TCASCODE (program's unique id) by regex
  2. Send the section text + explicit JSON schema to the LLM
  3. Validate the response, drop malformed rows
  4. Insert into admission_rounds (and child tables).

Idempotent: skips source_documents that already have admission_rounds rows.
"""
from __future__ import annotations
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))

from src.db import init_db, get_conn
from src.llm import chat, LLMError

PDF_DIR = Path("data/raw/tcas_pdfs")
MD_DIR = Path("data/cache/markdown")
DB_PATH = "data/university.db"

# Match `### สาขาวิชา...` headers (added by structure_markdowns.py)
PROGRAM_HEADER_RE = re.compile(r"^###\s+สาขาวิชา", re.MULTILINE)
# Match any subsequent higher-level header that ends this program block
SECTION_END_RE = re.compile(r"^#{1,3}\s+", re.MULTILINE)

# Pull TCASCODE out of a program's section
TCASCODE_RE = re.compile(r"TCASCODE:\s*([\dA-Z]+)")

# JSON schema for LLM output
SCHEMA_INSTRUCTIONS = """\
Return ONLY valid JSON in this exact shape (no commentary, no markdown fences):
{
  "rounds": [
    {"round_no": <int>, "round_name": "<short Thai>", "seats": <int or null>}
  ],
  "requirements": [
    {"kind": "eligibility"|"doc"|"condition", "text": "<Thai sentence>"}
  ]
}

Rules:
- round_no: TCAS round number (1, 2, 3, or 4).
- round_name: e.g. "Portfolio", "Quota", "Admission", "Direct".
- seats: integer if stated, else null. Do NOT invent values.
- requirements: list every bullet requirement from the section, in order.
- kind: 'eligibility' for คุณสมบัติผู้สมัคร bullets, 'doc' for เอกสาร bullets, 'condition' for other เงื่อนไข.
- If a section has no rounds/requirements, return empty arrays.
- Output ONLY the JSON object.
"""


def split_into_programs(text: str) -> list[tuple[str, str]]:
    """Yield (program_label, section_text) for each program in the markdown."""
    matches = list(PROGRAM_HEADER_RE.finditer(text))
    out: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        # First line is the header like "### สาขาวิชาภาษาอังกฤษ"
        first_line = section.split("\n", 1)[0]
        label = first_line.lstrip("#").strip()
        out.append((label, section))
    return out


def extract_for_section(section: str, max_attempts: int = 2) -> dict:
    """Call the LLM for one program section. Return parsed dict."""
    # Truncate to keep prompt reasonable
    if len(section) > 6000:
        section = section[:6000] + "\n\n[... truncated ...]"
    prompt = (
        SCHEMA_INSTRUCTIONS
        + "\n\n---PROGRAM SECTION---\n"
        + section
    )
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            raw = chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=1500,
            )
            # Strip fences if any
            stripped = raw.strip()
            fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", stripped, re.DOTALL)
            if fence:
                stripped = fence.group(1).strip()
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                raise ValueError("LLM returned non-object JSON")
            return parsed
        except (LLMError, json.JSONDecodeError, ValueError) as e:
            last_err = e
            continue
    raise RuntimeError(f"LLM extraction failed after {max_attempts} attempts: {last_err}")


def persist_program(
    conn, sd_id: int, year: int, tcas_id: str, extracted: dict
) -> tuple[int, int]:
    """Insert rounds/requirements for one program. Return (rounds, requirements) counts."""
    pid_row = conn.execute(
        "SELECT id FROM programs WHERE tcas_program_id=?", (tcas_id,)
    ).fetchone()
    if pid_row is None:
        return (0, 0)
    pid = pid_row[0]

    rounds = extracted.get("rounds") or []
    reqs = extracted.get("requirements") or []
    n_rounds = n_reqs = 0

    for r in rounds:
        try:
            round_no = int(r["round_no"])
        except (KeyError, TypeError, ValueError):
            continue
        round_name = (r.get("round_name") or "").strip() or None
        seats = r.get("seats")
        try:
            seats_i = int(seats) if seats is not None else None
        except (TypeError, ValueError):
            seats_i = None

        cur = conn.execute(
            "INSERT OR IGNORE INTO admission_rounds ("
            "program_id, year, round_no, seats, source_document_id) "
            "VALUES (?,?,?,?,?)",
            (pid, year, round_no, seats_i, sd_id),
        )
        if cur.rowcount == 0:
            continue  # already exists (idempotent)
        round_id = cur.lastrowid
        n_rounds += 1

        # Filter requirements that match this round (LLM doesn't always set round_no)
        for q in reqs:
            text = (q.get("text") or "").strip()
            if not text:
                continue
            kind = q.get("kind") or "condition"
            if kind not in ("eligibility", "doc", "condition"):
                kind = "condition"
            conn.execute(
                "INSERT INTO requirements (round_id, kind, text, source_document_id) "
                "VALUES (?,?,?,?)",
                (round_id, kind, text, sd_id),
            )
            n_reqs += 1

    return (n_rounds, n_reqs)


def process_pdf(pdf: Path, year: int = 2569) -> dict:
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    md_path = MD_DIR / f"{sha}.md"
    if not md_path.exists():
        return {"pdf": pdf.name, "skipped": "no markdown"}

    with get_conn(DB_PATH, read_only=True) as conn:
        sd_row = conn.execute(
            "SELECT id FROM source_documents WHERE sha256=?", (sha,)
        ).fetchone()
    if not sd_row:
        return {"pdf": pdf.name, "skipped": "no source_document row"}
    sd_id = sd_row[0]

    # Idempotency: skip if rounds already populated for this source
    with get_conn(DB_PATH, read_only=True) as conn:
        n_existing = conn.execute(
            "SELECT COUNT(*) FROM admission_rounds WHERE source_document_id=?",
            (sd_id,),
        ).fetchone()[0]
    if n_existing:
        return {"pdf": pdf.name, "skipped": f"already has {n_existing} rounds"}

    text = md_path.read_text(encoding="utf-8")
    programs = split_into_programs(text)
    if not programs:
        return {"pdf": pdf.name, "skipped": "no program headers"}

    stats = {"pdf": pdf.name, "programs": len(programs),
             "rounds": 0, "requirements": 0, "errors": 0}

    init_db(DB_PATH)
    for label, section in programs:
        m = TCASCODE_RE.search(section)
        if not m:
            continue  # No TCASCODE → can't link to programs table
        tcas_id = m.group(1)

        try:
            extracted = extract_for_section(section)
        except Exception as e:
            stats["errors"] += 1
            continue

        with get_conn(DB_PATH, read_only=False) as conn:
            n_rounds, n_reqs = persist_program(conn, sd_id, year, tcas_id, extracted)
            conn.commit()
        stats["rounds"] += n_rounds
        stats["requirements"] += n_reqs

    return stats


def main() -> int:
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {PDF_DIR}")
        return 1

    init_db(DB_PATH)
    print(f"Extracting structured data from {len(pdfs)} PDFs\n", flush=True)

    total = {"rounds": 0, "requirements": 0, "errors": 0, "skipped": 0}
    t0 = time.time()
    for pdf in pdfs:
        t1 = time.time()
        result = process_pdf(pdf, year=2569)
        dt = time.time() - t1
        if "skipped" in result:
            print(f"  SKIP  {pdf.name[:50]:<50}  ({result['skipped']})  ({dt:.1f}s)", flush=True)
            total["skipped"] += 1
            continue
        print(
            f"  OK    {pdf.name[:50]:<50}  programs={result['programs']:>4}  "
            f"rounds={result['rounds']:>4}  reqs={result['requirements']:>4}  "
            f"errs={result['errors']}  ({dt:.1f}s)",
            flush=True,
        )
        total["rounds"] += result["rounds"]
        total["requirements"] += result["requirements"]
        total["errors"] += result["errors"]

    print(f"\nTotal: {total}  wall={time.time()-t0:.1f}s", flush=True)

    # Final DB state
    with get_conn(DB_PATH, read_only=True) as conn:
        for t in ("admission_rounds", "cutoff_scores", "requirements"):
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t}: {n} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())