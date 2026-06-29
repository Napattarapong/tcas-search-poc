"""Add Markdown structure (headers, bullets, bold) to cleaned markdowns.

This is the human-readability layer on top of clean_markdowns.py. It:

  * promotes `คณะXXX` / `สาขาวิชาXXX` lines to ## / ### headers
  * promotes section labels (คุณสมบัติผู้สมัคร, เกณฑ์ขั้นต่ำ, สัดส่วน)
    to #### headers
  * rewrites `⬢` / `⬡` glyph bullets to standard `-`
  * bolds key numbers (Adj. T-SCORE minimums, weight percentages)
  * drops empty table rows
  * drops residual page markers like "1 / 7"

Idempotent: safe to re-run on already-structured output.
"""
from __future__ import annotations
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Line starting with faculty name "คณะ<NAME>" (not in middle of sentence,
# not the table-header "คณะ/สาขาวิชา").
FACULTY_LINE_RE = re.compile(r"^คณะ(?P<name>[^/\s].*)$")
PROGRAM_LINE_RE = re.compile(r"^สาขาวิชา(?P<name>\S.*)$")

# Inline TCASCODE marker (program unique id)
TCASCODE_RE = re.compile(r"(TCASCODE:\s*[\dA-Z]+)")

# Combining Thai marks we strip before matching (so "น้ํา" == "น้ำ").
COMBINING_MARKS = "ัิีึืฺุู" \
                  "็่้๊๋์ํ๎"

def _normalize_thai(text: str) -> str:
    """Strip Thai combining/vowel marks for matching purposes only."""
    return "".join(c for c in text if c not in COMBINING_MARKS)

# Section labels -> header text (matched against normalized text).
SECTION_HEADERS_NORM: dict[str, str] = {
    _normalize_thai("คุณสมบัติผู้สมัคร"): "## คุณสมบัติผู้สมัคร",
    _normalize_thai("เกณฑ์ขั้นต่ำ"): "## เกณฑ์ขั้นต่ำ",
    _normalize_thai("สัดส่วนที่ใช้ในการคัดเลือก"): "## สัดส่วนคะแนน",
    _normalize_thai("จำนวนรับตามประกาศ(คน)"): "## จำนวนรับ",
    _normalize_thai("จำนวนรับตาม"): "## จำนวนรับ",
}

# Drop lines that are just an empty markdown table row like "|     |     |"
EMPTY_TABLE_ROW_RE = re.compile(r"^\|[\s|]+\|$")

# Drop page markers like "1 / 7" or "2 / 433"
PAGE_MARKER_RE = re.compile(r"^\s*\d+\s*/\s*\d+\s*$")

# Lines that look like "รหัสโครงการ 00420101201014 ⬢ ..."
PROJECT_CODE_RE = re.compile(r"^รหัสโครงการ\s+(\d+)\s*(⬢)?\s*(.*)$")

# Score line: "Adj. T-SCORE ไม่น้อยกว่า 50" -> bold the number
SCORE_MIN_RE = re.compile(r"(ไม่น้อยกว่า)\s+(\d+(?:\.\d+)?)")

# Weight: "ค่าน้ำหนักร้อยละ 5" -> bold the percentage.
# "น้ำ" in the actual text is composed as น + nikhahit + mai tho + sara aa (4 chars).
WEIGHT_RE = re.compile(r"(?:ค่า)?นํ้าหนักร้อยละ\s+(\d+(?:\.\d+)?)")

# Diamond bullets -> "-"
DIAMOND_BULLET_RE = re.compile(r"^[⬢⬡]\s*")

# Bare bullet (just "-" or "- " or after stripping diamond becomes empty)
BARE_BULLET_RE = re.compile(r"^-\s*$")


def _is_university_header_line(line: str) -> bool:
    """The first content line is the university title block."""
    stripped = line.strip()
    if not stripped:
        return False
    return (
        "การรับนักศึกษา" in stripped
        or "การคัดเลือกบุคคล" in stripped
        or stripped.startswith("ข้อมูลการรับ")
    )


def structure_markdown(text: str) -> tuple[str, dict]:
    stats = {
        "faculty_headers": 0,
        "program_headers": 0,
        "section_headers": 0,
        "bullets_normalized": 0,
        "empty_table_rows_dropped": 0,
        "page_markers_dropped": 0,
        "tcascode_bolded": 0,
        "scores_bolded": 0,
        "weights_bolded": 0,
        "university_header_added": 0,
    }

    out: list[str] = []
    first_meaningful = True
    lines = text.split("\n")

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()

        # 1) Drop empty table rows
        if EMPTY_TABLE_ROW_RE.match(line):
            stats["empty_table_rows_dropped"] += 1
            i += 1
            continue

        # 2) Drop page markers like "1 / 7"
        if PAGE_MARKER_RE.match(line):
            stats["page_markers_dropped"] += 1
            i += 1
            continue

        # 3) Add university-level header once
        if first_meaningful and _is_university_header_line(line):
            out.append(f"# {line.strip()}")
            stats["university_header_added"] += 1
            first_meaningful = False
            i += 1
            # skip any immediate blank lines / repeated header runs
            while i < len(lines) and (
                not lines[i].strip()
                or _is_university_header_line(lines[i])
                or lines[i].strip().startswith("TCAS69")
                or lines[i].strip().startswith("ปีการศึกษา")
            ):
                if lines[i].strip().startswith("TCAS69"):
                    out.append(f"**{lines[i].strip()}**")
                i += 1
            out.append("")
            continue

        # 4) Faculty header
        m = FACULTY_LINE_RE.match(line)
        if m:
            rest = m.group(1).strip()
            # strip continuation tag like "(ต่อ)"
            rest = re.sub(r"\s*\(ต่อ\)\s*$", "", rest)
            out.append(f"## คณะ{rest}")
            out.append("")
            stats["faculty_headers"] += 1
            i += 1
            continue

        # 5) Program header (สาขาวิชา)
        m = PROGRAM_LINE_RE.match(line)
        if m:
            rest = m.group(1).strip()
            out.append(f"### สาขาวิชา{rest}")
            stats["program_headers"] += 1
            i += 1
            continue

        # 6) Inline TCASCODE on the same line as program info: bold it
        if TCASCODE_RE.search(line):
            line = TCASCODE_RE.sub(lambda m: f"**{m.group(1)}**", line)
            stats["tcascode_bolded"] += 1

        # 7) Section labels (คุณสมบัติ / เกณฑ์ / สัดส่วน).
        # Match against normalized text so nikhahit/etc don't break detection.
        stripped_norm = _normalize_thai(line.strip())
        matched_section = False
        for label_norm, header in SECTION_HEADERS_NORM.items():
            if stripped_norm == label_norm or stripped_norm.startswith(label_norm):
                out.append("")
                out.append(header)
                out.append("")
                stats["section_headers"] += 1
                matched_section = True
                break
        if matched_section:
            i += 1
            continue

        # 8) Project code line
        m = PROJECT_CODE_RE.match(line)
        if m:
            code = m.group(1)
            tail = m.group(3).strip()
            tail = DIAMOND_BULLET_RE.sub("", tail)
            out.append(f"**รหัสโครงการ:** `{code}` {tail}".rstrip())
            i += 1
            continue

        # 9) Diamond bullet at start of line -> "- "
        if DIAMOND_BULLET_RE.match(line):
            line = DIAMOND_BULLET_RE.sub("- ", line)
            stats["bullets_normalized"] += 1

        # Drop bare bullet lines (just "- " with no content)
        if BARE_BULLET_RE.match(line):
            i += 1
            continue

        # 10) Bold scores and weights within the line
        new_line, n_score = SCORE_MIN_RE.subn(r"\1 **\2**", line)
        if n_score:
            line = new_line
            stats["scores_bolded"] += n_score
        new_line, n_weight = WEIGHT_RE.subn(r"น้ำหนัก **\1%**", line)
        if n_weight:
            line = new_line
            stats["weights_bolded"] += n_weight

        out.append(line)
        first_meaningful = False
        i += 1

    # Collapse 3+ blank lines to 2
    final = "\n".join(out)
    final = re.sub(r"\n{3,}", "\n\n", final)
    return final, stats


def main() -> int:
    md_dir = Path("data/cache/markdown")
    backups_dir = md_dir / "_originals_structured"
    backups_dir.mkdir(exist_ok=True)

    total = {
        "faculty_headers": 0, "program_headers": 0, "section_headers": 0,
        "bullets_normalized": 0, "empty_table_rows_dropped": 0,
        "page_markers_dropped": 0, "tcascode_bolded": 0,
        "scores_bolded": 0, "weights_bolded": 0,
        "university_header_added": 0,
    }
    for f in sorted(md_dir.glob("*.md")):
        orig = f.read_text(encoding="utf-8")
        structured, stats = structure_markdown(orig)
        backup = backups_dir / f.name
        if not backup.exists():
            backup.write_text(orig, encoding="utf-8")
        f.write_text(structured, encoding="utf-8")
        for k, v in stats.items():
            total[k] += v
        print(
            f"  {f.name[:20]}...  uni={stats['university_header_added']}  "
            f"fac={stats['faculty_headers']:>3}  prog={stats['program_headers']:>3}  "
            f"sect={stats['section_headers']:>3}  bullets={stats['bullets_normalized']:>4}  "
            f"scores={stats['scores_bolded']:>4}  "
            f"weights={stats['weights_bolded']:>4}  "
            f"tbl_drop={stats['empty_table_rows_dropped']:>3}  "
            f"size {len(orig):>10,} -> {len(structured):>10,}"
        )
    print()
    print(f"TOTAL: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
