"""Clean PDF-derived markdowns: fix Mac Thai PUA mojibake, strip page footers.

Why this exists:
  markitdown preserves Private Use Area glyphs from Mac Thai-encoded PDFs.
  These appear as U+F701..U+F71A instead of real Thai vowels/tone marks.
  This script maps them back to standard Thai Unicode and strips repeated
  page footer noise so downstream extraction (regex + LLM) sees clean text.

This is a deterministic, fast, free alternative to LLM-cleaning the whole file.
"""
from __future__ import annotations
import re
from pathlib import Path


def _build_mac_thai_pua_map() -> dict[str, str]:
    """Map Mac Thai PUA codepoints (U+F700..U+F71A) to Unicode Thai.

    Derived empirically from observed contexts in เชียงใหม่ PDFs:
      F701=ิ (เปิด), F702=ี (ปี), F703=ึ (ฝึก), F704=ื (เฟือน),
      F706=ู (แฟ้ม), F70A=่ (ใหม่), F70B=้ (ข้อมูล), F70E=์ (ศาสตร์),
      F712=็ (เป็น)
    Other PUA codepoints that show up in smaller quantities get dropped.
    """
    pua_to_thai: dict[int, str] = {
        0xF701: "ิ",  # sara i          (เปิด)
        0xF702: "ี",  # sara ii         (ปีการศึกษา)
        0xF703: "ึ",  # sara ue         (ฝึก)
        0xF704: "ื",  # sara uea        (เฟือน)
        0xF706: "้",  # mai tho (alt)   (แฟ้ม)  -- Mac Thai duplicate slot
        0xF70A: "่",  # mai ek          (ใหม่)
        0xF70B: "้",  # mai tho         (ข้อมูล)
        0xF70E: "์",  # thanthakhat     (ศาสตร์)
        0xF712: "็",  # mai yamakkan    (เป็น)
    }
    return {chr(k): v for k, v in pua_to_thai.items()}


MAC_THAI_PUA_TO_THAI: dict[str, str] = _build_mac_thai_pua_map()

# Build a single char-class regex covering U+E000..U+F8FF so unknown PUA chars
# get stripped without confusing downstream regex/embedding models.
PUA_RANGE_RE = re.compile(
    "[" + "".join(chr(cp) for cp in range(0xE000, 0xF900)) + "]"
)

# Lines that are pure page-footer noise — drop entirely.
FOOTER_LINE_RE = re.compile(
    r"^\s*พิมพ[์ี]?เอกสาร"
    r"วันที่\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+"
    r"หน[้่]าที่\s+\d+\s+จาก\s+\d+\s*$"
)

# Collapse runs of 3+ blank lines down to 2.
BLANK_RUN_RE = re.compile(r"\n{3,}")

# Collapse 3+ spaces (excluding line breaks) down to 1.
MANY_SPACES_RE = re.compile(r"[ \t]{3,}")


def clean_markdown(text: str) -> tuple[str, dict]:
    """Return (cleaned_text, stats)."""
    stats = {
        "pua_replaced": 0,
        "pua_dropped": 0,
        "footer_lines_dropped": 0,
        "blank_runs_collapsed": 0,
    }

    # 1) Replace known PUA chars with Thai Unicode; drop unknown PUA chars.
    def _sub_pua(m: re.Match) -> str:
        ch = m.group(0)
        replacement = MAC_THAI_PUA_TO_THAI.get(ch)
        if replacement is not None:
            stats["pua_replaced"] += 1
            return replacement
        stats["pua_dropped"] += 1
        return ""

    text = PUA_RANGE_RE.sub(_sub_pua, text)

    # 2) Strip page footer lines (one per page).
    out_lines: list[str] = []
    for line in text.split("\n"):
        if FOOTER_LINE_RE.match(line):
            stats["footer_lines_dropped"] += 1
            continue
        out_lines.append(line)
    text = "\n".join(out_lines)

    # 3) Collapse runs of blank lines and long whitespace runs.
    new_text, n = BLANK_RUN_RE.subn("\n\n", text)
    stats["blank_runs_collapsed"] = n
    text = MANY_SPACES_RE.sub(" ", new_text)

    return text, stats


def main() -> int:
    md_dir = Path("data/cache/markdown")
    backups_dir = md_dir / "_originals"
    backups_dir.mkdir(exist_ok=True)

    total_stats = {
        "pua_replaced": 0,
        "pua_dropped": 0,
        "footer_lines_dropped": 0,
        "blank_runs_collapsed": 0,
    }
    for f in sorted(md_dir.glob("*.md")):
        orig = f.read_text(encoding="utf-8")
        cleaned, stats = clean_markdown(orig)
        backup = backups_dir / f.name
        if not backup.exists():
            backup.write_text(orig, encoding="utf-8")
        f.write_text(cleaned, encoding="utf-8")
        for k, v in stats.items():
            total_stats[k] += v
        print(
            f"  {f.name[:20]}...  PUA->Thai={stats['pua_replaced']:>5}  "
            f"PUA_drop={stats['pua_dropped']:>4}  "
            f"footer={stats['footer_lines_dropped']:>3}  "
            f"blank_runs={stats['blank_runs_collapsed']:>3}  "
            f"size {len(orig):>10,} -> {len(cleaned):>10,}"
        )
    print()
    print(f"TOTAL: {total_stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
