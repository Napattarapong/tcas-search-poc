"""PDF -> markdown conversion (PyMuPDF table mode) + page/chunk utilities.

Tables are rendered as markdown (nested headers merged column-wise) so each
program stays on one aligned row. Pages are split on the `<!-- page N -->`
marker that convert emits.
"""
import re
import sys

import fitz  # PyMuPDF


def _clean(cell):
    return (cell or "").replace("\n", " ").replace("|", "/").strip()


def _is_code(cell):
    c = (cell or "").strip()
    return bool(c) and c.isdigit()


def render_table(rows):
    if not rows:
        return ""
    start = 0
    for i, r in enumerate(rows):
        if any(_is_code(c) for c in r[1:3]):
            start = i
            break
    header_rows = rows[:start] if start > 0 else rows[:1]
    data_rows = rows[start:] if start > 0 else rows[1:]
    ncol = max(len(r) for r in rows)
    header = []
    for ci in range(ncol):
        parts = []
        for hr in header_rows:
            v = _clean(hr[ci]) if ci < len(hr) else ""
            if v and v not in parts:
                parts.append(v)
        header.append(" / ".join(parts) if parts else f"col{ci}")
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * ncol) + "|"]
    for r in data_rows:
        cells = [_clean(r[ci]) if ci < len(r) else "" for ci in range(ncol)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pdf_to_markdown(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, 1):
        tables = page.find_tables().tables
        chunks = [f"<!-- page {i} -->"]
        if tables:
            for t in tables:
                md = render_table(t.extract())
                if md:
                    chunks.append(md)
        else:
            txt = "\n".join(ln.rstrip() for ln in (page.get_text("text") or "").splitlines())
            txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
            if txt:
                chunks.append(txt)
        pages.append("\n\n".join(c for c in chunks if c))
    doc.close()
    return "\n\n".join(pages)


def split_pages(md):
    """Return list of page blocks (each starts with <!-- page N -->)."""
    parts = re.split(r"(<!-- page \d+ -->)", md)
    pages, cur = [], ""
    for tok in parts:
        if re.match(r"<!-- page \d+ -->", tok):
            if cur.strip():
                pages.append(cur)
            cur = tok
        else:
            cur += tok
    if cur.strip():
        pages.append(cur)
    return pages


def chunk_pages(md, per):
    """Group `per` page-blocks per chunk."""
    pages = split_pages(md)
    return ["\n\n".join(pages[i:i + per]) for i in range(0, len(pages), per)]


def ensure_markdown(t):
    """Convert PDF -> markdown for target t if not already present."""
    if not t["md_path"] or not t["pdf_path"]:
        return
    import os
    if os.path.exists(t["md_path"]):
        return
    os.makedirs(os.path.dirname(t["md_path"]), exist_ok=True)
    md = pdf_to_markdown(t["pdf_path"])
    with open(t["md_path"], "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[convert] {os.path.basename(t['md_path'])} ({len(md)} chars)", file=sys.stderr, flush=True)
