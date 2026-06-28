"""Streamlit chat UI for Thai university grounded RAG (hybrid pipeline)."""
from __future__ import annotations
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # picks up .env at repo root if present

import streamlit as st

from src.db import init_db, get_conn
from src.nl_to_sql import generate_sql, run_sql
from src.router import route
from src.compose import compose_prose

DB_PATH = os.environ.get("DB_PATH", "data/university.db")

st.set_page_config(page_title="ถามเรื่องมหาวิทยาลัย", page_icon="🎓")
st.title("🎓 ถามเรื่องมหาวิทยาลัย")
st.caption("คำตอบอ้างอิงจากฐานข้อมูลมหาวิทยาลัย 3 แห่ง (จุฬาฯ, มหิดล, เชียงใหม่) — ไม่มีการเดาคำตอบจากความรู้ภายนอก")


@st.cache_resource
def _setup_db():
    init_db(DB_PATH)
    return DB_PATH


def _render_sidebar():
    with st.sidebar:
        st.header("สถานะฐานข้อมูล")
        if not Path(DB_PATH).exists():
            st.warning(f"ยังไม่มีฐานข้อมูลที่ {DB_PATH} — กรุณารัน `python -m src.ingest tcas` ก่อน")
            return
        with get_conn(DB_PATH, read_only=True) as conn:
            counts = {
                "มหาวิทยาลัย": conn.execute("SELECT COUNT(*) FROM universities").fetchone()[0],
                "หลักสูตร": conn.execute("SELECT COUNT(*) FROM programs").fetchone()[0],
                "รอบรับสมัคร": conn.execute("SELECT COUNT(*) FROM admission_rounds").fetchone()[0],
                "คะแนนขั้นต่ำ": conn.execute("SELECT COUNT(*) FROM cutoff_scores").fetchone()[0],
                "คุณสมบัติ/เอกสาร": conn.execute("SELECT COUNT(*) FROM requirements").fetchone()[0],
            }
            unis = conn.execute(
                "SELECT name_th FROM universities ORDER BY name_th"
            ).fetchall()
        for k, v in counts.items():
            st.write(f"- **{k}:** {v}")
        st.write("---")
        st.write("**มหาวิทยาลัยในระบบ:**")
        for (name,) in unis:
            st.write(f"- {name}")


def _handle_question(question: str) -> dict | None:
    """Run hybrid pipeline. Returns rendered content dict or None."""
    db = _setup_db()
    if not Path(db).exists():
        st.error("ไม่พบฐานข้อมูล")
        return None

    path = route(question)
    rows = None
    cols = None
    sql = None
    intent = None
    chunks = None

    try:
        if path == "structured":
            with st.status("กำลังค้นหาในฐานข้อมูล..."):
                sql, intent = generate_sql(question, db)
                cols, rs = run_sql(sql, db)
                rows = [dict(zip(cols, r)) for r in rs] if rs else None
                # Attach table name to each row so composer can cite it
                if rows and sql:
                    table_match = re.search(r"FROM\s+(\w+)", sql, re.IGNORECASE)
                    tbl = table_match.group(1) if table_match else "rows"
                    # Multi-table: composer uses 'rows' generic label
                    if len(set(re.findall(r"FROM\s+(\w+)", sql, re.IGNORECASE))) > 1:
                        tbl = "rows"
                    for r in rows:
                        r.setdefault("table", tbl)
            # Fallback to vector search if SQL returned 0 rows
            if not rows:
                with st.status("ไม่พบในฐานข้อมูล ลองค้นหาในข้อความ..."):
                    chunks = _vector_search(question)
        else:
            with st.status("กำลังค้นหาในข้อความต้นฉบับ..."):
                chunks = _vector_search(question)
    except Exception as e:
        st.error(f"การค้นหาล้มเหลว: {e}")
        return None

    if not rows and not chunks:
        st.warning("ไม่พบข้อมูลในฐานข้อมูล")
        return None

    try:
        with st.status("กำลังเรียบเรียงคำตอบ...", state="complete"):
            prose, sources = compose_prose(question, rows=rows, chunks=chunks)
    except Exception as e:
        st.error(f"ไม่สามารถเรียบเรียงคำตอบได้: {e}")
        return {"prose": None, "rows": rows, "chunks": chunks, "sql": sql, "intent": intent,
                "path": path, "sources": []}

    return {"prose": prose, "rows": rows, "chunks": chunks, "sql": sql, "intent": intent,
            "path": path, "sources": sources}


@st.cache_resource(show_spinner="กำลังโหลดโมเดลฝัง...")
def _vector_search(question: str) -> list[dict]:
    """Embed the question and search the FAISS index.

    Streamlit caches the embedder + index across reruns, so they're built
    once and reused. Returns search results directly.
    """
    from src.vector_search import BgeM3Embedder, build_index, search
    from src.db import init_db, get_conn

    init_db(DB_PATH)
    embedder = BgeM3Embedder(cache_dir="data/models")
    with get_conn(DB_PATH, read_only=True) as conn:
        rows = conn.execute(
            "SELECT id, source_document_id, text FROM chunks ORDER BY id"
        ).fetchall()
    chunk_dicts = [{"id": r[0], "source_document_id": r[1], "text": r[2]} for r in rows]
    if not chunk_dicts:
        return []
    index = build_index(chunk_dicts, embedder=embedder)
    return search(index, question, embedder=embedder, k=5, threshold=0.3)


def _render_rendered(rendered: dict) -> None:
    path = rendered.get("path", "?")
    path_label = {
        "structured": "จากฐานข้อมูล",
        "free": "จากข้อความต้นฉบับ",
    }.get(path, path)
    st.info(f"เส้นทาง: {path_label}")

    prose = rendered.get("prose")
    if prose:
        st.markdown(prose)
    elif prose is None and (rendered.get("rows") or rendered.get("chunks")):
        st.info("ไม่สามารถสร้างคำตอบแบบอธิบายได้ — กรุณาดูข้อมูลด้านล่าง")

    if rendered.get("rows"):
        with st.expander("ตารางข้อมูลดิบ"):
            st.dataframe(rendered["rows"], hide_index=True)
    if rendered.get("chunks"):
        with st.expander("ข้อความต้นฉบับที่ค้นพบ"):
            for c in rendered["chunks"]:
                st.write(f"- **[chunk#{c['chunk_id']}]** (score={c['score']:.2f}): {c['text']}")
    if rendered.get("sql"):
        with st.expander("SQL ที่ใช้"):
            st.code(rendered["sql"], language="sql")
    if rendered.get("sources"):
        with st.expander("แหล่งอ้างอิง"):
            for s in rendered["sources"]:
                if s["type"] == "row":
                    st.write(f"- row `{s['table']}#{s['id']}`")
                else:
                    st.write(f"- chunk `#{s['chunk_id']}` (source_doc #{s['source_document_id']})")


_render_sidebar()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            if "rendered" in msg:
                _render_rendered(msg["rendered"])
            else:
                rendered = _handle_question(msg["content"])
                if rendered is not None:
                    msg["rendered"] = rendered

prompt = st.chat_input("ลองถาม เช่น: วิศวะจุฬารอบ 1 ปี 2569 รับกี่คน")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        rendered = _handle_question(prompt)
        if rendered is not None:
            _render_rendered(rendered)
            st.session_state.messages.append({"role": "assistant", "content": prompt, "rendered": rendered})
        else:
            st.session_state.messages.append({"role": "assistant", "content": prompt, "rendered": None})
