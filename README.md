# Thai University Q&A (Hybrid: NL→SQL + Vector Search)

A Thai-language Q&A chatbot over Thai university admission data. Combines NL→SQL for structured fields (seats, dates, costs) with vector search for free-text requirements (scholarships, eligibility). Every cited sentence points to a verifiable DB row or text chunk.

## Architecture

```
Thai question → Router (keywords)
   ├─ structured → NL→SQL → rows ─┐
   └─ free       → bge-m3 + FAISS → chunks ─┤
                                          ↓
                          Composer → Thai prose + citations
                                          ↓
                                Validator (drops uncited)
                                          ↓
                                   Streamlit UI
```

See `docs/superpowers/specs/2026-06-28-thai-uni-qa-hybrid-design.md` for the full design.

## Setup

```bash
# Use the existing venv at C:/Users/Napattarapong/.venv
C:/Users/Napattarapong/.venv/Scripts/pip install -e ".[dev]"
cp .env.example .env
# Edit .env and set LLM_API_KEY
```

## Ingest

```bash
# 1) Pull TCAS JSON for our 3 universities
python -m src.ingest tcas

# 2) Ingest a per-university admission PDF (also builds the FAISS index)
python -m src.ingest pdf "C:/Users/Napattarapong/Downloads/some-announcement.pdf" --year 2569
```

## Run

```bash
streamlit run app.py
```

## Test

```bash
pytest -v
```

## Demo Questions

| # | Question | Path | Citation |
|---|---|---|---|
| 1 | วิศวะจุฬารอบ 1 ปี 2569 รับกี่คน | structured | row |
| 2 | ค่าเทอมคณะวิทยาศาสตร์ มหิดลเท่าไหร่ | structured | row |
| 3 | ทุนเรียนดีของมหิดลมีอะไรบ้าง | free | chunk |
| 4 | คุณสมบัติผู้สมัครวิศวะจุฬารอบ 1 | free | chunk |
| 5 | อาจารย์ที่ปรึกษาดีๆ ของจุฬามีใครบ้าง | structured | (ไม่พบข้อมูล) |

## Files

- `src/llm.py` — OpenAI-compatible HTTP client
- `src/db.py` — 7-table SQLite schema
- `src/router.py` — keyword-based path selector
- `src/nl_to_sql.py` — Thai question → safe SQL
- `src/chunking.py` — Thai sentence splitter + paragraph chunker
- `src/vector_search.py` — bge-m3 embedder + FAISS index
- `src/compose.py` — rows + chunks → cited prose
- `src/validator.py` — citation regex
- `src/ingest.py` — CLI: `tcas` or `pdf`
- `app.py` — Streamlit UI