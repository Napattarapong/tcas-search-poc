# 🎓 Thai University QA - Hybrid Search System

ระบบตอบคำถามเกี่ยวกับมหาวิทยาลัยไทย (TCAS) ด้วย RAG แบบ Hybrid ที่ผสมผสานระหว่าง SQL Database และ Vector Search

## Overview

ระบบนี้ตอบคำถามเกี่ยวกับหลักสูตร การรับสมัคร และคะแนนของมหาวิทยาลัยชั้นนำ 3 แห่งในประเทศไทย:
- จุฬาลงกรณ์มหาวิทยาลัย (CU)
- มหาวิทยาลัยมหิดล (MU)
- มหาวิทยาลัยเชียงใหม่ (CMU)

### Key Features

- **Hybrid Pipeline**: เลือกเส้นทางอัตโนมัติระหว่าง SQL query และ vector search
- **Natural Language to SQL**: ใช้ LLM แปลงคำถามภาษาธรรมชาติเป็น SQL query
- **Thai NLP**: ประมวลผลภาษาไทยด้วย PyThaiNLP
- **Grounded RAG**: คำตอบอ้างอิงจากข้อมูลจริงในฐานข้อมูล ไม่ hallucinate
- **Streamlit UI**: อินเตอร์เฟซสนทนาภาษาไทย

## Architecture

```
┌─────────────────┐
│   User Question  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Router      │ ← Classifies: structured vs free
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│ SQL   │ │Vector │ → NL-to-SQL  → FAISS Search
│ Path  │ │ Path │
└───┬───┘ └───┬───┘
    │         │
    └────┬────┘
         ▼
┌─────────────────┐
│    Composer     │ ← LLM generates prose + citations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Final Answer  │
└─────────────────┘
```

## Project Structure

```
thai-uni-qa-hybrid/
├── app.py                 # Streamlit chat UI
├── src/
│   ├── db.py             # SQLite database operations
│   ├── router.py         # Intent classification (structured vs free)
│   ├── nl_to_sql.py     # Natural language to SQL translation
│   ├── vector_search.py # FAISS vector search with BGE-M3
│   ├── compose.py      # LLM-powered answer generation
│   ├── chunking.py     # Document chunking strategies
│   ├── ingest.py      # TCAS PDF ingestion pipeline
│   ├── llm.py        # LLM API client
│   └── validator.py   # SQL query validation
├── scripts/
│   ├── download_tcas_pdfs.py
│   ├── convert_pdfs_to_markdown.py
│   ├── structure_markdowns.py
│   └── ...
├── tests/
│   ├── test_router.py
│   ├── test_nl_to_sql_safety.py
│   ├── test_vector_search.py
│   └── ...
└── pyproject.toml
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Napattarapong/tcas-search-poc.git
cd tcas-search-poc

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Edit `.env` file:

```env
# Required: LLM API key (OpenAI compatible)
OPENAI_API_KEY=sk-...

# Optional
DB_PATH=data/university.db
MODEL_CACHE_DIR=data/models
```

## Usage

### 1. Ingest TCAS Data (first time only)

```bash
python -m src.ingest tcas
```

This will:
- Download TCAS PDFs from university websites
- Convert PDFs to markdown
- Create SQLite database with tables
- Build FAISS index for vector search

### 2. Run the Chat UI

```bash
streamlit run app.py
```

### 3. Ask Questions

Example questions:
- "วิศวะจุฬารอบ 1 ปี 2569 รับกี่คน"
- "คะแนนขั้นต่ำ อักษร ศาสตร์ มหิดล ปี 2568"
- "มีคณะอะไรบ้างที่รับ Portfolio"

## Database Schema

```sql
universities      -- มหาวิทยาลัย
programs       -- หลักสูตร/คณะ
admission_rounds -- รอบการรับสมัคร
cutoff_scores   -- คะแนนขั้นต่ำ
requirements   -- คุณสมบัติ/เอกสารที่ต้องยื่น
chunks         -- ข้อความต้นฉบับ (สำหรับ vector search)
source_documents -- เอกสารต้นฉบับ
```

## Testing

```bash
pytest tests/
```

## Demo Questions

| # | Question | Path | Citation |
|---|---|---|---|
| 1 | วิศวะจุฬารอบ 1 ปี 2569 รับกี่คน | structured | row |
| 2 | ค่าเทอมคณะวิทยาศาสตร์ มหิดลเท่าไหร่ | structured | row |
| 3 | ทุนเรียนดีของมหิดลมีอะไรบ้าง | free | chunk |
| 4 | คุณสมบัติผู้สมัครวิศวะจุฬารอบ 1 | free | chunk |
| 5 | อาจารย์ที่ปรึกษาดีๆ ของจุฬามีใครบ้าง | structured | (ไม่พบข้อมูล) |

## Tech Stack

- **Database**: SQLite with FAISS
- **Embedder**: BGE-M3 (multilingual, Thai-capable)
- **LLM**: OpenAI API (or compatible)
- **Thai NLP**: PyThaiNLP
- **UI**: Streamlit
- **PDF**: markitdown

## License

MIT License

## Author

Napattarapong Chen