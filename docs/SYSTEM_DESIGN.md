# TCAS Admission Search — System Design

A system that ingests Thai university admission (TCAS) PDFs, extracts structured
program/criteria data, and serves a **LLM-free** search over it — designed so new
documents can flow in at ~zero marginal cost.

## 1. Goals & constraints
- **LLM-free at query time** (and, via rule-based extraction, ~LLM-free at ingest).
- **Thai-first**: word-segmentation is stage 1 of every input path.
- **New docs keep arriving**: the pipeline must onboard a new university/round with
  minimal effort and no per-doc API spend.
- **Forgiving input**: free-form Thai/English queries, typos tolerated.
- Modest scale: ~1.4k programs now → ~10k at all 89 universities.

## 2. Architecture
```
                INGEST (per document)                         QUERY (per user)
   PDF ──PyMuPDF──▶ markdown ──(empty? OCR.space)──▶ text          typed text
                         │                                       │
                archetype detect (rules)                  1. normalize + tokenize
                         │                                 2. fuzzy parse signals
        ┌────────────────┼────────────────┐                 (university/subjects/seats/kw)
   matrix/index/list   criteria-text      UNKNOWN              │
   table-row parse     Thai regex         LLM fallback   3. filter on combined table
                         │                                       (university ∧ subjects ∧ seats)
                         ▼                                 4. fuzzy rank by name
              components ─▶ component_map (unsupervised)        │
                         │   (bge-m3 + clustering)         5. top-k programs
                         ▼                                       │
        data/search/programs.jsonl  ◀── combined table          ▼
        (one row per program)                                 web UI
                         │
              (optional) topics + MLP classify
```

## 3. Components

| Component | File(s) | Role | LLM? |
|---|---|---|---|
| Convert | `pipeline/convert.py` | PDF→markdown (PyMuPDF tables) | no |
| OCR | `pipeline/ocr.py` | scanned pages → text (OCR.space, `tha`) | no (cheap API) |
| Tokenize | `pipeline/thai_tokenize.py` | normalize + newmm + TCAS domain Trie | no |
| Extract (matrix) | `pipeline/extract_rb.py` | rule-based table-row parse | **no** ✅ |
| Extract (criteria/list/index) | `pipeline/extractors/*` + future RB | per-archetype; RB where possible, LLM fallback | LLM only for UNKNOWN |
| Normalize | `ml/cluster_components.py` | group label synonyms unsupervised | no |
| Combined table | `build_search_table.py` *(todo)* | denormalize → `programs.jsonl` | no |
| Query | `ml/query.py` | fuzzy parse + filter + rank | no |
| Web | `web/app.py` | input → results UI | no |

## 4. Data model — locked feature schema (`data/search/programs.jsonl`)
One row per program. This is the single contract everything reads.

| Column | Type | Example | Used for |
|---|---|---|---|
| `university` | str | "Chulalongkorn" | filter / display |
| `university_id` | str | "001" | id |
| `round` | str | "R3" | filter |
| `tcas_code` | str | "10040101902501A" | id |
| `faculty_th` | str | "คณะวิศวกรรมศาสตร์" | display |
| `program_name_th` | str | "วิศวกรรมคอมพิวเตอร์" | search text |
| `program_name_en` | str | "Computer Engineering" | search text (cross-script) |
| `seats` | int/null | 60 | filter `≥ N` |
| `min_gpax` | float/null | 2.0 | filter |
| `subject_codes` | list[str] | `["61","64","65"]` | **requires-subject filter** (normalized via `component_map`) |
| `topic` | str | "STEM" | grouping / facets |
| `weights` | dict | `{"61":20,"64":20}` | advanced rank (optional) |

**Key decision:** subjects stored as a normalized **`subject_codes` list** (not weight
columns) — sufficient for "requires physics+math" and keeps the table narrow. Weight
columns stay optional (`weights` dict) for future "physics ≥ 20%" filters.

## 5. Data flows
- **Ingest a new doc:** convert → (OCR if needed) → detect archetype → rule-based
  extract → map components via `component_map` (assign new labels to nearest cluster
  centroid) → append rows to `programs.jsonl`.
- **Query:** tokenize → fuzzy-lift signals → filter combined table → fuzzy rank → top-k.

## 6. Storage layout
```
data/
  extracted/<uni>/          source JSONs per university (source of truth)
  text/                     converted markdown
  ml/
    component_map.json      unsupervised synonym→canonical map (+centroids)
    unified_scores.csv      canonical features + topics (ML view)
  search/
    programs.jsonl          ONE combined denormalized table (search index)
```

## 7. Why one combined table (not per-university partition)
- Current/future scale (≤~10k) scans in ms — partitioning gives no speed gain.
- University is already a **filter** in the query → "search university first" pruning
  for free, without N indexes.
- Cross-university queries stay natural. Per-university folders remain source-of-truth.
- Partition only if you later hit ~100k+ or need per-uni deployments (shards by `university` cleanly).

## 8. Cost profile
- Query: **$0** (PyThaiNLP dict + rapidfuzz + stdlib).
- Ingest: **$0** for known archetypes (rule-based); LLM only for UNKNOWN formats (rare);
  OCR.space (cheap) only for scanned PDFs.
- One-time local model: bge-m3 (~2.3 GB) for normalization embeddings — free at inference.

## 9. Build status
- ✅ convert, OCR (OCR.space), tokenize (domain Trie)
- ✅ rule-based **matrix** extractor (99% vs LLM)
- ✅ unsupervised component normalization (210→35 concepts)
- ✅ unify + cluster + MLP (university-from-scores 94%)
- ✅ fuzzy query pipeline + web PoC
- ⏳ rule-based **criteria-text / index / list** extractors
- ⏳ archetype detector + `--no-llm` mode in `pipeline/run.py`
- ⏳ **`build_search_table.py`** → `programs.jsonl` (next)
- ⏳ incremental nearest-centroid assignment in normalizer
