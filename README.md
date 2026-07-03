# TCAS Search PoC

Search Thai university admission (TCAS) criteria, **LLM-free** at query time.

Pipeline: download entrance-condition PDFs → convert/OCR → **rule-based** extract →
normalize → combined search table → fuzzy Thai/English query → web UI.

## Why LLM-free
- **Query path** (the hot path): PyThaiNLP dictionary segmenter + rapidfuzz + stdlib.
  No LLM, no neural net, no API. `$0/query`, runs offline.
- **Extraction**: rule-based parsers (matrix / criteria / index / list) — no LLM.
  (LLM kept only as an optional fallback for unknown prose formats.)

## Repo layout
```
pipeline/            ingest + extract core
  convert.py           PDF -> markdown (PyMuPDF tables)
  ocr.py               scanned PDFs -> text (OCR.space, tha)
  thai_tokenize.py     normalize + word-segment (domain Trie)
  extract_rb.py        rule-based matrix extractor
  rb_all.py            rule-based extract all archetypes + detector
  extractors/          (LLM) extractors — fallback
  run.py               driver
ml/                  search + analytics
  query.py             fuzzy typed-query pipeline (parse signals -> filter -> rank)
  cluster_components.py  unsupervised synonym grouping (bge-m3 + clustering)
  score_topics.py      unify -> cluster -> MLP classify
web/app.py           basic Flask UI (input -> results)
scrape_all.py        download all universities' PDFs (mytcas S3 bucket)
convert_all.py       convert all PDFs -> markdown (+ scanned inventory)
ocr_scanned.py       OCR all empty/scanned markdowns
build_search_table.py  -> data/search/programs.jsonl (the combined table)
docs/                SYSTEM_DESIGN.md, QUERY_PIPELINE.md, LLM_FREE_PIPELINE.md
data/search/programs.jsonl   the combined search table (one row per program)
data/extracted/<uni>/        structured extractions per university
```

## Run the search
```bash
python -m ml.query "วิศวะคอม จุฬา"
python -m ml.query "physics and math, more than 100 seats"
python web/app.py            # -> http://localhost:5000
```

## Rebuild from scratch
```bash
python scrape_all.py                         # download PDFs (all universities)
python convert_all.py                        # PDF -> markdown
OCRSPACE_API_KEY=... python ocr_scanned.py   # OCR scanned docs
python -m pipeline.rb_all                    # rule-based extract
python build_search_table.py                 # -> data/search/programs.jsonl
```

## Coverage note
Rule-based extraction covers **table-based** criteria docs (matrix / index / list /
TCAS-CODE criteria tables). Prose-style announcement PDFs need the optional LLM
fallback. See `docs/SYSTEM_DESIGN.md`.

## Env vars
- `OCRSPACE_API_KEY` — only for OCR of scanned PDFs (cheap).
- `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` — only if using the LLM fallback extractor.
