# LLM-Free Pipeline Roadmap

**Goal:** process incoming Thai university admission PDFs with ~zero marginal LLM cost,
so new docs can flow in without API spend. LLM kept only as a fallback for unknown formats.

## Principle: only EXTRACTION used the LLM
Every other stage is already free:

| Stage | Tool | Cost |
|---|---|---|
| PDF → markdown | PyMuPDF (tables) | free |
| OCR (scanned) | OCR.space | cheap (per page) |
| **Extraction** (text → structured) | **rule-based parsers (this roadmap)** | **free** |
| Component normalization | bge-m3 + clustering (local) | free |
| Topics + classifier | sklearn (local) | free |

## Architecture (incoming doc)
```
PDF ──PyMuPDF──▶ markdown ──(empty? OCR.space)──▶ text
                                                   │
                            rule-based archetype detector
                          (matrix │ index │ criteria │ list │ UNKNOWN)
                                  │
            ┌─────────────────────┼─────────────────────┐
       matrix/index/list      criteria-text          UNKNOWN
       table-row parser       Thai regex parser     → LLM fallback (rare)
                                  │
                          components ──bge-m3──▶ nearest cluster centroid
                                  │
                         unify → topics → MLP classify
```

## Rule-based extractors (one per archetype)
Each maps the markdown to the same `weighted_components` schema the LLM produced,
so downstream code is unchanged.

| Archetype | Example | Method | Status |
|---|---|---|---|
| **matrix** | Chula R3 | fixed-column table-row parse | ✅ **done — 138/138 vs LLM, 99% seats** |
| **index** | Thammasat R1 | table-row parse (code\|name\|seats) | todo (easy) |
| **list** | Chula R1/R2 | numbered-item regex (`^\d+\.`) | todo (easy) |
| **criteria-text** | CMU R1-4, Thammasat R3 | regex: `TCAS CODE`, `จำนวนรับ`, `ค่าน้ำหนักร้อยละ X`, `รหัสวิชา \d+` | todo (hardest, biggest $ saver) |
| UNKNOWN | new format | LLM fallback | reuse existing extractors |

## Incremental normalization (new labels, no LLM, no re-train)
- Keep cluster **centroids** from `cluster_components.py`.
- A new label → bge-m3 embed → assign to nearest centroid (cosine).
- Periodically (e.g. quarterly) re-run full clustering to refresh centroids.
- No per-doc LLM call.

## Build order
1. ✅ matrix extractor + validation harness (`pipeline/extract_rb.py`)
2. index + list extractors (quick wins, table/regex)
3. criteria-text regex extractor (bulk of CMU; validate vs existing LLM JSON)
4. archetype detector (rule-based router)
5. wire into `pipeline/run.py` with `--no-llm` flag (LLM only when detector returns UNKNOWN)
6. incremental centroid assignment in the normalizer

## Trade-off (explicit)
Rule-based parsers are **format-specific**: a genuinely new layout needs a new parser
(or falls back to the LLM once). For the finite TCAS format set this is cheap to maintain,
and the payoff is per-doc extraction at $0.
