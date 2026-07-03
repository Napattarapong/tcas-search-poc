# Query Pipeline (typed free-form → ranked programs)

**100% LLM-free and model-free.** No generative LLM, no neural net, no API.
Stack = Python stdlib + PyThaiNLP dictionary segmenter + rapidfuzz (classical
string similarity). Runs offline, deterministic, $0/query.

## Stages
```
typed text (Thai/English, any spelling)
  │
  1. normalize        NFKC + collapse whitespace
  │
  2. tokenize         PyThaiNLP newmm + TCAS domain Trie   (pipeline/thai_tokenize.py)
  │
  3. parse signals    fuzzy-lift (rapidfuzz WRatio):
        university    จุฬา/chula, เชียงใหม่/cmu, ธรรมศาสตร์/thammasat
        subjects      ฟิสิกส์/physics→64, เคมี/chem→65, คณิต/math→61, … (incl. typos)
        min seats     "more than N" / "มากกว่า N" / ">N"
        keywords      leftover meaningful tokens
  │
  4. filter           university  ∧  required-subjects ⊆ program codes  ∧  seats ≥ min
  │
  5. rank             per-keyword max(partial_ratio, token_set_ratio) vs
                      program+faculty+English name  →  average, seats tiebreak
  │
  6. return           top-k program records (university, program, seats, codes)
```

## Files
- `ml/query.py` — the pipeline: `parse_signals(text)`, `search(text, k=8)`, CLI.
- `pipeline/thai_tokenize.py` — stage 1–2 (normalize + domain-augmented tokenize).
- `data/extracted/<uni>/*.json` — the program dataset it searches.

## Run
```bash
python -m ml.query "วิศวะคอม จุฬา"
python -m ml.query "compter enginering at chula"          # typos OK
python -m ml.query "phisics and math, more than 100 seats"
```

## Why no-LLM works here
- **Tokenize**: PyThaiNLP `newmm` is a *dictionary maximal-matching algorithm*, not a
  neural net — just a wordlist + rules.
- **Parse signals**: regex + dictionary lookups + rapidfuzz (edit/substring similarity).
- **Rank**: rapidfuzz ratios — pure string math.
- No `langchain` / `anthropic` / `torch` / `openai` anywhere in the path (grep-verified).

## Honest limitations
- **Cross-script matching** (English query → Thai-only program name) works only when an
  English name is stored (`major_name_en`). Run the enricher once to add English names;
  otherwise use Thai queries for those programs.
- **No semantics**: "AI" won't map to "computer" unless it's a substring/synonym in the
  dictionaries. (If you later want semantics, that's the one place a model — bge-m3 —
  would help; it stays optional and out of the no-LLM core.)
- Tunable: `FUZZ_THR` (synonym cutoff) and `K` (top-k) at the top of `ml/query.py`.

## Extending the dictionaries (grows with the corpus)
- `UNI` / `SUBJ` synonym tables in `ml/query.py`.
- The tokenizer's domain Trie auto-seeds from `data/ml/component_map*.json`, so new
  subject labels improve tokenization automatically.
