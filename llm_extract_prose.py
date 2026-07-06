#!/usr/bin/env python3
"""One-time LLM extraction for prose-format docs (where rule-based gave 0 records).
Reuses the criteria extractor (flat schema + 8-page chunks + glm-5.1).

Usage: python llm_extract_prose.py [start:end]   # slice of the prose list (for sharding)
"""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from pipeline.rb_all import best_parse, _uni_name_map  # noqa: E402
from pipeline.extractors import criteria  # noqa: E402


def main():
    names = _uni_name_map()
    prose = []
    for md in sorted(glob.glob(os.path.join(ROOT, "data/text/[0-9]*.md"))):
        txt = open(md, encoding="utf-8").read()
        if len(txt) < 800:
            continue
        _, recs, _ = best_parse(txt)
        if not recs:
            prose.append(md)
    start, end = (0, len(prose))
    if len(sys.argv) > 1:
        start, end = map(int, sys.argv[1].split(":"))
    print(f"[*] shard {start}:{end} of {len(prose)} prose docs", flush=True)
    for md in prose[start:end]:
        base = os.path.basename(md)[:-3]
        uid, rk = base[:3], base[4:6]
        folder = os.path.join(ROOT, "data/extracted", f"{uid}_{(names.get(uid,uid)).replace(' ','_')}")
        os.makedirs(folder, exist_ok=True)
        t = {"university_id": uid, "university_name": names.get(uid, uid), "round": rk,
             "round_label": rk, "md_path": md, "out_json": os.path.join(folder, base + ".json")}
        try:
            criteria.run(t)
        except Exception as e:
            print(f"[ERR] {base}: {type(e).__name__}: {e}", flush=True)
    print(f"[+] DONE shard {start}:{end}", flush=True)


if __name__ == "__main__":
    main()
