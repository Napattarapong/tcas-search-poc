#!/usr/bin/env python3
"""Re-extract the 'not good' Round 1 files via LLM (criteria extractor):
fixes index docs missing program names and replaces junk list/section-header
extractions with real programs (or 0 for pure announcements)."""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from pipeline.extractors import criteria  # noqa: E402
from pipeline.rb_all import _uni_name_map  # noqa: E402

names = _uni_name_map()
targets = [l.strip() for l in open("/tmp/r1_redo.txt") if l.strip()]
print(f"[*] re-extracting {len(targets)} R1 docs via LLM", flush=True)
for t in targets:
    uid, rk = t.split(":")
    md = os.path.join(ROOT, "data/text", f"{uid}_{rk}.md")
    if not os.path.exists(md) or os.path.getsize(md) < 800:
        # delete bad json so it doesn't pollute; nothing to re-extract
        for f in glob.glob(os.path.join(ROOT, "data/extracted", f"{uid}_*", f"{uid}_{rk}.json")):
            os.remove(f)
        print(f"[del] {t} (no text)", flush=True)
        continue
    folder = os.path.join(ROOT, "data/extracted", f"{uid}_{(names.get(uid,uid)).replace(' ','_')}")
    os.makedirs(folder, exist_ok=True)
    out = os.path.join(folder, f"{uid}_{rk}.json")
    if os.path.exists(out):
        os.remove(out)
    T = {"university_id": uid, "university_name": names.get(uid, uid), "round": rk,
         "round_label": rk, "md_path": md, "out_json": out}
    try:
        criteria.run(T)
    except Exception as e:
        print(f"[ERR] {t}: {e}", flush=True)
print("[+] done", flush=True)
