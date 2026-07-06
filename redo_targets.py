#!/usr/bin/env python3
"""Re-extract junk target files via LLM. Reads /tmp/junk_targets.txt (uid:round:chunk)."""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from pipeline.extractors import criteria  # noqa: E402
from pipeline.rb_all import _uni_name_map  # noqa: E402

names = _uni_name_map()
target_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/junk_targets.txt"
targets = [l.strip() for l in open(target_file) if l.strip()]
print(f"[*] re-extracting {len(targets)} targets", flush=True)
for line in targets:
    uid, rk, chunk = line.split(":")
    criteria.PAGES_PER_CHUNK = int(chunk)        # smaller chunks for 911
    md = os.path.join(ROOT, "data/text", f"{uid}_{rk}.md")
    if not os.path.exists(md) or os.path.getsize(md) < 800:
        for f in glob.glob(os.path.join(ROOT, "data/extracted", f"{uid}_*", f"{uid}_{rk}.json")):
            os.remove(f)
        print(f"[del] {uid} {rk} (no text)", flush=True)
        continue
    folder = os.path.join(ROOT, "data/extracted", f"{uid}_{(names.get(uid,uid)).replace(' ','_')}")
    os.makedirs(folder, exist_ok=True)
    out = os.path.join(folder, f"{uid}_{rk}.json")
    if os.path.exists(out):
        os.remove(out)
    t = {"university_id": uid, "university_name": names.get(uid, uid), "round": rk,
         "round_label": rk, "md_path": md, "out_json": out}
    try:
        criteria.run(t)
    except Exception as e:
        print(f"[ERR] {uid} {rk}: {e}", flush=True)
print("[+] done", flush=True)
