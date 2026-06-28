"""Inspect the downloaded TCAS universities.json."""
import gzip
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else r"C:\tmp\tcas_universities.json"
with open(path, "rb") as f:
    data = json.loads(gzip.decompress(f.read()))

print(f"count: {len(data)}")
if data:
    print("--- first record ---")
    print(json.dumps(data[0], ensure_ascii=False, indent=2)[:800])

print("--- our 3 IDs (001, 004, 006) ---")
for u in data:
    uid = u.get("university_id", "")
    if uid in ("001", "004", "006"):
        print(f"  {uid}: {u.get('university_name_th')} ({u.get('university_name_en')})")
