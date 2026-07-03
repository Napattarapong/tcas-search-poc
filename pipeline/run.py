"""Driver: convert PDF -> markdown, route by archetype, extract -> JSON.

Run with the hyperextract python (has fitz + hyperextract + langchain-anthropic):
    HE_PY=~/.local/share/uv/tools/hyperextract/bin/python
    ANTHROPIC_API_KEY=... ANTHROPIC_BASE_URL=... $HE_PY -m pipeline.run [opts]

Options:
    --only uid:ROUND,uid:ROUND   restrict to specific files (e.g. 001:R1,001:R2)
    --force                       re-extract even if output JSON exists
    --no-convert                  skip PDF->md (assume markdown already present)
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline import config as C  # noqa: E402
from pipeline.convert import ensure_markdown  # noqa: E402
from pipeline.extractors import list as list_ex, matrix as matrix_ex, criteria as criteria_ex  # noqa: E402

EXTRACTORS = {"list": list_ex, "matrix": matrix_ex, "criteria": criteria_ex}


def parse_only(arg):
    if not arg:
        return None
    out = set()
    for part in arg.split(","):
        uid, rk = part.split(":")
        out.add((uid.strip(), rk.strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-convert", action="store_true")
    args = ap.parse_args()
    only = parse_only(args.only)

    for t in C.targets(only=only):
        uid, rk, arch = t["university_id"], t["round"], t["archetype"]
        if arch not in EXTRACTORS:
            print(f"[skip] {uid} {rk}: archetype '{arch}' not implemented", file=sys.stderr, flush=True)
            continue
        if os.path.exists(t["out_json"]) and not args.force:
            print(f"[exists] {uid} {rk} -> {os.path.basename(t['out_json'])}", file=sys.stderr, flush=True)
            continue
        if not args.no_convert:
            ensure_markdown(t)
        if not os.path.exists(t["md_path"]):
            print(f"[no-md] {uid} {rk}: {t['md_path']} missing", file=sys.stderr, flush=True)
            continue
        os.makedirs(t.get("out_folder", os.path.dirname(t["out_json"])), exist_ok=True)
        print(f"=== {uid} {rk} ({arch}) ===", file=sys.stderr, flush=True)
        EXTRACTORS[arch].run(t)


if __name__ == "__main__":
    main()
