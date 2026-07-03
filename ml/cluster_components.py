#!/usr/bin/env python3
"""
Group admission-score component labels UNSUPERVISED (no LLM prompting):
embed every distinct label text with a multilingual encoder (bge-m3), then
agglomerative-cluster by cosine similarity. Labels that mean the same thing
(GPAX / เกรดเฉลี่ย / GPA) land in one cluster = one canonical concept.

Writes data/ml/component_map.json (label_key -> cluster representative) so
score_topics.py uses the unsupervised canonicals.

Usage: python cluster_components.py [distance_threshold]   (default 0.40)
"""
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(ROOT, "data", "extracted")
OUT = os.path.join(ROOT, "data", "ml", "component_map.json")
LLM_MAP = os.path.join(ROOT, "data", "ml", "component_map_llm.json")  # backup of LLM map


def text_of(cat, code, name):
    t = " ".join(p for p in [code, name] if p).strip()
    return t or cat or "other"


def collect_labels():
    labels = Counter()
    for f in glob.glob(os.path.join(EXT, "**", "*.json"), recursive=True):
        d = json.load(open(f, encoding="utf-8"))
        for p in d.get("programs", []):
            for c in (p.get("weighted_components") or p.get("weighted_subjects") or []):
                cat = (c.get("category") or "").strip()
                code = (c.get("test_or_subject_code") or c.get("subject_code") or "").strip()
                name = (c.get("subject_name_th") or "").strip()
                labels[(cat, code, name)] += 1
    return labels


def main():
    thresh = float(sys.argv[1]) if len(sys.argv) > 1 else 0.40
    labels = collect_labels()
    keys = list(labels.keys())
    texts = [text_of(*k) for k in keys]
    print(f"[*] {len(keys)} distinct labels -> embedding with bge-m3", file=sys.stderr)

    model = SentenceTransformer("BAAI/bge-m3")
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    cl = AgglomerativeClustering(n_clusters=None, metric="cosine",
                                 linkage="average", distance_threshold=thresh)
    ids = cl.fit_predict(emb)
    n_clusters = ids.max() + 1
    print(f"[*] threshold={thresh} -> {n_clusters} clusters", file=sys.stderr)

    # representative for each cluster = text of its most frequent label
    rep = {}
    for cid in range(n_clusters):
        members = [(labels[keys[i]], texts[i]) for i in range(len(keys)) if ids[i] == cid]
        rep[cid] = max(members)[1]
    # de-duplicate representatives that collide (rare) by appending cluster id
    seen = {}
    for cid in range(n_clusters):
        r = rep[cid]
        rep[cid] = r if r not in seen else f"{r}#{cid}"
        seen[rep[cid]] = cid

    # build label -> canonical map + report
    by_cluster = {}
    mapping = {}
    for i, k in enumerate(keys):
        cid = int(ids[i])
        mapping[f"{k[0]}||{k[1]}||{k[2]}"] = rep[cid]
        by_cluster.setdefault(cid, []).append((labels[k], texts[i]))

    print(f"\n=== {n_clusters} unsupervised clusters (representative : members) ===", file=sys.stderr)
    for cid in sorted(by_cluster, key=lambda c: -len(by_cluster[c])):
        members = sorted(by_cluster[cid], reverse=True)
        ex = ", ".join(t for _, t in members[:4])
        print(f"  [{rep[cid][:28]:28s}] ({len(members):3d}): {ex[:60]}", file=sys.stderr)

    # back up existing (LLM) map, then write unsupervised map
    if os.path.exists(OUT) and not os.path.exists(LLM_MAP):
        os.replace(OUT, LLM_MAP)
    json.dump(mapping, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[+] wrote unsupervised map -> {OUT}", file=sys.stderr)
    if os.path.exists(LLM_MAP):
        print(f"[+] (LLM map backed up at {LLM_MAP})", file=sys.stderr)


if __name__ == "__main__":
    main()
