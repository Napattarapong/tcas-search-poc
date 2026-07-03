#!/usr/bin/env python3
"""
Unify all 3 universities' score-bearing programs into one feature table,
group programs by scoring similarity (KMeans -> "score topics"), and train a
deep-learning classifier (sklearn MLPClassifier) to predict a program's score
topic from its weight profile.

Stage 1  unify   : every program -> 21-dim weight vector (GPAX/TGAT/TPAT/A-Level codes)
Stage 2  group   : KMeans on standardized vectors -> K score topics
Stage 3  classify: MLPClassifier predicts topic from the weight vector (train/test)

Outputs: data/ml/unified_scores.csv (the unified table + topic labels).
"""
import glob
import json
import os
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(ROOT, "data", "extracted")
OUT = os.path.join(ROOT, "data", "ml")
os.makedirs(OUT, exist_ok=True)

K = 8  # number of score topics


_MAP_PATH = os.path.join(ROOT, "data", "ml", "component_map.json")
_MAP = json.load(open(_MAP_PATH, encoding="utf-8")) if os.path.exists(_MAP_PATH) else {}


def canon(cat, code, name):
    """Model-derived canonical concept for a label (see normalize_components.py)."""
    return _MAP.get(f"{cat}||{code}||{name}", "other")


SUBJ_NAME = {
    "61": "Math1", "62": "Math2", "63": "Stat", "64": "Physics", "65": "Chem",
    "66": "Bio", "70": "Social", "81": "Thai", "82": "English", "83": "French",
    "84": "German", "85": "Japanese", "86": "Korean", "87": "Chinese",
    "88": "Pali", "89": "Spanish",
}


def feat_vec(p, feats):
    fv = {}
    for c in (p.get("weighted_components") or p.get("weighted_subjects") or []):
        cat = (c.get("category") or "").strip()
        code = (c.get("test_or_subject_code") or c.get("subject_code") or "").strip()
        name = (c.get("subject_name_th") or "").strip()
        w = c.get("weight_percent")
        try:
            w = float(w)
        except (TypeError, ValueError):
            continue
        k = canon(cat, code, name)
        if k and k != "other":
            fv[k] = fv.get(k, 0.0) + w  # sum weights that map to the same concept
    return [fv.get(f, 0.0) for f in feats]


def main():
    UID_NAME = {"001": "Chulalongkorn", "004": "Chiang Mai", "005": "Thammasat"}
    MIN_DF = 5  # drop canonicals appearing in <5 programs (unsupervised noise pruning)
    # pass 1: load programs + discover canonical features (model-derived)
    progs = []
    feats_seen = Counter()
    for f in sorted(glob.glob(os.path.join(EXT, "**", "*.json"), recursive=True)):
        d = json.load(open(f, encoding="utf-8"))
        uid = os.path.basename(f)[:3]
        uni = UID_NAME.get(uid, "?")
        rnd = d.get("round") or os.path.basename(f)[4:6]
        for p in d.get("programs", []):
            pcanon = set()
            for c in (p.get("weighted_components") or p.get("weighted_subjects") or []):
                k = canon((c.get("category") or "").strip(),
                          (c.get("test_or_subject_code") or c.get("subject_code") or "").strip(),
                          (c.get("subject_name_th") or "").strip())
                if k and k != "other":
                    pcanon.add(k)
            for k in pcanon:
                feats_seen[k] += 1
            progs.append((uni, rnd, p))

    def order(k):
        if k == "GPAX":
            return (0, k)
        if k.startswith("TGAT"):
            return (1, k)
        if k.startswith("TPAT"):
            return (2, k)
        if k.isdigit():
            return (3, int(k))
        return (4, k)
    feats = sorted([k for k, df in feats_seen.items() if df >= MIN_DF], key=order)

    # pass 2: build feature vectors
    rows, X = [], []
    for uni, rnd, p in progs:
        v = feat_vec(p, feats)
        if sum(v) == 0:
            continue  # no score data
        X.append(v)
        rows.append({
            "university": uni, "round": rnd,
            "faculty": p.get("faculty_major_th") or p.get("faculty_th") or "",
            "program": p.get("faculty_major_th") or p.get("program_name_th") or "",
            "seats": p.get("seats"),
        })
    X = np.array(X, dtype=float)
    print(f"[*] unified dataset: {X.shape[0]} programs x {X.shape[1]} features "
          f"(canonical concepts: {feats})")

    # --- Stage 2: cluster into score topics ---
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(Xs)
    topics = km.labels_

    # characterize each topic by its heaviest features
    topic_label = {}
    print(f"\n=== {K} score topics (by dominant weights) ===")
    for t in range(K):
        mask = topics == t
        mean = X[mask].mean(axis=0)
        top = sorted(zip(feats, mean), key=lambda kv: -kv[1])[:4]
        top = [(f, round(w)) for f, w in top if w > 1]
        names = [SUBJ_NAME.get(f, f) for f, _ in top]
        label = " + ".join(names) if names else "(misc)"
        topic_label[t] = label
        print(f"  topic {t} ({mask.sum():3d} programs): {label}")

    # --- Stage 3: MLP classifier topic <- weight vector ---
    Xtr, Xte, ytr, yte = train_test_split(Xs, topics, test_size=0.2, random_state=0)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=0)
    mlp.fit(Xtr, ytr)
    print(f"\n=== MLP classifier (deep learning) ===")
    print(f"  train acc: {mlp.score(Xtr, ytr):.3f}")
    print(f"  test  acc: {mlp.score(Xte, yte):.3f}")
    print(classification_report(yte, mlp.predict(Xte), zero_division=0, digits=2))

    # --- meaningful task: predict UNIVERSITY from the score profile ---
    # (not circular: the label is independent of the weight features)
    unis = sorted({r["university"] for r in rows})
    uni_y = np.array([unis.index(r["university"]) for r in rows])
    Xtr, Xte, ytr, yte = train_test_split(Xs, uni_y, test_size=0.2, random_state=0, stratify=uni_y)
    mlp_u = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=0).fit(Xtr, ytr)
    print(f"\n=== MLP: predict UNIVERSITY from score profile (real task) ===")
    print(f"  test acc: {mlp_u.score(Xte, yte):.3f}")
    print(classification_report(yte, mlp_u.predict(Xte), target_names=unis, zero_division=0, digits=2))

    # --- save unified table ---
    import csv
    out_csv = os.path.join(OUT, "unified_scores.csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(["university", "round", "faculty", "program", "seats", "topic_id", "topic"]
                   + feats)
        for r, v, tid in zip(rows, X, topics):
            w.writerow([r["university"], r["round"], r["faculty"], r["program"], r["seats"],
                        int(tid), topic_label[int(tid)]] + [round(x, 1) for x in v])
    print(f"\n[+] unified table -> {out_csv}")


if __name__ == "__main__":
    main()
