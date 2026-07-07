#!/usr/bin/env python3
"""
My Fit — given a student's scores, rank programs by how each program's
weighting favors the student's strengths.

fit = weighted average of the student's scores using the program's weights
      (over the subjects the student has). High fit = the program weights
      subjects where the student is strong.

Usage:
    python -m ml.myfit 61:80 64:75 65:70 82:65 GPAX:3.8 TGAT:75
    (codes are A-Level 61-89 / GPAX / TGAT / TPAT3 ; scores are 0-100, GPAX 0-4)
"""
import os
import sqlite3
import sys
from collections import defaultdict

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "data", "search", "tcas.db")


def parse_scores(args):
    """'61:80 GPAX:3.8' -> {'61':80.0, 'GPAX':95.0}  (GPAX 0-4 -> 0-100)"""
    scores = {}
    for a in args:
        if ":" not in a:
            continue
        code, val = a.split(":", 1)
        try:
            v = float(val)
        except ValueError:
            continue
        scores[code.strip()] = v * 25 if code.strip() == "GPAX" else v
    return scores


def myfit(scores, limit=15):
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT p.id,p.program_name_th,p.seats,p.round,u.name_en,"
        "ps.subject_code,ps.weight_percent "
        "FROM program p JOIN university u ON u.id=p.university_id "
        "JOIN program_subject ps ON ps.program_id=p.id "
        "WHERE ps.weight_percent IS NOT NULL").fetchall()

    subs, meta = defaultdict(list), {}
    for r in rows:
        subs[r["id"]].append((r["subject_code"], r["weight_percent"]))
        meta[r["id"]] = (r["program_name_th"], r["seats"], r["round"], r["name_en"])

    results = []
    for pid, components in subs.items():
        total_w = sum(w for _, w in components)
        matched = [(c, w, scores[c]) for c, w in components if c in scores]
        if not matched:
            continue
        # require at least one non-GPAX subject match (else GPAX-only programs flood)
        if not any(c != "GPAX" for c, _, _ in matched):
            continue
        got_w = sum(w for _, w, _ in matched)
        fit = sum(w * s for _, w, s in matched) / got_w
        coverage = got_w / total_w if total_w else 0
        results.append({"fit": fit, "coverage": coverage, "pid": pid,
                        "matched": matched, "total_w": total_w})
    # rank: full-coverage eligible first, then by fit
    results.sort(key=lambda r: (r["coverage"], r["fit"]), reverse=True)
    return results[:limit], meta


if __name__ == "__main__":
    scores = parse_scores(sys.argv[1:])
    if not scores:
        print("usage: python -m ml.myfit 61:80 64:75 65:70 82:65 GPAX:3.8 TGAT:75")
        sys.exit(1)
    print(f"Your scores: {scores}\n")
    results, meta = myfit(scores)
    print(f"{'#':>2} {'fit':>5} {'cov':>4} {'uni':12s} {'program':36s} {'seats':>5}  weighting")
    for i, r in enumerate(results, 1):
        name, seats, rnd, uni = meta[r["pid"]]
        wt = " ".join(f"{c}:{w:g}%×{s:g}" for c, w, s in r["matched"])
        print(f"{i:>2} {r['fit']:>5.1f} {r['coverage']*100:>3.0f}% {uni[:12]:12s} "
              f"{(name or '')[:36]:36s} {seats or '-':>5}  {wt}")
