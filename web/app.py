#!/usr/bin/env python3
"""
Web app: keyword search + My Fit (score-based program matcher).
LLM-free. Run: python web/app.py -> http://localhost:5000
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flask import Flask, request, render_template_string  # noqa: E402
from ml.query import search  # noqa: E402
from ml.myfit import myfit  # noqa: E402

app = Flask(__name__)

CODE_NAME = {"61": "Math1", "62": "Math2", "63": "Stat", "64": "Physics", "65": "Chem",
             "66": "Bio", "70": "Social", "81": "Thai", "82": "English", "83": "French",
             "84": "German", "85": "Japanese", "86": "Korean", "87": "Chinese",
             "88": "Pali", "89": "Spanish"}

FIT_FIELDS = [
    ("gpax", "GPAX", "GPAX", "0-4"), ("tgat", "TGAT", "TGAT", "0-100"),
    ("tpat3", "TPAT3", "TPAT 3", "0-100"), ("s61", "61", "Math1 (61)", "0-100"),
    ("s64", "64", "Physics (64)", "0-100"), ("s65", "65", "Chem (65)", "0-100"),
    ("s66", "66", "Bio (66)", "0-100"), ("s82", "82", "English (82)", "0-100"),
    ("s81", "81", "Thai (81)", "0-100"), ("s70", "70", "Social (70)", "0-100"),
]

LAYOUT = """<!doctype html><html lang="th"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{{ title }}</title>
<style>
body{font-family:-apple-system,"Sarabun",sans-serif;max-width:820px;margin:30px auto;padding:0 16px;color:#222}
h1{font-size:1.3rem;margin-bottom:.2em} .sub{color:#666;font-size:.85rem;margin-bottom:1em}
nav{margin-bottom:1.5em;padding-bottom:.5em;border-bottom:1px solid #eee}
nav a{color:#1a477f;text-decoration:none;margin-right:1em;font-weight:600}
nav a.cur{color:#222;border-bottom:2px solid #1a477f}
input[type=text]{padding:10px;font-size:1rem;border:1px solid #ccc;border-radius:6px;width:100%;box-sizing:border-box}
button{padding:10px 18px;font-size:1rem;border:0;background:#1a477f;color:#fff;border-radius:6px;cursor:pointer;margin-top:8px}
.hit{border:1px solid #eee;border-radius:8px;padding:10px 12px;margin-bottom:8px}
.uni{color:#1a477f;font-weight:600;font-size:.85rem} .prog{font-size:1rem;margin:2px 0}
.meta{color:#888;font-size:.8rem} .none{color:#888}
.fit-row{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:1em}
.fit-field{display:flex;flex-direction:column;gap:2px}
.fit-field label{font-size:.75rem;color:#666}
.fit-field input{width:70px;padding:6px;border:1px solid #ccc;border-radius:4px;font-size:.9rem}
.badge{display:inline-block;background:#e8f0e8;color:#2d6a2d;padding:2px 8px;border-radius:4px;font-size:.75rem;font-weight:600;margin-left:6px}
.signals{margin-bottom:1em}
.chip{display:inline-block;background:#eef3f8;color:#1a477f;padding:4px 10px;border-radius:12px;font-size:.8rem;margin:2px}
.chip b{color:#d2691e}
</style></head><body>
<h1>{{ title }}</h1><div class="sub">{{ subtitle }}</div>
<nav><a {% if tab=='search' %}class="cur"{% endif %} href="/">Search</a>
<a {% if tab=='fit' %}class="cur"{% endif %} href="/myfit">My Fit</a></nav>
{{ body|safe }}
</body></html>"""


def page(title, subtitle, tab, body):
    return render_template_string(LAYOUT, title=title, subtitle=subtitle, tab=tab, body=body)


def _hit(uni, prog, seats, extra=""):
    return (f'<div class="hit"><div class="uni">{uni}</div><div class="prog">{prog}</div>'
            f'<div class="meta">seats: {seats or "—"} {extra}</div></div>')


@app.get("/")
def index():
    body = '<form method="get" action="/search"><input type="text" name="q" placeholder="ค้นหาสาขา / มหาวิทยาลัย / วิชา" autofocus><button>Search</button></form>'
    return page("🎓 TCAS Search", "LLM-free · Thai or English (typos OK)", "search", body)


@app.get("/search")
def do_search():
    q = (request.args.get("q") or "").strip()
    body = f'<form method="get" action="/search"><input type="text" name="q" value="{q}" placeholder="ค้นหาสาขา / มหาวิทยาลัย / วิชา" autofocus><button>Search</button></form>'
    if not q:
        return page("🎓 TCAS Search", "LLM-free · Thai or English (typos OK)", "search", body)
    sig, hits = search(q)
    # visual signal chips
    chips = ""
    if sig["university"]:
        chips += f'<span class="chip">🎓 <b>{sig["university"]}</b></span>'
    if sig.get("faculty"):
        chips += f'<span class="chip">🏛️ <b>{sig["faculty"]}</b></span>'
    if sig.get("major"):
        chips += f'<span class="chip">📚 <b>{sig["major"]}</b></span>'
    if sig["subjects"]:
        subs = ", ".join(CODE_NAME.get(c, c) for c in sig["subjects"])
        chips += f'<span class="chip">📐 <b>{subs}</b></span>'
    if sig["seats_min"]:
        chips += f'<span class="chip">💺 ≥<b>{sig["seats_min"]}</b> seats</span>'
    if sig.get("round"):
        chips += f'<span class="chip">🔄 <b>{sig["round"]}</b></span>'
    if sig.get("gpax"):
        chips += f'<span class="chip">📊 GPAX <b>{sig["gpax"]}</b></span>'
    if sig.get("intl"):
        chips += f'<span class="chip">🌍 <b>International</b></span>'
    if sig["keywords"]:
        chips += f'<span class="chip">🔍 <b>{", ".join(sig["keywords"])}</b></span>'
    if sig.get("format"):
        chips += f'<span class="chip">📝 <b>{sig["format"]}</b></span>'
    if sig.get("intent"):
        chips += f'<span class="chip">🎯 <b>{sig["intent"].title()}</b></span>'
    body += f'<div class="signals">{chips}</div>'
    if hits:
        body += f"<div>{len(hits)} matches</div>"
        for p in hits:
            codes = sorted(p.get("codes") or [])
            subj = ", ".join(CODE_NAME.get(c, c) for c in codes[:5]) if codes else ""
            body += _hit(p["university"], p["program"], p["seats"],
                         f"· requires: {subj}" if subj else "")
    else:
        body += '<div class="none">No matches.</div>'
    return page("🎓 TCAS Search", "LLM-free · Thai or English (typos OK)", "search", body)


@app.route("/myfit", methods=["GET", "POST"])
def myfit_page():
    vals = {f: (request.form.get(f) or "") for f, _, _, _ in FIT_FIELDS} if request.method == "POST" else {}
    body = '<form method="post" action="/myfit"><div class="fit-row">'
    for f, code, label, ph in FIT_FIELDS:
        body += f'<div class="fit-field"><label>{label}</label><input type="text" name="{f}" placeholder="{ph}" value="{vals.get(f, "")}"></div>'
    body += '</div><button>Find My Fit</button></form>'
    if request.method == "POST":
        scores = {}
        for field, code, _, _ in FIT_FIELDS:
            v = (request.form.get(field) or "").strip()
            if v:
                try:
                    scores[code] = float(v) * 25 if code == "GPAX" else float(v)
                except ValueError:
                    pass
        if scores:
            ranked, meta = myfit(scores, limit=20)
            body += f"<h3>{len(ranked)} best-fit programs</h3>"
            for i, r in enumerate(ranked, 1):
                name, seats, rnd, uni = meta[r["pid"]]
                wt = " ".join(f"{CODE_NAME.get(c, c)}:{w:g}%×{s:g}" for c, w, s in r["matched"])
                elig = ' <span class="badge" style="background:#d4edda">eligible</span>' if r["coverage"] >= 1 else ""
                body += _hit(f"{i}. {uni} <span class='badge'>fit {r['fit']:.1f}</span>{elig}", name, seats, f"· round {rnd} · {wt}")
    return page("🎯 My Fit", "Enter your scores → programs that favor your strengths", "fit", body)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
