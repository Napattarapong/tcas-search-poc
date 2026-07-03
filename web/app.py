#!/usr/bin/env python3
"""
Basic web PoC: a search box that returns matching programs.
Uses the LLM-free query pipeline (ml.query.search). Run:
    python web/app.py
then open http://localhost:5000
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flask import Flask, request, render_template_string  # noqa: E402
from ml.query import search  # noqa: E402

app = Flask(__name__)

CODE_NAME = {"61": "Math1", "62": "Math2", "63": "Stat", "64": "Physics", "65": "Chem",
             "66": "Bio", "70": "Social", "81": "Thai", "82": "English", "83": "French",
             "84": "German", "85": "Japanese", "86": "Korean", "87": "Chinese",
             "88": "Pali", "89": "Spanish"}

PAGE = """<!doctype html>
<html lang="th"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TCAS Program Search</title>
<style>
  body{font-family:-apple-system,"Sarabun",sans-serif;max-width:760px;margin:40px auto;padding:0 16px;color:#222}
  h1{font-size:1.4rem;margin-bottom:.2em}
  .sub{color:#666;font-size:.9rem;margin-bottom:1.2em}
  form{display:flex;gap:8px;margin-bottom:1em}
  input[type=text]{flex:1;padding:10px;font-size:1rem;border:1px solid #ccc;border-radius:6px}
  button{padding:10px 18px;font-size:1rem;border:0;background:#1a477f;color:#fff;border-radius:6px;cursor:pointer}
  .sig{background:#f4f7fb;padding:8px 12px;border-radius:6px;font-size:.85rem;margin-bottom:1em;color:#345}
  .hit{border:1px solid #eee;border-radius:8px;padding:10px 12px;margin-bottom:8px}
  .hit .uni{color:#1a477f;font-weight:600;font-size:.85rem}
  .hit .prog{font-size:1rem;margin:2px 0}
  .hit .meta{color:#888;font-size:.8rem}
  .none{color:#888}
  code{background:#f0f0f0;padding:1px 5px;border-radius:3px}
</style></head><body>
<h1>🎓 TCAS Program Search</h1>
<div class="sub">LLM-free · type Thai or English (typos OK). e.g.
  <code>วิศวะคอม จุฬา</code>, <code>physics and math, more than 100 seats</code></div>
<form method="get" action="/search">
  <input type="text" name="q" value="{{ q }}" placeholder="ค้นหาสาขา / มหาวิทยาลัย / วิชาที่ต้องการ" autofocus>
  <button type="submit">Search</button>
</form>
{% if sig %}
<div class="sig">Parsed →
  university: <b>{{ sig.university or 'any' }}</b> ·
  subjects: <b>{% if sig.subjects %}{% for c in sig.subjects %}{{ CODE_NAME.get(c, c) }} ({{c}}){% if not loop.last %}, {% endif %}{% endfor %}{% else %}any{% endif %}</b> ·
  min seats: <b>{{ sig.seats_min or '—' }}</b>
</div>
{% endif %}
{% if hits is not none %}
  {% if hits %}
    <div>{{ hits|length }} matches</div>
    {% for p in hits %}
    <div class="hit">
      <div class="uni">{{ p.university }}</div>
      <div class="prog">{{ p.program }}</div>
      <div class="meta">seats: {{ p.seats if p.seats is not none else '—' }}</div>
    </div>
    {% endfor %}
  {% else %}
    <div class="none">No matches. Try broader terms.</div>
  {% endif %}
{% endif %}
</body></html>
"""


@app.get("/")
def index():
    return render_template_string(PAGE, q="", sig=None, hits=None, CODE_NAME=CODE_NAME)


@app.get("/search")
def do_search():
    q = (request.args.get("q") or "").strip()
    if not q:
        return render_template_string(PAGE, q="", sig=None, hits=None, CODE_NAME=CODE_NAME)
    sig, hits = search(q)
    return render_template_string(PAGE, q=q, sig=sig, hits=hits, CODE_NAME=CODE_NAME)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
