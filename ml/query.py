#!/usr/bin/env python3
"""
Query pipeline — typed free-form Thai/English -> ranked programs.
100% LLM-free and model-free: stdlib + PyThaiNLP dictionary segmenter + rapidfuzz
(classical string similarity, no neural net, no API).

Stages
  1. normalize      NFKC + collapse whitespace
  2. tokenize       PyThaiNLP newmm + TCAS domain Trie  (pipeline.thai_tokenize)
  3. parse signals  fuzzy-lift: university, required subjects, min seats, keywords
  4. filter         university / required-subjects ⊆ program codes / seats >= min
  5. rank           rapidfuzz token_set_ratio(keywords, program+faculty name)
  6. return         top-k program records

Usage
    python -m ml.query "compter enginering at chula"
    python -m ml.query "phisics and math, more than 100 seats"
"""
import glob
import json
import os
import re
import sys
import unicodedata

from rapidfuzz import fuzz, process

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline.thai_tokenize import tokenize as _tok  # noqa: E402

EXT = os.path.join(ROOT, "data", "extracted")
K = 8
FUZZ_THR = 85  # WRatio cutoff for fuzzy synonym matching

# university synonyms -> canonical (substring-matched against university name)
UNI = {
    "chulalongkorn": "Chulalongkorn", "chula": "Chulalongkorn", "จุฬา": "Chulalongkorn",
    "จุฬาลงกรณ์": "Chulalongkorn",
    "เชียงใหม่": "Chiang Mai", "cmu": "Chiang Mai", "chiang mai": "Chiang Mai",
    "ธรรมศาสตร์": "Thammasat", "thammasat": "Thammasat", "tu": "Thammasat",
    "ราชมงคล": "Rajamangala", "rajamangala": "Rajamangala",
    "มหิดล": "Mahidol", "mahidol": "Mahidol", "mu": "Mahidol",
    "เกษตร": "Kasetsart", "kasetsart": "Kasetsart",
    "ขอนแก่น": "Khon Kaen", "khon kaen": "Khon Kaen",
    "สงขลา": "Prince of Songkla", "songkla": "Prince of Songkla", "psu": "Prince of Songkla",
    "สิลปากร": "Silpakorn", "silpakorn": "Silpakorn",
    "นเรศวร": "Naresuan", "naresuan": "Naresuan",
    "แม่ฟ้าหลวง": "Mae Fah Luang", "mae fah luang": "Mae Fah Luang",
    "มหาสารคาม": "Mahasarakham", "mahasarakham": "Mahasarakham",
    "ศรีนครินทรวิโรฒ": "Srinakharinwirot", "srinakharinwirot": "Srinakharinwirot",
    "ทักษิณ": "Thaksin", "thaksin": "Thaksin",
    "วลัยลักษณ์": "Walailak", "walailak": "Walailak",
    "สุรนารี": "Suranaree", "suranaree": "Suranaree",
    "พระจอมเกล้า": "King Mongkut", "kmitl": "King Mongkut",
    "บูรพา": "Burapha", "burapha": "Burapha",
    "แม่โจ้": "Maejo", "maejo": "Maejo",
    "รามคำแหง": "Ramkhamhaeng", "ramkhamhaeng": "Ramkhamhaeng",
    "เชียงราย": "Mae Fah Luang",
    # Thai university abbreviations
    "มช": "Chiang Mai", "มธ": "Thammasat", "มก": "Kasetsart",
    "มข": "Khon Kaen", "มศว": "Srinakharinwirot", "มรม": "Ramkhamhaeng",
    "มอ": "Prince of Songkla", "มน": "Naresuan", "มจ": "Maejo",
    "มฟล": "Mae Fah Luang", "มทส": "Suranaree", "มส": "Mahasarakham",
    "มทร": "Rajamangala", "มกฬ": "Mahidol", "สมช": "Srinakharinwirot",
}
# subject synonyms -> A-Level code
SUBJ = {
    "physics": "64", "ฟิสิกส์": "64", "chem": "65", "chemistry": "65", "เคมี": "65",
    "bio": "66", "biology": "66", "ชีววิทยา": "66", "ชีวะ": "66",
    "math": "61", "maths": "61", "mathematics": "61", "คณิตศาสตร์": "61", "คณิต": "61",
    "english": "82", "ภาษาอังกฤษ": "82", "อังกฤษ": "82", "thai": "81", "ภาษาไทย": "81",
    "ไทย": "81", "social": "70", "สังคม": "70", "สังคมศึกษา": "70",
    "french": "83", "ฝรั่งเศส": "83", "chinese": "87", "จีน": "87",
    "japanese": "85", "ญี่ปุ่น": "85", "statistics": "63", "stat": "63", "สถิติ": "63",
}
STOP = set("ที่ ของ และ the a an of with needing need require requiring requires "
           "and more than over at in for to is are มี ต้องการ อยาก เรียน เข้า "
           "หลักสูตร programs program seats ที่นั่ง มากกว่า คน".split())

ROUND_KW = {
    "portfolio": "R1", "แฟ้มสะสม": "R1", "รอบ 1": "R1", "รอบ1": "R1", "รอบที่1": "R1",
    "quota": "R2", "โควตา": "R2", "รอบ 2": "R2", "รอบ2": "R2", "รอบที่2": "R2",
    "admission": "R3", "รอบ 3": "R3", "รอบ3": "R3", "รอบที่3": "R3",
    "รับตรง": "R4", "รอบ 4": "R4", "รอบ4": "R4", "รอบที่4": "R4", "direct": "R4",
}

# faculty/field keywords -> (canonical label, extra match terms for fuzzy)
FACULTY = {
    "วิศวกรรม": ("Engineering", "วิศวกรรม engineering"), "วิศวะ": ("Engineering", "วิศวกรรม engineering"),
    "engineering": ("Engineering", "วิศวกรรม engineering"),
    "engineer": ("Engineering", "วิศวกรรม engineering"),
    "แพทย์": ("Medicine", "แพทยศาสตร์ medicine"), "medicine": ("Medicine", "แพทยศาสตร์ medicine"),
    "พยาบาล": ("Nursing", "พยาบาลศาสตร์ nursing"), "nursing": ("Nursing", "พยาบาลศาสตร์ nursing"),
    "เภสัช": ("Pharmacy", "เภสัชศาสตร์ pharmacy"), "ทันตแพทย์": ("Dentistry", "ทันตแพทย์ dentistry"),
    "วิทยาศาสตร์": ("Science", "วิทยาศาสตร์ science"), "science": ("Science", "วิทยาศาสตร์ science"),
    "อักษร": ("Arts", "อักษรศาสตร์ arts"), "ศิลปศาสตร์": ("Arts", "ศิลปศาสตร์ arts"),
    "arts": ("Arts", "ศิลปศาสตร์ arts"), "ศิลปกรรม": ("Fine Arts", "ศิลปกรรม fine arts"),
    "พาณิชย์": ("Business", "พาณิชย์ commerce business"),
    "บริหาร": ("Business", "บริหารธุรกิจ business"), "business": ("Business", "บริหารธุรกิจ business"),
    "นิติ": ("Law", "นิติศาสตร์ law"), "law": ("Law", "นิติศาสตร์ law"),
    "ครุ": ("Education", "ครุศาสตร์ education"), "ศึกษาศาสตร์": ("Education", "ศึกษาศาสตร์ education"),
    "นิเทศ": ("Communication", "นิเทศศาสตร์ communication"),
    "รัฐศาสตร์": ("Political Science", "รัฐศาสตร์ political"),
    "เศรษฐศาสตร์": ("Economics", "เศรษฐศาสตร์ economics"), "economics": ("Economics", "เศรษฐศาสตร์ economics"),
    "สถาปัตย": ("Architecture", "สถาปัตยกรรม architecture"),
    "เกษตร": ("Agriculture", "เกษตรศาสตร์ agriculture"),
    "สัตวแพทย์": ("Veterinary", "สัตวแพทย์ veterinary"),
    "เทคโนโลยี": ("Technology", "เทคโนโลยี technology"),
    "สาธารณสุข": ("Public Health", "สาธารณสุข public health"),
}

# major/specialization keywords -> (label, extra match terms). More specific than faculty.
MAJOR = {
    "คอมพิวเตอร์": ("Computer", "คอมพิวเตอร์ computer"), "computer": ("Computer", "คอมพิวเตอร์ computer"),
    "คอม": ("Computer", "คอมพิวเตอร์ computer"),
    "ไอที": ("IT", "สารสนเทศ information technology IT"),
    "สารสนเทศ": ("IT", "สารสนเทศ information technology"),
    "ซอฟต์แวร์": ("Software", "ซอฟต์แวร์ software"),
    "ปัญญาประดิษฐ์": ("AI", "ปัญญาประดิษฐ์ artificial intelligence"),
    "ข้อมูล": ("Data Science", "ข้อมูล data"),
    "การตลาด": ("Marketing", "การตลาด marketing"), "marketing": ("Marketing", "การตลาด marketing"),
    "การเงิน": ("Finance", "การเงิน finance"), "finance": ("Finance", "การเงิน finance"),
    "การจัดการ": ("Management", "การจัดการ management"), "management": ("Management", "การจัดการ management"),
    "บัญชี": ("Accounting", "บัญชี accounting"), "accounting": ("Accounting", "บัญชี accounting"),
    "การออกแบบ": ("Design", "การออกแบบ design"), "design": ("Design", "การออกแบบ design"),
    "จิตวิทยา": ("Psychology", "จิตวิทยา psychology"), "psychology": ("Psychology", "จิตวิทยา psychology"),
    "กายภาพ": ("Physiotherapy", "กายภาพบำบัด physiotherapy physical therapy"),
    "โภชนาการ": ("Nutrition", "โภชนาการ nutrition"), "อาหาร": ("Food Science", "อาหาร food"),
    "การท่องเที่ยว": ("Tourism", "การท่องเที่ยว tourism"), "tourism": ("Tourism", "การท่องเที่ยว tourism"),
    "โลจิสติกส์": ("Logistics", "โลจิสติกส์ logistics"),
    "สื่อสารมวลชน": ("Mass Com", "สื่อสารมวลชน mass communication"),
    "ดนตรี": ("Music", "ดนตรี music"), "ทัศนศิลป์": ("Visual Arts", "ทัศนศิลป์ visual arts"),
    "สังคมวิทยา": ("Sociology", "สังคมวิทยา sociology"),
    "วิศวกรรมคอมพิวเตอร์": ("Computer Eng", "วิศวกรรมคอมพิวเตอร์ computer engineering"),
}

# program format -> (label, thai keyword to filter program names by)
FORMAT = {
    "ภาคปกติ": ("Regular", None), "regular": ("Regular", None),
    "ภาคพิเศษ": ("Special", "ภาคพิเศษ"), "special": ("Special", "ภาคพิเศษ"),
    "evening": ("Special", "ภาคพิเศษ"), "พิเศษ": ("Special", "ภาคพิเศษ"),
    "ทวิภาคี": ("Dual Degree", "ทวิภาคี"), "dual": ("Dual Degree", "ทวิภาคี"),
    "double": ("Dual Degree", "ทวิภาคี"),
    "bilingual": ("Bilingual", "bilingual"), "สองภาษา": ("Bilingual", "bilingual"),
    "ต่อเนื่อง": ("Integrated", "ต่อเนื่อง"),
    "sandbox": ("Sandbox", "sandbox"),
}

# query intent -> what the user wants to do (routing signal, no filter)
INTENT = {
    "เกณฑ์": "criteria", "คะแนน": "criteria", "น้ำหนัก": "criteria",
    "ต้องใช้": "criteria", "คุณสมบัติ": "criteria", "criteria": "criteria",
    "เทียบ": "compare", "เปรียบเทียบ": "compare", "ดีกว่า": "compare", "อันไหนดี": "compare",
    "อาชีพ": "career", "เงินเดือน": "career", "หางาน": "career", "จ้าง": "career",
    "ทุน": "cost", "เรียนฟรี": "cost", "ค่าเทอม": "cost", "ค่าเล่าเรียน": "cost", "ถูก": "cost",
}

# Thai abbreviations -> full form (expanded per-token, no cascading)
ABBREV = {
    "วิศวะ": "วิศวกรรม",
    "คอม": "คอมพิวเตอร์",
    "แพทย์": "แพทยศาสตร์", "เมด": "แพทยศาสตร์",
    "พยาบาล": "พยาบาลศาสตร์",
    "บัญชี": "บัญชีบัณฑิต",
    "สถาปัตย์": "สถาปัตยกรรม",
    "นิเทศ": "นิเทศศาสตร์",
    "เภสัช": "เภสัชศาสตร์",
    "ทันต": "ทันตแพทย์",
    "สัตว์": "สัตวแพทย์",
    "เกษตร": "เกษตรศาสตร์",
    "นิติ": "นิติศาสตร์",
    "ครุ": "ครุศาสตร์",
}


# ---------- stages ----------
def normalize(text):
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _fuzzy(token, table, thr=FUZZ_THR):
    """Exact, then fuzzy (WRatio) synonym match for tokens len>=4."""
    if token in table:
        return table[token]
    if len(token) >= 4:
        best = process.extractOne(token, list(table.keys()), scorer=fuzz.WRatio)
        if best and best[1] >= thr:
            return table[best[0]]
    return None


def _is_noise(t):
    return t.isdigit() or all(not c.isalnum() for c in t)


def parse_signals(text):
    norm = normalize(text)
    toks = _tok(norm)
    low = [t.lower() for t in toks]
    uni, subjects, consumed = None, set(), set()
    for t, tl in zip(toks, low):
        u = _fuzzy(t, UNI) or _fuzzy(tl, UNI)
        if u and not uni:
            uni, consumed = u, consumed | {t}
            continue
        s = _fuzzy(t, SUBJ) or _fuzzy(tl, SUBJ)
        if s:
            subjects.add(s); consumed.add(t)
    # fallback: check full text for multi-token Thai university names
    if not uni:
        for kw, val in UNI.items():
            if len(kw) > 1 and kw in norm.lower():
                uni = val
                break
    m = re.search(r"(?:more than|over|มากกว่า|>)\s*(\d+)", text)
    seats_min = int(m.group(1)) if m else None
    joined_low = " ".join(low)
    norm_low = norm.lower()
    round_label = next((rk for kw, rk in ROUND_KW.items() if kw in norm_low), None)
    gm = re.search(r"(?:GPAX|เกรดเฉลี่ย|เกรด)\s*[: ]?\s*(\d+\.?\d*)", norm, re.I)
    gpax = float(gm.group(1)) if gm else None
    intl = "นานาชาติ" in norm_low or "international" in norm_low
    keywords = [t for t in toks
                if t not in consumed and t.lower() not in STOP and not _is_noise(t)]
    # phrase = original content keywords joined (before enrichment), space-normalized
    # used for precise substring matching: "วิศวกรรมคอมพิวเตอร์" should NOT match "วิทยาการคอมพิวเตอร์"
    phrase_kw = keywords.copy()
    # expand each token's abbreviation independently (no cascading)
    phrase_expanded = [ABBREV.get(kw, kw) for kw in phrase_kw]
    phrase = re.sub(r"\s+", "", "".join(phrase_expanded)).lower()
    phrase_active = len(phrase_kw) >= 2 and len(phrase) > 3 and any("฀" <= c <= "๿" for c in phrase)
    faculty = None
    for kw, (label, match) in FACULTY.items():
        if kw in norm_low:
            faculty = label
            for m in match.split():
                if m not in keywords and m.lower() not in STOP:
                    keywords.append(m)
            break
    major = None
    for kw, (label, match) in MAJOR.items():
        if kw in norm_low:
            major = label
            for m in match.split():
                if m not in keywords and m.lower() not in STOP:
                    keywords.append(m)
            break
    fmt_label, fmt_kw = None, None
    for kw, (label, match) in FORMAT.items():
        if kw in norm_low:
            fmt_label, fmt_kw = label, match
            break
    intent = next((label for kw, label in INTENT.items() if kw in norm_low), None)
    return {"university": uni, "subjects": sorted(subjects),
            "seats_min": seats_min, "keywords": keywords,
            "round": round_label, "gpax": gpax, "intl": intl,
            "faculty": faculty, "major": major,
            "format": fmt_label, "format_kw": fmt_kw, "intent": intent,
            "phrase": phrase, "phrase_active": phrase_active, "raw": text}


_TABLE = None


def load_programs():
    """Load the combined search table once (cached) — data/search/programs.jsonl."""
    global _TABLE
    if _TABLE is None:
        path = os.path.join(ROOT, "data", "search", "programs.jsonl")
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        _TABLE = [{
            "university": r["university"],
            "program": r["program_name_th"],
            "name": (r.get("program_name_en", "") + " " + r["faculty_th"]
                     + " " + r["program_name_th"]).strip(),
            "seats": r.get("seats"),
            "round": r.get("round"),
            "min_gpax": r.get("min_gpax"),
            "codes": set(r.get("subject_codes") or []),
        } for r in rows]
    return _TABLE


def search(text, k=K):
    sig = parse_signals(text)
    progs = load_programs()
    kw = " ".join(sig["keywords"])

    def passes(p):
        if sig["university"] and sig["university"] not in p["university"]:
            return False
        if sig["subjects"] and not set(sig["subjects"]).issubset(p["codes"]):
            return False
        if sig["seats_min"] and (p["seats"] or 0) < sig["seats_min"]:
            return False
        if sig["round"] and p.get("round") != sig["round"]:
            return False
        if sig["gpax"] and (p.get("min_gpax") or 0) > sig["gpax"]:
            return False
        if sig["intl"] and "นานาชาติ" not in p.get("name", "").lower() \
                and "international" not in p.get("name", "").lower():
            return False
        if sig.get("format_kw") and sig["format_kw"] not in p.get("name", "").lower():
            return False
        if sig.get("phrase_active"):
            name_nospace = re.sub(r"\s+", "", p.get("name", "")).lower()
            if sig["phrase"] not in name_nospace:
                return False
        return True

    cands = [p for p in progs if passes(p)]
    kws = sig["keywords"]

    def kscore(p):
        if not kws:
            return (0, p["seats"] or 0)
        name = p["name"]
        per = [max(fuzz.partial_ratio(k, name), fuzz.token_set_ratio(k, name)) for k in kws]
        return (sum(per) / len(per), p["seats"] or 0)

    for p in cands:
        p["_score"] = kscore(p)
    cands.sort(key=lambda p: p["_score"], reverse=True)
    return sig, cands[:k]


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "compter enginering at chula"
    sig, hits = search(q)
    print("QUERY :", q)
    print("PARSED:", {k: v for k, v in sig.items() if k != "raw"})
    print(f"\n{len(hits)} matches:")
    for p in hits:
        print(f"  {p['university']:12s} | {p['program'][:42]:42s} | seats {p['seats']}")
