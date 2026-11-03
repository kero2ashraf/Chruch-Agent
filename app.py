# -*- coding: utf-8 -*-
"""
apply_update.py  —  upgrades app.py (final-v9) to final-v10 using the church feature spec.
Usage:  python apply_update.py app.py        (writes app.py, keeps app.py.bak)
Every edit is anchored on exact text; if ANY anchor is missing nothing is written.
Then run:  python app.py --selftest
"""
import re, sys, shutil

path = sys.argv[1] if len(sys.argv) > 1 else "app.py"
src = open(path, encoding="utf-8").read()
orig = src
EDITS = []   # (label, old, new)

def E(label, old, new):
    EDITS.append((label, old, new))

# ───────────────────────── 0. build tag ─────────────────────────
E("build", 'BUILD = "final-v9"', 'BUILD = "final-v10"')

# ───────────────────────── 1. helpers (before schemas) ─────────────────────────
HELPERS = r'''# ════════════════════════════════════════════════════════════════════════════
# v10 ADDITIONS: church template, age stage / venue, schedule, confidence, library
# ════════════════════════════════════════════════════════════════════════════
SCHEDULE_PATH = os.path.join(DATA_DIR, "schedule.json")
LIBRARY_PATH = os.path.join(DATA_DIR, "library.json")
PROFILE_PATH = os.path.join(DATA_DIR, "church_profile.json")
LOGO_PATH = os.path.join(DATA_DIR, "church_logo.bin")
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

AGE_STAGES = {
    "حضانة": "قصص بسيطة جدًا، صور وحركات وأغاني وألعاب قصيرة؛ جمل قصيرة وفكرة واحدة فقط، وكل نشاط 3-5 دقائق.",
    "ابتدائي": "قصص كتابية، أسئلة بسيطة، أنشطة عملية وتطبيقات يومية قريبة من حياة الطفل.",
    "إعدادي": "مناقشات ومواقف حياتية وأسئلة تفكير وتطبيقات أعمق، بأسلوب أقل طفولية.",
    "ثانوي": "حوار وتحليل مواقف وربط الدرس بالحياة اليومية وتحديات فكرية.",
    "جامعة وشباب": "دراسة كتابية أعمق ومناقشة عقائدية وتطبيقات حياتية وربط الدرس بالواقع.",
}
VENUE_GUIDE = {
    "فصل صغير (5-10 أطفال)": "ألعاب تفاعلية ومناقشات فردية وأنشطة تسمح بمشاركة كل طفل.",
    "فصل كبير": "مسابقات جماعية وتقسيم إلى فرق، وأنشطة تناسب الأعداد الكبيرة بخامات قليلة.",
    "داخل الكنيسة": "وسائل إيضاح هادئة، بدون جري أو حركة كبيرة أو أدوات قد تُتلف المكان.",
    "النادي": "ألعاب حركية ومسابقات وأنشطة جماعية في مساحة مفتوحة.",
    "اجتماع أسرة / مجموعة كبيرة": "طريقة شرح تناسب الجميع، مع تقسيم الحاضرين إلى مجموعات.",
}
TPL_KEYS = {
    "dates": "التاريخ الميلادي والقبطي", "objective": "هدف الدرس", "verse": "الآية الرئيسية",
    "intro": "أسئلة تمهيدية", "elements": "العناصر الأساسية", "lesson": "شرح الدرس",
    "reflection": "أفكار للتأمل", "application": "التطبيق العملي", "conclusion": "الختام",
}
COPTIC_MONTHS = ["توت", "بابه", "هاتور", "كيهك", "طوبة", "أمشير", "برمهات", "برمودة", "بشنس", "بؤونة", "أبيب", "مسرى", "النسئ"]


def load_json(path: str, default: Any) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: str, obj: Any):
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, default=str)
        os.replace(tmp, path)
    except Exception as e:
        log.warning("could not save %s: %s", path, e)


def gregorian_to_coptic(d: dt.date) -> Tuple[int, int, int]:
    cy = d.year - 283
    if d < coptic_to_gregorian(cy, 1, 1):
        cy -= 1
    n = (d - coptic_to_gregorian(cy, 1, 1)).days
    return cy, n // 30 + 1, n % 30 + 1


def coptic_str(d: dt.date) -> str:
    cy, cm, cd = gregorian_to_coptic(d)
    return f"{cd} {COPTIC_MONTHS[cm - 1]} {cy} ش"


def parse_date(s: Any) -> Optional[dt.date]:
    if isinstance(s, dt.datetime):
        return s.date()
    if isinstance(s, dt.date):
        return s
    parts = str(s or "").translate(AR_DIGITS).strip().split()
    if not parts:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(parts[0], fmt).date()
        except ValueError:
            continue
    return None


# -- church lesson schedule (Excel / CSV / PDF) ----------------------------------
_COLS = {"date": ("التاريخ", "تاريخ", "date", "اليوم"), "verses": ("آية", "آيات", "شواهد", "verse"),
         "stage": ("مرحلة", "فصل", "stage", "class", "الفئة"), "owner": ("مسؤول", "خادم", "owner", "servant", "المدرس"),
         "topic": ("نوع", "تصنيف", "topic", "type"), "title": ("عنوان", "الدرس", "موضوع", "title", "lesson")}


def rows_from_table(table: List[List[Any]]) -> Tuple[List[dict], str]:
    hdr_i, colmap = None, {}
    for i, row in enumerate(table[:10]):
        cm: Dict[str, int] = {}
        for j, cell in enumerate(row):
            h = ar_norm(str(cell or ""))
            if not h or "قبط" in h:
                continue
            for fld, syns in _COLS.items():
                if fld not in cm and any(ar_norm(s) in h for s in syns):
                    cm[fld] = j
                    break
        if "title" in cm:
            hdr_i, colmap = i, cm
            break
    if hdr_i is None:
        return [], "لم أجد عمود «عنوان الدرس» في الجدول."
    rows = []
    for row in table[hdr_i + 1:]:
        def get(f):
            j = colmap.get(f)
            return str(row[j]).strip() if j is not None and j < len(row) and row[j] is not None else ""
        title = get("title")
        if not title:
            continue
        d = parse_date(get("date"))
        rows.append({"date": d.isoformat() if d else "", "title": title, "stage": get("stage"),
                     "owner": get("owner"), "topic": get("topic"), "verses": get("verses")})
    return rows, ("" if rows else "الجدول فارغ.")


def parse_schedule(name: str, data: bytes) -> Tuple[List[dict], str]:
    low = name.lower()
    try:
        if low.endswith(".csv"):
            import csv
            return rows_from_table(list(csv.reader(io.StringIO(data.decode("utf-8-sig", errors="replace")))))
        if low.endswith(".xlsx"):
            import openpyxl
            wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
            return rows_from_table([list(r) for ws in wb.worksheets for r in ws.iter_rows(values_only=True)])
        if low.endswith(".pdf"):
            text, err = extract_doc_text(name, data)
            if err:
                return [], err
            rows = []
            for ln in text.splitlines():
                ln = ln.translate(AR_DIGITS)
                m = re.search(r"(\d{1,4}[/\-]\d{1,2}[/\-]\d{1,4})", ln)
                d = parse_date(m.group(1)) if m else None
                if d:
                    title = re.sub(r"\s+", " ", ln.replace(m.group(1), "")).strip(" -–|:،")
                    if len(title) >= 3:
                        rows.append({"date": d.isoformat(), "title": title, "stage": "", "owner": "", "topic": "", "verses": ""})
            return rows, ("" if rows else "لم أستطع قراءة جدول من الـ PDF — جرّب Excel أو CSV (الصور تحتاج تحويلها لجدول أولًا).")
    except ImportError:
        return [], "مكتبة قراءة Excel غير مثبتة (pip install openpyxl)."
    except Exception:
        return [], "تعذر قراءة ملف الجدول."
    return [], "نوع الملف غير مدعوم (Excel أو CSV أو PDF)."


def merge_schedule(old: List[dict], new: List[dict]) -> List[dict]:
    key = lambda r: (r.get("date", ""), r.get("stage", ""), "" if r.get("date") else r["title"])
    d = {key(r): r for r in old}
    d.update({key(r): r for r in new})
    return sorted(d.values(), key=lambda r: (r.get("date") or "9999", r["title"]))


def upcoming(rows: List[dict], today: dt.date, n: int = 5) -> List[dict]:
    return [r for r in sorted(rows, key=lambda r: r.get("date", "")) if r.get("date") and r["date"] >= today.isoformat()][:n]


def search_schedule(rows: List[dict], q: str) -> List[dict]:
    qn = ar_norm(q.translate(AR_DIGITS).strip())
    return [r for r in rows if qn in ar_norm(r["title"]) or qn in r.get("date", "") or qn in ar_norm(r.get("owner", ""))]


def previous_lesson(rows: List[dict], greg: str, title: str) -> str:
    past = [r for r in rows if r.get("date") and greg and r["date"] < greg and r["title"] != title]
    return max(past, key=lambda r: r["date"])["title"] if past else ""


# -- preparation context sent to the model -----------------------------------------
def default_prep() -> dict:
    return {"stage": "", "count": 0, "venue": "", "tools": "", "space": "", "greg": "", "coptic": "",
            "servant": "", "church": "", "tpl": [], "prev": ""}


def prep_context(state: dict) -> str:
    p = state.get("prep") or {}
    parts = []
    if p.get("stage") in AGE_STAGES:
        parts.append(f"المرحلة العمرية المستهدفة في هذا الطلب: {p['stage']} — تتقدم على أي فئة عمرية مذكورة في التعليمات العامة. "
                     f"طريقة التحضير المناسبة لها: {AGE_STAGES[p['stage']]}")
    if p.get("venue") in VENUE_GUIDE:
        parts.append(f"نوع الاجتماع: {p['venue']} — {VENUE_GUIDE[p['venue']]}")
    if p.get("count"):
        parts.append(f"عدد الأطفال/الحاضرين: {p['count']} — اختر أنشطة تناسب هذا العدد" + (" وقسّمهم إلى مجموعات." if int(p["count"]) > 20 else "."))
    if p.get("space"):
        parts.append(f"طبيعة المكان والمساحة: {p['space']}")
    if p.get("tools"):
        parts.append(f"الأدوات المتوفرة: {p['tools']} — لا تقترح خامات خارجها إلا كبديل بسيط.")
    if p.get("greg"):
        parts.append(f"تاريخ الاجتماع: {p['greg']}" + (f" ({p['coptic']})" if p.get("coptic") else ""))
    if p.get("prev"):
        parts.append(f"الدرس السابق في جدول الكنيسة: {p['prev']} — اربط الدرس الجديد به في البداية.")
    return ("\n".join(parts) + "\n\n") if parts else ""


# -- source confidence (High / Medium / Low) — computed in the BACKEND from the sources ----
_PATRISTIC = re.compile(r"(قال|يقول|قول|كما قال)\s+(القديس|الأنبا|البابا|الأب|أبونا|الآباء)")
_BIBLE_REF = re.compile(r"\(\s*[\u0600-\u06FF]+\s*\d+\s*[:：]\s*\d+")


def _overlap(text: str, ref_tokens: set) -> float:
    ta = set(tokens(text))
    return len(ta & ref_tokens) / len(ta) if ta else 0.0


def assess_confidence(state: dict) -> dict:
    """Grades the lesson's source-bearing parts. High = backed by the uploaded curriculum; Medium = web reference
    only (shown with a warning); Low = no supporting reference (excluded from the final export)."""
    lz = state.get("lesson") or {}
    chunks = [Chunk(**c) for c in (state.get("refs") or {}).get("lesson", [])]
    ref_tokens = {t for c in chunks for t in tokens(c.text)}
    rep: Dict[str, Any] = {"sections": [], "excluded_headings": [], "verse": "", "verse_excluded": False}
    verse = re.sub(r"\(.*?\)", "", lz.get("memory_verse") or "").strip()
    if verse:
        r = _overlap(verse, ref_tokens) if chunks else 0.0
        rep["verse"] = "high" if r >= .85 else "medium" if r >= .5 else "low"
        rep["verse_excluded"] = rep["verse"] == "low"
    for s in lz.get("sections", []):
        body, tag = str(s.get("body", "")), s.get("source_tag", "")
        if not (tag in ("منهج", "مرجع") or _PATRISTIC.search(body) or _BIBLE_REF.search(body)):
            continue
        valid = [chunks[r - 1] for r in s.get("refs", []) if isinstance(r, int) and 1 <= r <= len(chunks)]
        if any(c.origin == "curriculum" for c in valid):
            lvl, why = "high", "مسنود بمنهج الكنيسة المرفوع"
        elif valid:
            lvl, why = "medium", "مسنود بمرجع إلكتروني فقط — يحتاج مراجعة"
        elif chunks and _overlap(body, ref_tokens) >= 0.6:
            lvl, why = "medium", "يتطابق مع نص المراجع لكن بدون إشارة صريحة لها"
        else:
            lvl, why = "low", "لا يوجد مرجع مسترجع يسنده"
        rep["sections"].append({"heading": s.get("heading", ""), "level": lvl, "reason": why})
        if lvl == "low":
            rep["excluded_headings"].append(s.get("heading", ""))
    return rep


def filtered_lesson(state: dict) -> dict:
    """The lesson as it is exported: Low-confidence sections removed, Medium ones flagged, template applied."""
    lz = dict(state["lesson"])
    conf = state.get("confidence") or {}
    lvl = {s["heading"]: s["level"] for s in conf.get("sections", [])}
    secs = []
    for s in lz.get("sections", []):
        l = lvl.get(s.get("heading", ""))
        if l == "low":
            continue
        s = dict(s)
        if l == "medium":
            s["heading"] = (s.get("heading", "") + " ⚠ يحتاج مراجعة المصدر").strip()
        secs.append(s)
    lz["sections"] = secs
    if conf.get("verse_excluded"):
        lz["memory_verse"] = ""
    elif conf.get("verse") == "medium" and lz.get("memory_verse"):
        lz["memory_verse"] += " ⚠ تحتاج مراجعة"
    lz["_tpl"] = (state.get("prep") or {}).get("tpl") or list(TPL_KEYS)
    return lz


def header_blocks(state: dict) -> List[Tuple[str, str]]:
    p = state.get("prep") or {}
    tpl = set(p.get("tpl") or TPL_KEYS)
    B: List[Tuple[str, str]] = []
    if p.get("church"):
        B.append(("h2", p["church"]))
    if "dates" in tpl:
        ds = ([f"التاريخ الميلادي: {p['greg']}"] if p.get("greg") else []) + ([f"التاريخ القبطي: {p['coptic']}"] if p.get("coptic") else [])
        if ds:
            B.append(("p", "   |   ".join(ds)))
    meta = ([f"المرحلة: {p['stage']}"] if p.get("stage") else []) + ([f"الخادم: {p['servant']}"] if p.get("servant") else [])
    if meta:
        B.append(("p", "   |   ".join(meta)))
    return B


class Analysis(BaseModel):'''
E("helpers", "class Analysis(BaseModel):", HELPERS)

# ───────────────────────── 2. schemas ─────────────────────────
E("lesson-schema", '    conclusion: str = ""\n    review_flags: List[str] = []\n\n\nclass Exercise(BaseModel):',
r'''    conclusion: str = ""
    review_flags: List[str] = []
    elements: List[str] = []                 # العناصر الأساسية للدرس
    reflection: List[str] = []               # أفكار للتأمل
    application: List[str] = []              # تطبيق عملي في حياة الطفل


class Experiment(BaseModel):
    suitable: bool = True                    # False when the topic does not suit a science experiment
    title: str = ""
    tools: List[str] = []
    steps: List[str] = []
    spiritual_link: str = ""
    safety: List[str] = []
    alternative: str = ""
    review_flags: List[str] = []
    refs: List[int] = []


class HymnItem(BaseModel):
    title: str
    why: str = ""
    stage_note: str = ""
    source: str = ""


class Hymns(BaseModel):
    items: List[HymnItem]
    review_flags: List[str] = []


class Summary(BaseModel):
    bullets: List[str]
    previous_link: str = ""
    review_flags: List[str] = []


class Exercise(BaseModel):''')

# ───────────────────────── 3. tasks ─────────────────────────
E("tasks-head", "TASKS: Dict[str, Tuple[type, str, str]] = {",
r'''LESSON_EXTRA_INSTR = (
    "\n7) elements: «العناصر الأساسية للدرس» 3 إلى 5 نقاط قصيرة.\n"
    "8) reflection: 2 إلى 3 أفكار للتأمل.\n"
    "9) application: 2 إلى 3 أفكار للتطبيق العملي في حياة الطفل.\n"
    "10) لكل قسم في sections ضع source_tag (منهج/مرجع/شرح/إبداع/مراجعة) وأرقام المراجع التي اعتمد عليها في refs. "
    "أي آية أو قول لأحد الآباء لا بد أن يكون من المراجع المرفقة، مع ذكر قائله ومرجعه، وتوضيح هل هو حرفي أم شرح للمعنى. "
    "عند اختلاف التفاسير اذكر الاختلاف ولا تقدّم رأيًا غير مؤكد كأنه حقيقة."
)

TASKS: Dict[str, Tuple[type, str, str]] = {''')
E("tasks-lesson", '"lesson": (Lesson, "الدرس الكامل", LESSON_INSTR),',
  '"lesson": (Lesson, "الدرس الكامل", LESSON_INSTR + LESSON_EXTRA_INSTR),')
E("tasks-tail",
  '    "quiz": (Quiz, "الاختبار", "أنشئ اختبارًا من 6 أسئلة اختيار من متعدد بإجابة واحدة صحيحة وشرح قصير."),\n}',
r'''    "quiz": (Quiz, "الاختبار", "أنشئ اختبارًا من 6 أسئلة اختيار من متعدد بإجابة واحدة صحيحة وشرح قصير."),
    "experiment": (Experiment, "التجربة العلمية",
                   "اقترح تجربة علمية بسيطة تخدم فكرة الدرس: الأدوات في tools، الخطوات في steps، الربط بالمعنى الروحي/الكتابي في spiritual_link، "
                   "احتياطات السلامة في safety، وبديل بسيط لو الأدوات غير متوفرة في alternative. التجربة وسيلة إيضاح وليست بديلًا عن المعنى الكتابي. "
                   "إن لم يسمح موضوع الدرس بتجربة مناسبة اجعل suitable=false واترك الباقي فارغًا."),
    "hymns": (Hymns, "الترانيم والألحان",
              "اقترح 3 ترانيم أو ألحان أو مقاطع كتابية ملحنة مناسبة لموضوع الدرس والمرحلة العمرية، واربط كل واحدة بالهدف الروحي (why) "
              "وبملاحظة عن مناسبتها للعمر (stage_note). لا تخترع ترنيمة؛ اقترح فقط ما هو معروف ومتداول، واذكر مصدره في source إن عرفته، "
              "وإلا اتركه فارغًا وضع تنبيهًا في review_flags."),
    "summary": (Summary, "ملخص الخادم",
                "اكتب ملخصًا سريعًا للخادم قبل الاجتماع (6 إلى 8 نقاط): الفكرة، الآية، ترتيب الاجتماع، الأدوات، وأهم سؤال للنقاش. "
                "ثم اربط الدرس بالدرس السابق في previous_link إن ذُكر في طلب الخادم."),
}''')
E("task-icon", 'TASK_ICON = {"analysis": "🔎", "lesson": "📖", "activities": "🎲", "poster": "🖼️", "worksheet": "📝", "quiz": "❓"}',
  'TASK_ICON = {"analysis": "🔎", "lesson": "📖", "activities": "🎲", "poster": "🖼️", "worksheet": "📝", "quiz": "❓",\n             "experiment": "🔬", "hymns": "🎵", "summary": "🧾"}')

# ───────────────────────── 4. state / run_task / validation ─────────────────────────
E("new-state", '"validation": None, "gate": None, "refs": {}, "chat": []}',
  '"validation": None, "gate": None, "refs": {}, "chat": [],\n            "prep": default_prep(), "confidence": None}')
E("prompt-prep", '''    user = (f"طلب الخادم/الدرس:\\n{state['input']}\\n\\n"''',
  '''    user = (f"طلب الخادم/الدرس:\\n{state['input']}\\n\\n{prep_context(state)}"''')
E("reset-conf", '        state["validation"], state["gate"] = None, None',
  '        state["validation"], state["gate"], state["confidence"] = None, None, None')
E("hands-on", "    hands_on = json.dumps({k: state[k] for k in ('activities', 'poster', 'worksheet') if state.get(k)}, ensure_ascii=False)",
r'''    hands = {k: state[k] for k in ('activities', 'poster', 'worksheet') if state.get(k)}
    if state.get("experiment"):          # only tools/steps are hazards; the safety notes may legitimately say "no fire"
        hands["experiment"] = {k: state["experiment"].get(k) for k in ("tools", "steps")}
    hands_on = json.dumps(hands, ensure_ascii=False)''')
E("conf-after-gate", '    state["gate"] = status\n    return state["validation"]',
  '    state["gate"] = status\n    state["confidence"] = assess_confidence(state)\n    return state["validation"]')

# ───────────────────────── 5. render blocks & export ─────────────────────────
NEW_LESSON_BLOCKS = r'''def lesson_blocks(d: dict) -> List[Tuple[str, str]]:
    """Church template (v10). Sections can be switched off through d["_tpl"] (keys of TPL_KEYS):
    ❖ هدف الدرس / ❖ آية الدرس / ❖ أسئلة تمهيدية / ❖ العناصر الأساسية / ❖ الدرس / ❖ أفكار للتأمل / ❖ تطبيق عملي."""
    tpl = set(d.get("_tpl") or TPL_KEYS)
    B: List[Tuple[str, str]] = [("h1", d.get("title", ""))]
    if "objective" in tpl:
        B.append(("h2", "❖ هدف الدرس"))
        if d.get("main_idea"):
            B.append(("p", d["main_idea"]))
        if d.get("objectives"):
            B.append(("p", "مساعدة الطفل أن:"))
            B += [("li", o) for o in d["objectives"]]
    if "verse" in tpl and d.get("memory_verse"):
        B += [("h2", "❖ آية الدرس"), ("p", d["memory_verse"])]
    if "intro" in tpl and d.get("intro_questions"):
        B.append(("h2", "❖ أسئلة تمهيدية للدرس"))
        B += [("li", q) for q in d["intro_questions"]]
    if "elements" in tpl and d.get("elements"):
        B.append(("h2", "❖ العناصر الأساسية للدرس"))
        B += [("li", e) for e in d["elements"]]
    if "lesson" in tpl:
        B.append(("h2", "❖ الدرس"))
        for s in d.get("sections", []):
            tag = s.get("source_tag", "")
            if s.get("heading"):
                B.append(("h3", s["heading"] + (f" [{tag}]" if tag in ("إبداع", "مراجعة") else "")))
            for line in str(s.get("body", "")).splitlines():
                line = line.strip()
                if not line:
                    continue
                if _NUM_LINE.match(line):
                    B.append(("p", line))
                elif _BUL_LINE.match(line):
                    B.append(("li", _BUL_LINE.sub("", line)))
                else:
                    B.append(("p", line))
    if "reflection" in tpl and d.get("reflection"):
        B.append(("h2", "❖ أفكار للتأمل"))
        B += [("li", x) for x in d["reflection"]]
    if "application" in tpl and d.get("application"):
        B.append(("h2", "❖ تطبيق عملي في حياة الطفل"))
        B += [("li", x) for x in d["application"]]
    if "conclusion" in tpl and d.get("conclusion"):
        B += [("h3", "الختام"), ("p", d["conclusion"])]
    return B


'''
EDITS.append(("lesson-blocks", re.compile(r"def lesson_blocks\(d: dict\) -> List\[Tuple\[str, str\]\]:.*?(?=\ndef component_blocks)", re.S), NEW_LESSON_BLOCKS))

E("component-blocks", "    elif name == \"lesson\":\n        B = lesson_blocks(d)\n",
r'''    elif name == "lesson":
        B = lesson_blocks(d)
    elif name == "experiment":
        B.append(("h1", "تجربة علمية مرتبطة بالدرس"))
        if not d.get("suitable", True) or not d.get("title"):
            B.append(("p", "موضوع الدرس لا يستدعي تجربة علمية؛ يُكتفى بالأنشطة والوسائل."))
        else:
            B += [("h2", d["title"]), ("h3", "الأدوات")] + [("li", x) for x in d.get("tools", [])]
            B += [("h3", "خطوات التنفيذ")] + [("li", x) for x in d.get("steps", [])]
            if d.get("spiritual_link"):
                B += [("h3", "الربط بالمعنى الروحي"), ("p", d["spiritual_link"])]
            if d.get("safety"):
                B += [("h3", "احتياطات السلامة")] + [("li", x) for x in d["safety"]]
            if d.get("alternative"):
                B += [("h3", "بديل بسيط"), ("p", d["alternative"])]
    elif name == "hymns":
        B.append(("h1", "ترانيم وألحان مقترحة"))
        for h in d.get("items", []):
            B.append(("h3", h["title"]))
            for lab, key in (("المناسبة للدرس", "why"), ("ملاحظة للمرحلة", "stage_note"), ("المصدر", "source")):
                if h.get(key):
                    B.append(("li", f"{lab}: {h[key]}"))
            if not h.get("source"):
                B.append(("li", "المصدر غير مؤكد — راجع النص واللحن قبل الاستخدام"))
    elif name == "summary":
        B.append(("h1", "ملخص سريع للخادم"))
        B += [("li", b) for b in d.get("bullets", [])]
        if d.get("previous_link"):
            B += [("h3", "الربط بالدرس السابق"), ("p", d["previous_link"])]
''')

E("export-loop", '''    lesson = []
    for k in ("lesson", "activities", "poster"):
        if state.get(k):
            lesson += component_blocks(k, state[k])''',
r'''    lesson = header_blocks(state)
    for k in ("lesson", "experiment", "hymns", "activities", "poster"):
        d = state.get(k)
        if not d or (k == "experiment" and not d.get("suitable", True)):
            continue
        lesson += component_blocks(k, filtered_lesson(state) if k == "lesson" else d)''')
E("export-sets", '    out = {"الدرس": lesson, "أوراق الأطفال": student, "مفتاح الإجابة (للخادم فقط)": key}',
r'''    summ = component_blocks("summary", state["summary"]) if state.get("summary") else []
    out = {"الدرس": lesson, "ملخص الخادم": summ, "أوراق الأطفال": student, "مفتاح الإجابة (للخادم فقط)": key}''')

# DOCX / PDF with logo
E("docx-sig", "def blocks_to_docx(blocks: List[Tuple[str, str]]) -> bytes:", 'def blocks_to_docx(blocks: List[Tuple[str, str]], logo: bytes = b"") -> bytes:')
E("docx-logo", "    d = docx.Document()\n    for kind, text in blocks:",
r'''    d = docx.Document()
    if logo:
        try:
            d.add_picture(io.BytesIO(logo), width=docx.shared.Inches(1.0))
            d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        except Exception:
            log.warning("logo could not be embedded in DOCX")
    for kind, text in blocks:''')
E("pdf-sig", "def blocks_to_pdf(blocks: List[Tuple[str, str]]) -> bytes:", 'def blocks_to_pdf(blocks: List[Tuple[str, str]], logo: bytes = b"") -> bytes:')
E("pdf-logo", "    y = H - margin\n    shape = lambda s: get_display(arabic_reshaper.reshape(s))",
r'''    y = H - margin
    if logo:
        try:
            from reportlab.lib.utils import ImageReader
            c.drawImage(ImageReader(io.BytesIO(logo)), (W - 60) / 2, y - 60, 60, 60, preserveAspectRatio=True, mask="auto")
            y -= 72
        except Exception:
            log.warning("logo could not be drawn in PDF")
    shape = lambda s: get_display(arabic_reshaper.reshape(s))''')
E("docx-cache", "def _docx_cached(blocks_json: str) -> bytes:\n    return blocks_to_docx([tuple(b) for b in json.loads(blocks_json)])",
  'def _docx_cached(blocks_json: str, logo: bytes = b"") -> bytes:\n    return blocks_to_docx([tuple(b) for b in json.loads(blocks_json)], logo)')
E("pdf-cache", "def _pdf_cached(blocks_json: str) -> bytes:\n    return blocks_to_pdf([tuple(b) for b in json.loads(blocks_json)])",
  'def _pdf_cached(blocks_json: str, logo: bytes = b"") -> bytes:\n    return blocks_to_pdf([tuple(b) for b in json.loads(blocks_json)], logo)')
E("dl-docx", 'c2.download_button("DOCX", _docx_cached(bj), f"{title}.docx", key=f"d_{title}")',
  'c2.download_button("DOCX", _docx_cached(bj, ss.get("logo", b"")), f"{title}.docx", key=f"d_{title}")')
E("dl-pdf", 'c3.download_button("PDF", _pdf_cached(bj), f"{title}.pdf", key=f"p_{title}")',
  'c3.download_button("PDF", _pdf_cached(bj, ss.get("logo", b"")), f"{title}.pdf", key=f"p_{title}")')

# ───────────────────────── 6. UI ─────────────────────────
E("ui-defaults", '                 ("link_url", ""), ("link_follow", False)):',
r'''                 ("link_url", ""), ("link_follow", False), ("p_stage", "ابتدائي"), ("p_count", 15),
                 ("p_venue", "فصل صغير (5-10 أطفال)"), ("p_tools", ""), ("p_space", ""), ("p_date", dt.date.today()),
                 ("p_servant", ""), ("p_church", ""), ("sched_q", ""), ("logo", b"")):''')
E("ui-profile-load", "    # A KnowledgeBase kept in the session by an OLDER version of this file lacks newer methods -> rebuild it.",
r'''    if "profile_loaded" not in ss:                      # church profile, template, schedule, library (shared on disk)
        pr = load_json(PROFILE_PATH, {})
        ss["profile_loaded"] = True
        ss["p_church"] = pr.get("church", ss["p_church"])
        ss["p_servant"] = pr.get("servant", ss["p_servant"])
        ss["tpl"] = [k for k in pr.get("tpl", list(TPL_KEYS)) if k in TPL_KEYS] or list(TPL_KEYS)
        ss["schedule"] = load_json(SCHEDULE_PATH, [])
        ss["library"] = load_json(LIBRARY_PATH, [])
        try:
            with open(LOGO_PATH, "rb") as f_:
                ss["logo"] = f_.read()
        except OSError:
            pass
    # A KnowledgeBase kept in the session by an OLDER version of this file lacks newer methods -> rebuild it.''')

E("ui-callbacks", "    def add_link():",
r'''    def use_row(row: dict):
        new_lesson()
        ss["title"] = row.get("title", "")
        ss["notes"] = " — ".join(x for x in (row.get("topic", ""), row.get("verses", "")) if x)
        ss["p_date"] = parse_date(row.get("date", "")) or dt.date.today()
        for s_ in AGE_STAGES:
            if row.get("stage") and (s_ in row["stage"] or row["stage"] in s_):
                ss["p_stage"] = s_
        if row.get("owner"):
            ss["p_servant"] = row["owner"]

    def save_to_library():
        S_ = ss["S"]
        keep = {k: S_.get(k) for k in list(TASKS) + ["input", "refs", "prep", "gate", "validation", "confidence"]}
        keep["occasion"] = None
        pr_ = S_.get("prep") or {}
        entry = {"id": str(int(time.time() * 1000)), "title": (S_["input"].splitlines() or [""])[0],
                 "date": pr_.get("greg", ""), "saved_at": dt.datetime.now().isoformat(timespec="seconds"), "state": keep}
        ss["library"] = [e for e in ss["library"] if (e["title"], e["date"]) != (entry["title"], entry["date"])] + [entry]
        save_json(LIBRARY_PATH, ss["library"])

    def open_from_library(eid: str):
        e = next((x for x in ss["library"] if x["id"] == eid), None)
        if not e:
            return
        ns = new_state()
        ns.update(e["state"])
        ss["S"] = ns
        first, _, rest = ns["input"].partition("\n")
        ss["title"], ss["notes"], ss["approved"] = first, rest, False
        pr_ = ns.get("prep") or {}
        if pr_.get("stage") in AGE_STAGES:
            ss["p_stage"] = pr_["stage"]
        ss["p_date"] = parse_date(pr_.get("greg", "")) or dt.date.today()

    def delete_from_library(eid: str):
        ss["library"] = [x for x in ss["library"] if x["id"] != eid]
        save_json(LIBRARY_PATH, ss["library"])

    def add_link():''')

E("ui-schedule", "    # ---------------- 1) lesson title ----------------",
r'''    # ---------------- 0) church lesson schedule ----------------
    st.markdown('<div class="sec">📅 جدول دروس الكنيسة</div>', unsafe_allow_html=True)
    sched = ss["schedule"]
    nxt = upcoming(sched, dt.date.today(), 1)
    if nxt:
        st.info(f"🔔 الدرس القادم: **{nxt[0]['title']}** — {nxt[0]['date']}" + (f" — {nxt[0]['stage']}" if nxt[0].get("stage") else ""))
    with st.expander(f"📅 الجدول ({len(sched)} درس) — رفع / بحث / بدء التحضير"):
        up = st.file_uploader("ارفع الجدول (Excel أو CSV أو PDF) — الأعمدة: التاريخ، عنوان الدرس، المرحلة، المسؤول، نوع الموضوع، الآيات",
                              type=["xlsx", "csv", "pdf"], key="sched_up")
        if up is not None and ss.get("sched_sig") != (up.name, up.size):
            rows_, err_ = parse_schedule(up.name, up.getvalue())
            if err_:
                st.error(err_)
            else:
                ss["schedule"] = merge_schedule(ss["schedule"], rows_)
                save_json(SCHEDULE_PATH, ss["schedule"])
                ss["sched_sig"] = (up.name, up.size)
                sched = ss["schedule"]
                st.success(f"تم تحديث الجدول بـ {len(rows_)} درس (المكرر يُحدَّث بدون إدخال يدوي).")
        st.text_input("بحث بالعنوان أو التاريخ أو الخادم", key="sched_q")
        shown = search_schedule(sched, ss["sched_q"]) if ss["sched_q"].strip() else upcoming(sched, dt.date.today(), 8)
        if not shown:
            st.caption("لا توجد دروس لعرضها.")
        for i_, r_ in enumerate(shown):
            c1_, c2_ = st.columns([4, 1])
            c1_.write(f"**{r_['title']}** — {r_.get('date') or 'بدون تاريخ'}"
                      + "".join(f" — {r_[k]}" for k in ("stage", "owner", "verses") if r_.get(k)))
            c2_.button("ابدأ التحضير", key=f"sr_{i_}", on_click=use_row, args=(r_,))
        if sched and st.button("🗑️ مسح الجدول", key="sched_clear"):
            ss["schedule"] = []
            save_json(SCHEDULE_PATH, [])
            st.rerun()

    # ---------------- 1) lesson title ----------------''')

E("ui-prep-sync", '    S["input"] = (ss["title"].strip() + ("\\n" + notes if notes else "")).strip()',
r'''    S["input"] = (ss["title"].strip() + ("\n" + notes if notes else "")).strip()
    gd_ = ss["p_date"] if isinstance(ss["p_date"], dt.date) else None
    S["prep"] = {"stage": ss["p_stage"], "count": int(ss["p_count"]), "venue": ss["p_venue"], "tools": ss["p_tools"].strip(),
                 "space": ss["p_space"].strip(), "greg": gd_.isoformat() if gd_ else "", "coptic": coptic_str(gd_) if gd_ else "",
                 "servant": ss["p_servant"].strip(), "church": ss["p_church"].strip(), "tpl": list(ss["tpl"]),
                 "prev": previous_lesson(ss["schedule"], gd_.isoformat() if gd_ else "", ss["title"].strip())}''')

E("ui-prep-widgets", "    # ---------------- 2) curriculum upload ----------------",
r'''    # ---------------- 1b) preparation details ----------------
    with st.expander("🎯 تفاصيل التحضير: المرحلة · العدد · المكان · التاريخ · Template الكنيسة"):
        a1, a2 = st.columns(2)
        a1.selectbox("المرحلة العمرية", list(AGE_STAGES), key="p_stage")
        a2.number_input("عدد الأطفال / الحاضرين", 1, 500, key="p_count")
        a1.selectbox("نوع الاجتماع والمكان", list(VENUE_GUIDE), key="p_venue")
        a2.text_input("المساحة المتاحة / طبيعة المكان", key="p_space", placeholder="مثال: قاعة صغيرة بكراسي ثابتة")
        st.text_input("الأدوات المتوفرة", key="p_tools", placeholder="مثال: ورق ملون، أقلام، بروجكتور")
        b1, b2 = st.columns(2)
        b1.date_input("التاريخ الميلادي", key="p_date")
        b2.markdown(f"**التاريخ القبطي:** {S['prep']['coptic']}")
        st.divider()
        f1, f2 = st.columns(2)
        f1.text_input("اسم الكنيسة", key="p_church")
        f2.text_input("اسم الخادم / المسؤول", key="p_servant")
        st.multiselect("أقسام التحضير في الـ Template (تظهر في الدرس والتصدير)", list(TPL_KEYS), key="tpl",
                       format_func=lambda k: TPL_KEYS[k])
        lg = st.file_uploader("شعار الكنيسة (PNG / JPG)", type=["png", "jpg", "jpeg"], key="logo_up")
        if lg is not None and ss.get("logo_sig") != (lg.name, lg.size):
            ss["logo"], ss["logo_sig"] = lg.getvalue(), (lg.name, lg.size)
            try:
                os.makedirs(DATA_DIR, exist_ok=True)
                with open(LOGO_PATH, "wb") as f_:
                    f_.write(ss["logo"])
            except OSError:
                pass
        if st.button("💾 حفظ بيانات الكنيسة والـ Template"):
            save_json(PROFILE_PATH, {"church": ss["p_church"], "servant": ss["p_servant"], "tpl": list(ss["tpl"])})
            st.success("تم الحفظ — يظهر في كل تحضير جديد.")

    # ---------------- 2) curriculum upload ----------------''')

E("ui-order", '    ORDER = ["analysis", "lesson", "activities", "poster", "worksheet", "quiz"]',
  '    ORDER = ["analysis", "lesson", "experiment", "hymns", "activities", "poster", "worksheet", "quiz", "summary"]')

E("ui-confidence", '        st.caption("أي إعادة توليد تُبطل نتيجة المراجعة السابقة.")\n',
r'''        st.caption("أي إعادة توليد تُبطل نتيجة المراجعة السابقة.")
    conf = S.get("confidence")
    if conf and (conf["sections"] or conf["verse"]):
        LV = {"high": "🟢 عالية", "medium": "🟡 متوسطة — تحتاج مراجعة", "low": "🔴 منخفضة — استُبعدت من التصدير"}
        with st.expander("📊 تقرير الثقة في المعلومات (الاستبعاد تلقائي للمنخفضة)"):
            st.caption("التقييم محسوب من جودة المصدر وتطابق النص معه، وليس من رأي النموذج وحده.")
            if conf["verse"]:
                st.write(f"**آية الدرس:** {LV[conf['verse']]}")
            for s_ in conf["sections"]:
                st.write(f"**{s_['heading'] or 'قسم'}:** {LV[s_['level']]} — {s_['reason']}")
''')

E("ui-library", "    # ---------------- chat ----------------",
r'''    # ---------------- lesson library ----------------
    st.markdown('<div class="sec">🗂️ مكتبة الدروس</div>', unsafe_allow_html=True)
    st.button("💾 حفظ هذا التحضير في المكتبة", disabled=not has_content, on_click=save_to_library, key="lib_save")
    for e_ in reversed(ss["library"]):
        l1, l2, l3 = st.columns([4, 1, 1])
        l1.write(f"**{e_['title'] or 'بدون عنوان'}** — {e_.get('date') or e_['saved_at'][:10]}")
        l2.button("فتح", key=f"lo_{e_['id']}", on_click=open_from_library, args=(e_["id"],), disabled=running)
        l3.button("حذف", key=f"ld_{e_['id']}", on_click=delete_from_library, args=(e_["id"],))
    if not ss["library"]:
        st.caption("لا توجد تحضيرات محفوظة بعد.")

    # ---------------- chat ----------------''')

# ───────────────────────── 7. self-tests ─────────────────────────
E("selftests", "    suite = unittest.defaultTestLoader.loadTestsFromTestCase(T)",
r'''        def test_coptic_date(self):
            self.assertEqual(gregorian_to_coptic(dt.date(2026, 9, 11)), (1743, 1, 1))
            self.assertEqual(gregorian_to_coptic(dt.date(2026, 1, 7)), (1742, 4, 29))

        def test_schedule_csv_merge_upcoming(self):
            csv_ = "التاريخ,عنوان الدرس,المرحلة,المسؤول\n2026-11-08,الكنيسة,ابتدائي,مينا\n١٠/١١/٢٠٢٦,الصدق,إعدادي,\n".encode("utf-8")
            rows, err = parse_schedule("s.csv", csv_)
            self.assertEqual(err, "")
            self.assertEqual([r["date"] for r in rows], ["2026-11-08", "2026-11-10"])
            new = [{"date": "2026-11-08", "title": "الكنيسة (معدل)", "stage": "ابتدائي", "owner": "", "topic": "", "verses": ""}]
            m = merge_schedule(rows, new)
            self.assertEqual((len(m), m[0]["title"]), (2, "الكنيسة (معدل)"))
            self.assertEqual(upcoming(m, dt.date(2026, 11, 9), 5)[0]["title"], "الصدق")
            self.assertEqual(previous_lesson(m, "2026-11-10", "الصدق"), "الكنيسة (معدل)")

        def test_confidence_levels_and_exclusion(self):
            cur = Chunk("الكنيسة بيت الله ونحن نصلي فيها كل أسبوع", "c", "local:c", "local:c", "", "sunday_school", "s", origin="curriculum")
            S = new_state()
            S["refs"]["lesson"] = [asdict(cur)]
            S["lesson"] = {"title": "ت", "memory_verse": "الكنيسة بيت الله (مز 1: 1)", "sections": [
                {"heading": "أ", "body": "قال القديس أثناسيوس كلام كثير جدا غير موجود", "source_tag": "شرح", "refs": []},
                {"heading": "ب", "body": "الكنيسة بيت الله", "source_tag": "منهج", "refs": [1]}]}
            r = assess_confidence(S)
            self.assertEqual(r["verse"], "high")
            self.assertEqual(r["excluded_headings"], ["أ"])
            S["confidence"] = r
            heads = [t for k, t in component_blocks("lesson", filtered_lesson(S)) if k == "h3"]
            self.assertEqual(heads, ["ب"])

        def test_prep_context_and_template(self):
            S = new_state()
            self.assertEqual(prep_context(S), "")
            S["prep"].update({"stage": "ثانوي", "venue": "النادي", "count": 30, "greg": "2026-11-08", "coptic": "29 بابه 1743 ش"})
            ctx = prep_context(S)
            self.assertIn("ثانوي", ctx)
            self.assertIn("مجموعات", ctx)
            blocks = lesson_blocks({"title": "ت", "main_idea": "م", "sections": [], "_tpl": ["lesson"]})
            self.assertNotIn("❖ هدف الدرس", [t for _, t in blocks])
            S["prep"]["church"] = "كنيسة"
            self.assertEqual(header_blocks(S)[0], ("h2", "كنيسة"))

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(T)''')

# ───────────────────────── apply ─────────────────────────
missing = []
for label, old, new in EDITS:
    if isinstance(old, re.Pattern):
        src, n = old.subn(lambda m: new, src, count=1)
        if n != 1:
            missing.append(label)
    else:
        if src.count(old) != 1:
            missing.append(f"{label} (found {src.count(old)}x)")
            continue
        src = src.replace(old, new)
if missing:
    sys.exit("NOT APPLIED — anchors not found/unique: " + ", ".join(missing) + "\nNothing was written.")
compile(src, path, "exec")          # refuse to write code that does not even parse
shutil.copy(path, path + ".bak")
open(path, "w", encoding="utf-8").write(src)
print(f"OK — {len(EDITS)} edits applied. Backup: {path}.bak\nNow run: python {path} --selftest")
