# -*- coding: utf-8 -*-
"""
Church Lesson Creative Teaching Agent — مساعد مدارس الأحد الذكي   (FINAL · free engines)
Streamlit UI + 4 free AI engines + source-aware retrieval + Coptic calendar
+ validation gate + DOCX/PDF export + built-in self-tests.

Run app :  streamlit run app.py
Run tests: python app.py --selftest        (no key / no internet needed)

Keys are read ONLY from the backend (env vars / .streamlit/secrets.toml) — nothing is typed in the UI.
Free engines (each falls back to the others automatically):
  1. Gemma     -> OpenRouter, free ":free" Gemma models discovered LIVE from the catalog
  2. Gemini    -> Google AI Studio free tier (key from aistudio.google.com/apikey)
                  (+ any free Gemini that OpenRouter happens to list)
  3. Auto      -> OpenRouter's own free router "openrouter/free" (always-on safety net)

Keys (env var, .env, or .streamlit/secrets.toml):
  OPENROUTER_API_KEY   for engines 1, 3, 4
  GEMINI_API_KEY       for engine 2          (optional GEMINI_MODEL_ID)
"""
from __future__ import annotations

import datetime as dt
import io
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib import robotparser
from urllib.parse import urljoin, urlparse

import requests
from pydantic import BaseModel, ValidationError, field_validator

# ════════════════════════════════════════════════════════════════════════════
# 0. CONFIG
# ════════════════════════════════════════════════════════════════════════════
APP_TITLE_AR = "مساعد مدارس الأحد الذكي"
BUILD = "final-v5"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
ROUTER_FREE = "openrouter/free"
GEMINI_DEFAULT = "gemini-flash-latest"
GEMINI_LITE = "gemini-flash-lite-latest"
FREE_MIN_INTERVAL = 3.0          # free tiers allow ~20 requests/min -> pace requests
DATA_DIR = os.environ.get("APP_DATA_DIR", "data")
KB_CACHE_PATH = os.path.join(DATA_DIR, "kb_cache.json")
USER_AGENT = "ChurchLessonAgent/1.0 (educational, respects robots.txt)"
STALE_DAYS = 30

ENGINES = ["gemma", "gemini", "auto"]
ENGINE_AR = {"gemma": "Gemma", "gemini": "Gemini", "auto": "تلقائي"}
ENGINE_ICON = {"gemma": "💎", "gemini": "✨", "auto": "🧭"}

log = logging.getLogger("church_agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def get_secret(name: str, default: str = "") -> str:
    """Environment first, then Streamlit secrets. Never logged."""
    v = os.environ.get(name)
    if v:
        return v.strip()
    try:
        import streamlit as st
        v = st.secrets.get(name, default)
        return (v or default).strip()
    except Exception:
        return default


def redact(text: str, key: str = "") -> str:
    if key:
        text = text.replace(key, "***")
    text = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "sk-***", text)
    return re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "AIza***", text)


def is_free_id(model_id: str) -> bool:
    return model_id == ROUTER_FREE or model_id.endswith(":free")


MODEL_ID_RE = re.compile(r"^~?[\w.\-]+/[\w.\-:~]+$")

# ════════════════════════════════════════════════════════════════════════════
# 1. LLM PROVIDER — OpenRouter (free only) + Gemini (Google AI Studio)
# ════════════════════════════════════════════════════════════════════════════
class LLMError(Exception):
    def __init__(self, code: str, msg_ar: str):
        super().__init__(msg_ar)
        self.code, self.msg_ar = code, msg_ar


FALLBACK_CODES = {"rate", "server", "network", "model", "credits", "empty", "provider", "bad_response", "http"}


@dataclass(frozen=True)
class Endpoint:
    base: str
    key: str
    model: str
    kind: str = "openrouter"       # "openrouter" | "gemini"
    label: str = ""

    @property
    def free_route(self) -> bool:
        return self.kind == "openrouter" and is_free_id(self.model)

    @property
    def name(self) -> str:
        return self.label or self.model


class LLMProvider:
    def __init__(self, endpoints: List[Endpoint], site_url: str = "",
                 app_name: str = "Church Lesson Creative Teaching Agent",
                 timeout: int = 120, max_retries: int = 2,
                 http=requests, sleep: Callable[[float], None] = time.sleep,
                 on_model: Optional[Callable[[str], None]] = None, min_interval: float = 0.0):
        self.endpoints = list(endpoints)
        self.site_url, self.app_name = site_url, app_name
        self.timeout, self.max_retries = timeout, max_retries
        self.http, self.sleep = http, sleep
        self.on_model, self.last_model = on_model, ""
        self.min_interval, self._last_call = min_interval, 0.0
        self._no_json_mode: set = set()

    @classmethod
    def single(cls, key: str, model: str, kind: str = "openrouter", **kw) -> "LLMProvider":
        base = GEMINI_BASE if kind == "gemini" else OPENROUTER_BASE
        return cls([Endpoint(base, (key or "").strip(), (model or "").strip(), kind)], **kw)

    # -- validation -------------------------------------------------------
    @staticmethod
    def check_ready(ep: Endpoint):
        if not ep.key:
            if ep.kind == "gemini":
                raise LLMError("missing_key", "مفتاح Gemini غير مُعد. أضف GEMINI_API_KEY (مجاني من Google AI Studio).")
            raise LLMError("missing_key", "مفتاح OpenRouter غير مُعد. أضف OPENROUTER_API_KEY في ملف .env أو في Streamlit secrets.")
        if not ep.model:
            raise LLMError("missing_model", "لم يتم اختيار نموذج.")
        if ep.kind == "openrouter":
            if not MODEL_ID_RE.match(ep.model):
                raise LLMError("bad_model", f"معرّف النموذج غير صالح الصيغة: {ep.model}")
            if not is_free_id(ep.model):
                raise LLMError("not_free", f"هذا التطبيق يعمل بالنماذج المجانية فقط (:free). تم رفض: {ep.model}")

    def _headers(self, ep: Endpoint) -> Dict[str, str]:
        h = {"Authorization": f"Bearer {ep.key}", "Content-Type": "application/json", "Accept": "application/json"}
        if ep.kind == "openrouter":
            h["X-Title"] = self.app_name
            if self.site_url:
                h["HTTP-Referer"] = self.site_url
        return h

    def _pace(self, ep: Endpoint):
        if self.min_interval > 0 and ep.free_route:
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                self.sleep(wait)
        self._last_call = time.monotonic()

    # -- core call --------------------------------------------------------
    def chat(self, messages: List[Dict[str, str]], **kw) -> str:
        """Tries each endpoint in order. Collects every failure so the user sees WHY."""
        if not self.endpoints:
            raise LLMError("missing_model", "لا يوجد نموذج مجاني متاح الآن. أضف مفتاح OpenRouter أو Gemini من الشريط الجانبي.")
        failures, bad_keys, last = [], set(), None
        for ep in self.endpoints:
            if ep.key in bad_keys:
                continue
            try:
                out = self._chat_one(ep, messages, **kw)
                self.last_model = ep.name
                if self.on_model:
                    self.on_model(ep.name)
                return out
            except LLMError as e:
                last = e
                failures.append(f"{ep.name}: {e.msg_ar}")
                if e.code == "auth":
                    bad_keys.add(ep.key)
                elif e.code not in FALLBACK_CODES:
                    raise
        if len(failures) > 1:
            raise LLMError(last.code, "فشلت كل النماذج المتاحة:\n" + "\n".join("• " + f for f in failures))
        raise last

    def _chat_one(self, ep: Endpoint, messages: List[Dict[str, str]], temperature: float = 0.7,
                  max_tokens: int = 2500, json_mode: bool = False) -> str:
        self.check_ready(ep)
        # Free OpenRouter routes: no response_format / reasoning (many free providers reject them
        # or return empty text). JSON is enforced by prompt + local validation + one repair pass.
        use_json = json_mode and not ep.free_route and ep.model not in self._no_json_mode
        payload: Dict[str, Any] = {"model": ep.model, "messages": messages, "temperature": temperature,
                                   "max_tokens": max_tokens, "stream": False}
        if use_json:
            payload["response_format"] = {"type": "json_object"}
        url = f"{ep.base}/chat/completions"
        last_exc: Optional[LLMError] = None
        empty_tries = 0
        for attempt in range(self.max_retries + 2):
            self._pace(ep)
            try:
                r = self.http.post(url, headers=self._headers(ep), json=payload, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as e:
                last_exc = LLMError("network", "تعذر الاتصال (مهلة أو شبكة). حاول مرة أخرى.")
                log.warning("network error attempt %d: %s", attempt + 1, redact(str(e), ep.key))
                if attempt < self.max_retries:
                    self.sleep(min(2 ** attempt, 8))
                    continue
                raise last_exc
            sc = r.status_code
            if sc == 200:
                try:
                    return self._extract(r, ep.model)
                except LLMError as e:
                    if e.code != "empty" or empty_tries >= 2:
                        raise
                    last_exc, empty_tries = e, empty_tries + 1
                    if use_json:
                        self._no_json_mode.add(ep.model)
                        payload.pop("response_format", None)
                        use_json = False
                    elif getattr(e, "finish", "") == "length":
                        payload["max_tokens"] = min(int(payload["max_tokens"]) * 2, 8000)
                    self.sleep(1)
                    continue
            body = self._safe_text(r)
            if sc in (401, 403):
                raise LLMError("auth", "مفتاح API غير صحيح أو غير مصرّح به.")
            if sc == 402:
                raise LLMError("credits", "هذا المسار يطلب رصيدًا.")
            if sc == 400 and use_json:
                self._no_json_mode.add(ep.model)
                payload.pop("response_format", None)
                use_json = False
                continue
            if sc == 404 or (sc == 400 and re.search(r"model", body, re.I)):
                raise LLMError("model", "النموذج غير متاح أو المعرّف غير صحيح.")
            if sc == 429 or sc >= 500:
                last_exc = LLMError("rate" if sc == 429 else "server",
                                    "حد الطلبات المجاني مشغول حاليًا." if sc == 429 else "خطأ مؤقت من الخادم.")
                if attempt < self.max_retries:
                    ra = r.headers.get("Retry-After", "") if getattr(r, "headers", None) else ""
                    base = float(ra) if ra.replace(".", "").isdigit() else 3 * (2 ** attempt)
                    self.sleep(min(base, 20))
                    continue
                raise last_exc
            raise LLMError("http", f"رفض الخادم الطلب (رمز {sc}).")
        raise last_exc or LLMError("unknown", "فشل غير معروف.")

    @staticmethod
    def _safe_text(r) -> str:
        try:
            return r.text[:500]
        except Exception:
            return ""

    @staticmethod
    def _extract(r, model: str) -> str:
        try:
            data = r.json()
        except Exception:
            raise LLMError("bad_response", "رد غير مفهوم من الخادم.")
        if isinstance(data, list) and data:
            data = data[0]
        if isinstance(data, dict) and data.get("error"):
            err = data["error"]
            msg = str(err.get("message", "")) if isinstance(err, dict) else str(err)
            raise LLMError("provider", f"فشل النموذج {model}: {msg[:200]}")
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except Exception:
            raise LLMError("bad_response", "رد النموذج لا يحتوي على نص.")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if not content or not str(content).strip():
            err = LLMError("empty", "النموذج أعاد ردًا فارغًا.")
            err.finish = choice.get("finish_reason") or ""
            raise err
        return str(content)

    # -- structured output -----------------------------------------------
    @staticmethod
    def _parse_json(text: str) -> Any:
        t = text.strip()
        t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
        try:
            return json.loads(t)
        except Exception:
            m = re.search(r"\{.*\}", t, re.S)
            if m:
                return json.loads(m.group(0))
            raise

    def structured(self, schema: type, messages: List[Dict[str, str]], **kw):
        sch = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        msgs = list(messages) + [{"role": "user", "content":
                "أعد الرد بصيغة JSON صالحة فقط (بدون أي نص خارجه) ومطابقة لهذا المخطط:\n" + sch
                + "\nكل النصوص بالعربية."}]
        raw = self.chat(msgs, json_mode=True, **kw)
        try:
            return schema.model_validate(self._parse_json(raw))
        except (ValueError, ValidationError) as first:
            log.info("malformed JSON; attempting one repair")
            fix = msgs + [{"role": "assistant", "content": raw[:6000]},
                          {"role": "user", "content": f"الـJSON غير صالح أو لا يطابق المخطط ({str(first)[:300]}). أعد JSON صحيحًا فقط."}]
            raw2 = self.chat(fix, json_mode=True, **kw)
            try:
                return schema.model_validate(self._parse_json(raw2))
            except (ValueError, ValidationError):
                raise LLMError("malformed", "النموذج لم يُنتج بيانات بالصيغة المطلوبة بعد محاولة تصحيح واحدة. أعد المحاولة أو غيّر المحرّك.")


# ---- OpenRouter live catalog -> engine model lists ---------------------------------
def fetch_catalog(http=requests) -> List[dict]:
    r = http.get(f"{OPENROUTER_BASE}/models", timeout=30, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.json().get("data", [])


def free_models(catalog: List[dict]) -> List[str]:
    """Text-output models priced exactly 0/0. Ranking: not-tiny, JSON-capable, preferred family, big context.
    The openrouter/free router is always appended last."""
    prefer = ("gemma", "gemini", "deepseek", "qwen", "llama", "gpt-oss", "mistral")
    out = []
    for m in catalog:
        try:
            mid = m["id"]
            p = m.get("pricing", {})
            if float(p.get("prompt", 1)) != 0 or float(p.get("completion", 1)) != 0 or mid == ROUTER_FREE:
                continue
            outs = (m.get("architecture") or {}).get("output_modalities")
            if outs and "text" not in outs:
                continue
            sp = m.get("supported_parameters") or []
            tiny = 1 if re.search(r"[-_e](1|2|3|4)b\b", mid.lower()) else 0
            fam = next((i for i, f in enumerate(prefer) if f in mid.lower()), len(prefer))
            out.append((tiny, 0 if ("response_format" in sp or "structured_outputs" in sp) else 1, fam,
                        -int(m.get("context_length") or 0), mid))
        except (TypeError, ValueError, KeyError):
            continue
    return [i for *_, i in sorted(out)] + [ROUTER_FREE]


def pick_engine_models(free: List[str]) -> Dict[str, List[str]]:
    low = lambda s: s.lower()
    return {"gemma": [i for i in free if "gemma" in low(i)][:3],
            "gemini_or": [i for i in free if "gemini" in low(i)][:2]}


EMPTY_MAP: Dict[str, List[str]] = {"gemma": [], "gemini_or": []}


def build_endpoints(choice: str, free_map: Dict[str, List[str]], or_key: str, gem_key: str,
                    gem_model: str = GEMINI_DEFAULT, fallback: bool = True) -> List[Endpoint]:
    """The chosen engine first, then the others (auto router last). Engines without a key are skipped."""
    def eps(name: str) -> List[Endpoint]:
        out: List[Endpoint] = []
        if name == "gemma" and or_key:
            out = [Endpoint(OPENROUTER_BASE, or_key, m, "openrouter", f"Gemma · {m}") for m in free_map.get("gemma", [])]
        elif name == "gemini":
            if gem_key:
                for gm in dict.fromkeys([gem_model or GEMINI_DEFAULT, GEMINI_LITE]):
                    out.append(Endpoint(GEMINI_BASE, gem_key, gm, "gemini", f"Gemini · {gm}"))
            if or_key:
                out += [Endpoint(OPENROUTER_BASE, or_key, m, "openrouter", f"Gemini · {m}") for m in free_map.get("gemini_or", [])]
        elif name == "auto" and or_key:
            out = [Endpoint(OPENROUTER_BASE, or_key, ROUTER_FREE, "openrouter", "Auto · openrouter/free")]
        return out

    order = [choice] + [n for n in ENGINES if n != choice] if fallback else [choice]
    res, seen = [], set()
    for n in order:
        for e in eps(n):
            if (e.base, e.model) not in seen:
                seen.add((e.base, e.model))
                res.append(e)
    return res


def engine_status(free_map: Dict[str, List[str]], or_key: str, gem_key: str) -> Dict[str, Tuple[bool, str]]:
    g, go = free_map.get("gemma", []), free_map.get("gemini_or", [])
    return {
        "gemma": (bool(or_key and g), f"{len(g)} نموذج مجاني عبر OpenRouter" if (or_key and g)
                  else ("المفتاح غير مضبوط في الخادم" if not or_key else "لا يوجد Gemma مجاني الآن")),
        "gemini": (bool(gem_key or (or_key and go)), "عبر Google AI Studio (مجاني)" if gem_key
                   else (f"{len(go)} مجاني عبر OpenRouter" if (or_key and go) else "المفتاح غير مضبوط في الخادم")),
        "auto": (bool(or_key), "الموزّع المجاني openrouter/free" if or_key else "المفتاح غير مضبوط في الخادم"),
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. COPTIC CALENDAR (computed; flagged for human review)
# ════════════════════════════════════════════════════════════════════════════
def coptic_to_gregorian(cy: int, cm: int, cd: int) -> dt.date:
    jdn = 1825029 + 365 * (cy - 1) + cy // 4 + 30 * (cm - 1) + cd
    return dt.date.fromordinal(jdn - 1721425)


def orthodox_easter(year: int) -> dt.date:
    """Julian computus converted to Gregorian (valid 1900-2099)."""
    a, b, c = year % 4, year % 7, year % 19
    d = (19 * c + 15) % 30
    e = (2 * a + 4 * b - d + 34) % 7
    month, day = (d + e + 114) // 31, (d + e + 114) % 31 + 1
    return dt.date(year, month, day) + dt.timedelta(days=13)


FIXED_FEASTS = {
    "عيد النيروز (رأس السنة القبطية)": (1, 1), "عيد الصليب": (1, 17),
    "عيد الميلاد المجيد": (4, 29), "عيد الغطاس": (5, 11), "عيد البشارة": (7, 29),
    "عيد دخول المسيح أرض مصر": (9, 24), "عيد الرسل": (11, 5),
    "عيد التجلي": (12, 13), "عيد صعود جسد السيدة العذراء": (12, 16),
}
MOVABLE_OFFSETS = {
    "بداية الصوم الكبير": -55, "أحد الشعانين": -7, "خميس العهد": -3,
    "الجمعة العظيمة": -2, "عيد القيامة المجيد": 0, "عيد الصعود": 39, "عيد العنصرة (الخمسين)": 49,
}
OCCASION_KEYWORDS = {
    "عيد الرسل": ["عيد الرسل", "الرسل", "بطرس وبولس"], "عيد القيامة المجيد": ["القيامة", "قيامة"],
    "عيد الميلاد المجيد": ["الميلاد", "ميلاد"], "عيد الغطاس": ["الغطاس"], "عيد البشارة": ["البشارة"],
    "عيد الصليب": ["عيد الصليب"], "عيد التجلي": ["التجلي", "التجلى"], "عيد العنصرة (الخمسين)": ["العنصرة", "الخمسين"],
    "عيد الصعود": ["الصعود"], "أحد الشعانين": ["الشعانين"], "الجمعة العظيمة": ["الجمعة العظيمة"],
    "بداية الصوم الكبير": ["الصوم الكبير"], "عيد النيروز (رأس السنة القبطية)": ["النيروز"],
    "عيد دخول المسيح أرض مصر": ["دخول المسيح", "أرض مصر"], "عيد صعود جسد السيدة العذراء": ["صعود جسد", "العذراء"],
}


def occasions_for_year(year: int) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for name, (m, d) in FIXED_FEASTS.items():
        for cy in (year - 284, year - 283):
            g = coptic_to_gregorian(cy, m, d)
            if g.year == year:
                out[name] = {"date": g, "kind": "ثابت", "coptic": f"{d}/{m}"}
    easter = orthodox_easter(year)
    for name, off in MOVABLE_OFFSETS.items():
        out[name] = {"date": easter + dt.timedelta(days=off), "kind": "متحرك (يعتمد على القيامة)", "coptic": ""}
    return out


def find_occasion(text: str, year: Optional[int] = None) -> Optional[dict]:
    year = year or dt.date.today().year
    norm = text or ""
    for name, kws in OCCASION_KEYWORDS.items():
        if any(k in norm for k in kws):
            info = occasions_for_year(year).get(name)
            if info:
                return {"name": name, **info, "year": year,
                        "note": "التاريخ محسوب برمجيًا — يُراجع مع التقويم الكنسي الرسمي."}
    return None


# ════════════════════════════════════════════════════════════════════════════
# 3. KNOWLEDGE: registry, ingestion, chunking, retrieval
# ════════════════════════════════════════════════════════════════════════════
START = "https://st-takla.org"
REGISTERED_SOURCES = [
    {"id": "takla-feasts", "name": "St-Takla — الأعياد والمناسبات", "kind": "feasts",
     "url": START + "/Feastes-%26-Special-Events/Coptic-Orthodox-Feasts-Events-Fasting_.html"},
    {"id": "takla-calendar", "name": "St-Takla — التقويم القبطي", "kind": "calendar",
     "url": START + "/Feastes-%26-Special-Events/00-St-Takla.org_Orthodox-Monthly-Coptic-Calendar/00-Al-Natiga-Al-Keptia-Current-Year-index.html"},
    {"id": "takla-sunday", "name": "St-Takla — مناهج مدارس الأحد", "kind": "sunday_school",
     "url": START + "/Coptic-Service-Corner/Sunday-School-Syllabus/Manhag-Al-Khedma-00-index.html"},
    {"id": "takla-belief", "name": "St-Takla — العقيدة الأرثوذكسية", "kind": "doctrine",
     "url": START + "/Coptic-Faith-Creed-Dogma/Coptic-Orthodox-Belief-00-index_.html"},
    {"id": "takla-rites", "name": "St-Takla — الطقوس", "kind": "doctrine",
     "url": START + "/Coptic-Faith-Creed-Dogma/Coptic-Rite-n-Ritual-Taks-Al-Kanisa/Orthodox-Rites-n-Rituals-000-index.html"},
    {"id": "takla-saints", "name": "St-Takla — القديسون", "kind": "saints",
     "url": START + "/Saints/Coptic-Saint-Hagiography-Kediseen-00-index.html"},
    {"id": "takla-home", "name": "St-Takla — الصفحة الرئيسية", "kind": "general",
     "url": START + "/P-1_.html"},
    {"id": "orsozoxi", "name": "Orsozoxi.net", "kind": "general", "url": "https://www.orsozoxi.net"},
]
TOPIC_KEYWORDS = {
    "feasts": ["عيد", "أعياد"], "fasts": ["صوم", "الصوم"], "saints": ["قديس", "شهيد", "الأنبا", "القديسة"],
    "biblical": ["الكتاب المقدس", "إنجيل", "الإنجيل", "سفر", "معجزة"], "doctrine": ["عقيدة", "إيمان", "سر", "طقس"],
    "sunday_school": ["مدارس الأحد", "منهج", "درس"],
}
ORIGIN_AR = {"curriculum": "منهج الكنيسة المرفوع", "web": "مرجع إلكتروني"}


def ar_norm(s: str) -> str:
    s = re.sub(r"[\u064B-\u0652\u0640]", "", s or "")
    s = re.sub("[إأآ]", "ا", s).replace("ى", "ي").replace("ة", "ه")
    return s.lower()


def tokens(s: str) -> List[str]:
    out = []
    for w in re.findall(r"[\w]+", ar_norm(s)):
        if w.startswith("ال") and len(w) > 4:
            w = w[2:]
        if len(w) > 1:
            out.append(w)
    return out


@dataclass
class Chunk:
    text: str
    title: str
    url: str
    canonical: str
    retrieved_at: str
    topic: str
    section: str
    origin: str = "web"
    verification: str = "غير مؤكد — مرجع إلكتروني"


def classify_topic(text: str, default: str = "general") -> str:
    for t, kws in TOPIC_KEYWORDS.items():
        if any(k in text for k in kws):
            return t
    return default


def parse_html(html: str, url: str) -> dict:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "noscript", "form"]):
        t.decompose()
    title = (soup.title.get_text(strip=True) if soup.title else "") or url
    can = soup.find("link", rel="canonical")
    canonical = urljoin(url, can["href"]) if can and can.get("href") else url
    links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]
    lines = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "div", "td"]):
        if el.find(["p", "div", "li", "td"]) and el.name in ("div", "td"):
            continue
        t = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if re.search(r"[\u0600-\u06FF]", t) and (len(t) >= 25 or el.name.startswith("h")):
            lines.append((el.name, t))
    return {"title": title, "canonical": canonical, "lines": lines, "links": links}


def make_chunks(parsed: dict, url: str, kind: str, now: str, max_chars: int = 700) -> List[Chunk]:
    chunks, buf, section = [], "", parsed["title"]

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(Chunk(buf.strip(), parsed["title"], url, parsed["canonical"], now,
                                classify_topic(section + " " + buf, kind), section))
        buf = ""
    for name, t in parsed["lines"]:
        if name.startswith("h"):
            flush()
            section = t[:120]
            continue
        if len(buf) + len(t) > max_chars:
            flush()
        buf += (" " if buf else "") + t
    flush()
    return chunks


class KnowledgeBase:
    """Web pages are cached on disk (public data only). Uploaded curriculum stays in memory per session."""

    def __init__(self, cache_path: str = KB_CACHE_PATH, http=requests, sleep=time.sleep, delay: float = 1.5):
        self.cache_path, self.http, self.sleep, self.delay = cache_path, http, sleep, delay
        self.pages: Dict[str, dict] = {}
        self._robots: Dict[str, Optional[robotparser.RobotFileParser]] = {}
        self.load()

    def load(self):
        try:
            with open(self.cache_path, encoding="utf-8") as f:
                self.pages = json.load(f)
        except Exception:
            self.pages = {}

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.pages, f, ensure_ascii=False)
        except Exception as e:
            log.warning("could not save KB cache: %s", e)

    def allowed(self, url: str) -> Tuple[bool, str]:
        p = urlparse(url)
        host = f"{p.scheme}://{p.netloc}"
        if host not in self._robots:
            rp = robotparser.RobotFileParser()
            try:
                r = self.http.get(host + "/robots.txt", timeout=15, headers={"User-Agent": USER_AGENT})
                if r.status_code == 200:
                    rp.parse(r.text.splitlines())
                    self._robots[host] = rp
                elif 400 <= r.status_code < 500:
                    self._robots[host] = None
                else:
                    return False, f"تعذر قراءة robots.txt (رمز {r.status_code})"
            except Exception:
                return False, "تعذر قراءة robots.txt (شبكة)"
        rp = self._robots[host]
        if rp is not None and not rp.can_fetch(USER_AGENT, url):
            return False, "robots.txt يمنع الجلب الآلي لهذه الصفحة"
        return True, ""

    def _record(self, url, kind, name, status, error="", parsed=None, chunks=None):
        self.pages[url] = {"url": url, "kind": kind, "name": name, "status": status, "error": error,
                           "title": (parsed or {}).get("title", ""), "canonical": (parsed or {}).get("canonical", url),
                           "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                           "chunks": [asdict(c) for c in (chunks or [])]}

    def fetch_page(self, url: str, kind: str, name: str) -> Optional[dict]:
        ok, why = self.allowed(url)
        if not ok:
            self._record(url, kind, name, "unavailable", why)
            return None
        try:
            r = self.http.get(url, timeout=25, headers={"User-Agent": USER_AGENT})
            if r.status_code != 200:
                self._record(url, kind, name, "unavailable", f"رمز الاستجابة {r.status_code}")
                return None
            if not r.encoding or r.encoding.lower() == "iso-8859-1":
                r.encoding = r.apparent_encoding or "utf-8"
            parsed = parse_html(r.text, url)
            now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            chunks = make_chunks(parsed, url, kind, now)
            if not chunks:
                self._record(url, kind, name, "unavailable", "لا يوجد نص عربي مفيد في الصفحة", parsed)
                return None
            self._record(url, kind, name, "ok", "", parsed, chunks)
            return parsed
        except Exception as e:
            self._record(url, kind, name, "unavailable", redact(str(e))[:150])
            return None
        finally:
            self.sleep(self.delay)

    def ingest(self, src: dict, discover: bool = True, max_pages: int = 5) -> int:
        """Fetch the start page + a *bounded* number of same-section links. Not a full crawl."""
        parsed = self.fetch_page(src["url"], src["kind"], src["name"])
        n = 1 if parsed else 0
        if parsed and discover and max_pages > 1:
            base_dir = src["url"].rsplit("/", 1)[0] + "/"
            seen = {src["url"]}
            for link in parsed["links"]:
                link = link.split("#")[0]
                if len(seen) >= max_pages:
                    break
                if link in seen or not link.startswith(base_dir) or not re.search(r"\.html?$", link):
                    continue
                seen.add(link)
                if self.fetch_page(link, src["kind"], src["name"]):
                    n += 1
        self.save()
        return n

    def remove(self, url: str):
        self.pages.pop(url, None)
        self.save()

    def is_stale(self, page: dict) -> bool:
        try:
            t = dt.datetime.fromisoformat(page["retrieved_at"])
            return (dt.datetime.now(dt.timezone.utc) - t).days > STALE_DAYS
        except Exception:
            return True

    def web_chunks(self) -> List[Chunk]:
        return [Chunk(**c) for p in self.pages.values() if p["status"] == "ok" for c in p["chunks"]]

    def stats(self) -> dict:
        ok = sum(1 for p in self.pages.values() if p["status"] == "ok")
        return {"ok": ok, "failed": len(self.pages) - ok,
                "chunks": sum(len(p["chunks"]) for p in self.pages.values())}


def local_doc_to_chunks(name: str, data: bytes) -> Tuple[List[Chunk], str]:
    """PDF / DOCX curriculum -> chunks (kept in session memory only)."""
    text = ""
    if name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        text = "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    elif name.lower().endswith(".docx"):
        import docx
        d = docx.Document(io.BytesIO(data))
        text = "\n".join(p.text for p in d.paragraphs)
    else:
        return [], "نوع الملف غير مدعوم (PDF أو DOCX فقط)."
    if len(re.findall(r"[\u0600-\u06FF]", text)) < 50:
        return [], "لم يُستخرج نص عربي كافٍ (قد يكون الملف ممسوحًا ضوئيًا)."
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    parsed = {"title": name, "canonical": f"local:{name}",
              "lines": [("p", ln.strip()) for ln in text.splitlines() if len(ln.strip()) >= 20]}
    ch = make_chunks(parsed, f"local:{name}", "sunday_school", now)
    for c in ch:
        c.origin, c.verification = "curriculum", "منهج معتمد مرفوع من المشغّل"
    return ch, ""


def retrieve(query: str, chunks: List[Chunk], k: int = 5, min_score: float = 0.5) -> List[Chunk]:
    """BM25 over normalized Arabic tokens. Curriculum chunks get a boost. k=5 keeps prompts lean for free models."""
    q = tokens(query)
    if not q or not chunks:
        return []
    docs = [tokens(c.text + " " + c.section) for c in chunks]
    N, avg = len(docs), (sum(len(d) for d in docs) / max(len(docs), 1)) or 1
    df = Counter(t for d in docs for t in set(d))
    scored = []
    for c, d in zip(chunks, docs):
        tf = Counter(d)
        s = 0.0
        for t in set(q):
            if tf[t]:
                idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(d) / avg))
        if c.origin == "curriculum":
            s *= 1.5
        if s >= min_score:
            scored.append((s, c))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:k]]


def build_context(chunks: List[Chunk], occasion: Optional[dict]) -> str:
    parts = []
    if occasion:
        parts.append(f"مناسبة كنسية مرصودة: {occasion['name']} — {occasion['kind']} — تاريخ {occasion['year']}: "
                     f"{occasion['date'].isoformat()} (محسوب برمجيًا، يحتاج مراجعة).")
    if not chunks:
        parts.append("لا توجد مراجع مسترجعة لهذا الطلب. لا تنسب أي معلومة لمرجع، وضع كل معلومة عقائدية أو تاريخية غير مؤكدة في review_flags.")
    for i, c in enumerate(chunks, 1):
        parts.append(f"[م{i}] ({ORIGIN_AR.get(c.origin, c.origin)}) {c.title} — {c.url}\n{c.text[:700]}")
    return "\n\n".join(parts)


def invalid_citations(obj: Any, n_refs: int) -> List[int]:
    txt = json.dumps(obj, ensure_ascii=False)
    nums = {int(x) for x in re.findall(r"م(\d+)", txt)}
    return sorted(n for n in nums if n < 1 or n > n_refs)


def unverified_quotes(text: str, chunks: List[Chunk], min_words: int = 5) -> List[str]:
    corpus = ar_norm(" ".join(c.text for c in chunks))
    bad = []
    for q in re.findall(r"[«\"“]([^»\"”]{10,300})[»\"”]", text):
        if len(q.split()) >= min_words and ar_norm(q).strip() not in corpus:
            bad.append(q.strip())
    return bad


# ════════════════════════════════════════════════════════════════════════════
# 4. SCHEMAS + VALIDATION GATE
# ════════════════════════════════════════════════════════════════════════════
class Analysis(BaseModel):
    topic: str
    is_church_occasion: bool = False
    occasion_name: str = ""
    key_message: str
    objectives: List[str]
    bible_refs: List[str] = []
    review_flags: List[str] = []
    refs: List[int] = []


class Activity(BaseModel):
    title: str
    description: str
    materials: List[str] = []
    steps: List[str] = []
    minutes: int = 10
    safety_note: str = ""
    source_tag: str = "إبداع"
    refs: List[int] = []


class Activities(BaseModel):
    items: List[Activity]


class Poster(BaseModel):
    title: str
    layout: str
    materials: List[str]
    construction_steps: List[str]
    teaching_script: str
    children_participation: str
    discussion_questions: List[str]
    refs: List[int] = []


class LessonSection(BaseModel):
    heading: str
    body: str
    source_tag: str = "شرح"
    refs: List[int] = []


class Lesson(BaseModel):
    title: str
    objectives: List[str]
    sections: List[LessonSection]
    memory_verse: str = ""
    conclusion: str = ""
    review_flags: List[str] = []


class Exercise(BaseModel):
    kind: str
    prompt: str
    items: List[str] = []
    answer: str = ""


class Worksheet(BaseModel):
    title: str
    exercises: List[Exercise]


class QuizQ(BaseModel):
    question: str
    options: List[str]
    answer_index: int
    explanation: str = ""

    @field_validator("answer_index")
    @classmethod
    def _in_range(cls, v, info):
        opts = info.data.get("options", [])
        if not (0 <= v < max(len(opts), 1)):
            raise ValueError("answer_index out of range")
        return v


class Quiz(BaseModel):
    title: str
    questions: List[QuizQ]


class Issue(BaseModel):
    severity: str  # critical | major | minor
    text: str


class Validation(BaseModel):
    biblical_alignment_ok: bool
    safety_ok: bool
    age_appropriate: bool = True
    confidence: float = 0.0
    issues: List[Issue] = []
    unsupported_claims: List[str] = []


UNSAFE_WORDS = ["نار", "لهب", "كبريت", "ولاعة", "سكين", "مشرط", "مسمار", "ماء مغلي", "كهرباء", "مادة كيميائية", "بنزين", "مسدس", "سلاح"]


def local_safety_scan(text: str) -> List[Issue]:
    return [Issue(severity="critical", text=f"كلمة تدل على خطر محتمل: «{w}» — يلزم تعديل النشاط أو مراجعته.")
            for w in UNSAFE_WORDS if w in text]


def gate(v: Validation, extra: Optional[List[Issue]] = None) -> Tuple[str, List[Issue]]:
    """Critical failures can never be overridden by a high confidence score."""
    issues = list(v.issues) + list(extra or [])
    if (not v.safety_ok) or (not v.biblical_alignment_ok) or any(i.severity == "critical" for i in issues):
        return "blocked", issues
    if any(i.severity == "major" for i in issues) or v.unsupported_claims or not v.age_appropriate:
        return "needs_review", issues
    return "ok", issues


# ════════════════════════════════════════════════════════════════════════════
# 5. PROMPTS + TASK ENGINE
# ════════════════════════════════════════════════════════════════════════════
SYSTEM_BASE = (
    "أنت مساعد إبداعي لخدام مدارس الأحد في الكنيسة القبطية الأرثوذكسية، والفئة المستهدفة أطفال 8–10 سنوات "
    "(تالتة ورابعة ابتدائي). المحتوى التعليمي بالعربية المبسطة المناسبة للطفل. "
    "قواعد صارمة: (1) منهج الكنيسة المرفوع له الأولوية على أي تفسير عام. (2) لا تخترع آيات أو اقتباسات أو مصادر؛ "
    "اقتبس فقط من المراجع المرفقة، واذكر رقم المرجع بالشكل [م1]. (3) ميّز بين: منهج / مرجع / شرح / إبداع / مراجعة عبر source_tag. "
    "(4) إن لم تتأكد من معلومة عقائدية أو تاريخية ضعها في review_flags. (5) إن اختلفت المراجع لا تدمجها؛ اذكر الاختلاف. "
    "(6) الأنشطة آمنة تمامًا للأطفال: بلا نار أو أدوات حادة أو مواد خطرة. (7) لا تذكر الأعياد إلا إذا كانت ذات صلة بالطلب. "
    "(8) كن موجزًا ومحدد البنية."
)
CHAT_SYSTEM = SYSTEM_BASE + " في المحادثة تكلم مع الخادم بالعامية المصرية الودودة، واقترح أفكارًا عملية وبسيطة الخامات، واسأل سؤالًا واحدًا فقط عند الحاجة."

TASKS: Dict[str, Tuple[type, str, str]] = {
    "analysis": (Analysis, "تحليل الدرس", "حلل الدرس: الموضوع، هل هو مناسبة كنسية، الرسالة الأساسية، أهداف التعلم، الشواهد (من المراجع فقط)."),
    "lesson": (Lesson, "الدرس الكامل", "اكتب خطة درس كاملة (مقدمة، قصة/شرح، تطبيق، ختام) بأقسام واضحة وآية حفظ إن وردت في المراجع."),
    "activities": (Activities, "الأنشطة", "اقترح 4 أنشطة وألعاب إبداعية متنوعة وآمنة (حركية، فنية، جماعية، تأملية) بخامات بسيطة."),
    "poster": (Poster, "اللوحة التعليمية", "صمم لوحة/وسيلة إيضاح تفاعلية بخامات بسيطة: التصميم، الخامات، خطوات التنفيذ، سكريبت الشرح، مشاركة الأطفال، أسئلة نقاش."),
    "worksheet": (Worksheet, "ورقة العمل", "أنشئ ورقة عمل قابلة للطباعة من 5 تمارين متنوعة (توصيل، إكمال، ترتيب، رسم) مع الإجابة النموذجية في حقل answer."),
    "quiz": (Quiz, "الاختبار", "أنشئ اختبارًا من 6 أسئلة اختيار من متعدد بإجابة واحدة صحيحة وشرح قصير."),
}
TASK_ICON = {"analysis": "🔎", "lesson": "📖", "activities": "🎲", "poster": "🖼️", "worksheet": "📝", "quiz": "❓"}
STATE_KEYS_FOR_CONTEXT = ["analysis", "lesson", "activities", "poster"]


def new_state() -> dict:
    return {"input": "", "curriculum_chunks": [], "curriculum_names": [], "occasion": None,
            **{k: None for k in TASKS}, "validation": None, "gate": None, "refs": {}, "chat": []}


def retrieve_for(state: dict, kb: KnowledgeBase, extra: str = "") -> Tuple[List[Chunk], Optional[dict]]:
    q = f"{state['input']} {extra}"
    occ = find_occasion(q)
    pool = list(state["curriculum_chunks"]) + kb.web_chunks()
    return retrieve(q + (" " + occ["name"] if occ else ""), pool), occ


def run_task(name: str, state: dict, kb: KnowledgeBase, provider: LLMProvider, extra: str = "") -> Any:
    """Regenerates ONLY the requested component; everything else in `state` is untouched."""
    schema, _, instr = TASKS[name]
    chunks, occ = retrieve_for(state, kb, extra)
    state["occasion"] = occ or state.get("occasion")
    existing = {k: state[k] for k in STATE_KEYS_FOR_CONTEXT if state.get(k) and k != name}
    user = (f"طلب الخادم/الدرس:\n{state['input']}\n\n"
            f"المراجع المسترجعة:\n{build_context(chunks, occ)}\n\n"
            + (f"مكونات الدرس الحالية (حافظ على اتساقها):\n{json.dumps(existing, ensure_ascii=False)[:4000]}\n\n" if existing else "")
            + (f"ملاحظة إضافية: {extra}\n\n" if extra else "") + f"المهمة: {instr}")
    obj = provider.structured(schema, [{"role": "system", "content": SYSTEM_BASE}, {"role": "user", "content": user}],
                              max_tokens=3500)
    data = obj.model_dump()
    bad = invalid_citations(data, len(chunks))
    if bad:
        data.setdefault("review_flags", []).append(f"إشارات مراجع غير موجودة {bad} — تم تجاهلها وتحتاج مراجعة.")
    state[name] = data
    state["refs"][name] = [asdict(c) for c in chunks]
    state["validation"], state["gate"] = None, None
    return data


def run_validation(state: dict, provider: LLMProvider) -> dict:
    comps = {k: state[k] for k in TASKS if state.get(k)}
    if not comps:
        raise LLMError("empty_state", "لا يوجد محتوى للمراجعة بعد.")
    chunks = [Chunk(**c) for refs in state["refs"].values() for c in refs]
    flat = json.dumps(comps, ensure_ascii=False)
    user = (f"راجع المحتوى التالي لدرس مدارس الأحد (8–10 سنوات): مطابقة لتعليم الكنيسة، سلامة الأنشطة، مناسبة العمر، "
            f"وادعاءات غير مدعومة بالمراجع. كن صارمًا.\nالمراجع:\n{build_context(chunks[:8], state.get('occasion'))}\n\nالمحتوى:\n{flat[:8000]}")
    v = provider.structured(Validation, [{"role": "system", "content": SYSTEM_BASE}, {"role": "user", "content": user}], temperature=0.2)
    extra = local_safety_scan(flat)
    # Quote check runs in the BACKEND only: logged and stored, never shown in the UI or used by the gate
    # (it flags every «...» phrase, including the model's own activity text, so it was mostly noise).
    notes = [q[:120] for q in unverified_quotes(flat, chunks)]
    if notes:
        log.info("backend: %d unverified quoted phrases (hidden from UI)", len(notes))
    status, issues = gate(v, extra)
    state["validation"] = {**v.model_dump(), "issues": [i.model_dump() for i in issues], "backend_notes": notes}
    state["gate"] = status
    return state["validation"]


# ════════════════════════════════════════════════════════════════════════════
# 6. RENDER BLOCKS + EXPORTS (DOCX / PDF, Arabic RTL)
# ════════════════════════════════════════════════════════════════════════════
def component_blocks(name: str, d: dict, answers: bool = False) -> List[Tuple[str, str]]:
    B: List[Tuple[str, str]] = []
    tag = lambda x: f"[{x.get('source_tag')}] " if x.get("source_tag") else ""
    if name == "analysis":
        B += [("h2", "تحليل الدرس"), ("p", f"الموضوع: {d['topic']}"), ("p", f"الرسالة الأساسية: {d['key_message']}")]
        B += [("li", o) for o in d["objectives"]]
        if d.get("bible_refs"):
            B += [("h3", "الشواهد")] + [("li", x) for x in d["bible_refs"]]
    elif name == "lesson":
        B += [("h1", d["title"]), ("h3", "أهداف الدرس")] + [("li", o) for o in d["objectives"]]
        for s in d["sections"]:
            B += [("h2", s["heading"]), ("p", tag(s) + s["body"])]
        if d.get("memory_verse"):
            B += [("h3", "آية الحفظ"), ("p", d["memory_verse"])]
        if d.get("conclusion"):
            B += [("h3", "الختام"), ("p", d["conclusion"])]
    elif name == "activities":
        B.append(("h1", "الأنشطة"))
        for a in d["items"]:
            B += [("h2", f"{a['title']} ({a['minutes']} دقيقة)"), ("p", tag(a) + a["description"])]
            if a["materials"]:
                B.append(("p", "الخامات: " + "، ".join(a["materials"])))
            B += [("li", s) for s in a["steps"]]
            if a["safety_note"]:
                B.append(("p", "تنبيه أمان: " + a["safety_note"]))
    elif name == "poster":
        B += [("h1", d["title"]), ("h3", "التصميم"), ("p", d["layout"]), ("h3", "الخامات")]
        B += [("li", m) for m in d["materials"]] + [("h3", "خطوات التنفيذ")] + [("li", s) for s in d["construction_steps"]]
        B += [("h3", "سكريبت الشرح"), ("p", d["teaching_script"]), ("h3", "مشاركة الأطفال"), ("p", d["children_participation"]),
              ("h3", "أسئلة للنقاش")] + [("li", q) for q in d["discussion_questions"]]
    elif name == "worksheet":
        B += [("h1", d["title"]), ("p", "الاسم: ........................   التاريخ: ................")]
        for i, e in enumerate(d["exercises"], 1):
            B += [("h3", f"تمرين {i} ({e['kind']})"), ("p", e["prompt"])] + [("li", x) for x in e["items"]]
            if answers and e.get("answer"):
                B.append(("p", "الإجابة: " + e["answer"]))
    elif name == "quiz":
        B.append(("h1", d["title"]))
        for i, q in enumerate(d["questions"], 1):
            B.append(("h3", f"{i}. {q['question']}"))
            B += [("li", f"{'أبجد'[j] if j < 4 else j+1}) {o}") for j, o in enumerate(q["options"])]
            if answers:
                B.append(("p", f"الإجابة الصحيحة: {q['options'][q['answer_index']]} — {q['explanation']}"))
    return B


def refs_blocks(state: dict) -> List[Tuple[str, str]]:
    seen, B = set(), [("h2", "المراجع المسترجعة")]
    for refs in state["refs"].values():
        for c in refs:
            if c["url"] not in seen:
                seen.add(c["url"])
                B.append(("li", f"{c['title']} — {c['url']} — تاريخ الجلب {c['retrieved_at'][:10]} — {c['verification']}"))
    return B if len(B) > 1 else []


def export_sets(state: dict) -> Dict[str, List[Tuple[str, str]]]:
    lesson = []
    for k in ("lesson", "activities", "poster"):
        if state.get(k):
            lesson += component_blocks(k, state[k])
    lesson += refs_blocks(state)
    if state.get("gate") in ("blocked", "needs_review"):
        lesson.insert(0, ("p", "تنبيه: هذا المحتوى يحتاج مراجعة بشرية قبل الاستخدام."))
    student = []
    for k in ("worksheet", "quiz"):
        if state.get(k):
            student += component_blocks(k, state[k], answers=False)
    key = []
    for k in ("worksheet", "quiz"):
        if state.get(k):
            key += component_blocks(k, state[k], answers=True)
    out = {"الدرس": lesson, "أوراق الأطفال": student, "مفتاح الإجابة (للخادم فقط)": key}
    return {k: v for k, v in out.items() if v}


def blocks_to_docx(blocks: List[Tuple[str, str]]) -> bytes:
    import docx
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    d = docx.Document()
    for kind, text in blocks:
        p = d.add_paragraph(style="List Bullet" if kind == "li" else None)
        run = p.add_run(text)
        run.bold = kind.startswith("h")
        run.font.size = docx.shared.Pt({"h1": 20, "h2": 16, "h3": 14}.get(kind, 12))
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        ppr = p._p.get_or_add_pPr()
        ppr.append(OxmlElement("w:bidi"))
        rpr = run._r.get_or_add_rPr()
        rpr.append(OxmlElement("w:rtl"))
        rfonts = rpr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = OxmlElement("w:rFonts")
            rpr.append(rfonts)
        rfonts.set(qn("w:cs"), "Arial")
    bio = io.BytesIO()
    d.save(bio)
    return bio.getvalue()


def find_arabic_font() -> Optional[str]:
    cands = [os.environ.get("ARABIC_FONT_PATH", ""), "fonts/Amiri-Regular.ttf", "fonts/NotoNaskhArabic-Regular.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
             "C:/Windows/Fonts/arial.ttf", "/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"]
    return next((p for p in cands if p and os.path.exists(p)), None)


def blocks_to_pdf(blocks: List[Tuple[str, str]]) -> bytes:
    import arabic_reshaper
    from bidi.algorithm import get_display
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
    fp = find_arabic_font()
    if not fp:
        raise RuntimeError("لا يوجد خط عربي. ضع ملف خط (مثل Amiri-Regular.ttf) في مجلد fonts/ أو حدد ARABIC_FONT_PATH.")
    pdfmetrics.registerFont(TTFont("AR", fp))
    W, H = A4
    margin = 50
    bio = io.BytesIO()
    c = canvas.Canvas(bio, pagesize=A4)
    y = H - margin
    shape = lambda s: get_display(arabic_reshaper.reshape(s))
    for kind, text in blocks:
        size = {"h1": 20, "h2": 16, "h3": 14}.get(kind, 12)
        prefix = "• " if kind == "li" else ""
        words, lines, cur = (prefix + text).split(), [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if pdfmetrics.stringWidth(shape(trial), "AR", size) <= W - 2 * margin:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        for ln in lines:
            if y < margin + size:
                c.showPage()
                y = H - margin
            c.setFont("AR", size)
            c.drawRightString(W - margin, y, shape(ln))
            y -= size * 1.6
        y -= 4
    c.save()
    return bio.getvalue()


# ════════════════════════════════════════════════════════════════════════════
# 7. STREAMLIT UI  (RTL, parchment + navy + gold theme)
# ════════════════════════════════════════════════════════════════════════════
CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
:root{--navy:#14213d;--navy2:#2a3f6f;--gold:#c8963e;--gold2:#e6b95c;--cream:#fbf7ef;--sand:#f1e8d6;
      --ink:#1f2a44;--ok:#2e7d5b;--bad:#b4443c;--card:#ffffff;--line:#e6dcc6}
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, label{
  font-family:'Cairo','Segoe UI',Tahoma,sans-serif !important}
.stApp{direction:rtl;text-align:right;color:var(--ink);
  background:radial-gradient(1200px 500px at 90% -10%,#fff7e0 0,transparent 60%),
             linear-gradient(180deg,var(--cream) 0%,#f4ecdc 100%)}
.block-container{padding-top:1.4rem;max-width:1180px}
h1,h2,h3,h4{color:var(--navy);font-weight:800}
.hero{position:relative;overflow:hidden;border-radius:24px;padding:30px 34px;margin-bottom:20px;color:#fff;
  background:linear-gradient(135deg,var(--navy) 0%,var(--navy2) 62%,#8a6a2c 150%);
  box-shadow:0 14px 34px rgba(20,33,61,.28)}
.hero:before{content:"✝";position:absolute;left:-10px;top:-40px;font-size:230px;opacity:.07;color:#fff}
.hero h1{color:#fff;margin:0 0 6px;font-size:2rem}
.hero p{margin:0;opacity:.88;font-size:1.02rem}
.hero .chips{margin-top:14px;display:flex;gap:8px;flex-wrap:wrap}
.chip{background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.28);padding:4px 12px;border-radius:999px;font-size:.85rem}
.ecard{background:var(--card);border:1.5px solid var(--line);border-radius:18px;padding:14px 16px;min-height:112px;
  box-shadow:0 4px 14px rgba(20,33,61,.06);transition:.2s}
.ecard.sel{border-color:var(--gold);box-shadow:0 6px 20px rgba(200,150,62,.28);background:linear-gradient(180deg,#fffdf6,#fff)}
.ecard.off{opacity:.62;background:#f6f2e8}
.ecard .t{font-weight:800;color:var(--navy);font-size:1.08rem}
.ecard .n{font-size:.82rem;color:#5b6684;margin-top:4px}
.pill{display:inline-block;font-size:.72rem;font-weight:700;padding:2px 9px;border-radius:999px;margin-right:6px}
.pill.ok{background:#e4f4ec;color:var(--ok)} .pill.off{background:#f3e3e1;color:var(--bad)}
.stButton>button,.stDownloadButton>button{border-radius:14px;border:1.5px solid var(--navy);background:#fff;color:var(--navy);
  font-weight:700;padding:.45rem 1rem;transition:.18s}
.stButton>button:hover,.stDownloadButton>button:hover{background:var(--navy);color:#fff;transform:translateY(-1px);
  box-shadow:0 6px 16px rgba(20,33,61,.22)}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,var(--gold),#a97420);border:none;color:#fff;
  font-size:1.05rem;padding:.7rem 1.2rem;box-shadow:0 8px 20px rgba(200,150,62,.38)}
.stButton>button[kind="primary"]:hover{filter:brightness(1.07);background:linear-gradient(135deg,var(--gold2),#b4812c)}
.stTextArea textarea,.stTextInput input{direction:rtl;text-align:right;border-radius:14px !important;
  border:1.5px solid var(--line) !important;background:#fff !important;color:var(--ink) !important}
.stTextArea textarea:focus,.stTextInput input:focus{border-color:var(--gold) !important;box-shadow:0 0 0 3px rgba(200,150,62,.2) !important}
.stChatInput textarea{direction:rtl;text-align:right}
div[data-testid="stExpander"]{background:var(--card);border:1.5px solid var(--line) !important;border-radius:18px !important;
  box-shadow:0 4px 14px rgba(20,33,61,.05);margin-bottom:10px;overflow:hidden}
div[data-testid="stExpander"] summary{font-weight:800;color:var(--navy)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#fffaf0,#f3e9d4);border-left:1px solid var(--line);overflow:hidden}
[data-testid="stSidebarContent"]{direction:rtl;text-align:right}
[data-testid="stSidebar"][aria-expanded="true"]{min-width:300px}
/* collapsed sidebar: hide its content so it can't squeeze into a narrow vertical strip */
[data-testid="stSidebar"][aria-expanded="false"]{min-width:0 !important;width:0 !important;border:none}
[data-testid="stSidebar"][aria-expanded="false"] *{visibility:hidden !important;width:0 !important;overflow:hidden !important}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:var(--navy)}
.stProgress>div>div>div>div{background:linear-gradient(90deg,var(--gold),var(--gold2))}
div[data-testid="stAlert"]{border-radius:14px}
.sec{display:flex;align-items:center;gap:10px;margin:26px 0 10px;font-weight:800;color:var(--navy);font-size:1.25rem}
.sec:after{content:"";flex:1;height:2px;background:linear-gradient(90deg,var(--gold),transparent);opacity:.55}
.badge{display:inline-block;padding:6px 16px;border-radius:999px;font-weight:800;font-size:.95rem}
.badge.ok{background:#e4f4ec;color:var(--ok)} .badge.rev{background:#fff1d6;color:#9a6a12} .badge.blk{background:#f8e0de;color:var(--bad)}
/* Arabic (RTL) text: right-aligned paragraphs, bullets on the right next to the text */
[data-testid="stMarkdownContainer"],[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,[data-testid="stMarkdownContainer"] h3,[data-testid="stMarkdownContainer"] h4{
  direction:rtl;text-align:right}
[data-testid="stMarkdownContainer"] ul,[data-testid="stMarkdownContainer"] ol{direction:rtl;text-align:right;
  padding-right:1.5rem;padding-left:0;margin-right:0;list-style-position:outside}
[data-testid="stMarkdownContainer"] li{direction:rtl;text-align:right;margin-bottom:.35rem}
[data-testid="stMarkdownContainer"] li::marker{color:var(--gold)}
div[data-testid="stExpander"] details,div[data-testid="stExpander"] summary{direction:rtl;text-align:right}
div[data-testid="stAlert"]{direction:rtl;text-align:right}
.foot{text-align:center;color:#7a829c;font-size:.8rem;margin:30px 0 6px}
</style>"""


def main_app():
    import streamlit as st
    st.set_page_config(page_title=APP_TITLE_AR, page_icon="✝️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    ss = st.session_state
    ss.setdefault("S", new_state())
    ss.setdefault("auto_fb", True)
    ss.setdefault("skip_done", True)
    if "kb" not in ss:
        ss["kb"] = KnowledgeBase()
    if "free_map" not in ss:
        try:
            ss["free_map"], ss["catalog_ok"] = pick_engine_models(free_models(fetch_catalog())), True
        except Exception:
            ss["free_map"], ss["catalog_ok"] = dict(EMPTY_MAP), False
    S, kb = ss["S"], ss["kb"]

    def keys() -> Tuple[str, str]:
        return (get_secret("OPENROUTER_API_KEY").strip(),
                (get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")).strip())

    def endpoints(choice: str, fallback: bool) -> List[Endpoint]:
        ok, gk_ = keys()
        return build_endpoints(choice, ss["free_map"], ok, gk_, get_secret("GEMINI_MODEL_ID", GEMINI_DEFAULT).strip(), fallback)

    def provider(choice: Optional[str] = None, fallback: Optional[bool] = None) -> LLMProvider:
        return LLMProvider(endpoints(choice or ss["engine"], ss["auto_fb"] if fallback is None else fallback),
                           site_url=get_secret("OPENROUTER_SITE_URL"), min_interval=FREE_MIN_INTERVAL,
                           on_model=lambda m: ss.__setitem__("last_model", m),
                           app_name=get_secret("OPENROUTER_APP_NAME", "Church Lesson Creative Teaching Agent"))

    def guarded(fn, spinner="جاري العمل..."):
        try:
            with st.spinner(spinner):
                return fn()
        except LLMError as e:
            st.error(e.msg_ar)
        except Exception as e:
            st.error("حدث خطأ غير متوقع: " + redact(str(e), "".join(keys()))[:250])
        return None

    ork, gk = keys()
    status = engine_status(ss["free_map"], ork, gk)
    if ss.get("engine") not in ENGINES:
        ss["engine"] = next((n for n in ENGINES if status[n][0]), "auto")

    # ---------------- sidebar ----------------
    with st.sidebar:
        st.markdown("### ⚙️ الإعدادات")
        st.caption(f"الإصدار: {BUILD}")
        st.checkbox("إن فشل المحرّك، جرّب المحركات الأخرى تلقائيًا", key="auto_fb")
        if st.button("🔄 تحديث قائمة النماذج المجانية"):
            try:
                ss["free_map"], ss["catalog_ok"] = pick_engine_models(free_models(fetch_catalog())), True
            except Exception:
                ss["catalog_ok"] = False
            st.rerun()
        if not ss.get("catalog_ok", True):
            st.warning("تعذر قراءة كتالوج OpenRouter؛ متاح فقط الموزّع التلقائي وGemini.")
        if st.button("🧪 اختبار كل المحركات"):
            for n in ENGINES:
                if not endpoints(n, False):
                    st.write(f"❌ {ENGINE_AR[n]} — {status[n][1]}")
                    continue
                try:
                    pv = provider(n, False)
                    pv.chat([{"role": "user", "content": "قل: تمام"}], max_tokens=20, temperature=0)
                    st.write(f"✅ {ENGINE_AR[n]} — {pv.last_model}")
                except LLMError as e:
                    st.write(f"❌ {ENGINE_AR[n]} — {e.msg_ar[:160]}")
        st.caption("الحد المجاني على OpenRouter ≈ 20 طلب/دقيقة، لذلك يبطّئ التطبيق طلباته عبره فقط.")
        st.divider()
        if st.button("🗑️ بدء درس جديد"):
            ss["S"] = new_state()
            st.rerun()

    # ---------------- hero ----------------
    st.markdown(f"""<div class="hero"><h1>✝️ {APP_TITLE_AR}</h1>
      <p>خطط درسك، أنشطتك، لوحتك التعليمية وأوراق الأطفال في دقائق — لأطفال 8–10 سنوات</p>
      <div class="chips"><span class="chip">🆓 نماذج مجانية فقط</span><span class="chip">📚 يعتمد على منهج الكنيسة</span>
      <span class="chip">🛡️ مراجعة عقيدة وأمان</span><span class="chip">📤 DOCX / PDF</span></div></div>""",
                unsafe_allow_html=True)

    # ---------------- engine selector ----------------
    st.markdown('<div class="sec">🧠 اختر المحرّك المجاني</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, n in zip(cols, ENGINES):
        ok, note = status[n]
        cls = ("sel " if ss["engine"] == n else "") + ("" if ok else "off")
        col.markdown(f"""<div class="ecard {cls}"><div class="t">{ENGINE_ICON[n]} {ENGINE_AR[n]}
          <span class="pill {'ok' if ok else 'off'}">{'جاهز' if ok else 'غير متاح'}</span></div>
          <div class="n">{note}</div></div>""", unsafe_allow_html=True)
    st.radio("المحرّك", ENGINES, key="engine", horizontal=True, format_func=lambda n: f"{ENGINE_ICON[n]} {ENGINE_AR[n]}",
             label_visibility="collapsed")
    if ss.get("last_model"):
        st.caption(f"آخر نموذج استجاب: {ss['last_model']}")
    has_engine = bool(endpoints(ss["engine"], ss["auto_fb"]))
    if has_engine and not status[ss["engine"]][0] and ss["auto_fb"]:
        st.info("هذا المحرّك غير متاح الآن — سيُستخدم أول محرّك متاح تلقائيًا.")
    if not has_engine:
        st.warning("لا يوجد محرّك جاهز: مفاتيح الخادم غير مضبوطة (OPENROUTER_API_KEY / GEMINI_API_KEY).")

    # ---------------- lesson input ----------------
    st.markdown('<div class="sec">📝 وصف الدرس</div>', unsafe_allow_html=True)
    S["input"] = st.text_area("وصف الدرس أو طلبك", value=S["input"], height=110, label_visibility="collapsed",
                              placeholder="عندي درس عن عيد الرسل. عايز لوحة وأنشطة لتالتة ورابعة ابتدائي.")
    can_run = has_engine and bool(S["input"].strip())
    occ = find_occasion(S["input"]) if S["input"] else None
    if occ:
        st.info(f"مناسبة مرصودة: **{occ['name']}** ({occ['kind']}) — {occ['date'].isoformat()} — {occ['note']}")

    ORDER = ["analysis", "lesson", "activities", "poster", "worksheet", "quiz"]

    def render(name):
        d = S.get(name)
        if not d:
            st.caption("لم يتم التوليد بعد.")
            return
        for kind, text in component_blocks(name, d, answers=(name == "quiz")):
            st.markdown({"h1": "## ", "h2": "### ", "h3": "#### ", "li": "- "}.get(kind, "") + text)
        if d.get("review_flags"):
            st.warning("يحتاج مراجعة:\n" + "\n".join("- " + f for f in d["review_flags"]))
        if S["refs"].get(name) is not None and not S["refs"][name]:
            st.warning("لم يُسترجع أي مرجع لهذا المكوّن — المحتوى إبداعي/عام ويحتاج مراجعة بشرية.")

    # ---------------- generate all (sequential + paced: safe for free rate limits) ----------------
    if st.button("✨ ولّد كل شيء: تحليل + درس + أنشطة + لوحة + ورقة عمل + اختبار + مراجعة",
                 type="primary", disabled=not can_run, key="gen_all", use_container_width=True):
        steps = ORDER + ["validation"]
        bar = st.progress(0.0, text="جاري البدء...")
        finished = True
        todo = [n for n in ORDER if not (S.get(n) and ss["skip_done"])]
        eps0 = endpoints(ss["engine"], ss["auto_fb"])
        fast = bool(eps0) and eps0[0].kind == "gemini" and len(todo) > 1   # Gemini has no 20/min cap -> run parts together
        if fast:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            errs, done = {}, len(ORDER) - len(todo)
            bar.progress(done / len(steps), text=f"⚡ توليد {len(todo)} أجزاء معًا ...")
            with ThreadPoolExecutor(max_workers=3) as ex:
                futs = {}
                for n in todo:
                    pv = provider()
                    pv.on_model = None            # worker threads never touch Streamlit state
                    futs[ex.submit(run_task, n, S, kb, pv)] = n
                for fut in as_completed(futs):
                    n = futs[fut]
                    done += 1
                    try:
                        fut.result()
                    except LLMError as e:
                        errs[n] = e.msg_ar
                    except Exception as e:
                        errs[n] = redact(str(e), "".join(keys()))[:200]
                    bar.progress(done / len(steps), text=f"اكتمل {done} من {len(ORDER)}: {TASKS[n][1]}")
            for n, m in errs.items():
                st.error(f"{TASKS[n][1]}: {m}")
            finished = not errs
            if finished:
                bar.progress(len(ORDER) / len(steps), text="جاري المراجعة ...")
                finished = guarded(lambda: run_validation(S, provider()), "جاري المراجعة...") is not None
            else:
                st.info("الأجزاء الناجحة محفوظة — اضغط الزر مرة أخرى لاستكمال الناقص فقط.")
        else:
            for i, name in enumerate(steps):
                label = "المراجعة" if name == "validation" else TASKS[name][1]
                if name != "validation" and S.get(name) and ss["skip_done"]:
                    bar.progress((i + 1) / len(steps), text=f"✓ {label} (موجود بالفعل)")
                    continue
                bar.progress(i / len(steps), text=f"الخطوة {i + 1} من {len(steps)}: {label} ...")
                if name == "validation":
                    res = guarded(lambda: run_validation(S, provider()), f"جاري {label}...")
                else:
                    res = guarded(lambda n=name: run_task(n, S, kb, provider()), f"جاري توليد {label}...")
                if res is None:
                    finished = False
                    st.info("توقف التوليد. الأجزاء الناجحة محفوظة — اضغط الزر مرة أخرى لاستكمال الناقص فقط.")
                    break
        bar.progress(1.0 if finished else 0.0, text="✅ اكتمل التوليد" if finished else "توقف التوليد")
    st.checkbox("عند إعادة الضغط، تخطَّ الأجزاء المولّدة بالفعل", key="skip_done")

    # ---------------- results ----------------
    st.markdown('<div class="sec">📦 المحتوى</div>', unsafe_allow_html=True)
    for name in ORDER:
        _, label, _ = TASKS[name]
        with st.expander(f"{TASK_ICON[name]} {label} {'✅' if S.get(name) else ''}", expanded=bool(S.get(name))):
            c1, c2 = st.columns([3, 1])
            extra = c1.text_input("ملاحظة للتعديل (اختياري)", key=f"x_{name}")
            if c2.button("🔄 إعادة توليد هذا الجزء فقط", key=f"b_{name}", disabled=not can_run):
                guarded(lambda n=name, e=extra: run_task(n, S, kb, provider(), e), f"جاري توليد {label}...")
                st.rerun()
            render(name)

    # ---------------- review ----------------
    st.markdown('<div class="sec">🛡️ المراجعة</div>', unsafe_allow_html=True)
    if st.button("إعادة المراجعة", disabled=not has_engine, key="re_validate"):
        guarded(lambda: run_validation(S, provider()), "جاري المراجعة...")
    if S.get("validation"):
        v = S["validation"]
        n = len(v["issues"]) + len(v["unsupported_claims"])
        if S["gate"] == "ok" and n == 0:
            st.markdown('<span class="badge ok">✅ اكتملت المراجعة — لا ملاحظات</span>', unsafe_allow_html=True)
        else:
            cls = "blk" if S["gate"] == "blocked" else "rev"
            st.markdown(f'<span class="badge {cls}">✅ اكتمل التوليد — {n} ملاحظة تحتاج مراجعة الخادم قبل الاستخدام</span>',
                        unsafe_allow_html=True)
            with st.expander(f"عرض ملاحظات المراجعة ({n})", expanded=False):
                st.caption("المراجعة بنموذج مجاني مساعدة فقط — المراجعة البشرية من الخادم المسؤول واجبة دائمًا.")
                for i in v["issues"]:
                    fn = {"critical": st.error, "major": st.warning, "high": st.warning}.get(i["severity"], st.info)
                    fn(f"[{i['severity']}] {i['text']}")
                for u in v["unsupported_claims"]:
                    st.warning("ادعاء غير مدعوم بالمراجع: " + u)
                st.info("لإصلاحها: أعد توليد الجزء المعني من زر «🔄 إعادة توليد» مع ملاحظة التعديل، ثم أعد المراجعة.")
    else:
        st.caption("أي إعادة توليد تُبطل نتيجة المراجعة السابقة.")

    # ---------------- export ----------------
    st.markdown('<div class="sec">📤 التصدير</div>', unsafe_allow_html=True)
    sets = export_sets(S)
    if S.get("gate") == "blocked":
        st.error("المحتوى مرفوض في المراجعة — التصدير يتضمن تنبيهًا. عدّله أولًا.")
    if not sets:
        st.caption("لا يوجد محتوى للتصدير بعد.")
    for title, blocks in sets.items():
        c1, c2, c3 = st.columns([2, 1, 1])
        c1.write(f"**{title}**")
        c2.download_button("DOCX", blocks_to_docx(blocks), f"{title}.docx", key=f"d_{title}")
        try:
            c3.download_button("PDF", blocks_to_pdf(blocks), f"{title}.pdf", key=f"p_{title}")
        except Exception as e:
            c3.caption("PDF: " + str(e)[:120])

    # ---------------- chat ----------------
    st.markdown('<div class="sec">💬 محادثة عصف ذهني</div>', unsafe_allow_html=True)
    chat_box = st.container()
    msg = st.chat_input("احكيلي عن الدرس وهنفكر مع بعض...", disabled=not has_engine)
    with chat_box:
        for m in S["chat"]:
            st.chat_message(m["role"]).markdown(m["content"])
        if msg:
            S["chat"].append({"role": "user", "content": msg})
            st.chat_message("user").markdown(msg)
            if not S["input"].strip():
                S["input"] = msg
            chunks, o = retrieve_for(S, kb, msg)
            hist = [{"role": m["role"], "content": m["content"]} for m in S["chat"][-8:]]
            sysmsg = CHAT_SYSTEM + "\n\nالمراجع المسترجعة:\n" + build_context(chunks, o)
            ans = guarded(lambda: provider().chat([{"role": "system", "content": sysmsg}] + hist), "...")
            if ans:
                S["chat"].append({"role": "assistant", "content": ans})
                st.chat_message("assistant").markdown(ans)

    # ---------------- sources ----------------
    with st.expander("📚 المراجع والمنهج"):
        st.subheader("منهج الكنيسة (المصدر الأساسي)")
        ups = st.file_uploader("ارفع دروسًا معتمدة PDF أو DOCX", type=["pdf", "docx"], accept_multiple_files=True)
        for f in ups or []:
            if f.name not in S["curriculum_names"]:
                ch, err = local_doc_to_chunks(f.name, f.getvalue())
                if err:
                    st.error(f"{f.name}: {err}")
                else:
                    S["curriculum_chunks"] += ch
                    S["curriculum_names"].append(f.name)
                    st.success(f"{f.name}: {len(ch)} مقطع")
        st.caption("المنهج المرفوع يبقى في ذاكرة هذه الجلسة فقط ولا يُحفظ على القرص.")
        st.divider()
        st.subheader("المراجع الإلكترونية")
        mp = st.slider("أقصى عدد صفحات لكل مصدر (لا يتم زحف الموقع كاملًا)", 1, 10, 3)
        if st.button("⬇️ جلب المصادر المسجلة"):
            for src in REGISTERED_SOURCES:
                guarded(lambda s=src: kb.ingest(s, True, mp), f"جلب {src['name']}...")
        stt = kb.stats()
        st.write(f"صفحات ناجحة: **{stt['ok']}** — فاشلة/غير متاحة: **{stt['failed']}** — مقاطع: **{stt['chunks']}**")
        st.caption("تنبيه: هذه الأرقام تخص الصفحات التي جُلبت فقط، ولا تعني فهرسة المواقع بالكامل.")
        reg = {s["url"]: s for s in REGISTERED_SOURCES}
        for url, p in list(kb.pages.items()):
            c1, c2, c3 = st.columns([5, 1, 1])
            stale = " ⏳ قديم" if kb.is_stale(p) else ""
            c1.write(f"{'✅' if p['status']=='ok' else '⚠️'} {p['name']} — {p['title'][:60] or url}{stale}  \n`{url}`" + (f"  \n{p['error']}" if p["error"] else ""))
            if c2.button("تحديث", key="r_" + url):
                kb.ingest(reg.get(url, {"url": url, "kind": p["kind"], "name": p["name"]}), False, 1)
                st.rerun()
            if c3.button("حذف", key="rm_" + url):
                kb.remove(url)
                st.rerun()
        extra_url = st.text_input("إضافة رابط مرجع معتمد")
        if st.button("إضافة") and extra_url.startswith("http"):
            guarded(lambda: kb.ingest({"url": extra_url, "kind": "general", "name": urlparse(extra_url).netloc}, False, 1))
            st.rerun()
        unreg = [s["name"] for s in REGISTERED_SOURCES if s["url"] not in kb.pages]
        if unreg:
            st.caption("مصادر مسجلة لم تُجلب بعد: " + "، ".join(unreg))

    st.markdown('<div class="foot">✝️ للخدمة — كل محتوى مُولَّد يحتاج مراجعة الخادم المسؤول قبل الاستخدام</div>',
                unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 8. SELF-TESTS  (python app.py --selftest) — all HTTP is mocked
# ════════════════════════════════════════════════════════════════════════════
def run_selftests() -> int:
    import tempfile
    import unittest

    class R:
        def __init__(self, code=200, js=None, text="", headers=None):
            self.status_code, self._js, self.text, self.headers = code, js, text, headers or {}
            self.encoding = "utf-8"
            self.apparent_encoding = "utf-8"

        def json(self):
            if self._js is None:
                raise ValueError
            return self._js

    def ok_resp(content):
        return R(200, {"choices": [{"message": {"content": content}}]})

    class FakeHTTP:
        def __init__(self, seq):
            self.seq, self.calls = list(seq), []

        def post(self, url, **kw):
            self.calls.append({**kw, "url": url, "json": json.loads(json.dumps(kw.get("json", {})))})
            x = self.seq.pop(0)
            if isinstance(x, Exception):
                raise x
            return x

        def get(self, url, **kw):
            self.calls.append({"url": url, **kw})
            x = self.seq.pop(0) if self.seq else R(404)
            if isinstance(x, Exception):
                raise x
            return x

    nosleep = lambda s: None
    M = "a/b:free"
    MSG = [{"role": "user", "content": "x"}]
    P = lambda key, model, **kw: LLMProvider.single(key, model, **kw)
    OR = lambda model, key="k": Endpoint(OPENROUTER_BASE, key, model, "openrouter")
    HTML = ("<html><head><title>عيد الرسل</title><link rel='canonical' href='https://x.org/a.html'></head><body>"
            "<h2>عيد الرسل</h2><p>يحتفل الكنيسة القبطية بعيد الرسل في الخامس من أبيب كل عام تكريما للقديسين بطرس وبولس.</p>"
            "<p>رسالة العيد أن الرسل حملوا بشارة الإنجيل إلى العالم كله بقوة الروح القدس.</p>"
            "<a href='https://x.org/sec/b.html'>b</a></body></html>")

    class T(unittest.TestCase):
        def test_missing_key_messages(self):
            with self.assertRaises(LLMError) as c:
                P("", M).chat(MSG)
            self.assertEqual(c.exception.code, "missing_key")
            self.assertIn("OPENROUTER_API_KEY", c.exception.msg_ar)
            with self.assertRaises(LLMError) as c:
                P("", "gm", kind="gemini").chat(MSG)
            self.assertIn("GEMINI_API_KEY", c.exception.msg_ar)
            with self.assertRaises(LLMError) as c:
                LLMProvider([]).chat(MSG)
            self.assertEqual(c.exception.code, "missing_model")

        def test_redact(self):
            self.assertNotIn("sk-abcdef123456", redact("fail sk-abcdef123456", "sk-abcdef123456"))
            self.assertNotIn("AIzaSyA1234567890abcdefghij", redact("k=AIzaSyA1234567890abcdefghij"))

        def test_paid_openrouter_refused_without_request(self):
            h = FakeHTTP([])
            for mid in ("google/gemini-3.8-flash", "deepseek/deepseek-v4-flash-0731"):
                with self.assertRaises(LLMError) as c:
                    P("k", mid, http=h, sleep=nosleep).chat(MSG)
                self.assertEqual(c.exception.code, "not_free")
            self.assertEqual(h.calls, [])

        def test_free_route_payload_minimal(self):
            h = FakeHTTP([ok_resp("تمام"), ok_resp("تمام")])
            for mid in (ROUTER_FREE, M):
                P("k", mid, http=h, sleep=nosleep).chat(MSG, json_mode=True)
            for c in h.calls:
                self.assertNotIn("response_format", c["json"])
                self.assertNotIn("reasoning", c["json"])
                self.assertIs(c["json"]["stream"], False)
            self.assertEqual(h.calls[0]["json"]["model"], ROUTER_FREE)
            self.assertEqual(h.calls[0]["headers"]["Authorization"], "Bearer k")
            self.assertIn("X-Title", h.calls[0]["headers"])

        def test_gemini_endpoint(self):
            h = FakeHTTP([ok_resp("تمام")])
            P("gk", "gm", kind="gemini", http=h, sleep=nosleep).chat(MSG, json_mode=True)
            c = h.calls[0]
            self.assertTrue(c["url"].startswith(GEMINI_BASE) and c["url"].endswith("/chat/completions"))
            self.assertEqual(c["json"]["model"], "gm")
            self.assertEqual(c["headers"]["Authorization"], "Bearer gk")
            self.assertNotIn("X-Title", c["headers"])
            self.assertIn("response_format", c["json"])

        def test_invalid_model(self):
            with self.assertRaises(LLMError) as c:
                P("k", "nonsense", sleep=nosleep).chat(MSG)
            self.assertEqual(c.exception.code, "bad_model")
            h = FakeHTTP([R(404, text="no such model")])
            with self.assertRaises(LLMError) as c:
                P("k", M, http=h, sleep=nosleep).chat(MSG)
            self.assertEqual(c.exception.code, "model")

        def test_auth_not_retried_and_key_skipped(self):
            h = FakeHTTP([R(401)])
            with self.assertRaises(LLMError) as c:
                P("k", M, http=h, sleep=nosleep).chat(MSG)
            self.assertEqual(c.exception.code, "auth")
            self.assertEqual(len(h.calls), 1)
            h = FakeHTTP([R(401)])
            with self.assertRaises(LLMError):
                LLMProvider([OR(M), OR("c/d:free")], http=h, sleep=nosleep).chat(MSG)
            self.assertEqual(len(h.calls), 1)
            h = FakeHTTP([R(401), ok_resp("تمام")])
            p = LLMProvider([OR(M), Endpoint(GEMINI_BASE, "gk", "gm", "gemini")], http=h, sleep=nosleep)
            self.assertEqual(p.chat(MSG), "تمام")
            self.assertTrue(h.calls[1]["url"].startswith(GEMINI_BASE))

        def test_rate_limit_bounded_retry(self):
            h = FakeHTTP([R(429), R(429), R(429)])
            with self.assertRaises(LLMError) as c:
                P("k", M, http=h, sleep=nosleep, max_retries=2).chat(MSG)
            self.assertEqual(c.exception.code, "rate")
            self.assertEqual(len(h.calls), 3)

        def test_transient_then_success(self):
            h = FakeHTTP([requests.ConnectionError("x"), R(503), ok_resp("تمام")])
            self.assertEqual(P("k", M, http=h, sleep=nosleep).chat(MSG), "تمام")

        def test_network_failure(self):
            h = FakeHTTP([requests.Timeout()] * 4)
            with self.assertRaises(LLMError) as c:
                P("k", M, http=h, sleep=nosleep).chat(MSG)
            self.assertEqual(c.exception.code, "network")

        def test_pacing(self):
            waits = []
            h = FakeHTTP([ok_resp("1"), ok_resp("2")])
            p = P("k", M, http=h, sleep=waits.append, min_interval=5.0)
            p.chat(MSG)
            p.chat(MSG)
            self.assertTrue(waits and waits[0] > 0)

        def test_no_pacing_for_gemini(self):
            waits = []
            h = FakeHTTP([ok_resp("1"), ok_resp("2")])
            p = P("gk", "gm", kind="gemini", http=h, sleep=waits.append, min_interval=5.0)
            p.chat(MSG)
            p.chat(MSG)
            self.assertEqual(waits, [])

        def test_provider_error_in_200(self):
            h = FakeHTTP([R(200, {"error": {"message": "upstream down", "code": 502}})])
            with self.assertRaises(LLMError) as c:
                P("k", M, http=h, sleep=nosleep).chat(MSG)
            self.assertEqual(c.exception.code, "provider")
            self.assertIn(M, c.exception.msg_ar)

        def test_fallback_and_combined_failures(self):
            h = FakeHTTP([R(429), ok_resp("تمام")])
            seen = []
            p = LLMProvider([OR(M), OR("c/d:free")], http=h, sleep=nosleep, max_retries=0, on_model=seen.append)
            self.assertEqual(p.chat(MSG), "تمام")
            self.assertEqual(h.calls[-1]["json"]["model"], "c/d:free")
            self.assertEqual(seen, ["c/d:free"])
            h = FakeHTTP([R(429), R(429)])
            with self.assertRaises(LLMError) as c:
                LLMProvider([OR(M), OR("c/d:free")], http=h, sleep=nosleep, max_retries=0).chat(MSG)
            self.assertEqual(c.exception.code, "rate")
            self.assertEqual(c.exception.msg_ar.count("•"), 2)

        def test_free_models_rank(self):
            z = {"prompt": "0", "completion": "0"}
            cat = [{"id": "z/plain:free", "pricing": z, "context_length": 5},
                   {"id": "b/paid", "pricing": {"prompt": "0.1", "completion": "0.2"}},
                   {"id": "a/x:free", "pricing": z, "supported_parameters": ["response_format"]},
                   {"id": "g/img:free", "pricing": z, "architecture": {"output_modalities": ["image"]}},
                   {"id": "google/gemma-3n-e2b-it:free", "pricing": z},
                   {"id": "google/gemma-9:free", "pricing": z, "context_length": 9},
                   {"id": ROUTER_FREE, "pricing": z}]
            out = free_models(cat)
            self.assertEqual(out[0], "a/x:free")
            self.assertEqual(out[1:3], ["google/gemma-9:free", "z/plain:free"])
            self.assertEqual(out[-2:], ["google/gemma-3n-e2b-it:free", ROUTER_FREE])
            self.assertTrue(not {"b/paid", "g/img:free"} & set(out))
            self.assertEqual(free_models([]), [ROUTER_FREE])

        def test_pick_and_build_endpoints(self):
            free = ["google/gemma-4-31b-it:free", "deepseek/x:free", "other/y:free", ROUTER_FREE]
            fm = pick_engine_models(free)
            self.assertEqual(fm, {"gemma": ["google/gemma-4-31b-it:free"], "gemini_or": []})
            eps = build_endpoints("gemini", fm, "ork", "gk", "gm")
            self.assertEqual([e.model for e in eps],
                             ["gm", GEMINI_LITE, "google/gemma-4-31b-it:free", ROUTER_FREE])
            self.assertEqual(eps[0].kind, "gemini")
            self.assertEqual([e.model for e in build_endpoints("gemma", fm, "ork", "", "gm", fallback=False)],
                             ["google/gemma-4-31b-it:free"])
            self.assertEqual(build_endpoints("gemma", fm, "", "", "gm"), [])
            self.assertEqual([e.model for e in build_endpoints("auto", fm, "", "gk", "gm")], ["gm", GEMINI_LITE])
            st = engine_status(EMPTY_MAP, "ork", "")
            self.assertFalse(engine_status(EMPTY_MAP, "ork", "")["gemma"][0])
            self.assertTrue(st["auto"][0])
            self.assertTrue(engine_status(EMPTY_MAP, "", "gk")["gemini"][0])

        def test_empty_reply_retried(self):
            h = FakeHTTP([ok_resp(""), ok_resp("تمام")])
            self.assertEqual(P("gk", "gm", kind="gemini", http=h, sleep=nosleep).chat(MSG, json_mode=True), "تمام")
            self.assertIn("response_format", h.calls[0]["json"])
            self.assertNotIn("response_format", h.calls[1]["json"])
            h = FakeHTTP([R(200, {"choices": [{"finish_reason": "length", "message": {"content": None}}]}), ok_resp("تمام")])
            P("k", M, http=h, sleep=nosleep).chat(MSG, max_tokens=1000)
            self.assertEqual(h.calls[1]["json"]["max_tokens"], 2000)
            h = FakeHTTP([ok_resp("")] * 3)
            with self.assertRaises(LLMError) as c:
                P("k", M, http=h, sleep=nosleep).chat(MSG)
            self.assertEqual(c.exception.code, "empty")

        def test_json_mode_fallback(self):
            h = FakeHTTP([R(400, text="response_format unsupported"), ok_resp('{"title":"ت","questions":[]}')])
            q = P("gk", "gm", kind="gemini", http=h, sleep=nosleep).structured(Quiz, MSG)
            self.assertEqual(q.title, "ت")
            self.assertNotIn("response_format", h.calls[1]["json"])

        def test_malformed_json_repair_then_fail(self):
            good = '{"title":"ت","questions":[]}'
            h = FakeHTTP([ok_resp("not json"), ok_resp(good)])
            self.assertEqual(P("k", M, http=h, sleep=nosleep).structured(Quiz, MSG).title, "ت")
            h = FakeHTTP([ok_resp("bad"), ok_resp("still bad")])
            with self.assertRaises(LLMError) as c:
                P("k", M, http=h, sleep=nosleep).structured(Quiz, MSG)
            self.assertEqual(c.exception.code, "malformed")

        def _kb(self, seq):
            d = tempfile.mkdtemp()
            return KnowledgeBase(os.path.join(d, "kb.json"), http=FakeHTTP(seq), sleep=nosleep, delay=0)

        def test_ingest_metadata_and_retrieval(self):
            kb = self._kb([R(404), R(200, text=HTML), R(404), R(200, text=HTML.replace("sec/b", "sec/c"))])
            src = {"id": "t", "name": "T", "kind": "feasts", "url": "https://x.org/sec/a.html"}
            self.assertGreaterEqual(kb.ingest(src, True, 2), 1)
            ch = kb.web_chunks()
            self.assertTrue(ch and ch[0].url == "https://x.org/sec/a.html" and ch[0].canonical == "https://x.org/a.html" and ch[0].retrieved_at)
            top = retrieve("عندي درس عن عيد الرسل", ch)
            self.assertTrue(top and "الرسل" in top[0].text)

        def test_unavailable_recorded(self):
            kb = self._kb([R(404), R(500)])
            kb.ingest({"id": "t", "name": "T", "kind": "general", "url": "https://x.org/a.html"}, False, 1)
            self.assertEqual(kb.pages["https://x.org/a.html"]["status"], "unavailable")
            self.assertEqual(kb.stats()["ok"], 0)

        def test_robots_block(self):
            kb = self._kb([R(200, text="User-agent: *\nDisallow: /")])
            kb.ingest({"id": "t", "name": "T", "kind": "general", "url": "https://x.org/a.html"}, False, 1)
            self.assertIn("robots", kb.pages["https://x.org/a.html"]["error"])

        def test_stale_and_remove(self):
            kb = self._kb([])
            kb.pages["u"] = {"retrieved_at": "2020-01-01T00:00:00+00:00", "status": "ok", "chunks": []}
            self.assertTrue(kb.is_stale(kb.pages["u"]))
            kb.remove("u")
            self.assertNotIn("u", kb.pages)

        def test_citation_check(self):
            self.assertEqual(invalid_citations({"x": "حسب [م1] و [م5]"}, 2), [5])

        def test_unverified_quote(self):
            c = [Chunk("قال الرب لتلاميذه اذهبوا إلى العالم كله", "t", "u", "u", "", "biblical", "s")]
            self.assertEqual(unverified_quotes("«كلام مخترع تماما لا وجود له هنا»", c), ["كلام مخترع تماما لا وجود له هنا"])
            self.assertEqual(unverified_quotes("«اذهبوا إلى العالم كله»", c), [])

        def test_calendar(self):
            self.assertEqual(orthodox_easter(2026), dt.date(2026, 4, 12))
            self.assertEqual(occasions_for_year(2026)["عيد الرسل"]["date"], dt.date(2026, 7, 12))
            self.assertEqual(occasions_for_year(2026)["عيد الميلاد المجيد"]["date"], dt.date(2026, 1, 7))
            self.assertIsNone(find_occasion("درس عن المحبة"))
            self.assertEqual(find_occasion("درس عن عيد الرسل", 2026)["name"], "عيد الرسل")

        def _prov(self, outputs):
            h = FakeHTTP([ok_resp(o) for o in outputs])
            return P("k", "google/test:free", http=h, sleep=nosleep), h

        ACT = json.dumps({"items": [{"title": "لعبة", "description": "د", "steps": ["1"], "refs": [1]}]}, ensure_ascii=False)

        def test_workflow_and_preservation(self):
            kb = self._kb([])
            kb.pages["u"] = {"status": "ok", "chunks": [asdict(c) for c in make_chunks(parse_html(HTML, "https://x.org/a.html"), "https://x.org/a.html", "feasts", "2026-01-01")]}
            S = new_state()
            S["input"] = "عايز أفكار لدرس عن عيد الرسل."
            S["lesson"] = {"title": "قديم"}
            S["poster"] = {"title": "لوحة"}
            p, h = self._prov([self.ACT, self.ACT])
            run_task("activities", S, kb, p)
            sent = h.calls[0]["json"]["messages"][1]["content"]
            self.assertIn("[م1]", sent)
            self.assertIn("https://x.org/a.html", sent)
            self.assertEqual(S["occasion"]["name"], "عيد الرسل")
            self.assertEqual(S["lesson"], {"title": "قديم"})
            self.assertEqual(S["poster"], {"title": "لوحة"})
            self.assertEqual(S["activities"]["items"][0]["title"], "لعبة")
            run_task("activities", S, kb, p)
            self.assertEqual(S["lesson"], {"title": "قديم"})

        def test_no_church_context_for_unrelated_lesson(self):
            kb = self._kb([])
            S = new_state()
            S["input"] = "درس عن الصدق"
            ch, occ = retrieve_for(S, kb)
            self.assertEqual((ch, occ), ([], None))
            self.assertIn("لا توجد مراجع", build_context(ch, occ))

        def test_gate_overrides_confidence(self):
            v = Validation(biblical_alignment_ok=True, safety_ok=False, confidence=0.99)
            self.assertEqual(gate(v)[0], "blocked")
            v = Validation(biblical_alignment_ok=True, safety_ok=True, confidence=0.99, issues=[Issue(severity="critical", text="x")])
            self.assertEqual(gate(v)[0], "blocked")
            self.assertEqual(gate(Validation(biblical_alignment_ok=True, safety_ok=True))[0], "ok")
            self.assertEqual(gate(Validation(biblical_alignment_ok=True, safety_ok=True, unsupported_claims=["x"]))[0], "needs_review")

        def test_unsafe_activity_blocked(self):
            val = json.dumps({"biblical_alignment_ok": True, "safety_ok": True, "confidence": 0.95, "issues": []})
            S = new_state()
            S["activities"] = {"items": [{"title": "ش", "description": "نشعل نار صغيرة", "steps": []}]}
            p, _ = self._prov([val])
            run_validation(S, p)
            self.assertEqual(S["gate"], "blocked")

        def test_missing_sources_warns(self):
            S = new_state()
            S["input"] = "درس عن الصدق"
            p, _ = self._prov([self.ACT])
            run_task("activities", S, self._kb([]), p)
            self.assertEqual(S["refs"]["activities"], [])

        def test_exports(self):
            S = new_state()
            S["lesson"] = Lesson(title="درس", objectives=["أ"], sections=[LessonSection(heading="ع", body="نص عربي")]).model_dump()
            S["quiz"] = Quiz(title="اختبار", questions=[QuizQ(question="س", options=["أ", "ب"], answer_index=1, explanation="ش")]).model_dump()
            sets = export_sets(S)
            self.assertIn("مفتاح الإجابة (للخادم فقط)", sets)
            self.assertFalse(any("الإجابة الصحيحة" in t for _, t in sets["أوراق الأطفال"]))
            try:
                import docx  # noqa
                self.assertTrue(blocks_to_docx(sets["الدرس"]).startswith(b"PK"))
            except ImportError:
                self.skipTest("python-docx missing")
            try:
                import arabic_reshaper, bidi, reportlab  # noqa
                if find_arabic_font():
                    self.assertTrue(blocks_to_pdf(sets["الدرس"]).startswith(b"%PDF"))
                else:
                    self.skipTest("no Arabic font")
            except ImportError:
                self.skipTest("pdf deps missing")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(T)
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(run_selftests())
    main_app()
