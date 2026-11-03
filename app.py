# -*- coding: utf-8 -*-
"""
Church Lesson Creative Teaching Agent — مساعد مدارس الأحد الذكي   (FINAL v9 · title-only helper · free engines · curriculum-style lesson)
Streamlit UI + 3 free AI engines + source-aware retrieval + Coptic calendar
+ validation gate + DOCX/PDF export + built-in self-tests.

NEW in v8: the full lesson follows the uploaded church curriculum format:
   ❖ هدف الدرس  ❖ آية الدرس  ❖ أسئلة تمهيدية  ❖ الدرس (عامية مصرية، نقاط مرقمة)
   Same layout on screen and in DOCX / PDF.

Run app :  streamlit run app.py
Run tests: python app.py --selftest        (no key / no internet needed)

Keys are read ONLY from the backend (env vars / .streamlit/secrets.toml) — nothing is typed in the UI.
Free engines (each falls back to the others automatically):
  1. Gemma     -> OpenRouter, free ":free" Gemma models discovered LIVE from the catalog
  2. Gemini    -> Google AI Studio free tier (key from aistudio.google.com/apikey)
                  (+ any free Gemini that OpenRouter happens to list)
  3. Auto      -> OpenRouter's own free router "openrouter/free" (always-on safety net)

Keys (env var, .env, or .streamlit/secrets.toml):
  OPENROUTER_API_KEY   for engines 1, 3
  GEMINI_API_KEY       for engine 2          (optional GEMINI_MODEL_ID)
"""
from __future__ import annotations

import datetime as dt
import functools
import io
import json
import logging
import math
import os
import re
import sys
import ipaddress
import socket
import threading
import time
import unicodedata
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
BUILD = "final-v10.1-openpyxl"
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
CURRICULUM_FULL_CHARS = 9000      # a curriculum up to this size is sent to the model whole
AUTO_FETCH_PAGES = 2             # pages per registered source on the automatic background fetch
RETRIEVE_K = 8                   # curriculum-style lessons need more of the uploaded curriculum in context

ENGINES = ["gemma", "gemini", "auto"]
ENGINE_AR = {"gemma": "Gemma", "gemini": "Gemini", "auto": "تلقائي"}
ENGINE_ICON = {"gemma": "💎", "gemini": "✨", "auto": "🧭"}

log = logging.getLogger("church_agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Shared across every LLMProvider instance (a new one is created per task), so pacing really holds.
_PACE_LOCK = threading.Lock()
_LAST_CALL = {"t": 0.0}
# Guards writes to the shared lesson state when tasks run in parallel threads.
_STATE_LOCK = threading.Lock()


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
        self.min_interval = min_interval
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
        """Global (process-wide) spacing between free-route requests, thread-safe."""
        if self.min_interval > 0 and ep.free_route:
            with _PACE_LOCK:
                wait = self.min_interval - (time.monotonic() - _LAST_CALL["t"])
                if wait > 0:
                    self.sleep(wait)
                _LAST_CALL["t"] = time.monotonic()

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


MAX_DOWNLOAD_BYTES = 8_000_000


def safe_public_url(url: str, resolver=None) -> Tuple[bool, str]:
    """Only public http(s) addresses may be fetched: blocks localhost, private/link-local IPs and bad schemes."""
    p = urlparse((url or "").strip())
    if p.scheme not in ("http", "https") or not p.netloc or not p.hostname:
        return False, "الرابط غير صحيح — لازم يبدأ بـ http:// أو https://"
    host = p.hostname.lower()
    if host == "localhost" or host.endswith((".local", ".internal", ".localhost")):
        return False, "هذا العنوان داخلي ولا يمكن جلبه."
    try:
        infos = (resolver or socket.getaddrinfo)(host, None)
    except Exception:
        return False, "تعذر العثور على الموقع — تأكد من كتابة الرابط."
    for info in infos:
        try:
            ip = ipaddress.ip_address(str(info[4][0]).split("%")[0])
        except ValueError:
            return False, "عنوان الموقع غير صالح."
        if not ip.is_global:
            return False, "هذا العنوان داخلي/خاص ولا يمكن جلبه."
    return True, ""


def extract_doc_text(name: str, data: bytes) -> Tuple[str, str]:
    """PDF / DOCX bytes -> (text, error_ar). Arabic presentation forms are normalised (NFKC)."""
    lower = name.lower()
    try:
        if lower.endswith(".pdf"):
            from pypdf import PdfReader
            text = "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
        elif lower.endswith(".docx"):
            import docx
            d = docx.Document(io.BytesIO(data))
            text = "\n".join(p.text for p in d.paragraphs)
        else:
            return "", "نوع الملف غير مدعوم (PDF أو DOCX فقط)."
    except Exception:
        return "", "تعذر قراءة الملف (قد يكون تالفًا أو محميًا)."
    text = unicodedata.normalize("NFKC", text)
    if len(re.findall(r"[\u0600-\u06FF]", text)) < 50:
        return "", "لم يُستخرج نص عربي كافٍ (قد يكون الملف ممسوحًا ضوئيًا)."
    return text, ""


def doc_text_to_parsed(title: str, canonical: str, text: str) -> dict:
    return {"title": title, "canonical": canonical, "links": [],
            "lines": [("p", ln.strip()) for ln in text.splitlines() if len(ln.strip()) >= 20]}


class KnowledgeBase:
    """Registered public pages are cached on disk (shared). Pages the user adds (session=True) and the uploaded
    curriculum live in this session's memory only. Every URL passes `check_url` (public addresses only)."""

    def __init__(self, cache_path: str = KB_CACHE_PATH, http=requests, sleep=time.sleep, delay: float = 1.5,
                 check_url: Optional[Callable[[str], Tuple[bool, str]]] = None):
        self.cache_path, self.http, self.sleep, self.delay = cache_path, http, sleep, delay
        self.check_url = check_url or safe_public_url
        self.pages: Dict[str, dict] = {}
        self._robots: Dict[str, Optional[robotparser.RobotFileParser]] = {}
        self._lock = threading.RLock()
        self.load()

    def load(self):
        try:
            with open(self.cache_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
        with self._lock:
            self.pages = data if isinstance(data, dict) else {}

    def save(self):
        """Atomic write; session-only pages (user links) are never written to disk."""
        try:
            with self._lock:
                shared = {u: p for u, p in self.pages.items() if not p.get("session")}
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            tmp = f"{self.cache_path}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(shared, f, ensure_ascii=False)
            os.replace(tmp, self.cache_path)
        except Exception as e:
            log.warning("could not save KB cache: %s", e)

    def snapshot(self) -> List[Tuple[str, dict]]:
        with self._lock:
            return list(self.pages.items())

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

    def _record(self, url, kind, name, status, error="", parsed=None, chunks=None, session=False):
        rec = {"url": url, "kind": kind, "name": name, "status": status, "error": error,
               "title": (parsed or {}).get("title", ""), "canonical": (parsed or {}).get("canonical", url),
               "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "chunks": [asdict(c) for c in (chunks or [])]}
        if session:
            rec["session"] = True
        with self._lock:
            self.pages[url] = rec

    def fetch_page(self, url: str, kind: str, name: str, session: bool = False) -> Optional[dict]:
        rec = lambda status, err, parsed=None, chunks=None: self._record(url, kind, name, status, err, parsed, chunks, session)
        ok, why = self.check_url(url)
        if not ok:
            rec("unavailable", why)
            return None
        ok, why = self.allowed(url)
        if not ok:
            rec("unavailable", why)
            return None
        try:
            r = self.http.get(url, timeout=25, headers={"User-Agent": USER_AGENT})
            final = getattr(r, "url", "") or url
            if final != url:                       # redirects must also stay on public addresses
                ok2, why2 = self.check_url(final)
                if not ok2:
                    rec("unavailable", why2)
                    return None
            if r.status_code != 200:
                rec("unavailable", f"رمز الاستجابة {r.status_code}")
                return None
            hdrs = getattr(r, "headers", None) or {}
            try:
                size = int(hdrs.get("Content-Length") or 0)
            except (TypeError, ValueError):
                size = 0
            if size > MAX_DOWNLOAD_BYTES:
                rec("unavailable", "الملف كبير جدًا")
                return None
            ctype = str(hdrs.get("Content-Type", "")).lower()
            path = urlparse(url).path.lower()
            now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            if "pdf" in ctype or path.endswith(".pdf") or "wordprocessingml" in ctype or path.endswith(".docx"):
                fname = os.path.basename(urlparse(url).path) or "document"
                if not fname.lower().endswith((".pdf", ".docx")):
                    fname += ".pdf" if "pdf" in ctype else ".docx"
                text, err = extract_doc_text(fname, getattr(r, "content", b""))
                if err:
                    rec("unavailable", err)
                    return None
                parsed = doc_text_to_parsed(fname, url, text)
            else:
                if not r.encoding or r.encoding.lower() == "iso-8859-1":
                    r.encoding = r.apparent_encoding or "utf-8"
                parsed = parse_html(r.text, url)
            chunks = make_chunks(parsed, url, kind, now)
            if not chunks:
                rec("unavailable", "لا يوجد نص عربي مفيد في الصفحة", parsed)
                return None
            rec("ok", "", parsed, chunks)
            return parsed
        except Exception as e:
            rec("unavailable", redact(str(e))[:150])
            return None
        finally:
            self.sleep(self.delay)

    def ingest(self, src: dict, discover: bool = True, max_pages: int = 5, session: bool = False) -> int:
        """Fetch the start page + a *bounded* number of same-section links. Not a full crawl."""
        parsed = self.fetch_page(src["url"], src["kind"], src["name"], session)
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
                if self.fetch_page(link, src["kind"], src["name"], session):
                    n += 1
        self.save()
        return n

    def remove(self, url: str):
        with self._lock:
            self.pages.pop(url, None)
        self.save()

    def is_stale(self, page: dict) -> bool:
        """Good pages go stale after STALE_DAYS; failed ones are retried after one day."""
        try:
            t = dt.datetime.fromisoformat(page["retrieved_at"])
            limit = STALE_DAYS if page.get("status") == "ok" else 1
            return (dt.datetime.now(dt.timezone.utc) - t).total_seconds() > limit * 86400
        except Exception:
            return True

    def web_chunks(self) -> List[Chunk]:
        with self._lock:
            return [Chunk(**c) for p in self.pages.values() if p["status"] == "ok" for c in p["chunks"]]

    def stats(self) -> dict:
        with self._lock:
            ps = list(self.pages.values())
        ok = sum(1 for p in ps if p["status"] == "ok")
        return {"ok": ok, "failed": len(ps) - ok, "chunks": sum(len(p["chunks"]) for p in ps)}


def local_doc_to_chunks(name: str, data: bytes) -> Tuple[List[Chunk], str]:
    """PDF / DOCX curriculum -> chunks (kept in session memory only)."""
    text, err = extract_doc_text(name, data)
    if err:
        return [], err
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    parsed = doc_text_to_parsed(name, f"local:{name}", text)
    ch = make_chunks(parsed, f"local:{name}", "sunday_school", now)
    for c in ch:
        c.origin, c.verification = "curriculum", "منهج معتمد مرفوع من المشغّل"
    return ch, ""


def retrieve(query: str, chunks: List[Chunk], k: int = 5, min_score: float = 0.5) -> List[Chunk]:
    """BM25 over normalized Arabic tokens. Curriculum chunks get a boost."""
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


# Each quote style is matched with its own opener/closer, so a quote can no longer span unrelated text.
QUOTE_RE = re.compile(r"«([^»]{10,300})»|“([^”]{10,300})”|\"([^\"]{10,300})\"")


def unverified_quotes(text: str, chunks: List[Chunk], min_words: int = 5) -> List[str]:
    corpus = ar_norm(" ".join(c.text for c in chunks))
    bad = []
    for m in QUOTE_RE.finditer(text):
        q = next(g for g in m.groups() if g)
        if len(q.split()) >= min_words and ar_norm(q).strip() not in corpus:
            bad.append(q.strip())
    return bad


# Free models sometimes write schema field names INSIDE the text, e.g. "(source_tag: منهج)" or
# "[review_flags: ...]". Those are removed from the visible text; review_flags content is moved to the flags list.
_FLAG_LEAK = re.compile(r"[\[\(]\s*review_flags\s*[:：]\s*(.*?)[\]\)]", re.S)
_TAG_LEAK = re.compile(r"[\[\(]\s*source_tag\s*[:：]?\s*[^\]\)]*[\]\)]")
_BARE_LEAK = re.compile(r"\b(?:source_tag|review_flags|bible_refs)\b\s*[:：]?", re.I)


def clean_leaks(text: str, flags: List[str]) -> str:
    def grab(m):
        f = m.group(1).strip()
        if f:
            flags.append(f)
        return ""
    t = _FLAG_LEAK.sub(grab, text)
    t = _TAG_LEAK.sub("", t)
    t = _BARE_LEAK.sub("", t)
    t = re.sub(r"\(\s*\)|\[\s*\]", "", t)
    return re.sub(r"[ \t]{2,}", " ", t).strip() if t != text else text


def scrub_leaks(obj: Any, flags: List[str]) -> Any:
    if isinstance(obj, str):
        return clean_leaks(obj, flags)
    if isinstance(obj, list):
        return [scrub_leaks(x, flags) for x in obj]
    if isinstance(obj, dict):
        return {k: scrub_leaks(v, flags) for k, v in obj.items()}
    return obj


# ════════════════════════════════════════════════════════════════════════════
# 4. SCHEMAS + VALIDATION GATE
# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
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
            try:
                from openpyxl import load_workbook
            except ModuleNotFoundError as exc:
                if exc.name != "openpyxl":
                    raise
                return [], ("مكتبة openpyxl غير مثبتة في بيئة تشغيل Streamlit. "
                            "ثبّتها باستخدام: python3 -m pip install openpyxl "
                            "أو أضف openpyxl>=3.1.0 إلى requirements.txt ثم أعد تشغيل التطبيق.")
            wb = load_workbook(io.BytesIO(data), data_only=True, read_only=True)
            try:
                rows = []
                for ws in wb.worksheets:
                    rows.extend([list(r) for r in ws.iter_rows(values_only=True)])
                return rows_from_table(rows)
            finally:
                wb.close()
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
    except ImportError as exc:
        return [], f"تعذر تحميل مكتبة مطلوبة: {exc.name or 'غير معروفة'}. ثبّت المتطلبات عبر python3 -m pip install -r requirements.txt."
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
    """Curriculum-style lesson: هدف الدرس / آية الدرس / أسئلة تمهيدية / الدرس.
    Old-format data (title, objectives, sections, memory_verse, conclusion) still validates."""
    title: str
    main_idea: str = ""                      # the key sentence under «هدف الدرس»
    objectives: List[str] = []               # completes «مساعدة الطفل أن:»
    memory_verse: str = ""                   # آية الدرس (from references only) + its reference
    intro_questions: List[str] = []          # أسئلة تمهيدية للدرس
    sections: List[LessonSection]            # «الدرس»: body = numbered/bulleted lines
    conclusion: str = ""
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
    """Whole-word match (with و/ف/ب/ل/ك + ال prefixes), so «منارة» is not flagged for «نار»."""
    toks = [ar_norm(t) for t in re.findall(r"[\u0600-\u06FF]+", text or "")]
    stems = {re.sub(r"^[وفبلك]?(?:ال)?", "", t) for t in toks} | set(toks)
    flat = " ".join(toks)
    out = []
    for w in UNSAFE_WORDS:
        wn = ar_norm(w)
        hit = (wn in flat) if " " in wn else (wn in stems)
        if hit:
            out.append(Issue(severity="critical", text=f"كلمة تدل على خطر محتمل: «{w}» — يلزم تعديل النشاط أو مراجعته."))
    return out


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
    "(8) كن موجزًا ومحدد البنية. (9) لا تكتب أسماء الحقول التقنية (source_tag, review_flags, refs) داخل النصوص إطلاقًا؛ ضع قيمها في حقولها المخصصة فقط. "
    "(10) في الدرس الكامل اكتب بالعامية المصرية المبسطة بنفس أسلوب ملف المنهج المرفوع، وحافظ على ترتيب وعناوين أقسامه. "
    "(11) أنت مساعد للخادم فقط: كل ما تكتبه مسودة اقتراحات يراجعها الخادم ويقرّها ويعدّلها، ولا تتصرف كمرجع نهائي."
)
CHAT_SYSTEM = SYSTEM_BASE + " في المحادثة تكلم مع الخادم بالعامية المصرية الودودة، واقترح أفكارًا عملية وبسيطة الخامات، واسأل سؤالًا واحدًا فقط عند الحاجة."

LESSON_INSTR = (
    "اكتب الدرس بنفس بنية ملف المنهج المرفق في المراجع بالضبط:\n"
    "1) title: عنوان قصير.\n"
    "2) main_idea: جملة واحدة تلخص فكرة الدرس (مثل: «الكنيسة هي ملكوت السموات على الأرض»).\n"
    "3) objectives: 3 أهداف تبدأ بفعل مضارع، تكمل جملة «مساعدة الطفل أن: يعرف.. / يشعر.. / يتدرب..».\n"
    "4) memory_verse: آية الدرس مع رقمها، فقط إن وردت نصًا في المراجع؛ وإلا اتركها فارغة وضع ذلك في review_flags.\n"
    "5) intro_questions: 4 إلى 5 أسئلة تمهيدية قصيرة بالعامية المصرية تفتح النقاش.\n"
    "6) sections: أقسام «الدرس». القسم الأول مدخل بسيط، ثم أقسام للنقاط الرئيسية. داخل body اكتب كل نقطة في سطر مستقل "
    "مرقمًا هكذا «1. العنوان: الشرح» أو بشرطة «- » للتفاصيل، واشرح رمزية كل شيء بـ«لأن...».\n"
    "الأسلوب: عامية مصرية مبسطة للأطفال كما في ملف المنهج، جمل قصيرة. التزم بمعلومات المراجع (منهج الكنيسة المرفوع) "
    "ولا تضف رمزيات أو تفاصيل غير واردة فيها؛ أي إضافة ضعها في review_flags. "
    "إن لم يوجد منهج مرفوع، اكتب بنفس البنية مع تنبيه في review_flags أن المحتوى يحتاج مراجعة."
)

LESSON_EXTRA_INSTR = (
    "\n7) elements: «العناصر الأساسية للدرس» 3 إلى 5 نقاط قصيرة.\n"
    "8) reflection: 2 إلى 3 أفكار للتأمل.\n"
    "9) application: 2 إلى 3 أفكار للتطبيق العملي في حياة الطفل.\n"
    "10) لكل قسم في sections ضع source_tag (منهج/مرجع/شرح/إبداع/مراجعة) وأرقام المراجع التي اعتمد عليها في refs. "
    "أي آية أو قول لأحد الآباء لا بد أن يكون من المراجع المرفقة، مع ذكر قائله ومرجعه، وتوضيح هل هو حرفي أم شرح للمعنى. "
    "عند اختلاف التفاسير اذكر الاختلاف ولا تقدّم رأيًا غير مؤكد كأنه حقيقة."
)

TASKS: Dict[str, Tuple[type, str, str]] = {
    "analysis": (Analysis, "تحليل الدرس", "حلل الدرس: الموضوع، هل هو مناسبة كنسية، الرسالة الأساسية، أهداف التعلم، الشواهد (من المراجع فقط)."),
    "lesson": (Lesson, "الدرس الكامل", LESSON_INSTR + LESSON_EXTRA_INSTR),
    "activities": (Activities, "الأنشطة", "اقترح 4 أنشطة وألعاب إبداعية متنوعة وآمنة (حركية، فنية، جماعية، تأملية) بخامات بسيطة."),
    "poster": (Poster, "اللوحة التعليمية", "صمم لوحة/وسيلة إيضاح تفاعلية بخامات بسيطة: التصميم، الخامات، خطوات التنفيذ، سكريبت الشرح، مشاركة الأطفال، أسئلة نقاش."),
    "worksheet": (Worksheet, "ورقة العمل", "أنشئ ورقة عمل قابلة للطباعة من 5 تمارين متنوعة (توصيل، إكمال، ترتيب، رسم) مع الإجابة النموذجية في حقل answer."),
    "quiz": (Quiz, "الاختبار", "أنشئ اختبارًا من 6 أسئلة اختيار من متعدد بإجابة واحدة صحيحة وشرح قصير."),
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
}
TASK_ICON = {"analysis": "🔎", "lesson": "📖", "activities": "🎲", "poster": "🖼️", "worksheet": "📝", "quiz": "❓",
             "experiment": "🔬", "hymns": "🎵", "summary": "🧾"}
STATE_KEYS_FOR_CONTEXT = ["analysis", "lesson", "activities", "poster"]


def new_state() -> dict:
    return {"input": "", "curriculum_chunks": [], "curriculum_names": [], "occasion": None,
            **{k: None for k in TASKS}, "validation": None, "gate": None, "refs": {}, "chat": [],
            "prep": default_prep(), "confidence": None}


def retrieve_for(state: dict, kb: KnowledgeBase, extra: str = "") -> Tuple[List[Chunk], Optional[dict]]:
    """A short uploaded curriculum goes to the model WHOLE (a title-only query can't miss parts of it);
    a long one is searched with BM25. Web references are always searched."""
    q = f"{state['input']} {extra}"
    occ = find_occasion(q)
    query = q + (" " + occ["name"] if occ else "")
    cur = list(state["curriculum_chunks"])
    if cur and sum(len(c.text) for c in cur) <= CURRICULUM_FULL_CHARS:
        return cur + retrieve(query, kb.web_chunks(), k=3), occ
    return retrieve(query, cur + kb.web_chunks(), k=RETRIEVE_K), occ


def run_task(name: str, state: dict, kb: KnowledgeBase, provider: LLMProvider, extra: str = "") -> Any:
    """Regenerates ONLY the requested component; everything else in `state` is untouched."""
    schema, _, instr = TASKS[name]
    chunks, occ = retrieve_for(state, kb, extra)
    existing = {k: state[k] for k in STATE_KEYS_FOR_CONTEXT if state.get(k) and k != name}
    user = (f"طلب الخادم/الدرس:\n{state['input']}\n\n{prep_context(state)}"
            f"المراجع المسترجعة:\n{build_context(chunks, occ)}\n\n"
            + (f"مكونات الدرس الحالية (حافظ على اتساقها):\n{json.dumps(existing, ensure_ascii=False)[:4000]}\n\n" if existing else "")
            + (f"ملاحظة إضافية: {extra}\n\n" if extra else "") + f"المهمة: {instr}")
    obj = provider.structured(schema, [{"role": "system", "content": SYSTEM_BASE}, {"role": "user", "content": user}],
                              max_tokens=3500)
    leaked: List[str] = []
    data = scrub_leaks(obj.model_dump(), leaked)
    if leaked:
        data.setdefault("review_flags", []).extend(leaked)
    bad = invalid_citations(data, len(chunks))
    if bad:
        data.setdefault("review_flags", []).append(f"إشارات مراجع غير موجودة {bad} — تم تجاهلها وتحتاج مراجعة.")
    # All shared-state writes happen under one lock (tasks may run in parallel threads).
    with _STATE_LOCK:
        state["occasion"] = occ or state.get("occasion")
        state[name] = data
        state["refs"][name] = [asdict(c) for c in chunks]
        state["validation"], state["gate"], state["confidence"] = None, None, None
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
    hands = {k: state[k] for k in ('activities', 'poster', 'worksheet') if state.get(k)}
    if state.get("experiment"):          # only tools/steps are hazards; the safety notes may legitimately say "no fire"
        hands["experiment"] = {k: state["experiment"].get(k) for k in ("tools", "steps")}
    hands_on = json.dumps(hands, ensure_ascii=False)
    extra = local_safety_scan(hands_on)      # fire/knives are only a hazard in hands-on parts, not in lesson text
    # Quote check runs in the BACKEND only: logged and stored, never shown in the UI or used by the gate
    # (it flags every «...» phrase, including the model's own activity text, so it was mostly noise).
    notes = [q[:120] for q in unverified_quotes(flat, chunks)]
    if notes:
        log.info("backend: %d unverified quoted phrases (hidden from UI)", len(notes))
    status, issues = gate(v, extra)
    state["validation"] = {**v.model_dump(), "issues": [i.model_dump() for i in issues], "backend_notes": notes}
    state["gate"] = status
    state["confidence"] = assess_confidence(state)
    return state["validation"]


class Job:
    """Background generation job. Plain data + lock: worker threads never touch Streamlit state,
    so the UI stays fully interactive while a lesson is being generated."""

    def __init__(self, total: int):
        self.lock = threading.Lock()
        self.total, self.done = max(total, 1), 0
        self.text, self.errors = "جاري البدء...", {}
        self.finished, self.ok, self.last_model = False, False, ""

    def update(self, text: Optional[str] = None, inc: int = 0):
        with self.lock:
            if text:
                self.text = text
            self.done += inc

    def fraction(self) -> float:
        return min(self.done / self.total, 1.0)


def run_job(job: Job, S: dict, kb: KnowledgeBase, make_provider: Callable[[], LLMProvider],
            names: List[str], validate: bool = True, skip_done: bool = True, extra: str = "", fast: bool = False):
    """Runs the requested tasks (+ optional validation). Never raises: results/errors are stored on `job`."""
    def fail(key: str, e: Exception):
        job.errors[key] = e.msg_ar if isinstance(e, LLMError) else redact(str(e))[:200]
    try:
        todo = [n for n in names if not (S.get(n) and skip_done)]
        job.update(inc=len(names) - len(todo))
        if fast and len(todo) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            job.update(f"⚡ توليد {len(todo)} أجزاء معًا ...")
            with ThreadPoolExecutor(max_workers=3) as ex:
                futs = {ex.submit(run_task, n, S, kb, make_provider(), extra): n for n in todo}
                for fut in as_completed(futs):
                    n = futs[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        fail(n, e)
                    job.update(f"اكتمل: {TASKS[n][1]}", inc=1)
        else:
            for n in todo:
                job.update(f"جاري توليد {TASKS[n][1]} ...")
                try:
                    run_task(n, S, kb, make_provider(), extra)
                except Exception as e:
                    fail(n, e)
                    break
                job.update(inc=1)
        if validate and not job.errors:
            job.update("جاري المراجعة ...")
            try:
                run_validation(S, make_provider())
            except Exception as e:
                fail("validation", e)
            job.update(inc=1)
    except Exception as e:           # last-resort guard so the UI never waits forever
        fail("job", e)
    finally:
        job.ok = not job.errors
        job.text = "✅ اكتمل التوليد" if job.ok else "توقف التوليد"
        job.finished = True


def run_fetch(job: Job, kb: KnowledgeBase, sources: List[dict], discover: bool = True,
              max_pages: int = 3, session: bool = False):
    """Background web fetching (never blocks the UI). Never raises."""
    try:
        for s in sources:
            job.update(f"جاري جلب {s['name']} ...")
            try:
                kb.ingest(s, discover, max_pages, session)
            except Exception as e:
                job.errors[s["name"]] = redact(str(e))[:150]
            job.update(inc=1)
    except Exception as e:
        job.errors["fetch"] = redact(str(e))[:150]
    finally:
        job.ok = not job.errors
        job.text = "✅ اكتمل الجلب" if job.ok else "اكتمل الجلب مع أخطاء"
        job.finished = True


# ════════════════════════════════════════════════════════════════════════════
# 6. RENDER BLOCKS + EXPORTS (DOCX / PDF, Arabic RTL)
# ════════════════════════════════════════════════════════════════════════════
_NUM_LINE = re.compile(r"^\s*(\d+|[١-٩]+)\s*[\.\-\)]\s*")
_BUL_LINE = re.compile(r"^\s*[-•▪]\s*")


def lesson_blocks(d: dict) -> List[Tuple[str, str]]:
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



def component_blocks(name: str, d: dict, answers: bool = False) -> List[Tuple[str, str]]:
    B: List[Tuple[str, str]] = []
    tag = lambda x: f"[{x.get('source_tag')}] " if x.get("source_tag") else ""
    if name == "analysis":
        B += [("h2", "تحليل الدرس"), ("p", f"الموضوع: {d['topic']}"), ("p", f"الرسالة الأساسية: {d['key_message']}")]
        B += [("li", o) for o in d["objectives"]]
        if d.get("bible_refs"):
            B += [("h3", "الشواهد")] + [("li", x) for x in d["bible_refs"]]
    elif name == "lesson":
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


def export_sets(state: dict, approved: bool = True) -> Dict[str, List[Tuple[str, str]]]:
    lesson = header_blocks(state)
    for k in ("lesson", "experiment", "hymns", "activities", "poster"):
        d = state.get(k)
        if not d or (k == "experiment" and not d.get("suitable", True)):
            continue
        lesson += component_blocks(k, filtered_lesson(state) if k == "lesson" else d)
    lesson += refs_blocks(state)
    if state.get("gate") in ("blocked", "needs_review"):
        lesson.insert(0, ("p", "تنبيه: هذا المحتوى يحتاج مراجعة بشرية قبل الاستخدام."))
    elif not approved:
        lesson.insert(0, ("p", "مسودة اقتراحات من المساعد — لم تُراجَع بعد، وتحتاج مراجعة الخادم قبل الاستخدام."))
    student = []
    for k in ("worksheet", "quiz"):
        if state.get(k):
            student += component_blocks(k, state[k], answers=False)
    key = []
    for k in ("worksheet", "quiz"):
        if state.get(k):
            key += component_blocks(k, state[k], answers=True)
    summ = component_blocks("summary", state["summary"]) if state.get("summary") else []
    out = {"الدرس": lesson, "ملخص الخادم": summ, "أوراق الأطفال": student, "مفتاح الإجابة (للخادم فقط)": key}
    return {k: v for k, v in out.items() if v}


def blocks_to_docx(blocks: List[Tuple[str, str]], logo: bytes = b"") -> bytes:
    import docx
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    d = docx.Document()
    if logo:
        try:
            d.add_picture(io.BytesIO(logo), width=docx.shared.Inches(1.0))
            d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        except Exception:
            log.warning("logo could not be embedded in DOCX")
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


def blocks_to_pdf(blocks: List[Tuple[str, str]], logo: bytes = b"") -> bytes:
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
    if logo:
        try:
            from reportlab.lib.utils import ImageReader
            c.drawImage(ImageReader(io.BytesIO(logo)), (W - 60) / 2, y - 60, 60, 60, preserveAspectRatio=True, mask="auto")
            y -= 72
        except Exception:
            log.warning("logo could not be drawn in PDF")
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


# Cached exports: rebuilt only when the content changes, not on every Streamlit rerun.
@functools.lru_cache(maxsize=32)
def _docx_cached(blocks_json: str, logo: bytes = b"") -> bytes:
    return blocks_to_docx([tuple(b) for b in json.loads(blocks_json)], logo)


@functools.lru_cache(maxsize=32)
def _pdf_cached(blocks_json: str, logo: bytes = b"") -> bytes:
    return blocks_to_pdf([tuple(b) for b in json.loads(blocks_json)], logo)


# ════════════════════════════════════════════════════════════════════════════
# 7. STREAMLIT UI  (RTL content, adaptive light/dark theme, fluid sizing)
# ════════════════════════════════════════════════════════════════════════════
# Rules that keep the layout working on every screen:
#  * direction:rtl is applied ONLY to content areas (stMain / stSidebarContent), never to .stApp or the
#    flex shell -> the sidebar is no longer squeezed to a thin strip on phones.
#  * no overflow:hidden on html/body/stApp/stAppViewContainer -> Streamlit's mobile sidebar overlay works.
#  * every colour is a token; dark mode only redefines tokens (prefers-color-scheme).
#  * sizes use clamp()/min() and columns wrap, so it adapts without per-device hacks.
CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');

/* ========== design tokens (light) ========== */
:root{
  --navy:#14213d; --navy2:#2a3f6f; --gold:#c8963e; --gold2:#e6b95c;
  --bg1:#fbf7ef; --bg2:#f4ecdc; --glow:#fff7e0;
  --card:#ffffff; --card-sel:#fffdf6; --card-off:#f6f2e8;
  --ink:#1f2a44; --muted:#5b6684; --heading:#14213d;
  --line:#e6dcc6; --input:#ffffff;
  --ok:#2e7d5b; --ok-bg:#e4f4ec; --bad:#b4443c; --bad-bg:#f8e0de; --warn:#9a6a12; --warn-bg:#fff1d6;
  --side1:#fffaf0; --side2:#f3e9d4;
  --shadow:rgba(20,33,61,.08);
  --r:clamp(12px,2vw,18px);
  --gap:clamp(.5rem,1.6vw,1rem);
}
/* ========== design tokens (dark) ========== */
@media (prefers-color-scheme:dark){
  :root{
    --bg1:#0f1626; --bg2:#141d33; --glow:#26304f;
    --card:#1a2440; --card-sel:#212c4c; --card-off:#161e36;
    --ink:#e8ecf7; --muted:#a3adc8; --heading:#f0d9a8;
    --line:#2e3a5c; --input:#16203a;
    --ok:#6fd0a0; --ok-bg:#16382a; --bad:#ff9a92; --bad-bg:#3d1f1f; --warn:#f2c46b; --warn-bg:#3b2f14;
    --side1:#121a30; --side2:#0f1626;
    --shadow:rgba(0,0,0,.4);
  }
}

html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, label{
  font-family:'Cairo','Segoe UI',Tahoma,sans-serif !important}

/* ========== APP SHELL: never set direction / overflow here ========== */
.stApp{
  color:var(--ink);
  background:radial-gradient(1200px 500px at 90% -10%,var(--glow) 0,transparent 60%),
             linear-gradient(180deg,var(--bg1) 0%,var(--bg2) 100%)}

/* ========== RTL only on CONTENT ========== */
[data-testid="stMain"], [data-testid="stSidebarContent"]{direction:rtl;text-align:right}
[data-testid="stMain"]{overflow-x:clip}
.block-container{width:100%;max-width:1180px;
  padding:clamp(.8rem,2vw,1.4rem) clamp(.8rem,3vw,2rem) 3rem}

h1,h2,h3,h4{color:var(--heading);font-weight:800}

/* ========== SIDEBAR: sane width, wrap words not letters ========== */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,var(--side1),var(--side2));
  border:none !important;box-shadow:none !important}
[data-testid="stSidebar"][aria-expanded="true"]{min-width:min(86vw,320px)}
[data-testid="stSidebarContent"]{word-break:normal;overflow-wrap:break-word}
[data-testid="stSidebarContent"] *{word-break:normal}
[data-testid="stSidebar"] [data-testid*="esize"],
[data-testid="stSidebar"] [class*="esize"]{display:none !important}
[data-testid="stSidebar"]::before,[data-testid="stSidebar"]::after{display:none !important}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:var(--heading)}

/* ========== hero ========== */
.hero{position:relative;overflow:hidden;border-radius:calc(var(--r) + 6px);
  padding:clamp(18px,4vw,30px) clamp(16px,4vw,34px);margin-bottom:var(--gap);color:#fff;
  background:linear-gradient(135deg,var(--navy) 0%,var(--navy2) 62%,#8a6a2c 150%);
  box-shadow:0 14px 34px var(--shadow)}
.hero:before{content:"✝";position:absolute;left:-10px;top:-40px;font-size:clamp(120px,30vw,230px);opacity:.07;color:#fff}
.hero h1{color:#fff;margin:0 0 6px;font-size:clamp(1.3rem,4.5vw,2rem);line-height:1.35}
.hero p{margin:0;opacity:.9;font-size:clamp(.88rem,2.4vw,1.02rem)}
.hero .chips{margin-top:14px;display:flex;gap:6px;flex-wrap:wrap}
.chip{background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.28);
  padding:3px clamp(8px,2vw,12px);border-radius:999px;font-size:clamp(.72rem,2vw,.85rem)}

/* ========== engine cards ========== */
.ecard{background:var(--card);border:1.5px solid var(--line);border-radius:var(--r);padding:14px 16px;
  box-shadow:0 4px 14px var(--shadow);transition:.2s}
.ecard.sel{border-color:var(--gold);background:var(--card-sel);box-shadow:0 6px 20px rgba(200,150,62,.28)}
.ecard.off{opacity:.62;background:var(--card-off)}
.ecard .t{font-weight:800;color:var(--heading);font-size:clamp(.95rem,2.6vw,1.08rem)}
.ecard .n{font-size:.82rem;color:var(--muted);margin-top:4px}
.pill{display:inline-block;font-size:.72rem;font-weight:700;padding:2px 9px;border-radius:999px;margin-right:6px}
.pill.ok{background:var(--ok-bg);color:var(--ok)} .pill.off{background:var(--bad-bg);color:var(--bad)}

/* ========== buttons / inputs ========== */
.stButton>button,.stDownloadButton>button{border-radius:14px;border:1.5px solid var(--gold);
  background:var(--card);color:var(--heading);font-weight:700;padding:.45rem 1rem;transition:.18s}
.stButton>button:hover,.stDownloadButton>button:hover{background:var(--navy);color:#fff;
  transform:translateY(-1px);box-shadow:0 6px 16px var(--shadow)}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,var(--gold),#a97420);border:none;color:#fff;
  font-size:clamp(.95rem,2.4vw,1.05rem);padding:.7rem 1.2rem;box-shadow:0 8px 20px rgba(200,150,62,.38)}
.stButton>button[kind="primary"]:hover{filter:brightness(1.07)}
.stTextArea textarea,.stTextInput input{direction:rtl;text-align:right;border-radius:14px !important;
  border:1.5px solid var(--line) !important;background:var(--input) !important;color:var(--ink) !important}
.stTextArea textarea:focus,.stTextInput input:focus{border-color:var(--gold) !important;
  box-shadow:0 0 0 3px rgba(200,150,62,.2) !important}
.stChatInput textarea{direction:rtl;text-align:right}

/* ========== expanders / alerts / progress ========== */
div[data-testid="stExpander"]{background:var(--card);border:1.5px solid var(--line) !important;
  border-radius:var(--r) !important;box-shadow:0 4px 14px var(--shadow);margin-bottom:10px;overflow:hidden}
div[data-testid="stExpander"] summary{font-weight:800;color:var(--heading)}
div[data-testid="stExpander"] details,div[data-testid="stExpander"] summary,
div[data-testid="stAlert"]{direction:rtl;text-align:right;border-radius:14px}
.stProgress>div>div>div>div{background:linear-gradient(90deg,var(--gold),var(--gold2))}

/* ========== section titles / badges ========== */
.sec{display:flex;align-items:center;gap:10px;margin:clamp(18px,4vw,26px) 0 10px;font-weight:800;
  color:var(--heading);font-size:clamp(1.05rem,3vw,1.25rem)}
.sec:after{content:"";flex:1;height:2px;background:linear-gradient(90deg,var(--gold),transparent);opacity:.55}
.badge{display:inline-block;padding:6px 16px;border-radius:999px;font-weight:800;font-size:.95rem}
.badge.ok{background:var(--ok-bg);color:var(--ok)} .badge.rev{background:var(--warn-bg);color:var(--warn)}
.badge.blk{background:var(--bad-bg);color:var(--bad)}

/* ========== Arabic markdown ========== */
[data-testid="stMarkdownContainer"],[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] h1,[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,[data-testid="stMarkdownContainer"] h4{direction:rtl;text-align:right}
[data-testid="stMarkdownContainer"] ul,[data-testid="stMarkdownContainer"] ol{direction:rtl;text-align:right;
  padding-right:1.5rem;padding-left:0;margin-right:0;list-style-position:outside}
[data-testid="stMarkdownContainer"] li{direction:rtl;text-align:right;margin-bottom:.35rem}
[data-testid="stMarkdownContainer"] li::marker{color:var(--gold)}
.foot{text-align:center;color:var(--muted);font-size:.8rem;margin:30px 0 6px}

/* ========== responsive columns: wrap instead of overflowing ========== */
[data-testid="stHorizontalBlock"]{flex-wrap:wrap;gap:var(--gap)}
[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]{min-width:min(100%,220px)}
[data-testid="stRadio"] div[role="radiogroup"]{flex-wrap:wrap;gap:.4rem}
img,svg,canvas,table{max-width:100%}

@media (max-width:768px){
  .stButton>button,.stDownloadButton>button{width:100%}
  [data-testid="stHorizontalBlock"]>[data-testid="stColumn"]{flex:1 1 100% !important;width:100% !important}
}

/* ========== motion ========== */
@keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.hero,.ecard,div[data-testid="stExpander"]{animation:rise .35s ease both}
@media (prefers-reduced-motion:reduce){*{animation:none !important;transition:none !important}}
</style>"""


def main_app():
    import streamlit as st
    st.set_page_config(page_title=APP_TITLE_AR, page_icon="✝️", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)

    ss = st.session_state
    ss.setdefault("S", new_state())
    for k, v in (("auto_fb", True), ("skip_done", True), ("approved", False), ("title", ""), ("notes", ""),
                 ("link_url", ""), ("link_follow", False), ("p_stage", "ابتدائي"), ("p_count", 15),
                 ("p_venue", "فصل صغير (5-10 أطفال)"), ("p_tools", ""), ("p_space", ""), ("p_date", dt.date.today()),
                 ("p_servant", ""), ("p_church", ""), ("sched_q", ""), ("logo", b"")):
        ss.setdefault(k, v)
    if "profile_loaded" not in ss:                      # church profile, template, schedule, library (shared on disk)
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
    # A KnowledgeBase kept in the session by an OLDER version of this file lacks newer methods -> rebuild it.
    if "kb" not in ss or not all(hasattr(ss["kb"], m) for m in ("snapshot", "check_url", "is_stale")):
        ss["kb"] = KnowledgeBase()
        ss.pop("auto_fetched", None)
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

    def start_job(names: List[str], validate: bool, extra: str = ""):
        """Runs generation in a background thread so every widget keeps working meanwhile."""
        eps = endpoints(ss["engine"], ss["auto_fb"])
        site, app_name = get_secret("OPENROUTER_SITE_URL"), get_secret("OPENROUTER_APP_NAME", "Church Lesson Creative Teaching Agent")
        job = Job(len(names) + (1 if validate else 0))
        make = lambda: LLMProvider(eps, site_url=site, min_interval=FREE_MIN_INTERVAL, app_name=app_name,
                                   on_model=lambda m: setattr(job, "last_model", m))
        fast = bool(eps) and eps[0].kind == "gemini"      # Gemini has no 20/min cap -> run parts together
        ss["approved"] = False                            # new content -> the teacher must approve again
        ss["job"] = job
        threading.Thread(target=run_job, args=(job, S, kb, make, names, validate, ss["skip_done"], extra, fast),
                         daemon=True).start()

    def start_fetch(sources: List[dict], discover: bool, max_pages: int, session: bool = False):
        fj = Job(len(sources))
        ss["fjob"] = fj
        threading.Thread(target=run_fetch, args=(fj, kb, sources, discover, max_pages, session), daemon=True).start()

    def new_lesson():
        ss["S"] = new_state()
        ss["title"], ss["notes"], ss["approved"] = "", "", False

    def use_row(row: dict):
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

    def add_link():
        raw = (ss.get("link_url") or "").strip()
        if not raw:
            ss["link_msg"] = ("warn", "اكتب الرابط أولًا.")
            return
        ok, why = kb.check_url(raw)
        if not ok:
            ss["link_msg"] = ("err", why)
            return
        start_fetch([{"url": raw, "kind": "general", "name": urlparse(raw).netloc}], bool(ss.get("link_follow")), 3, True)
        ss["link_msg"] = ("ok", "بدأ جلب الرابط في الخلفية — تابع حالته في القائمة أدناه. (يبقى في جلستك فقط)")
        ss["link_url"] = ""

    # Registered sources are fetched once per session in the background (shared cache, polite pacing).
    if not ss.get("auto_fetched"):
        ss["auto_fetched"] = True
        todo = [s for s in REGISTERED_SOURCES if s["url"] not in kb.pages or kb.is_stale(kb.pages[s["url"]])]
        if todo:
            start_fetch(todo, True, AUTO_FETCH_PAGES)

    job, fjob = ss.get("job"), ss.get("fjob")
    running = bool(job and not job.finished)
    fetching = bool(fjob and not fjob.finished)

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
        st.button("🗑️ بدء درس جديد", disabled=running, on_click=new_lesson)

    # ---------------- hero ----------------
    st.markdown(f"""<div class="hero"><h1>✝️ {APP_TITLE_AR}</h1>
      <p>اكتب عنوان الدرس والمساعد يجهّز لك مسودة: درس، أنشطة، لوحة، ورقة عمل واختبار — لأطفال 8–10 سنوات</p>
      <div class="chips"><span class="chip">🤝 مساعد فقط — القرار ليك</span><span class="chip">📚 يعتمد على منهج الكنيسة</span>
      <span class="chip">🛡️ مراجعة عقيدة وأمان</span><span class="chip">📤 DOCX / PDF</span></div></div>""",
                unsafe_allow_html=True)

    # ---------------- engine (advanced, collapsed) ----------------
    has_engine = bool(endpoints(ss["engine"], ss["auto_fb"]))
    with st.expander("🧠 المحرّك المجاني (متقدم)"):
        cols = st.columns(3)
        for col, n in zip(cols, ENGINES):
            ok, note = status[n]
            cls = ("sel " if ss["engine"] == n else "") + ("" if ok else "off")
            col.markdown(f"""<div class="ecard {cls}"><div class="t">{ENGINE_ICON[n]} {ENGINE_AR[n]}
              <span class="pill {'ok' if ok else 'off'}">{'جاهز' if ok else 'غير متاح'}</span></div>
              <div class="n">{note}</div></div>""", unsafe_allow_html=True)
        st.radio("المحرّك", ENGINES, key="engine", horizontal=True, format_func=lambda n: f"{ENGINE_ICON[n]} {ENGINE_AR[n]}",
                 label_visibility="collapsed")
        if job and job.last_model:
            ss["last_model"] = job.last_model
        if ss.get("last_model"):
            st.caption(f"آخر نموذج استجاب: {ss['last_model']}")
        if has_engine and not status[ss["engine"]][0] and ss["auto_fb"]:
            st.info("هذا المحرّك غير متاح الآن — سيُستخدم أول محرّك متاح تلقائيًا.")
    if not has_engine:
        st.warning("لا يوجد محرّك جاهز: مفاتيح الخادم غير مضبوطة (OPENROUTER_API_KEY / GEMINI_API_KEY).")

    st.info("🤝 المساعد بيقترح مسودة بس — أنت اللي بتراجع وتعدّل وتقرر قبل ما تستخدم أي حاجة.")

    # ---------------- 0) church lesson schedule ----------------
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

    # ---------------- 1) lesson title ----------------
    st.markdown('<div class="sec">📝 عنوان الدرس</div>', unsafe_allow_html=True)
    st.text_input("عنوان الدرس", key="title", placeholder="مثال: الكنيسة", label_visibility="collapsed")
    with st.expander("➕ تفاصيل إضافية (اختياري)"):
        st.text_area("ملاحظات", key="notes", height=90, label_visibility="collapsed",
                     placeholder="مثال: عايز لوحة وأنشطة لتالتة ورابعة ابتدائي")
    notes = ss["notes"].strip()
    S["input"] = (ss["title"].strip() + ("\n" + notes if notes else "")).strip()
    gd_ = ss["p_date"] if isinstance(ss["p_date"], dt.date) else None
    S["prep"] = {"stage": ss["p_stage"], "count": int(ss["p_count"]), "venue": ss["p_venue"], "tools": ss["p_tools"].strip(),
                 "space": ss["p_space"].strip(), "greg": gd_.isoformat() if gd_ else "", "coptic": coptic_str(gd_) if gd_ else "",
                 "servant": ss["p_servant"].strip(), "church": ss["p_church"].strip(), "tpl": list(ss["tpl"]),
                 "prev": previous_lesson(ss["schedule"], gd_.isoformat() if gd_ else "", ss["title"].strip())}
    can_run = has_engine and bool(S["input"])
    occ = find_occasion(S["input"]) if S["input"] else None
    if occ:
        st.info(f"مناسبة مرصودة: **{occ['name']}** ({occ['kind']}) — {occ['date'].isoformat()} — {occ['note']}")

    # ---------------- 1b) preparation details ----------------
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

    # ---------------- 2) curriculum upload ----------------
    st.markdown('<div class="sec">📚 منهج الدرس (موصى به)</div>', unsafe_allow_html=True)
    ups = st.file_uploader("ارفع ملف المنهج PDF أو DOCX", type=["pdf", "docx"], accept_multiple_files=True,
                           label_visibility="collapsed")
    for f in ups or []:
        if f.name not in S["curriculum_names"]:
            ch, err = local_doc_to_chunks(f.name, f.getvalue())
            if err:
                st.error(f"{f.name}: {err}")
            else:
                S["curriculum_chunks"] += ch
                S["curriculum_names"].append(f.name)
    if S["curriculum_names"]:
        st.success("المنهج المحمّل: " + "، ".join(S["curriculum_names"]))
    else:
        st.caption("بدون ملف منهج هيعتمد المساعد على المراجع الإلكترونية فقط، وهيطلب منك مراجعة أدق. "
                   "المنهج المرفوع يبقى في ذاكرة الجلسة فقط ولا يُحفظ على القرص.")

    ORDER = ["analysis", "lesson", "experiment", "hymns", "activities", "poster", "worksheet", "quiz", "summary"]

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

    # ---------------- 3) generate (background job: the page stays interactive) ----------------
    if st.button("✨ ساعدني في تحضير الدرس", type="primary", disabled=not can_run or running,
                 key="gen_all", use_container_width=True):
        start_job(ORDER, True)
        st.rerun()
    if not S["input"]:
        st.caption("اكتب عنوان الدرس أولًا ثم اضغط الزر.")
    st.checkbox("عند إعادة الضغط، تخطَّ الأجزاء المولّدة بالفعل", key="skip_done")

    def _job_view():
        j = ss["job"]
        st.progress(j.fraction(), text=j.text)
        st.caption("⏳ المساعد شغال في الخلفية — تقدر تتصفح وتستخدم باقي الأدوات.")
        if j.finished:
            st.rerun()                       # refresh the whole page once, to show the results

    if running:
        if hasattr(st, "fragment"):
            st.fragment(run_every=2)(_job_view)()
        else:                                 # very old Streamlit: simple polling fallback
            _job_view()
            time.sleep(2)
            st.rerun()
    elif job:
        st.progress(1.0 if job.ok else job.fraction(), text=job.text)
        for k, m in job.errors.items():
            st.error(f"{TASKS[k][1] if k in TASKS else 'المراجعة'}: {m}")
        if job.errors:
            st.info("الأجزاء الناجحة محفوظة — اضغط الزر مرة أخرى لاستكمال الناقص فقط.")

    # ---------------- results ----------------
    st.markdown('<div class="sec">📦 المسودة</div>', unsafe_allow_html=True)
    for name in ORDER:
        _, label, _ = TASKS[name]
        with st.expander(f"{TASK_ICON[name]} {label} {'✅' if S.get(name) else ''}", expanded=bool(S.get(name))):
            c1, c2 = st.columns([3, 1])
            extra = c1.text_input("ملاحظة للتعديل (اختياري)", key=f"x_{name}")
            if c2.button("🔄 إعادة توليد هذا الجزء فقط", key=f"b_{name}", disabled=not can_run or running):
                start_job([name], False, extra)
                st.rerun()
            if running:
                c1.caption("⏳ جاري التوليد في الخلفية...")
            render(name)

    # ---------------- review ----------------
    st.markdown('<div class="sec">🛡️ المراجعة</div>', unsafe_allow_html=True)
    if st.button("إعادة المراجعة", disabled=not has_engine or running, key="re_validate"):
        start_job([], True)
        st.rerun()
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
    conf = S.get("confidence")
    if conf and (conf["sections"] or conf["verse"]):
        LV = {"high": "🟢 عالية", "medium": "🟡 متوسطة — تحتاج مراجعة", "low": "🔴 منخفضة — استُبعدت من التصدير"}
        with st.expander("📊 تقرير الثقة في المعلومات (الاستبعاد تلقائي للمنخفضة)"):
            st.caption("التقييم محسوب من جودة المصدر وتطابق النص معه، وليس من رأي النموذج وحده.")
            if conf["verse"]:
                st.write(f"**آية الدرس:** {LV[conf['verse']]}")
            for s_ in conf["sections"]:
                st.write(f"**{s_['heading'] or 'قسم'}:** {LV[s_['level']]} — {s_['reason']}")

    # ---------------- export (teacher approval controls the draft notice) ----------------
    st.markdown('<div class="sec">📤 التصدير</div>', unsafe_allow_html=True)
    has_content = any(S.get(k) for k in TASKS)
    if has_content:
        st.checkbox("✅ راجعت المحتوى بنفسي وأقرّه للاستخدام (بدونها يُكتب على الملف «مسودة تحتاج مراجعة»)", key="approved")
    sets = export_sets(S, bool(ss["approved"]))
    if S.get("gate") == "blocked":
        st.error("المحتوى مرفوض في المراجعة — التصدير يتضمن تنبيهًا. عدّله أولًا.")
    if not sets:
        st.caption("لا يوجد محتوى للتصدير بعد.")
    for title, blocks in sets.items():
        bj = json.dumps(blocks, ensure_ascii=False)
        c1, c2, c3 = st.columns([2, 1, 1])
        c1.write(f"**{title}**")
        c2.download_button("DOCX", _docx_cached(bj, ss.get("logo", b"")), f"{title}.docx", key=f"d_{title}")
        try:
            c3.download_button("PDF", _pdf_cached(bj, ss.get("logo", b"")), f"{title}.pdf", key=f"p_{title}")
        except Exception as e:
            c3.caption("PDF: " + str(e)[:120])

    # ---------------- lesson library ----------------
    st.markdown('<div class="sec">🗂️ مكتبة الدروس</div>', unsafe_allow_html=True)
    st.button("💾 حفظ هذا التحضير في المكتبة", disabled=not has_content, on_click=save_to_library, key="lib_save")
    for e_ in reversed(ss["library"]):
        l1, l2, l3 = st.columns([4, 1, 1])
        l1.write(f"**{e_['title'] or 'بدون عنوان'}** — {e_.get('date') or e_['saved_at'][:10]}")
        l2.button("فتح", key=f"lo_{e_['id']}", on_click=open_from_library, args=(e_["id"],), disabled=running)
        l3.button("حذف", key=f"ld_{e_['id']}", on_click=delete_from_library, args=(e_["id"],))
    if not ss["library"]:
        st.caption("لا توجد تحضيرات محفوظة بعد.")

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
            chunks, o = retrieve_for(S, kb, msg)
            hist = [{"role": m["role"], "content": m["content"]} for m in S["chat"][-8:]]
            sysmsg = CHAT_SYSTEM + "\n\nالمراجع المسترجعة:\n" + build_context(chunks, o)
            ans = guarded(lambda: provider().chat([{"role": "system", "content": sysmsg}] + hist), "...")
            if ans:
                S["chat"].append({"role": "assistant", "content": ans})
                st.chat_message("assistant").markdown(ans)

    # ---------------- web sources ----------------
    def _fetch_view():
        j = ss["fjob"]
        st.progress(j.fraction(), text=j.text)
        if j.finished:
            st.rerun()

    with st.expander("🌐 المراجع الإلكترونية"):
        st.caption("المصادر المسجلة تُجلب تلقائيًا في الخلفية أول مرة، وتُحدَّث إذا قدمت. الجلب يحترم robots.txt ولا يزحف الموقع كاملًا.")
        mp = st.slider("أقصى عدد صفحات لكل مصدر", 1, 10, 3)
        if st.button("⬇️ جلب/تحديث المصادر المسجلة", disabled=fetching):
            start_fetch(REGISTERED_SOURCES, True, mp)
            st.rerun()
        if fetching:
            if hasattr(st, "fragment"):
                st.fragment(run_every=2)(_fetch_view)()
            else:
                st.caption("⏳ " + fjob.text)
        elif fjob:
            st.caption(fjob.text)
        stt = kb.stats()
        st.write(f"صفحات ناجحة: **{stt['ok']}** — فاشلة/غير متاحة: **{stt['failed']}** — مقاطع: **{stt['chunks']}**")
        st.caption("هذه الأرقام تخص الصفحات التي جُلبت فقط، ولا تعني فهرسة المواقع بالكامل.")
        for url, p in kb.snapshot():
            c1, c2, c3 = st.columns([5, 1, 1])
            stale = " ⏳ قديم" if kb.is_stale(p) else ""
            mine = " 👤 (جلستك فقط)" if p.get("session") else ""
            c1.write(f"{'✅' if p['status']=='ok' else '⚠️'} {p['name']} — {(p['title'] or url)[:60]}{stale}{mine}  \n`{url}`"
                     + (f"  \n{p['error']}" if p["error"] else ""))
            if c2.button("تحديث", key="r_" + url, disabled=fetching):
                start_fetch([{"url": url, "kind": p["kind"], "name": p["name"]}], False, 1, bool(p.get("session")))
                st.rerun()
            if c3.button("حذف", key="rm_" + url):
                kb.remove(url)
                st.rerun()
        st.divider()
        st.text_input("إضافة رابط مرجع (صفحة ويب أو ملف PDF/DOCX)", key="link_url", placeholder="https://...")
        st.checkbox("اجلب أيضًا الصفحات القريبة في نفس القسم (حتى 3 صفحات)", key="link_follow")
        st.button("إضافة", on_click=add_link, disabled=fetching)
        lm = ss.pop("link_msg", None)
        if lm:
            {"ok": st.success, "warn": st.warning, "err": st.error}[lm[0]](lm[1])
        unreg = [s["name"] for s in REGISTERED_SOURCES if s["url"] not in kb.pages]
        if unreg:
            st.caption("مصادر مسجلة لم تُجلب بعد: " + "، ".join(unreg))

    st.markdown('<div class="foot">✝️ للخدمة — كل محتوى مُولَّد مسودة تحتاج مراجعة الخادم المسؤول قبل الاستخدام</div>',
                unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 8. SELF-TESTS  (python app.py --selftest) — all HTTP is mocked
# ════════════════════════════════════════════════════════════════════════════
def run_selftests() -> int:
    import tempfile
    import unittest

    class R:
        def __init__(self, code=200, js=None, text="", headers=None, content=b""):
            self.status_code, self._js, self.text, self.headers = code, js, text, headers or {}
            self.content = content
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
        def setUp(self):
            _LAST_CALL["t"] = 0.0          # pacing is global now -> isolate every test

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

        def test_pacing_shared_across_providers(self):
            waits = []
            h = FakeHTTP([ok_resp("1"), ok_resp("2")])
            P("k", M, http=h, sleep=waits.append, min_interval=5.0).chat(MSG)
            P("k", M, http=h, sleep=waits.append, min_interval=5.0).chat(MSG)   # a NEW provider must still wait
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
            return KnowledgeBase(os.path.join(d, "kb.json"), http=FakeHTTP(seq), sleep=nosleep, delay=0, check_url=lambda u: (True, ""))

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

        def test_quote_regex_does_not_span_styles(self):
            c = [Chunk("نص مرجعي", "t", "u", "u", "", "biblical", "s")]
            txt = "«قصير» ثم كلام طويل بين الاقتباسين لا علاقة له بشيء « وهذا اقتباس آخر مختلف تماما عن الأول هنا »"
            out = unverified_quotes(txt, c)
            self.assertEqual(out, ["وهذا اقتباس آخر مختلف تماما عن الأول هنا"])

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

        def test_parallel_run_task_keeps_all_parts(self):
            from concurrent.futures import ThreadPoolExecutor
            S = new_state()
            S["input"] = "درس عن الصدق"
            kb = self._kb([])
            quiz = json.dumps({"title": "ق", "questions": []}, ensure_ascii=False)

            def go(n, out):
                p, _ = self._prov([out])
                run_task(n, S, kb, p)
            with ThreadPoolExecutor(max_workers=2) as ex:
                list(ex.map(lambda a: go(*a), [("activities", self.ACT), ("quiz", quiz)]))
            self.assertTrue(S["activities"] and S["quiz"])

        def test_scrub_leaks(self):
            flags = []
            out = scrub_leaks({"a": "حسب (source_tag: منهج) ثم [review_flags: التأكد من الدقة.] نص",
                               "b": ["عنوان الدرس (source_tag)", "عادي"]}, flags)
            blob = json.dumps(out, ensure_ascii=False)
            self.assertNotIn("source_tag", blob)
            self.assertNotIn("review_flags", blob)
            self.assertEqual(flags, ["التأكد من الدقة."])
            self.assertEqual(out["b"][1], "عادي")

        def test_task_output_is_scrubbed(self):
            leaky = json.dumps({"items": [{"title": "لعبة (source_tag: إبداع)", "description": "د [review_flags: تأكد]",
                                           "steps": ["1"]}]}, ensure_ascii=False)
            S = new_state()
            S["input"] = "درس عن الصدق"
            p, _ = self._prov([leaky])
            run_task("activities", S, self._kb([]), p)
            it = S["activities"]["items"][0]
            self.assertEqual((it["title"], it["description"]), ("لعبة", "د"))
            self.assertIn("تأكد", S["activities"]["review_flags"])

        def test_background_job(self):
            S = new_state()
            S["input"] = "درس عن الصدق"
            job = Job(1)
            run_job(job, S, self._kb([]), lambda: self._prov([self.ACT])[0], ["activities"], False, True)
            self.assertTrue(job.finished and job.ok and S["activities"])
            self.assertEqual(job.fraction(), 1.0)
            bad = Job(1)
            run_job(bad, S, self._kb([]), lambda: LLMProvider([]), ["quiz"], False, True)
            self.assertTrue(bad.finished and not bad.ok and "quiz" in bad.errors)

        def test_curriculum_style_lesson(self):
            lesson = json.dumps({
                "title": "الكنيسة", "main_idea": "الكنيسة هي ملكوت السموات على الأرض",
                "objectives": ["يعرف قدسية الكنيسة", "يشعر بحضور السمائيين"],
                "memory_verse": "أَمَّا أَنَا فَبِكَثْرَةِ رَحْمَتِكَ أَدْخُلُ بَيْتَكَ (مز 5: 7)",
                "intro_questions": ["ايه هي الكنيسة؟", "ايه هو شكل كنيستنا؟"],
                "sections": [{"heading": "أشكال الكنيسة",
                              "body": "1. سفينة: بتفكرنا بفلك نوح\n2. صليب: لأن ربنا فدانا على الصليب\n- تفصيلة صغيرة"}],
            }, ensure_ascii=False)
            S = new_state()
            S["input"] = "درس عن الكنيسة"
            p, _ = self._prov([lesson])
            run_task("lesson", S, self._kb([]), p)
            B = component_blocks("lesson", S["lesson"])
            texts = [t for _, t in B]
            order = [texts.index(x) for x in ("❖ هدف الدرس", "❖ آية الدرس", "❖ أسئلة تمهيدية للدرس", "❖ الدرس")]
            self.assertEqual(order, sorted(order))
            self.assertIn("مساعدة الطفل أن:", texts)
            self.assertIn(("p", "1. سفينة: بتفكرنا بفلك نوح"), B)
            self.assertIn(("li", "تفصيلة صغيرة"), B)
            # old-format lesson data still renders
            old = component_blocks("lesson", Lesson(title="درس", objectives=["أ"], sections=[LessonSection(heading="ع", body="نص")]).model_dump())
            self.assertIn(("h1", "درس"), old)

        def test_url_validation(self):
            res = lambda ip: (lambda host, port: [(2, 1, 6, "", (ip, 0))])
            self.assertFalse(safe_public_url("ftp://a.org/x")[0])
            self.assertFalse(safe_public_url("not a url")[0])
            self.assertFalse(safe_public_url("http://localhost/x")[0])
            self.assertFalse(safe_public_url("https://a.org", res("10.0.0.5"))[0])
            self.assertFalse(safe_public_url("https://a.org", res("169.254.169.254"))[0])
            self.assertFalse(safe_public_url("https://a.org", res("127.0.0.1"))[0])
            self.assertTrue(safe_public_url("https://a.org", res("93.184.216.34"))[0])

            def boom(h, p):
                raise OSError("nx")
            self.assertFalse(safe_public_url("https://nx.invalid", boom)[0])

        def test_kb_blocks_unsafe_url_without_request(self):
            d = tempfile.mkdtemp()
            h = FakeHTTP([])
            kb = KnowledgeBase(os.path.join(d, "kb.json"), http=h, sleep=nosleep, delay=0)
            kb.check_url = lambda u: safe_public_url(u, lambda host, port: [(2, 1, 6, "", ("10.1.1.1", 0))])
            kb.ingest({"url": "https://intranet.example/a.html", "kind": "general", "name": "x"}, False, 1)
            self.assertEqual(h.calls, [])
            self.assertEqual(kb.pages["https://intranet.example/a.html"]["status"], "unavailable")

        def test_session_link_not_persisted(self):
            kb = self._kb([R(404), R(200, text=HTML)])
            url = "https://x.org/a.html"
            kb.ingest({"url": url, "kind": "general", "name": "x"}, False, 1, session=True)
            self.assertTrue(kb.pages[url].get("session") and kb.pages[url]["status"] == "ok")
            kb2 = KnowledgeBase(kb.cache_path, http=FakeHTTP([]), sleep=nosleep, delay=0)
            self.assertNotIn(url, kb2.pages)

        def test_shared_page_persisted(self):
            kb = self._kb([R(404), R(200, text=HTML)])
            url = "https://x.org/a.html"
            kb.ingest({"url": url, "kind": "feasts", "name": "x"}, False, 1)
            kb2 = KnowledgeBase(kb.cache_path, http=FakeHTTP([]), sleep=nosleep, delay=0)
            self.assertEqual(kb2.pages[url]["status"], "ok")

        def test_pdf_link_dispatch(self):
            txt = "\n".join(["الكنيسة بيت الله ونحن نجتمع فيها لنصلي لربنا كل أسبوع في يوم الأحد"] * 3)
            real = globals()["extract_doc_text"]
            globals()["extract_doc_text"] = lambda name, data: (txt, "")
            try:
                kb = self._kb([R(404), R(200, headers={"Content-Type": "application/pdf"})])
                kb.ingest({"url": "https://x.org/lesson.pdf", "kind": "general", "name": "x"}, False, 1)
            finally:
                globals()["extract_doc_text"] = real
            p = kb.pages["https://x.org/lesson.pdf"]
            self.assertEqual(p["status"], "ok")
            self.assertEqual(p["chunks"][0]["origin"], "web")

        def test_stale_failed_retried_after_a_day(self):
            kb = self._kb([])
            recent = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2)).isoformat(timespec="seconds")
            self.assertFalse(kb.is_stale({"status": "ok", "retrieved_at": old}))
            self.assertFalse(kb.is_stale({"status": "unavailable", "retrieved_at": recent}))
            self.assertTrue(kb.is_stale({"status": "unavailable", "retrieved_at": old}))

        def test_fetch_job(self):
            kb = self._kb([R(404), R(200, text=HTML)])
            job = Job(1)
            run_fetch(job, kb, [{"url": "https://x.org/a.html", "kind": "general", "name": "x"}], False, 1)
            self.assertTrue(job.finished and job.ok)
            self.assertEqual(kb.stats()["ok"], 1)

        def test_safety_word_boundary(self):
            self.assertEqual(local_safety_scan("بنينا منارة جميلة وقصة نارية"), [])
            self.assertTrue(local_safety_scan("نشعل النار في الساحة"))
            self.assertTrue(local_safety_scan("نستخدم ماء مغلي"))

        def test_fire_in_lesson_text_not_blocked_but_in_activity_is(self):
            val = json.dumps({"biblical_alignment_ok": True, "safety_ok": True, "confidence": 0.9, "issues": []})
            S = new_state()
            S["lesson"] = {"title": "العنصرة", "sections": [{"heading": "ألسنة من نار", "body": "حل الروح القدس"}]}
            p, _ = self._prov([val])
            run_validation(S, p)
            self.assertEqual(S["gate"], "ok")

        def test_short_curriculum_sent_whole(self):
            kb = self._kb([])
            S = new_state()
            S["input"] = "الصدق"          # a title that matches nothing in the curriculum text
            cur = [Chunk("نص عن الكنيسة بيت الله", "c", "local:c", "local:c", "", "sunday_school", "s", origin="curriculum"),
                   Chunk("تقسيمة الكنيسة من الداخل", "c", "local:c", "local:c", "", "sunday_school", "s", origin="curriculum")]
            S["curriculum_chunks"] = cur
            ch, _ = retrieve_for(S, kb)
            self.assertEqual(ch, cur)

        def test_export_draft_notice_until_approved(self):
            S = new_state()
            S["lesson"] = Lesson(title="درس", sections=[LessonSection(heading="ع", body="نص")]).model_dump()
            self.assertEqual(export_sets(S, False)["الدرس"][0][0], "p")
            self.assertIn("مسودة", export_sets(S, False)["الدرس"][0][1])
            self.assertEqual(export_sets(S, True)["الدرس"][0], ("h1", "درس"))

        def test_exports(self):
            S = new_state()
            S["lesson"] = Lesson(title="درس", objectives=["أ"], sections=[LessonSection(heading="ع", body="نص عربي")]).model_dump()
            S["quiz"] = Quiz(title="اختبار", questions=[QuizQ(question="س", options=["أ", "ب"], answer_index=1, explanation="ش")]).model_dump()
            sets = export_sets(S)
            self.assertIn("مفتاح الإجابة (للخادم فقط)", sets)
            self.assertFalse(any("الإجابة الصحيحة" in t for _, t in sets["أوراق الأطفال"]))
            try:
                import docx  # noqa
                bj = json.dumps(sets["الدرس"], ensure_ascii=False)
                self.assertTrue(blocks_to_docx(sets["الدرس"]).startswith(b"PK"))
                self.assertTrue(_docx_cached(bj).startswith(b"PK"))
                self.assertIs(_docx_cached(bj), _docx_cached(bj))      # served from cache
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

        def test_coptic_date(self):
            self.assertEqual(gregorian_to_coptic(dt.date(2026, 9, 11)), (1743, 1, 1))
            self.assertEqual(gregorian_to_coptic(dt.date(2026, 1, 7)), (1742, 4, 29))

        def test_schedule_xlsx_openpyxl(self):
            from openpyxl import Workbook
            book = Workbook()
            sheet = book.active
            sheet.append(["التاريخ", "اسم الموضوع", "مسؤول اليوم", "نوع الموضوع"])
            sheet.append(["2026-11-08", "درس الكنيسة", "مينا", "سلوكي"])
            stream = io.BytesIO()
            book.save(stream)
            rows, error = parse_schedule("schedule.xlsx", stream.getvalue())
            self.assertEqual(error, "")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["title"], "درس الكنيسة")
            self.assertEqual(rows[0]["date"], "2026-11-08")

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

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(T)
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(run_selftests())
    main_app()
