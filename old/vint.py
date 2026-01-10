"""
vint.py (helpers only)
----------------------
Legacy helper module used by features/* while you migrate.

IMPORTANT:
- No Streamlit UI here (no st.set_page_config, no tabs, no rendering)
- No DB migrations executed on import
- Pure functions + lightweight constants only
"""

from __future__ import annotations

import os
import io
import re
import csv
import hashlib
import tempfile
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz
from unidecode import unidecode
from sqlmodel import Session, select

# ✅ Single source of truth for models
from domain.models import Product, Order, OrderLine

# ✅ Prefer your existing core DB session/engine
from core.db import engine


def get_session() -> Session:
    return Session(engine)


# =========================
# Parsing constants
# =========================

NUM_WORDS = {
    # English
    "zero": 0, "one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
    # Spanish
    "cero": 0, "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12,
    "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16, "dieciséis": 16,
    "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20,
    # Greek (accented + unaccented)
    "μηδεν": 0, "μηδέν": 0, "ενα": 1, "ένα": 1, "μια": 1, "μία": 1, "ενας": 1, "ένας": 1,
    "δυο": 2, "δύο": 2, "τρια": 3, "τρία": 3, "τεσσερα": 4, "τέσσερα": 4,
    "πεντε": 5, "πέντε": 5, "εξι": 6, "έξι": 6, "επτα": 7, "επτά": 7,
    "οκτω": 8, "οκτώ": 8, "εννια": 9, "εννιά": 9, "δεκα": 10, "δέκα": 10,
    "εντεκα": 11, "έντεκα": 11, "δωδεκα": 12, "δώδεκα": 12,
    "δεκατρια": 13, "δεκατρία": 13, "δεκατεσσερα": 14, "δεκατέσσερα": 14,
    "δεκαπεντε": 15, "δεκαπέντε": 15, "δεκαεξι": 16, "δεκαέξι": 16,
    "δεκαεπτα": 17, "δεκαεπτά": 17, "δεκαοκτω": 18, "δεκαοκτώ": 18,
    "δεκαεννια": 19, "δεκαεννιά": 19, "εικοσι": 20, "είκοσι": 20,
}

UNITS = {
    "unidad", "unidades", "botella", "botellas", "kg", "kilo", "kilos", "gramo", "gramos", "g",
    "lata", "latas", "caja", "cajas", "pack", "packs", "u", "ud", "uds", "paquete", "paquetes",
    "κιλο", "κιλά", "κιλα", "γραμμάριο", "γραμμάρια",
    "τεμάχιο", "τεμαχιο", "τεμάχια",
    "μπουκάλι", "μπουκαλι", "μπουκάλια", "φιάλη", "φιαλη",
}

NEXT_SEPARATORS = [" next ", " siguiente ", " sig ", ",", ";", " y ", " and ", "also ", "tambien "]

GREETINGS = {
    "hola", "hello", "hi", "hey", "buenosdias", "buenastardes", "buenasnoches",
    "γεια", "γειά", "γεια σου", "καλημερα", "καλημέρα", "καλησπερα", "καλησπέρα",
    "καληνυχτα", "καληνύχτα",
}

FILLER_PREFIX = re.compile(
    r"^\s*(?:quiero|ponme|pon|me pones|dame|trae(?:me)?|tráeme|anade|añade|agrega|mete|suma|sumar|agregar)\s+",
    flags=re.IGNORECASE,
)

# =========================
# Small generic helpers
# =========================

def safe_str(x, default: str = "") -> str:
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    return str(x)

def str_or_default(x, default="") -> str:
    try:
        if x is pd.NA or pd.isna(x):
            return default
    except Exception:
        pass
    s = str(x).strip()
    return s if s else default

def normalize_key(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "").strip().lower())

def distinct_units(products: List[Product]) -> List[str]:
    units = sorted({normalize_key(getattr(p, "unit", "") or "") for p in products if getattr(p, "unit", None)})
    units = [u for u in units if u]
    return units or ["unidad", "kg", "g", "l", "caja", "botella"]

def num_or_default(x, default=0.0):
    try:
        if x is pd.NA or pd.isna(x):
            return default
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return default

def coalesce_unit(x, prod_obj=None, default="unidad"):
    s = str_or_default(x, "")
    if s:
        return s.lower()
    if prod_obj is not None:
        p = getattr(prod_obj, "unit", None)
        p = str_or_default(p, "")
        if p:
            return p.lower()
    return default


# =========================
# Normalization + parsing
# =========================

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def singularize_es(word: str) -> str:
    w = word.strip()
    if len(w) <= 3:
        return w
    if w.endswith("es") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = unidecode(text.strip().lower())
    rules = [
        (r"([^aeiou])ies$", r"\1y"),
        (r"([aeiou])s$", r"\1"),
        (r"([nrlsdz])es$", r"\1"),
        (r"s$", ""),
    ]
    for pattern, repl in rules:
        if re.search(pattern, text):
            text = re.sub(pattern, repl, text)
            break
    return text

def tokenize_items(raw_text: str) -> List[str]:
    text = unidecode((raw_text or "").lower())
    text = re.sub(r"buenos\s+d[ií]as", "buenosdias", text)
    text = re.sub(r"buenas\s+tardes", "buenastardes", text)
    text = re.sub(r"buenas\s+noches", "buenasnoches", text)

    sep_text = text
    for tok in NEXT_SEPARATORS:
        sep_text = sep_text.replace(tok, " | ")
    sep_text = re.sub(r"\s+y\s+", " | ", sep_text)

    parts = [p.strip(" |\t\n,") for p in sep_text.split("|")]
    parts = [p for p in parts if p and p not in GREETINGS and len(p) > 1]
    return parts

def strip_filler_prefix(text: str) -> str:
    t = text or ""
    while True:
        t2 = FILLER_PREFIX.sub("", t)
        if t2 == t:
            break
        t = t2
    return t.strip()

def replace_number_words(text: str) -> str:
    choices = "|".join(sorted(NUM_WORDS.keys(), key=len, reverse=True))
    patt = re.compile(rf"\b({choices})\b", flags=re.IGNORECASE)

    def repl(m):
        w = m.group(1).lower()
        return str(NUM_WORDS.get(w, w))

    return patt.sub(repl, text)

def parse_item(fragment: str) -> Optional[Tuple[str, float, Optional[str]]]:
    frag = (fragment or "").strip().lower()
    frag = unidecode(frag)
    if not frag:
        return None

    frag = re.sub(r"buenos\s+d[ií]as", "buenosdias", frag)
    frag = re.sub(r"buenas\s+tardes", "buenastardes", frag)
    frag = re.sub(r"buenas\s+noches", "buenasnoches", frag)

    frag = strip_filler_prefix(frag)
    frag = replace_number_words(frag)
    frag = re.sub(r"^\s*(?:de|del|la|el|los|las)\s+", "", frag)

    only_letters = re.sub(r"[^a-z\s]", " ", frag).strip()
    if only_letters in GREETINGS:
        return None

    # qty first
    m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s+(.*)$", frag)
    if m:
        qty_s = m.group(1)
        rest = m.group(2).strip()

        parts = re.split(r"\s+", rest)
        unit: Optional[str] = None
        name_part = rest

        if parts and parts[0] in UNITS:
            unit = parts[0]
            name_part = " ".join(parts[1:]).strip()
            name_part = re.sub(r"^(?:de|del)\s+", "", name_part)

        name = normalize_text(name_part)
        name = re.sub(r"[^a-z\s]", " ", name).strip()
        name = re.sub(r"\s+", " ", name)

        if name in GREETINGS or len(name) < 2:
            return None

        try:
            qty = float(qty_s.replace(",", "."))
        except Exception:
            qty = 1.0
        if qty <= 0:
            qty = 1.0

        return (name, qty, unit if unit in UNITS else None)

    # qty last
    m2 = re.match(r"^(.*\D)\s+(\d+(?:[.,]\d+)?)\s*$", frag)
    if m2:
        name_part = m2.group(1).strip()
        qty_s = m2.group(2)

        unit: Optional[str] = None
        parts = re.split(r"\s+", name_part)
        if parts and parts[0] in UNITS:
            unit = parts[0]
            name_part = " ".join(parts[1:]).strip()
            name_part = re.sub(r"^(?:de|del)\s+", "", name_part)

        name = normalize_text(name_part)
        name = re.sub(r"[^a-z\s]", " ", name).strip()
        name = re.sub(r"\s+", " ", name)

        if name in GREETINGS or len(name) < 2:
            return None

        try:
            qty = float(qty_s.replace(",", "."))
        except Exception:
            qty = 1.0
        if qty <= 0:
            qty = 1.0

        return (name, qty, unit if unit in UNITS else None)

    # name only
    name_only = re.sub(r"[^a-z\s]", " ", frag).strip()
    name_only = re.sub(r"\s+", " ", name_only)
    if len(name_only) >= 2:
        name_only = normalize_text(name_only)
        if name_only in GREETINGS or len(name_only) < 2:
            return None
        return (name_only, 1.0, None)

    return None

def fuzzy_match(name: str, catalog: List[str]) -> Tuple[Optional[str], float]:
    if not catalog:
        return None, 0.0
    if name in GREETINGS or len(name) < 3:
        return None, 0.0

    stripped = re.sub(r"\b(the|a|an|de|del|la|el|los|las)\b", " ", name).strip()
    best = process.extractOne(stripped or name, catalog, scorer=fuzz.WRatio)
    if best is None:
        return None, 0.0

    match_name, score, _ = best
    if score < 88.0:
        return None, 0.0

    return match_name, float(score)


# =========================
# DataFrame helper
# =========================

def add_provider_column(sess: Session, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "provider" not in df.columns:
        df["provider"] = pd.NA

    prods = sess.exec(select(Product)).all()

    def _key(x: str) -> str:
        return unidecode(str(x or "").strip().lower())

    name_to_provider = {
        _key(getattr(p, "name", "")): (getattr(p, "provider_name", "") or "")
        for p in prods
        if p is not None
    }

    def _prov(matched_name):
        if matched_name is pd.NA or matched_name is None:
            return ""
        return name_to_provider.get(_key(matched_name), "")

    df["provider"] = df["matched_name"].apply(_prov).astype("string")
    return df


# =========================
# Audio input + ASR
# =========================

def mic_or_upload_audio(label: str, key: str, sample_rate: int = 16000):
    if hasattr(st, "audio_input"):
        try:
            return st.audio_input(label, key=key, sample_rate=sample_rate)
        except TypeError:
            return st.audio_input(label, key=key)

    return st.file_uploader(
        "🎙️ Tu Streamlit no soporta grabación. Sube un audio (wav/mp3/m4a)",
        type=["wav", "mp3", "m4a", "ogg"],
        key=f"{key}_upload",
    )

def read_audio_bytes(audio_obj) -> bytes:
    if audio_obj is None:
        return b""
    if hasattr(audio_obj, "getvalue"):
        return audio_obj.getvalue()
    if hasattr(audio_obj, "read"):
        return audio_obj.read()
    return bytes(audio_obj)

def asr_openai_whisper(audio_bytes: bytes, vocab, language: str = "es") -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Missing dependency: pip install openai") from e

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    prompt = (
        "Transcribe pedidos de productos HORECA. Mantén cantidades y unidades. "
        "Productos/marcas frecuentes: " + ", ".join(vocab[:120])
    )

    bio = io.BytesIO(audio_bytes)
    bio.name = "audio.wav"

    out = client.audio.transcriptions.create(
        file=bio,
        model="whisper-1",
        language=language,
        prompt=prompt,
        temperature=0,
    )
    return (out.text or "").strip()

def asr_faster_whisper(audio_bytes: bytes, vocab, language: str = "es") -> str:
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise RuntimeError("Missing dependency: pip install faster-whisper") from e

    if "fw_model_small" not in st.session_state:
        st.session_state["fw_model_small"] = WhisperModel(
            "small",
            device="cpu",
            compute_type="int8",
        )

    model = st.session_state["fw_model_small"]

    initial_prompt = (
        "Pedido HORECA en español. Mantén cantidades y unidades. "
        "Separador: siguiente. Productos y marcas: " + ", ".join(vocab[:120])
    )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        segments, _info = model.transcribe(
            tmp_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            initial_prompt=initial_prompt,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
