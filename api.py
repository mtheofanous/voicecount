# Voice Inventory — Modern Streamlit App (single file)
# ==================================================
# 🚀 Highlights
# - Clean two-panel layout with sidebar navigation and helpful tooltips
# - Products DB: add/edit/delete, CSV import/export, filtering & quick stats
# - Voice Order: browser mic (WebRTC component) *or* local mic (sounddevice)
# - Whisper: choose OpenAI API or Faster-Whisper (cached) with vocab bias from your catalog
# - Robust parsing, fuzzy match, aggregation, and provider-ready Email/WhatsApp exports
# - Session-safe, graceful fallbacks if DB/helper modules are missing
#
# ▶ Run locally
#   pip install -U streamlit sqlmodel rapidfuzz unidecode python-dateutil faster-whisper torch numpy sounddevice pydub streamlit-audiorecorder openai
#   streamlit run app.py
#
# Notes
# - Set env var OPENAI_API_KEY if you want the OpenAI Whisper API option.
# - Local mic mode (sounddevice) requires microphone access on the machine running Streamlit.

from __future__ import annotations
import os
import io
import re
import csv
import json
import tempfile
import wave
from datetime import datetime, date
from typing import List, Optional, Tuple, Dict
from collections import defaultdict
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import streamlit as st
from sqlmodel import SQLModel, Field, Session, create_engine, select
from rapidfuzz import process, fuzz
from unidecode import unidecode

# Optional: browser mic component
try:
    from audiorecorder import audiorecorder  # returns pydub.AudioSegment
    from pydub import AudioSegment
except Exception:
    audiorecorder = None
    AudioSegment = None

# Optional: Faster-Whisper & local mic
try:
    from faster_whisper import WhisperModel
    import sounddevice as sd
except Exception:
    WhisperModel = None
    sd = None
load_dotenv()
# -------------------------
# Global App Config & Styles
# -------------------------

st.set_page_config(page_title="Voice Inventory", page_icon="🎤", layout="wide")

CSS = """
<style>
/**** Subtle modern look ****/
.block-container {padding-top: 1rem;}
section[data-testid="stSidebar"] {width: 320px !important;}
/* Tweak tables */
div[data-testid="stDataFrame"] {border-radius: 14px;}
/* Buttons */
.stButton>button {border-radius: 12px; padding: 0.5rem 0.9rem;}
/* Info boxes */
.stAlert {border-radius: 14px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -------------------------
# Database Models
# -------------------------

SQLModel.metadata.clear()

class Product(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    category: Optional[str] = None
    unit: Optional[str] = Field(default="unidad")  # e.g., botella, kg, caja, lata, unidad
    quantity: Optional[float] = Field(default=0)    # current stock (optional)

    # Provider fields
    provider_name: Optional[str] = None
    provider_email: Optional[str] = None
    provider_phone: Optional[str] = None
    provider_address: Optional[str] = None

    created_at: date = Field(default_factory=lambda: date.today())

class Order(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now())
    note: Optional[str] = None
    
    status: str = Field(default="draft")
    title: Optional[str] = Field(default=None)

class OrderLine(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    order_id: int = Field(foreign_key="order.id")
    product_id: Optional[int] = Field(default=None, foreign_key="product.id")

    # Captured data
    spoken_name: str
    matched_name: Optional[str] = None
    confidence: Optional[float] = None

    # Quantity & unit
    quantity: float
    unit: Optional[str] = None
    


# -------------------------
# DB Engine & Session
# -------------------------

if "engine" not in st.session_state:
    engine = create_engine("sqlite:///voice_inventory.db")
    SQLModel.metadata.create_all(engine)
    st.session_state.engine = engine

def get_session() -> Session:
    return Session(st.session_state.engine)

# -------------------------
# Normalization & Helpers
# -------------------------

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
    # ES/EN
    "unidad", "unidades", "botella", "botellas", "kg", "kilo", "kilos", "gramo", "gramos", "g",
    "lata", "latas", "caja", "cajas", "pack", "packs", "u", "ud", "uds", "paquete", "paquetes",
    # Greek
    "κιλο", "κιλά", "κιλα", "γραμμάριο", "γραμμάρια",
    "τεμάχιο", "τεμαχιο", "τεμάχια",
    "μπουκάλι", "μπουκαλι", "μπουκάλια", "φιάλη", "φιαλη"
}


GREETINGS = {
    # ES/EN
    "hola", "hello", "hi", "hey", "buenosdias", "buenastardes", "buenasnoches",
    # Greek
    "γεια", "γειά", "γεια σου", "καλημερα", "καλημέρα", "καλησπερα", "καλησπέρα",
    "καληνυχτα", "καληνύχτα"
}



FILLER_PREFIX = re.compile(
    r"^\s*(?:quiero|ponme|pon|me pones|dame|trae(?:me)?|tráeme|anade|añade|agrega|mete|suma|sumar|agregar)\s+",
    flags=re.IGNORECASE,
)

NEXT_SEPARATORS = [" next ", " siguiente ", " sig ", ",", ";", " y ", " and ", "also ", "tambien "]

def str_or_default(x, default=""):
    if x is pd.NA or pd.isna(x):
        return default
    s = str(x).strip()
    return s if s != "" else default

def num_or_default(x, default=0.0):
    if x is pd.NA or pd.isna(x):
        return default
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

def add_provider_column(sess, df: pd.DataFrame) -> pd.DataFrame:
    """Adds/refreshes df['provider'] based on df['matched_name'] -> Product.provider_name (case-insensitive)."""
    df = df.copy()
    if "provider" not in df.columns:
        df["provider"] = pd.NA

    res = sess.exec(select(Product))
    try:
        prods = res.all()          # SQLModel ScalarResult
    except Exception:
        prods = res.all()  # SQLAlchemy Result

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



def product_for_name(sess, name):
    if not name:
        return None
    return sess.exec(select(Product).where(Product.name == name)).first()




def coerce_to_df(obj, expected_cols):
    """Convierte obj a DataFrame y garantiza columnas/orden."""
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    elif isinstance(obj, (list, tuple)):
        # lista de dicts -> OK; si no, vacío con columnas esperadas
        if len(obj) == 0:
            df = pd.DataFrame(columns=expected_cols)
        elif isinstance(obj[0], dict):
            df = pd.DataFrame(obj)
        else:
            df = pd.DataFrame(columns=expected_cols)
    elif isinstance(obj, dict):
        # dict de listas/series (columnas) -> DataFrame; si no, dict fila única
        vals = list(obj.values())
        if vals and all(isinstance(v, (list, tuple, pd.Series)) for v in vals):
            try:
                df = pd.DataFrame(obj)
            except Exception:
                df = pd.DataFrame([obj])
        else:
            df = pd.DataFrame([obj])  # fila única
    else:
        df = pd.DataFrame(columns=expected_cols)

    # Garantiza columnas y orden
    for c in expected_cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[expected_cols]

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = unidecode(text.strip().lower())
    rules = [
        (r"([^aeiou])ies$", r"\1y"),     # berries -> berry
        (r"([aeiou])s$", r"\1"),         # kilos -> kilo
        (r"([nrlsdz])es$", r"\1"),       # limones -> limon, botellas -> botella
        (r"s$", ""),                      # packs -> pack
    ]
    for pattern, repl in rules:
        if re.search(pattern, text):
            text = re.sub(pattern, repl, text)
            break
    return text


def normalize_model(obj):
    for name, value in vars(obj).items():
        if isinstance(value, str):
            setattr(obj, name, normalize_text(value))


def replace_number_words(text: str) -> str:
    choices = "|".join(sorted(NUM_WORDS.keys(), key=len, reverse=True))
    patt = re.compile(rf"\b({choices})\b", flags=re.IGNORECASE)
    def repl(m):
        w = m.group(1).lower()
        return str(NUM_WORDS.get(w, w))
    return patt.sub(repl, text)


def strip_filler_prefix(text: str) -> str:
    t = text or ""
    while True:
        t2 = FILLER_PREFIX.sub("", t)
        if t2 == t:
            break
        t = t2
    return t.strip()


def tokenize_items(raw_text: str) -> List[str]:
    text = unidecode((raw_text or "").lower())

    # Normalize "buenos dias" etc. so we can filter reliably
    text = re.sub(r"buenos\s+d[ií]as", "buenosdias", text)
    text = re.sub(r"buenas\s+tardes", "buenastardes", text)
    text = re.sub(r"buenas\s+noches", "buenasnoches", text)

    sep_text = text
    for tok in NEXT_SEPARATORS:
        sep_text = sep_text.replace(tok, " | ")
    # Also split on standalone ' y ' (Spanish AND)
    sep_text = re.sub(r"\s+y\s+", " | ", sep_text)

    parts = [p.strip(" |\t\n,") for p in sep_text.split("|")]
    # Drop empty and greeting-only parts early
    parts = [p for p in parts if p and p not in GREETINGS and len(p) > 1]

    return parts



def parse_item(fragment: str) -> Optional[Tuple[str, float, Optional[str]]]:
    # Normalize
    frag = (fragment or "").strip().lower()
    frag = unidecode(frag)
    if not frag:
        return None

    # Normalize spaced greetings so we can filter reliably
    frag = re.sub(r"buenos\s+d[ií]as", "buenosdias", frag)
    frag = re.sub(r"buenas\s+tardes", "buenastardes", frag)
    frag = re.sub(r"buenas\s+noches", "buenasnoches", frag)

    # Remove filler at the very beginning (your helper)
    frag = strip_filler_prefix(frag)

    # Convert number words -> digits (your helper: e.g., “two” -> 2)
    frag = replace_number_words(frag)

    # Drop leading Spanish articles/preps
    frag = re.sub(r"^\s*(?:de|del|la|el|los|las)\s+", "", frag)

    # Early discard: greeting-only fragments (e.g., "hola")
    only_letters = re.sub(r"[^a-z\s]", " ", frag).strip()
    if only_letters in GREETINGS:
        return None

    # ----------------------------
    # Pattern 1: number-first
    # "<qty> <maybe-unit> [de] <name>"
    # ----------------------------
    m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s+(.*)$", frag)
    if m:
        qty_s = m.group(1)
        rest  = m.group(2).strip()

        parts = re.split(r"\s+", rest)
        unit: Optional[str] = None
        name_part = rest

        if parts:
            first = parts[0]
            if first in UNITS:
                unit = first
                name_part = " ".join(parts[1:]).strip()
                name_part = re.sub(r"^(?:de|del)\s+", "", name_part)

        # Keep only letters/spaces in name, normalize
        name = normalize_text(name_part)
        name = re.sub(r"[^a-z\s]", " ", name).strip()
        name = re.sub(r"\s+", " ", name)

        # Reject greetings / too-short names
        if name in GREETINGS or len(name) < 2:
            return None

        try:
            qty = float(qty_s.replace(",", "."))
        except Exception:
            qty = 1.0
        if qty <= 0:
            qty = 1.0

        return (name, qty, unit if unit in UNITS else None)

    # ----------------------------
    # Pattern 2: name-first + number-last
    # "<maybe-unit> <name> <qty>"
    # ----------------------------
    m2 = re.match(r"^(.*\D)\s+(\d+(?:[.,]\d+)?)\s*$", frag)
    if m2:
        name_part = m2.group(1).strip()
        qty_s     = m2.group(2)

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

    # ----------------------------
    # Pattern 3: fallback name-only
    # ----------------------------
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

    # Ignore fragments that are obviously not items
    if name in GREETINGS or len(name) < 3:
        return None, 0.0

    # Optional: strip very generic words before matching
    stripped = re.sub(r"\b(the|a|an|de|del|la|el|los|las)\b", " ", name).strip()

    best = process.extractOne(stripped or name, catalog, scorer=fuzz.WRatio)
    if best is None:
        return None, 0.0

    match_name, score, _ = best

    # Require a higher threshold to avoid spurious matches on short tokens
    MIN_SCORE = 88.0
    if score < MIN_SCORE:
        return None, 0.0

    return match_name, float(score)



def aggregate_parsed_rows(rows: List[Dict], name_to_product: Dict[str, Product], default_unit: str = "unidad") -> List[Dict]:
    by_name: Dict[str, Dict] = {}
    unit_count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        name_key = (r.get("matched_name") or r.get("spoken_name") or "").strip().lower()
        if not name_key:
            name_key = f"__raw__:{(r.get('spoken_name') or '').strip().lower()}"
        qty = float(r.get("quantity") or 1)
        unit = (r.get("unit") or "").strip().lower()
        conf = float(r.get("confidence") or 0.0)
        status = r.get("status") or ""
        if name_key not in by_name:
            by_name[name_key] = {
                "spoken_name": r.get("spoken_name"),
                "matched_name": r.get("matched_name"),
                "confidence": conf,
                "quantity": 0.0,
                "unit": "",
                "status": status or "Revisar",
            }
        by_name[name_key]["quantity"] += qty
        by_name[name_key]["confidence"] = max(by_name[name_key]["confidence"], conf)
        if by_name[name_key]["status"] != "OK" and status == "OK":
            by_name[name_key]["status"] = "OK"
        if unit:
            unit_count[name_key][unit] += 1
    for k, row in by_name.items():
        chosen_unit = ""
        if unit_count[k]:
            chosen_unit = max(unit_count[k].items(), key=lambda kv: kv[1])[0]
        if not chosen_unit:
            matched = (row.get("matched_name") or "").strip()
            prod = name_to_product.get(matched) if matched else None
            if prod and (prod.unit or "").strip():
                chosen_unit = prod.unit.strip().lower()
        if not chosen_unit:
            chosen_unit = default_unit
        row["unit"] = chosen_unit
        q = row["quantity"]
        row["quantity"] = int(q) if abs(q - int(q)) < 1e-9 else round(q, 2)
    return list(by_name.values())

# -------------------------
# Transcription backends
# -------------------------

@st.cache_resource(show_spinner=False)
def load_whisper(model_size: str = "small"):
    if WhisperModel is None:
        raise RuntimeError("Faster-Whisper not installed")
    return WhisperModel(model_size, device="auto", compute_type="auto")


def record_local(seconds: int = 6, samplerate: int = 16000) -> tuple[np.ndarray, int]:
    if sd is None:
        raise RuntimeError("sounddevice not installed")
    st.info(f"Grabando {seconds}s… Habla ahora.")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()
    st.success("Grabación completa.")
    return audio.reshape(-1), samplerate


def save_wav(path: str, audio: np.ndarray, samplerate: int):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())


def transcribe_faster_whisper(wav_path: str, model_size: str, lang_code: str | None, vocab_hint: List[str] | None) -> str:
    model = load_whisper(model_size)
    opts = {
        "vad_filter": True,
        "beam_size": 5,
        "best_of": 5,
        "without_timestamps": True,
        "temperature": 0.0,
        "no_speech_threshold": 0.5,
        "compression_ratio_threshold": 2.6,
        "condition_on_previous_text": False,
        "task": "transcribe",
        "patience": 0.2,  # speeds up for short orders
    }
    if lang_code and lang_code != "auto":
        opts["language"] = lang_code

    # Compact domain prompt; avoids truncation and keeps decoder on shopping items
    if vocab_hint:
        vocab = sorted(set(vocab_hint))
        vocab = [w for w in vocab if 2 <= len(w) <= 30][:120]
        opts["initial_prompt"] = (
            "Lista de la compra / Λίστα αγορών / Shopping list: " + ", ".join(vocab)
        )
    else:
        opts["initial_prompt"] = (
            "Lista de la compra, supermercado. Λίστα αγορών, σούπερ μάρκετ. Shopping list."
        )

    segments, _ = model.transcribe(wav_path, **opts)
    text = " ".join(s.text.strip() for s in segments).strip()
    text = re.sub(r"\b(yes|sí)\.?$", "", text, flags=re.IGNORECASE).strip()
    return text




def export_audiosegment_to_wav_bytes(seg: AudioSegment) -> bytes:
    wav_io = io.BytesIO()
    seg.export(wav_io, format="wav")
    return wav_io.getvalue()


def transcribe_openai(wav_bytes: bytes, lang: str, vocab_hint: list[str] | None = None) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    try:
        from openai import OpenAI
        client = OpenAI()

        # Build a short decoding bias from your catalog
        prompt = ""
        if vocab_hint:
            vocab = sorted(set(vocab_hint))
            vocab = [w for w in vocab if 2 <= len(w) <= 30][:120]
            if vocab:
                prompt = "Shopping list / Lista de la compra / Λίστα αγορών: " + ", ".join(vocab)

        with io.BytesIO(wav_bytes) as f:
            f.name = "audio.wav"
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=lang,   # "es" / "el" / "en" (or "auto" you map to default)
                temperature=0,
                prompt=prompt,   # << bias toward your products
            )
        return (getattr(result, "text", "") or "").strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI transcription failed: {e}")



# -------------------------
# Sidebar Navigation & Global Controls
# -------------------------

st.sidebar.title("🎤 Voice Inventory")
mode = st.sidebar.radio(
    "Modo",
    ["Products DB", "Voice Order", "Orders"],  # NEW "Orders"
    index=0,
    help="Cambia entre crear catálogo, tomar pedidos por voz y gestionar pedidos."
)


with st.sidebar.expander("⚙️ Opciones de transcripción"):
    asr_backend = st.selectbox("Backend ASR", ["OpenAI Whisper API", "Faster-Whisper (local)"])
    # Focus on ES, EL, EN; keep 'auto' if you want fallback
    lang_code = st.selectbox("Idioma", ["auto", "es", "el", "en"], index=1)
    model_size = st.selectbox("Modelo (local)", ["tiny", "base", "small"], index=2, help="Para Faster-Whisper")
    seconds = st.slider("Segundos a grabar (local)", 3, 20, 8)
    samplerate = st.selectbox("Samplerate (local)", [16000, 22050, 24000], index=0)


with st.sidebar.expander("🎛️ Preferencias de UI"):
    show_stats = st.toggle("Ver KPIs de catálogo", value=True)
    compact_tables = st.toggle("Tablas compactas", value=False)

# -------------------------
# Products DB page
# -------------------------

if mode == "Products DB":
    st.title("📦 Products Database")
    st.caption("Crea y mantiene tu catálogo. Estos datos alimentan el pedido por voz y los mensajes a proveedores.")

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("➕ Añadir producto")
        with st.form("add_product"):
            name = st.text_input("Nombre del producto *")
            category = st.text_input("Categoría")
            unit = st.selectbox("Unidad base", ["unidad", "botella", "kg", "caja", "lata", "pack"], index=0)
            quantity = st.number_input("Cantidad (stock opcional)", min_value=0.0, step=1.0, value=0.0)
            with st.expander("Proveedor (opcional)", expanded=False):
                provider_name = st.text_input("Nombre")
                provider_email = st.text_input("Email")
                provider_phone = st.text_input("Teléfono")
                provider_address = st.text_area("Dirección", height=72)
            created_at = st.date_input("Fecha", value=date.today())
            submitted = st.form_submit_button("Guardar producto")
        if submitted:
            if not name:
                st.error("El nombre es obligatorio.")
            else:
                with get_session() as s:
                    p = Product(
                        name=name,
                        category=category,
                        unit=unit,
                        quantity=quantity,
                        provider_name=provider_name,
                        provider_email=provider_email,
                        provider_phone=provider_phone,
                        provider_address=provider_address,
                        created_at=created_at,
                    )
                    normalize_model(p)
                    s.add(p)
                    s.commit()
                    st.success(f"Producto '{p.name}' guardado (normalizado).")

        st.subheader("⬆️⬇️ Importar / Exportar")
        c1, c2 = st.columns(2)
        with c1:
            up = st.file_uploader("Importar CSV (name,category,unit,quantity,provider_*)", type=["csv"])
            if up is not None:
           
                try:
                    df = pd.read_csv(up)
                    needed = {"name"}
                    if not needed.issubset(df.columns):
                        st.error("El CSV debe incluir la columna 'name'.")
                    else:
                        with get_session() as s:
                            def _str_or_none(x: object) -> str | None:
                                # Devuelve None si está vacío/NaN, si no devuelve str(x).strip()
                         
                                if pd.isna(x):
                                    return None
                                s = str(x).strip()
                                return s if s != "" else None

                            def _unit_or_default(x: object, default: str = "unidad") -> str:
                            
                                if pd.isna(x):
                                    return default
                                s = str(x).strip().lower()
                                return s if s != "" else default

                            def _float_or_default(x: object, default: float = 0.0) -> float:
                            
                                if pd.isna(x):
                                    return default
                                try:
                                    return float(x)
                                except Exception:
                                    return default

                            for _, row in df.iterrows():
                                p = Product(
                                    name=str(row.get("name", "")).strip(),  # 'name' es obligatorio
                                    category=_str_or_none(row.get("category")),
                                    unit=_unit_or_default(row.get("unit"), "unidad"),
                                    quantity=_float_or_default(row.get("quantity"), 0.0),

                                    # Proveedor (guárdalos como texto; pueden llevar ceros a la izquierda)
                                    provider_name=_str_or_none(row.get("provider_name")),
                                    provider_email=_str_or_none(row.get("provider_email")),
                                    provider_phone=_str_or_none(row.get("provider_phone")),
                                    provider_address=_str_or_none(row.get("provider_address")),
                                )
                                normalize_model(p)
                                s.add(p)
                            s.commit()

                        st.success("Productos importados.")
                except Exception as e:
                    st.error(f"No se pudo importar: {e}")
        with c2:
            if st.button("Exportar catálogo CSV"):
                with get_session() as s:
                    prods = s.exec(select(Product).order_by(Product.id.desc())).all()
                if prods:
                    df = pd.DataFrame([p.dict() for p in prods])
                    csv_buf = io.StringIO()
                    df.to_csv(csv_buf, index=False)
                    st.download_button("Descargar catalogo.csv", csv_buf.getvalue(), file_name="catalogo.csv", mime="text/csv")
                else:
                    st.info("No hay productos para exportar.")

    with right:
        st.subheader("Catálogo actual")
        with get_session() as s:
            products = s.exec(select(Product).order_by(Product.id.desc())).all()
        import pandas as pd
        if not products:
            st.info("Aún no hay productos. Añade el primero con el formulario.")
        else:
            # Filters
            f1, f2, f3 = st.columns([2, 1, 1])
            q = f1.text_input("🔎 Buscar (nombre/categoría)")
            unit_filter = f2.selectbox("Unidad", ["(todas)"] + sorted({p.unit or "unidad" for p in products}))
            cat_filter = f3.selectbox("Categoría", ["(todas)"] + sorted({p.category or "" for p in products}))

            rows = [p.dict() for p in products]
            if q:
                qn = unidecode(q.lower())
                rows = [r for r in rows if qn in unidecode((r.get("name") or "").lower()) or qn in unidecode((r.get("category") or "").lower())]
            if unit_filter != "(todas)":
                rows = [r for r in rows if (r.get("unit") or "unidad") == unit_filter]
            if cat_filter != "(todas)":
                rows = [r for r in rows if (r.get("category") or "") == cat_filter]

            df = pd.DataFrame(rows)
            if compact_tables:
                st.dataframe(df, use_container_width=True, height=360)
            else:
                st.dataframe(df, use_container_width=True)

            if show_stats:
                total_items = len(df)
                total_stock = float(df["quantity"].fillna(0).sum()) if "quantity" in df else 0
                st.caption(f"**{total_items}** productos · Stock total: **{total_stock:.0f}** (unidades agregadas)")

            # Inline delete
            if df.shape[0] > 0:
                st.divider()
                del_id = st.number_input("Eliminar por ID", min_value=0, step=1, value=0)
                if st.button("Eliminar") and del_id:
                    with get_session() as s:
                        obj = s.get(Product, int(del_id))
                        if obj:
                            s.delete(obj)
                            s.commit()
                            st.success(f"Producto {del_id} eliminado. Recarga para ver cambios.")
                        else:
                            st.warning("ID no encontrado.")

# -------------------------
# Voice Order page
# -------------------------

if mode == "Voice Order":
    st.title("🗣️ Voice Order")
    st.caption("Dicta productos y cantidades. Se admite número→nombre o nombre→número y el separador 'next'.")

    with get_session() as s:
        products = s.exec(select(Product).order_by(Product.name.asc())).all()
    if not products:
        st.warning("Primero crea tu catálogo en 'Products DB'.")
        st.stop()

    catalog_names = [normalize_text(p.name) for p in products]
    name_to_product = {p.name: p for p in products}

    # ------------- Mic Choice -------------
    mic_mode = st.radio("Entrada de audio", ["🎙️ Navegador (audiorecorder)", "💻 Local (sounddevice)"])

    transcript = None

    if mic_mode == "🎙️ Navegador (audiorecorder)":
        if audiorecorder is None or AudioSegment is None:
            st.error("Falta 'streamlit-audiorecorder' / 'pydub'. Instálalos para usar el micro del navegador.")
        else:
            st.subheader("🎧 Grabar en navegador")
            audio_seg = audiorecorder("Start", "Stop")
            if len(audio_seg) > 0:
                with st.expander("Preview", expanded=False):
                    wav_preview = io.BytesIO()
                    audio_seg.export(wav_preview, format="wav")
                    st.audio(wav_preview.getvalue(), format="audio/wav")
                try:
                    wav_bytes = export_audiosegment_to_wav_bytes(audio_seg)
                    with st.spinner("Transcribiendo…"):
                        if asr_backend == "OpenAI Whisper API":
                            transcript = transcribe_openai(
                                            wav_bytes,
                                            lang_code if lang_code != "auto" else "es",
                                            catalog_names,  # <— feed product vocabulary
                                        )
                        else:
                            if WhisperModel is None:
                                raise RuntimeError("Faster-Whisper no disponible")
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                                tf.write(wav_bytes)
                                wav_path = tf.name
                            transcript = transcribe_faster_whisper(wav_path, model_size, lang_code, catalog_names)
                except Exception as e:
                    st.error(f"No se pudo transcribir: {e}")

    else:
        st.subheader("💻 Grabar en equipo local")
        colA, colB, colC = st.columns([1,1,1])
        with colA:
            st.write(f"Modelo local: **{model_size}**")
        with colB:
            st.write(f"Idioma: **{lang_code}**")
        with colC:
            st.write(f"Duración: **{seconds}s** @ {samplerate} Hz")
        if st.button("Grabar y transcribir (local)"):
            try:
                audio_i16, sr = record_local(seconds=seconds, samplerate=samplerate)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    save_wav(tf.name, audio_i16, sr)
                    wav_path = tf.name
                with st.spinner("Transcribiendo con Whisper local…"):
                    transcript = transcribe_faster_whisper(wav_path, model_size, lang_code, catalog_names)
            except Exception as e:
                st.error(f"No se pudo grabar/transcribir: {type(e).__name__}: {e}")

    if transcript:
        st.session_state["transcript"] = transcript
        st.success("Transcripción lista")
    st.text_area("Transcripción", value=st.session_state.get("transcript", ""), height=120, key="transcript_area")

    # ------------- Parse & Review -------------
    # ------------- Parse & Review -------------
    st.subheader("🧩 Parseo y revisión")
    raw_text = st.text_area(
        "Texto de pedido (puedes editar la transcripción)",
        height=120,
        value=st.session_state.get("transcript", ""),
        placeholder="3 limones next gin 2 next olivas 1 next 1 aperol",
        key="parse_input",
    )

    parse_clicked = st.button("Parsear pedido")
    
    if parse_clicked:
        if not raw_text.strip():
            st.error("Introduce o genera una transcripción.")
        else:
            items = tokenize_items(raw_text)
            parsed_rows: List[Dict] = []

            for frag in items:
                parsed = parse_item(frag)
                if not parsed:
                    parsed_rows.append({
                        "spoken_name": frag,
                        "matched_name": None,
                        "confidence": 0.0,
                        "quantity": None,
                        "unit": None,
                        "status": "No interpretado",
                    })
                    continue

                name, qty, unit = parsed
                match_name, score = fuzzy_match(name, catalog_names)

                matched_product: Optional[Product] = None
                if match_name is not None:
                    for p in products:
                        if unidecode(p.name.lower()) == unidecode(str(match_name).lower()):
                            matched_product = p
                            break

                parsed_rows.append({
                    "spoken_name": name,
                    "matched_name": matched_product.name if matched_product else None,
                    "confidence": round(score or 0.0, 1),
                    "quantity": qty,
                    "unit": unit,
                    "status": "OK" if matched_product else "Revisar",
                })

            aggregated_rows = aggregate_parsed_rows(parsed_rows, name_to_product)

            # ✅ store parsed_df WITH provider (do NOT overwrite afterwards)
            with get_session() as s:
                st.session_state["parsed_df"] = add_provider_column(s, pd.DataFrame(aggregated_rows))

            st.success("✅ Parseo listo. Revisa/edita abajo y usa los botones.")




    # 🚩 CLAVE: Bloque persistente (editor + botones) fuera del if parse_clicked
    if "parsed_df" in st.session_state:
        st.subheader("Resultado del parseo")

        # El editor siempre se muestra desde sesión
        edited = st.data_editor(
            st.session_state["parsed_df"],
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "confidence": st.column_config.NumberColumn("Confianza", help="0-100"),
                "quantity": st.column_config.NumberColumn("Cantidad"),
                "unit": st.column_config.TextColumn("Unidad"),
                "matched_name": st.column_config.TextColumn("Producto (catálogo)"),
                "provider": st.column_config.TextColumn("Proveedor (catálogo)", disabled=True),

            },
            hide_index=True,
            key="parse_editor",  # mantiene edición entre reruns
        )

        # Actualizar la sesión con lo editado
        # Actualizar la sesión con lo editado + recalcular proveedor
        with get_session() as s:
            st.session_state["parsed_df"] = add_provider_column(s, edited)

        # Botones SIEMPRE visibles
        b1, b2, b3, b4 = st.columns([1,1,1,1])
        with b1:
            save_draft_clicked = st.button("💾 Guardar como borrador")
        with b2:
            export_csv_clicked = st.button("⬇️ Exportar CSV (sin guardar)")
        with b3:
            export_txt_clicked = st.button("⬇️ TXT por proveedor")
        with b4:
            add_to_existing_clicked = st.button("➕ Añadir a borrador abierto")

        # Usar siempre el DF de sesión
        df_to_use = st.session_state["parsed_df"]

        if save_draft_clicked:
            with get_session() as s:
                # 1) Crear Order y obtener su id sin cerrar la sesión
                order = Order(status="draft", title=f"Pedido {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                s.add(order)
                s.flush()  # asigna order.id sin commitear
                oid = order.id  # ← GUARDA EL ID MIENTRAS LA SESIÓN SIGUE ABIERTA

                # 2) Insertar líneas usando oid
                # 2) Insertar líneas usando oid
                for _, row in df_to_use.iterrows():
                    matched_name = str_or_default(row.get("matched_name"), "")
                    prod_obj = (
                        s.exec(select(Product).where(Product.name == matched_name)).first()
                        if matched_name
                        else None
                    )

                    unit_val = coalesce_unit(row.get("unit"), prod_obj, "unidad")

                    s.add(OrderLine(
                        order_id=oid,
                        product_id=prod_obj.id if prod_obj else None,
                        spoken_name=str_or_default(row.get("spoken_name"), "").lower(),
                        matched_name=(matched_name.strip() if matched_name else None),
                        confidence=num_or_default(row.get("confidence"), 0.0),
                        quantity=num_or_default(row.get("quantity"), 0.0),
                        unit=unit_val,
                    ))


                # 3) Un solo commit al final
                s.commit()

            # 4) Ya fuera del with, usa el id que guardaste (ya NO toques order.id)
            st.session_state["active_order_id"] = oid
            st.success(f"Borrador guardado (ID {oid}). Ahora puedes abrirlo en 'Orders' y seguir editando.")

            # Si quieres limpiar la tabla tras guardar, descomenta:
            # st.session_state.pop("parsed_df", None)

        if add_to_existing_clicked:
            oid = st.session_state.get("active_order_id")
            if not oid:
                st.warning("No hay un borrador abierto en esta sesión. Guarda como borrador primero o usa la página 'Orders'.")
            else:
                with get_session() as s:
                    for _, row in df_to_use.iterrows():
                        matched_name = (row.get("matched_name") or "").strip()
                        prod_obj = s.exec(select(Product).where(Product.name == matched_name)).first() if matched_name else None
                        candidate_unit = (row.get("unit") or "").strip().lower()
                        if not candidate_unit and prod_obj and getattr(prod_obj, "unit", None):
                            candidate_unit = str(prod_obj.unit).strip().lower()
                        unit_val = candidate_unit or "unidad"

                        s.add(OrderLine(
                            order_id=oid,
                            product_id=prod_obj.id if prod_obj else None,
                            spoken_name=str(row.get("spoken_name") or "").lower(),
                            matched_name=(matched_name.lower() if matched_name else None),
                            confidence=float(row.get("confidence") or 0.0),
                            quantity=float(row.get("quantity") or 0.0),
                            unit=unit_val,
                        ))
                    s.commit()
                st.success(f"Líneas añadidas al borrador existente (ID {oid}).")


        if export_csv_clicked:
            out = io.StringIO()
            writer = csv.writer(out)
            writer.writerow(["spoken_name", "matched_name", "quantity", "unit", "confidence"])
            for _, row in df_to_use.iterrows():
                writer.writerow([
                    row.get("spoken_name"), row.get("matched_name"),
                    row.get("quantity"), row.get("unit"), row.get("confidence")
                ])
            st.download_button(
                label="Descargar pedido.csv",
                data=out.getvalue().encode("utf-8"),
                file_name=f"pedido_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )

        if export_txt_clicked:
            grouped: Dict[str, Dict[str, object]] = {}
            for _, row in df_to_use.iterrows():
                matched_name = (row.get("matched_name") or '').strip()
                if not matched_name:
                    continue
                prod = name_to_product.get(matched_name)
                if not prod:
                    continue
                prov = getattr(prod, "provider_name", None) or "(Sin proveedor)"
                grouped.setdefault(prov, {
                    "provider_email": getattr(prod, "provider_email", "") or "",
                    "provider_phone": getattr(prod, "provider_phone", "") or "",
                    "provider_address": getattr(prod, "provider_address", "") or "",
                    "lines": []
                })
                grouped[prov]["lines"].append({
                    "product": matched_name,
                    "quantity": num_or_default(row.get("quantity"), None),
                    "unit": coalesce_unit(row.get("unit"), prod, "unidad"),
                })


            if not grouped:
                st.info("No hay líneas con producto del catálogo para agrupar por proveedor.")
            else:
                cc = st.text_input("Código país para WhatsApp (ej. +34 España, +30 Grecia)", value="+34")

                def normalize_phone(raw: str) -> str:
                    raw = (raw or "").strip()
                    digits = ''.join(ch for ch in raw if ch.isdigit() or ch == '+')
                    if not digits:
                        return ''
                    if digits.startswith('+'):
                        return digits
                    return f"{cc}{digits if not digits.startswith(('0',)) else digits.lstrip('0')}"

                for prov, meta in grouped.items():
                    st.markdown(f"#### {prov}")
                    body_lines = [
                        f"Pedido automático — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        "",
                        "Por favor, confirmar disponibilidad y plazos:",
                        ""
                    ]
                    for ln in meta["lines"]:  # type: ignore
                        qty = ln["quantity"] if ln["quantity"] is not None else ""
                        unit = ln["unit"]
                        body_lines.append(f"- {ln['product']}: {qty} {unit}".strip())
                    body_text = "\n".join(body_lines)
                    subject = f"Pedido — {datetime.now().strftime('%Y-%m-%d')}"
                    provider_email = meta.get("provider_email", "")  # type: ignore
                    if provider_email:
                        import urllib.parse as up
                        mailto = f"mailto:{up.quote(str(provider_email))}?subject={up.quote(subject)}&body={up.quote(body_text)}"
                        st.markdown(f"[📧 Abrir email]({mailto})  ")
                    else:
                        st.caption("(Sin email del proveedor)")
                    provider_phone = meta.get("provider_phone", "")  # type: ignore
                    phone_norm = normalize_phone(str(provider_phone))
                    if phone_norm:
                        import urllib.parse as up
                        wa = f"https://wa.me/{phone_norm.replace('+','')}?text={up.quote(body_text)}"
                        st.markdown(f"[📲 Abrir WhatsApp]({wa})  ")
                    else:
                        st.caption("(Sin teléfono del proveedor)")
                    txt_name = f"pedido_{prov}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
                    st.download_button(
                        label="⬇️ Descargar TXT",
                        data=body_text.encode("utf-8"),
                        file_name=txt_name,
                        mime="text/plain",
                    )
                    st.divider()


    st.markdown("""
---
### 🧩 Consejos de dictado
- Di **producto + número** o **número + producto**.
- Usa **"next"** / **"siguiente"** para separar artículos.
- Ejemplo: `3 limones next gin 2 next olivas 1 next 1 aperol`
""")

# -------------------------
# Footer / Help
# -------------------------
with st.expander("Ayuda y ajustes avanzados"):
    st.markdown(
        """
        - En servidores sin GPU, el modo local usa int8 y funciona bien. En GPU puedes ajustar a float16.
        - Para producción, migra la base de datos a Postgres y añade autenticación.
        - Si el navegador bloquea el micro, revisa los permisos del sitio.
        """
    )
if mode == "Orders":
    # =========================
    # 📜 Orders — Management UI
    # =========================

    from typing import Dict, List, Optional
    import io, csv
    from datetime import datetime


    st.title("📜 Orders")

    # ---------- Safe helpers (avoid pd.NA truthiness) ----------
    def str_or_default(x, default=""):
        if x is pd.NA or pd.isna(x):
            return default
        s = str(x).strip()
        return s if s != "" else default

    def num_or_default(x, default=0.0):
        if x is pd.NA or pd.isna(x):
            return default
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

    # ---------- Small helpers ----------
    expected_cols = ["line_id", "spoken_name", "matched_name", "quantity", "unit",
                  "provider", "confidence", "eliminar"]


    def coerce_to_df(obj, columns=expected_cols) -> pd.DataFrame:
        """Convert anything (df/list[dict]/dict/None) to a DataFrame with expected columns+order."""
        if isinstance(obj, pd.DataFrame):
            df = obj.copy()
        elif isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                df = pd.DataFrame(columns=columns)
            elif isinstance(obj[0], dict):
                df = pd.DataFrame(obj)
            else:
                df = pd.DataFrame(columns=columns)
        elif isinstance(obj, dict):
            vals = list(obj.values())
            if vals and all(isinstance(v, (list, tuple, pd.Series)) for v in vals):
                try:
                    df = pd.DataFrame(obj)
                except Exception:
                    df = pd.DataFrame([obj])
            else:
                df = pd.DataFrame([obj])
        else:
            df = pd.DataFrame(columns=columns)
        for c in columns:
            if c not in df.columns:
                df[c] = pd.NA
        return df[columns]

    def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        for col in ["spoken_name", "matched_name", "unit", "provider"]:
            if col in df.columns:
                df[col] = df[col].astype("string")
        for col in ["confidence", "quantity"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "eliminar" in df.columns:
            df["eliminar"] = df["eliminar"].fillna(False).astype(bool)
        return df


    def unit_from(row, prod_obj) -> str:
        return coalesce_unit(row.get("unit"), prod_obj, "unidad")

    def parse_and_merge(text_in: str, df_current: pd.DataFrame) -> pd.DataFrame:
        """Parse extra text and merge as new lines into df_current."""
        base_cols = ["spoken_name", "matched_name", "provider", "quantity", "unit", "confidence"]
        df_current = coerce_to_df(df_current, expected_cols)

        items = tokenize_items(text_in)
        parsed_rows = []

        with get_session() as s2:
            prods = s2.exec(select(Product).order_by(Product.name.asc())).all()
        catalog_names = [normalize_text(p.name) for p in prods]

        for frag in items:
            parsed = parse_item(frag)
            if not parsed:
                parsed_rows.append({
                    "spoken_name": frag,
                    "matched_name": None,
                    "provider": "",
                    "confidence": 0.0,
                    "quantity": None,
                    "unit": None,
                })
                continue

            name, qty, unit = parsed
            match_name, score = fuzzy_match(name, catalog_names)
            parsed_rows.append({
                "spoken_name": name,
                "matched_name": match_name if match_name else None,
                "provider": "",
                "confidence": round(score or 0.0, 1),
                "quantity": qty,
                "unit": unit,
            })

        add_df = pd.DataFrame(parsed_rows, columns=base_cols)
        add_df["line_id"] = pd.NA
        add_df["eliminar"] = False

        merged = pd.concat([df_current[expected_cols], add_df[expected_cols]], ignore_index=True)
        merged = normalize_types(merged)

        # ✅ recompute provider from Products DB
        with get_session() as s3:
            merged = add_provider_column(s3, merged)

        return merged


    # ---------- Load orders list ----------
    with get_session() as s:
        orders = (
            s.exec(select(Order).order_by(Order.created_at.desc()))
            .all()
        )


    if not orders:
        st.info("No hay pedidos aún. Guarda un borrador desde 'Voice Order'.")
        st.stop()

    # Selector del pedido
    oid_labels = [
        f"#{o.id} — {o.title or o.created_at.strftime('%Y-%m-%d %H:%M')} — {o.status}"
        for o in orders
    ]
    choice = st.selectbox("Selecciona un pedido", options=list(range(len(orders))), format_func=lambda i: oid_labels[i])
    order = orders[choice]
    st.session_state["active_order_id"] = order.id

    cA, cB, cC = st.columns([2,1,1])
    with cA:
        order_title = st.text_input("Título del pedido", value=order.title or "", key=f"order_title_{order.id}")
    with cB:
        order_status = st.selectbox("Estado", ["draft", "final"], index=0 if order.status == "draft" else 1, key=f"order_status_{order.id}")
    with cC:
        st.write(" ")
        st.caption(f"Creado: {order.created_at.strftime('%Y-%m-%d %H:%M')}")

    # Save order header if changed
    if (order.title or "") != order_title or order.status != order_status:
        with get_session() as s:
            o = s.exec(select(Order).where(Order.id == order.id)).first()
            if o:
                o.title = order_title or None
                o.status = order_status
                s.add(o)
                s.commit()

    # ---------- Load lines for selected order ----------
    with get_session() as s:
        lines = (
            s.exec(select(OrderLine).where(OrderLine.order_id == order.id))
            .all()
        )


    rows = [{
        "line_id": ln.id,
        "spoken_name": ln.spoken_name,
        "matched_name": ln.matched_name,
        "provider": "",
        "quantity": ln.quantity,
        "unit": ln.unit,
        "confidence": ln.confidence,
        "eliminar": False,
    } for ln in lines]


    # ---------- Prepare editor data state BEFORE rendering widget ----------
    state_key_df = f"order_editor_df_{order.id}"
    state_key_widget = f"order_editor_widget_{order.id}"

    if state_key_df not in st.session_state:
        st.session_state[state_key_df] = pd.DataFrame(rows, columns=expected_cols)
    else:
        # refresh from DB if df is empty but we have rows
        if st.session_state[state_key_df].empty and rows:
            st.session_state[state_key_df] = pd.DataFrame(rows, columns=expected_cols)

    # Guarantee cols & normalize
    st.session_state[state_key_df] = coerce_to_df(st.session_state[state_key_df], expected_cols)
    st.session_state[state_key_df] = normalize_types(st.session_state[state_key_df])
    with get_session() as s:
        st.session_state[state_key_df] = add_provider_column(s, st.session_state[state_key_df])


    # ---------- Editable UI ----------
    st.subheader("🧾 Líneas del pedido")

    colcfg = {
        "line_id": st.column_config.Column("ID", help="ID interno", disabled=True, width="small"),
        "spoken_name": st.column_config.TextColumn("Nombre hablado", help="Como se dictó por voz"),
        "matched_name": st.column_config.TextColumn("Producto (catálogo)"),
        "provider": st.column_config.TextColumn("Proveedor (catálogo)", disabled=True),
        "quantity": st.column_config.NumberColumn("Cantidad", min_value=0.0, step=1.0, format="%.2f"),
        "unit": st.column_config.TextColumn("Unidad", help="Ej: unidad, kg, l, ..."),
        "confidence": st.column_config.NumberColumn("Confianza", help="0-100", disabled=True, format="%.1f"),
        "eliminar": st.column_config.CheckboxColumn("Eliminar", help="Marca para borrar"),
    }


    st.caption("Edita las celdas. Usa **+** para añadir. Marca **Eliminar** para borrar una fila.")
    edited = st.data_editor(
        st.session_state[state_key_df],
        use_container_width=True,
        num_rows="dynamic",
        column_config=colcfg,
        hide_index=True,
        key=state_key_widget,
    )

    # Sync widget -> session data
    st.session_state[state_key_df] = coerce_to_df(edited, expected_cols)
    st.session_state[state_key_df] = normalize_types(st.session_state[state_key_df])
    with get_session() as s:
        st.session_state[state_key_df] = add_provider_column(s, st.session_state[state_key_df])


    # ---------- Actions ----------
    c1, c2 = st.columns([1,1])
    with c1:
        apply_btn = st.button("💾 Guardar cambios", type="primary")
    with c2:
        discard_btn = st.button("↩️ Deshacer cambios")

    if discard_btn:
        # reload from DB
        st.session_state[state_key_df] = pd.DataFrame(rows, columns=expected_cols)
        st.session_state[state_key_df] = normalize_types(st.session_state[state_key_df])
        st.rerun()

    if apply_btn:
        df_edit = st.session_state[state_key_df].copy()

        # Build delete / upsert lists
        to_delete_ids = []
        to_upsert = []

        with get_session() as s:
            # For product unit inference
            def product_for_name(name):
                name = str_or_default(name, "")
                if not name:
                    return None
                return s.exec(select(Product).where(Product.name == name)).first()


            for _, r in df_edit.iterrows():
                # skip completely blank new lines
                if (
                    str_or_default(r.get("matched_name"), "") == "" and
                    num_or_default(r.get("quantity"), 0.0) == 0.0 and
                    str_or_default(r.get("unit"), "") == "" and
                    str_or_default(r.get("spoken_name"), "") == ""
                ):
                    continue


                rid = str_or_default(r.get("line_id"), "")
                if bool(r.get("eliminar", False)):
                    if rid:
                        to_delete_ids.append(rid)
                    continue

                matched_name = str_or_default(r.get("matched_name"), "")
                prod_obj = product_for_name(matched_name)

                payload = dict(
                    order_id = order.id,
                    product_id = (prod_obj.id if prod_obj else None),
                    spoken_name = str_or_default(r.get("spoken_name"), "").lower(),
                    matched_name = (matched_name.strip() if matched_name else None),
                    confidence = num_or_default(r.get("confidence"), 0.0),
                    quantity = num_or_default(r.get("quantity"), 0.0),
                    unit = coalesce_unit(r.get("unit"), prod_obj, "unidad"),
                )

                if rid:
                    payload["id"] = rid
                to_upsert.append(payload)

            # Persist: delete first
            for lid in to_delete_ids:
                line = s.exec(select(OrderLine).where(OrderLine.id == lid)).first()


                if line:
                    s.delete(line)

            # Persist upserts (update or insert)
            for lp in to_upsert:
                lid = lp.pop("id", None)
                if lid:
                    line = s.exec(select(OrderLine).where(OrderLine.id == lid)).first()


                    if line:
                        for k, v in lp.items():
                            setattr(line, k, v)
                        s.add(line)
                else:
                    s.add(OrderLine(**lp))

            s.commit()

        st.success("Líneas actualizadas correctamente ✅")


    # =============================
    # ✉️ Drafts for Suppliers (nice)
    # =============================
    st.subheader("✉️ Borradores para proveedores")

    # Build a quick catalog lookup by name
    with get_session() as s:
        prods = s.exec(select(Product)).all()
    name_to_product = {getattr(p, "name", ""): p for p in prods}

    df_current = coerce_to_df(st.session_state[state_key_df], expected_cols)

    # Group lines by supplier (using product metadata)
    grouped: Dict[str, Dict[str, object]] = {}
    for _, row in df_current.iterrows():
        matched_name = (row.get("matched_name") or "").strip()
        if not matched_name:
            continue
        prod = name_to_product.get(matched_name)
        if not prod:
            continue
        prov = getattr(prod, "provider_name", None) or "(Sin proveedor)"
        grouped.setdefault(prov, {
            "provider_email": getattr(prod, "provider_email", "") or "",
            "provider_phone": getattr(prod, "provider_phone", "") or "",
            "provider_address": getattr(prod, "provider_address", "") or "",
            "lines": []
        })
        grouped[prov]["lines"].append({
            "product": matched_name,
            "quantity": num_or_default(row.get("quantity"), None),
            "unit": coalesce_unit(row.get("unit"), prod, "unidad"),
        })


    if not grouped:
        st.info("No hay líneas con producto del catálogo para agrupar por proveedor.")
    else:
        cc = st.text_input("Código país para WhatsApp (ej. +34 España, +30 Grecia)", value="+34", key=f"wa_cc_{order.id}")

        def normalize_phone(raw: str) -> str:
            raw = (raw or "").strip()
            digits = ''.join(ch for ch in raw if ch.isdigit() or ch == '+')
            if not digits:
                return ''
            if digits.startswith('+'):
                return digits
            # remove leading zero if any after country code
            return f"{cc}{digits if not digits.startswith(('0',)) else digits.lstrip('0')}"

        # Global subject/body templates (user friendly)
        default_subject = f"Pedido — {order_title or ('#'+str(order.id))} — {datetime.now().strftime('%Y-%m-%d')}"
        default_open = "Hola,\n\nAdjunto el pedido actualizado. Por favor, confirma disponibilidad y plazos:"
        default_close = "\n\nGracias y un saludo."

        subject_tpl = st.text_input("Asunto del email", value=default_subject, key=f"email_subject_{order.id}")
        opening_tpl = st.text_area("Cabecera del mensaje", value=default_open, height=80, key=f"email_open_{order.id}")
        closing_tpl = st.text_area("Cierre del mensaje", value=default_close, height=60, key=f"email_close_{order.id}")

        st.caption("Ajusta estos textos; abajo verás los borradores por proveedor con enlaces directos y TXT descargable.")

        for prov, meta in grouped.items():
            st.markdown(f"### 🧑‍💼 {prov}")

            # Build body lines
            body_lines = [opening_tpl.strip(), ""]
            for ln in meta["lines"]:  # type: ignore
                qty = ln["quantity"] if ln["quantity"] is not None else ""
                unit = ln["unit"]
                body_lines.append(f"- {ln['product']}: {qty} {unit}".strip())
            body_lines.append(closing_tpl.strip())
            body_text = "\n".join(body_lines)

            # Render provider contact info
            provider_email = str(meta.get("provider_email", "") or "")
            provider_phone = str(meta.get("provider_phone", "") or "")
            provider_addr  = str(meta.get("provider_address", "") or "")

            c1, c2, c3 = st.columns([1,1,2])
            with c1:
                st.caption("📧 Email")
                st.write(provider_email or "—")
            with c2:
                st.caption("📱 Teléfono")
                st.write(provider_phone or "—")
            with c3:
                st.caption("📍 Dirección")
                st.write(provider_addr or "—")

            # Quick actions: open mail & WhatsApp and download TXT
            ca, cb, ccx = st.columns([1,1,2])

            with ca:
                if provider_email:
                    import urllib.parse as up
                    mailto = f"mailto:{up.quote(provider_email)}?subject={up.quote(subject_tpl)}&body={up.quote(body_text)}"
                    st.markdown(f"[📧 Abrir email]({mailto})")
                else:
                    st.caption("(Sin email del proveedor)")

            with cb:
                phone_norm = normalize_phone(provider_phone)
                if phone_norm:
                    import urllib.parse as up
                    wa = f"https://wa.me/{phone_norm.replace('+','')}?text={up.quote(body_text)}"
                    st.markdown(f"[📲 Abrir WhatsApp]({wa})")
                else:
                    st.caption("(Sin teléfono del proveedor)")

            with ccx:
                txt_name = f"pedido_{prov}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
                st.download_button(
                    label="⬇️ Descargar TXT del mensaje",
                    data=body_text.encode("utf-8"),
                    file_name=txt_name,
                    mime="text/plain",
                    key=f"dl_txt_{order.id}_{prov}"
                )

            # Show the final composed preview
            with st.expander("👁️ Vista previa del mensaje"):
                st.code(f"Asunto: {subject_tpl}\n\n{body_text}", language="text")

            st.divider()
