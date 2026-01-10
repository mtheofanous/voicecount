
# api.py — Modern minimal VOICECOUNT (single-page UX)
# ---------------------------------------------------
# Features
# - Tabs: Catálogo (Products) • Nuevo pedido (Parse) • Pedidos (Orders)
# - Supports same product name with multiple providers (multiple Product rows)
# - Unit dropdown + Provider dropdown only when product has >1 provider
# - Robust Streamlit reruns: no duplicate CSV imports, no “select twice” provider bug
# - SQLite auto-migration for new columns (provider/matched_name/product_id, etc.)

from __future__ import annotations
import os
import tempfile
import io
import re
import csv
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import unicodedata
from rapidfuzz import process, fuzz
from collections import defaultdict
from modelsold import Order, OrderLine, Product
from features.auth_and_manage.auth_multi_tenant import init_auth_db, auth_gate, current_active_venue, require_login, manage_organization_ui
from old.style_button import style_button
from audiorecorder import audiorecorder 
import pandas as pd
import streamlit as st
from sqlmodel import SQLModel, Field, Session, create_engine, select
from sqlalchemy import text
from dotenv import load_dotenv
from unidecode import unidecode
load_dotenv()
st.set_page_config(page_title="Voi", page_icon="🧾")

DB_URL = "sqlite:///voicecount2.db"
engine = create_engine(DB_URL, echo=False)

class Product(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    venue_id: int = Field(index=True)
    name: str = Field(index=True)

    category: Optional[str] = None
    unit: Optional[str] = Field(default="unidad")
    
    # ✅ NEW: default quantity (your "cantidad")
    default_qty: float = Field(default=1.0)

    provider_name: Optional[str] = Field(default=None, index=True)
    provider_email: Optional[str] = None
    provider_phone: Optional[str] = None
    provider_address: Optional[str] = None
    
    aliases: Optional[str] = Field(default=None)   # ✅ ADD THIS

    created_at: datetime = Field(default_factory=datetime.utcnow)


class Order(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    venue_id: int = Field(index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field(default="draft")  # draft|final
    title: Optional[str] = None
    note: Optional[str] = None


class OrderLine(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    venue_id: int = Field(index=True)
    order_id: int = Field(foreign_key="order.id")
    product_id: Optional[int] = Field(default=None, foreign_key="product.id")

    spoken_name: str
    quantity: float = 1.0
    unit: Optional[str] = Field(default="unidad")
    confidence: Optional[float] = Field(default=0.0)

    matched_name: Optional[str] = None
    provider: Optional[str] = None


def get_session() -> Session:
    return Session(engine)


def ensure_sqlite_columns() -> None:
    '''Tiny SQLite “migration” so older DBs don't crash when models change.'''
    SQLModel.metadata.create_all(engine)

    with engine.begin() as conn:
        cols = conn.execute(text("PRAGMA table_info('orderline')")).fetchall()
        existing = {row[1] for row in cols}
        if "provider" not in existing:
            conn.execute(text("ALTER TABLE orderline ADD COLUMN provider TEXT"))
        if "matched_name" not in existing:
            conn.execute(text("ALTER TABLE orderline ADD COLUMN matched_name TEXT"))
        if "product_id" not in existing:
            conn.execute(text("ALTER TABLE orderline ADD COLUMN product_id INTEGER"))
        if "confidence" not in existing:
            conn.execute(text("ALTER TABLE orderline ADD COLUMN confidence REAL"))
        if "unit" not in existing:
            conn.execute(text("ALTER TABLE orderline ADD COLUMN unit TEXT"))
        if "quantity" not in existing:
            conn.execute(text("ALTER TABLE orderline ADD COLUMN quantity REAL"))
        if "spoken_name" not in existing:
            conn.execute(text("ALTER TABLE orderline ADD COLUMN spoken_name TEXT"))

        
        # --- venue scoping ---
        # Add venue_id to orderline
        if "venue_id" not in existing:
            conn.execute(text("ALTER TABLE orderline ADD COLUMN venue_id INTEGER"))

        # Add venue_id to order
        ocols = conn.execute(text("PRAGMA table_info('order')")).fetchall()
        oexisting = {row[1] for row in ocols}
        if "venue_id" not in oexisting:
            conn.execute(text('ALTER TABLE "order" ADD COLUMN venue_id INTEGER'))
        pcols = conn.execute(text("PRAGMA table_info('product')")).fetchall()
        pexisting = {row[1] for row in pcols}
        for col, ddl in [
            ("venue_id", "ALTER TABLE product ADD COLUMN venue_id INTEGER"),
            ("provider_name", "ALTER TABLE product ADD COLUMN provider_name TEXT"),
            ("provider_email", "ALTER TABLE product ADD COLUMN provider_email TEXT"),
            ("provider_phone", "ALTER TABLE product ADD COLUMN provider_phone TEXT"),
            ("provider_address", "ALTER TABLE product ADD COLUMN provider_address TEXT"),
            ("unit", "ALTER TABLE product ADD COLUMN unit TEXT"),
            ("category", "ALTER TABLE product ADD COLUMN category TEXT"),
            ("created_at", "ALTER TABLE product ADD COLUMN created_at TEXT"),
            ("aliases", "ALTER TABLE product ADD COLUMN aliases TEXT"),
            ("default_qty", "ALTER TABLE product ADD COLUMN default_qty REAL"),
        ]:
            if col not in pexisting:
                conn.execute(text(ddl))


ensure_sqlite_columns()


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

NEXT_SEPARATORS = [" next ", " siguiente ", " sig ", ",", ";", " y ", " and ", "also ", "tambien "]

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


#===============Normalize helpers==============================================================
def norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def clean_unit(u: Optional[str]) -> str:
    return norm(u)

def product_exists_exact(session, name, provider, unit) -> bool:
    stmt = select(Product).where(
        Product.name == name,
        Product.provider_name == provider,
        Product.unit == unit,
    )
    return session.exec(stmt).first() is not None

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def singularize_es(word: str) -> str:
    w = word.strip()
    # reglas simples y seguras para tu caso (papeles->papel, vasos->vaso)
    if len(w) <= 3:
        return w
    if w.endswith("es") and len(w) > 4:
        return w[:-2]   # papeles -> papel
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]   # vasos -> vaso
    return w

def normalize_term(s: str) -> str:
    s = strip_accents(s.lower())
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # singulariza palabra a palabra
    s = " ".join(singularize_es(tok) for tok in s.split())
    return s

def save_provider_info_now(venue_id: int, provider_name: str, email: str, phone: str, address: str):
    with get_session() as s:
        updated = update_provider_info_for_all_products(
            s,
            venue_id,
            provider_name.strip(),
            (email or "").strip(),
            (phone or "").strip(),
            (address or "").strip(),
        )
        s.commit()
    return updated

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

#========================Catalogo - Proveedores Helpers========================================
def file_signature(uploaded_file):
    if uploaded_file is None:
        return None
    data = uploaded_file.getvalue()
    return hashlib.md5(data).hexdigest()

def get_provider_info_from_products(products: list[Product], provider_name: str):
    """Pick a 'canonical' provider info record from existing products."""
    provider_name = (provider_name or "").strip()
    if not provider_name:
        return {"name": "", "email": "", "phone": "", "address": ""}

    # pick the first row that has any provider detail; fallback to first matching row
    matches = [p for p in products if (p.provider_name or "").strip().lower() == provider_name.lower()]
    if not matches:
        return {"name": provider_name, "email": "", "phone": "", "address": ""}

    best = None
    for p in matches:
        if any([(p.provider_email or "").strip(), (p.provider_phone or "").strip(), (p.provider_address or "").strip()]):
            best = p
            break
    if best is None:
        best = matches[0]

    return {
        "name": provider_name,
        "email": (best.provider_email or ""),
        "phone": (best.provider_phone or ""),
        "address": (best.provider_address or ""),
    }


def update_provider_info_for_all_products(session, venue_id: int, provider_name: str, email: str, phone: str, address: str):
    """Because provider fields live on Product rows, update every Product with this provider_name."""
    provider_name = (provider_name or "").strip()
    if not provider_name:
        return 0

    rows = session.exec(select(Product).where(Product.venue_id == venue_id, Product.provider_name == provider_name)).all()
    for p in rows:
        p.provider_email = email or None
        p.provider_phone = phone or None
        p.provider_address = address or None
        session.add(p)
    return len(rows)


#================HELPERS============================================================================

def safe_str(x, default=""):
    # Handles None, pd.NA, NaN
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    return str(x)


def header():
    st.markdown(
        '''
        <style>
          .block-container {padding-top: 1rem;}
          div[data-testid="stMetric"] {background: rgba(0,0,0,0.03); padding: 12px; border-radius: 16px;}
        </style>
        ''',
        unsafe_allow_html=True,
    )
    st.title("🧾 VOICECOUNT")
    st.caption("Minimal • rápido • catálogo con proveedores • pedidos por proveedor")


def str_or_default(x, default="") -> str:
    if x is pd.NA or pd.isna(x):
        return default
    s = str(x).strip()
    return s if s else default

def normalize_key(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "").strip().lower())

def file_signature(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    return hashlib.md5(data).hexdigest()

def distinct_units(products: List[Product]) -> List[str]:
    units = sorted({normalize_key(getattr(p, "unit", "") or "") for p in products if getattr(p, "unit", None)})
    units = [u for u in units if u]
    return units or ["unidad", "kg", "g", "l", "caja", "botella"]

def mic_or_upload_audio(label: str, key: str, sample_rate: int = 16000):
    """
    Audio input helper with optional sample rate.
    """
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
    """
    Extract raw audio bytes from Streamlit audio widgets or uploads.
    """
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
    bio.name = "audio.wav"  # important: OpenAI uses filename metadata

    out = client.audio.transcriptions.create(
        file=bio,
        model="whisper-1",
        language=language,
        prompt=prompt,
        temperature=0,
    )
    return (out.text or "").strip()

def asr_faster_whisper(audio_bytes: bytes, vocab, language: str = "es") -> str:
    """
    Local Faster-Whisper backend
    - Always uses SMALL model
    - CPU-optimized
    """
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise RuntimeError("Missing dependency: pip install faster-whisper") from e

    # Load SMALL model once per session
    if "fw_model_small" not in st.session_state:
        st.session_state["fw_model_small"] = WhisperModel(
            "small",
            device="cpu",
            compute_type="int8",
        )

    model = st.session_state["fw_model_small"]

    initial_prompt = (
        "Pedido HORECA en español. "
        "Mantén cantidades y unidades. "
        "Separador: siguiente. "
        "Productos y marcas: " + ", ".join(vocab[:120])
    )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        segments, info = model.transcribe(
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

def num_or_default(x, default=0.0):
    if x is pd.NA or pd.isna(x):
        return default
    try:
        return float(x)
    except Exception:
        return default

def canonical_manual_name(raw_name: str) -> str:
    """
    Reuse your parse logic from new_order_tab() to normalize a manual product name.
    We trick parse_item by prepending a qty so it parses like an order line.
    """
    raw_name = (raw_name or "").strip()
    if not raw_name:
        return ""
    parsed = parse_item(f"1 {raw_name}")  # parse_item returns (name, qty, unit) :contentReference[oaicite:2]{index=2}
    if parsed:
        name_norm, _, _ = parsed
        return (name_norm or "").strip()
    # fallback
    return normalize_text(raw_name).strip()  # normalize_text is used in your matching pipeline :contentReference[oaicite:3]{index=3}

#===CATALOG=======================================================================================
#===CATALOG=======================================================================================

def catalog_tab(venue_id: int, role: str | None = None):
    st.subheader("📦 Catálogo de productos")

    # ---- tiny modern styling ----
    st.markdown(
        """
        <style>
          .block-container {padding-top: 1rem;}
          .vc-card {background: rgba(0,0,0,0.03); padding: 14px 16px; border-radius: 18px;}
          .vc-muted {opacity: .7;}
          div[data-testid="stMetric"] {background: rgba(0,0,0,0.03); padding: 12px; border-radius: 16px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    with get_session() as s:
        products = s.exec(
            select(Product).where(Product.venue_id == venue_id).order_by(
                Product.name.asc(),
                Product.provider_name.asc(),
                Product.unit.asc(),
            )
        ).all()

    # KPIs
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Productos", len(products))
    with m2:
        st.metric("Nombres únicos", len({(p.name or "").strip().lower() for p in products if p.name}))
    with m3:
        st.metric("Proveedores", len({(p.provider_name or "").strip().lower() for p in products if p.provider_name}))

    # Units from DB
    unit_opts = distinct_units(products)
    if "unidad" not in [u.lower() for u in unit_opts]:
        unit_opts = ["unidad"] + unit_opts

    # =========================================================
    # Tabs: Import / Manual add / Edit
    # =========================================================
    t_import, t_manual, t_edit = st.tabs(["📥 Importar (CSV/Excel)", "➕ Añadir manual", "✏️ Editar catálogo"])

    # -------------------------
    # Helpers: key logic (SKIP duplicates exact)
    # -------------------------
    def _norm(s: str) -> str:
        return (s or "").strip().lower()

    def _unit(u: str) -> str:
        u = (u or "").strip().lower()
        return u or "unidad"

    def _key_name_provider_unit(name: str, provider: str, unit: str):
        return (_norm(name), _norm(provider), _unit(unit))

    # Build exact existing keys from DB once per run
    existing_keys_db = {
        _key_name_provider_unit(p.name or "", p.provider_name or "", p.unit or "unidad")
        for p in products
    }

    # -------------------------
    # 1) IMPORT (CSV/EXCEL)
    # -------------------------
    with t_import:
        st.markdown('<div class="vc-card">', unsafe_allow_html=True)
        st.markdown("### Importar archivo")
        st.caption(
            "Columnas: name (obligatorio), unit, default_qty, provider_name, provider_email, "
            "provider_phone, provider_address, category, aliases (opcionales)."
        )

        # --- UNDO state ---
        if "last_import_product_ids" not in st.session_state:
            st.session_state["last_import_product_ids"] = []
        if "last_import_summary" not in st.session_state:
            st.session_state["last_import_summary"] = None

        uploaded = st.file_uploader("Sube CSV o Excel", type=["csv", "xlsx", "xls"], key="products_file_uploader")

        if "imported_file_sigs" not in st.session_state:
            st.session_state["imported_file_sigs"] = set()
        if "import_force_save" not in st.session_state:
            st.session_state["import_force_save"] = False
        if "import_mode" not in st.session_state:
            st.session_state["import_mode"] = None

        def _read_products_file(up):
            name = (up.name or "").lower()
            if name.endswith(".csv"):
                df_ = pd.read_csv(up)
            else:
                df_ = pd.read_excel(up)
            df_.columns = [c.strip().lower() for c in df_.columns]
            return df_

        # Undo
        undo_col, _ = st.columns([1, 3])
        with undo_col:
            if st.session_state["last_import_product_ids"]:
                if st.button("↩️ Undo last import", type="secondary", key="undo_last_import_btn"):
                    ids_to_delete = list(st.session_state["last_import_product_ids"])
                    deleted = 0
                    with get_session() as s:
                        for pid in ids_to_delete:
                            obj = s.exec(select(Product).where(Product.id == pid)).first()
                            if obj:
                                s.delete(obj)
                                deleted += 1
                        s.commit()

                    st.session_state["last_import_product_ids"] = []
                    st.session_state["last_import_summary"] = None
                    st.success(f"Undo completado ✅ Eliminados {deleted} productos.")
                    st.rerun()
            else:
                st.caption("No hay un import reciente para deshacer.")

        if uploaded is None:
            st.info("Sube un archivo para empezar.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            sig = file_signature(uploaded)

            try:
                df = _read_products_file(uploaded)
            except Exception as e:
                st.error(f"No pude leer el archivo: {e}")
                st.markdown("</div>", unsafe_allow_html=True)
                df = None

            if df is None:
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.dataframe(df.head(50).reset_index(drop=True), width="stretch")

                c1, c2, c3 = st.columns([1, 1, 2])
                with c1:
                    do_import = st.button("✅ Importar", type="primary", key="do_import_btn")
                with c2:
                    reset = st.button("↩️ Permitir re-importar", key="reset_reimport_btn")
                with c3:
                    st.caption("El import solo se ejecuta al hacer click. Evita doble-import del mismo archivo en esta sesión.")

                if reset:
                    st.session_state["imported_file_sigs"].discard(sig)
                    st.success("OK — este archivo se puede re-importar en esta sesión.")

                if do_import:
                    if sig in st.session_state["imported_file_sigs"]:
                        st.warning("Este archivo ya se importó en esta sesión. Si quieres re-importarlo, pulsa reset.")
                        st.markdown("</div>", unsafe_allow_html=True)
                    elif "name" not in df.columns:
                        st.error("El archivo debe tener una columna 'name'.")
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        # Normalize missing cols
                        for c in ["unit", "default_qty", "provider_name", "provider_email", "provider_phone",
                                  "provider_address", "category", "aliases"]:
                            if c not in df.columns:
                                df[c] = pd.NA

                        # ---- Build existing normalized names per (provider, unit) from DB ----
                        # This allows: same name+provider with different unit WITHOUT conflict.
                        provunit_to_existing_norm = {}
                        for p in products:
                            prov = _norm(p.provider_name or "")
                            unit = _unit(p.unit or "unidad")
                            key_pu = (prov, unit)
                            provunit_to_existing_norm.setdefault(key_pu, []).append(normalize_text(p.name))

                        provunit_to_exact_set = {k: set(v) for k, v in provunit_to_existing_norm.items()}

                        conflicts = []
                        clean_row_idxs = []
                        seen_in_file = {}

                        for idx, r in df.iterrows():
                            raw_name = str_or_default(r.get("name"), "").strip()
                            if not raw_name:
                                continue

                            prov_raw = str_or_default(r.get("provider_name"), "").strip()
                            prov_key = _norm(prov_raw)

                            unit_raw = str_or_default(r.get("unit"), "unidad").strip()
                            unit_key = _unit(unit_raw)

                            pu_key = (prov_key, unit_key)

                            norm_name = canonical_manual_name(raw_name)

                            # exact dup checks are per (provider, unit)
                            exact_dup_db = norm_name in provunit_to_exact_set.get(pu_key, set())

                            seen_in_file.setdefault(pu_key, set())
                            exact_dup_file = norm_name in seen_in_file[pu_key]

                            # fuzzy only within same (provider, unit)
                            existing_norm_list = provunit_to_existing_norm.get(pu_key, [])
                            fuzzy_name, score = fuzzy_match(norm_name, existing_norm_list) if existing_norm_list else (None, 0.0)

                            if exact_dup_db or exact_dup_file or fuzzy_name:
                                conflicts.append({
                                    "row": int(idx) + 1,
                                    "provider_name": prov_raw,
                                    "unit": unit_key,
                                    "input_name": raw_name,
                                    "normalized": norm_name,
                                    "conflict_type": (
                                        "EXACT_DB" if exact_dup_db else
                                        "EXACT_FILE" if exact_dup_file else
                                        "FUZZY_DB"
                                    ),
                                    "matched_existing": fuzzy_name if fuzzy_name else "(exact match)",
                                    "score": float(score or 0.0),
                                })
                            else:
                                clean_row_idxs.append(idx)
                                seen_in_file[pu_key].add(norm_name)

                        if conflicts and not st.session_state["import_force_save"]:
                            st.warning("He detectado productos duplicados o muy parecidos (por proveedor + unidad). Revisa antes de importar.")
                            st.dataframe(pd.DataFrame(conflicts).reset_index(drop=True), width="stretch")

                            b1, b2, b3 = st.columns([1, 1, 2])
                            with b1:
                                if st.button("✅ Importar solo los no conflictivos", type="primary", key="import_clean_only_btn"):
                                    st.session_state["import_mode"] = "clean_only"
                                    st.session_state["import_force_save"] = True
                                    st.rerun()
                            with b2:
                                if st.button("⚠️ Forzar importación de TODO", type="secondary", key="import_force_all_btn"):
                                    st.session_state["import_mode"] = "force_all"
                                    st.session_state["import_force_save"] = True
                                    st.rerun()
                            with b3:
                                st.caption("Tip: si querías otra unidad (SKU distinto), mantenla distinta en la columna unit.")

                        else:
                            import_mode = st.session_state["import_mode"] or ("force_all" if not conflicts else "clean_only")
                            st.session_state["import_force_save"] = False
                            st.session_state["import_mode"] = None

                            import_df = df if import_mode == "force_all" else df.loc[clean_row_idxs]

                            created, skipped = 0, 0
                            inserted_ids = []

                            with get_session() as s:
                                # exact keys: (name, provider, unit)  -> SKIP
                                existing_keys = set(existing_keys_db)

                                for _, r in import_df.iterrows():
                                    name = str_or_default(r.get("name"), "").strip()
                                    if not name:
                                        continue

                                    provider = str_or_default(r.get("provider_name"), "").strip()
                                    unit = _unit(str_or_default(r.get("unit"), "unidad"))

                                    key = _key_name_provider_unit(name, provider, unit)
                                    if key in existing_keys:
                                        skipped += 1
                                        continue

                                    qty = r.get("default_qty", 1.0)
                                    try:
                                        qty = float(qty) if not pd.isna(qty) else 1.0
                                    except Exception:
                                        qty = 1.0

                                    p = Product(
                                        venue_id=venue_id,
                                        name=name,
                                        unit=unit,
                                        default_qty=qty,
                                        category=(str_or_default(r.get("category"), "").strip() or None),
                                        provider_name=(provider or None),
                                        provider_email=(str_or_default(r.get("provider_email"), "").strip() or None),
                                        provider_phone=(str_or_default(r.get("provider_phone"), "").strip() or None),
                                        provider_address=(str_or_default(r.get("provider_address"), "").strip() or None),
                                        aliases=(str_or_default(r.get("aliases"), "").strip() or None),
                                    )
                                    s.add(p)
                                    s.flush()
                                    inserted_ids.append(p.id)

                                    existing_keys.add(key)
                                    created += 1

                                s.commit()

                            st.session_state["imported_file_sigs"].add(sig)
                            st.session_state["last_import_product_ids"] = inserted_ids
                            st.session_state["last_import_summary"] = {"created": created, "skipped": skipped, "mode": import_mode}

                            st.success(f"Importado: {created} ✅  | SKIP exactos: {skipped} | Modo: {import_mode}")

            st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # 2) MANUAL ADD WIZARD
    # -------------------------
    with t_manual:
        st.markdown("### ➕ Añadir manual (modo rápido)")

        providers = sorted({(p.provider_name or "").strip() for p in products if (p.provider_name or "").strip()})
        provider_choices = ["➕ Nuevo proveedor…"] + providers

        if "selected_provider" not in st.session_state:
            st.session_state["selected_provider"] = "➕ Nuevo proveedor…"
        if "provider_edit_enabled" not in st.session_state:
            st.session_state["provider_edit_enabled"] = False

        selected = st.selectbox(
            "Proveedor",
            provider_choices,
            index=provider_choices.index(st.session_state["selected_provider"])
            if st.session_state["selected_provider"] in provider_choices else 0,
            key="provider_selectbox",
        )

        if selected != st.session_state["selected_provider"]:
            st.session_state["selected_provider"] = selected
            st.session_state["provider_edit_enabled"] = False

        is_new_provider = (selected == "➕ Nuevo proveedor…")

        st.markdown("#### Información del proveedor")

        if is_new_provider:
            prov_name = st.text_input("Nombre proveedor *", value="", placeholder="Ej: Frutas Paco", key="prov_name_new")
            c1, c2 = st.columns(2)
            with c1:
                prov_email = st.text_input("Email", value="", placeholder="ventas@proveedor.com", key="prov_email_new")
                prov_phone = st.text_input("Teléfono", value="", placeholder="+34 ...", key="prov_phone_new")
            with c2:
                prov_address = st.text_input("Dirección", value="", placeholder="Calle ...", key="prov_addr_new")

            chosen_provider = prov_name.strip()
            can_edit = True

        else:
            chosen_provider = selected.strip()
            info = get_provider_info_from_products(products, chosen_provider)

            if "last_existing_provider" not in st.session_state:
                st.session_state["last_existing_provider"] = None

            if chosen_provider != st.session_state["last_existing_provider"]:
                st.session_state["last_existing_provider"] = chosen_provider
                st.session_state["provider_edit_enabled"] = False
                st.session_state["prov_email_existing"] = info["email"]
                st.session_state["prov_phone_existing"] = info["phone"]
                st.session_state["prov_addr_existing"]  = info["address"]

            b1, b2, _ = st.columns([2, 2, 3])

            with b1:
                if st.button(
                    "✏️ Editar" if not st.session_state["provider_edit_enabled"] else "🔒 Bloquear edición",
                    key="toggle_provider_edit",
                ):
                    st.session_state["provider_edit_enabled"] = not st.session_state["provider_edit_enabled"]
                    st.rerun()

            with b2:
                if st.session_state["provider_edit_enabled"]:
                    if st.button("💾 Guardar", type="primary", key="save_provider_info_btn"):
                        updated = save_provider_info_now(
                            venue_id,
                            chosen_provider,
                            st.session_state.get("prov_email_existing", ""),
                            st.session_state.get("prov_phone_existing", ""),
                            st.session_state.get("prov_addr_existing", ""),
                        )
                        st.session_state["provider_edit_enabled"] = False
                        st.success(f"Información guardada ✅ (actualizada en {updated} productos)")
                        st.rerun()

            can_edit = st.session_state["provider_edit_enabled"]

            st.text_input("Nombre proveedor", value=chosen_provider, disabled=True, key="prov_name_existing")

            c1, c2 = st.columns(2)
            with c1:
                prov_email = st.text_input("Email", disabled=(not can_edit), key="prov_email_existing")
                prov_phone = st.text_input("Teléfono", disabled=(not can_edit), key="prov_phone_existing")
            with c2:
                prov_address = st.text_input("Dirección", disabled=(not can_edit), key="prov_addr_existing")

            prov_name = info["name"]

        # --- Products form ---
        st.markdown("#### Productos")

        if "manual_rows" not in st.session_state:
            st.session_state["manual_rows"] = [
                {"name": "", "unit": "unidad", "default_qty": 1.0, "category": "", "aliases": "", "eliminar": False}
            ]

        with st.form("manual_add_products_form"):
            manual_df = st.data_editor(
                pd.DataFrame(st.session_state["manual_rows"]),
                width="stretch",
                num_rows="dynamic",
                hide_index=True,
                column_config={
                    "name": st.column_config.TextColumn("Producto *", required=True),
                    "unit": st.column_config.SelectboxColumn("Unidad", options=unit_opts),
                    "default_qty": st.column_config.NumberColumn("Cantidad", min_value=0.0, step=1.0, format="%.2f"),
                    "category": st.column_config.TextColumn("Categoría"),
                    "aliases": st.column_config.TextColumn("Aliases (coma-separated)"),
                    "eliminar": st.column_config.CheckboxColumn("Eliminar", help="Marca para borrar la línea al guardar.", width="small"),
                },
                key="manual_products_editor",
            )

            save_manual = st.form_submit_button("✅ Guardar productos", type="primary")

        st.session_state["manual_rows"] = manual_df.to_dict("records")

        if save_manual:
            if not chosen_provider:
                st.error("Falta el nombre del proveedor.")
                st.stop()

            # filter out rows marked eliminar
            if "eliminar" in manual_df.columns:
                manual_df_save = manual_df[manual_df["eliminar"] != True].copy()
            else:
                manual_df_save = manual_df.copy()

            created, skipped = 0, 0

            with get_session() as s:
                # build existing keys from DB live (name+provider+unit) -> SKIP
                existing_rows = s.exec(select(Product.name, Product.provider_name, Product.unit).where(Product.venue_id == venue_id)).all()
                existing_keys = {
                    _key_name_provider_unit(str(n or ""), str(pv or ""), str(u or "unidad"))
                    for n, pv, u in existing_rows
                }

                for _, r in manual_df_save.iterrows():
                    name_raw = str_or_default(r.get("name"), "").strip()
                    if not name_raw:
                        continue

                    unit_val = _unit(str_or_default(r.get("unit"), "unidad"))
                    key = _key_name_provider_unit(name_raw, chosen_provider, unit_val)

                    # ✅ SKIP exact duplicate (same name+provider+unit)
                    if key in existing_keys:
                        skipped += 1
                        continue

                    try:
                        qty = float(r.get("default_qty", 1.0))
                    except Exception:
                        qty = 1.0

                    p = Product(
                        venue_id=venue_id,
                        name=name_raw,
                        unit=unit_val,
                        default_qty=qty,
                        category=(str_or_default(r.get("category"), "").strip() or None),
                        aliases=(str_or_default(r.get("aliases"), "").strip() or None),
                        provider_name=chosen_provider,
                        provider_email=(prov_email.strip() or None),
                        provider_phone=(prov_phone.strip() or None),
                        provider_address=(prov_address.strip() or None),
                    )
                    s.add(p)
                    existing_keys.add(key)
                    created += 1

                s.commit()

            st.success(f"Guardado ✅  Nuevos: {created} | SKIP exactos: {skipped}")
            st.rerun()

    # -------------------------
    # 3) EDIT CATALOG (existing)
    # -------------------------
    with t_edit:
        st.caption("Tip: mismo producto con 2 unidades → 2 filas con mismo name+provider_name pero unit distinta.")

        df = pd.DataFrame([{
            "id": p.id,
            "name": p.name,
            "unit": (p.unit or "unidad"),
            "default_qty": float(getattr(p, "default_qty", 1.0) or 1.0),
            "provider_name": (p.provider_name or ""),
            "provider_email": (p.provider_email or ""),
            "provider_phone": (p.provider_phone or ""),
            "provider_address": (p.provider_address or ""),
            "category": (getattr(p, "category", "") or ""),
            "aliases": (getattr(p, "aliases", "") or ""),
        } for p in products])

        if df.empty:
            df = pd.DataFrame(columns=[
                "id","name","unit","default_qty","provider_name",
                "provider_email","provider_phone","provider_address","category","aliases"
            ])

        edited = st.data_editor(
            df,
            width="stretch",
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
                "name": st.column_config.TextColumn("Producto", required=True),
                "unit": st.column_config.SelectboxColumn("Unidad", options=unit_opts),
                "default_qty": st.column_config.NumberColumn("Cantidad", min_value=0.0, step=1.0, format="%.2f"),
                "provider_name": st.column_config.TextColumn("Proveedor"),
                "provider_email": st.column_config.TextColumn("Email"),
                "provider_phone": st.column_config.TextColumn("Teléfono"),
                "provider_address": st.column_config.TextColumn("Dirección"),
                "category": st.column_config.TextColumn("Categoría"),
                "aliases": st.column_config.TextColumn("Aliases (coma-separated)"),
            },
            key="catalog_editor",
        )

        c1, c2 = st.columns([1,2])
        with c1:
            save = st.button("💾 Guardar catálogo", type="primary")
        with c2:
            st.caption("Se sincroniza con SQLite. Borra filas para eliminar productos.")

        if save:
            with get_session() as s:
                # delete removed rows
                existing_ids = {p.id for p in products if p.id is not None}
                edited_ids = {int(x) for x in edited["id"].dropna().astype(int).tolist()} if "id" in edited.columns else set()
                to_delete = sorted(existing_ids - edited_ids)

                for pid in to_delete:
                    obj = s.exec(select(Product).where(Product.id == pid)).first()
                    if obj:
                        s.delete(obj)

                # upsert edited rows
                for _, r in edited.iterrows():
                    rid = r.get("id")
                    name = str_or_default(r.get("name"), "").strip()
                    if not name:
                        continue

                    unit_val = _unit(str_or_default(r.get("unit"), "unidad"))
                    provider_name = str_or_default(r.get("provider_name"), "").strip()

                    try:
                        qty = float(r.get("default_qty", 1.0))
                    except Exception:
                        qty = 1.0

                    payload = dict(
                        name=name,
                        unit=unit_val,
                        default_qty=qty,
                        provider_name=(provider_name or None),
                        provider_email=(str_or_default(r.get("provider_email"), "").strip() or None),
                        provider_phone=(str_or_default(r.get("provider_phone"), "").strip() or None),
                        provider_address=(str_or_default(r.get("provider_address"), "").strip() or None),
                        category=(str_or_default(r.get("category"), "").strip() or None),
                        aliases=(str_or_default(r.get("aliases"), "").strip() or None),
                    )

                    if rid is not None and not pd.isna(rid) and str(rid).strip() != "":
                        obj = s.exec(select(Product).where(Product.id == int(rid))).first()
                        if obj:
                            for k, v in payload.items():
                                setattr(obj, k, v)
                            s.add(obj)
                    else:
                        # inserting from editor: allow same name+provider if unit differs
                        s.add(Product(venue_id=venue_id, **payload))

                s.commit()

            st.success("Catálogo guardado ✅")
            st.rerun()


#======================Voice Order=========================================================================

# ====================== Voice Order ===========================================

def new_order_tab(venue_id: int, role: str | None = None):
    import time
    import hashlib
    from typing import Dict, List
    import pandas as pd
    import streamlit as st
    from sqlmodel import select

    # ---------------------------
    # Session state init
    # ---------------------------
    if "transcript_area" not in st.session_state:
        st.session_state["transcript_area"] = ""
    if "last_audio_hash" not in st.session_state:
        st.session_state["last_audio_hash"] = None
    if "adding_to_draft_mode" not in st.session_state:
        st.session_state["adding_to_draft_mode"] = False
    if "audio_widget_key" not in st.session_state:
        st.session_state["audio_widget_key"] = f"asr_audio_in_{int(time.time())}"


    # ---------------------------
    # Sidebar options
    # ---------------------------
    with st.sidebar.expander("⚙️ Opciones de transcripción", expanded=False):
        asr_backend = st.selectbox("Backend ASR", ["OpenAI Whisper API", "Faster-Whisper (local)"])
        lang_code = st.selectbox("Idioma", ["auto", "es", "el", "en"], index=1)
        samplerate = st.selectbox("Samplerate", [16000, 22050, 24000], index=0)

    # ---------------------------
    # Load catalog
    # ---------------------------
    with get_session() as s:
        products = s.exec(
            select(Product)
            .where(Product.venue_id == venue_id)
            .order_by(Product.name.asc(), Product.provider_name.asc())
        ).all()

    if not products:
        st.warning("Primero crea tu catálogo en la pestaña 'Catálogo'.")
        return

    catalog_names = [normalize_text(p.name) for p in products]
    catalog_norm_to_product = {normalize_text(p.name): p for p in products}
    name_to_product = {p.name: p for p in products}

    # ---------------------------
    # Segmented control mode switch (horizontal radio)
    # ---------------------------
    mode = st.radio(
        "Modo",
        ["🎙️ Voz", "⌨️ Manual"],
        horizontal=True,
        label_visibility="collapsed",
        key="new_order_mode",
    )

    # Helper: clean all current order state
    def _clear_all(used_component_recorder: bool):
        st.session_state["transcript_area"] = ""
        st.session_state["last_audio_hash"] = None
        st.session_state.pop("parsed_df", None)
        st.session_state["adding_to_draft_mode"] = False

        # Clear fallback audio widget by rotating key
        if not used_component_recorder:
            st.session_state["audio_widget_key"] = f"asr_audio_in_{int(time.time())}"

        st.success("Pedido limpiado completamente ✅")
        st.rerun()

    # =========================================================
    # VOICE MODE
    # =========================================================
    audio_bytes = b""
    used_component_recorder = False

    if mode == "🎙️ Voz":
        st.markdown('<div class="vc-card">', unsafe_allow_html=True)
        st.markdown('<div class="vc-h">🎙️ Audio</div>', unsafe_allow_html=True)
        st.markdown('<div class="vc-caption">Graba y luego transcribe manualmente (no hay transcripción automática).</div>', unsafe_allow_html=True)

        try:
            from audiorecorder import audiorecorder  # pip install streamlit-audiorecorder
            used_component_recorder = True

            # ✅ Modern prompts, no pause
            audio_seg = audiorecorder(
                start_prompt="🎤 Grabar",
                stop_prompt="⏹️ Listo",
            )

            if audio_seg is not None and len(audio_seg) > 0:
                audio_bytes = audio_seg.export(format="wav").read()
                st.audio(audio_bytes, format="audio/wav")

        except Exception:
            st.info("Recorder tipo app no disponible. Usando grabador estándar.")
            audio = mic_or_upload_audio(
                "Mantén pulsado para grabar / o sube audio",
                key=st.session_state["audio_widget_key"],
                sample_rate=samplerate
            )
            if audio is not None:
                audio_bytes = read_audio_bytes(audio)

        st.markdown("</div>", unsafe_allow_html=True)

        audio_hash = hashlib.sha1(audio_bytes).hexdigest() if audio_bytes else None

        st.markdown('<div class="vc-card">', unsafe_allow_html=True)
        st.markdown('<div class="vc-h">📝 Transcripción</div>', unsafe_allow_html=True)
        st.markdown('<div class="vc-caption">Pulsa transcribir (manual). Luego puedes editar el texto.</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([2, 1])
        with c1:
            transcribe_clicked = st.button(
                "✨ Transcribir audio (manual)",
                type="primary",
                disabled=not bool(audio_bytes),
                key="btn_transcribe_manual",
            )
        with c2:
            clear_clicked = st.button("🧹 Limpiar", key="btn_clear_all_voice")

        if clear_clicked:
            _clear_all(used_component_recorder=used_component_recorder)

        if transcribe_clicked:
            if audio_hash and audio_hash == st.session_state.get("last_audio_hash"):
                st.info("Este audio ya fue transcrito (no vuelvo a llamar a Whisper).")
            else:
                try:
                    with st.spinner("Transcribiendo..."):
                        if asr_backend == "OpenAI Whisper API":
                            transcript = asr_openai_whisper(audio_bytes, catalog_names, language=lang_code)
                        else:
                            transcript = asr_faster_whisper(audio_bytes, catalog_names, language=lang_code)

                    prev = (st.session_state.get("transcript_area") or "").strip()
                    st.session_state["transcript_area"] = (prev + "\n" + transcript).strip() if prev else transcript
                    st.session_state["last_audio_hash"] = audio_hash

                    st.success("Transcripción lista ✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error transcribiendo: {e}")

        st.text_area(
            "Transcripción (editable)",
            height=170,
            key="transcript_area",
            placeholder="Aquí aparecerá la transcripción… o escribe/pega manualmente."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================
    # MANUAL MODE
    # =========================================================
    if mode == "⌨️ Manual":
        st.markdown('<div class="vc-card">', unsafe_allow_html=True)
        st.markdown('<div class="vc-h">⌨️ Entrada manual</div>', unsafe_allow_html=True)
        st.markdown('<div class="vc-caption">Escribe o pega el pedido aquí. Sin audio, sin transcripción.</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([2, 1])
        with c1:
            st.button("✅ Listo para parsear", type="primary", disabled=True, key="btn_ready_manual")
        with c2:
            clear_clicked = st.button("🧹 Limpiar", key="btn_clear_all_manual")

        if clear_clicked:
            _clear_all(used_component_recorder=False)

        st.text_area(
            "Pedido",
            height=200,
            key="transcript_area",
            placeholder="Ej: 3 cajas cerveza, 2kg limones, 10 botellas agua..."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================
    # Shared: PARSE
    # =========================================================
    raw_text = (st.session_state.get("transcript_area") or "").strip()

    st.markdown('<div class="vc-card">', unsafe_allow_html=True)
    st.markdown('<div class="vc-h">🔎 Parseo</div>', unsafe_allow_html=True)
    st.markdown('<div class="vc-caption">Convierte el texto en líneas con producto/cantidad/unidad.</div>', unsafe_allow_html=True)

    parse_clicked = st.button(
        "✨ Parsear pedido",
        type="primary",
        disabled=not bool(raw_text),
        key="btn_parse_order"
    )

    if parse_clicked:
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
            name_norm = normalize_text(name)
            match_norm, score = fuzzy_match(name_norm, catalog_names)
            prod = catalog_norm_to_product.get(match_norm) if match_norm else None

            parsed_rows.append({
                "spoken_name": name,
                "matched_name": prod.name if prod else None,
                "confidence": round(score or 0.0, 1),
                "quantity": qty,
                "unit": unit,
                "status": "OK" if prod else "Revisar",
            })

        aggregated = aggregate_parsed_rows(parsed_rows, name_to_product)
        with get_session() as s:
            st.session_state["parsed_df"] = add_provider_column(s, pd.DataFrame(aggregated))

        st.success("Parseo listo ✅")
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================
    # Result editor
    # =========================================================
    if "parsed_df" in st.session_state:
        st.markdown('<div class="vc-card">', unsafe_allow_html=True)
        st.markdown('<div class="vc-h">✅ Resultado</div>', unsafe_allow_html=True)

        edited = st.data_editor(
            st.session_state["parsed_df"],
            width="stretch",
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "confidence": st.column_config.NumberColumn("Confianza", help="0-100"),
                "quantity": st.column_config.NumberColumn("Cantidad"),
                "unit": st.column_config.TextColumn("Unidad"),
                "matched_name": st.column_config.TextColumn("Producto (catálogo)"),
                "provider": st.column_config.TextColumn("Proveedor (catálogo)", disabled=True),
            },
            key="parse_editor",
        )
        with get_session() as s:
            st.session_state["parsed_df"] = add_provider_column(s, edited)

        st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================
    # Save actions
    # =========================================================
    has_parsed = (
        "parsed_df" in st.session_state
        and isinstance(st.session_state["parsed_df"], pd.DataFrame)
        and not st.session_state["parsed_df"].empty
    )
    if not has_parsed:
        return

    df_to_use = st.session_state["parsed_df"]

    st.markdown('<div class="vc-card">', unsafe_allow_html=True)
    st.markdown('<div class="vc-h">💾 Guardar</div>', unsafe_allow_html=True)
    st.markdown('<div class="vc-caption">Crea un nuevo borrador o añade a uno existente.</div>', unsafe_allow_html=True)

    b1, b2 = st.columns(2)
    save_draft_clicked = b1.button("💾 Nuevo borrador", type="primary", key="btn_save_new_draft")
    add_to_existing_clicked = b2.button("➕ Añadir a borrador", key="btn_add_to_existing")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- SAVE AS NEW DRAFT ----------------
    if save_draft_clicked:
        with get_session() as s:
            new_order = Order(status="draft", title=None, venue_id=venue_id)
            s.add(new_order)
            s.commit()
            s.refresh(new_order)
            oid = new_order.id

            merged = {}
            for _, row in df_to_use.iterrows():
                matched_name = safe_str(row.get("matched_name")).strip()
                qty = float(row.get("quantity") or 0.0)
                unit_raw = safe_str(row.get("unit")).strip().lower()
                conf = float(row.get("confidence") or 0.0)

                if not matched_name and qty == 0.0 and not unit_raw:
                    continue

                prod_obj = (
                    s.exec(select(Product).where(Product.venue_id == venue_id, Product.name == matched_name)).first()
                    if matched_name else None
                )
                pid = prod_obj.id if prod_obj else None

                unit_val = (str(prod_obj.unit).strip().lower() if (prod_obj and getattr(prod_obj, "unit", None)) else unit_raw)
                unit_val = unit_val or "unidad"

                key = ("pid", pid) if pid is not None else ("name", matched_name.lower())

                if key in merged:
                    merged[key]["quantity"] = float(merged[key]["quantity"] or 0.0) + qty
                    merged[key]["confidence"] = max(float(merged[key]["confidence"] or 0.0), conf)
                    if not (merged[key].get("unit") or "").strip():
                        merged[key]["unit"] = unit_val
                else:
                    merged[key] = dict(
                        venue_id=venue_id,
                        order_id=oid,
                        product_id=pid,
                        spoken_name=str(row.get("spoken_name") or "").lower(),
                        matched_name=(matched_name if matched_name else None),
                        confidence=conf,
                        quantity=qty,
                        unit=unit_val,
                    )

            for payload in merged.values():
                s.add(OrderLine(**payload))

            s.commit()

        st.session_state["active_order_id"] = oid
        st.session_state["adding_to_draft_mode"] = False
        st.success(f"Borrador guardado ✅ (ID {oid})")
        st.rerun()

    # ---------------- ADD TO EXISTING DRAFT ----------------
    with get_session() as s:
        drafts = s.exec(
            select(Order)
            .where(Order.venue_id == venue_id, Order.status == "draft")
            .order_by(Order.created_at.desc())
        ).all()

    if add_to_existing_clicked:
        if not drafts:
            st.warning("No hay borradores disponibles. Crea uno nuevo primero.")
        else:
            st.session_state["adding_to_draft_mode"] = True

    if st.session_state.get("adding_to_draft_mode", False) and drafts:
        st.markdown('<div class="vc-card">', unsafe_allow_html=True)
        st.markdown('<div class="vc-h">➕ Añadir a borrador</div>', unsafe_allow_html=True)

        draft_options = [(o.id, f"#{o.id} — {o.title or o.created_at.strftime('%Y-%m-%d %H:%M')}") for o in drafts]
        labels = [lbl for _, lbl in draft_options]
        ids = [oid for oid, _ in draft_options]

        chosen_label = st.selectbox("Borrador", options=labels, key="choose_draft_select")
        selected_draft_id = ids[labels.index(chosen_label)]

        cA, cB = st.columns([1, 1])
        confirm_add = cA.button("✅ Confirmar", type="primary", key="confirm_add_lines")
        cancel_add = cB.button("❌ Cancelar", key="cancel_add_lines")

        if cancel_add:
            st.session_state["adding_to_draft_mode"] = False
            st.rerun()

        if confirm_add:
            oid = selected_draft_id
            from sqlmodel import and_

            with get_session() as s:
                for _, row in df_to_use.iterrows():
                    matched_name = safe_str(row.get("matched_name")).strip()
                    qty_to_add = float(row.get("quantity") or 0.0)
                    if not matched_name and qty_to_add == 0.0:
                        continue

                    prod_obj = (
                        s.exec(select(Product).where(Product.venue_id == venue_id, Product.name == matched_name)).first()
                        if matched_name else None
                    )
                    pid = prod_obj.id if prod_obj else None

                    unit_val = (
                        str(prod_obj.unit).strip().lower()
                        if (prod_obj and getattr(prod_obj, "unit", None))
                        else safe_str(row.get("unit")).strip().lower()
                    )
                    unit_val = unit_val or "unidad"

                    existing_line = None
                    if pid is not None:
                        existing_line = s.exec(
                            select(OrderLine).where(and_(OrderLine.order_id == oid, OrderLine.product_id == pid))
                        ).first()

                    if existing_line is None and matched_name:
                        existing_line = s.exec(
                            select(OrderLine).where(and_(OrderLine.order_id == oid, OrderLine.matched_name == matched_name))
                        ).first()

                    if existing_line:
                        existing_line.quantity = float(existing_line.quantity or 0.0) + qty_to_add
                        if not (existing_line.unit or "").strip():
                            existing_line.unit = unit_val
                        existing_line.confidence = max(
                            float(existing_line.confidence or 0.0),
                            float(row.get("confidence") or 0.0),
                        )
                        s.add(existing_line)
                    else:
                        s.add(OrderLine(
                            venue_id=venue_id,
                            order_id=oid,
                            product_id=pid,
                            spoken_name=str(row.get("spoken_name") or "").lower(),
                            matched_name=(matched_name if matched_name else None),
                            confidence=float(row.get("confidence") or 0.0),
                            quantity=qty_to_add,
                            unit=unit_val,
                        ))

                s.commit()

            st.session_state["active_order_id"] = oid
            st.session_state["adding_to_draft_mode"] = False
            st.success(f"Líneas añadidas ✅ (ID {oid}).")
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


 
#=============================ORDER===========================================================================

def orders_tab(venue_id: int, role: str | None = None):
    st.subheader("📜 Pedidos")
    
    expected_cols = ["line_id", "matched_name", "quantity", "unit",
                  "provider"]
    
    
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
    
    with get_session() as s:
        orders = s.exec(select(Order).where(Order.venue_id == venue_id).order_by(Order.created_at.desc())).all()
    if not orders:
        st.info("No hay pedidos aún. Crea uno en 'Nuevo pedido'.")
        return
    
       # Selector del pedido
    oid_labels = [
        f"#{o.id} — {o.title or o.created_at.strftime('%Y-%m-%d %H:%M')}"
        for o in orders
    ]
    choice = st.selectbox("Selecciona un pedido", options=list(range(len(orders))), format_func=lambda i: oid_labels[i])
    order = orders[choice]
    st.session_state["active_order_id"] = order.id
    
    # ✅ Si cambias de pedido, fuerza recarga del editor desde DB
    if st.session_state.get("last_orders_tab_order_id") != order.id:
        st.session_state["last_orders_tab_order_id"] = order.id
        st.session_state.pop(f"order_editor_df_{order.id}", None)


    cA, _ = st.columns([2,1])
    with cA:
        order_title = st.text_input("Título del pedido", value=order.title or "", key=f"order_title_{order.id}")


    # Save order header if changed
    if (order.title or "") != order_title:
        with get_session() as s:
            o = s.exec(select(Order).where(Order.id == order.id)).first()
            if o:
                o.title = order_title or None
                s.add(o)
                s.commit()
                
        
    # ---------- Load products once (for dropdowns & supplier drafts) ----------
    with get_session() as s:
        products = s.exec(select(Product).where(Product.venue_id == venue_id).order_by(Product.name.asc(), Product.provider_name.asc())).all()

    # UX: dropdown labels (unique by id)
    def _product_label(p: Product) -> str:
        prov = getattr(p, 'provider_name', None) or '(Sin proveedor)'
        return f"{p.name} — {prov} (#{p.id})"

    placeholder_product = "— Selecciona producto —"
    product_labels = [placeholder_product] + [_product_label(p) for p in products]
    label_to_product = {lbl: p for lbl, p in zip(product_labels[1:], products)}
    id_to_product = {getattr(p, "id", None): p for p in products}

    # Units dropdown (DB-driven + sensible fallbacks)
    with get_session() as s:
        unit_opts = distinct_units(products)
    unit_placeholder = "—"
    unit_options = [unit_placeholder] + sorted({u.strip().lower() for u in (unit_opts or []) if str(u).strip()})
    # Always include common units
    for u in ["unidad", "kg", "g", "l", "ml", "caja", "paquete"]:
        if u not in unit_options:
            unit_options.append(u)

    def _apply_product_selection(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize df based on product_label/product_id. Ensures matched_name/provider/unit are consistent."""
        if df.empty:
            return df

        df = df.copy()

        for i, row in df.iterrows():
            lbl = safe_str(row.get("product_label")).strip()
            pid = row.get("product_id")

            prod = None
            if lbl and lbl != placeholder_product and lbl in label_to_product:
                prod = label_to_product[lbl]
            elif pd.notna(pid) and pid in id_to_product:
                prod = id_to_product.get(pid)

            if prod:
                df.at[i, "product_id"] = getattr(prod, "id", None)
                df.at[i, "matched_name"] = getattr(prod, "name", "") or ""
                df.at[i, "provider"] = getattr(prod, "provider_name", "") or ""
                # ✅ unit SIEMPRE viene del catálogo (otra unidad = otro producto_id)
                df.at[i, "unit"] = safe_str(getattr(prod, "unit", "")).strip().lower() or "unidad"

                # keep product_label coherent
                if lbl == "" or lbl == placeholder_product:
                    df.at[i, "product_label"] = f"{prod.name} — {getattr(prod, 'provider_name', None) or '(Sin proveedor)'} (#{prod.id})"
            else:
                # No product selected: keep editable values but blank computed fields
                if lbl == placeholder_product or not lbl:
                    df.at[i, "product_id"] = pd.NA
                    df.at[i, "matched_name"] = ""
                    df.at[i, "provider"] = ""
                    # keep unit as-is
        return df

    def render_order_editor(order_obj: Order) -> pd.DataFrame:
        """Editor compacto:
        - UI muestra SOLO: ID, Producto, Cantidad, Unidad, Eliminar
        - Producto y Unidad BLOQUEADOS (vienen del catálogo / product_id)
        - Para cambiar “unidad” => es otro Product (otro product_id) => se añade con ➕ Añadir línea.
        """

        visible_cols = ["product_label", "quantity", "unit", "eliminar"]
        full_cols = ["line_id", "product_label", "product_id", "matched_name", "provider", "quantity", "unit", "eliminar"]

        def coerce_df(obj, columns):
            if isinstance(obj, pd.DataFrame):
                df = obj.copy()
            elif isinstance(obj, (list, tuple)):
                df = pd.DataFrame(obj) if obj else pd.DataFrame(columns=columns)
            elif isinstance(obj, dict):
                df = pd.DataFrame([obj])
            else:
                df = pd.DataFrame(columns=columns)

            for c in columns:
                if c not in df.columns:
                    df[c] = pd.NA

            # checkbox must be bool-compatible
            if "eliminar" in df.columns:
                df["eliminar"] = df["eliminar"].fillna(False)

            return df[columns]

        def norm_types(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            df = df.copy()

            for col in ["product_label", "matched_name", "provider", "unit"]:
                if col in df.columns:
                    df[col] = df[col].astype("string")

            for col in ["quantity", "line_id", "product_id"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            if "eliminar" in df.columns:
                df["eliminar"] = df["eliminar"].fillna(False).astype(bool)

            return df

        # ---------- Load lines from DB ----------
        with get_session() as s:
            lines = s.exec(select(OrderLine).where(OrderLine.order_id == order_obj.id)).all()

        rows = []
        for ln in lines:
            pid = getattr(ln, "product_id", None)
            prod = id_to_product.get(pid) if pid is not None else None

            if prod:
                lbl = _product_label(prod)  # must match product_labels options
                matched_name = getattr(prod, "name", "") or (ln.matched_name or "")
                provider = getattr(prod, "provider_name", "") or ""
                unit_val = safe_str(getattr(prod, "unit", "")).strip().lower() or "unidad"  # ✅ DB manda
                qty_val = getattr(ln, "quantity", None)
            else:
                lbl = placeholder_product
                matched_name = (ln.matched_name or "")
                provider = ""
                unit_val = safe_str(getattr(ln, "unit", "")).strip().lower() or "unidad"
                qty_val = getattr(ln, "quantity", None)

            rows.append({
                "line_id": getattr(ln, "id", pd.NA),
                "product_label": lbl,
                "product_id": pid if pid is not None else pd.NA,
                "matched_name": matched_name,
                "provider": provider,
                "quantity": qty_val,
                "unit": unit_val,
                "eliminar": False,
            })

        # ---------- Session keys ----------
        state_key_df = f"order_editor_df_{order_obj.id}"
        state_key_widget = f"order_editor_widget_{order_obj.id}"
        baseline_key = f"{state_key_df}__baseline"

        # Init/refresh DF (FULL, hidden cols included)
        if state_key_df not in st.session_state:
            df0 = coerce_df(rows, full_cols)
            df0 = norm_types(df0)
            df0 = _apply_product_selection(df0)

            # drop only placeholders at init
            def _keep_init(r):
                lbl = safe_str(r.get("product_label")).strip()
                return lbl not in ("", placeholder_product)

            df0 = df0[df0.apply(_keep_init, axis=1)].reset_index(drop=True)

            st.session_state[state_key_df] = df0
            st.session_state[baseline_key] = df0.copy(deep=True)
        else:
            # if wiped but DB has rows => reload
            cur = st.session_state.get(state_key_df)
            if isinstance(cur, pd.DataFrame) and cur.empty and rows:
                df0 = coerce_df(rows, full_cols)
                df0 = norm_types(df0)
                df0 = _apply_product_selection(df0)
                st.session_state[state_key_df] = df0.reset_index(drop=True)
                st.session_state[baseline_key] = st.session_state[state_key_df].copy(deep=True)

        # ---------- Add line popover (only product_id not already present) ----------
        df_full_for_add = coerce_df(st.session_state.get(state_key_df), full_cols)
        df_full_for_add = norm_types(df_full_for_add)

        pid_series = df_full_for_add["product_id"] if "product_id" in df_full_for_add.columns else pd.Series([], dtype="float")
        used_ids = set(pd.to_numeric(pid_series, errors="coerce").dropna().astype(int).tolist())

        remaining_products = [p for p in products if getattr(p, "id", None) not in used_ids]
        remaining_labels = [_product_label(p) for p in remaining_products]

        with st.popover("➕ Añadir línea"):
            if not remaining_products:
                st.info("Ya has añadido todos los productos del catálogo a este pedido.")
            else:
                new_lbl = st.selectbox("Producto", options=remaining_labels, key=f"add_prod_{order_obj.id}")
                new_qty = st.number_input("Cantidad", min_value=0.0, value=1.0, step=1.0, key=f"add_qty_{order_obj.id}")
                add_confirm = st.button("✅ Añadir", type="primary", key=f"add_confirm_{order_obj.id}")

                if add_confirm:
                    p = label_to_product.get(new_lbl)
                    if p is None:
                        p = next((pp for pp in products if _product_label(pp) == new_lbl), None)

                    if p is not None:
                        df_tmp = coerce_df(st.session_state.get(state_key_df), full_cols)
                        df_tmp = norm_types(df_tmp)

                        df_tmp.loc[len(df_tmp)] = {
                            "line_id": pd.NA,
                            "product_label": _product_label(p),
                            "product_id": int(getattr(p, "id")),
                            "matched_name": getattr(p, "name", "") or "",
                            "provider": getattr(p, "provider_name", "") or "",
                            "quantity": float(new_qty),
                            "unit": safe_str(getattr(p, "unit", "")).strip().lower() or "unidad",  # ✅ DB manda
                            "eliminar": False,
                        }

                        df_tmp = norm_types(df_tmp)
                        df_tmp = _apply_product_selection(df_tmp).reset_index(drop=True)

                        st.session_state[state_key_df] = df_tmp
                        st.session_state[baseline_key] = df_tmp.copy(deep=True)

                        st.session_state.pop(state_key_widget, None)
                        st.rerun()

        # ---------- Editor (VISIBLE ONLY) ----------
        st.subheader("🧾 Líneas del pedido")

        colcfg = {
            "line_id": st.column_config.Column("ID", disabled=True, width="small"),
            "product_label": st.column_config.Column(
                "Producto",
                disabled=True,
                width="large",
                help="Producto fijado. Para cambiarlo: marca Eliminar y añade una nueva línea.",
            ),
            "quantity": st.column_config.NumberColumn("Cantidad", min_value=0.0, step=1.0, format="%.2f"),
            "unit": st.column_config.Column(
                "Unidad",
                disabled=True,
                width="small",
                help="Unidad fija del catálogo. Otra unidad = otro producto (otro ID).",
            ),
            "eliminar": st.column_config.CheckboxColumn("Eliminar", help="Marca para eliminar esta línea"),
        }

        df_full = coerce_df(st.session_state.get(state_key_df), full_cols)
        df_full = norm_types(df_full).reset_index(drop=True)

        df_view = df_full[visible_cols].copy()

        edited_view = st.data_editor(
            df_view.reset_index(drop=True),
            width="stretch",
            num_rows="fixed",
            column_config=colcfg,
            hide_index=True,
            key=state_key_widget,
        )

        # Sync visible -> full (product/unit stay as-is, but we still re-apply for safety)
        df_vis = coerce_df(edited_view, visible_cols)
        df_vis = norm_types(df_vis).reset_index(drop=True)

        n = min(len(df_full), len(df_vis))
        df_full = df_full.iloc[:n].reset_index(drop=True)
        df_vis = df_vis.iloc[:n].reset_index(drop=True)

        for c in visible_cols:
            df_full[c] = df_vis[c].values

        df_full = norm_types(df_full)
        df_full = _apply_product_selection(df_full).reset_index(drop=True)

        # keep rows unless completely empty (or marked delete)
        def _keep_row(r):
            if bool(r.get("eliminar") or False):
                return True
            lbl = safe_str(r.get("product_label")).strip()
            qty = r.get("quantity")
            has_product = lbl not in ("", placeholder_product)
            has_qty = qty is not None and str(qty) != "nan" and float(qty or 0) != 0.0
            return has_product or has_qty

        df_full = df_full[df_full.apply(_keep_row, axis=1)].reset_index(drop=True)

        st.session_state[state_key_df] = df_full

        # ---------- Actions ----------
        c1, c2 = st.columns([1, 1])
        with c1:
            apply_btn = st.button("💾 Guardar cambios", type="primary", key=f"apply_changes_{order_obj.id}")
        with c2:
            discard_btn = st.button("↩️ Deshacer cambios", key=f"discard_changes_{order_obj.id}")

        # ---- Discard ----
        if discard_btn:
            if baseline_key in st.session_state:
                st.session_state[state_key_df] = st.session_state[baseline_key].copy(deep=True)
            else:
                df0 = coerce_df(rows, full_cols)
                df0 = norm_types(df0)
                df0 = _apply_product_selection(df0)
                st.session_state[state_key_df] = df0.reset_index(drop=True)
                st.session_state[baseline_key] = st.session_state[state_key_df].copy(deep=True)

            st.session_state.pop(state_key_widget, None)
            st.rerun()

        # ---- Apply (delete + upsert) ----
        if apply_btn:
            df_edit = coerce_df(st.session_state.get(state_key_df), full_cols)
            df_edit = norm_types(df_edit)

            to_delete_ids = (
                pd.to_numeric(df_edit.loc[df_edit["eliminar"] == True, "line_id"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )

            df_keep = df_edit[df_edit["eliminar"] != True].copy()

            df_keep["product_id"] = pd.to_numeric(df_keep["product_id"], errors="coerce")
            df_keep["line_id"] = pd.to_numeric(df_keep["line_id"], errors="coerce")
            df_keep["quantity"] = pd.to_numeric(df_keep["quantity"], errors="coerce")
            df_keep = df_keep.dropna(subset=["product_id"]).copy()
            df_keep["product_id"] = df_keep["product_id"].astype(int)

            # merge duplicates by product_id: sum quantities (unit fixed by DB anyway)
            if not df_keep.empty:
                df_keep = (
                    df_keep.groupby("product_id", as_index=False)
                    .agg({
                        "line_id": "first",
                        "product_label": "first",
                        "matched_name": "first",
                        "provider": "first",
                        "unit": "first",
                        "quantity": "sum",
                    })
                )

            with get_session() as s:
                if to_delete_ids:
                    lines_db = s.exec(select(OrderLine).where(OrderLine.id.in_(to_delete_ids))).all()
                    for ln in lines_db:
                        s.delete(ln)

                for _, row in df_keep.iterrows():
                    pid = int(row["product_id"])
                    prod = id_to_product.get(pid)
                    matched_name = getattr(prod, "name", None) if prod else safe_str(row.get("matched_name")).strip() or None
                    unit_val = safe_str(getattr(prod, "unit", "") if prod else row.get("unit")).strip().lower() or "unidad"
                    qty = float(row.get("quantity") or 0.0)

                    line_id = row.get("line_id")
                    line_id_int = int(line_id) if line_id is not None and str(line_id) != "nan" else None

                    if line_id_int is not None:
                        ln = s.exec(select(OrderLine).where(OrderLine.id == line_id_int)).first()
                        if ln:
                            ln.product_id = pid
                            ln.matched_name = matched_name
                            ln.quantity = qty
                            ln.unit = unit_val
                            s.add(ln)
                        else:
                            s.add(OrderLine(order_id=order_obj.id, product_id=pid, matched_name=matched_name, quantity=qty, unit=unit_val, spoken_name=(matched_name or "")))
                    else:
                        s.add(OrderLine(order_id=order_obj.id, product_id=pid, matched_name=matched_name, quantity=qty, unit=unit_val, spoken_name=(matched_name or "")))

                s.commit()

            # refresh from DB so IDs are correct
            with get_session() as s:
                lines2 = s.exec(select(OrderLine).where(OrderLine.order_id == order_obj.id)).all()

            rows2 = []
            for ln in lines2:
                pid = getattr(ln, "product_id", None)
                prod = id_to_product.get(pid) if pid is not None else None

                if prod:
                    lbl = _product_label(prod)
                    provider = getattr(prod, "provider_name", "") or ""
                    matched_name = getattr(prod, "name", "") or ""
                    unit_val = safe_str(getattr(prod, "unit", "")).strip().lower() or "unidad"
                else:
                    lbl = placeholder_product
                    provider = ""
                    matched_name = getattr(ln, "matched_name", "") or ""
                    unit_val = safe_str(getattr(ln, "unit", "")).strip().lower() or "unidad"

                rows2.append({
                    "line_id": getattr(ln, "id", pd.NA),
                    "product_label": lbl,
                    "product_id": pid if pid is not None else pd.NA,
                    "matched_name": matched_name,
                    "provider": provider,
                    "quantity": getattr(ln, "quantity", None),
                    "unit": unit_val,
                    "eliminar": False,
                })

            df_saved = coerce_df(rows2, full_cols)
            df_saved = norm_types(df_saved)
            df_saved = _apply_product_selection(df_saved)

            st.session_state[state_key_df] = df_saved.reset_index(drop=True)
            st.session_state[baseline_key] = st.session_state[state_key_df].copy(deep=True)
            st.session_state.pop(state_key_widget, None)

            st.success("Cambios guardados ✅")
            st.rerun()

        return st.session_state[state_key_df]



    # Render the upgraded editor and keep current df for downstream (supplier drafts)
    df_current = render_order_editor(order)
# =============================
    # ✉️ Drafts for Suppliers (nice)
    # =============================
    st.subheader("✉️ Borradores para proveedores")

    # Catalog lookups (already loaded above for dropdowns)
    name_to_product = {getattr(p, "name", ""): p for p in products}

    # df_current comes from the editor above; just ensure it is a DataFrame
    if not isinstance(df_current, pd.DataFrame):
        df_current = pd.DataFrame(df_current or [])

    # Group lines by supplier (using product metadata)
    grouped: Dict[str, Dict[str, object]] = {}
    for _, row in df_current.iterrows():
        matched_name = safe_str(row.get("matched_name")).strip()
        pid = row.get("product_id") if "product_id" in df_current.columns else None

        prod = None
        if pid is not None and pd.notna(pid):
            try:
                prod = id_to_product.get(int(pid))
            except Exception:
                prod = None

        if not prod and matched_name:
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
       
        
def main():
    header()
    init_auth_db()

    # Login + top bar + venue selector (ONLY ONCE)
    auth_gate(show_manage_org=False)
    require_login()

    active = current_active_venue()
    if not active:
        st.warning("No venue access yet. Ask an admin to grant you access.")
        st.stop()

    venue, venue_role = active  # venue_role is owner/manager/staff/viewer

    # Show Manage tab only for owner/manager
    if venue_role in {"owner", "manager"}:
        tabs = st.tabs(["🏢 Manage organization", "📦 Catálogo", "🆕 Nuevo pedido", "📜 Pedidos"])
        with tabs[0]:
            manage_organization_ui(venue_role=venue_role)  # pass role in
        with tabs[1]:
            catalog_tab(venue.id, venue_role)
        with tabs[2]:
            new_order_tab(venue.id, venue_role)
        with tabs[3]:
            orders_tab(venue.id, venue_role)
    else:
        tabs = st.tabs(["🆕 Nuevo pedido", "📜 Pedidos"])
        with tabs[0]:
            new_order_tab(venue.id, venue_role)
        with tabs[1]:
            orders_tab(venue.id, venue_role)

main()
