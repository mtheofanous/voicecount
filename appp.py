"""
Streamlit Voice Inventory MVP (single-file app)
------------------------------------------------
Run locally:
  pip install streamlit sqlmodel rapidfuzz unidecode python-dateutil faster-whisper torch numpy sounddevice

Then:
  streamlit run app.py

This single file provides two modes via sidebar:
  1) Products DB  — create and manage your product catalog (with provider fields)
  2) Voice Order  — dictate items to order and quantities (supports number-first or name-first, and the keyword 'next')
"""

from __future__ import annotations
import io
import csv
from datetime import datetime, date
from typing import List, Optional, Dict,Tuple

import os
import re
import tempfile
import wave
import numpy as np
import streamlit as st
from sqlmodel import SQLModel, Field, Session, create_engine, select
SQLModel.metadata.clear()
from rapidfuzz import process, fuzz
from unidecode import unidecode
from collections import defaultdict
from audiorecorder import audiorecorder
import unicodedata

# -------------------------
# Database Models
# -------------------------

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
# App Setup / DB Engine
# -------------------------

st.set_page_config(page_title="Voice Inventory MVP", page_icon="🎤", layout="wide")

if "engine" not in st.session_state:
    engine = create_engine("sqlite:///voice_inventory.db")
    SQLModel.metadata.create_all(engine)
    st.session_state.engine = engine

def get_session() -> Session:
    return Session(st.session_state.engine)


# -------------------------
# Fuzzy Matching & Parsing (kept from your app)
# -------------------------

UNITS = {
    "unidad", "unidades", "botella", "botellas", "kg", "kilo", "kilos", "lata", "latas",
    "caja", "cajas", "pack", "packs", "u", "ud", "uds"
}

# Patterns accept number-first and name-first (e.g., "3 botellas de gin", "aperol 1")
PATTERNS = [
    re.compile(r"\b(\d+[\.,]?\d*)\s*(\w+)?\s*(?:de\s)?([a-zA-Záéíóúñçüöä\s]+?)\b"),
    re.compile(r"\b([a-zA-Záéíóúñçüöä\s]+?)\s(\d+[\.,]?\d*)\s*(\w+)?\b"),
]

NEXT_TOKENS = {"next", "siguiente", "sig", ",", ";", " y ", " and "}

# Common filler verbs/phrases at the beginning of a spoken fragment
FILLER_PREFIX = re.compile(
    r"^\s*(?:quiero|ponme|pon|me pones|dame|trae(?:me)?|tr\u00e1eme|anade|añade|agrega|mete|sum[ae])\s+",
    flags=re.IGNORECASE,
)

def strip_filler_prefix(text: str) -> str:
    t = text or ""
    # remove repeated fillers if user says "quiero quiero ..."
    while True:
        t2 = FILLER_PREFIX.sub("", t)
        if t2 == t:
            break
        t = t2
    return t.strip()


# Spanish number words mapping + normalizer
NUM_WORDS_ES = {
    "cero": 0, "un": 1, "una": 1, "uno": 1,
    "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
    "diez": 10, "once": 11, "doce": 12
}

def normalize_text(text: str) -> str:
    """Return lowercase, stripped, and singular version of a word or phrase."""
    if not isinstance(text, str):
        return text
    text = unidecode(text.strip().lower())

    # Simple plural → singular rules (both English + Spanish basics)
    # Handles words ending with: s, es, ies, les, nes, etc.
    rules = [
        (r"([^aeiou])ies$", r"\1y"),      # berries -> berry
        (r"([aeiou])s$", r"\1"),          # kilos -> kilo
        (r"([nrlsdz])es$", r"\1"),        # limones -> limon, botellas -> botella
        (r"s$", ""),                      # packs -> pack
    ]
    for pattern, repl in rules:
        if re.search(pattern, text):
            text = re.sub(pattern, repl, text)
            break
    return text

def normalize_model(obj):
    """Normalize all string fields of a model to lowercase & singular."""
    for name, value in vars(obj).items():
        if isinstance(value, str):
            setattr(obj, name, normalize_text(value))
            
            
def aggregate_parsed_rows(rows: list[dict], name_to_product: dict[str, Product], default_unit: str = "unidad") -> list[dict]:
    """
    Agrupa por producto, suma cantidades y decide la unidad:
    1) unidad más frecuente entre filas agregadas
    2) si no hay, unidad del producto en DB (si existe)
    3) si tampoco hay, usa 'unidad'
    """
    by_name: dict[str, dict] = {}
    unit_count: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

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
                "unit": "",          # se decide al final
                "status": status or "Revisar",
            }

        by_name[name_key]["quantity"] += qty
        by_name[name_key]["confidence"] = max(by_name[name_key]["confidence"], conf)
        if by_name[name_key]["status"] != "OK" and status == "OK":
            by_name[name_key]["status"] = "OK"

        if unit:
            unit_count[name_key][unit] += 1

    # Asignar unidad final
    for k, row in by_name.items():
        # 1) unidad más frecuente si hubo
        chosen_unit = ""
        if unit_count[k]:
            chosen_unit = max(unit_count[k].items(), key=lambda kv: kv[1])[0]

        # 2) si no hay, mira la unidad del producto en DB
        if not chosen_unit:
            matched = (row.get("matched_name") or "").strip()
            prod = name_to_product.get(matched) if matched else None
            if prod and (prod.unit or "").strip():
                chosen_unit = prod.unit.strip().lower()

        # 3) si sigue vacío, usa default
        if not chosen_unit:
            chosen_unit = default_unit

        row["unit"] = chosen_unit

        # pretty quantity
        q = row["quantity"]
        row["quantity"] = int(q) if abs(q - int(q)) < 1e-9 else round(q, 2)

    return list(by_name.values())

def replace_number_words(text: str) -> str:
    choices = "|".join(sorted(NUM_WORDS_ES.keys(), key=len, reverse=True))
    patt = re.compile(rf"\b({choices})\b", flags=re.IGNORECASE)
    def repl(m):
        w = m.group(1).lower()
        return str(NUM_WORDS_ES.get(w, w))
    return patt.sub(repl, text)

def tokenize_items(raw_text: str) -> List[str]:
    """Split dictation into item-like chunks using 'next' and punctuation."""
    text = unidecode(raw_text.lower())
    sep_text = text
    for tok in [" next ", " siguiente ", " sig ", ",", ";", " y ", " and "]:
        sep_text = sep_text.replace(tok, " | ")
    sep_text = re.sub(r"\bnext\b", "|", sep_text)
    parts = [p.strip(" |\t\n") for p in sep_text.split("|")]
    return [p for p in parts if p]

def parse_item(fragment: str) -> Optional[Tuple[str, float, Optional[str]]]:
    """
    Extract (name, qty, unit) robustly.
    Handles:
      - "1 caja de coca cola" / "una caja de coca cola" -> qty=1, unit=caja, name=coca cola
      - "2 kg tomate" / "2 botellas de gin" -> qty, unit, name
      - "tomate 3" (name-first) -> qty=3, name=tomate
      - "pepino" (no number) -> qty=1, name=pepino
    """
    frag = (fragment or "").strip().lower()
    frag = unidecode(frag)
    if not frag:
        return None

    # NEW: remove filler verbs like "quiero", "ponme", etc.
    frag = strip_filler_prefix(frag)

    # number words -> digits (e.g., "una" -> "1", "tres" -> "3")
    try:
        frag = replace_number_words(frag)
    except Exception:
        pass

    # drop leading articles like "la/el/los/las de/del"
    frag = re.sub(r"^\s*(?:de|del|la|el|los|las)\s+", "", frag)

    # 1) number-first: "<qty> <maybe-unit> [de] <name>"
    m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s+(.*)$", frag)
    if m:
        qty_s = m.group(1)
        rest  = m.group(2).strip()

        parts = re.split(r"\s+", rest)
        unit = None
        name_part = rest

        if parts:
            first = parts[0]
            if first in UNITS:
                unit = first
                name_part = " ".join(parts[1:]).strip()
                name_part = re.sub(r"^(?:de|del)\s+", "", name_part)

        # if no valid unit, treat whole rest as name (e.g., "pepinos")
        name = normalize_text(name_part)
        name = re.sub(r"[^a-z\s]", "", name).strip()

        try:
            qty = float(qty_s.replace(",", "."))
        except Exception:
            qty = 1.0

        if len(name) >= 2:
            return (name, qty, unit if unit in UNITS else None)

    # 2) name-first + number at the end: "<maybe-unit> <name> <qty>"
    m2 = re.match(r"^(.*\D)\s+(\d+(?:[.,]\d+)?)\s*$", frag)
    if m2:
        name_part = m2.group(1).strip()
        qty_s     = m2.group(2)

        unit = None
        parts = re.split(r"\s+", name_part)
        if parts and parts[0] in UNITS:
            unit = parts[0]
            name_part = " ".join(parts[1:]).strip()
            name_part = re.sub(r"^(?:de|del)\s+", "", name_part)

        name = normalize_text(name_part)
        name = re.sub(r"[^a-z\s]", "", name).strip()

        try:
            qty = float(qty_s.replace(",", "."))
        except Exception:
            qty = 1.0

        if len(name) >= 2:
            return (name, qty, unit if unit in UNITS else None)

    # 3) fallback: just a name -> qty=1
    name_only = re.sub(r"[^a-z\s]", "", frag).strip()
    if len(name_only) >= 2:
        name_only = normalize_text(name_only)
        return (name_only, 1.0, None)

    return None


def fuzzy_match(name: str, catalog: List[str]) -> Tuple[Optional[str], float]:
    if not catalog:
        return None, 0.0
    best = process.extractOne(name, catalog, scorer=fuzz.WRatio)
    if best is None:
        return None, 0.0
    match_name, score, _ = best
    return match_name, float(score)

# -------------------------
# Whisper (offline) + local mic helpers (NEW)
# -------------------------

from faster_whisper import WhisperModel
import sounddevice as sd

@st.cache_resource(show_spinner=False)
def load_whisper(model_size: str = "small"):
    # CPU-friendly by default; change compute_type if you have GPU
    return WhisperModel(model_size, compute_type="int8")

def _save_wav(path: str, audio: np.ndarray, samplerate: int):
    """Write mono int16 PCM to WAV."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())

def _record_audio(seconds: int = 6, samplerate: int = 16000) -> tuple[np.ndarray, int]:
    """Record from local/server microphone using sounddevice (blocking)."""
    st.info(f"Grabando {seconds} s… Habla ahora.")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()
    st.success("Grabación completa.")
    return audio.reshape(-1), samplerate

def _transcribe_with_whisper(wav_path: str, model_size: str, lang_code: str | None, vocab_hint: list[str] | None) -> str:
    """Transcribe with faster-whisper, adding a gentle catalog vocabulary bias."""
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
    }
    if lang_code and lang_code != "auto":
        opts["language"] = lang_code

    if vocab_hint:
        vocab = sorted(set(vocab_hint))
        vocab = [w for w in vocab if 2 <= len(w) <= 30][:120]
        if vocab:
            opts["initial_prompt"] = "Lista de la compra, inventario, productos: " + ", ".join(vocab)

    segments, info = model.transcribe(wav_path, **opts)
    text = " ".join(s.text.strip() for s in segments).strip()
    # Light cleanup of occasional trailing "sí/yes."
    text = re.sub(r"\b(yes|sí)\.?$", "", text, flags=re.IGNORECASE).strip()
    return text

# -------------------------
# UI: Sidebar Navigation
# -------------------------

st.sidebar.title("🎤 Voice Inventory MVP")
mode = st.sidebar.radio("Modo", ["Products DB", "Voice Order"], index=0)

# -------------------------
# Page 1: Products DB (unchanged)
# -------------------------

if mode == "Products DB":
    st.title("📦 Products Database")
    st.caption("Crea tu catálogo con proveedor. Estos campos se usan para el pedido por voz.")

    col_form, col_table = st.columns([1, 2], gap="large")

    with col_form:
        st.subheader("Añadir producto")
        with st.form("add_product"):
            name = st.text_input("Nombre del producto *")
            category = st.text_input("Categoría")
            unit = st.selectbox("Unidad base", ["unidad", "botella", "kg", "caja", "lata", "pack"], index=0)
            quantity = st.number_input("Cantidad (stock opcional)", min_value=0.0, step=1.0, value=0.0)
            provider_name = st.text_input("Proveedor - nombre")
            provider_email = st.text_input("Proveedor - email")
            provider_phone = st.text_input("Proveedor - teléfono")
            provider_address = st.text_input("Proveedor - dirección")
            created_at = st.date_input("Fecha", value=date.today())
            submitted = st.form_submit_button("➕ Guardar producto")
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

                    with get_session() as s:
                        s.add(p)
                        s.commit()
                        st.success(f"Producto '{p.name}' guardado (normalizado).")

    with col_table:
        st.subheader("Catálogo actual")
        with get_session() as s:
            products = s.exec(select(Product).order_by(Product.id.desc())).all()
        if not products:
            st.info("Aún no hay productos. Añade el primero con el formulario.")
        else:
            import pandas as pd
            df = pd.DataFrame([p.dict() for p in products])
            st.dataframe(df, use_container_width=True)

# -------------------------
# Page 2: Voice Order (NEW mic section; Parse & Review unchanged)
# -------------------------

if mode == "Voice Order":
    
        # ----------------------- Page & Controls -----------------------
    st.set_page_config(page_title="Voice → Order List", page_icon="🛒", layout="centered")
    st.title("🛒 Voice → Order List (Start / Stop)")
    st.caption("Click **Start**, speak your order, then click **Stop**. We'll show only product + quantity.")

    # Language control (forces ASR to this language to avoid mis-detection)
    language = st.selectbox("Recognition language", ["en", "es", "de", "fr", "it", "pt"], index=0, help="Force Whisper to transcribe in this language.")

    # Recorder UI (component provides Start/Stop buttons already)
    audio = audiorecorder("Start", "Stop")  # returns pydub.AudioSegment after stopping

    status = st.empty()
    transcript_area = st.empty()
    order_table_container = st.empty()
    download_placeholder = st.empty()

    # ----------------------- Transcription -----------------------

    def transcribe_with_openai(wav_bytes: bytes, lang: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        try:
            from openai import OpenAI
            client = OpenAI()
            with io.BytesIO(wav_bytes) as f:
                f.name = "audio.wav"
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language=lang,
                    # temperature=0 helps keep outputs stable
                    temperature=0,
                )
            return (result.text or "").strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI transcription failed: {e}")


    def transcribe_with_faster_whisper(wav_bytes: bytes, lang: str) -> str:
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel("small", device="auto", compute_type="auto")  # "small" for better accuracy than tiny
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_bytes)
                tmp_path = tmp.name
            segments, info = model.transcribe(tmp_path, beam_size=1, language=lang)
            text = " ".join(s.text.strip() for s in segments).strip()
            return text
        except Exception as e:
            raise RuntimeError(f"faster-whisper failed: {e}")


    def transcribe_audiosegment(audio_segment, lang: str) -> str:
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_bytes = wav_io.getvalue()
        try:
            return transcribe_with_openai(wav_bytes, lang)
        except Exception:
            return transcribe_with_faster_whisper(wav_bytes, lang)

    # ----------------------- Order Parsing -----------------------

    NUMBER_WORDS = {
        "zero": 0, "one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
        "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
        "eighteen": 18, "nineteen": 19, "twenty": 20
    }

    PLURAL_EXCEPTIONS = {
        "tomatoes": "tomato",
        "potatoes": "potato",
    }

    UNIT_WORDS = {"bottle", "bottles", "kilo", "kilos", "kg", "gram", "grams", "g", "pack", "packs"}
    STOP_WORDS = {"of", "and", "please", "i", "would", "like", "to", "order", "hello", "hi", "hey"}


    def singularize(word: str) -> str:
        if word in PLURAL_EXCEPTIONS:
            return PLURAL_EXCEPTIONS[word]
        if word.endswith("ies") and len(word) > 3:
            return word[:-3] + "y"
        if word.endswith("oes") and len(word) > 3:
            return word[:-2]  # e.g., heroes->heroe (rare); handled tomatoes above
        if word.endswith("es") and len(word) > 2:
            return word[:-2]
        if word.endswith("s") and len(word) > 1:
            return word[:-1]
        return word


    def normalize_item(tokens: List[str]) -> str:
        # remove unit words & stop words, singularize last token
        core = [t for t in tokens if t not in UNIT_WORDS and t not in STOP_WORDS]
        if not core:
            return ""
        # If multiple tokens remain, keep last as head noun, prepend modifiers
        head = singularize(core[-1])
        mods = core[:-1]
        item = " ".join(mods + [head]).strip()
        return item


    def parse_quantity(tok: str) -> Tuple[bool, int]:
        if tok.isdigit():
            return True, int(tok)
        return (tok in NUMBER_WORDS), NUMBER_WORDS.get(tok, 0)


    def split_phrases(text: str) -> List[str]:
        # Split on commas and ' and ' while keeping meaningful chunks
        text = re.sub(r"\s+and\s+", ", ", text)
        parts = [p.strip() for p in text.split(",") if p.strip()]
        return parts


    def extract_orders(transcript: str) -> List[Dict[str, str]]:
        t = transcript.lower().strip()
        # remove politeness prefix
        t = re.sub(r"^(hello|hi|hey)[^a-z]*", "", t)
        phrases = split_phrases(t)

        results: List[Dict[str, str]] = []
        for ph in phrases:
            tokens = re.findall(r"[a-zA-Z]+|\d+", ph)
            if not tokens:
                continue
            qty = None
            # find first quantity token
            for i, tok in enumerate(tokens):
                is_q, num = parse_quantity(tok)
                if is_q:
                    qty = num
                    # item is everything after the quantity
                    tail = tokens[i + 1 :]
                    item = normalize_item(tail)
                    if item:
                        results.append({"product": item, "quantity": qty})
                    break
            else:
                # no explicit quantity found; default to 1 and use tokens as item
                item = normalize_item(tokens)
                if item:
                    results.append({"product": item, "quantity": 1})
        return results


    # ----------------------- Main flow -----------------------
    if len(audio) > 0:  # Only true after Stop is pressed
        status.info("Transcribing...")

        # Audio preview
        with st.expander("Preview recording", expanded=False):
            wav_preview = io.BytesIO()
            audio.export(wav_preview, format="wav")
            st.audio(wav_preview.getvalue(), format="audio/wav")

        try:
            text = transcribe_audiosegment(audio, language)
            transcript_area.text_area("Transcript", value=text, height=140)
            status.success("Transcribed")

            orders = extract_orders(text)
            if orders:
                order_table_container.table(orders)
                # CSV download
                csv_io = io.StringIO()
                writer = csv.DictWriter(csv_io, fieldnames=["product", "quantity"])
                writer.writeheader()
                writer.writerows(orders)
                download_placeholder.download_button(
                    "Download order CSV", data=csv_io.getvalue(), file_name="order.csv", mime="text/csv"
                )
            else:
                order_table_container.info("No products recognized. Try speaking clearly, e.g., 'two bottles of gin, two kilos of lemon'.")
        except Exception as e:
            status.error(f"Error: {e}")
    else:
        status.info("Click **Start**, then **Stop** to capture your order.")


    # -------------------------
    # Parse & review (unchanged)
    # -------------------------
    st.subheader("🧩 Parseo y revisión")
    default_text = st.session_state.get("transcript", "")
    raw_text = st.text_area(
        "Texto de pedido (puedes editar la transcripción)",
        height=120,
        value=default_text,
        placeholder="3 limones next gin 2 next olivas 1 next 1 aperol",
    )

    if st.button("Parsear pedido"):
        if not raw_text.strip():
            st.error("Introduce o genera una transcripción.")
        else:
            items = tokenize_items(raw_text)
            parsed_rows = []
            for frag in items:
                parsed = parse_item(frag)
                if not parsed:
                    parsed_rows.append({
                        "spoken_name": frag,
                        "matched_name": None,
                        "confidence": 0.0,
                        "quantity": None,
                        "unit": None,
                        "status": "No interpretado"
                    })
                    continue

                name, qty, unit = parsed

                # ⬇️ If you implemented the improved fuzzy with catalog_index, use this:
                # match_name, score = fuzzy_match(name, catalog_index)

                # ⬇️ Otherwise keep your old catalog_names version:
                match_name, score = fuzzy_match(name, catalog_names)

                matched_product: Optional[Product] = None
                if match_name is not None:
                    for p in products:
                        # be tolerant with case/accents
                        if unidecode(p.name.lower()) == unidecode(str(match_name).lower()):
                            matched_product = p
                            break

                parsed_rows.append({
                    "spoken_name": name,
                    "matched_name": matched_product.name if matched_product else None,
                    "confidence": round(score, 1),
                    "quantity": qty,
                    "unit": unit,
                    "status": "OK" if matched_product else "Revisar"
                })

            # -------------------------
            # ✅ Aggregate BEFORE showing the editor, with unit fallback:
            #    1) most frequent spoken unit
            #    2) product.unit from DB
            #    3) "unidad"
            # -------------------------
            name_to_product = {p.name: p for p in products}
            aggregated_rows = aggregate_parsed_rows(parsed_rows, name_to_product)  # <- pass mapping

            st.subheader("Resultado del parseo")

            import pandas as pd
            df = pd.DataFrame(aggregated_rows)
            edited = st.data_editor(
                df,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "confidence": st.column_config.NumberColumn("Confianza", help="0-100"),
                    "quantity": st.column_config.NumberColumn("Cantidad"),
                    "unit": st.column_config.TextColumn("Unidad"),
                    "matched_name": st.column_config.TextColumn("Producto (catálogo)")
                },
                hide_index=True,
            )

            # Save Order
            if st.button("💾 Guardar pedido"):
                with get_session() as s:
                    order = Order()
                    s.add(order)
                    s.commit()
                    s.refresh(order)

                    for _, row in edited.iterrows():
                        matched_name = (row.get("matched_name") or '').strip()

                        # ✅ avoid unbound variable
                        prod_obj = None
                        product_unit = ""
                        if matched_name:
                            prod_obj = s.exec(select(Product).where(Product.name == matched_name)).first()
                            if prod_obj and (prod_obj.unit or "").strip():
                                product_unit = prod_obj.unit.strip().lower()

                        unit_val = (row.get("unit") or "").strip().lower()
                        if not unit_val:
                            unit_val = product_unit or "unidad"

                        line = OrderLine(
                            order_id=order.id,
                            product_id=prod_obj.id if prod_obj else None,
                            spoken_name=str(row.get("spoken_name") or "").lower(),
                            matched_name=(matched_name.lower() if matched_name else None),
                            confidence=float(row.get("confidence") or 0.0),
                            quantity=float(row.get("quantity") or 0.0),
                            unit=unit_val,
                        )
                        s.add(line)
                    s.commit()
                st.success("Pedido guardado.")

            # Export CSV (without saving)
            if st.button("⬇️ Exportar CSV (sin guardar)"):
                out = io.StringIO()
                writer = csv.writer(out)
                writer.writerow(["spoken_name", "matched_name", "quantity", "unit", "confidence"])
                for _, row in edited.iterrows():
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

            # -------------------------
            # Group by provider & prepare Emails / WhatsApp
            # -------------------------
            st.subheader("✉️ Emails y 📲 WhatsApp por proveedor")

            grouped = {}
            for _, row in edited.iterrows():
                matched_name = (row.get("matched_name") or '').strip()
                if not matched_name:
                    continue
                prod = name_to_product.get(matched_name)
                if not prod:
                    continue
                prov = prod.provider_name or "(Sin proveedor)"
                grouped.setdefault(prov, {
                    "provider_email": prod.provider_email or "",
                    "provider_phone": prod.provider_phone or "",
                    "provider_address": prod.provider_address or "",
                    "lines": []
                })
                grouped[prov]["lines"].append({
                    "product": matched_name,
                    "quantity": row.get("quantity"),
                    "unit": row.get("unit") or prod.unit or "unidad"  # <- final fallback
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
                    for ln in meta["lines"]:
                        qty = ln["quantity"] if ln["quantity"] is not None else ""
                        unit = ln["unit"]
                        body_lines.append(f"- {ln['product']}: {qty} {unit}".strip())
                    # ✅ keep line breaks
                    body_text = "\n".join(body_lines)

                    subject = f"Pedido — {datetime.now().strftime('%Y-%m-%d')}"

                    # mailto link
                    if meta["provider_email"]:
                        import urllib.parse as up
                        mailto = f"mailto:{up.quote(meta['provider_email'])}?subject={up.quote(subject)}&body={up.quote(body_text)}"
                        st.markdown(f"[📧 Abrir email]({mailto})  ")
                    else:
                        st.caption("(Sin email del proveedor)")

                    # WhatsApp link
                    phone_norm = normalize_phone(meta["provider_phone"])
                    if phone_norm:
                        import urllib.parse as up
                        wa = f"https://wa.me/{phone_norm.replace('+','')}?text={up.quote(body_text)}"
                        st.markdown(f"[📲 Abrir WhatsApp]({wa})  ")
                    else:
                        st.caption("(Sin teléfono del proveedor)")

                    # Download TXT
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
### 🧩 Cómo hablar para este MVP
- Di **producto + número** o **número + producto**.
- Usa la palabra **"next"** para pasar al siguiente artículo (también vale "siguiente" o ",").
- Ejemplo: `3 limones next gin 2 next olivas 1 next 1 aperol`
""")

# -------------------------
# Advanced notes / troubleshooting
# -------------------------
with st.expander("Ajustes avanzados y consejos"):
    st.markdown(
        """
        - Si el audio sale saturado, baja el volumen del micro o aumenta la distancia.
        - Cambia el **modelo Whisper**: `tiny` = rápido, `small` = mejor precisión.
        - Idioma: usa **auto** si mezclas español/inglés/griego; fija `es` o `el` para mejorar precisión.
        - En servidores sin GPU, `int8` es suficiente; en GPU puedes cambiar a `float16`.
        - Para producción, migra la base de datos a Postgres y añade autenticación.
        """
    )
