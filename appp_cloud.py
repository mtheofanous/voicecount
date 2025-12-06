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
from typing import List, Optional, Tuple
import time
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av  # noqa: F401
    HAS_WEBRTC = True
except Exception:
    HAS_WEBRTC = False

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
# App Setup / DB Engine  ✅ LIMPIO
# -------------------------

st.set_page_config(page_title="Voice Inventory MVP", page_icon="🎤", layout="wide")

import os
import streamlit as st
from sqlmodel import SQLModel, Session, create_engine

def _sqlite_path() -> str:
    # Carpeta escribible en Streamlit Cloud (persiste solo mientras el contenedor vive)
    data_dir = os.environ.get("STREAMLIT_DATA_DIR", "/mount/data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "voicecount.db")

from sqlalchemy.engine import URL
import socket

def _first_ipv4(host: str) -> str | None:
    try:
        for fam, _, _, _, sockaddr in socket.getaddrinfo(host, None):
            if fam == socket.AF_INET:  # IPv4 only
                return sockaddr[0]
    except Exception:
        pass
    return None

def build_engine():
    # Prefer structured secrets (no URL-encoding headaches)
    host = st.secrets.get("DB_HOST")
    if host:
        user = st.secrets.get("DB_USER", "postgres")
        pwd  = st.secrets.get("DB_PASSWORD", "")
        port = int(st.secrets.get("DB_PORT", "5432"))
        name = st.secrets.get("DB_NAME", "postgres")

        # If user supplied an explicit IPv4, use it; else resolve one.
        hostaddr = st.secrets.get("DB_HOSTADDR") or _first_ipv4(host)

        query = {"sslmode": "require"}
        if hostaddr:
            query["hostaddr"] = hostaddr  # psycopg3: connect via IPv4, but keep hostname for TLS/SNI

        url = URL.create(
            "postgresql+psycopg",
            username=user,
            password=pwd,
            host=host,   # hostname stays for TLS
            port=port,
            database=name,
            query=query,
        )
        return create_engine(url, pool_pre_ping=True)

    # Fallback: single DATABASE_URL (we'll still enforce sslmode=require)
    raw = st.secrets.get("DATABASE_URL") or os.getenv("DATABASE_URL")
    if not raw:
        raw = f"sqlite:///{_sqlite_path()}"
        return create_engine(raw, connect_args={"check_same_thread": False}, pool_pre_ping=True)

    # normalize + force SSL
    if raw.startswith("postgres://"):
        raw = "postgresql://" + raw[len("postgres://"):]
    if raw.startswith("postgresql://") and "+psycopg" not in raw.split("://", 1)[0]:
        raw = raw.replace("postgresql://", "postgresql+psycopg://", 1)
    if "sslmode=" not in raw:
        raw += ("&" if "?" in raw else "?") + "sslmode=require"
    return create_engine(raw, pool_pre_ping=True)


@st.cache_resource
def get_engine():
    # Cachea el engine para toda la vida de la app
    return build_engine()

ENGINE = get_engine()

# from sqlalchemy import text
# def _mask_url(u: str) -> str:
#     import re
#     return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", u)

# with st.expander("🔧 DB Debug", expanded=False):
#     try:
#         st.write("Driver:", ENGINE.url.get_backend_name())  # debe ser 'postgresql+psycopg'
#         st.write("DB URL:", _mask_url(str(ENGINE.url)))
#         with ENGINE.connect() as conn:
#             st.write("SELECT 1 ->", conn.execute(text("SELECT 1")).fetchone())
#             rows = conn.execute(text("""
#                 SELECT table_name
#                 FROM information_schema.tables
#                 WHERE table_schema='public'
#                 ORDER BY 1
#             """)).fetchall()
#             st.write("Tablas public:", rows)
#     except Exception as e:
#         st.error(f"DB connection error: {e}")

def init_db() -> None:
    # Crea tablas una vez (tras declarar los modelos)
    SQLModel.metadata.create_all(ENGINE)

def get_session() -> Session:
    # Úsalo como:  with get_session() as s: ...
    return Session(ENGINE)

# Llamar una vez tras definir modelos
init_db()
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


@st.cache_resource(show_spinner=False)
def load_whisper(model_size: str = "small"):
    # CPU-friendly by default; change compute_type if you have GPU
    return WhisperModel(model_size, compute_type="int8")


def _save_wav(path: str, audio_i16: np.ndarray, samplerate: int):
    import wave
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(samplerate)
        wf.writeframes(audio_i16.tobytes())
        
# WebRTC recorder → returns temp .wav path (48kHz mono int16)

def record_webrtc_wav(seconds: int = 6, sample_rate: int = 48000, key: str = "voice_mic") -> str | None:
    """
    Graba ~N segundos desde el micro del navegador vía WebRTC, guarda un WAV mono
    temporal a `sample_rate` y devuelve la ruta. Devuelve None si el usuario no inició el mic.
    Requiere: streamlit-webrtc, av, numpy y la helper `_save_wav`.
    """
    import numpy as np
    import time
    import tempfile

    # STUN público de Google (funciona en Cloud)
    rtc_cfg = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    # Enviar solo audio: navegador -> servidor
    ctx = webrtc_streamer(
        key=key,
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=rtc_cfg,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    if not ctx.state.playing:
        st.info("Pulsa **Start** arriba para activar el micrófono y luego vuelve a pulsar **Grabar**.")
        return None

    st.write(f"🎙️ Grabando {seconds} s…")
    t_end = time.time() + seconds
    chunks: list[np.ndarray] = []

    # Ir recogiendo frames mientras dure la grabación
    while time.time() < t_end:
        if ctx.audio_receiver:
            frames = ctx.audio_receiver.get_frames(timeout=0.2)
            for f in frames:  # f es av.AudioFrame
                pcm = f.to_ndarray()  # suele venir (channels, samples) o (samples,)
                # Asegurar forma (samples, channels)
                if pcm.ndim == 2 and pcm.shape[0] < pcm.shape[1]:
                    pcm = pcm.T
                # Pasar a mono
                if pcm.ndim == 2 and pcm.shape[1] > 1:
                    pcm = pcm.mean(axis=1)
                else:
                    pcm = pcm.reshape(-1)
                # Convertir a int16 de forma segura
                if pcm.dtype != np.int16:
                    pcm = np.clip(pcm, -1, 1)
                    pcm = (pcm * 32767.0).astype(np.int16)
                chunks.append(pcm)
        else:
            time.sleep(0.1)

    if not chunks:
        st.warning("No se recibieron frames de audio. ¿Concediste permiso al micrófono?")
        return None

    audio_i16 = np.concatenate(chunks)

    # Guardar WAV temporal (Whisper re-muestrea internamente si hace falta)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        _save_wav(tf.name, audio_i16, sample_rate)  # asegúrate de tener esta helper definida
        wav_path = tf.name

    st.success("✅ Grabación lista")
    return wav_path

#

#

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

def transcribe_file(
    wav_path: str,
    model_size: str = "small",
    lang_code: str = "auto",
    vocab_hint: list[str] | None = None,
) -> str:
    return _transcribe_with_whisper(
        wav_path=wav_path,
        model_size=model_size,
        lang_code=lang_code,
        vocab_hint=vocab_hint,
    )

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
    st.title("🗣️ Voice Order (Mic + Whisper)")
    st.caption(
        "Dicta productos y cantidades. Puedes decir el número antes o después del nombre "
        "Ejemplo: '3 limones, gin 2, olivas 1, 1 aperol'"
    )

    # Load catalog for bias & mapping
    with get_session() as s:
        products = s.exec(select(Product).order_by(Product.name.asc())).all()
    catalog_names = [normalize_text(p.name) for p in products]

    if not products:
        st.warning("Primero crea tu catálogo en 'Products DB'.")
        st.stop()

    # -------------------------
    # Microphone section (REWRITTEN: local sounddevice + offline Whisper)
    # -------------------------
    st.subheader("🎙️ Micrófono")

    # Choose mic mode
    choices = ["Local (servidor)"]
    if HAS_WEBRTC:
        choices.insert(0, "Browser (WebRTC)")
    mic_mode = st.radio(
        "Origen del audio",
        choices,
        horizontal=True,
        help="WebRTC funciona en Streamlit Cloud. 'Local' requiere acceso a un micrófono en el host."
    )


    # Common params
    colA, colB = st.columns([1, 1])
    with colA:
        lang = st.selectbox("Idioma", ["auto", "es", "en", "el"], index=1, help="'auto' detecta automáticamente")
    with colB:
        model_size = st.selectbox("Modelo Whisper", ["tiny", "base", "small"], index=2, help="'small' = mejor precisión")

    vocab_hint = catalog_names  # bias desde tu catálogo

    if mic_mode == "Browser (WebRTC)":
        # --- WebRTC path (Cloud-friendly) ---
        colC, colD = st.columns([1, 1])
        with colC:
            seconds = st.slider("Segundos a grabar", 3, 20, 6)
        with colD:
            st.caption("El navegador suele capturar a 48 kHz (se re-muestrea internamente).")

        if st.button("🎧 Grabar (WebRTC)"):
            wav_path = record_webrtc_wav(seconds=seconds, sample_rate=48000, key="voice_mic")
            if wav_path:
                with st.spinner("Transcribiendo con Whisper…"):
                    transcript = transcribe_file(
                        wav_path,
                        model_size=model_size,  # "small" para mejor precisión; "tiny/base" si quieres más rapidez
                        lang_code=lang,
                        vocab_hint=vocab_hint
                    )
                st.session_state.transcript = transcript
                st.success("Transcripción lista")
                st.text_area("Transcripción", value=st.session_state.transcript, height=120)


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
