# features/catalog/utils.py
# features/catalog/utils.py
from __future__ import annotations
from domain.models import Product



import re
import unicodedata
from typing import Dict, Any, Optional

# ----------------------------
# 1) Text normalization
# ----------------------------
def normalize_text(s: Any) -> str:
    """
    Normalize text for DB storage / matching:
    - cast to str
    - strip
    - lowercase
    - remove Greek tonos/diacritics (and any diacritics)
    - collapse internal whitespace
    """
    if s is None:
        return ""
    s = str(s).strip().lower()
    if not s:
        return ""

    # Remove diacritics (Greek tonos etc.)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = unicodedata.normalize("NFKC", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ----------------------------
# 2) CSV header mapping (Greek/Spanish/English → canonical columns)
# ----------------------------
# Canonical columns your app expects:
# name, unit, quantity, price, category, provider_name, provider_email, provider_phone, provider_address, aliases
HEADER_SYNONYMS: Dict[str, str] = {
    # name
    "name": "name",
    "nombre": "name",
    "ονομα": "name",
    "όνομα": "name",
    "προϊόν": "name",
    "προιον": "name",
    "περιγραφη": "name",
    "περιγραφή": "name",

    # unit
    "unit": "unit",
    "unidad": "unit",
    "unidades": "unit",
    "μονάδα": "unit",
    "μοναδα": "unit",
    "μ/δ": "unit",

    # quantity
    "quantity": "quantity",
    "qty": "quantity",
    "cantidad": "quantity",
    "ποσότητα": "quantity",
    "ποσοτητα": "quantity",
    "ποσοτ": "quantity",
    "τεμάχια": "quantity",
    "τεμαχια": "quantity",

    # price
    "price": "price",
    "precio": "price",
    "τιμή": "price",
    "τιμη": "price",
    "κόστος": "price",
    "κοστος": "price",

    # category
    "category": "category",
    "categoria": "category",
    "κατηγορία": "category",
    "κατηγορια": "category",

    # provider fields
    "provider_name": "provider_name",
    "proveedor": "provider_name",
    "supplier": "provider_name",
    "προμηθευτής": "provider_name",
    "προμηθευτης": "provider_name",

    "provider_email": "provider_email",
    "email": "provider_email",
    "ηλ. ταχυδρομείο": "provider_email",
    "ηλ ταχυδρομειο": "provider_email",
    "ηλεκτρονικο ταχυδρομειο": "provider_email",

    "provider_phone": "provider_phone",
    "telefono": "provider_phone",
    "teléfono": "provider_phone",
    "τηλέφωνο": "provider_phone",
    "τηλεφωνο": "provider_phone",

    "provider_address": "provider_address",
    "direccion": "provider_address",
    "dirección": "provider_address",
    "διεύθυνση": "provider_address",
    "διευθυνση": "provider_address",

    # aliases
    "aliases": "aliases",
    "alias": "aliases",
    "sinonimos": "aliases",
    "συνώνυμα": "aliases",
    "συνωνυμα": "aliases",
}

def canonicalize_headers(raw_columns: list[str]) -> list[str]:
    """
    Convert CSV headers to canonical column names.
    Uses normalize_text to handle Greek tonos + case + spaces.
    Unknown headers are left normalized (so you can debug).
    """
    out = []
    for c in raw_columns:
        key = normalize_text(c)
        out.append(HEADER_SYNONYMS.get(key, key))
    return out


# ----------------------------
# 3) Unit normalization (Spanish/English/Greek → canonical unit code)
# ----------------------------
UNIT_SYNONYMS = {
    # canonical: unit
    "unit": "unit",
    "unidad": "unit",
    "unidades": "unit",
    "ud": "unit",
    "uds": "unit",
    "pz": "unit",
    "pieza": "unit",
    "piezas": "unit",

    # Greek "piece"
    "τεμ": "unit",
    "τεμ.": "unit",
    "τεμαχιο": "unit",
    "τεμαχια": "unit",
    "τμχ": "unit",
    "τμχ.": "unit",

    # canonical: kg
    "kg": "kg",
    "kgr": "kg",
    "kilo": "kg",
    "kilos": "kg",
    "κιλ": "kg",
    "κιλο": "kg",
    "κιλο.": "kg",
    "κιλό": "kg",  # just in case (with tonos)
    "κιλό": "kg",
    "κιλά": "kg",
    "κιλά.": "kg",
    "κιλα": "kg",
    

    # canonical: g
    "g": "g",
    "gr": "g",
    "gram": "g",
    "grams": "g",
    "gramo": "g",
    "gramos": "g",

    # Greek grams
    "γραμ": "g",
    "γραμ.": "g",
    "γραμμαριο": "g",
    "γραμμαρια": "g",
    "γραμμάριο": "g",
    "γραμμάρια": "g",
    "γραμμαριo": "g",  # occasional typo

    # canonical: pack
    "pack": "pack",
    "paquete": "pack",
    "paq": "pack",
    "πακ": "pack",
    "πακετο": "pack",
    "πακέτο": "pack",

    # canonical: box
    "box": "box",
    "caja": "box",
    "κιβ": "box",
    "κιβ.": "box",
    "κιβωτιο": "box",
    "κιβωτιο.": "box",
    "κιβώτιο": "box",
}

def normalize_unit(u: Any) -> str:
    """
    Returns canonical unit among: unit, kg, g, pack, box.
    Default is 'unit'.
    """
    key = normalize_text(u)
    if not key:
        return "unit"
    return UNIT_SYNONYMS.get(key, key)  # if unknown, keep normalized value


# ----------------------------
# 4) Row → normalized product payload (ready for DB)
# ----------------------------
def to_float(val: Any, default: float) -> float:
    try:
        if val is None:
            return default
        s = str(val).strip()
        if s == "":
            return default
        # allow European decimal comma
        s = s.replace(",", ".")
        return float(s)
    except Exception:
        return default

def normalize_product_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes a dict with canonical headers and returns a dict with normalized values
    ready for Product(**payload).
    """
    name = normalize_text(row.get("name"))
    category = normalize_text(row.get("category")) or None

    provider_name = normalize_text(row.get("provider_name")) or None
    provider_email = normalize_text(row.get("provider_email")) or None
    provider_phone = normalize_text(row.get("provider_phone")) or None
    provider_address = normalize_text(row.get("provider_address")) or None

    aliases = normalize_text(row.get("aliases")) or None

    unit = normalize_unit(row.get("unit"))
    quantity = to_float(row.get("quantity"), 1.0)
    price = to_float(row.get("price"), 0.0)
    iva = to_float(row.get("iva"), 21.0)

    return {
        "name": name,
        "category": category,
        "unit": unit,
        "quantity": quantity,
        "price": price,
        "iva": iva,
        "provider_name": provider_name,
        "provider_email": provider_email,
        "provider_phone": provider_phone,
        "provider_address": provider_address,
        "aliases": aliases,
    }

def product_from_csv_row(row: dict, venue_id: int) -> Product:
    """
    Convert a CSV row (dict-like) into a Product instance.
    No DB access. No Streamlit. Pure mapping logic.
    """

    def _clean(val):
        return val.strip() if isinstance(val, str) else val

    def _to_float(val, default):
        try:
            if val is None:
                return default
            if isinstance(val, str):
                val = val.strip().replace(",", ".")
            return float(val)
        except Exception:
            return default

    return Product(
        venue_id=venue_id,
        name=_clean(row.get("name", "")),
        category=_clean(row.get("category")),
        unit=_clean(row.get("unit")) or "unit",
        quantity=_to_float(row.get("quantity"), 1.0),
        price=_to_float(row.get("price"), 0.0),
        provider_name=_clean(row.get("provider_name")),
        provider_email=_clean(row.get("provider_email")),
        provider_phone=_clean(row.get("provider_phone")),
        provider_address=_clean(row.get("provider_address")),
    )

def price_with_iva(price: float, iva: float) -> float:
    return round(price * (1 + iva / 100), 2)