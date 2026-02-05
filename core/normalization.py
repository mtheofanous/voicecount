import re
import unicodedata
from typing import Any, Dict, List, Iterable, Optional

try:
    from unidecode import unidecode  # type: ignore
except Exception:  # pragma: no cover
    def unidecode(s: str) -> str:  # type: ignore
        return s


# ============================================================
# Header normalization (CSV import, etc.)
# ============================================================

HEADER_SYNONYMS: Dict[str, str] = {
    # name
    "name": "name",
    "nombre": "name",
    "ονομα": "name",
    "ονομασια": "name",
    "προιον": "name",
    "προϊόν": "name",
  
    "description": "description",
    "descripcion": "description",
    "descripción": "description",
    "περιγραφη": "description",
    "περιγραφή": "description",


    # unit
    "unit": "unit",
    "unidad": "unit",
    "unidades": "unit",
    "μοναδα": "unit",
    "μονάδα": "unit",

    # quantity
    "quantity": "quantity",
    "qty": "quantity",
    "cantidad": "quantity",
    "ποσοτητα": "quantity",
    "ποσότητα": "quantity",

    # legacy compatibility
    "default_qty": "default_qty",

    # price
    "price": "price",
    "precio": "price",
    "τιμη": "price",
    "τιμή": "precio",
    "κοστος": "price",
    "κόστος": "precio",
    
    # iva
    "iva": "iva",
    "iva": "φπα",
    "φπα": "iva",
    "vat": "iva",
    "impuesto": "iva",

    # category
    "category": "category",
    "categoria": "category",
    "κατηγορια": "category",
    "κατηγορία": "categoria",
    "category": "κατηγορια",
    "categoria": "κατηγορια",

    # provider fields
    "provider_name": "provider_name",
    "proveedor": "provider_name",
    "supplier": "provider_name",
    "προμηθευτης": "provider_name",
    "προμηθευτής": "provider_name",

    "provider_email": "provider_email",
    "email": "provider_email",

    "provider_phone": "provider_phone",
    "telefono": "provider_phone",
    "τηλεφωνο": "provider_phone",
    "τηλέφωνο": "provider_phone",

    "provider_address": "provider_address",
    "direccion": "provider_address",
    "διευθυνση": "provider_address",
    "διεύθυνση": "provider_address",

    # aliases
    "aliases": "aliases",
    "alias": "aliases",
    "συνωνυμα": "aliases",
    "συνώνυμα": "aliases",
}



# ============================================================
# Core text normalization
# ============================================================

_ENIE_TOKEN = "__enie__"

def normalize_text(val: Any) -> str:
    """
    Canonical text normalization (single source of truth).

    - lower + strip
    - collapse whitespace
    - remove diacritics (Greek tonos etc.)
    - preserve Spanish ñ (so 'jalapeño' doesn't become 'jalapeno')
    - DO NOT transliterate scripts (no unidecode here)
    """
    if val is None:
        return ""
    s = str(val).strip().lower()
    if not s:
        return ""

    # Preserve ñ explicitly (NFKD would split it into n + combining mark)
    s = s.replace("ñ", _ENIE_TOKEN)

    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = unicodedata.normalize("NFKC", s)

    s = s.replace(_ENIE_TOKEN, "ñ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonicalize_headers(cols: List[str]) -> List[str]:
    out: List[str] = []
    for c in cols:
        key = normalize_text(c)
        out.append(HEADER_SYNONYMS.get(key, key))
    return out


# ============================================================
# Unit normalization
# ============================================================

UNIT_SYNONYMS: Dict[str, str] = {
    # unit / piece
    "unit": "unit",
    "unidad": "unit",
    "ud": "unit",
    "uds": "unit",
    "pieza": "unit",
    "piezas": "unit",
    "τεμ": "unit",
    "τεμ.": "unit",
    "τεμαχιο": "unit",
    "τεμαχια": "unit",
    "τμχ": "unit",
    "τμχ.": "unit",

    # kg
    "kg": "kg",
    "kgs": "kg",
    "kilo": "kg",
    "kilos": "kg",
    "κιλ": "kg",
    "κιλο": "kg",
    "κιλο.": "kg",
    "κιλό": "kg",
    "κιλα": "kg",
    "κιλά": "kg",

    # grams
    "g": "g",
    "gr": "g",
    "gram": "g",
    "grams": "g",
    "gramo": "g",
    "gramos": "g",
    "γραμ": "g",
    "γραμ.": "g",
    "γραμμαριο": "g",
    "γραμμαρια": "g",
    "γραμμάριο": "g",
    "γραμμάρια": "g",
    "γρ": "g",
    "γρ.": "g",

    # pack
    "pack": "pack",
    "packs": "pack",
    "paquete": "pack",
    "paquetes": "pack",
    "paq": "pack",
    "πακ": "pack",
    "πακετο": "pack",
    "πακέτο": "pack",
    "πακετα": "pack",
    "πακέτα": "pack",

    # box
    "box": "box",
    "boxes": "box",
    "caja": "box",
    "cajas": "box",
    "κιβ": "box",
    "κιβ.": "box",
    "κιβωτιο": "box",
    "κιβωτιο.": "box",
    "κιβώτιο": "box",
    "κιβωτια": "box",
    "κιβώτια": "box",
}

def normalize_unit(val: Any) -> str:
    key = normalize_text(val)
    if not key:
        return "unit"
    return UNIT_SYNONYMS.get(key, key)


# ============================================================
# ASR normalization helpers (text cleanup before parsing)
# ============================================================

def cleanup_asr_transcript(text: Any) -> str:
    """
    Clean common ASR artifacts (Whisper/faster-whisper) before tokenization.
    Examples:
      - "0, 3 αλάιμ" -> "3 αλάιμ"
      - collapse weird bullets/punctuation
    """
    s = str(text or "").strip()
    if not s:
        return ""

    # "0, 3 xxx" -> "3 xxx"
    s = re.sub(r"^\s*0\s*[,.:]\s*(\d+)\b", r"\1", s)

    # normalize bullets / middots used by some ASR
    s = re.sub(r"[·•]+", " ", s)

    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================================
# Matching keys (normalization for catalog lookup)
# ============================================================

def match_keys(text: Any) -> List[str]:
    """
    Generate multiple normalized keys for matching.

    - primary: normalize_text (keeps script)
    - secondary: unidecode transliteration (for matching only, not display)
    """
    def _collapse_repeats(s: str) -> str:
        return re.sub(r"(.)\1{2,}", r"\1\1", s)
    
    base = normalize_text(text)
    keys = {base, _collapse_repeats(base)}
    if not base:
        return []
    keys = {base}

    try:
        u = unidecode(base)
        if u:
            keys.add(u)
            keys.add(normalize_text(u))
    except Exception:
        pass

    return [k for k in keys if k]
