from __future__ import annotations

"""
voice_and_orders_utils.py

Goal
----
Utilities for turning messy, multilingual (EL/ES/EN) order text (typed or ASR)
into clean item fragments, then parsing each fragment into:

    (product_name, quantity, unit)

This module intentionally does NOT decide *which catalog product* to pick.
Catalog matching + ambiguity handling ("Elegir") should live in the UI layer
(e.g., new_order_tab.py) so you can swap matching strategies without touching
speech/text parsing.

Design principles
-----------------
1) Keep parsing fast and deterministic (no heavy NLP).
2) Be tolerant to ASR variability (missing accents, weird punctuation).
3) Remove intent/polite prefixes via `strip_prefix_phrases` (no giant regex).
4) Support EL/ES/EN number words -> digits.
5) Never invent products. If we can't parse, return None and let caller handle.
"""

import io
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st
from domain.models import Product

from core.normalization import normalize_text, match_keys
from features.utils.asr_google import asr_google
from features.utils.prefix_stripper import strip_prefix_phrases
from core.normalization import normalize_text
# Optional dependency: unidecode (not required, but sometimes useful in admin scripts)
try:
    from unidecode import unidecode  # type: ignore
except Exception:  # pragma: no cover
    def unidecode(s: str) -> str:  # type: ignore
        return s


# =============================================================================
# Generic helpers
# =============================================================================

def safe_str(x: Any, default: str = "") -> str:
    """Safe conversion to string; treats NaN/NA as empty."""
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    return str(x)


def str_or_default(x: Any, default: str = "") -> str:
    """Strip string; treat NaN/NA/None as default."""
    s = safe_str(x, default=default).strip()
    return s if s else default


def normalize_key(x: str) -> str:
    """Lowercase + collapse whitespace; used for canonicalizing units."""
    return re.sub(r"\s+", " ", (x or "").strip().lower())


def num_or_default(x: Any, default: float = 0.0) -> float:
    """Safe float conversion; treats NA as default."""
    try:
        if x is pd.NA or pd.isna(x):
            return default
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return default


def distinct_units(products: List[Product]) -> List[str]:
    """Return distinct units from catalog; provides a small default set."""
    units = sorted({normalize_key(getattr(p, "unit", "") or "") for p in products if getattr(p, "unit", None)})
    units = [u for u in units if u]
    return units or ["unidad", "kg", "g", "l", "caja", "botella"]


def coalesce_unit(x: Any, prod_obj: Optional[Product] = None, default: str = "unidad") -> str:
    """Choose unit from x else from product.unit else default."""
    s = str_or_default(x, "")
    if s:
        return s.lower()
    if prod_obj is not None:
        p = str_or_default(getattr(prod_obj, "unit", None), "")
        if p:
            return p.lower()
    return default


# =============================================================================
# Constants: greetings, separators, number words, units, ASR aliases
# =============================================================================

GREETINGS: Set[str] = {
    # ES/EN
    "hola", "hello", "hi", "hey", "buenosdias", "buenastardes", "buenasnoches",
    # EL (accented and not)
    "γεια", "γειά", "γεια σου", "καλημερα", "καλημέρα", "καλησπερα", "καλησπέρα",
    "καληνυχτα", "καληνύχτα",
}

# Common "next item" separators. We split on punctuation and these phrases.
NEXT_SEPARATORS: List[str] = [
    # punctuation / formatting
    ",", ";", "|", "/", " - ", " – ", " — ",

    # EN
    " and then ", " after that ", " followed by ", " next ", " then ",
    " also ", " plus ", " another ", " additionally ",

    # ES
    " y luego ", " y después ", " y despues ", " y también ", " y tambien ",
    " y más ", " y mas ", " siguiente ", " siguientes ", " luego ",
    " después ", " despues ", " también ", " tambien ", " además ", " ademas ",
    " más ", " mas ",

    # EL
    " και μετά ", " και μετα ", " στη συνέχεια ", " στη συνεχεια ",
    " μετά από ", " μετα απο ", " ύστερα ", " υστερα ",
    " κατόπιν ", " κατοποιν ",
    " επίσης ", " επισης ", " και επίσης ", " και επισης ",
    " ακόμα ", " ακομα ",

    # IMPORTANT: keep simple "και" split
    " και ",

    # ASR noise
    " ... ", " … ",
]

# Number words (EN/ES/EL) -> numeric values.
# Keep both accented and unaccented keys because ASR often drops accents.
NUM_WORDS: Dict[str, float] = {
    # EN
    "zero": 0, "oh": 0, "nil": 0,
    "one": 1, "a": 1, "an": 1, "single": 1,
    "two": 2, "couple": 2,
    "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000,
    "half": 0.5, "quarter": 0.25,

    # ES
    "cero": 0,
    "un": 1, "uno": 1, "una": 1,
    "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
    "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
    "dieciseis": 16, "dieciséis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19,
    "veinte": 20,
    "veintiuno": 21, "veintiun": 21, "veintiún": 21, "veintiuna": 21,
    "veintidos": 22, "veintidós": 22,
    "veintitres": 23, "veintitrés": 23,
    "veinticuatro": 24, "veinticinco": 25,
    "veintiseis": 26, "veintiséis": 26,
    "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
    "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90,
    "cien": 100, "ciento": 100, "mil": 1000,
    "medio": 0.5, "media": 0.5,
    "cuarto": 0.25, "cuarta": 0.25,

    # EL
    "μηδεν": 0, "μηδέν": 0,
    "ενα": 1, "ένα": 1, "μια": 1, "μία": 1,
    "δυο": 2, "δύο": 2, "τρια": 3, "τρία": 3, "τεσσερα": 4, "τέσσερα": 4,
    "πεντε": 5, "πέντε": 5, "εξι": 6, "έξι": 6, "επτα": 7, "επτά": 7,
    "οκτω": 8, "οκτώ": 8, "εννεα": 9, "εννέα": 9, "δεκα": 10, "δέκα": 10,
    "εντεκα": 11, "έντεκα": 11, "δωδεκα": 12, "δώδεκα": 12,
    "δεκατρια": 13, "δεκατρία": 13, "δεκατεσσερα": 14, "δεκατέσσερα": 14,
    "δεκαπεντε": 15, "δεκαπέντε": 15, "δεκαεξι": 16, "δεκαέξι": 16,
    "δεκαεπτα": 17, "δεκαεπτά": 17, "δεκαοκτω": 18, "δεκαοκτώ": 18,
    "δεκαεννεα": 19, "δεκαεννέα": 19,
    "εικοσι": 20, "είκοσι": 20,
    "τριαντα": 30, "τριάντα": 30,
    "σαραντα": 40, "σαράντα": 40,
    "πενηντα": 50, "πενήντα": 50,
    "εξηντα": 60, "εξήντα": 60,
    "εβδομηντα": 70, "εβδομήντα": 70,
    "ογδοντα": 80, "ογδόντα": 80,
    "ενενηντα": 90, "ενενήντα": 90,
    "εκατο": 100, "εκατό": 100,
    "χιλια": 1000, "χίλια": 1000, "χιλιοι": 1000, "χίλιοι": 1000, "χιλιες": 1000, "χίλιες": 1000,
    "μισο": 0.5, "μισό": 0.5, "μιση": 0.5, "μισή": 0.5,
    "τεταρτο": 0.25, "τέταρτο": 0.25,
}

# Units / unit-like tokens we recognize as the first token after quantity.
UNITS: Set[str] = {
    # generic piece
    "u", "ud", "uds", "unit", "units", "piece", "pieces",
    "unidad", "unidades", "pieza", "piezas",
    # Greek
    "τεμ", "τεμ.", "τεμαχιο", "τεμαχια", "τεμάχιο", "τεμάχια", "τμχ", "τμχ.",

    # box/pack
    "box", "boxes", "caja", "cajas", "pack", "packs", "paquete", "paquetes",
    # Greek
    "πακετο", "πακετα", "πακέτο", "πακέτα",
    "κιβ", "κιβ.", "κιβωτιο", "κιβωτια", "κιβώτιο", "κιβώτια",

    # bottles/cans
    "bottle", "bottles", "can", "cans",
    "botella", "botellas", "lata", "latas",
    "μπουκαλι", "μπουκαλια", "μπουκάλι", "μπουκάλια",
    "κουτι", "κουτια", "κουτί", "κουτιά",

    # weight
    "kg", "kgs", "kilo", "kilos", "kilogram", "kilograms",
    "g", "gr", "gram", "grams", "gramo", "gramos",
    "κιλο", "κιλα", "κιλό", "κιλά", "κιλά.",
    "γρ", "γρ.", "γραμμαριο", "γραμμαρια", "γραμ", "γραμ.", "γραμμάριο", "γραμμάρια",

    # volume
    "l", "lt", "liter", "liters", "litro", "litros",
    "λιτρο", "λιτρα", "λίτρο", "λίτρα",
    "ml", "milliliter", "milliliters",
    "χιλιοστολιτρο", "χιλιοστολιτρα", "χιλιοστόλιτρο", "χιλιοστόλιτρα",

    # catering-ish
    "slice", "slices", "portion", "portions",
    "racion", "raciones", "porcion", "porciones",
    "μεριδα", "μεριδες", "μερίδα", "μερίδες",

    # misc (some ASR noise tokens)
    "par", "pares", "set", "sets",
}

# Small ASR correction map for common mis-hearings (keep tiny & safe).
ASR_ALIASES: Dict[str, str] = {
    # lime examples (extend as you discover)
    "αλαϊμ": "λαιμ",
    "αλαιμ": "λαιμ",
    "αλάιμ": "λαιμ",
    "λαϊμ": "λαιμ",
    "lime": "λαιμ",
    "laim": "λαιμ",
}


# =============================================================================
# Quantity token set for splitting (anchors)
# =============================================================================

def _build_quantity_token_set(num_words: Dict[str, float]) -> Set[str]:
    """All textual tokens that could represent a quantity."""
    return {k.lower() for k in num_words.keys() if isinstance(k, str) and k.strip()}


_QTY_TOKENS: Set[str] = _build_quantity_token_set(NUM_WORDS)


# =============================================================================
# Sentence splitting into item fragments
# =============================================================================

_DECIMAL_TOKEN = "§DEC§"


def _protect_decimals(text: str) -> str:
    """Protect decimals so we don't split '2,5' into '2' and '5'."""
    text = re.sub(r"(\d)\s*,\s*(\d)", rf"\1{_DECIMAL_TOKEN}\2", text)
    text = re.sub(r"(\d)\s*\.\s*(\d)", rf"\1{_DECIMAL_TOKEN}\2", text)
    return text


def _restore_decimals(text: str) -> str:
    """Restore protected decimals (comma by default)."""
    return text.replace(_DECIMAL_TOKEN, ",")


def _build_separator_regex(next_separators: List[str]) -> re.Pattern:
    """
    Build one regex matching punctuation separators + phrase separators.
    This is faster/cleaner than repeated str.replace loops.
    """
    punct_pat = r"[,;|/]+"
    dash_pat = r"(?:\s[-–—]\s)"

    phrases: List[str] = []
    for sep in next_separators:
        s = (sep or "").strip()
        if not s:
            continue
        # ignore punct separators already covered
        if len(s) <= 2 and any(ch in s for ch in [",", ";", "|", "/", "-"]):
            continue
        phrases.append(re.escape(s))

    if phrases:
        phrase_pat = r"(?:\s+(?:" + "|".join(p.replace(r"\ ", r"\s+") for p in phrases) + r")\s+)"
    else:
        phrase_pat = r"(?!x)x"

    return re.compile(rf"(?:{punct_pat}|{dash_pat}|{phrase_pat})", flags=re.IGNORECASE)


_HALF_WORDS: Set[str] = {
    # EN
    "half", "a half",
    # ES
    "medio", "media",
    # EL
    "μισο", "μισό", "μιση", "μισή",
}


def _looks_like_half_expression(fragment: str) -> bool:
    """
    Detect patterns like:
      - "two and a half"
      - "dos y medio"
      - "δυο και μισο"
    so we don't split "and/y/και" incorrectly.
    """
    f = (fragment or "").strip().lower()
    if not f:
        return False
    if (" and " in f or " y " in f or " και " in f) and any(hw in f for hw in _HALF_WORDS):
        return bool(re.search(r"\d", f))
    return False


def split_sentence_into_item_fragments(text: str) -> List[str]:
    """
    Quantity-aware split using NUM_WORDS as anchors.

    Example:
      "θελω ενα γαλα αμυγδαλου δυο γαλα βρωμης"
    -> ["ενα γαλα αμυγδαλου", "δυο γαλα βρωμης"]
    """
    if not text:
        return []

    t = text.lower().strip()
    t = re.sub(r"\s+(και|y|and)\s+", " | ", t)
    t = re.sub(r"[,\n;]+", " | ", t)

    words = t.split()
    if not words:
        return []

    chunks: List[List[str]] = []
    current: List[str] = []

    for w in words:
        is_qty = w.isdigit() or w in _QTY_TOKENS
        if is_qty and current:
            chunks.append(current)
            current = [w]
        else:
            current.append(w)

    if current:
        chunks.append(current)

    out: List[str] = []
    for c in chunks:
        frag = " ".join(c).strip(" |")
        if frag:
            out.extend([p.strip() for p in frag.split("|") if p.strip()])
    return out


def cleanup_asr_transcript(s: str) -> str:
    """Light cleanup of ASR transcript."""
    s = (s or "").strip()
    if not s:
        return s
    # "0, 3 xxx" -> "3 xxx"
    s = re.sub(r"^\s*0\s*[,.:]\s*(\d+)\b", r"\1", s)
    # collapse repeated punctuation/spaces
    s = re.sub(r"[·•]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def apply_asr_aliases(s: str) -> str:
    """Replace known mis-hearings (kept intentionally small)."""
    t = normalize_text(s)
    for k, v in ASR_ALIASES.items():
        t = re.sub(rf"\b{re.escape(normalize_text(k))}\b", v, t, flags=re.IGNORECASE)
    return t



def _find_alias_spans(tokens: List[str], alias_phrases: Set[str], max_len: int = 4) -> List[Tuple[int,int,str]]:
    """
    Returns spans (start_idx, end_idx_exclusive, matched_phrase) for alias phrase hits.
    Longest-first, non-overlapping.
    """
    hits: List[Tuple[int,int,str]] = []
    n = len(tokens)

    # Build phrase candidates up to max_len
    for i in range(n):
        for L in range(max_len, 0, -1):
            j = i + L
            if j > n:
                continue
            phrase = " ".join(tokens[i:j])
            if phrase in alias_phrases:
                hits.append((i, j, phrase))
                break  # longest at this i

    # Now keep non-overlapping, prefer longer spans
    hits.sort(key=lambda x: (-(x[1]-x[0]), x[0]))
    chosen: List[Tuple[int,int,str]] = []
    occupied = [False]*n

    for i,j,ph in hits:
        if any(occupied[k] for k in range(i, j)):
            continue
        for k in range(i, j):
            occupied[k] = True
        chosen.append((i,j,ph))

    chosen.sort(key=lambda x: x[0])
    return chosen

def split_fragment_by_catalog_aliases(fragment: str, alias_to_products: Dict[str, List[str]]) -> List[str]:
    """
    If a fragment contains multiple product aliases, split it into multiple fragments.
    Example: "αλευρι σκληρο ανιθα" -> ["αλευρι σκληρο", "ανιθα"]
    """
    f = normalize_text(fragment)
    if not f:
        return []

    # Use alias keys as a phrase set (already normalized in your pipeline)
    alias_phrases = set(alias_to_products.keys())

    tokens = f.split()
    spans = _find_alias_spans(tokens, alias_phrases, max_len=4)

    # If 0 or 1 product hit, keep as-is
    if len(spans) <= 1:
        return [fragment.strip()]

    # Convert spans into fragments: include any "glue" words before the first hit inside that hit
    out: List[str] = []
    for (i, j, _ph) in spans:
        piece = " ".join(tokens[i:j]).strip()
        if piece:
            out.append(piece)

    # Deduplicate while preserving order
    seen = set()
    final = []
    for x in out:
        if x not in seen:
            seen.add(x)
            final.append(x)
    return final

def tokenize_items(raw_text: str) -> List[str]:
    """
    Split a message into item fragments, robust for ES/EN/EL + ASR.

    Output: list[str] fragments like:
      ["2 kg tomates", "1 botella aceite", "γαλα", "2 καφε"]

    Notes:
    - We strip intent/polite prefixes both on the full text and per fragment.
    - We drop obvious greetings / conversational replies.
    """
    text = normalize_text(raw_text)
    if not text:
        return []

    # Strip leading intent once (e.g., "θέλω", "quiero", "i want")
    text = strip_prefix_phrases(text, lang="auto")
    if not text:
        return []

    # Normalize greetings so they don't split into pieces
    text = re.sub(r"\bbuenos\s+d[ií]as\b", "buenosdias", text)
    text = re.sub(r"\bbuenas\s+tardes\b", "buenastardes", text)
    text = re.sub(r"\bbuenas\s+noches\b", "buenasnoches", text)
    text = re.sub(r"\bgood\s+morning\b", "goodmorning", text)
    text = re.sub(r"\bgood\s+afternoon\b", "goodafternoon", text)
    text = re.sub(r"\bgood\s+evening\b", "goodevening", text)
    text = re.sub(r"\bκαλη\s*μερα\b", "καλημερα", text)
    text = re.sub(r"\bκαλη\s*σπερα\b", "καλησπερα", text)
    text = re.sub(r"\bκαλη\s*νυχτα\b", "καληνυχτα", text)

    # Protect decimals like 2,5 or 2.5
    text = _protect_decimals(text)

    # Split on separators (punctuation + phrases)
    sep_re = _build_separator_regex(NEXT_SEPARATORS)
    raw_parts = [p.strip(" \t\n,;|/") for p in sep_re.split(text)]
    raw_parts = [p for p in raw_parts if p]
    raw_parts = [_restore_decimals(p) for p in raw_parts]

    # Strip filler prefix for each fragment too (e.g., "επίσης θέλω ...")
    raw_parts = [strip_prefix_phrases(p, lang="auto") for p in raw_parts]
    raw_parts = [p for p in raw_parts if p]

    STOP_FRAGMENTS: Set[str] = {
        # ES
        "y", "e", "o", "tambien", "también", "ademas", "además",
        "vale", "ok", "okay", "perfecto", "bien", "gracias",
        "si", "sí", "no", "algo mas", "algo más",
        # EN
        "and", "or", "ok", "okay", "thanks", "thank you", "perfect", "great", "yes", "no",
        # EL
        "και", "ή", "οκ", "ενταξει", "εντάξει", "τελεια", "τέλεια",
        "ναι", "οχι", "όχι", "ευχαριστω", "ευχαριστώ",
        "αλλο", "άλλο", "ακομα", "ακόμα",
    }

    QUESTION_STARTERS = (
        # ES
        "quieres ", "quiere ", "deseas ", "necesitas ", "te hace falta ",
        "confirmas", "confirmar", "seguro", "esta bien", "está bien",
        # EN
        "do you want ", "would you like ", "need ", "are you sure ", "confirm ",
        # EL
        "θελεις ", "θέλεις ", "χρειαζεσαι ", "χρειάζεσαι ",
        "να επιβεβαιωσω", "να επιβεβαιώσω", "επιβεβαιωνεις", "επιβεβαιώνεις",
    )

    def _is_question_like(p: str) -> bool:
        p2 = (p or "").strip().lower()
        if not p2:
            return True
        if "?" in p2 or p2.startswith("¿"):
            return True
        if p2.startswith(QUESTION_STARTERS):
            return True
        if p2 in STOP_FRAGMENTS:
            return True
        return False

    def _is_noise_fragment(p: str) -> bool:
        p2 = normalize_text(p)
        if not p2:
            return True
        if p2 in GREETINGS or p2 in STOP_FRAGMENTS:
            return True
        # short fragments are often noise (keep digits-only if you want)
        if len(p2) <= 2 and not re.fullmatch(r"\d{1,2}", p2):
            return True
        # connector-only without digits
        if p2 in {"y", "and", "και"} and not re.search(r"\d", p2):
            return True
        return False

    def _split_if_two_quantities(p: str) -> List[str]:
        """
        If fragment contains 2+ numbers and a connector, it's likely 2 items glued together.
        Split on connector.
        """
        p2 = (p or "").strip()
        if not p2:
            return []
        if len(re.findall(r"\d+(?:[.,]\d+)?", p2)) >= 2 and re.search(r"(?:\sκαι\s|\sy\s|\sand\s)", p2.lower()):
            tmp = re.sub(r"\sκαι\s", " | ", p2, flags=re.IGNORECASE)
            tmp = re.sub(r"\sy\s", " | ", tmp, flags=re.IGNORECASE)
            tmp = re.sub(r"\sand\s", " | ", tmp, flags=re.IGNORECASE)
            return [x.strip() for x in tmp.split("|") if x.strip()]
        return [p2]

    parts: List[str] = []
    for p in raw_parts:
        if _is_question_like(p) or _is_noise_fragment(p):
            continue

        expanded = _split_if_two_quantities(p)
        for item in expanded:
            if not item:
                continue

            # half-expression re-join heuristic
            if parts:
                prev = parts[-1]
                combined = f"{prev} {item}"
                if _looks_like_half_expression(combined):
                    parts[-1] = combined.strip()
                    continue

            parts.append(item)

    # final cleanup
    parts = [re.sub(r"\s+", " ", (p or "").strip()) for p in parts]
    parts = [p for p in parts if p and p not in GREETINGS and p.lower() not in STOP_FRAGMENTS]
    return parts


# =============================================================================
# Parsing a single item fragment -> (name, qty, unit)
# =============================================================================

def _replace_number_words(text: str) -> str:
    """Replace number words with digits, e.g., 'δυο' -> '2'."""
    if not text:
        return text
    choices = "|".join(sorted(NUM_WORDS.keys(), key=len, reverse=True))
    patt = re.compile(rf"\b({choices})\b", flags=re.IGNORECASE)

    def repl(m: re.Match) -> str:
        w = m.group(1).lower()
        return str(NUM_WORDS.get(w, w))

    return patt.sub(repl, text)


def parse_item(fragment: str) -> Optional[Tuple[str, float, Optional[str]]]:
    """
    Parse a single fragment into (name, qty, unit).

    Supported patterns:
      - "3 cajas cerveza"   -> ("cerveza", 3, "cajas")
      - "cerveza 3"         -> ("cerveza", 3, None)
      - "γαλα"              -> ("γαλα", 1, None)

    Notes:
    - We preserve original script (Greek stays Greek).
    - Quantity defaults to 1 if missing or invalid.
    """
    frag = normalize_text(fragment)
    if not frag:
        return None

    # Remove intent/polite prefixes
    frag = strip_prefix_phrases(frag, lang="auto")

    # Replace number words with digits
    frag = _replace_number_words(frag)

    # Drop leading connector remnants
    frag = re.sub(r"^(?:και|κι|κ|επισης|επίσης|ακομα|ακόμα|y|and|also|plus)\s+", "", frag).strip()

    # If fragment is just greeting, ignore
    only_letters = re.sub(r"[^\w\s]", " ", frag, flags=re.UNICODE).strip()
    only_letters = re.sub(r"[\d_]+", " ", only_letters).strip()
    only_letters = re.sub(r"\s+", " ", only_letters)
    if only_letters in GREETINGS:
        return None

    # qty-first: "3 kg tomates"
    m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s+(.*)$", frag)
    if m:
        qty_s = m.group(1)
        rest = m.group(2).strip()

        tokens = rest.split()
        unit = None
        name_part = rest

        if tokens and tokens[0] in UNITS:
            unit = tokens[0]
            name_part = " ".join(tokens[1:]).strip()
            # remove ES prepositions/articles
            name_part = re.sub(r"^(?:de|del|la|el|los|las)\s+", "", name_part).strip()

        name = normalize_text(name_part)
        name = re.sub(r"[^\w\s]", " ", name, flags=re.UNICODE).strip()
        name = re.sub(r"\s+", " ", name)

        try:
            qty = float(qty_s.replace(",", "."))
        except Exception:
            qty = 1.0
        if qty <= 0:
            qty = 1.0

        if len(name) < 2 or name in GREETINGS:
            return None
        return name, qty, unit

    # qty-last: "tomates 3"
    m2 = re.match(r"^(.*\D)\s+(\d+(?:[.,]\d+)?)\s*$", frag)
    if m2:
        name_part = m2.group(1).strip()
        qty_s = m2.group(2)

        name_part = re.sub(r"^(?:de|del|la|el|los|las)\s+", "", name_part).strip()

        name = normalize_text(name_part)
        name = re.sub(r"[^\w\s]", " ", name, flags=re.UNICODE).strip()
        name = re.sub(r"\s+", " ", name)

        try:
            qty = float(qty_s.replace(",", "."))
        except Exception:
            qty = 1.0
        if qty <= 0:
            qty = 1.0

        if len(name) < 2 or name in GREETINGS:
            return None
        return name, qty, None

    # name-only fallback
    name_only = re.sub(r"[^\w\s]", " ", frag, flags=re.UNICODE).strip()
    name_only = re.sub(r"\s+", " ", name_only)
    if len(name_only) >= 2 and name_only not in GREETINGS:
        return normalize_text(name_only), 1.0, None

    return None


# =============================================================================
# Simple morphology helpers (used by UI matcher)
# =============================================================================

def singularize_el(text: str) -> str:
    """
    Very light EL singular heuristic (kept intentionally small).
    The robust solution is aliases + match_keys() sigma variants.
    """
    t = normalize_text(text)
    if not t:
        return t
    toks = t.split()
    out: List[str] = []
    for w in toks:
        if len(w) > 4 and w.endswith("ια"):
            out.append(w[:-1])  # "λεμονια" -> "λεμονι"
        else:
            out.append(w)
    return " ".join(out).strip()


def singularize_es(text: str) -> str:
    """Light ES singularization heuristic for plural forms."""
    t = normalize_text(text)
    if not t:
        return t
    toks = t.split()
    out: List[str] = []
    for w in toks:
        if len(w) > 4 and w.endswith("es") and not w.endswith(("ses", "xes", "zes")):
            out.append(w[:-2])
        elif len(w) > 3 and w.endswith("s"):
            out.append(w[:-1])
        else:
            out.append(w)
    return " ".join(out).strip()


# =============================================================================
# Matching helpers (used by UI matcher) — we keep these generic
# =============================================================================

def fuzzy_match(query_norm: str, candidates_norm: List[str], threshold: float = 82.0) -> Tuple[Optional[str], float]:
    """
    Returns (best_match_candidate, score_0_100) using RapidFuzz if available.

    Why threshold=82 (default)
    --------------------------
    Greek ASR + accents/sigma variants can easily drop 5–15 points.
    Ambiguity is resolved in the UI matcher ("Elegir"), so we prefer higher recall here.
    """
    q0 = normalize_text(query_norm)
    if not q0 or not candidates_norm:
        return None, 0.0

    q_keys = match_keys(q0) or [q0]

    # Prefer RapidFuzz
    try:
        from rapidfuzz import fuzz, process  # type: ignore

        cand_list = [normalize_text(c) for c in candidates_norm]

        def combined_scorer(q: str, c: str) -> float:
            # partial_ratio helps when ASR truncates/drops words
            if len(q) < 5:
                return float(fuzz.ratio(q, c))
            return float(max(fuzz.ratio(q, c), fuzz.partial_ratio(q, c)))

        best: Optional[str] = None
        best_score = 0.0

        for q in q_keys:
            q = normalize_text(q)
            if not q:
                continue
            res = process.extractOne(q, cand_list, scorer=combined_scorer)
            if not res:
                continue
            _choice, score, idx = res
            if score > best_score:
                best_score = float(score)
                best = candidates_norm[idx]

        if best_score >= threshold:
            return best, best_score
        return None, best_score

    except Exception:
        # Fallback to difflib
        from difflib import SequenceMatcher

        best: Optional[str] = None
        best_score = 0.0

        for q in q_keys:
            q = normalize_text(q)
            if not q:
                continue
            for c in candidates_norm:
                c2 = normalize_text(c)
                if not c2:
                    continue
                score = SequenceMatcher(None, q, c2).ratio() * 100.0
                if score > best_score:
                    best_score = score
                    best = c

        if best_score >= threshold:
            return best, float(best_score)
        return None, float(best_score)


def suggest_matches(query_norm: str, candidates_norm: List[str], limit: int = 8) -> List[Tuple[str, float]]:
    """
    Return top-N candidate matches: [(candidate, score), ...]
    Uses match_keys() (Greek + transliteration variants) like fuzzy_match.
    """
    q0 = normalize_text(query_norm)
    if not q0 or not candidates_norm:
        return []

    q_keys = match_keys(q0) or [q0]

    try:
        from rapidfuzz import fuzz, process  # type: ignore

        cand_list = [normalize_text(c) for c in candidates_norm]

        def combined_scorer(q: str, c: str) -> float:
            if len(q) < 5:
                return float(fuzz.ratio(q, c))
            return float(max(fuzz.ratio(q, c), fuzz.partial_ratio(q, c)))

        best: Dict[str, float] = {}
        for q in q_keys:
            q = normalize_text(q)
            if not q:
                continue
            res = process.extract(q, cand_list, scorer=combined_scorer, limit=limit)
            for _choice, score, idx in res or []:
                original = candidates_norm[idx]
                best[original] = max(best.get(original, 0.0), float(score))

        out = sorted(best.items(), key=lambda x: x[1], reverse=True)
        return out[:limit]

    except Exception:
        return []


# =============================================================================
# Audio input helpers (Streamlit)
# =============================================================================

def mic_or_upload_audio(label: str, key: str, sample_rate: int = 16000):
    """
    Uses st.audio_input if available (newer Streamlit), otherwise file_uploader fallback.
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
    """Streamlit audio_input returns an object with getvalue(); uploader returns file-like."""
    if audio_obj is None:
        return b""
    if hasattr(audio_obj, "getvalue"):
        return audio_obj.getvalue()
    if hasattr(audio_obj, "read"):
        return audio_obj.read()
    return bytes(audio_obj)


# =============================================================================
# ASR backends
# =============================================================================

def collapse_comma_repeats(text: str, max_consecutive: int = 3) -> str:
    """Collapse consecutive repeated comma-separated phrases from ASR."""
    if not text:
        return text
    parts = [p.strip() for p in text.split(",")]
    out: List[str] = []
    last: Optional[str] = None
    run = 0
    for p in parts:
        if not p:
            continue
        if p.lower() == (last or "").lower():
            run += 1
            if run <= max_consecutive:
                out.append(p)
        else:
            last = p
            run = 1
            out.append(p)
    return ", ".join(out)


def asr_openai_whisper(audio_bytes: bytes, vocab: List[str], language: str = "es") -> str:
    """
    OpenAI Whisper API transcription (cloud).
    Requires:
      - OPENAI_API_KEY in env
      - `openai` python package
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: pip install openai") from e

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    vocab_clean: List[str] = []
    seen: Set[str] = set()
    for v in (vocab or []):
        v2 = (v or "").strip()
        if not v2 or v2 in seen:
            continue
        seen.add(v2)
        vocab_clean.append(v2)

    # Prompt: force a comma-separated "shopping list" format.
    prompt = (
        "Είσαι σύστημα μεταγραφής για παραγγελίες. Στόχος: μόνο λίστα προϊόντων.\n\n"
        "ΚΑΝΟΝΕΣ:\n"
        "1) Γράψε ΜΟΝΟ προϊόντα και (αν ακούγονται) ποσότητες/μονάδες.\n"
        "2) ΜΗΝ γράφεις προτάσεις ή λέξεις intent (θέλω, θα ήθελα, παρακαλώ, επίσης, κλπ).\n"
        "3) Χώρισε προϊόντα με κόμμα \", \".\n"
        "4) Ποσότητες: χρησιμοποίησε ψηφία.\n"
        "5) Μονάδες: kg, g, l, ml, τεμ, κουτί, μπουκάλι, πακέτο.\n"
        "6) Μην μαντεύεις επιπλέον προϊόντα.\n\n"
        "Κατάλογος προϊόντων (προτίμησε αυτά τα ονόματα): "
        + ", ".join(vocab_clean[:150])
    )

    bio = io.BytesIO(audio_bytes)
    bio.name = "audio.wav"

    out = client.audio.transcriptions.create(
        file=bio,
        model="whisper-1",
        language=None if language == "auto" else language,
        prompt=prompt,
        temperature=0,
    )

    txt = (getattr(out, "text", "") or "").strip()
    txt = collapse_comma_repeats(txt, max_consecutive=3)
    return txt


def asr_faster_whisper(audio_bytes: bytes, vocab: List[str], language: str = "es") -> str:
    """
    Local transcription via faster-whisper.
    Requires: pip install faster-whisper
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
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
        "HORECA order. Speaker may be Greek/Spanish/English; product names may be English. "
        "Keep quantities and units. Separate items with commas. "
        "Common products/brands: " + ", ".join((vocab or [])[:120])
    )

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        segments, _info = model.transcribe(
            tmp_path,
            language=None if language == "auto" else language,
            beam_size=8,
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
