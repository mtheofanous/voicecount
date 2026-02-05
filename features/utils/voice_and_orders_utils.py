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
from features.utils.prefix_stripper import clean_fragment
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
"hola", "hello", "hi", "hey",
"buenosdias", "buenastardes", "buenasnoches",
"goodmorning", "goodafternoon", "goodevening",
# EL (accented and not)
"γεια", "γειά", "γεια σου",
"καλημερα", "καλημέρα",
"καλησπερα", "καλησπέρα",
"καληνυχτα", "καληνύχτα",
"καλημερα σας", "καλημέρα σας",
}

# Common "next item" separators. We split on punctuation and these phrases.
NEXT_SEPARATORS: List[str] = [
    # punctuation / formatting (handled separately too, but ok to keep)
    ",", ";", "|", "/",
    " - ", " – ", " — ",
    "...", "…",


    # EN phrases
    " and then ", " after that ", " followed by ",
    " next ", " then ",
    " also ", " plus ", " another ", " additionally ",


    # ES phrases
    " y luego ", " y después ", " y despues ",
    " y también ", " y tambien ",
    " y más ", " y mas ",
    " siguiente ", " siguientes ",
    " luego ", " después ", " despues ",
    " también ", " tambien ",
    " además ", " ademas ",
    " más ", " mas ",


    # EL phrases
    " και μετά ", " και μετα ",
    " στη συνέχεια ", " στη συνεχεια ",
    " μετά από ", " μετα απο ",
    " ύστερα ", " υστερα ",
    " κατόπιν ", " κατοποιν ",
    " επίσης ", " επισης ",
    " και επίσης ", " και επισης ",
    " ακόμα ", " ακομα ",


    # # IMPORTANT: keep simple "και" split (Greek "and")
    # " και ",

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


_DECIMAL_TOKEN = "§DEC§"          # existing (comma-style)
_DECIMAL_DOT_TOKEN = "§DECDOT§"   # new (dot-style)

def _protect_decimals(text: str) -> str:
    """
    Protect decimals so we don't split '2,5' or '2.5'.
    Handles ASR spacing variants: '2 , 5', '2 .5', etc.
    """
    if not text:
        return text

    # Protect comma decimals: 2,5 / 2 ,5 / 2, 5 / 2 , 5
    text = re.sub(r"(\d)\s*,\s*(\d)", rf"\1{_DECIMAL_TOKEN}\2", text)

    # Protect dot decimals: 2.5 / 2 .5 / 2. 5 / 2 . 5
    text = re.sub(r"(\d)\s*\.\s*(\d)", rf"\1{_DECIMAL_DOT_TOKEN}\2", text)

    return text


def _restore_decimals(text: str, prefer_comma: bool = True) -> str:
    """
    Restore protected decimals.

    - §DEC§     -> ','   (always)
    - §DECDOT§ -> ',' if prefer_comma=True, else '.'

    Your qty parsing already does:
        float(qty_s.replace(",", "."))
    so both are safe.
    """
    if not text:
        return text

    text = text.replace(_DECIMAL_TOKEN, ",")

    if prefer_comma:
        text = text.replace(_DECIMAL_DOT_TOKEN, ",")
    else:
        text = text.replace(_DECIMAL_DOT_TOKEN, ".")

    return text


def _build_separator_regex(next_separators: List[str]) -> re.Pattern:
    """
    Build one regex matching punctuation separators + phrase separators.
    Faster/cleaner than repeated replace loops.


    Splits on:
    - punctuation: , ; | /
    - dashes surrounded by spaces: " - ", " – ", " — "
    - phrase separators (from NEXT_SEPARATORS), matched flexibly on whitespace
    """
    # Basic punctuation separators
    punct_pat = r"[,;|/]+"


    # Dash separators with surrounding spaces (avoid splitting inside product codes)
    dash_pat = r"(?:\s[-–—]\s)"


    # Phrase-based separators
    phrases: List[str] = []
    for sep in next_separators:
        s = (sep or "").strip()
        if not s:
            continue


    # ignore single-char punct separators already handled
        if s in {",", ";", "|", "/"}:
            continue


    # escape literal text, then later we relax spaces into \s+
        phrases.append(re.escape(s))


    if phrases:
    # Turn escaped spaces into flexible whitespace
    # Example: "y\ luego" -> "y\s+luego"
        phrase_pat = r"(?:\s*(?:" + "|".join(p.replace(r"\ ", r"\s+") for p in phrases) + r")\s*)"
    else:
    # matches nothing
        phrase_pat = r"(?!x)x"


    combined = rf"(?:{punct_pat}|{dash_pat}|{phrase_pat})"
    return re.compile(combined, flags=re.IGNORECASE)




_HALF_WORDS: Set[str] = {
    # -----------------
    # EN
    # -----------------
    "half", "a half",
    "quarter", "a quarter",

    # -----------------
    # ES
    # -----------------
    "medio", "media",
    "cuarto", "cuarta",

    # -----------------
    # EL (accented + unaccented)
    # -----------------
    "μισο", "μισό",
    "μιση", "μισή",
    "τεταρτο", "τέταρτο",
}



def _looks_like_half_expression(fragment: str) -> bool:
    """
    Detect fractional quantity phrases so we don't split on and/y/και incorrectly.

    Examples that should return True:
      - "two and a half"
      - "2 and a half"
      - "dos y medio"
      - "2 y medio"
      - "δυο και μισο"
      - "2 και μισό"
      - "half kilo", "medio kilo", "μισό κιλό"
      - "one and a quarter", "uno y cuarto", "ενα και τεταρτο"
      - "1/2", "0.5", "2,5"
    """
    f = (fragment or "").strip().lower()
    if not f:
        return False

    # quick reject: if there is no connector and no fraction word, it's not a half-expression
    has_connector = (" and " in f) or (" y " in f) or (" και " in f)
    has_fraction_word = any(hw in f for hw in _HALF_WORDS)  # you already have this set/list
    has_fraction_symbol = bool(re.search(r"\b\d+\s*/\s*\d+\b", f))
    has_decimal = bool(re.search(r"\b\d+[.,]\d+\b", f))

    # If we already see a decimal or explicit fraction, treat as fractional quantity
    if has_decimal or has_fraction_symbol:
        return True

    # If it contains half/quarter word but no connector, still likely a fractional qty (e.g., "medio kilo")
    if has_fraction_word and not has_connector:
        # ensure it's quantity-ish: either a number word or digit exists somewhere
        if re.search(r"\b\d+\b", f):
            return True
        # number word presence: use your NUM_WORDS mapping
        tokens = re.findall(r"[^\W_]+", f, flags=re.UNICODE)
        if any(t in NUM_WORDS for t in tokens):
            return True
        return False

    # If it has connector + fraction word, it's likely "X and a half" etc.
    if has_connector and has_fraction_word:
        # digit present?
        if re.search(r"\b\d+\b", f):
            return True

        # number-word present?
        tokens = re.findall(r"[^\W_]+", f, flags=re.UNICODE)
        if any(t in NUM_WORDS for t in tokens):
            return True

        # special case: "a half" / "un medio" without explicit number (rare but happens)
        # If it's basically "and a half" / "y medio" / "και μισό" -> treat as fractional phrase
        if re.search(r"\b(and|y|και)\b.*\b(" + "|".join(map(re.escape, _HALF_WORDS)) + r")\b", f):
            return True

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

    Guardrails:
    - Don't split very short / generic fragments (single token like "tomate").
    - Don't split if hits overlap or are basically the same "base" token.
    """
    f = normalize_text(fragment)
    if not f:
        return []

    tokens = f.split()

    # ✅ Guardrail 1: generic / short queries should NOT be split
    # (these need catalog expansion later, not early splitting)
    if len(tokens) <= 2:
        return [fragment.strip()]

    alias_phrases = set(alias_to_products.keys())
    spans = _find_alias_spans(tokens, alias_phrases, max_len=4)

    # If 0 or 1 product hit, keep as-is
    if len(spans) <= 1:
        return [fragment.strip()]

    # ✅ Guardrail 2: require non-overlapping spans
    # (overlaps often come from variants like "tomate" vs "tomate rama")
    spans_sorted = sorted(spans, key=lambda x: (x[0], -(x[1] - x[0])))
    non_overlapping = []
    last_end = -1
    for i, j, ph in spans_sorted:
        if i >= last_end:
            non_overlapping.append((i, j, ph))
            last_end = j

    # If after removing overlaps we are left with <=1, don't split
    if len(non_overlapping) <= 1:
        return [fragment.strip()]

    # ✅ Guardrail 3: if all hits share the same first token, don't split
    # e.g., "tomate rama tomate cherry" is arguably multiple, but most of the time
    # "tomate ..." phrases are variants and should be handled by suggestions, not splitting.
    first_tokens = set()
    for i, j, _ph in non_overlapping:
        if i < len(tokens):
            first_tokens.add(tokens[i])

    if len(first_tokens) == 1:
        return [fragment.strip()]

    # Convert spans into fragments
    out: List[str] = []
    for (i, j, _ph) in non_overlapping:
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

    # ✅ Strip global intent/polite once (handles "I want ...", "quiero ...", "θέλω ...")
    text = clean_fragment(text, lang="auto")
    if not text:
        return []

    # Normalize greetings so they don't split into pieces
    text = re.sub(r"\bbuenos\s+d[ií]as\b", "buenosdias", text, flags=re.IGNORECASE)
    text = re.sub(r"\bbuenas\s+tardes\b", "buenastardes", text, flags=re.IGNORECASE)
    text = re.sub(r"\bbuenas\s+noches\b", "buenasnoches", text, flags=re.IGNORECASE)
    text = re.sub(r"\bgood\s+morning\b", "goodmorning", text, flags=re.IGNORECASE)
    text = re.sub(r"\bgood\s+afternoon\b", "goodafternoon", text, flags=re.IGNORECASE)
    text = re.sub(r"\bgood\s+evening\b", "goodevening", text, flags=re.IGNORECASE)
    text = re.sub(r"\bκαλη\s*μερα\b", "καλημερα", text, flags=re.IGNORECASE)
    text = re.sub(r"\bκαλη\s*σπερα\b", "καλησπερα", text, flags=re.IGNORECASE)
    text = re.sub(r"\bκαλη\s*νυχτα\b", "καληνυχτα", text, flags=re.IGNORECASE)

    # Protect decimals like 2,5 or 2.5
    text = _protect_decimals(text)

    # Extra: normalize common separators from speech
    # "and/y/και" between items should behave like separators in many ASR cases
    # We only do this at top-level (later we still handle glued fragments)
    text = re.sub(r"\s+(?:and|y|e|και|κι)\s+", " | ", text, flags=re.IGNORECASE)

    # Split on separators (punctuation + phrases)
    sep_re = _build_separator_regex(NEXT_SEPARATORS)
    raw_parts = [p.strip(" \t\n,;|/") for p in sep_re.split(text)]
    raw_parts = [p for p in raw_parts if p]
    raw_parts = [_restore_decimals(p) for p in raw_parts]

    # ✅ FIX: clean each fragment, not the full text
    raw_parts = [clean_fragment(p, lang="auto") for p in raw_parts]
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
        p2 = normalize_text(p).strip()
        if not p2:
            return True
        if p2 in GREETINGS or p2.lower() in STOP_FRAGMENTS:
            return True

        # Avoid discarding legit short products like "tea", "ice", "ham"
        # Only drop very short fragments if they have no letters (or are pure connector)
        if len(p2) <= 2:
            if re.fullmatch(r"\d{1,2}", p2):
                return False
            if re.search(r"[A-Za-zΑ-Ωα-ω]", p2):
                return False
            return True

        # connector-only without digits
        if p2.lower() in {"y", "and", "και"} and not re.search(r"\d", p2):
            return True
        return False

    def _split_glued_items(p: str) -> List[str]:
        """
        Split if fragment likely contains multiple items.
        Heuristics:
          - 2+ numbers and connector
          - explicit " | " inserted earlier
        """
        p2 = (p or "").strip()
        if not p2:
            return []

        if " | " in p2:
            return [x.strip() for x in p2.split("|") if x.strip()]

        nums = re.findall(r"\d+(?:[.,]\d+)?", p2)
        if len(nums) >= 2 and re.search(r"(?:\sκαι\s|\sy\s|\se\s|\sand\s)", p2.lower()):
            tmp = re.sub(r"\sκαι\s", " | ", p2, flags=re.IGNORECASE)
            tmp = re.sub(r"\sy\s", " | ", tmp, flags=re.IGNORECASE)
            tmp = re.sub(r"\se\s", " | ", tmp, flags=re.IGNORECASE)
            tmp = re.sub(r"\sand\s", " | ", tmp, flags=re.IGNORECASE)
            return [x.strip() for x in tmp.split("|") if x.strip()]

        return [p2]

    parts: List[str] = []
    for p in raw_parts:
        if _is_question_like(p) or _is_noise_fragment(p):
            continue

        for item in _split_glued_items(p):
            if not item:
                continue

            item = clean_fragment(item, lang="auto")
            if not item:
                continue

            # half-expression re-join heuristic
            if parts:
                prev = parts[-1]
                combined = f"{prev} {item}".strip()
                if _looks_like_half_expression(combined):
                    parts[-1] = combined
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
    """
    Replace number words with digits.
    Handles fractional expressions like:
      - "two and a half"      -> "2.5"
      - "dos y medio"        -> "2.5"
      - "δυο και μισο"       -> "2.5"
      - "half kilo"          -> "0.5 kilo"
      - "medio kilo"         -> "0.5 kilo"
      - "μισό κιλό"          -> "0.5 κιλό"
    """
    if not text:
        return text

    t = text.lower()

    # --------------------------------------------------
    # 1) X + and/y/και + half/quarter  -> decimal
    # --------------------------------------------------
    FRACTION_CONNECTORS = r"(?:and|y|και)"
    FRACTION_WORDS = {
        # EN
        "half": 0.5,
        "quarter": 0.25,
        # ES
        "medio": 0.5, "media": 0.5,
        "cuarto": 0.25, "cuarta": 0.25,
        # EL
        "μισο": 0.5, "μισό": 0.5,
        "μιση": 0.5, "μισή": 0.5,
        "τεταρτο": 0.25, "τέταρτο": 0.25,
    }

    def repl_fraction(m: re.Match) -> str:
        base = m.group("base")
        frac_word = m.group("frac")
        base_val = NUM_WORDS.get(base, None)
        frac_val = FRACTION_WORDS.get(frac_word, None)
        if base_val is None or frac_val is None:
            return m.group(0)
        return str(float(base_val) + frac_val)

    frac_words_pat = "|".join(map(re.escape, FRACTION_WORDS.keys()))
    num_words_pat = "|".join(sorted(NUM_WORDS.keys(), key=len, reverse=True))

    t = re.sub(
        rf"\b(?P<base>{num_words_pat})\s+{FRACTION_CONNECTORS}\s+(?:a\s+)?(?P<frac>{frac_words_pat})\b",
        repl_fraction,
        t,
        flags=re.IGNORECASE,
    )

    # --------------------------------------------------
    # 2) standalone half / quarter  -> 0.5 / 0.25
    # --------------------------------------------------
    for w, v in FRACTION_WORDS.items():
        t = re.sub(rf"\b{re.escape(w)}\b", str(v), t, flags=re.IGNORECASE)

    # --------------------------------------------------
    # 3) simple number word -> digit
    # --------------------------------------------------
    choices = "|".join(sorted(NUM_WORDS.keys(), key=len, reverse=True))
    patt = re.compile(rf"\b({choices})\b", flags=re.IGNORECASE)

    def repl_simple(m: re.Match) -> str:
        w = m.group(1).lower()
        return str(NUM_WORDS.get(w, w))

    t = patt.sub(repl_simple, t)

    return t


def parse_item(fragment: str) -> Optional[Tuple[str, float, Optional[str]]]:
    """
    Parse a single fragment into (name, qty, unit).

    Supported patterns:
      - "3 cajas cerveza"   -> ("cerveza", 3, "cajas")
      - "cerveza 3"         -> ("cerveza", 3, None)
      - "γαλα"              -> ("γαλα", 1, None)
      - "i want 3 cucumbers please" -> ("cucumbers", 3, None)

    Notes:
    - We preserve original script (Greek stays Greek).
    - Quantity defaults to 1 if missing or invalid.
    """
    # Normalize early but keep script
    frag = normalize_text(fragment)
    if not frag:
        return None

    # ✅ robust cleanup: strips intent prefixes AND polite suffixes (multi-lang auto)
    # requires the updated prefix_stripper.py that defines clean_fragment()
    frag = clean_fragment(frag, lang="auto")
    if not frag:
        return None

    # Replace number words with digits (your existing function)
    frag = _replace_number_words(frag)
    frag = normalize_text(frag).strip()
    if not frag:
        return None

    # Common "x" multiplier noise: "3x tomates", "3 x tomates"
    frag = re.sub(r"^\s*(\d+(?:[.,]\d+)?)\s*x\s+", r"\1 ", frag, flags=re.IGNORECASE).strip()

    # Drop leading connector remnants repeatedly (kitchen speech often starts with "and", "y", "και")
    # We loop because people say: "and also 3 tomatoes"
    LEAD_JUNK = r"(?:και|κι|κ|επισης|επίσης|ακομα|ακόμα|y|e|and|also|plus|ademas|además|tambien|también|luego|then|porfavor|por\s+favor)\b"
    for _ in range(6):
        new_frag = re.sub(rf"^\s*{LEAD_JUNK}\s+", "", frag, flags=re.IGNORECASE).strip()
        if new_frag == frag:
            break
        frag = new_frag

    if not frag:
        return None

    # If fragment is just greeting / filler, ignore
    only_letters = re.sub(r"[^\w\s]", " ", frag, flags=re.UNICODE).strip()
    only_letters = re.sub(r"[\d_]+", " ", only_letters).strip()
    only_letters = re.sub(r"\s+", " ", only_letters).strip()
    if only_letters in GREETINGS:
        return None

    # Helper to sanitize name consistently
    def _clean_name(s: str) -> str:
        s = normalize_text(s or "")
        s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE).strip()
        s = re.sub(r"\s+", " ", s).strip()
        # remove ES prepositions/articles commonly spoken: "de la", "del", etc.
        s = re.sub(r"^(?:de|del|la|el|los|las)\s+", "", s, flags=re.IGNORECASE).strip()
        return s

    # ---------------------------------------------------------
    # 0) qty+unit glued: "3kg tomates", "2l leche"
    # ---------------------------------------------------------
    m0 = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*([A-Za-zΑ-Ωα-ω]+)\s+(.*)$", frag)
    if m0:
        qty_s, unit_candidate, rest = m0.group(1), m0.group(2), m0.group(3)
        unit_candidate_norm = normalize_text(unit_candidate)
        if unit_candidate_norm in UNITS:
            name = _clean_name(rest)
            try:
                qty = float(qty_s.replace(",", "."))
            except Exception:
                qty = 1.0
            if qty <= 0:
                qty = 1.0
            if len(name) < 2 or name in GREETINGS:
                return None
            return name, qty, unit_candidate_norm

    # ---------------------------------------------------------
    # 1) qty-first: "3 kg tomates" OR "3 tomates"
    # ---------------------------------------------------------
    m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s+(.*)$", frag)
    if m:
        qty_s = m.group(1)
        rest = m.group(2).strip()

        tokens = rest.split()
        unit = None
        name_part = rest

        if tokens:
            t0 = normalize_text(tokens[0])
            if t0 in UNITS:
                unit = t0
                name_part = " ".join(tokens[1:]).strip()

        name = _clean_name(name_part)

        try:
            qty = float(qty_s.replace(",", "."))
        except Exception:
            qty = 1.0
        if qty <= 0:
            qty = 1.0

        if len(name) < 2 or name in GREETINGS:
            return None
        return name, qty, unit

    # ---------------------------------------------------------
    # 2) qty-last: "tomates 3"
    # ---------------------------------------------------------
    m2 = re.match(r"^(.*\D)\s+(\d+(?:[.,]\d+)?)\s*$", frag)
    if m2:
        name_part = m2.group(1).strip()
        qty_s = m2.group(2)

        name = _clean_name(name_part)

        try:
            qty = float(qty_s.replace(",", "."))
        except Exception:
            qty = 1.0
        if qty <= 0:
            qty = 1.0

        if len(name) < 2 or name in GREETINGS:
            return None
        return name, qty, None

    # ---------------------------------------------------------
    # 3) name-only fallback
    # ---------------------------------------------------------
    name_only = _clean_name(frag)
    if len(name_only) >= 2 and name_only not in GREETINGS:
        return name_only, 1.0, None

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
