from __future__ import annotations

import os
import re
import unicodedata
from typing import List, Dict, Any


# -----------------------------------------------------------------------------
# Normalization helpers
# -----------------------------------------------------------------------------
def _norm(s: str) -> str:
    """
    Lowercase + remove accents/tonos + normalize apostrophes + collapse spaces.
    Keeps only a normalized view for matching/stripping; returned output is normalized.
    """
    s = (s or "").strip().lower()
    if not s:
        return ""

    # normalize unicode
    s = unicodedata.normalize("NFKC", s)

    # normalize common apostrophes/quotes
    s = s.replace("’", "'").replace("`", "'").replace("´", "'")

    # strip accents/tonos
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = unicodedata.normalize("NFC", s)

    # collapse whitespace
    s = " ".join(s.split())
    return s


def _strip_leading_punct(s: str) -> str:
    # keep digits/letters; remove noisy leading punctuation
    return (s or "").lstrip(" \t\r\n,.:;!?-–—()[]{}\"'“”«»")


def _strip_trailing_punct(s: str) -> str:
    return (s or "").rstrip(" \t\r\n,.:;!?-–—()[]{}\"'“”«»")


# -----------------------------------------------------------------------------
# Prefix/suffix phrase dictionaries
# -----------------------------------------------------------------------------
# NOTE: We work in normalized space; you can include accented variants but they will
# normalize to the same string anyway.

PREFIX_PHRASES = {
    # "auto" should be lightweight *connectors/polite* but we will, in code,
    # expand auto to include en/es/el intent phrases too for robustness.
    "auto": [
        # connectors (EN/ES)
        "and", "or", "also", "plus",
        "and a", "and an", "and one",
        "y", "e", "o", "ademas", "tambien",
        "y un", "y una", "y uno",
        "luego", "entonces",

        # Greek connectors
        "και", "κι", "επισης", "ακομα", "λοιπον",

        # polite
        "please", "por favor", "porfa",
        "παρακαλω", "παρακαλώ",
    ],
    "en": [
        # intent
        "i want", "i would like", "i'd like", "id like", "i need",
        "can i have", "could i get", "may i have", "can we have",
        "we need", "we want", "we would like",
        "add", "put", "give me", "bring me", "send me",

        # softeners
        "please", "also", "then",
        "for me", "for us",
    ],
    "es": [
        "quiero", "quisiera", "me gustaria", "me gustaría", "necesito",
        "ponme", "pon", "me pones", "dame", "traeme", "tráeme",
        "anade", "añade", "agrega", "mete", "suma",
        "por favor", "porfa",
        "ademas", "además", "tambien", "también", "luego",
        "para mi", "para nosotros",
    ],
    "el": [
        # intent (Greek)
        "θελω", "θέλω",
        "θελω να", "θέλω να",
        "θα ηθελα", "θα ήθελα",
        "θα θελα", "θα 'θελα",
        "θα θελω", "θα θέλω",
        "μπορω να εχω", "μπορώ να έχω",
        "μπορεις να μου βαλεις", "μπορείς να μου βάλεις",
        "βαλε", "βάλε", "βαλτε", "βάλτε",
        "προσθεσε", "πρόσθεσε",
        "δωσε", "δώσε", "δωστε", "δώστε",
        "φερε", "φέρε", "φερτε", "φέρτε",

        # polite/connectors
        "παρακαλω", "παρακαλώ",
        "επισης", "επίσης",
        "ακομα", "ακόμα",
        "και μετα", "και μετά",
    ],
}

# Suffix fillers often spoken at the end
SUFFIX_PHRASES = {
    "auto": [
        "please", "thanks", "thank you", "thankyou",
        "por favor", "gracias", "muchas gracias",
        "παρακαλω", "παρακαλώ", "ευχαριστω", "ευχαριστώ",
    ],
    "en": [
        "please", "thanks", "thank you", "thankyou",
    ],
    "es": [
        "por favor", "gracias", "muchas gracias",
    ],
    "el": [
        "παρακαλω", "παρακαλώ", "ευχαριστω", "ευχαριστώ",
    ],
}

# Greek number words commonly used after a connector. We remove the connector but keep the number.
GREEK_NUMBER_WORDS = {
    "ενα", "εναν", "ενας", "μια",
    "δυο", "τρια", "τρεια", "τεσσερα", "πεντε",
    "εξι", "επτα", "οκτω", "εννεα", "δεκα",
}


# -----------------------------------------------------------------------------
# Language / phrase selection
# -----------------------------------------------------------------------------
def _phrases_for_lang(bucket: Dict[str, List[str]], lang: str) -> List[str]:
    """
    Return phrases for given lang.
    For lang="auto": include en+es+el + auto to robustly strip intent words regardless of detected language.
    """
    if lang == "auto":
        phrases = []
        phrases += bucket.get("en", [])
        phrases += bucket.get("es", [])
        phrases += bucket.get("el", [])
        phrases += bucket.get("auto", [])
        return phrases

    phrases = []
    phrases += bucket.get(lang, [])
    phrases += bucket.get("auto", [])
    return phrases


def _compile_phrase_list(phrases: List[str]) -> List[str]:
    """
    Normalize, de-duplicate, sort longest-first.
    """
    normed = sorted({ _norm(p) for p in phrases if (p or "").strip() }, key=len, reverse=True)
    return [p for p in normed if p]


# -----------------------------------------------------------------------------
# Core stripping functions
# -----------------------------------------------------------------------------
def strip_prefix_phrases(text: str, lang: str = "auto", max_loops: int = 10) -> str:
    """
    Repeatedly strip known intent/polite/connectors from the beginning.
    Returns a normalized, whitespace-collapsed string suitable for parsing/matching.

    Examples:
      "I want 3 cucumbers" -> "3 cucumbers"
      "y 2 limones" -> "2 limones"
      "και δυο γαλατα" -> "δυο γαλατα"   (keeps the number word)
    """
    raw = (text or "").strip()
    if not raw:
        return ""

    # Work in normalized space to avoid mismatch from punctuation/accents.
    t = _norm(raw)

    # Pre-strip leading punctuation once
    t = _strip_leading_punct(t)
    t = " ".join(t.split()).strip()
    if not t:
        return ""

    phrases = _compile_phrase_list(_phrases_for_lang(PREFIX_PHRASES, lang))

    for _ in range(max_loops):
        t = _strip_leading_punct(t)
        t = " ".join(t.split()).strip()
        if not t:
            return ""

        # SPECIAL: Greek connector + number word -> drop connector only
        # e.g. "και δυο γαλατα" => "δυο γαλατα"
        m = re.match(r"^(και|κι)\s+(\S+)\b", t)
        if m:
            maybe_num = _norm(m.group(2))
            if maybe_num in GREEK_NUMBER_WORDS:
                # Remove only first token (connector)
                parts = t.split()
                t = " ".join(parts[1:]).strip()
                continue

        changed = False
        for p in phrases:
            if t == p:
                t = ""
                changed = True
                break
            if t.startswith(p + " "):
                t = t[len(p):].strip()
                changed = True
                break

        if not changed:
            break

    t = _strip_leading_punct(t)
    t = " ".join(t.split()).strip()
    return t


def strip_suffix_phrases(text: str, lang: str = "auto", max_loops: int = 6) -> str:
    """
    Repeatedly strip common polite/filler phrases from the end.
    Useful for ASR-like phrases: "3 cucumbers please" -> "3 cucumbers"
    """
    raw = (text or "").strip()
    if not raw:
        return ""

    t = _norm(raw)
    t = _strip_trailing_punct(t)
    t = " ".join(t.split()).strip()
    if not t:
        return ""

    suffixes = _compile_phrase_list(_phrases_for_lang(SUFFIX_PHRASES, lang))

    for _ in range(max_loops):
        t = _strip_trailing_punct(t)
        t = " ".join(t.split()).strip()
        if not t:
            return ""

        changed = False
        for sfx in suffixes:
            if t == sfx:
                t = ""
                changed = True
                break
            if t.endswith(" " + sfx):
                t = t[: -len(sfx)].strip()
                changed = True
                break

        if not changed:
            break

    t = _strip_trailing_punct(t)
    t = " ".join(t.split()).strip()
    return t


def clean_fragment(text: str, lang: str = "auto") -> str:
    """
    Convenience: strip prefix + suffix fillers and normalize punctuation/spacing.
    """
    t = strip_prefix_phrases(text, lang=lang)
    t = strip_suffix_phrases(t, lang=lang)
    t = _strip_leading_punct(_strip_trailing_punct(t))
    return " ".join((t or "").split()).strip()


# -----------------------------------------------------------------------------
# Optional: LLM extractor (kept from your original file)
# -----------------------------------------------------------------------------
def llm_extract_items(fragments: List[str], language_hint: str = "auto") -> List[Dict[str, Any]]:
    """
    Returns list of {name, qty, unit} extracted from fragments.
    Uses OpenAI Structured Outputs (JSON Schema).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Missing dependency: pip install openai") from e

    client = OpenAI()

    schema = {
        "name": "order_items",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "qty": {"type": "number"},
                            "unit": {"type": ["string", "null"]},
                        },
                        "required": ["name", "qty", "unit"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        },
    }

    prompt = (
        "You extract order items from user text.\n"
        "Return ONLY structured JSON matching the schema.\n"
        "Rules:\n"
        "- Remove filler/intent words (e.g., 'I want', 'quiero', 'θα ήθελα', 'παρακαλώ').\n"
        "- Keep only product name, quantity, and unit (if present).\n"
        "- If quantity missing, set qty=1.\n"
        "- If unit missing, unit=null.\n"
        "- Do NOT invent products.\n"
        f"- language_hint: {language_hint}\n"
        "Input fragments:\n"
        + "\n".join(f"- {f}" for f in fragments)
    )

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "You are a precise information extraction engine."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_schema", "json_schema": schema},
        temperature=0,
    )

    data = resp.choices[0].message.content
    import json
    obj = json.loads(data)
    return obj.get("items", [])