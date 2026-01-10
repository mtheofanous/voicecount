from __future__ import annotations
import unicodedata
import os
import re
from typing import List, Optional, Dict, Any

def _norm(s: str) -> str:
    """Lowercase + remove accents/tonos + collapse spaces."""
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")  # remove accents
    s = unicodedata.normalize("NFC", s)
    s = " ".join(s.split())
    return s

# Multi-lingual phrases you don't want at the START of a fragment
PREFIX_PHRASES = {
    "auto": [
        # connectors
        "and", "or", "also", "plus",
        "and a", "and an", "and one",
        "y", "e", "o", "ademas", "tambien",
        "y un", "y una", "y uno",
        # Greek connectors (single + connector+article/number combos)
        "και", "κι", "επισης", "ακομα", "λοιπον",
        # polite
        "please", "por favor", "παρακαλω",
    ],
    "en": [
        "i want", "i would like", "i'd like", "i need",
        "can i have", "could i get", "may i have",
        "add", "put", "give me", "bring me",
        "please", "also", "then",
    ],
    "es": [
        "quiero", "quisiera", "me gustaria", "necesito",
        "ponme", "pon", "me pones", "dame", "traeme", "traeme",
        "anade", "añade", "agrega", "mete", "suma",
        "por favor", "ademas", "tambien", "luego",
    ],
    "el": [
        "θελω", "θελω να", "θελω ενα", "θελω μια",
        "θα ηθελα", "θα θελα", "θα θελω",
        "μπορω να εχω", "μπορείς να μου βάλεις", "μπορεις να μου βαλεις",
        "βαλε", "βαλτε", "βαζε", "προσθεσε", "δωσε", "δωστε", "φερε", "φερτε",
        "παρακαλω", "επισης", "ακομα", "και μετα",
    ],
}

def strip_prefix_phrases(text: str, lang: str = "auto", max_loops: int = 8) -> str:
    """
    Repeatedly strip known intent/polite/connectors from the beginning.
    No regex: just normalized prefix matching.
    """
    raw = (text or "").strip()
    if not raw:
        return ""

    # work on normalized view but remove from the original progressively
    t = raw
    for _ in range(max_loops):
        changed = False

        t_norm = _norm(t)
        if not t_norm:
            return ""

        # SPECIAL: keep quantity when fragment starts with Greek connector + number
        # e.g. 'και δυο γαλατα' -> 'δυο γαλατα' (do NOT drop 'δυο')
        m = re.match(r"^(και|κι)\s+(ενα|εναν|ενας|μια|δυο|τρια|τρεια|τεσσερα)\b", t_norm)
        if m:
            t = t_norm.split(maxsplit=1)[1] if len(t_norm.split()) > 1 else ""
            # continue loop to allow stripping additional polite prefixes after connector removal
            continue

        phrases = []
        if lang in PREFIX_PHRASES:
            phrases += PREFIX_PHRASES[lang]
        phrases += PREFIX_PHRASES["auto"]

        # longest-first prevents stripping "i" before "i would like"
        phrases = sorted(set(_norm(p) for p in phrases if p.strip()), key=len, reverse=True)

        for p in phrases:
            if t_norm == p or t_norm.startswith(p + " "):
                # remove the same number of words from the ORIGINAL string
                # easiest: rebuild from normalized remainder
                remainder = t_norm[len(p):].strip()
                t = remainder  # from now on keep normalized version (fine for parsing)
                changed = True
                break

        # strip leading punctuation
        t = t.lstrip(" ,.:;!?-–—")
        if not changed:
            break

    return " ".join(t.split()).strip()

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
                    }
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

    # Chat Completions API with response_format json_schema
    # Docs: response_format json_schema ensures schema adherence. :contentReference[oaicite:2]{index=2}
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
    # content is JSON string; parse it
    import json
    obj = json.loads(data)
    return obj.get("items", [])