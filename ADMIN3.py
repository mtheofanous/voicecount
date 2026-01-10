# from __future__ import annotations

import io
import os
import re
import json
import unicodedata
import zipfile
from datetime import datetime, timedelta
from typing import List, Set, Dict, Optional, Tuple, Any

import pandas as pd
import streamlit as st
from openpyxl import Workbook
from sqlmodel import SQLModel, Field, Session, select  # type: ignore

# ---------------------------------------------------------
# Imports from your project (fallbacks for single-file runs)
# ---------------------------------------------------------
try:
    from features.auth_and_manage.auth_multi_tenant import (
        init_auth_db,
        auth_gate,
        require_login,
        current_user,
        current_account,
        get_auth_session,
        hash_password,
        Account,
        User,
        Venue,
        VenueUser,
    )
except Exception:
    from auth_multi_tenant import (  # type: ignore
        init_auth_db,
        auth_gate,
        require_login,
        current_user,
        current_account,
        get_auth_session,
        hash_password,
        Account,
        User,
        Venue,
        VenueUser,
    )

try:
    from core.init import init_db  # type: ignore
except Exception:
    from init import init_db  # type: ignore

try:
    from core.db import get_session  # type: ignore
except Exception:
    from db import get_session  # type: ignore

try:
    from domain.models import Product  # type: ignore
except Exception:
    try:
        from models import Product  # type: ignore
    except Exception:
        Product = None  # type: ignore


# =========================================================
# Superadmin configuration
# =========================================================
SUPERADMIN_EMAILS = {
    e.strip().lower()
    for e in os.getenv("SUPERADMIN_EMAILS", "").split(",")
    if e.strip()
}

def is_superadmin(user: dict) -> bool:
    return user.get("email", "").lower() in SUPERADMIN_EMAILS


# =========================================================
# Audit Log model (stored in the same DB as auth tables)
# =========================================================
class AuditLog(SQLModel, table=True):
    __tablename__ = "audit_log"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)

    account_id: Optional[int] = Field(default=None, index=True)
    venue_id: Optional[int] = Field(default=None, index=True)

    actor_user_id: Optional[int] = Field(default=None, index=True)
    actor_email: Optional[str] = Field(default=None, index=True)

    action: str = Field(index=True)
    entity_type: str = Field(index=True)
    entity_id: Optional[int] = Field(default=None, index=True)

    before_json: Optional[str] = Field(default=None)
    after_json: Optional[str] = Field(default=None)
    meta_json: Optional[str] = Field(default=None)

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


# =========================================================
# JSON + audit helpers
# =========================================================
def _split_aliases_cell(cell) -> List[str]:
    if cell is None:
        return []
    if isinstance(cell, list):
        return [str(x).strip() for x in cell if str(x).strip()]
    try:
        if pd.isna(cell):
            return []
    except Exception:
        pass

    s = str(cell).strip()
    if not s:
        return []
    parts = re.split(r"[|,;]", s)
    out: List[str] = []
    seen: Set[str] = set()
    for p in parts:
        p2 = clean_basic(p)
        if p2 and p2 not in seen:
            seen.add(p2)
            out.append(p2)
    return out


def _jsonable(x: Any) -> Any:
    if isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, pd.Timestamp):
        return x.to_pydatetime().isoformat()
    if isinstance(x, pd.Series):
        return x.to_dict()
    if isinstance(x, pd.DataFrame):
        return x.to_dict(orient="records")
    try:
        import numpy as np  # type: ignore
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
    except Exception:
        pass
    return x


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=_jsonable)


def log_audit(
    *,
    action: str,
    entity_type: str,
    entity_id: Optional[int] = None,
    account_id: Optional[int] = None,
    venue_id: Optional[int] = None,
    before: Optional[dict] = None,
    after: Optional[dict] = None,
    meta: Optional[dict] = None,
) -> None:
    try:
        u = current_user() or {}
        entry = AuditLog(
            account_id=account_id or u.get("account_id"),
            venue_id=venue_id,
            actor_user_id=u.get("id"),
            actor_email=u.get("email"),
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            before_json=_dumps(before) if before else None,
            after_json=_dumps(after) if after else None,
            meta_json=_dumps(meta) if meta else None,
        )
        with get_auth_session() as s:
            s.add(entry)
            s.commit()
    except Exception:
        return


# =========================================================
# XLSX repair helpers (broken styles.xml / invalid XML)
# =========================================================
@st.cache_data(show_spinner=False)
def _minimal_styles_xml() -> bytes:
    wb = Workbook()
    bio = io.BytesIO()
    wb.save(bio)
    bio.seek(0)
    with zipfile.ZipFile(bio, "r") as z:
        return z.read("xl/styles.xml")


def repair_xlsx_styles(xlsx_bytes: bytes) -> Tuple[bytes, bool]:
    did_repair = False
    min_styles = _minimal_styles_xml()

    zin = zipfile.ZipFile(io.BytesIO(xlsx_bytes), "r")
    out = io.BytesIO()
    names = set(zin.namelist())

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "xl/styles.xml":
                data = min_styles
                did_repair = True
            zout.writestr(item.filename, data)

        if "xl/styles.xml" not in names:
            zout.writestr("xl/styles.xml", min_styles)
            did_repair = True

    return out.getvalue(), did_repair


def safe_excel_file(uploaded_file) -> Tuple[pd.ExcelFile, bytes, bool]:
    b = uploaded_file.getvalue()
    try:
        xls = pd.ExcelFile(io.BytesIO(b))
        return xls, b, False
    except Exception as e:
        msg = str(e).lower()
        if ("stylesheet" in msg) or ("invalid xml" in msg) or ("workbook" in msg and "xml" in msg):
            fixed, did = repair_xlsx_styles(b)
            xls = pd.ExcelFile(io.BytesIO(fixed))
            return xls, fixed, did
        raise


# =========================================================
# Normalization helpers
# =========================================================
GREEK_RANGE = ("\u0370", "\u03FF")
GREEK_EXT_RANGE = ("\u1F00", "\u1FFF")

def has_greek(s: str) -> bool:
    return any(
        (GREEK_RANGE[0] <= ch <= GREEK_RANGE[1]) or (GREEK_EXT_RANGE[0] <= ch <= GREEK_EXT_RANGE[1])
        for ch in (s or "")
    )

def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)

def clean_basic(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s\u0370-\u03FF\u1F00-\u1FFF]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def dedupe_keep_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        x = (x or "").strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def normalize_unit(u: str) -> str:
    u = clean_basic(u).replace(".", "").strip()
    u = re.sub(r"\s+", " ", u)
    return u


# =========================================================
# Alias generation (provider NOT used) + conflict safety
# =========================================================
GREEK_PLURAL_EXCEPTIONS = {"γάλα": "γάλατα", "γαλα": "γαλατα", "γιαούρτι": "γιαούρτια", "γιαουρτι": "γιαουρτια"}

def greek_plural_token(tok: str) -> str:
    t = clean_basic(tok)
    return clean_basic(GREEK_PLURAL_EXCEPTIONS[t]) if t in GREEK_PLURAL_EXCEPTIONS else t

CONFUSION_GROUPS = [("ι",["ι","η","υ"]),("ει",["ει","ι"]),("οι",["οι","ι"]),("ο",["ο","ω"]),("ε",["ε","αι"])]

def greek_variants_token(tok: str, max_variants: int = 6) -> Set[str]:
    tok0 = clean_basic(tok)
    if not tok0 or not has_greek(tok0) or len(tok0) < 4:
        return {tok0} if tok0 else set()
    variants: Set[str] = {tok0, tok0.replace("ς","σ"), tok0.replace("σ","ς"), re.sub(r"(.)\1+", r"\1", tok0)}
    for base, opts in CONFUSION_GROUPS:
        new_set = set(variants)
        for v in variants:
            if base in v:
                for o in opts:
                    new_set.add(v.replace(base, o))
        variants = new_set
        if len(variants) > max_variants:
            break
    return set(list(variants)[:max_variants])

def greek_variants_phrase(text: str, max_total: int = 12) -> Set[str]:
    t = clean_basic(text)
    if not t or not has_greek(t):
        return set()
    toks = t.split()
    token_vars = [list(greek_variants_token(tok)) for tok in toks]
    out: Set[str] = set()
    def rec(i: int, acc: List[str]):
        if len(out) >= max_total:
            return
        if i == len(token_vars):
            out.add(clean_basic(" ".join(acc)))
            return
        for v in token_vars[i]:
            rec(i+1, acc+[v])
            if len(out) >= max_total:
                return
    rec(0, [])
    return out

def generate_aliases_smart(
    name: str,
    lang: str = "auto",
    *,
    include_misspellings: bool = True,
    include_english_in_greek: bool = False,
    max_aliases: int = 60,
) -> List[str]:
    raw = (name or "").strip()
    if not raw:
        return []
    if lang == "auto":
        lang = "el" if has_greek(raw) else "es"
    base = clean_basic(raw)
    no_acc = clean_basic(strip_accents(raw))
    aliases: List[str] = [base, no_acc]

    if lang == "el" and base:
        plur = " ".join(greek_plural_token(t) for t in base.split())
        aliases += [clean_basic(plur), clean_basic(strip_accents(plur))]

    if include_misspellings:
        if lang == "el" and has_greek(base):
            aliases += list(greek_variants_phrase(base))
            aliases += list(greek_variants_phrase(no_acc))
        else:
            aliases += [re.sub(r"(.)\1+", r"\1", base)]

    aliases = [a for a in aliases if a and len(a) >= 2]
    aliases = dedupe_keep_order(aliases)[:max_aliases]
    return sorted(aliases)

# stopwords / claims
STOPWORDS = {
    "και","κι","σε","στο","στη","στην","των","του","της","με","απο","από","για",
    # units / packaging / generic abbreviations
    "x","τεμ","τεμ.","τμχ","τμχ.","tmx","stk","stick","sticks","sachet","portion","portions","pack","box","bottle","can",
    # greek packaging words
    "μεριδα","μερίδα","μεριδες","μερίδες","στικ","στικάκι","στικακι","φακελακι","φακελάκι",
    # measures
    "kg","g","l","ml","ltr","lt"
    "κιλο",
    "κιλά",
    "κιλα",
    "κιλ",
    "κουβας",
    "κουβα",
    "κιβ",
    "κιβωτιο",
    "κιβώτιο",
    "κιβωτια",
    "κιβώτια",
    "χ",
}
CLAIM_TERMS = {"0","0%","zero","sugar","sugarfree","light","free","bio","organic","eco","ζαχαρη","ζάχαρη","χωρις","χωρίς","natrue"}

GENERIC_TERMS = {"γαλα","γάλα","milk","leche","καφε","καφες","καφές","coffee","λαδι","λάδι","oil","water","νερο","νερό","agua","mince","κιμα","κιμά","κιμας","κιμάς"}

# Only these very-generic single words are collision-filtered (to avoid deleting useful tokens like 'μεγαρων').
GENERIC_SINGLE_WORD_COLLISION = {clean_basic(strip_accents(x)) for x in GENERIC_TERMS}

def _greek_sigma_variants(tok: str) -> Set[str]:
    t = clean_basic(tok)
    if not t:
        return set()
    out = {t, strip_accents(t)}
    if has_greek(t):
        if t.endswith("ς"):
            out.add(t[:-1])
            out.add(t[:-1] + "σ")
        if t.endswith("σ"):
            out.add(t[:-1] + "ς")
        out.add(t.replace("ς", "σ"))
    return {clean_basic(x) for x in out if x}

def _tokenize_mixed(text: str) -> List[str]:
    t = clean_basic(strip_accents(text or ""))
    toks = [w for w in t.split() if w]
    out = []
    for w in toks:
        if w.isdigit():
            continue
        if sum(ch.isdigit() for ch in w) >= max(1, len(w)//2):
            continue
        if w in STOPWORDS or w in CLAIM_TERMS:
            continue
        out.append(w)
    return out

# --- Controlled Latin token extraction from description (brands/product terms) ---
DESC_LATIN_STOP = {
    "sv","kg","gr","g","lt","l","ml","pcs","pc","pack","pkt","tmx","stk","stick","sticks"
}

def _extract_desc_latin_tokens(desc: str, *, max_tokens: int = 10) -> List[str]:
    """Extract meaningful Latin words from the description (e.g., 'prosciutto', 'motta').
    Filters out quantities and packaging abbreviations."""
    if not desc:
        return []
    raw = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]{3,}", str(desc))
    out: List[str] = []
    seen: Set[str] = set()
    for w in raw:
        wn = clean_basic(strip_accents(w))
        if not wn or wn in DESC_LATIN_STOP or wn in STOPWORDS or wn in CLAIM_TERMS:
            continue
        if wn.isnumeric():
            continue
        if wn in seen:
            continue
        seen.add(wn)
        out.append(wn)
        if len(out) >= max_tokens:
            break
    return out

def _extract_desc_latin_phrases(desc: str, *, max_phrases: int = 6) -> List[str]:
    toks = _extract_desc_latin_tokens(desc, max_tokens=12)
    phrases: List[str] = []
    for i in range(len(toks) - 1):
        phrases.append(f"{toks[i]} {toks[i+1]}")
        if len(phrases) >= max_phrases:
            break
    return phrases


# --- Controlled Greek token extraction from description (brands/locations/product terms) ---
GREEK_DESC_STOP = {
    # units / packaging / logistics
    "κιλο","κιλά","κιλα","κιλ","τεμ","τεμ.","τμχ","τμχ.","πακ","κουβας","κουβα","κιβ","κιβωτιο","κιβώτιο","κιβωτια","κιβώτια",
    "x","χ","tmx","stk","stick","sticks","στικ","στικάκι","στικακι","φακελακι","φακελάκι",
    # generic glue words
    "με","σε","και","του","της","των","α","β",
}

def _is_numberish(tok: str) -> bool:
    t = (tok or "").replace(",", ".")
    if not t:
        return False
    if re.fullmatch(r"\d+(\.\d+)?", t):
        return True
    return any(ch.isdigit() for ch in tok)

def _extract_desc_greek_tokens(desc: str, *, max_tokens: int = 12) -> List[str]:
    """Extract meaningful Greek words from description (e.g., 'μεγαρων', 'αττικον').
    Filters out units/packaging/numbers/single letters."""
    if not desc:
        return []
    s = str(desc)

    raw = re.findall(r"[Α-Ωα-ωάέήίόύώϊΐϋΰ]{2,}", s)
    out: List[str] = []
    seen: Set[str] = set()
    for w in raw:
        wn = clean_basic(strip_accents(w))
        if not wn:
            continue
        if wn in STOPWORDS or wn in CLAIM_TERMS or wn in GREEK_DESC_STOP:
            continue
        if len(wn) < 3:
            continue
        if _is_numberish(wn):
            continue
        if wn in seen:
            continue
        seen.add(wn)
        out.append(wn)
        if len(out) >= max_tokens:
            break
    return out

def _extract_desc_greek_phrases(desc: str, *, max_phrases: int = 6) -> List[str]:
    toks = _extract_desc_greek_tokens(desc, max_tokens=12)
    phrases: List[str] = []
    for i in range(len(toks) - 1):
        phrases.append(f"{toks[i]} {toks[i+1]}")
        if len(phrases) >= max_phrases:
            break
    return phrases

def _concept_aliases(name: str, desc: str = "", category: str = "") -> Set[str]:
    """
    Derive high-level concept aliases from name + description ONLY.
    Category is accepted for compatibility but intentionally ignored
    to avoid wrong concepts (e.g. sugar -> coffee).
    """
    t = " ".join(
        _tokenize_mixed(" ".join([name or "", desc or ""]))
    )

    out: Set[str] = set()

    # Oil
    if any(k in t for k in ["ελαιο", "ελαιολαδο", "sunflower", "olive", "aceite", "oil"]):
        out |= {"λαδι", "λάδι", "oil", "aceite"}

    # Coffee
    if any(k in t for k in ["καφε", "καφες", "espresso", "cafe", "coffee"]):
        out |= {"καφε", "καφες", "καφές", "coffee", "cafe"}

    # Milk
    if any(k in t for k in ["γαλα", "milk", "leche", "barista"]):
        out |= {"γαλα", "γάλα", "milk", "leche"}

    # Minced meat
    if any(k in t for k in ["κιμα", "κιμας", "mince", "ground", "carne picada"]):
        out |= {"κιμα", "κιμά", "κιμας", "κιμάς", "mince"}

    # normalize + dedupe
    return {clean_basic(x) for x in out if x}

def _openai_suggest_aliases_safe(name: str, desc: str, *, max_new: int = 12) -> List[str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return []
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return []

    base_tokens = set(_tokenize_mixed(" ".join([name or "", desc or ""])))
    allowed: Set[str] = set()
    for t in base_tokens:
        allowed |= _greek_sigma_variants(t)

    client = OpenAI(api_key=api_key)
    prompt = (
        "Propose 5–12 short, voice-friendly aliases for matching THIS product only.\n"
        "Rules:\n"
        "- Do NOT include provider/brand/supplier names.\n"
        "- Do NOT include nutrition/marketing claims (0%, sugar-free, light, bio, organic).\n"
        "- Do NOT include generic words like 'product' or 'food'.\n"
        "- Keep each alias 1–3 words.\n"
        "- Use only words already present in name/description (accent/sigma variants ok).\n"
        "Return ONLY a JSON array of strings.\n\n"
        f"Name: {name}\n"
        f"Description: {desc}\n"
    )

    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        arr = json.loads(raw) if raw.startswith("[") else []
        out: List[str] = []
        for a in arr[: max_new * 2]:
            a_norm = clean_basic(strip_accents(str(a)))
            if not a_norm or a_norm in STOPWORDS or a_norm in CLAIM_TERMS:
                continue
            if len(a_norm.split()) > 3:
                continue
            toks = a_norm.split()
            if toks and all(t in allowed for t in toks):
                out.append(a_norm)
        return dedupe_keep_order(out)[:max_new]
    except Exception:
        return []

def generate_aliases_robust(
    *,
    name: str,
    description: str = "",
    category: str = "",
    lang: str = "auto",
    include_misspellings: bool = True,
    include_english_in_greek: bool = False,
    max_aliases: int = 80,
    add_concepts: bool = True,
    openai_enrich: bool = False,
    openai_max_new: int = 10,
    # Description controls
    include_desc_latin: bool = True,
    include_desc_greek: bool = True,
    include_desc_phrases: bool = True,
    # Optional debug collector (filled in-place)
    debug: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    base = generate_aliases_smart(
        name,
        lang=lang,
        include_misspellings=include_misspellings,
        include_english_in_greek=include_english_in_greek,
        max_aliases=max(10, min(200, max_aliases)),
    )

    out: Set[str] = set(base)

    for a in list(out):
        for tok in a.split():
            out |= _greek_sigma_variants(tok)

    if add_concepts:
        out |= _concept_aliases(name, description, category)

    openai_added: List[str] = []
    if openai_enrich:
        openai_added = _openai_suggest_aliases_safe(name, description, max_new=openai_max_new)
        out |= set(openai_added)
    # Add controlled description tokens/phrases (Latin + Greek), filtered for packaging/claims
    desc_latin: List[str] = []
    desc_greek: List[str] = []
    desc_latin_ph: List[str] = []
    desc_greek_ph: List[str] = []

    if description:
        if include_desc_latin:
            desc_latin = _extract_desc_latin_tokens(description)
            out |= set(desc_latin)
            if include_desc_phrases:
                desc_latin_ph = _extract_desc_latin_phrases(description)
                out |= set(desc_latin_ph)

        if include_desc_greek:
            desc_greek = _extract_desc_greek_tokens(description)
            out |= set(desc_greek)
            if include_desc_phrases:
                desc_greek_ph = _extract_desc_greek_phrases(description)
                out |= set(desc_greek_ph)

    # Debug collector: fill sources (normalized later, but useful for UX)
    if debug is not None:
        debug.setdefault("name_base", []).extend(base)
        if add_concepts:
            debug.setdefault("concepts", []).extend(list(_concept_aliases(name, description, category)))
        if openai_enrich and openai_added:
            debug.setdefault("openai", []).extend(openai_added)
        if include_desc_latin:
            debug.setdefault("desc_latin", []).extend(desc_latin)
            if include_desc_phrases:
                debug.setdefault("desc_latin_phrases", []).extend(desc_latin_ph)
        if include_desc_greek:
            debug.setdefault("desc_greek", []).extend(desc_greek)
            if include_desc_phrases:
                debug.setdefault("desc_greek_phrases", []).extend(desc_greek_ph)

    cleaned: List[str] = []
    for a in out:
        a2 = clean_basic(strip_accents(a))
        if not a2 or len(a2) < 2:
            continue
        if a2 in STOPWORDS or a2 in CLAIM_TERMS:
            continue
        cleaned.append(a2)

    cleaned = dedupe_keep_order(cleaned)
    return sorted(cleaned[:max_aliases])

def _norm_name_for_conflict(s: str) -> str:
    return clean_basic(strip_accents(s or ""))

def _alias_conflicts_single_word_only(alias_norm: str, this_name_norm: str, all_name_norms: List[str]) -> bool:
    """Collision filter used during import alias generation.

    - Only applies to *single-word* aliases.
    - Only applies to *very generic* words (see GENERIC_SINGLE_WORD_COLLISION).
      This keeps useful single words like place/brand descriptors (e.g., 'μεγαρων', 'αττικον').
    - Multi-word aliases are always kept by this rule.
    """
    if not alias_norm or len(alias_norm) < 3:
        return True
    if len(alias_norm.split()) != 1:
        return False

    # Only filter collisions for very-generic words
    if alias_norm not in GENERIC_SINGLE_WORD_COLLISION:
        return False

    for n in all_name_norms:
        if n == this_name_norm:
            continue
        if alias_norm in n:
            return True
    return False



# =========================================================
# Admin helpers (users/venues/products)
# =========================================================
def require_admin() -> Tuple[dict, dict]:
    u = current_user() or {}
    acc = current_account() or {}
    role = (u.get("account_role") or "").lower()
    if not (is_superadmin(u) or role in {"owner", "admin", "manager"}):
        st.error("You don't have permission to access Admin.")
        st.stop()
    return u, acc

def list_venues_for_account(account_id: int) -> List[Venue]:
    with get_auth_session() as s:
        return s.exec(select(Venue).where(Venue.account_id == account_id).order_by(Venue.name.asc())).all()

def list_users_for_account(account_id: int) -> List[User]:
    with get_auth_session() as s:
        return s.exec(select(User).where(User.account_id == account_id).order_by(User.email.asc())).all()

def list_all_venues() -> List[Venue]:
    with get_auth_session() as s:
        return s.exec(select(Venue).order_by(Venue.name.asc())).all()

def list_all_users() -> List[User]:
    with get_auth_session() as s:
        return s.exec(select(User).order_by(User.email.asc())).all()

def update_user_basic(user_id: int, *, full_name: str, email: str, account_role: str, is_active: bool) -> None:
    with get_auth_session() as s:
        u = s.exec(select(User).where(User.id == user_id)).first()
        if not u:
            raise ValueError("User not found")
        before = {"full_name": u.full_name, "email": u.email, "account_role": u.account_role, "is_active": u.is_active}
        u.full_name = full_name
        u.email = email
        u.account_role = account_role
        u.is_active = bool(is_active)
        s.add(u)
        s.commit()
        after = {"full_name": u.full_name, "email": u.email, "account_role": u.account_role, "is_active": u.is_active}
    log_audit(action="USER_UPDATE", entity_type="user", entity_id=user_id, account_id=u.account_id, before=before, after=after)

def reset_user_password(user_id: int, new_password: str) -> None:
    if len(new_password or "") < 8:
        raise ValueError("Password must be at least 8 characters.")
    with get_auth_session() as s:
        u = s.exec(select(User).where(User.id == user_id)).first()
        if not u:
            raise ValueError("User not found")
        u.password_hash = hash_password(new_password)
        s.add(u)
        s.commit()
        acc_id = u.account_id
        email = u.email
    log_audit(action="USER_PASSWORD_RESET", entity_type="user", entity_id=user_id, account_id=acc_id, meta={"target_email": email})

def update_venue_basic(venue_id: int, *, name: str, tax_number: str, address: str, phone: str, email: str) -> None:
    with get_auth_session() as s:
        v = s.exec(select(Venue).where(Venue.id == venue_id)).first()
        if not v:
            raise ValueError("Venue not found")
        before = {"name": v.name, "tax_number": v.tax_number, "address": v.address, "phone": v.phone, "email": v.email}
        v.name = name
        v.tax_number = tax_number
        v.address = address
        v.phone = phone
        v.email = email
        s.add(v)
        s.commit()
        after = {"name": v.name, "tax_number": v.tax_number, "address": v.address, "phone": v.phone, "email": v.email}
        acc_id = v.account_id
    log_audit(action="VENUE_UPDATE", entity_type="venue", entity_id=venue_id, account_id=acc_id, venue_id=venue_id, before=before, after=after)

def list_products_for_venue(venue_id: int) -> pd.DataFrame:
    if Product is None:
        raise RuntimeError("Product model not importable.")
    with get_session() as s:
        rows = s.exec(select(Product).where(Product.venue_id == venue_id).order_by(Product.name.asc())).all()
    data = []
    for p in rows:
        data.append({
            "id": p.id,
            "venue_id": p.venue_id,
            "name": p.name,
            "description": p.description,
            "category": p.category,
            "unit": p.unit,
            "quantity": p.quantity,
            "price": p.price,
            "provider_name": p.provider_name,
            "aliases": p.aliases,
        })
    return pd.DataFrame(data)

def upsert_products_from_df(venue_id: int, df_upload: pd.DataFrame, mode: str = "upsert") -> Tuple[int, int]:
    if Product is None:
        raise RuntimeError("Product model not importable.")

    required = {"name","description","category","unit","quantity","price","provider_name","aliases"}
    missing = required - set(df_upload.columns)
    if missing:
        raise ValueError(f"Upload file missing columns: {sorted(missing)}")

    def key(name: str, unit: str) -> str:
        return clean_basic(name) + "||" + normalize_unit(unit or "unit")

    df_u = df_upload.copy()
    df_u["unit"] = df_u["unit"].astype(str).replace({"": "unit"})
    df_u["quantity"] = pd.to_numeric(df_u["quantity"], errors="coerce").fillna(1.0)
    df_u["price"] = pd.to_numeric(df_u["price"], errors="coerce").fillna(0.0)

    inserted = 0
    updated = 0
    audit_samples: List[dict] = []
    sample_limit = 50

    total = int(len(df_u))
    prog = st.progress(0.0)
    status = st.empty()

    with get_session() as s:
        existing = s.exec(select(Product).where(Product.venue_id == venue_id)).all()
        index = {key(p.name, p.unit): p for p in existing}

        for i, (_, r) in enumerate(df_u.iterrows(), start=1):
            k = key(str(r["name"]), str(r["unit"]))
            if mode == "upsert" and k in index:
                p = index[k]
                before = None
                if len(audit_samples) < sample_limit:
                    before = {"id": int(p.id), "name": p.name, "unit": p.unit, "price": p.price, "quantity": p.quantity}

                p.description = str(r.get("description") or "") or None
                p.category = str(r.get("category") or "") or None
                p.unit = str(r.get("unit") or "unit") or "unit"
                p.quantity = float(r.get("quantity") or 1.0)
                p.price = float(r.get("price") or 0.0)
                p.provider_name = str(r.get("provider_name") or "") or None
                p.aliases = str(r.get("aliases") or "") or None
                if hasattr(p, "updated_at"):
                    p.updated_at = datetime.utcnow()

                s.add(p)
                updated += 1

                if len(audit_samples) < sample_limit and before is not None:
                    after = {"id": int(p.id), "name": p.name, "unit": p.unit, "price": p.price, "quantity": p.quantity}
                    audit_samples.append({"action": "PRODUCT_UPDATE", "entity_id": int(p.id), "before": before, "after": after})

            else:
                p = Product(
                    venue_id=int(venue_id),
                    name=str(r.get("name") or "").strip(),
                    description=(str(r.get("description") or "").strip() or None),
                    category=(str(r.get("category") or "").strip() or None),
                    unit=(str(r.get("unit") or "unit").strip() or "unit"),
                    quantity=float(r.get("quantity") or 1.0),
                    price=float(r.get("price") or 0.0),
                    provider_name=(str(r.get("provider_name") or "").strip() or None),
                    aliases=(str(r.get("aliases") or "").strip() or None),
                )
                s.add(p)
                s.flush()
                inserted += 1
                if len(audit_samples) < sample_limit:
                    audit_samples.append({"action": "PRODUCT_INSERT", "entity_id": int(p.id), "before": None, "after": {"id": int(p.id), "name": p.name, "unit": p.unit}})

            if total > 0 and (i % 25 == 0 or i == total):
                prog.progress(min(1.0, i / total))
                status.text(f"Uploading… {i}/{total} rows")

        s.commit()

    prog.progress(1.0)
    status.text(f"Upload committed ✅ Inserted: {inserted} • Updated: {updated}")

    log_audit(
        action="PRODUCT_UPLOAD",
        entity_type="upload",
        venue_id=venue_id,
        meta={"mode": mode, "inserted": inserted, "updated": updated, "rows": total, "samples": audit_samples},
    )
    return inserted, updated


# =========================================================
# App boot
# =========================================================
st.set_page_config(page_title="Import + Admin", page_icon="🧩", layout="wide")

init_db()
init_auth_db()

auth_gate(show_manage_org=True)
require_login()

u = current_user() or {}
acc = current_account() or {}
account_role = (u.get("account_role") or "member").lower()

if is_superadmin(u):
    st.sidebar.success("🔐 SUPERADMIN MODE")

st.title("🧩 Import Assistant + Admin")
st.caption("Create DB-ready product files (with safe aliases) + manage venues/users/catalogs + audit logs.")

tabs = st.tabs(["🧩 Import Assistant", "🛠️ Admin"])


# =========================================================
# TAB 1: Import Assistant (DB-ready)
# =========================================================
with tabs[0]:
    st.subheader("Import Assistant (DB-ready)")

    with st.sidebar:
        st.header("Import Settings")

        venue_id_enabled = st.toggle("Include venue_id column (optional)", value=False, key="imp_vid_toggle")
        venue_id_value = None
        if venue_id_enabled:
            venue_id_value = st.number_input(
                "venue_id",
                min_value=1,
                value=int(st.session_state.get("active_venue_id") or 1),
                step=1,
                key="imp_vid_value",
            )

        alias_lang = st.selectbox("Alias language", ["auto", "el", "es", "en"], index=0, key="imp_alias_lang")
        include_misspellings = st.toggle("Include misspellings", value=True, key="imp_alias_miss")
        include_eng_to_gr = st.toggle("Include English-in-Greek (EL)", value=False, key="imp_alias_eng2gr")
        max_aliases = st.slider("Max aliases per product", min_value=10, max_value=200, value=80, step=10, key="imp_alias_max")
        add_concepts = st.toggle("Add concept aliases (milk/coffee/oil…)", value=True, key="imp_alias_concepts")
        use_desc_in_alias = st.toggle("Use description text for aliases", value=True, key="imp_alias_use_desc")
        include_desc_greek = st.toggle("Include Greek words from description", value=True, key="imp_desc_gr")
        include_desc_latin = st.toggle("Include Latin words from description", value=True, key="imp_desc_lat")
        include_desc_phrases = st.toggle("Include 2-word phrases from description", value=True, key="imp_desc_ph")
        show_alias_debug = st.toggle("Show 'Why these aliases?' debug", value=False, key="imp_alias_debug")

        openai_enrich = st.toggle("OpenAI alias suggestions (filtered)", value=False, key="imp_alias_openai")
        openai_max_new = st.slider("OpenAI max new aliases", min_value=0, max_value=25, value=10, step=1, key="imp_alias_openai_max")

        st.divider()
        st.subheader("Collision safety")
        allow_generic_shared = st.toggle("Allow shared generic aliases (e.g. 'water')", value=False, key="imp_allow_generic")
        filter_single_word_collisions = st.toggle("Filter 1-word aliases that appear in other product names", value=True, key="imp_alias_conflict_filter")

    uploaded = st.file_uploader("Upload client Excel (.xlsx)", type=["xlsx"], key="imp_upload")
    if not uploaded:
        st.info("Upload a client Excel to begin.")
        st.stop()

    try:
        xls, bytes_used, did_repair = safe_excel_file(uploaded)
        if did_repair:
            st.warning("⚠️ This Excel had invalid XML/styles. It was auto-repaired for reading.")
            st.download_button(
                "⬇️ Download repaired Excel",
                data=bytes_used,
                file_name="client_repaired.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="imp_dl_repaired",
            )

        sheet = st.selectbox("Select sheet", xls.sheet_names, index=0, key="imp_sheet")
        df_raw = pd.read_excel(xls, sheet_name=sheet)
    except Exception as e:
        st.error(f"Could not read the Excel file (even after repair attempt): {e}")
        st.stop()

    st.markdown("#### Raw preview")
    st.dataframe(df_raw.head(50), use_container_width=True)

    HEADER_SYNONYMS: Dict[str, List[str]] = {
        "name": ["name","product","item","descripcion","descripción","producto","προϊον","προϊόν","ειδος","είδος","ονομα","όνομα"],
        "description": ["description","desc","details","detalle","detalles","observaciones","nota","notas","περιγραφη","περιγραφή","σχόλια"],
        "category": ["category","categoria","categoría","κατηγορια","κατηγορία","familia","family"],
        "unit": ["unit","unidad","uds","ud","μονάδα","μοναδα","τεμ","τεμ.","uom","unidad de medida"],
        "quantity": ["quantity","qty","cantidad","ποσοτητα","ποσότητα","amount"],
        "price": ["price","precio","τιμη","τιμή","cost","costo","κόστος"],
        "provider_name": ["provider","supplier","proveedor","προμηθευτης","προμηθευτής"],
    }

    def guess_column(df0: pd.DataFrame, canonical: str) -> Optional[str]:
        targets = set(clean_basic(x) for x in HEADER_SYNONYMS.get(canonical, []))
        for c in df0.columns:
            if clean_basic(str(c)) in targets:
                return c
        return None

    guess_name = guess_column(df_raw, "name")
    guess_desc = guess_column(df_raw, "description")
    guess_category = guess_column(df_raw, "category")
    guess_unit = guess_column(df_raw, "unit")
    guess_qty = guess_column(df_raw, "quantity")
    guess_price = guess_column(df_raw, "price")
    guess_provider = guess_column(df_raw, "provider_name")

    cols = ["(none)"] + [str(c) for c in df_raw.columns]

    st.markdown("### Map columns")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        name_col = st.selectbox("name", cols, index=(cols.index(guess_name) if guess_name in cols else 1), key="imp_map_name")
        desc_col = st.selectbox("description", cols, index=(cols.index(guess_desc) if guess_desc in cols else 0), key="imp_map_desc")
    with c2:
        category_col = st.selectbox("category", cols, index=(cols.index(guess_category) if guess_category in cols else 0), key="imp_map_cat")
        unit_col = st.selectbox("unit", cols, index=(cols.index(guess_unit) if guess_unit in cols else 0), key="imp_map_unit")
    with c3:
        qty_col = st.selectbox("quantity", cols, index=(cols.index(guess_qty) if guess_qty in cols else 0), key="imp_map_qty")
        price_col = st.selectbox("price", cols, index=(cols.index(guess_price) if guess_price in cols else 0), key="imp_map_price")
    with c4:
        provider_col = st.selectbox("provider_name", cols, index=(cols.index(guess_provider) if guess_provider in cols else 0), key="imp_map_provider")

    if name_col == "(none)":
        st.error("You must select a name column.")
        st.stop()

    def get_series(colname: str, default: pd.Series) -> pd.Series:
        if colname == "(none)":
            return default
        return df_raw[colname]

    df = pd.DataFrame()
    df["name"] = get_series(name_col, pd.Series([""] * len(df_raw))).astype(str).map(lambda x: x.strip())
    df["description"] = get_series(desc_col, pd.Series([""] * len(df_raw))).astype(str).fillna("")
    df["category"] = get_series(category_col, pd.Series([""] * len(df_raw))).astype(str).fillna("")
    df["unit"] = get_series(unit_col, pd.Series([""] * len(df_raw))).astype(str).fillna("")
    df["quantity"] = pd.to_numeric(get_series(qty_col, pd.Series([None] * len(df_raw))), errors="coerce")
    df["price"] = pd.to_numeric(get_series(price_col, pd.Series([None] * len(df_raw))), errors="coerce")
    df["provider_name"] = get_series(provider_col, pd.Series([""] * len(df_raw))).astype(str).fillna("")

    if venue_id_enabled and venue_id_value is not None:
        df["venue_id"] = int(venue_id_value)

    # Clean minimal fields
    df["name_norm"] = df["name"].astype(str).map(_norm_name_for_conflict)
    df.loc[df["unit"].astype(str).str.strip().eq(""), "unit"] = "unit"
    df = df[df["name_norm"].str.len() > 0].copy()

    st.markdown("### Generate aliases")
    gen_clicked = st.button("✨ Generate aliases", type="primary", key="imp_gen_aliases")

    if gen_clicked:
        # build name index (uploaded names + optionally current venue catalog names)
        all_names: List[str] = df["name"].astype(str).tolist()
        if venue_id_enabled and venue_id_value is not None and Product is not None:
            try:
                with get_session() as s:
                    rows = s.exec(select(Product.name).where(Product.venue_id == int(venue_id_value))).all()
                all_names += [r[0] for r in rows if r and r[0]]
            except Exception:
                pass
        all_name_norms = dedupe_keep_order([_norm_name_for_conflict(n) for n in all_names if _norm_name_for_conflict(n)])

        prog = st.progress(0.0)
        with st.spinner("Generating safe aliases…"):
            out_aliases: List[str] = []
            debug_rows: List[Dict[str, List[str]]] = []
            total = max(1, len(df))
            for i, r in enumerate(df.to_dict(orient="records"), start=1):
                name = str(r.get("name") or "")
                desc = str(r.get("description") or "") if use_desc_in_alias else ""
                cat = str(r.get("category") or "")
                dbg: Dict[str, List[str]] = {}
                aliases_list = generate_aliases_robust(
                    name=name,
                    description=desc,
                    category=cat,
                    lang=alias_lang,
                    include_misspellings=include_misspellings,
                    include_english_in_greek=include_eng_to_gr,
                    max_aliases=int(max_aliases),
                    add_concepts=bool(add_concepts),
                    openai_enrich=bool(openai_enrich),
                    openai_max_new=int(openai_max_new),
                    include_desc_latin=bool(include_desc_latin) and bool(use_desc_in_alias),
                    include_desc_greek=bool(include_desc_greek) and bool(use_desc_in_alias),
                    include_desc_phrases=bool(include_desc_phrases) and bool(use_desc_in_alias),
                    debug=dbg,

                )
                name_norm = _norm_name_for_conflict(name)
                cleaned = []
                for a in aliases_list:
                    a_norm = clean_basic(strip_accents(a))
                    if not a_norm or a_norm in STOPWORDS or a_norm in CLAIM_TERMS:
                        continue
                    if filter_single_word_collisions and _alias_conflicts_single_word_only(a_norm, name_norm, all_name_norms):
                        continue
                    cleaned.append(a_norm)

                # optional: remove shared generic terms (only if user disables)
                if not allow_generic_shared:
                    cleaned = [a for a in cleaned if a not in {clean_basic(strip_accents(x)) for x in GENERIC_TERMS}]

                out_aliases.append(" | ".join(dedupe_keep_order(cleaned)))
                debug_rows.append(dbg)
                if i % 20 == 0 or i == total:
                    prog.progress(min(1.0, i / total))

            df["aliases"] = out_aliases
            st.session_state["alias_debug_rows"] = debug_rows

        st.success("Aliases generated. Preview + export are ready below.")

        # -----------------------------------------------------
        # Why these aliases? (debug)
        # -----------------------------------------------------
        if show_alias_debug and st.session_state.get("alias_debug_rows") and isinstance(df, pd.DataFrame) and not df.empty:
            with st.expander("🔎 Why these aliases? (debug)", expanded=False):
                st.caption("Pick a row to see which sources produced which aliases (name, description tokens, OpenAI, concepts).")
                idx_options = list(df.index)
                default_idx = idx_options[0]
                pick_idx = st.selectbox("Row", idx_options, index=0, key="imp_debug_pick_idx")
                row = df.loc[pick_idx].to_dict()
                dbg = st.session_state["alias_debug_rows"][idx_options.index(pick_idx)]
                st.markdown(f"**Product:** {row.get('name','')}")
                st.markdown(f"**Description:** {row.get('description','')}")
                st.markdown(f"**Final aliases:** {row.get('aliases','')}")
                st.divider()
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("**From name (base):**")
                    st.write(sorted(set(dbg.get("name_base", []))))
                    if dbg.get("concepts"):
                        st.markdown("**Concept aliases:**")
                        st.write(sorted(set(dbg.get("concepts", []))))
                with cols[1]:
                    if dbg.get("desc_greek") or dbg.get("desc_greek_phrases"):
                        st.markdown("**From description (Greek):**")
                        st.write(sorted(set(dbg.get("desc_greek", []))))
                        if dbg.get("desc_greek_phrases"):
                            st.markdown("**Greek phrases:**")
                            st.write(sorted(set(dbg.get("desc_greek_phrases", []))))
                    if dbg.get("desc_latin") or dbg.get("desc_latin_phrases"):
                        st.markdown("**From description (Latin):**")
                        st.write(sorted(set(dbg.get("desc_latin", []))))
                        if dbg.get("desc_latin_phrases"):
                            st.markdown("**Latin phrases:**")
                            st.write(sorted(set(dbg.get("desc_latin_phrases", []))))
                    if dbg.get("openai"):
                        st.markdown("**From OpenAI:**")
                        st.write(sorted(set(dbg.get("openai", []))))
        st.session_state["last_ready_df"] = df.drop(columns=["name_norm"], errors="ignore").copy()

    st.markdown("### Enriched preview (DB import)")
    df_show = df.drop(columns=["name_norm"], errors="ignore").copy()
    st.dataframe(df_show.head(50), use_container_width=True)

    def _df_to_excel_bytes(df0: pd.DataFrame, sheet_name: str = "enriched") -> bytes:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df0.to_excel(writer, index=False, sheet_name=sheet_name)
        return buf.getvalue()

    st.download_button(
        "⬇️ Download enriched Excel (with aliases)",
        data=_df_to_excel_bytes(df_show, sheet_name="client_enriched"),
        file_name="client_enriched_with_aliases.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="imp_dl_enriched",
    )


# =========================================================
# TAB 2: Admin
# =========================================================
with tabs[1]:
    st.subheader("Admin")
    u_admin, acc_admin = require_admin()
    acc_id = int(acc_admin["id"])

    admin_tabs = st.tabs(["🏬 Venues", "👤 Users", "📦 Venue Catalog", "⬆️ Upload to DB", "🧾 Audit Logs"])

    if is_superadmin(u_admin):
        venues_scope = list_all_venues()
        users_scope = list_all_users()
    else:
        venues_scope = list_venues_for_account(acc_id)
        users_scope = list_users_for_account(acc_id)

    with admin_tabs[0]:
        st.markdown("### Venues")
        if not venues_scope:
            st.info("No venues found.")
        else:
            venue_labels = [f"{v.name} (#{v.id})" for v in venues_scope]
            chosen = st.selectbox("Select venue", venue_labels, index=0, key="adm_venue_sel")
            v = venues_scope[venue_labels.index(chosen)]

            st.markdown("#### Edit venue")
            c1, c2 = st.columns(2)
            with c1:
                v_name = st.text_input("Name", value=v.name or "", key="adm_v_name")
                v_tax = st.text_input("Tax number", value=v.tax_number or "", key="adm_v_tax")
                v_email = st.text_input("Email", value=v.email or "", key="adm_v_email")
            with c2:
                v_phone = st.text_input("Phone", value=v.phone or "", key="adm_v_phone")
                v_addr = st.text_area("Address", value=v.address or "", key="adm_v_addr")

            if st.button("Save venue changes", type="primary", key="adm_save_venue"):
                try:
                    update_venue_basic(int(v.id), name=v_name, tax_number=v_tax, address=v_addr, phone=v_phone, email=v_email)
                    st.success("Venue updated ✅")
                except Exception as e:
                    st.error(str(e))

    with admin_tabs[1]:
        st.markdown("### Users")
        if not users_scope:
            st.info("No users found.")
        else:
            user_labels = [f"{uu.email} — {uu.full_name} (#{uu.id})" for uu in users_scope]
            chosen_u = st.selectbox("Select user", user_labels, index=0, key="adm_user_sel")
            usr = users_scope[user_labels.index(chosen_u)]

            st.markdown("#### Edit user")
            c1, c2 = st.columns(2)
            with c1:
                u_full = st.text_input("Full name", value=usr.full_name or "", key="adm_u_full")
                u_email = st.text_input("Email", value=usr.email or "", key="adm_u_email")
            with c2:
                u_role = st.selectbox("Account role", ["owner", "admin", "manager", "staff", "viewer"], index=["owner","admin","manager","staff","viewer"].index((usr.account_role or "staff")), key="adm_u_role")
                u_active = st.checkbox("Active", value=bool(usr.is_active), key="adm_u_active")

            if st.button("Save user changes", type="primary", key="adm_save_user"):
                try:
                    update_user_basic(int(usr.id), full_name=u_full, email=u_email.strip().lower(), account_role=u_role, is_active=u_active)
                    st.success("User updated ✅")
                except Exception as e:
                    st.error(str(e))

            st.markdown("#### Reset password")
            new_pwd = st.text_input("New password (min 8 chars)", type="password", key="adm_pwd_new")
            if st.button("Set new password", key="adm_set_pwd"):
                try:
                    reset_user_password(int(usr.id), new_pwd)
                    st.success("Password reset ✅")
                    st.session_state["adm_pwd_new"] = ""
                except Exception as e:
                    st.error(str(e))

    with admin_tabs[2]:
        if Product is None:
            st.error("Product model couldn't be imported. Make sure domain/models.py is available.")
        elif not venues_scope:
            st.info("No venues found.")
        else:
            st.markdown("### Venue catalog")
            venue_labels = [f"{v.name} (#{v.id})" for v in venues_scope]
            chosen = st.selectbox("Select venue", venue_labels, index=0, key="adm_cat_venue_sel")
            v = venues_scope[venue_labels.index(chosen)]
            venue_id = int(v.id)

            try:
                dfp = list_products_for_venue(venue_id)
            except Exception as e:
                st.error(str(e))
                dfp = pd.DataFrame()

            st.caption(f"{len(dfp)} products")
            with st.expander("🔁 Add aliases from Excel (Singular / Plural)", expanded=False):
                st.write("Upload an Excel/CSV with **2 columns**: Singular and Plural. Rows will be matched to products by name (preferred) or existing aliases, then both forms will be appended to the product's aliases.")
                up = st.file_uploader("Upload file", type=["xlsx","xls","csv"], key="adm_alias_sp_upload")
                if up is not None:
                    try:
                        if up.name.lower().endswith((".xlsx",".xls")):
                            df_sp = pd.read_excel(up)
                        else:
                            df_sp = pd.read_csv(up)
                        if df_sp.shape[1] < 2:
                            st.error("File must have at least 2 columns (Singular, Plural).")
                        else:
                            # pick columns: prefer names if present, otherwise first two columns
                            cols_lower = [str(c).strip().lower() for c in df_sp.columns]
                            def _pick_col(candidates):
                                for cand in candidates:
                                    if cand in cols_lower:
                                        return df_sp.columns[cols_lower.index(cand)]
                                return None
                            c_sing = _pick_col(["singular","sing","s","sg","singular_name","producto_singular"])
                            c_plur = _pick_col(["plural","plur","p","pl","plural_name","producto_plural"])
                            if c_sing is None or c_plur is None:
                                c_sing, c_plur = df_sp.columns[0], df_sp.columns[1]

                            df_sp = df_sp[[c_sing, c_plur]].rename(columns={c_sing:"singular", c_plur:"plural"})
                            df_sp["singular"] = df_sp["singular"].astype(str).fillna("").map(lambda x: x.strip())
                            df_sp["plural"] = df_sp["plural"].astype(str).fillna("").map(lambda x: x.strip())
                            df_sp = df_sp[(df_sp["singular"] != "") | (df_sp["plural"] != "")]
                            st.dataframe(df_sp.head(25), use_container_width=True, height=260)

                            if st.button("Apply aliases to venue products", type="primary", key="adm_alias_sp_apply"):
                                if len(dfp) == 0:
                                    st.warning("No products in this venue.")
                                else:
                                    # Build match maps
                                    name_to_pid: Dict[str,int] = {}
                                    alias_to_pid: Dict[str,int] = {}
                                    for _, prow in dfp.iterrows():
                                        pid = int(prow.get("id"))
                                        nm = str(prow.get("name") or "")
                                        nm_norm = _norm_name_for_conflict(nm)
                                        if nm_norm:
                                            name_to_pid[nm_norm] = pid
                                        for a in _split_aliases_cell(prow.get("aliases")):
                                            a_norm = _norm_name_for_conflict(a)
                                            if a_norm and a_norm not in alias_to_pid:
                                                alias_to_pid[a_norm] = pid

                                    updated: Dict[int, List[str]] = {}
                                    unmatched: List[Dict[str,str]] = []

                                    def _norm_sp(x: str) -> str:
                                        return _norm_name_for_conflict(x)

                                    for _, rr in df_sp.iterrows():
                                        sing = str(rr.get("singular") or "").strip()
                                        plur = str(rr.get("plural") or "").strip()
                                        candidates = [c for c in [sing, plur] if c]
                                        pid = None
                                        for c in candidates:
                                            n = _norm_sp(c)
                                            if n in name_to_pid:
                                                pid = name_to_pid[n]; break
                                        if pid is None:
                                            for c in candidates:
                                                n = _norm_sp(c)
                                                if n in alias_to_pid:
                                                    pid = alias_to_pid[n]; break

                                        if pid is None:
                                            unmatched.append({"singular": sing, "plural": plur})
                                            continue

                                        updated.setdefault(pid, [])
                                        if sing:
                                            updated[pid].append(sing)
                                        if plur:
                                            updated[pid].append(plur)

                                    # Persist updates
                                    added_total = 0
                                    updated_products = 0
                                    with get_session() as s:
                                        for pid, new_forms in updated.items():
                                            p = s.exec(select(Product).where(Product.id == pid)).first()
                                            if not p:
                                                continue
                                            current = _split_aliases_cell(getattr(p, "aliases", None))
                                            merged = dedupe_keep_order(current + [x for x in new_forms if x])
                                            if merged != current:
                                                setattr(p, "aliases", " | ".join(merged))
                                                s.add(p)
                                                updated_products += 1
                                                added_total += max(0, len(merged) - len(current))
                                        s.commit()

                                    st.success(f"Updated {updated_products} products. Added {added_total} new alias values.")
                                    if unmatched:
                                        st.warning(f"Unmatched rows: {len(unmatched)}")
                                        st.dataframe(pd.DataFrame(unmatched).head(50), use_container_width=True, height=240)
                    except Exception as e:
                        st.error(f"Failed to read/apply file: {e}")

            if len(dfp) == 0:
                st.info("No products yet for this venue.")
            else:
                editable_cols = ["name","description","category","unit","quantity","price","provider_name","aliases"]
                df_edit = dfp[["id"] + editable_cols].copy()

                edited = st.data_editor(
                    df_edit,
                    num_rows="fixed",
                    hide_index=True,
                    use_container_width=True,
                    key="adm_cat_editor",
                )

                if st.button("Save catalog edits", type="primary", key="adm_save_cat"):
                    try:
                        with get_session() as s:
                            before_map = {int(r["id"]): r for _, r in df_edit.iterrows()}
                            for _, r in edited.iterrows():
                                pid = int(r["id"])
                                p = s.exec(select(Product).where(Product.id == pid)).first()
                                if not p:
                                    continue

                                before = dict(before_map.get(pid, {}))
                                after = {c: r.get(c) for c in editable_cols}

                                for c in editable_cols:
                                    val = r.get(c)
                                    if c in {"quantity", "price"}:
                                        try:
                                            val = float(val)
                                        except Exception:
                                            val = float(getattr(p, c) or 0.0)
                                    setattr(p, c, val if val != "" else None)
                                if hasattr(p, "updated_at"):
                                    p.updated_at = datetime.utcnow()

                                s.add(p)

                                if _dumps({k: before.get(k) for k in editable_cols}) != _dumps(after):
                                    log_audit(
                                        action="PRODUCT_EDIT",
                                        entity_type="product",
                                        entity_id=pid,
                                        venue_id=venue_id,
                                        before={k: before.get(k) for k in editable_cols},
                                        after=after,
                                        meta={"source": "admin_catalog_editor"},
                                    )

                            s.commit()

                        log_audit(action="CATALOG_BULK_SAVE", entity_type="product", venue_id=venue_id, meta={"rows": int(len(edited))})
                        st.success("Catalog updated ✅")
                    except Exception as e:
                        st.error(str(e))

    with admin_tabs[3]:
        if Product is None:
            st.error("Product model couldn't be imported. Upload-to-DB requires Product.")
        elif not venues_scope:
            st.info("No venues found.")
        else:
            st.markdown("### Upload products to a venue database")
            venue_labels = [f"{v.name} (#{v.id})" for v in venues_scope]
            chosen = st.selectbox("Target venue", venue_labels, index=0, key="adm_up_venue")
            v = venues_scope[venue_labels.index(chosen)]
            venue_id = int(v.id)

            mode = st.radio("Mode", ["upsert", "append"], index=0, horizontal=True, key="adm_up_mode")

            st.markdown("#### Option A: Use the last generated file from Import Assistant")
            last_df = st.session_state.get("last_ready_df")
            if isinstance(last_df, pd.DataFrame) and len(last_df) > 0:
                st.dataframe(last_df.head(50), use_container_width=True)
                if st.button("Upload this to DB", type="primary", key="adm_up_last"):
                    try:
                        ins, upd = upsert_products_from_df(venue_id, last_df, mode=mode)
                        st.success(f"Upload complete ✅ Inserted: {ins} • Updated: {upd}")
                    except Exception as e:
                        st.error(str(e))
            else:
                st.info("Generate an enriched file in the Import Assistant tab first.")

            st.markdown("#### Option B: Upload an Excel/CSV (must have DB-ready columns)")
            up_file = st.file_uploader("Upload Ready_For_Upload.xlsx or .csv", type=["xlsx", "csv"], key="adm_up_file")

            if up_file is not None:
                try:
                    if up_file.name.lower().endswith(".csv"):
                        df_up = pd.read_csv(up_file)
                    else:
                        xls2, bytes2, did2 = safe_excel_file(up_file)
                        if did2:
                            st.warning("This upload file was auto-repaired for reading.")
                        df_up = pd.read_excel(xls2, sheet_name=0)
                    st.dataframe(df_up.head(50), use_container_width=True)

                    if st.button("Upload file to DB", type="primary", key="adm_up_file_btn"):
                        ins, upd = upsert_products_from_df(venue_id, df_up, mode=mode)
                        st.success(f"Upload complete ✅ Inserted: {ins} • Updated: {upd}")
                except Exception as e:
                    st.error(f"Could not load/upload file: {e}")

    with admin_tabs[4]:
        st.markdown("### Audit Logs")
        st.caption("Who changed what, when. Passwords are never stored or logged.")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            days = st.selectbox("Time range", [1, 3, 7, 14, 30, 90], index=2, key="aud_days")
        with col2:
            action_filter = st.text_input("Action contains", value="", key="aud_action")
        with col3:
            actor_filter = st.text_input("Actor email contains", value="", key="aud_actor")
        with col4:
            venue_filter = st.selectbox(
                "Venue",
                ["(all)"] + [f"{v.name} (#{v.id})" for v in venues_scope],
                index=0,
                key="aud_venue",
            )

        venue_id_filter = None
        if venue_filter != "(all)":
            m = re.search(r"#(\d+)\)", venue_filter)
            if m:
                venue_id_filter = int(m.group(1))

        since = datetime.utcnow() - timedelta(days=int(days))

        with get_auth_session() as s:
            q = select(AuditLog).where(AuditLog.created_at >= since)
            if not is_superadmin(u_admin):
                q = q.where(AuditLog.account_id == acc_id)
            if venue_id_filter is not None:
                q = q.where(AuditLog.venue_id == venue_id_filter)
            if action_filter.strip():
                q = q.where(AuditLog.action.ilike(f"%{action_filter.strip()}%"))
            if actor_filter.strip():
                q = q.where(AuditLog.actor_email.ilike(f"%{actor_filter.strip()}%"))
            q = q.order_by(AuditLog.created_at.desc()).limit(500)
            logs = s.exec(q).all()

        if not logs:
            st.info("No audit logs found for the selected filters.")
        else:
            rows = []
            for l in logs:
                rows.append({
                    "time": l.created_at.isoformat(sep=" ", timespec="seconds"),
                    "actor": l.actor_email,
                    "action": l.action,
                    "entity": f"{l.entity_type}#{l.entity_id}" if l.entity_id else l.entity_type,
                    "account_id": l.account_id,
                    "venue_id": l.venue_id,
                })
            df_logs = pd.DataFrame(rows)
            st.dataframe(df_logs, use_container_width=True, height=320)

            idx = st.number_input("Select row index for details (0..n-1)", min_value=0, max_value=len(logs)-1, value=0, step=1)
            chosen = logs[int(idx)]
            st.markdown("#### Details")
            st.write({
                "created_at": chosen.created_at.isoformat(),
                "actor_email": chosen.actor_email,
                "action": chosen.action,
                "entity_type": chosen.entity_type,
                "entity_id": chosen.entity_id,
                "account_id": chosen.account_id,
                "venue_id": chosen.venue_id,
            })
            cA, cB, cC = st.columns(3)
            with cA:
                st.caption("Before")
                st.code(chosen.before_json or "{}", language="json")
            with cB:
                st.caption("After")
                st.code(chosen.after_json or "{}", language="json")
            with cC:
                st.caption("Meta")
                st.code(chosen.meta_json or "{}", language="json")
