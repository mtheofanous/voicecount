from __future__ import annotations

import time
import hashlib
import re
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import streamlit as st
from sqlmodel import Session, select
from datetime import datetime
from sqlalchemy import desc, func, and_

from domain.models import Product, Order, OrderLine, VenueTranscriptionSettings, ProviderSendStatus
from core.db import get_session
from features.utils.asr_google import asr_google
from features.manage_orders.orders import current_actor
from features.utils.voice_and_orders_utils import *

# ✅ single source of truth for normalization + unit synonyms
from core.normalization import normalize_text, normalize_unit, UNIT_SYNONYMS

# ✅ Router / deep links (matches app.py router)
from core.url_nav import set_query_params


def unit_dropdown_options() -> List[str]:
    """Canonical unit options derived from UNIT_SYNONYMS."""
    return sorted(set(UNIT_SYNONYMS.values()))


def resolve_venue_id(passed_venue_id: Optional[int]) -> int:
    """Prefer session active venue, fallback to passed venue_id."""
    sid = st.session_state.get("active_venue_id")
    if sid:
        return int(sid)
    if passed_venue_id:
        return int(passed_venue_id)
    st.error("No active venue selected")
    st.stop()
    raise RuntimeError("Unreachable")


# ---------------------------
# Catalog helpers (moved to top-level for caching)
# ---------------------------
def build_google_phrases(products_: list[Product]) -> list[str]:
    phrases: list[str] = []
    for p in products_:
        if getattr(p, "name", None):
            phrases.append(str(p.name).strip())
        raw_aliases = (getattr(p, "aliases", "") or "")
        for a in raw_aliases.split("|"):
            a = a.strip()
            if a:
                phrases.append(a)

    seen = set()
    out: list[str] = []
    for x in phrases:
        k = x.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(x.strip())
    return out


@st.cache_data(show_spinner=False, ttl=300)
def load_catalog_and_indexes(venue_id: int, _cache_version: str = "v2"):
    """
    Cached catalog + indexes to avoid reloading on every Streamlit rerun.
    TTL 5 minutes. (You can also add a refresh token to the cache key later.)
    _cache_version param allows manual cache busting when data structure changes.
    """
    with get_session() as s:
        products = s.exec(
            select(Product)
            .where(Product.venue_id == venue_id)
            .order_by(Product.name.asc(), Product.provider_name.asc())
        ).all()

    products_by_id: dict[int, Product] = {int(p.id): p for p in products if getattr(p, "id", None) is not None}

    alias_indexes = build_alias_indexes(products)

    catalog_norm_to_pids: dict[str, list[int]] = {}
    for pid, p in products_by_id.items():
        kn = normalize_text(getattr(p, "name", "") or "")
        if kn:
            catalog_norm_to_pids.setdefault(kn, []).append(int(pid))

    catalog_prompt_names = build_google_phrases(products)

    # Provider lookup maps (used to avoid DB work during finalize)
    pid_to_provider: dict[int, str] = {}
    norm_name_to_provider: dict[str, str] = {}
    for pid, p in products_by_id.items():
        pid_to_provider[int(pid)] = (getattr(p, "provider_name", "") or "")
        k = normalize_text(getattr(p, "name", "") or "")
        if k:
            norm_name_to_provider[k] = (getattr(p, "provider_name", "") or "")

    return (
        products,
        products_by_id,
        alias_indexes,
        catalog_norm_to_pids,
        catalog_prompt_names,
        pid_to_provider,
        norm_name_to_provider,
    )


def get_last_sent_pid_for_venue_among_opts(venue_id: int, opts: list[int]) -> int | None:
    """🕒 Last product among opts that was included in an order that was SENT to its provider (per-provider send)."""
    if not opts:
        return None

    with get_session() as s:
        stmt = (
            select(OrderLine.product_id)
            .join(
                ProviderSendStatus,
                and_(
                    ProviderSendStatus.order_id == OrderLine.order_id,
                    ProviderSendStatus.venue_id == OrderLine.venue_id,
                    ProviderSendStatus.provider_name == OrderLine.provider,
                    ProviderSendStatus.sent == True,
                ),
            )
            .where(OrderLine.venue_id == int(venue_id))
            .where(OrderLine.product_id.in_(opts))
            .order_by(desc(ProviderSendStatus.sent_at), desc(OrderLine.id))
            .limit(1)
        )
        return s.exec(stmt).first()


def get_most_frequent_sent_pid_for_venue_among_opts(venue_id: int, opts: list[int]) -> int | None:
    """🔁 Most frequently SENT product among opts to its provider (per-provider send)."""
    if not opts:
        return None

    with get_session() as s:
        stmt = (
            select(OrderLine.product_id, func.count().label("c"))
            .join(
                ProviderSendStatus,
                and_(
                    ProviderSendStatus.order_id == OrderLine.order_id,
                    ProviderSendStatus.venue_id == OrderLine.venue_id,
                    ProviderSendStatus.provider_name == OrderLine.provider,
                    ProviderSendStatus.sent == True,
                ),
            )
            .where(OrderLine.venue_id == int(venue_id))
            .where(OrderLine.product_id.in_(opts))
            .group_by(OrderLine.product_id)
            .order_by(desc("c"), desc(OrderLine.product_id))
            .limit(1)
        )
        row = s.exec(stmt).first()
        return int(row[0]) if row else None


# ---------------------------
# Provider column enrichment (FAST, no DB, no apply-axis=1)
# ---------------------------
def add_provider_column_fast(
    df: pd.DataFrame,
    *,
    pid_to_provider: Dict[int, str],
    norm_name_to_provider: Dict[str, str],
) -> pd.DataFrame:
    """
    Adds/updates a 'provider' column based on:
      1) matched_product_id -> provider
      2) fallback: matched_name (normalized) -> provider
    Vectorized (much faster than df.apply(axis=1)).
    """
    out = df.copy()
    if "provider" not in out.columns:
        out["provider"] = ""

    if "matched_product_id" in out.columns:
        out["provider"] = out["matched_product_id"].map(pid_to_provider).fillna("")

    if "matched_name" in out.columns:
        missing = out["provider"] == ""
        if missing.any():
            out.loc[missing, "provider"] = (
                out.loc[missing, "matched_name"]
                .map(lambda x: norm_name_to_provider.get(normalize_text(x), ""))
                .fillna("")
            )

    out["provider"] = out["provider"].astype("string")
    return out


def apply_unit_choice(df: pd.DataFrame) -> pd.DataFrame:
    """
    If unit == 'Other…', use unit_custom; then normalize units.
    Produces a clean 'unit' column (canonical).
    """
    out = df.copy()
    if "unit" not in out.columns:
        out["unit"] = ""
    if "unit_custom" not in out.columns:
        out["unit_custom"] = ""

    def _final_unit(row) -> str:
        u = safe_str(row.get("unit")).strip()
        if u == "Other…":
            u = safe_str(row.get("unit_custom")).strip()
        return normalize_unit(u) or "unit"

    out["unit"] = out.apply(_final_unit, axis=1)
    return out


# ---------------------------
# Alias utilities (your existing)
# ---------------------------
def _split_aliases_cell(cell: str) -> List[str]:
    raw = str(cell or "").replace("\n", " ")
    parts: List[str] = []
    for chunk in raw.split("|"):
        for sub in re.split(r"[,;]+", chunk):
            v = normalize_text(sub)
            if v:
                parts.append(v)
    seen = set()
    out: List[str] = []
    for a in parts:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def build_alias_indexes(products: List[Product]) -> Dict[str, Any]:
    """
    Build alias indexes from Product.aliases.

    Returns:
        {
          'alias_to_pids': dict(alias_norm -> [product_id...]),
          'token_to_pids': dict(token -> set(product_id...)),
          'alias_to_products': dict(alias_norm -> [product_name...])  (used for fragment splitting)
        }
    """
    alias_to_pids: Dict[str, List[int]] = {}
    token_to_pids: Dict[str, set[int]] = {}
    alias_to_products: Dict[str, List[str]] = {}

    for p in products:
        pid_raw = getattr(p, "id", None)
        if pid_raw is None:
            continue
        pid = int(pid_raw)

        aliases = _split_aliases_cell(getattr(p, "aliases", "") or "")
        aliases.append(normalize_text(getattr(p, "name", "") or ""))

        prov = normalize_text(getattr(p, "provider_name", "") or "")
        if prov:
            aliases.append(prov)

        for a in aliases:
            a = normalize_text(a)
            if not a:
                continue

            alias_to_pids.setdefault(a, [])
            if pid not in alias_to_pids[a]:
                alias_to_pids[a].append(pid)

            alias_to_products.setdefault(a, [])
            if p.name not in alias_to_products[a]:
                alias_to_products[a].append(p.name)

            for tok in a.split():
                if len(tok) < 2:
                    continue
                token_to_pids.setdefault(tok, set()).add(pid)

    return {
        "alias_to_pids": alias_to_pids,
        "token_to_pids": token_to_pids,
        "alias_to_products": alias_to_products,
    }


def alias_suggestions(
    query_norm: str,
    *,
    alias_to_pids: Dict[str, List[int]],
    token_to_pids: Dict[str, set[int]],
    limit: int = 20,
) -> List[int]:
    """Return candidate product IDs using alias indexes (ID-safe)."""
    q = normalize_text(query_norm)
    if not q:
        return []

    exact = alias_to_pids.get(q, [])
    if exact:
        return exact[:limit]

    toks = [t for t in q.split() if len(t) >= 2]
    if not toks:
        return []

    scores: Dict[int, int] = {}
    for t in toks:
        for pid in token_to_pids.get(t, set()):
            scores[pid] = scores.get(pid, 0) + 1

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [pid for (pid, _s) in ranked[:limit]]


# =========================================================
# MAIN TAB
# =========================================================
def new_order_tab(venue_id: int, role: str | None = None) -> None:
    """
    1-screen mobile flow: Notas de faltantes → Pedido en preparación (draft)
    - Timeline notes (voice + typed)
    - Auto-parse in the same screen
    - Fix ambiguities inline
    - Save = add to active draft if exists, else create draft
    """

    # ---------------------------
    # Venue + namespacing
    # ---------------------------
    venue_id = resolve_venue_id(venue_id)
    NS = f"new_order_{venue_id}_"

    def K(name: str) -> str:
        return NS + name

    def S(name: str) -> str:
        return NS + name

    # ---------------------------
    # Session state init
    # ---------------------------
    st.session_state.setdefault(S("transcript_area"), "")
    st.session_state.setdefault(S("last_audio_hash"), None)
    st.session_state.setdefault(S("audio_widget_key"), f"{NS}asr_audio_in_{int(time.time())}")
    st.session_state.setdefault(S("audiorecorder_key"), f"{NS}audiorecorder_{int(time.time())}")
    st.session_state.setdefault(S("order_chat"), [])  # [{"role":"user"|"asr","text":str,"ts":float}]
    st.session_state.setdefault(S("auto_parse_pending"), False)
    st.session_state.setdefault(S("resolved_picks"), {})  # item_key -> picked_product_id
    st.session_state.setdefault(S("lang_code_ui"), None)  # UI-only language override

    # ---------------------------
    # Helpers
    # ---------------------------
    def reset_notes_only(clear_resolved_picks: bool = False, *, do_rerun: bool = True) -> None:
        """Clear the notes + parse output, keep venue context.
        If clear_resolved_picks=True, also clears ambiguity memory.
        """
        st.session_state[S("transcript_area")] = ""
        st.session_state[S("last_audio_hash")] = None
        st.session_state[S("order_chat")] = []
        st.session_state[S("auto_parse_pending")] = False

        # Clear parse outputs
        st.session_state.pop(S("parsed_df"), None)
        st.session_state.pop(S("parse_candidates_df"), None)
        st.session_state.pop(S("finalize_parse_pending"), None)
        st.session_state.pop(K("parse_editor"), None)

        # Optional: clear ambiguity resolutions
        if clear_resolved_picks:
            st.session_state.pop(S("resolved_picks"), None)

        # Clear UI language override too (so we return to venue defaults)
        st.session_state.pop(S("lang_code_ui"), None)

        # Reset audio widgets
        st.session_state[S("audio_widget_key")] = f"{NS}asr_audio_in_{int(time.time())}"
        st.session_state[S("audiorecorder_key")] = f"{NS}audiorecorder_{int(time.time())}"

        if do_rerun:
            st.rerun()

    def bump_orders_refresh_token() -> None:
        k = f"orders_refresh_token_{venue_id}"
        st.session_state[k] = int(st.session_state.get(k, 0)) + 1

    def _rebuild_transcript_from_chat() -> None:
        st.session_state[S("transcript_area")] = "\n".join(
            (m.get("text") or "").strip()
            for m in st.session_state[S("order_chat")]
            if (m.get("text") or "").strip()
        ).strip()

    def append_message(role_: str, text_: str) -> None:
        text_ = (text_ or "").strip()
        if not text_:
            return

        st.session_state[S("order_chat")].append({"role": role_, "text": text_, "ts": time.time()})
        _rebuild_transcript_from_chat()

        # auto-parse after every change
        st.session_state[S("auto_parse_pending")] = True
        st.session_state.pop(S("parsed_df"), None)
        st.session_state.pop(S("parse_candidates_df"), None)
        st.session_state.pop(S("finalize_parse_pending"), None)

    # ---------------------------
    # Draft targeting (shared key with orders.py)
    # ---------------------------
    ACTIVE_DRAFT_KEY = f"orders_active_order_id_{venue_id}"
    active_draft_id = st.session_state.get(ACTIVE_DRAFT_KEY)
    try:
        active_draft_id = int(active_draft_id) if active_draft_id is not None else None
    except Exception:
        active_draft_id = None

    # ✅ Validate that active_draft_id is actually a DRAFT in DB
    if active_draft_id:
        with get_session() as s:
            o = s.exec(
                select(Order).where(Order.venue_id == int(venue_id), Order.id == int(active_draft_id))
            ).first()
        o_status = (getattr(o, "status", "") or "").lower() if o else ""
        if o_status != "draft":
            st.session_state.pop(ACTIVE_DRAFT_KEY, None)
            active_draft_id = None

    def _set_active_draft(order_id: int) -> None:
        st.session_state[ACTIVE_DRAFT_KEY] = int(order_id)

    def _go_orders(order_id: int) -> None:
        # matches app.py router: ?page=orders&order_id=...&status=draft
        st.session_state["page"] = "orders"
        set_query_params(page="orders", order_id=int(order_id), status="draft")
        st.rerun()

    # =========================================================
    # 1) ASR CONFIG (READ-ONLY, per venue)
    # =========================================================
    ASR_CFG_TTL_SECONDS = 90
    defaults = {
        "asr_backend": "Google Speech-to-Text",
        "lang_code": "auto",
        "samplerate": 22050,
        "hide_user_controls": True,
    }

    cfg_key = S("asr_cfg")
    cfg_ts_key = S("asr_cfg_ts")
    now = time.time()

    cfg_cached = st.session_state.get(cfg_key)
    ts = float(st.session_state.get(cfg_ts_key) or 0.0)
    needs_refresh = (not isinstance(cfg_cached, dict)) or (now - ts > ASR_CFG_TTL_SECONDS)

    if needs_refresh:
        cfg_data = defaults.copy()
        with get_session() as s:
            cfg = s.exec(
                select(VenueTranscriptionSettings).where(VenueTranscriptionSettings.venue_id == int(venue_id))
            ).first()
            if cfg:
                cfg_data["asr_backend"] = getattr(cfg, "asr_backend", None) or cfg_data["asr_backend"]
                cfg_data["lang_code"] = getattr(cfg, "lang_code", None) or cfg_data["lang_code"]
                try:
                    cfg_data["samplerate"] = int(getattr(cfg, "samplerate", cfg_data["samplerate"]))
                except Exception:
                    cfg_data["samplerate"] = defaults["samplerate"]
                cfg_data["hide_user_controls"] = bool(getattr(cfg, "hide_user_controls", True))

        st.session_state[cfg_key] = cfg_data
        st.session_state[cfg_ts_key] = now
        cfg_cached = cfg_data

    asr_backend = cfg_cached.get("asr_backend", defaults["asr_backend"])
    lang_code = cfg_cached.get("lang_code", defaults["lang_code"])
    try:
        samplerate = int(cfg_cached.get("samplerate", defaults["samplerate"]))
    except Exception:
        samplerate = defaults["samplerate"]

    if bool(cfg_cached.get("hide_user_controls", True)):
        st.session_state.pop(S("lang_code_ui"), None)

    # =========================================================
    # 2) LOAD CATALOG + indexes (CACHED)
    # =========================================================
    (
        products,
        products_by_id,
        alias_indexes,
        catalog_norm_to_pids,
        catalog_prompt_names,
        pid_to_provider,
        norm_name_to_provider,
    ) = load_catalog_and_indexes(venue_id)

    if not products:
        st.warning("Primero crea tu catálogo en la pestaña 'Catálogo'.")
        return

    # Validate alias_indexes structure (in case of cache corruption)
    if not isinstance(alias_indexes, dict):
        st.error("Error en la estructura de datos del catálogo. Limpiando caché...")
        st.cache_data.clear()
        st.rerun()
        return
    
    required_keys = ["alias_to_pids", "token_to_pids", "alias_to_products"]
    missing_keys = [k for k in required_keys if k not in alias_indexes]
    if missing_keys:
        st.error(f"Estructura de catálogo incompleta (faltan: {missing_keys}). Limpiando caché...")
        st.cache_data.clear()
        st.rerun()
        return

    alias_to_pids = alias_indexes["alias_to_pids"]
    token_to_pids = alias_indexes["token_to_pids"]
    alias_to_products = alias_indexes["alias_to_products"]

    catalog_names = list(catalog_norm_to_pids.keys())

    # =========================================================
    # 3) MOBILE-FIRST STYLES
    # =========================================================
    st.markdown(
        """
        <style>
        .chat-bubble {
        display: inline-block;
        padding: 10px 12px;
        border-radius: 14px;
        margin: 4px 0;
        max-width: 92%;
        line-height: 1.35;
        font-size: 0.98rem;
        word-wrap: break-word;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
        }
        .bubble-user { background: #DCF8C6; border-top-right-radius: 7px; }
        .bubble-asr { background: #FFFFFF; border-top-left-radius: 7px; }

        .row-pill {
        display: inline-block;
        padding: 8px 10px;
        border-radius: 14px;
        margin: 6px 0;
        width: 100%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        font-size: 0.98rem;
        }

        .notes-wrap {
        display: flex;
        flex-direction: column;
        gap: 10px;
        }

        .note-row {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        }

        .note-bubble {
        flex: 1;
        border-radius: 16px;
        padding: 10px 12px;
        line-height: 1.35;
        word-break: break-word;
        border: 1px solid rgba(49, 51, 63, 0.18);
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        background: #fff;
        }

        .note-user {
        background: #DCF8C6;
        border-top-right-radius: 7px;
        }

        .note-asr {
        background: #FFFFFF;
        border-top-left-radius: 7px;
        }

        .note-meta {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 10px;
            font-size: 0.80rem;
        }

        .note-text {
            flex: 1;
            word-break: break-word;
        }

        .note-time {
            white-space: nowrap;
            font-size: 0.72rem;
            opacity: 0.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- WhatsApp-like redesign (design-only) ---
    st.markdown(
        """
    <style>
    /* Google Font for handwriting */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* ---------- Unified Notebook Design ---------- */
    .voi-notebook-unified {
      position: relative;
      background: #fef9e7;
      border: 1px solid #d4af37;
      border-bottom: none;
      border-left: 3px solid #c41e3a;
      border-radius: 4px 4px 0 0;
      padding: 16px 16px 12px 40px;
      font-family: 'Inter', sans-serif;
    }
    
    .voi-notebook-unified::before {
      content: '';
      position: absolute;
      left: 32px;
      top: 0;
      bottom: 0;
      width: 2px;
      background: #c41e3a;
      opacity: 0.5;
    }
    
    .voi-notebook-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 12px;
      border-bottom: 1px dashed rgba(0,0,0,0.15);
    }
    
    .voi-notebook-title {
      font-family: 'Inter', sans-serif;
      font-size: 1.3rem;
      font-weight: 600;
      color: #1a1a1a;
    }
    
    .voi-notebook-time {
      font-family: 'Inter', sans-serif;
      font-size: 1.1rem;
      font-weight: 500;
      color: #4a4a4a;
    }
    
    .voi-notebook-products {
      position: relative;
      background: linear-gradient(
        to bottom,
        #fef9e7 0%,
        #fef9e7 calc(100% - 1.5rem),
        transparent calc(100% - 1.5rem)
      ),
      repeating-linear-gradient(
        transparent,
        transparent 1.4rem,
        #b8b8b8 1.4rem,
        #b8b8b8 1.45rem
      );
      background-color: #fef9e7;
      border: 1px solid #d4af37;
      border-top: none;
      border-bottom: none;
      border-left: 3px solid #c41e3a;
      padding: 8px 16px 8px 40px;
    }
    
    .voi-notebook-products::before {
      content: '';
      position: absolute;
      left: 32px;
      top: 0;
      bottom: 0;
      width: 2px;
      background: #c41e3a;
      opacity: 0.5;
    }
    
    /* Make buttons look integrated */
    .voi-notebook-products button {
      font-family: 'Inter', sans-serif !important;
      font-size: 1.3rem !important;
      font-weight: 700 !important;
      padding: 2px 8px !important;
      min-width: 32px !important;
      height: 32px !important;
      border-radius: 4px !important;
      margin-top: 6px;
    }
    
    .voi-notebook-footer {
      position: relative;
      background: #fef9e7;
      border: 1px solid #d4af37;
      border-top: none;
      border-left: 3px solid #c41e3a;
      border-radius: 0 0 4px 4px;
      padding: 12px 16px 16px 40px;
      display: flex;
      align-items: center;
      gap: 10px;
      font-family: 'Inter', sans-serif;
      font-size: 1.15rem;
      font-weight: 500;
      color: #4a4a4a;
      box-shadow: 0 4px 12px rgba(0,0,0,0.08);
      margin-bottom: 14px;
    }
    
    .voi-notebook-footer::before {
      content: '';
      position: absolute;
      left: 32px;
      top: 0;
      bottom: 0;
      width: 2px;
      background: #c41e3a;
      opacity: 0.5;
    }
    
    .voi-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #f59e0b;
      box-shadow: 0 0 4px rgba(245,158,11,0.5);
      flex-shrink: 0;
    }

    /* ---------- Bottom WhatsApp-like bar ---------- */
    .voi-bottom-wrap{
      position: fixed;
      left: 0; right: 0;
      bottom: calc(62px + env(safe-area-inset-bottom));
      z-index: 9998;

      padding: 10px 12px calc(10px + env(safe-area-inset-bottom));
      background: rgba(255,255,255,.96);
      border-top: 1px solid rgba(148,163,184,.30);
      backdrop-filter: saturate(180%) blur(12px);
    }
    .voi-bottom-inner{
      max-width: 1100px;
      margin: 0 auto;
      display:flex;
      flex-direction:column;
      gap:10px;
    }
    .voi-bottom-inner [data-testid="stTextInput"] input{
      border-radius:16px !important;
      background:#f1f5f9 !important;
      border:1px solid rgba(148,163,184,.35) !important;
      padding: 12px 14px !important;
      font-weight:800 !important;
    }
    .voi-bottom-inner [data-testid="stTextInput"] label{display:none !important;}

    .voi-btnrow{
      display:flex; gap:10px;
    }
    .voi-btnrow > div{flex: 1;}
    .voi-bottom-inner .stVerticalBlock{gap:8px !important;}
    /* Compact the audio input container a bit */
    [data-testid="stAudioInput"]{
      border:1px solid rgba(148,163,184,.35);
      border-radius:14px;
      padding:6px 10px;
      background:#ffffff;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # =========================================================
    # HEADER (notes mindset) + Active draft display
    # =========================================================
    
    st.subheader("📝 Notas de faltantes")

            
    # =========================================================
    # Timeline (living notes) — mobile friendly + aligned
    # =========================================================
    # =========================================================
    # Timeline (living notes) — mobile friendly + aligned
    # =========================================================
    # with st.container(border=True):
    #     chat = st.session_state.get(S("order_chat"), [])

    #     if not chat:
    #         st.info("Empieza escribiendo abajo o graba un audio 👇")
    #     else:
    #         st.markdown('<div class="notes-wrap">', unsafe_allow_html=True)

    #         for i, msg in enumerate(list(chat)):
    #             role_msg = safe_str(msg.get("role"))
    #             txt = safe_str(msg.get("text", ""))
    #             tsf = float(msg.get("ts") or 0.0)

    #             who = "Tú" if role_msg == "user" else "Audio"
    #             cls = "note-user" if role_msg == "user" else "note-asr"
    #             tlabel = time.strftime("%H:%M", time.localtime(tsf)) if tsf else ""

    #             bubble_col, del_col = st.columns([20, 2], vertical_alignment="top")

    #             with bubble_col:
    #                 st.markdown(
    #                     f"""
    #                     <div class="note-row">
    #                     <div class="note-bubble {cls}">
    #                         <div class="note-meta">
    #                         <div class="note-text"><strong>{who}:</strong> {txt}</div>
    #                         <div class="note-time">{tlabel}</div>
    #                         </div>
    #                     </div>
    #                     </div>
    #                     """,
    #                     unsafe_allow_html=True,
    #                 )

    #             with del_col:
    #                 msg_key = f"{int(tsf * 1000)}" if tsf else f"idx_{i}"
    #                 if st.button(
    #                     "🗑️",
    #                     key=K(f"del_msg_{msg_key}"),
    #                     help="Eliminar esta nota",
    #                     type="secondary",
    #                 ):
    #                     st.session_state[S("order_chat")].pop(i)
    #                     _rebuild_transcript_from_chat()

    #                     st.session_state[S("auto_parse_pending")] = True
    #                     st.session_state.pop(S("parsed_df"), None)
    #                     st.session_state.pop(S("parse_candidates_df"), None)
    #                     st.session_state.pop(S("finalize_parse_pending"), None)
    #                     st.rerun()

    #         st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================
    # Convert df -> order lines
    # =========================================================
    def _df_to_orderline_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if df is None or df.empty:
            return rows

        for _, r in df.iterrows():
            pid = r.get("matched_product_id", None)
            try:
                if pid is None or pd.isna(pid):
                    continue
                pid_i = int(pid)
            except Exception:
                continue

            try:
                qty = float(r.get("quantity") or 0.0)
            except Exception:
                qty = 0.0
            if qty <= 0:
                continue

            unit = str(r.get("unit") or "unit").strip().lower() or "unit"
            rows.append({"product_id": pid_i, "quantity": qty, "unit": unit})
        return rows


    def _create_draft_and_insert_lines(*, venue_id: int, actor: str, df: pd.DataFrame) -> int:
        rows = _df_to_orderline_rows(df)
        if not rows:
            return 0

        with get_session() as s:
            o = Order(
                venue_id=int(venue_id),
                status="draft",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                created_by=actor,
                updated_by=actor,
                title=None,
                note=None,
            )
            s.add(o)
            s.commit()
            s.refresh(o)
            order_id = int(o.id)

            for it in rows:
                pid = int(it["product_id"])
                qty = float(it["quantity"])
                unit = str(it["unit"])

                prod = s.exec(select(Product).where(Product.venue_id == int(venue_id), Product.id == pid)).first()
                provider_name = (getattr(prod, "provider_name", "") or "") if prod else ""

                s.add(
                    OrderLine(
                        venue_id=int(venue_id),
                        order_id=int(order_id),
                        product_id=pid,
                        quantity=qty,
                        unit=unit,
                        provider=provider_name or None,
                        updated_at=datetime.utcnow(),
                        updated_by=actor,
                    )
                )

            o.updated_at = datetime.utcnow()
            o.updated_by = actor
            s.add(o)
            s.commit()

        return order_id


    def _add_lines_to_existing_draft(*, venue_id: int, order_id: int, actor: str, df: pd.DataFrame) -> None:
        rows = _df_to_orderline_rows(df)
        if not rows:
            return

        with get_session() as s:
            existing = s.exec(
                select(OrderLine).where(
                    OrderLine.venue_id == int(venue_id),
                    OrderLine.order_id == int(order_id),
                )
            ).all()
            by_pid = {int(getattr(ln, "product_id", 0) or 0): ln for ln in existing}

            for it in rows:
                pid = int(it["product_id"])
                qty = float(it["quantity"])
                unit = str(it["unit"])

                prod = s.exec(select(Product).where(Product.venue_id == int(venue_id), Product.id == pid)).first()
                provider_name = (getattr(prod, "provider_name", "") or "") if prod else ""

                if pid in by_pid and by_pid[pid] is not None:
                    ln = by_pid[pid]
                    ln.quantity = float(getattr(ln, "quantity", 0.0) or 0.0) + qty
                    ln.unit = unit or (getattr(ln, "unit", None) or "unit")
                    if provider_name:
                        ln.provider = provider_name
                    ln.updated_at = datetime.utcnow()
                    ln.updated_by = actor
                    s.add(ln)
                else:
                    s.add(
                        OrderLine(
                            venue_id=int(venue_id),
                            order_id=int(order_id),
                            product_id=pid,
                            quantity=qty,
                            unit=unit,
                            provider=provider_name or None,
                            updated_at=datetime.utcnow(),
                            updated_by=actor,
                        )
                    )

            o = s.exec(select(Order).where(Order.id == int(order_id))).first()
            if o:
                o.updated_at = datetime.utcnow()
                o.updated_by = actor
                s.add(o)

            s.commit()


    def finalize_candidates_to_df(
        candidates_df: pd.DataFrame,
        *,
        products_by_id: dict[int, Product],
        venue_id: int,
    ) -> pd.DataFrame:
        """
        Aggregate parse candidates (sum quantities of same product), normalize units,
        add provider column and return final df ready for editor / saving.
        """
        df_in = candidates_df.copy()

        merged: dict[object, dict[str, object]] = {}
        for _, r in df_in.iterrows():
            pid = r.get("matched_product_id", None)
            pid_i: int | None = None
            try:
                if pid is not None and not pd.isna(pid):
                    pid_i = int(pid)
            except Exception:
                pid_i = None

            matched_name = safe_str(r.get("matched_name")).strip()
            if pid_i is not None and pid_i in products_by_id:
                matched_name = safe_str(getattr(products_by_id[pid_i], "name", "") or matched_name).strip()

            qty = float(r.get("quantity") or 0.0) if r.get("quantity") is not None else 0.0
            unit_val = normalize_unit(r.get("unit")) or "unit"
            conf = float(r.get("confidence") or 0.0)
            spoken = safe_str(r.get("spoken_name")).strip()
            unit_custom = safe_str(r.get("unit_custom"))

            if (pid_i is None) and (not matched_name) and qty == 0.0 and not unit_val:
                continue

            key = ("pid", pid_i) if pid_i is not None else ("spoken", normalize_text(spoken))

            if key not in merged:
                merged[key] = {
                    "spoken_name": spoken,
                    "matched_product_id": pid_i,
                    "matched_name": matched_name or None,
                    "confidence": conf,
                    "quantity": qty,
                    "unit": unit_val,
                    "unit_custom": unit_custom,
                    "status": "OK" if (pid_i is not None or matched_name) else "Revisar",
                }
            else:
                merged[key]["quantity"] = float(merged[key]["quantity"] or 0.0) + qty
                merged[key]["confidence"] = max(float(merged[key]["confidence"] or 0.0), conf)

                if not safe_str(merged[key].get("unit_custom")) and unit_custom:
                    merged[key]["unit_custom"] = unit_custom

                if spoken and spoken not in (merged[key].get("spoken_name") or ""):
                    merged[key]["spoken_name"] = (str(merged[key]["spoken_name"]) + " | " + spoken).strip(" | ")

        df = pd.DataFrame(list(merged.values()))
        if "unit_custom" not in df.columns:
            df["unit_custom"] = ""
        if "matched_product_id" not in df.columns:
            df["matched_product_id"] = pd.NA

        # ✅ Fast provider enrichment without DB calls
        df = add_provider_column_fast(df, pid_to_provider=pid_to_provider, norm_name_to_provider=norm_name_to_provider)
        return df


    # =========================================================
    # Inline ambiguity resolver (still 1-screen)
    # =========================================================
    candidates_df = st.session_state.get(S("parse_candidates_df"))
    finalize_pending = st.session_state.get(S("finalize_parse_pending"), False)

    resolved_picks: dict[str, int] = st.session_state.setdefault(S("resolved_picks"), {})

    if finalize_pending and isinstance(candidates_df, pd.DataFrame) and not candidates_df.empty:
        def _is_unresolved_ambiguous(row) -> bool:
            if safe_str(row.get("status")) != "Elegir":
                return False
            sugg = row.get("suggestions")
            if not (isinstance(sugg, list) and len(sugg) > 0):
                return False
            item_key = safe_str(row.get("item_key")).strip()
            if item_key and item_key in resolved_picks:
                return False
            return True

        needs_df = candidates_df[candidates_df.apply(_is_unresolved_ambiguous, axis=1)].copy()

        if not needs_df.empty:
            with st.container(border=True):
                st.markdown("### 🔎 Elige productos (solo ambiguos)")
                st.caption("Toca una opción por línea y pulsa **OK**.")

                picks: dict[int, int] = {}
                pick_keys: dict[int, str] = {}

                for idx, row in needs_df.iterrows():
                    item_key = safe_str(row.get("item_key")).strip()
                    spoken = safe_str(row.get("spoken_name"))
                    qty = row.get("quantity")
                    unit = safe_str(row.get("unit") or "unit")

                    opts: list[int] = [int(x) for x in (row.get("suggestions") or []) if x is not None]
                    if not opts:
                        continue

                    def _price(pid: int) -> float:
                        p = products_by_id.get(int(pid))
                        price = getattr(p, "price", None)
                        return float(price) if price is not None else float("inf")

                    cheapest_pid = min(opts, key=_price) if opts else None
                    if cheapest_pid is not None and _price(cheapest_pid) == float("inf"):
                        cheapest_pid = None

                    last_sent_pid = get_last_sent_pid_for_venue_among_opts(venue_id=venue_id, opts=opts)
                    most_freq_pid = get_most_frequent_sent_pid_for_venue_among_opts(venue_id=venue_id, opts=opts)

                    def _opt_label(pid: int) -> str:
                        p = products_by_id.get(int(pid))
                        if not p:
                            return f"#{int(pid)}"

                        name2 = (getattr(p, "name", "") or "").strip()
                        prov2 = (getattr(p, "provider_name", "") or "").strip()
                        desc2 = (getattr(p, "description", "") or "").strip()
                        price2 = getattr(p, "price", None)
                        price_txt = f"{float(price2):.2f}€" if price2 is not None else "—"

                        core = name2 + (f" — {desc2}" if desc2 else "")
                        meta = " · ".join([x for x in [prov2, price_txt] if x])
                        label2 = f"{core} · {meta}" if meta else core

                        badges = []
                        if cheapest_pid is not None and int(pid) == int(cheapest_pid):
                            badges.append("💰")
                        if last_sent_pid is not None and int(pid) == int(last_sent_pid):
                            badges.append("🕒")
                        if most_freq_pid is not None and int(pid) == int(most_freq_pid):
                            badges.append("🔥")

                        badge_txt = ("".join(badges) + " ") if badges else ""
                        return badge_txt + label2

                    default_pid = (
                        int(last_sent_pid)
                        if (last_sent_pid is not None and int(last_sent_pid) in opts)
                        else (
                            int(cheapest_pid)
                            if (cheapest_pid is not None and int(cheapest_pid) in opts)
                            else int(opts[0])
                        )
                    )
                    default_index = opts.index(int(default_pid))

                    pick_pid = st.selectbox(
                        f"{spoken} — {qty or ''} {unit}".strip(),
                        options=opts,
                        index=int(default_index),
                        format_func=_opt_label,
                        key=K(f"resolve_pick_{item_key or idx}"),
                    )

                    picks[int(idx)] = int(pick_pid)
                    if item_key:
                        pick_keys[int(idx)] = item_key

                if st.button("✅ OK", type="primary", key=K("btn_finalize_parse")):
                    df2 = candidates_df.copy()

                    for idx, pick_pid in picks.items():
                        df2.at[idx, "matched_product_id"] = int(pick_pid)
                        p = products_by_id.get(int(pick_pid))
                        df2.at[idx, "matched_name"] = (p.name if p else None)
                        df2.at[idx, "status"] = "OK"
                        df2.at[idx, "confidence"] = max(float(df2.at[idx, "confidence"] or 0.0), 99.0)

                    for idx, pick_pid in picks.items():
                        key = pick_keys.get(int(idx))
                        if key:
                            resolved_picks[str(key)] = int(pick_pid)

                    st.session_state[S("resolved_picks")] = resolved_picks
                    st.session_state[S("parse_candidates_df")] = df2
                    st.session_state[S("finalize_parse_pending")] = False

                    st.session_state[S("parsed_df")] = finalize_candidates_to_df(
                        df2,
                        products_by_id=products_by_id,
                        venue_id=venue_id,
                    )
                    st.rerun()


    # =========================================================
    # Auto-parse engine (runs in same screen)
    # =========================================================
    def _parse_chat_to_candidates() -> pd.DataFrame:
        chat = st.session_state.get(S("order_chat"), []) or []
        if not chat:
            return pd.DataFrame([])

        resolved_picks: Dict[str, int] = st.session_state.setdefault(S("resolved_picks"), {})

        items: List[Dict[str, Any]] = []
        for m in chat:
            msg_text = (m.get("text") or "").strip()
            if not msg_text:
                continue

            msg_ts = float(m.get("ts") or 0.0)

            msg_items: List[str] = []
            for raw in tokenize_items(msg_text):
                msg_items.extend(split_fragment_by_catalog_aliases(raw, alias_to_products))

            for j, frag in enumerate(msg_items):
                frag_clean = (frag or "").strip()
                if not frag_clean:
                    continue
                items.append({"chat_ts": msg_ts, "item_idx": j, "frag": frag_clean})

        if not items:
            return pd.DataFrame([])

        parsed_rows: List[Dict[str, Any]] = []

        LOW_CONFIDENCE_SCORE = 80.0
        TOP_CHOICES = 10

        for it in items:
            frag = it["frag"]
            item_key = f'{it["chat_ts"]}:{it["item_idx"]}:{normalize_text(frag)}'

            parsed = parse_item(frag)
            if not parsed:
                parsed_rows.append({
                    "item_key": item_key,
                    "chat_ts": it["chat_ts"],
                    "item_idx": it["item_idx"],
                    "spoken_name": frag,
                    "matched_product_id": None,
                    "matched_name": None,
                    "confidence": 0.0,
                    "quantity": None,
                    "unit": "unit",
                    "unit_custom": "",
                    "suggestions": [],
                    "recommended_pid": None,
                    "status": "No interpretado",
                })
                continue

            name, qty, unit_raw = parsed
            name_norm = normalize_text(name)

            suggestions_pids: List[int] = []
            prod: Optional[Product] = None
            score: float = 0.0
            matched_pid: Optional[int] = None

            picked_pid = resolved_picks.get(item_key)
            if picked_pid is not None:
                try:
                    picked_pid_i = int(picked_pid)
                except Exception:
                    picked_pid_i = None

                if picked_pid_i is not None and picked_pid_i in products_by_id:
                    prod = products_by_id.get(picked_pid_i)
                    matched_pid = picked_pid_i
                    score = 99.0
                    suggestions_pids = []

            if prod is None:
                name_sing_es = singularize_es(name_norm)
                name_sing_el = singularize_el(name_norm)

                alias_keys = [name_norm]
                if name_sing_es and name_sing_es != name_norm:
                    alias_keys.append(name_sing_es)
                if name_sing_el and name_sing_el != name_norm:
                    alias_keys.append(name_sing_el)

                exact_hits_pids: List[int] = []
                for k in alias_keys:
                    hits = alias_to_pids.get(k, [])
                    if hits:
                        exact_hits_pids = [int(x) for x in hits]
                        break

                if exact_hits_pids:
                    if len(exact_hits_pids) == 1:
                        matched_pid = int(exact_hits_pids[0])
                        prod = products_by_id.get(matched_pid)
                        score = 100.0 if prod else 0.0
                    else:
                        suggestions_pids = exact_hits_pids[:TOP_CHOICES]

                if prod is None and not suggestions_pids:
                    match_norm, score2 = fuzzy_match(name_norm, catalog_names)
                    if match_norm:
                        score = float(score2 or 0.0)
                        pids = catalog_norm_to_pids.get(match_norm, [])

                        is_generic_query = (len(name_norm.split()) == 1 and len(name_norm) <= 10)
                        prefix_hits = [cn for cn in catalog_names if cn == name_norm or cn.startswith(name_norm + " ")]

                        if is_generic_query and len(prefix_hits) >= 2:
                            top = suggest_matches(name_norm, catalog_names, limit=TOP_CHOICES)
                            seen = set()
                            for cand_norm, _sc in top:
                                for pid in catalog_norm_to_pids.get(cand_norm, []):
                                    pid_i = int(pid)
                                    if pid_i not in seen:
                                        suggestions_pids.append(pid_i)
                                        seen.add(pid_i)
                            suggestions_pids = suggestions_pids[:TOP_CHOICES]

                        elif len(pids) == 1:
                            if score < LOW_CONFIDENCE_SCORE:
                                top = suggest_matches(name_norm, catalog_names, limit=TOP_CHOICES)
                                seen = set()
                                for cand_norm, _sc in top:
                                    for pid in catalog_norm_to_pids.get(cand_norm, []):
                                        pid_i = int(pid)
                                        if pid_i not in seen:
                                            suggestions_pids.append(pid_i)
                                            seen.add(pid_i)
                                suggestions_pids = suggestions_pids[:TOP_CHOICES]
                            else:
                                matched_pid = int(pids[0])
                                prod = products_by_id.get(matched_pid)

                        elif len(pids) > 1:
                            top = suggest_matches(name_norm, catalog_names, limit=TOP_CHOICES)
                            seen = set()
                            for cand_norm, _sc in top:
                                for pid in catalog_norm_to_pids.get(cand_norm, []):
                                    pid_i = int(pid)
                                    if pid_i not in seen:
                                        suggestions_pids.append(pid_i)
                                        seen.add(pid_i)
                            for pid in pids:
                                pid_i = int(pid)
                                if pid_i not in seen:
                                    suggestions_pids.insert(0, pid_i)
                                    seen.add(pid_i)
                            suggestions_pids = suggestions_pids[:TOP_CHOICES]

                if prod is None and not suggestions_pids:
                    alias_hits = alias_suggestions(
                        name_norm,
                        alias_to_pids=alias_to_pids,
                        token_to_pids=token_to_pids,
                        limit=TOP_CHOICES,
                    )
                    if alias_hits:
                        suggestions_pids = [int(x) for x in alias_hits][:TOP_CHOICES]
                    else:
                        top = suggest_matches(name_norm, catalog_names, limit=TOP_CHOICES)
                        seen = set()
                        for cand_norm, _sc in top:
                            for pid in catalog_norm_to_pids.get(cand_norm, []):
                                pid_i = int(pid)
                                if pid_i not in seen:
                                    suggestions_pids.append(pid_i)
                                    seen.add(pid_i)
                        suggestions_pids = suggestions_pids[:TOP_CHOICES]

            unit_val = normalize_unit(unit_raw) if unit_raw else ""
            if prod and getattr(prod, "unit", None):
                unit_val = normalize_unit(prod.unit)

            if prod and getattr(prod, "id", None) is not None:
                matched_pid = int(prod.id)

            recommended_pid = suggestions_pids[0] if suggestions_pids else (matched_pid if matched_pid is not None else None)
            status = "OK" if prod else ("Elegir" if suggestions_pids else "Revisar")

            parsed_rows.append({
                "item_key": item_key,
                "chat_ts": it["chat_ts"],
                "item_idx": it["item_idx"],
                "spoken_name": name,
                "matched_product_id": matched_pid,
                "matched_name": prod.name if prod else None,
                "confidence": round(float(score or 0.0), 1),
                "quantity": qty,
                "unit": unit_val or "unit",
                "unit_custom": "",
                "suggestions": suggestions_pids,
                "recommended_pid": recommended_pid,
                "status": status,
            })

        return pd.DataFrame(parsed_rows)


    # Run auto-parse if pending
    has_any_text = bool((st.session_state.get(S("transcript_area")) or "").strip())
    if st.session_state.get(S("auto_parse_pending"), False) and has_any_text:
        candidates_df = _parse_chat_to_candidates()
        st.session_state[S("parse_candidates_df")] = candidates_df
        st.session_state[S("finalize_parse_pending")] = True
        st.session_state[S("auto_parse_pending")] = False

        if isinstance(candidates_df, pd.DataFrame) and not candidates_df.empty:
            needs_choice = candidates_df[
                (candidates_df["status"] == "Elegir")
                & candidates_df["suggestions"].apply(lambda x: isinstance(x, list) and len(x) > 0)
            ]
            if needs_choice.empty:
                st.session_state[S("finalize_parse_pending")] = False
                st.session_state[S("parsed_df")] = finalize_candidates_to_df(
                    candidates_df,
                    products_by_id=products_by_id,
                    venue_id=venue_id,
                )
        st.rerun()


    # =========================================================
    # Parsed summary preview card (Unified notebook with delete buttons)
    # =========================================================
    parsed_df = st.session_state.get(S("parsed_df"))
    order_chat = st.session_state.get(S("order_chat"), []) or []
    
    # Initialize strikethrough state
    st.session_state.setdefault(S("striked_products"), set())
    striked_products = st.session_state.get(S("striked_products"), set())

    if isinstance(parsed_df, pd.DataFrame) and not parsed_df.empty:
        # Count non-striked products for status
        any_needs_review = False
        active_products = 0
        
        for idx, r in parsed_df.iterrows():
            pid = r.get("matched_product_id", None)
            status = safe_str(r.get("status") or "").lower()
            name = safe_str(r.get("matched_name") or r.get("spoken_name") or "").strip()
            qty = r.get("quantity", None)
            product_key = f"{idx}_{name}_{qty}"
            
            if product_key not in striked_products:
                active_products += 1
            
            if pid is None or (isinstance(pid, float) and pd.isna(pid)) or ("revis" in status):
                any_needs_review = True
        
        # Product list with delete buttons
        # st.markdown('<div class="voi-notebook-products">', unsafe_allow_html=True)
        css = """
        .st-key-my_blue_container {
            background-color: rgba(254, 249, 231, 1);
        }
        """

        st.html(f"<style>{css}</style>")
        with st.container(border=True, key="my_blue_container", height=800):
            for row_idx, r in parsed_df.iterrows():
                name = safe_str(r.get("matched_name") or r.get("spoken_name") or "").strip()
                qty = r.get("quantity", None)
                pid = r.get("matched_product_id", None)
                unit = safe_str(r.get("unit") or "unit").strip()
                provider = safe_str(r.get("provider") or "").strip()
                
                # Get description
                description = ""
                if pid is not None and not (isinstance(pid, float) and pd.isna(pid)):
                    try:
                        prod = products_by_id.get(int(pid))
                        if prod:
                            description = safe_str(getattr(prod, "description", "") or "").strip()
                    except Exception:
                        pass
                
                if not name:
                    continue
                
                product_key = f"{row_idx}_{name}_{qty}"
                is_striked = product_key in striked_products
                
                # Create columns for product and delete button
                # col_product, col_delete = st.columns([11, 1])
                
                with st.container(horizontal=True):
                    # Product name with quantity
                    strike_style = "text-decoration: line-through; text-decoration-color: #c41e3a; text-decoration-thickness: 2px; opacity: 0.4;" if is_striked else ""
                    
                    qty_badge = ""
                    if qty:
                        try:
                            qty_display = f"{float(qty):g}"
                            qty_badge = f'<span style="display: inline-flex; align-items: center; justify-content: center; min-width: 28px; padding: 2px 8px; background: rgba(196,30,58,0.15); border-radius: 4px; font-weight: 700; font-size: 1.15rem; color: #c41e3a; margin-right: 8px;">{qty_display}</span>'
                        except:
                            pass
                    
                    st.markdown(
                        f'<div style="font-family: \'Inter\', sans-serif; font-size: 0.85rem; font-weight: 600; color: #1a1a1a; padding-top: 8px; {strike_style}">{qty_badge}{name}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Product details
                    details_parts = []
                    if unit and unit != "unit":
                        details_parts.append(f'<span style="font-weight: 600; color: #2a2a2a;"></span> {unit}')
                    if description:
                        details_parts.append(f'<span style="font-weight: 600; color: #2a2a2a;"></span> {description}')
                    if provider:
                        details_parts.append(f'<span style="font-weight: 600; color: #2a2a2a;"></span> {provider}')
                    
                    if details_parts:
                        st.markdown(
                            f'<div style="font-family: \'Inter\', sans-serif; font-size: 0.75rem; color: #4a4a4a; padding: 4px 0 8px 0; {strike_style}">{" · ".join(details_parts)}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)
                
                # with col_delete:
                    # Delete button
                    btn_label = "↺" if is_striked else "✗"
                    if st.button(
                        btn_label,
                        key=K(f"strike_{product_key}"),
                        help=f"{'Restaurar' if is_striked else 'Tachar'} {name[:20]}",
                        type="secondary"
                    ):
                        if is_striked:
                            striked_products.discard(product_key)
                        else:
                            striked_products.add(product_key)
                        st.session_state[S("striked_products")] = striked_products
                        st.rerun()
                
                # Separator line
                st.markdown('<div style="border-bottom: 1px dotted rgba(0,0,0,0.1); margin: 0;"></div>', unsafe_allow_html=True)

        


    # Spacer so the fixed bottom bar doesn't cover the resolver
    st.markdown('<div style="height:110px"></div>', unsafe_allow_html=True)


    # =========================================================
    # Language control (BEFORE form - outside form)
    # =========================================================
    effective_lang_code = lang_code or "auto"
    if not bool(cfg_cached.get("hide_user_controls", True)):
        st.session_state.setdefault(S("lang_code_ui"), None)
        if st.session_state.get(S("lang_code_ui")) is None:
            st.session_state[S("lang_code_ui")] = lang_code or "auto"

        effective_lang_code = st.session_state.get(S("lang_code_ui")) or lang_code or "auto"

        try:
            with st.popover("🌐", use_container_width=False):
                picked = st.radio(
                    "Idioma",
                    options=["auto", "es", "en", "el"],
                    index=["auto", "es", "en", "el"].index(
                        effective_lang_code if effective_lang_code in ["auto", "es", "en", "el"] else "auto"
                    ),
                    format_func=lambda v: {
                        "auto": "🌐 Auto",
                        "es": "🇪🇸 Español",
                        "en": "🇬🇧 English",
                        "el": "🇬🇷 Ελληνικά",
                    }.get(v, v),
                    key=K("lang_picker_radio"),
                )
                st.session_state[S("lang_code_ui")] = picked
                effective_lang_code = picked
        except Exception:
            with st.expander("🌐", expanded=False):
                picked = st.radio(
                    "Idioma",
                    options=["auto", "es", "en", "el"],
                    index=["auto", "es", "en", "el"].index(
                        effective_lang_code if effective_lang_code in ["auto", "es", "en", "el"] else "auto"
                    ),
                    format_func=lambda v: {
                        "auto": "🌐 Auto",
                        "es": "🇪🇸 Español",
                        "en": "🇬🇧 English",
                        "el": "🇬🇷 Ελληνικά",
                    }.get(v, v),
                    key=K("lang_picker_radio_fallback"),
                )
                st.session_state[S("lang_code_ui")] = picked
                effective_lang_code = picked
    else:
        st.session_state.pop(S("lang_code_ui"), None)
        effective_lang_code = lang_code or "auto"


    # =========================================================
    # ASR processing (BEFORE form - handles audio transcription)
    # =========================================================
    st.session_state.setdefault(S("audio_bytes"), b"")
    audio_bytes = st.session_state.get(S("audio_bytes")) or b""
    audio_hash = hashlib.sha1(audio_bytes).hexdigest() if audio_bytes else None

    if audio_bytes and audio_hash and audio_hash != st.session_state.get(S("last_audio_hash")):
        try:
            with st.spinner("Transcribiendo…"):
                if asr_backend == "OpenAI Whisper API":
                    transcript = asr_openai_whisper(audio_bytes, catalog_prompt_names, language=effective_lang_code)
                elif asr_backend == "Faster-Whisper (local)":
                    transcript = asr_faster_whisper(audio_bytes, catalog_prompt_names, language=effective_lang_code)
                elif asr_backend == "Google Speech-to-Text":
                    transcript = asr_google(audio_bytes, catalog_prompt_names, language=effective_lang_code)
                else:
                    transcript = ""

            st.session_state[S("last_audio_hash")] = audio_hash
            transcript = cleanup_asr_transcript(transcript)
            transcript = " | ".join(tokenize_items(transcript))
            append_message("asr", transcript)
            st.rerun()
        except Exception as e:
            st.error(f"Error transcribiendo: {e}")


    # =========================================================
    # WhatsApp-like fixed bottom bar with AUDIO INPUT IN DICTAR SLOT
    # =========================================================
    st.session_state.setdefault(S("wa_text_input"), "")

    st.markdown('<div class="voi-bottom-wrap"><div class="voi-bottom-inner">', unsafe_allow_html=True)

    with st.form(key=K("wa_form"), clear_on_submit=True):
        
        typed = st.text_input(
            "",
            placeholder="Escribe como en WhatsApp... ej: 3 coca cola, hielo",
            key=K("wa_text_input_field"),
            label_visibility="collapsed",
        )

        # b1, b2, b3, b4 = st.columns([1, 2, 1, 1], vertical_alignment="center")
        
        with st.container(horizontal=True):
            # ✅ AUDIO INPUT NOW LIVES HERE (in the "Dictar" slot)
            
            audio_file = st.audio_input("", key=K("audio_msg"), label_visibility="collapsed")
            if audio_file is not None:
                st.session_state[S("audio_bytes")] = audio_file.read()
                
        # with b2:
            add_clicked = st.form_submit_button("Add", use_container_width=True, type="primary", key=K("btn_add_note"))

        # with b3:
            limpiar_clicked = st.form_submit_button("🗑️", use_container_width=True, key=K("btn_clear_text"))
            
        # # with b4:
        #     reset_clicked = st.form_submit_button("🔄", use_container_width=True, key=K("btn_reset_all"))

    st.markdown("</div></div>", unsafe_allow_html=True)

    # ✅ IMPROVED: Style the button using CSS instead of JavaScript
    # This is more reliable than style_button() which can fail due to timing issues
    st.markdown("""
    <style>
    /* Target primary buttons (the Add button) */
    button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: 1px solid rgba(59,130,246,.35) !important;
        border-radius: 14px !important;
        padding: 10px 14px !important;
        font-size: 16px !important;
        box-shadow: 0 10px 20px rgba(2,6,23,.10) !important;
        font-weight: 700 !important;
        transition: all 0.2s ease !important;
    }

    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        transform: translateY(-1px);
        box-shadow: 0 12px 24px rgba(2,6,23,.15) !important;
    }

    button[kind="primary"]:active {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        transform: translateY(0);
    }
    
    /* Make sure text inside button is also styled */
    button[kind="primary"] p,
    button[kind="primary"] div,
    button[kind="primary"] span {
        color: white !important;
        font-size: 16px !important;
        font-weight: 700 !important;
    }
    </style>
    """, unsafe_allow_html=True)


    # =========================================================
    # Handle form submission actions
    # =========================================================

    # Handle typed input submission
    if typed and typed.strip():
        append_message("user", typed.strip())
        st.rerun()

    # =========================================================
    # 🎯 IMPROVED ADD BUTTON LOGIC - Mobile-First UX
    # =========================================================
    
    # Initialize draft selection state if needed
    if S("awaiting_draft_selection") not in st.session_state:
        st.session_state[S("awaiting_draft_selection")] = False
    
    # If we're awaiting selection, show the modal-style selector FIRST
    if st.session_state.get(S("awaiting_draft_selection"), False):
        parsed_df = st.session_state.get(S("parsed_df_pending"))
        df_to_use = st.session_state.get(S("df_to_use_pending"))
        actor = st.session_state.get(S("actor_pending"))
        
        with get_session() as s:
            drafts = s.exec(
                select(Order)
                .where(Order.venue_id == venue_id, Order.status == "draft")
                .order_by(Order.created_at.desc())
            ).all()
        
        st.markdown("---")
        st.markdown("### 📋 Selecciona un borrador")
        st.info("Tienes varios borradores abiertos. ¿A cuál quieres añadir estos productos?")
        
        # Create mobile-friendly card-style buttons for each draft
        for draft in drafts:
            draft_date = draft.created_at.strftime('%d/%m/%Y %H:%M')
            draft_title = draft.title or "Sin título"
            
            # Count items in this draft
            with get_session() as s:
                line_count = s.exec(
                    select(func.count(OrderLine.id))
                    .where(OrderLine.order_id == draft.id)
                ).first() or 0
            
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 16px;
                    border-radius: 12px;
                    margin-bottom: 12px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                ">
                    <div style="color: white; font-size: 18px; font-weight: 600; margin-bottom: 4px;">
                        {draft_title}
                    </div>
                    <div style="color: rgba(255,255,255,0.8); font-size: 14px; margin-bottom: 4px;">
                        #{draft.id} • {draft_date}
                    </div>
                    <div style="color: rgba(255,255,255,0.9); font-size: 13px;">
                        📦 {line_count} productos
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button("✓", key=f"select_draft_{draft.id}", use_container_width=True, type="primary"):
                    # Save to this draft
                    _add_lines_to_existing_draft(
                        venue_id=venue_id,
                        order_id=int(draft.id),
                        actor=actor,
                        df=df_to_use,
                    )
                    _set_active_draft(draft.id)
                    bump_orders_refresh_token()
                    
                    # Clear state
                    st.session_state[S("awaiting_draft_selection")] = False
                    st.session_state.pop(S("parsed_df_pending"), None)
                    st.session_state.pop(S("df_to_use_pending"), None)
                    st.session_state.pop(S("actor_pending"), None)
                    st.session_state[S("striked_products")] = set()
                    reset_notes_only(do_rerun=False)
                    
                    st.success(f"✅ Productos añadidos al borrador #{draft.id}")
                    time.sleep(0.5)
                    _go_orders(draft.id)
        
        # Option to create new draft instead
        st.markdown("---")
        if st.button("➕ Crear nuevo borrador", key="create_new_draft_instead", use_container_width=True, type="secondary"):
            new_id = _create_draft_and_insert_lines(
                venue_id=venue_id,
                actor=actor,
                df=df_to_use,
            )
            if new_id:
                _set_active_draft(new_id)
                bump_orders_refresh_token()
                
                # Clear state
                st.session_state[S("awaiting_draft_selection")] = False
                st.session_state.pop(S("parsed_df_pending"), None)
                st.session_state.pop(S("df_to_use_pending"), None)
                st.session_state.pop(S("actor_pending"), None)
                st.session_state[S("striked_products")] = set()
                reset_notes_only(do_rerun=False)
                
                st.success(f"✅ Nuevo borrador creado #{new_id}")
                time.sleep(0.5)
                _go_orders(new_id)
        
        # Cancel button
        if st.button("✕ Cancelar", key="cancel_draft_selection", use_container_width=True):
            st.session_state[S("awaiting_draft_selection")] = False
            st.session_state.pop(S("parsed_df_pending"), None)
            st.session_state.pop(S("df_to_use_pending"), None)
            st.session_state.pop(S("actor_pending"), None)
            st.rerun()
        
        st.stop()
    
    if add_clicked:
        parsed_df = st.session_state.get(S("parsed_df"))

        if not isinstance(parsed_df, pd.DataFrame) or parsed_df.empty:
            st.warning("Aún no hay nada parseado para guardar. Añade texto o audio y espera a que se genere el resumen.")
            st.stop()

        # Filter out striked products (your existing logic)
        striked_products = st.session_state.get(S("striked_products"), set())
        df_filtered = parsed_df.copy()

        if striked_products:
            keep_mask = []
            for idx, r in df_filtered.iterrows():
                name = safe_str(r.get("matched_name") or r.get("spoken_name") or "").strip()
                qty = r.get("quantity", None)
                product_key = f"{idx}_{name}_{qty}"
                keep_mask.append(product_key not in striked_products)
            df_filtered = df_filtered[keep_mask]

        if df_filtered.empty:
            st.warning("Todos los productos están tachados. No hay nada que guardar.")
            st.stop()

        df_to_use = apply_unit_choice(df_filtered)
        actor = current_actor()

        # ---------------------------------------
        # Determine draft target intelligently
        # ---------------------------------------
        with get_session() as s:
            drafts = s.exec(
                select(Order)
                .where(Order.venue_id == venue_id, Order.status == "draft")
                .order_by(Order.created_at.desc())
            ).all()

        target_id: int = 0

        # Case 1️⃣ — Active draft already set
        if active_draft_id:
            target_id = int(active_draft_id)

        # Case 2️⃣ — Only ONE draft exists → auto use it
        elif len(drafts) == 1:
            target_id = int(drafts[0].id)
            _set_active_draft(target_id)

        # Case 3️⃣ — Multiple drafts → trigger selection mode (NO blocking)
        elif len(drafts) > 1:
            # Store data in session state for the selection UI
            st.session_state[S("parsed_df_pending")] = parsed_df
            st.session_state[S("df_to_use_pending")] = df_to_use
            st.session_state[S("actor_pending")] = actor
            st.session_state[S("awaiting_draft_selection")] = True
            st.rerun()

        # Case 4️⃣ — No drafts → create new
        else:
            new_id = _create_draft_and_insert_lines(
                venue_id=venue_id,
                actor=actor,
                df=df_to_use,
            )
            if new_id:
                target_id = int(new_id)

        # ---------------------------------------
        # Apply add-to-draft (if target determined)
        # ---------------------------------------
        if not target_id:
            # This means we triggered draft selection - do nothing here
            pass
        else:
            _add_lines_to_existing_draft(
                venue_id=venue_id,
                order_id=int(target_id),
                actor=actor,
                df=df_to_use,
            )

            _set_active_draft(target_id)
            bump_orders_refresh_token()

            st.success(f"Preparación guardada ✅ (#{target_id})")

            # Clear striked products after saving (your existing behavior)
            st.session_state[S("striked_products")] = set()

            reset_notes_only(do_rerun=False)
            _go_orders(target_id)


    if limpiar_clicked:
        reset_notes_only(clear_resolved_picks=False)
        st.success("Texto limpiado.")
        st.rerun()

    # if reset_clicked:
    #     reset_notes_only(clear_resolved_picks=True)
    #     st.success("Todo reiniciado.")
    #     st.rerun()


    # # =========================================================
    # # Draft selector (optional)
    # # =========================================================
    # with st.expander("⚙️ Cambiar borrador destino (opcional)", expanded=False):
    #     with get_session() as s:
    #         drafts = s.exec(
    #             select(Order)
    #             .where(Order.venue_id == venue_id, Order.status == "draft")
    #             .order_by(Order.created_at.desc())
    #         ).all()

    #     if not drafts:
    #         st.caption("No hay borradores aún. Se creará uno al guardar.")
    #     else:
    #         draft_options = [(o.id, f"#{o.id} — {o.title or o.created_at.strftime('%Y-%m-%d %H:%M')}") for o in drafts]
    #         labels = [lbl for _, lbl in draft_options]
    #         ids = [oid for oid, _ in draft_options]

    #         default_idx = 0
    #         if active_draft_id in ids:
    #             default_idx = ids.index(active_draft_id)

    #         chosen_label = st.selectbox("Borrador destino", options=labels, index=default_idx, key=K("choose_draft_select"))
    #         manual_target_id = int(ids[labels.index(chosen_label)])

    #         if st.button("Usar este borrador como activo", key=K("btn_set_active_draft")):
    #             _set_active_draft(manual_target_id)
    #             st.success(f"Activo: #{manual_target_id}")
    #             st.rerun()
