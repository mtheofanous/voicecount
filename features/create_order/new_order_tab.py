from __future__ import annotations

import time
import hashlib
import re
from typing import Dict, List, Optional, Any, Tuple
from functools import lru_cache
import math
import json
import pandas as pd
import streamlit as st
from sqlmodel import Session, select
from datetime import datetime
from sqlalchemy import desc, func, and_
from sqlalchemy.orm import load_only
from streamlit_float import float_init, float_css_helper, float_dialog
from domain.models import Product, Order, OrderLine, VenueTranscriptionSettings, ProviderSendStatus
from core.db import get_session
from features.utils.asr_google import asr_google
from features.manage_orders.orders import current_actor
from features.utils.voice_and_orders_utils import *

# ✅ single source of truth for normalization + unit synonyms
from core.normalization import normalize_text, normalize_unit, UNIT_SYNONYMS
from core.i18n import t

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
    st.error(t("msg.no_active_venue"))
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
        products = list(s.exec(
            select(Product)
            .options(load_only(
                Product.id, Product.venue_id, Product.name, Product.description,
                Product.category, Product.unit, Product.quantity, Product.price, Product.iva,
                Product.provider_name, Product.provider_email, Product.provider_phone,
                Product.provider_address, Product.aliases
            ))
            .where(Product.venue_id == venue_id)
            .order_by(Product.name.asc(), Product.provider_name.asc())
        ).all())

        # Force-load all attributes while session is active to prevent DetachedInstanceError
        for p in products:
            _ = p.id, p.name, p.description, p.category, p.unit, p.quantity, p.price, p.iva
            _ = p.provider_name, p.provider_email, p.provider_phone, p.provider_address, p.aliases

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


@st.cache_data(show_spinner=False, ttl=30)
def load_venue_drafts(venue_id: int, _refresh_token: int = 0):
    """
    Cached draft orders to avoid reloading on every rerun.
    TTL 30 seconds. Use _refresh_token to force refresh.
    """
    with get_session() as s:
        drafts = s.exec(
            select(Order)
            .options(load_only(Order.id, Order.title, Order.created_at, Order.status, Order.venue_id))
            .where(Order.venue_id == venue_id, Order.status == "draft", Order.title != "__autosave__")
            .order_by(Order.created_at.desc())
        ).all()
    return drafts


@st.cache_data(show_spinner=False, ttl=60)
def get_last_sent_pid_for_venue_among_opts(venue_id: int, opts: tuple[int, ...]) -> int | None:
    """🕒 Last product among opts that was included in an order that was SENT to its provider (per-provider send).
    Cached for 60 seconds. opts must be tuple for hashing."""
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


@st.cache_data(show_spinner=False, ttl=60)
def get_most_frequent_sent_pid_for_venue_among_opts(venue_id: int, opts: tuple[int, ...]) -> int | None:
    """🔁 Most frequently SENT product among opts to its provider (per-provider send).
    Cached for 60 seconds. opts must be tuple for hashing."""
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
    float_init()
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

        # Delete the auto-save draft from DB so it doesn't restore on next load
        _aid = st.session_state.pop(S("autosave_order_id"), None)
        st.session_state.pop(S("autosave_hash"), None)
        if _aid:
            try:
                _delete_autosave(order_id=_aid)
            except Exception:
                pass

        if do_rerun:
            st.rerun()

    def bump_orders_refresh_token() -> None:
        k = f"orders_refresh_token_{venue_id}"
        st.session_state[k] = int(st.session_state.get(k, 0)) + 1

    def bump_drafts_refresh_token() -> None:
        """Bump draft cache refresh token to force reload"""
        k = f"drafts_refresh_token_{venue_id}"
        st.session_state[k] = int(st.session_state.get(k, 0)) + 1

    def get_drafts_refresh_token() -> int:
        """Get current draft cache refresh token"""
        k = f"drafts_refresh_token_{venue_id}"
        return int(st.session_state.get(k, 0))

    def _rebuild_transcript_from_chat() -> None:
        st.session_state[S("transcript_area")] = "\n".join(
            (m.get("text") or "").strip()
            for m in st.session_state[S("order_chat")]
            if (m.get("text") or "").strip()
        ).strip()

    def _merge_manual_products_back(parsed_df: pd.DataFrame) -> pd.DataFrame:
        """Merge back any manually-added products (from Añadir productos) that were
        preserved before re-parsing from chat."""
        manual_backup = st.session_state.pop(S("_manual_products_backup"), None)
        if not isinstance(manual_backup, pd.DataFrame) or manual_backup.empty:
            return parsed_df

        if not isinstance(parsed_df, pd.DataFrame) or parsed_df.empty:
            return manual_backup

        # For each manual product, add it or merge quantities with existing
        for _, mrow in manual_backup.iterrows():
            pid = mrow.get("matched_product_id")
            if pid is not None and not pd.isna(pid):
                existing_mask = parsed_df["matched_product_id"] == int(pid)
                if existing_mask.any():
                    # Product also came from voice — sum quantities
                    idx = parsed_df.index[existing_mask][0]
                    parsed_df.at[idx, "quantity"] = float(parsed_df.at[idx, "quantity"] or 0) + float(mrow.get("quantity") or 0)
                    parsed_df.at[idx, "source"] = "manual"
                else:
                    parsed_df = pd.concat([parsed_df, pd.DataFrame([mrow])], ignore_index=True)
            else:
                parsed_df = pd.concat([parsed_df, pd.DataFrame([mrow])], ignore_index=True)

        return parsed_df

    def append_message(role_: str, text_: str) -> None:
        text_ = (text_ or "").strip()
        if not text_:
            return

        st.session_state[S("order_chat")].append({"role": role_, "text": text_, "ts": time.time()})
        _rebuild_transcript_from_chat()

        # auto-parse after every change
        st.session_state[S("auto_parse_pending")] = True

        # Preserve manually-added products (from Añadir productos) before clearing
        old_df = st.session_state.get(S("parsed_df"))
        if isinstance(old_df, pd.DataFrame) and not old_df.empty and "source" in old_df.columns:
            manual_rows = old_df[old_df["source"] == "manual"].copy()
            st.session_state[S("_manual_products_backup")] = manual_rows
        else:
            st.session_state.pop(S("_manual_products_backup"), None)

        st.session_state.pop(S("parsed_df"), None)
        st.session_state.pop(S("parse_candidates_df"), None)
        st.session_state.pop(S("finalize_parse_pending"), None)

    # ---------------------------
    # Draft targeting (shared key with orders.py)
    # ---------------------------
    ACTIVE_DRAFT_KEY = f"orders_active_order_id_{venue_id}"
    AUTOSAVE_TITLE = "__autosave__"

    # ------------------------------------------------------------------
    # Auto-save helpers (persist Summary card across tab-navigation reloads)
    # NOTE: defined early so they are available before the catalog is loaded.
    # Closures capture variables by reference — products_by_id is resolved
    # at call-time, after it has been set by load_catalog_and_indexes().
    # ------------------------------------------------------------------

    def _autosave_parsed_df(df: pd.DataFrame, existing_order_id: Optional[int] = None) -> int:
        """Upsert a hidden auto-save draft Order+OrderLines for the current session.

        Unlike _create_draft_and_insert_lines, this saves ALL rows — including
        unmatched items (product_id=None) — so the Summary card is fully restored.
        Returns the autosave order_id.
        """
        actor = current_actor()
        with get_session() as s:
            order = None

            # 1. Try to reuse the known autosave order (fastest path)
            if existing_order_id:
                order = s.exec(
                    select(Order).where(
                        Order.id == existing_order_id,
                        Order.venue_id == int(venue_id),
                        Order.title == AUTOSAVE_TITLE,
                    )
                ).first()

            # 2. Fall back to querying by marker + actor
            if order is None:
                order = s.exec(
                    select(Order).where(
                        Order.venue_id == int(venue_id),
                        Order.status == "draft",
                        Order.title == AUTOSAVE_TITLE,
                        Order.created_by == actor,
                    )
                ).first()

            if order is not None:
                order_id = int(order.id)
                # Delete existing lines (full replace)
                existing_lines = s.exec(
                    select(OrderLine).where(OrderLine.order_id == order_id)
                ).all()
                for ln in existing_lines:
                    s.delete(ln)
                s.commit()
                order.updated_at = datetime.utcnow()
                order.updated_by = actor
                s.add(order)
            else:
                # Create new autosave Order
                order = Order(
                    venue_id=int(venue_id),
                    status="draft",
                    title=AUTOSAVE_TITLE,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    created_by=actor,
                    updated_by=actor,
                )
                s.add(order)
                s.commit()
                s.refresh(order)
                order_id = int(order.id)

            # Insert all rows (including unmatched — product_id may be None)
            for _, r in df.iterrows():
                pid = r.get("matched_product_id", None)
                try:
                    pid_i = int(pid) if pid is not None and not pd.isna(pid) else None
                except Exception:
                    pid_i = None

                try:
                    qty = float(r.get("quantity") or 0.0)
                except Exception:
                    qty = 0.0
                if qty <= 0:
                    continue

                s.add(OrderLine(
                    venue_id=int(venue_id),
                    order_id=order_id,
                    product_id=pid_i,
                    spoken_name=str(r.get("spoken_name") or r.get("matched_name") or ""),
                    matched_name=str(r.get("matched_name") or ""),
                    quantity=qty,
                    unit=str(r.get("unit") or "unit").strip().lower() or "unit",
                    confidence=float(r.get("confidence") or 0.0),
                    provider=r.get("provider") or None,
                    updated_at=datetime.utcnow(),
                    updated_by=actor,
                ))

            s.commit()
        return order_id

    def _restore_from_autosave() -> Optional[tuple[pd.DataFrame, int]]:
        """Reload the auto-save draft back into parsed_df after a page reload.

        Returns (df, order_id) or None if no autosave exists for this user/venue.
        products_by_id is captured by closure — resolved at call-time.
        """
        actor = current_actor()
        with get_session() as s:
            order = s.exec(
                select(Order).where(
                    Order.venue_id == int(venue_id),
                    Order.status == "draft",
                    Order.title == AUTOSAVE_TITLE,
                    Order.created_by == actor,
                )
            ).first()
            if order is None:
                return None

            lines = s.exec(
                select(OrderLine).where(OrderLine.order_id == int(order.id))
            ).all()
            if not lines:
                return None

            rows = []
            for ln in lines:
                pid = getattr(ln, "product_id", None)
                prod = products_by_id.get(int(pid)) if pid is not None else None
                rows.append({
                    "spoken_name":        getattr(ln, "spoken_name", "") or "",
                    "matched_product_id": pid,
                    "matched_name":       getattr(ln, "matched_name", "") or (prod.name if prod else None),
                    "confidence":         float(getattr(ln, "confidence", 0) or 0),
                    "quantity":           float(getattr(ln, "quantity", 1) or 1),
                    "unit":               getattr(ln, "unit", "unit") or "unit",
                    "unit_custom":        "",
                    "provider":           getattr(ln, "provider", None) or (prod.provider_name if prod else None),
                    "source":             "autosave",
                    "status":             "OK" if pid is not None else "Revisar",
                })

        if not rows:
            return None
        return pd.DataFrame(rows), int(order.id)

    def _delete_autosave(order_id: Optional[int] = None) -> None:
        """Delete the hidden auto-save draft (and its lines) from the DB."""
        actor = current_actor()
        with get_session() as s:
            if order_id is not None:
                orders_to_delete = s.exec(
                    select(Order).where(
                        Order.id == int(order_id),
                        Order.title == AUTOSAVE_TITLE,
                    )
                ).all()
            else:
                orders_to_delete = s.exec(
                    select(Order).where(
                        Order.venue_id == int(venue_id),
                        Order.title == AUTOSAVE_TITLE,
                        Order.created_by == actor,
                    )
                ).all()

            for o in orders_to_delete:
                lines = s.exec(select(OrderLine).where(OrderLine.order_id == int(o.id))).all()
                for ln in lines:
                    s.delete(ln)
                s.delete(o)
            s.commit()
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
        """Navigate to Orders (draft) using a real URL redirect (faster than st.rerun).

        This mirrors the 'Option B' optimization used in the bottom tabbar:
        we update the URL (?page=...) and let the browser navigate, avoiding an extra
        full-script pass that happens when triggering navigation late + st.rerun().
        """
        token = st.session_state.get("_session_token", "")
        token_param = f"&st={token}" if token else ""
        url = f"?page=orders&order_id={int(order_id)}&status=draft{token_param}"

        # Use a small client-side redirect, then stop this run.
        # (Safe quoting to avoid breaking the script tag.)
        st.markdown(
            f"""<script>window.location.href={json.dumps(url)};</script>""",
            unsafe_allow_html=True,
        )
        st.stop()

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
        st.warning(t("msg.no_catalog"))
        return

    # Validate alias_indexes structure (in case of cache corruption)
    if not isinstance(alias_indexes, dict):
        st.error(t("msg.catalog_data_error"))
        st.cache_data.clear()
        st.rerun()
        return
    
    required_keys = ["alias_to_pids", "token_to_pids", "alias_to_products"]
    missing_keys = [k for k in required_keys if k not in alias_indexes]
    if missing_keys:
        st.error(t("new_order.catalog_incomplete", keys=str(missing_keys)))
        st.cache_data.clear()
        st.rerun()
        return

    alias_to_pids = alias_indexes["alias_to_pids"]
    token_to_pids = alias_indexes["token_to_pids"]
    alias_to_products = alias_indexes["alias_to_products"]

    catalog_names = list(catalog_norm_to_pids.keys())

    # =========================================================
    # RESTORE AUTO-SAVE (lost session state after tab navigation reload)
    # =========================================================
    _existing_parsed = st.session_state.get(S("parsed_df"))
    if not isinstance(_existing_parsed, pd.DataFrame) or _existing_parsed.empty:
        _restored = _restore_from_autosave()
        if _restored is not None:
            _df_restored, _autosave_id = _restored
            st.session_state[S("parsed_df")] = _df_restored
            st.session_state[S("autosave_order_id")] = _autosave_id
            st.toast("↩️ " + t("new_order.autosave_restored"), icon="✅")

    # =========================================================
    # FULL-PAGE PRODUCT ADDER (similar to _render_lines_editor)
    # =========================================================
    if st.session_state.get("product_adder_fullpage", False):
        # Back button
        with st.container(horizontal=True):
            if st.button(f"← {t('action.back')}", key=K("back_from_product_adder")):
                st.session_state.product_adder_fullpage = False
                st.rerun()

            # Search bar
            search_query = st.text_input(
                t("new_order.search_product"),
                placeholder=t("new_order.search_placeholder"),
                key=K("product_search_fullpage"),label_visibility="collapsed",
            ).strip().lower()

            # Create product lookup maps
            with st.spinner(t("new_order.loading_catalog")):
                products_by_cat = {}
                products_by_prov = {}
                all_categories = set()
                all_providers = set()

                for p in products:
                    cat = getattr(p, "category", "") or t("new_order.no_category")
                    prov = getattr(p, "provider_name", "") or t("new_order.no_provider")
                    all_categories.add(cat)
                    all_providers.add(prov)
                    products_by_cat.setdefault(cat, []).append(p)
                    products_by_prov.setdefault(prov, []).append(p)

                # Filter products by search
                filtered_products = products
                if search_query:
                    filtered_products = [
                        p for p in products
                        if search_query in (getattr(p, "name", "") or "").lower()
                        or search_query in (getattr(p, "provider_name", "") or "").lower()
                        or search_query in (getattr(p, "description", "") or "").lower()
                    ]

 
            # Get all providers (with "Todos")
            prov_list = [t("new_order.all_providers")] + sorted(all_providers)

            selected_prov = st.selectbox(
                "Proveedor",
                options=prov_list,
                key=K("provider_selector"),label_visibility="collapsed"
            )

        # Filter by provider
        if selected_prov == t("new_order.all_providers"):
            prov_filtered = filtered_products
        else:
            # Use the pre-built dictionary instead of filtering
            prov_filtered = products_by_prov.get(selected_prov, [])

        # Category sub-tabs (filtered by provider)
        categories_in_prov = sorted({getattr(p, "category", "") or t("new_order.no_category") for p in prov_filtered})
        cat_list = [t("new_order.all_categories")] + categories_in_prov
        cat_tabs = st.tabs(cat_list)

        for j, cat_tab in enumerate(cat_tabs):
            with cat_tab:
                selected_cat = cat_list[j]

                # Final filter
                final_products = prov_filtered
                if selected_cat != t("new_order.all_categories"):
                    final_products = [
                        p for p in prov_filtered
                        if (getattr(p, "category", "") or t("new_order.no_category")) == selected_cat
                    ]

                if not final_products:
                    st.info(t("new_order.no_products_cat"))
                    continue

                st.caption(f"{len(final_products)} producto(s)")

                # --------------------------------------------------
                # Pagination — kept OUTSIDE the form so st.button
                # works reliably on mobile (no CSS hacks needed).
                # --------------------------------------------------
                PAGE_SIZE = 30
                page_state_key = K(f"fullpage_page_{selected_prov}_{selected_cat}_{j}")
                total_pages = max(1, int(math.ceil(len(final_products) / PAGE_SIZE)))
                page_idx = int(st.session_state.get(page_state_key, 0) or 0)
                page_idx = max(0, min(page_idx, total_pages - 1))
                st.session_state[page_state_key] = page_idx

                if total_pages > 1:
                    # col_prev, col_info, col_next = st.columns(
                    #     [1, 4, 1], vertical_alignment="center"
                    # )
                    with st.container(horizontal=True):
                        if st.button(
                            "◀",
                            key=K(f"prev_{selected_prov}_{j}"),
                            disabled=(page_idx == 0),
                            use_container_width=True,
                        ):
                            st.session_state[page_state_key] = max(0, page_idx - 1)
                            st.rerun()
                    # with col_info:
                        st.markdown(
                            f'<div style="text-align:center;font-size:0.9rem;color:#888;">'
                            f'{page_idx + 1} / {total_pages}</div>',
                            unsafe_allow_html=True,
                        )
                    # with col_next:
                        if st.button(
                            "▶",
                            key=K(f"next_{selected_prov}_{j}"),
                            disabled=(page_idx >= total_pages - 1),
                            use_container_width=True,
                        ):
                            st.session_state[page_state_key] = min(total_pages - 1, page_idx + 1)
                            st.rerun()

                # Slice products for this page
                start_i = page_idx * PAGE_SIZE
                end_i = start_i + PAGE_SIZE
                page_products = final_products[start_i:end_i]

                with st.form(key=K(f"add_form_{selected_prov}_{j}"), border=False):
                    add_clicked = st.form_submit_button(
                        t("new_order.add_selected"), type="primary", use_container_width=True
                    )

                    # -----------------------------
                    # Render only this page
                    # (values persist via keys)
                    # -----------------------------
                    with st.container(height=600):
                        for prod in page_products:
                            pid = int(prod.id)
                            pname = getattr(prod, "name", "")
                            pprov = getattr(prod, "provider_name", "")
                            pdesc = getattr(prod, "description", "")
                            punit = getattr(prod, "unit", "") or "unit"
                            pprice = getattr(prod, "price", None)

                            with st.container(horizontal=True, border=True):
            
                                st.markdown(f"**{pname}**")
                                details = []
                                if pdesc:
                                    details.append(pdesc)
                                if pprov:
                                    details.append(pprov)
                                if pprice:
                                    details.append(f"{float(pprice):.2f}€")
                
                                if details:
                                    st.caption(" · ".join(details))
                
                                st.number_input(
                                    "Cant.",
                                    min_value=0,
                                    value=(st.session_state.get(K(f"fullpage_qty_{pid}_{j}"), 0) or 0),
                                    step=1,
                                    key=K(f"fullpage_qty_{pid}_{j}"),
                                    label_visibility="collapsed"
                                )

                    # -----------------------------
                    # On submit: collect across ALL pages
                    # -----------------------------
                    if add_clicked:
                        products_to_add = []
                        for prod in final_products:
                            pid = int(prod.id)
                            qty = float(st.session_state.get(K(f"fullpage_qty_{pid}_{j}"), 0.0) or 0.0)
                            if qty > 0:
                                pname = getattr(prod, "name", "")
                                pprov = getattr(prod, "provider_name", "")
                                punit = getattr(prod, "unit", "") or "unit"
                                products_to_add.append((pid, pname, qty, punit, pprov))

                        if products_to_add:
                            parsed_df = st.session_state.get(S("parsed_df"))

                            for pid, pname, qty, punit, pprov in products_to_add:
                                new_row = pd.DataFrame([{
                                    "spoken_name": pname,
                                    "matched_product_id": pid,
                                    "matched_name": pname,
                                    "confidence": 100.0,
                                    "quantity": qty,
                                    "unit": punit,
                                    "unit_custom": "",
                                    "suggestions": [],
                                    "recommended_pid": pid,
                                    "status": "OK",
                                    "provider": pprov,
                                    "source": "manual",
                                }])

                                if isinstance(parsed_df, pd.DataFrame) and not parsed_df.empty:
                                    existing_mask = parsed_df['matched_product_id'] == pid
                                    if existing_mask.any():
                                        idx = parsed_df.index[existing_mask][0]
                                        parsed_df.at[idx, 'quantity'] = float(parsed_df.at[idx, 'quantity']) + qty
                                    else:
                                        parsed_df = pd.concat([parsed_df, new_row], ignore_index=True)
                                else:
                                    if parsed_df is None:
                                        parsed_df = new_row
                                    else:
                                        parsed_df = pd.concat([parsed_df, new_row], ignore_index=True)

                            st.session_state[S("parsed_df")] = parsed_df
                            st.success(t("new_order.added_products", n=str(len(products_to_add))))
                            time.sleep(0.5)
                            st.session_state.product_adder_fullpage = False
                            st.rerun()
                        else:
                            st.warning(t("msg.no_product_selected"))

        # Stop rendering the rest of the page
        return

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
    st.markdown(t("new_order.header"), unsafe_allow_html=True)

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
                st.markdown(t("new_order.choose_ambiguous"))
                st.caption(t("msg.tap_option_per_line"))

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

                    # Convert to tuple for caching
                    opts_tuple = tuple(opts)
                    last_sent_pid = get_last_sent_pid_for_venue_among_opts(venue_id=venue_id, opts=opts_tuple)
                    most_freq_pid = get_most_frequent_sent_pid_for_venue_among_opts(venue_id=venue_id, opts=opts_tuple)

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

                if st.button(t("action.ok"), type="primary", key=K("btn_finalize_parse")):
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

                    new_parsed = finalize_candidates_to_df(
                        df2,
                        products_by_id=products_by_id,
                        venue_id=venue_id,
                    )
                    st.session_state[S("parsed_df")] = _merge_manual_products_back(new_parsed)
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


    # Run auto-parse if pending (optimization: only when text exists)
    has_any_text = bool((st.session_state.get(S("transcript_area")) or "").strip())
    auto_parse_pending = st.session_state.get(S("auto_parse_pending"), False)

    # Skip parsing if no text or already finalized
    if auto_parse_pending and has_any_text and not st.session_state.get(S("parsed_df")):
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
                new_parsed = finalize_candidates_to_df(
                    candidates_df,
                    products_by_id=products_by_id,
                    venue_id=venue_id,
                )
                st.session_state[S("parsed_df")] = _merge_manual_products_back(new_parsed)
        else:
            # No voice products parsed — restore manual products if any
            manual_backup = st.session_state.pop(S("_manual_products_backup"), None)
            if isinstance(manual_backup, pd.DataFrame) and not manual_backup.empty:
                st.session_state[S("parsed_df")] = manual_backup
                st.session_state[S("finalize_parse_pending")] = False
        st.rerun()


    # =========================================================
    # AUTO-SAVE parsed_df to DB (debounced by content hash)
    # =========================================================
    _autosave_df = st.session_state.get(S("parsed_df"))
    if isinstance(_autosave_df, pd.DataFrame) and not _autosave_df.empty:
        _df_hash = hashlib.md5(
            _autosave_df.astype(str).to_json().encode()
        ).hexdigest()
        if _df_hash != st.session_state.get(S("autosave_hash")):
            _aid = _autosave_parsed_df(
                _autosave_df,
                existing_order_id=st.session_state.get(S("autosave_order_id")),
            )
            st.session_state[S("autosave_order_id")] = _aid
            st.session_state[S("autosave_hash")] = _df_hash

    # =========================================================
    # Parsed summary preview card (Unified notebook with delete buttons)
    # =========================================================
    @st.fragment
    def render_product_list():
        """Fragment: Product list with strike/unstrike - isolated reruns for better performance"""
        parsed_df = st.session_state.get(S("parsed_df"))

        # Initialize strikethrough state
        st.session_state.setdefault(S("striked_products"), set())
        striked_products = st.session_state.get(S("striked_products"), set())

        if not isinstance(parsed_df, pd.DataFrame) or parsed_df.empty:
            return

        # Vectorized counting (much faster than iterrows)
        # Create product keys vectorized (qty excluded so strike survives edits)
        names = parsed_df['matched_name'].fillna(parsed_df['spoken_name']).fillna('').astype(str).str.strip()
        indices = parsed_df.index.astype(str)
        product_keys = indices + '_' + names

        # Count active (non-striked) products
        active_mask = ~product_keys.isin(striked_products)
        active_products = active_mask.sum()

        # Check if any need review (vectorized)
        status_series = parsed_df['status'].fillna('').astype(str).str.lower()
        pid_is_null = parsed_df['matched_product_id'].isna()
        has_revisar = status_series.str.contains('revis', na=False)
        any_needs_review = (pid_is_null | has_revisar).any()

        # Product list with delete buttons
        css = """
        .st-key-my_blue_container {
            background-color: rgba(254, 249, 231, 1);
        }
        /* Force single-line horizontal layout on all screen sizes */
        .st-key-my_blue_container [data-testid="stHorizontalBlock"] {
            flex-wrap: nowrap !important;
            align-items: center !important;
            gap: 4px !important;
            min-width: 0 !important;
        }
        /* Qty input column - shrinks to content width */
        .st-key-my_blue_container [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlockBorderWrapper"]:first-child {
            flex: 0 0 auto !important;
            min-width: 0 !important;
        }
        /* Name column - takes remaining space, clips overflow */
        .st-key-my_blue_container [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlockBorderWrapper"]:nth-child(2) {
            flex: 1 1 0% !important;
            min-width: 0 !important;
            overflow: hidden !important;
        }
        /* Strike button column - auto width */
        .st-key-my_blue_container [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlockBorderWrapper"]:last-child {
            flex: 0 0 auto !important;
        }
        /* Qty number input - collapse all wrappers to content size */
        .st-key-my_blue_container [data-testid="stNumberInput"],
        .st-key-my_blue_container [data-testid="stNumberInput"] > div,
        .st-key-my_blue_container [data-testid="stNumberInput"] > div > div {
            min-height: unset !important;
            width: auto !important;
        }
        .st-key-my_blue_container [data-testid="stNumberInput"] > label {
            display: none !important;
        }
        /* Hide +/- step buttons */
        .st-key-my_blue_container [data-testid="stNumberInput"] button {
            display: none !important;
        }
        /* Input sizes to its own content */
        .st-key-my_blue_container [data-testid="stNumberInput"] input {
            field-sizing: content;
            min-width: 2ch;
            width: auto !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            color: #c41e3a !important;
            text-align: center !important;
            padding: 2px 6px !important;
            background: rgba(196,30,58,0.08) !important;
            border-color: rgba(196,30,58,0.25) !important;
            border-radius: 4px !important;
        }
        .st-key-my_blue_container [data-testid="stNumberInput"] input:focus {
            background: rgba(196,30,58,0.15) !important;
            border-color: #c41e3a !important;
            box-shadow: 0 0 0 2px rgba(196,30,58,0.15) !important;
        }
        """

        st.html(f"<style>{css}</style>")
        with st.container(key="my_blue_container"):
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

                # qty excluded from key so strike state survives qty edits
                product_key = f"{row_idx}_{name}"
                is_striked = product_key in striked_products

                with st.container(horizontal=True):
                    strike_style = "text-decoration: line-through; text-decoration-color: #c41e3a; text-decoration-thickness: 2px; opacity: 0.4;" if is_striked else ""

                    # Column 1: editable quantity input (integer)
                    qty_val = 0
                    try:
                        qty_val = int(round(float(qty))) if qty is not None else 0
                    except Exception:
                        qty_val = 0

                    new_qty = st.number_input(
                        "qty",
                        value=qty_val,
                        min_value=0,
                        step=1,
                        label_visibility="collapsed",
                        key=K(f"qty_{row_idx}"), width=30,
                    )

                    if new_qty != qty_val:
                        parsed_df.at[row_idx, "quantity"] = new_qty
                        st.session_state[S("parsed_df")] = parsed_df
                        st.rerun(scope="fragment")

                    # Column 2: product name + details
                    details_parts = []
                    if unit and unit != "unit":
                        details_parts.append(unit)
                    if description:
                        details_parts.append(description)
                    if provider:
                        details_parts.append(provider)

                    details_html = ""
                    if details_parts:
                        details_html = f'<div style="font-family: \'Inter\', sans-serif; font-size: 0.75rem; color: #4a4a4a; padding: 2px 0 6px 0; {strike_style} overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{" · ".join(details_parts)}</div>'

                    st.markdown(
                        f'<div style="overflow: hidden;">'
                        f'<div style="font-family: \'Inter\', sans-serif; font-size: 0.85rem; font-weight: 600; color: #1a1a1a; padding-top: 8px; {strike_style} overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{name}</div>'
                        f'{details_html}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    # Column 3: strike/restore button (fragment rerun only)
                    btn_label = "↺" if is_striked else "✗"
                    if st.button(
                        btn_label,
                        key=K(f"strike_{product_key}"),
                        help=f"{t('new_order.restore') if is_striked else t('new_order.cross_out')} {name[:20]}",
                        type="secondary"
                    ):
                        if is_striked:
                            striked_products.discard(product_key)
                        else:
                            striked_products.add(product_key)
                        st.session_state[S("striked_products")] = striked_products
                        st.rerun(scope="fragment")  # Only rerun this fragment!

                # Separator line
                st.markdown('<div style="border-bottom: 1px dotted rgba(0,0,0,0.1); margin: 0;"></div>', unsafe_allow_html=True)

    # Call the fragment
    parsed_df = st.session_state.get(S("parsed_df"))
    order_chat = st.session_state.get(S("order_chat"), []) or []

    if isinstance(parsed_df, pd.DataFrame) and not parsed_df.empty:
        render_product_list()

        


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
            with st.spinner(t("new_order.transcribing")):
                if asr_backend == "OpenAI Whisper API":
                    transcript = asr_openai_whisper(audio_bytes, catalog_prompt_names, language=effective_lang_code)
                elif asr_backend == "Faster-Whisper (local)":
                    transcript = asr_faster_whisper(audio_bytes, catalog_prompt_names, language=effective_lang_code)
                elif asr_backend == "Google Speech-to-Text":
                    transcript = asr_google(audio_bytes, catalog_prompt_names, language=effective_lang_code)
                else:
                    transcript = ""

            st.session_state[S("last_audio_hash")] = audio_hash
            # Clear audio bytes to avoid recomputing hash and bloating session state on every rerun
            st.session_state[S("audio_bytes")] = b""
            transcript = cleanup_asr_transcript(transcript)
            transcript = " | ".join(tokenize_items(transcript))
            append_message("asr", transcript)
            st.rerun()
        except Exception as e:
            st.error(t("new_order.transcription_err", err=str(e)))


    # =========================================================
    # WhatsApp-like fixed bottom bar with AUDIO INPUT IN DICTAR SLOT
    # =========================================================
    st.session_state.setdefault(S("wa_text_input"), "")

    # Draft selector check (rendered as floating container below)
    parsed_df = st.session_state.get(S("parsed_df"))
    show_add_button = isinstance(parsed_df, pd.DataFrame) and not parsed_df.empty

    st.markdown('<div class="voi-bottom-wrap"><div class="voi-bottom-inner">', unsafe_allow_html=True)
    
    

    # float_init()

    # -------------------------------
    # state
    # -------------------------------
    if "show_micro" not in st.session_state:
        st.session_state.show_micro = False

    if "show_composer" not in st.session_state:
        st.session_state.show_composer = False

    if "show_product_adder" not in st.session_state:
        st.session_state.show_product_adder = False

    if "product_adder_fullpage" not in st.session_state:
        st.session_state.product_adder_fullpage = False

    if "fab_menu_open" not in st.session_state:
        st.session_state.fab_menu_open = False

    # =========================================================
    # FAB floating menu — isolated in a fragment for fast toggle
    # =========================================================
    @st.fragment
    def render_fabs(show_add_button):
        # Spacing knobs
        SIDE_PAD = "0.55rem"
        FAB_RIGHT = "1.10rem"
        FAB_SIZE = "3.2rem"
        FAB_GAP = "3.8rem"
        FAB_BASE = "5.5rem"

        FAB_GLASS_CSS = f"""
        padding: 0;
        & button {{
            width: {FAB_SIZE} !important;
            height: {FAB_SIZE} !important;
            min-height: {FAB_SIZE} !important;
            border-radius: 50% !important;
            background: rgba(255, 255, 255, 0.35) !important;
            backdrop-filter: saturate(180%) blur(20px) !important;
            -webkit-backdrop-filter: saturate(180%) blur(20px) !important;
            border: 1px solid rgba(255, 255, 255, 0.55) !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.10),
                        0 0 0 0.5px rgba(255, 255, 255, 0.4) inset !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 1.25rem !important;
            transition: transform 0.18s ease, box-shadow 0.18s ease !important;
        }}
        & button:hover {{
            transform: scale(1.08) !important;
            box-shadow: 0 6px 24px rgba(0, 0, 0, 0.15),
                        0 0 0 0.5px rgba(255, 255, 255, 0.5) inset !important;
            background: rgba(255, 255, 255, 0.50) !important;
        }}
        & button:active {{
            transform: scale(0.95) !important;
        }}
        """

        # =========================================================
        # 0) FAB menu trigger (⋯ / ✕) — fragment rerun only
        # =========================================================
        fab_menu_container = st.container()
        with fab_menu_container:
            menu_label = "✕" if st.session_state.fab_menu_open else "⋯"
            if st.button(menu_label, key="fab_menu_toggle", help=t("new_order.menu")):
                st.session_state.fab_menu_open = not st.session_state.fab_menu_open
                st.rerun(scope="fragment")  # fast: only reruns this fragment

        fab_menu_css = float_css_helper(
            right=FAB_RIGHT,
            bottom=FAB_BASE,
            width="auto",
            z_index="10001",
        )
        fab_menu_css += FAB_GLASS_CSS
        fab_menu_container.float(fab_menu_css)

        # Override: trigger button is bigger and always-visible (dark backdrop)
        st.html("""<style>
        .st-key-fab_menu_toggle button {
            width: 4rem !important;
            height: 4rem !important;
            min-height: 4rem !important;
            font-size: 1.7rem !important;
            background: rgba(18, 18, 18, 0.88) !important;
            backdrop-filter: saturate(180%) blur(24px) !important;
            -webkit-backdrop-filter: saturate(180%) blur(24px) !important;
            border: 1.5px solid rgba(255,255,255,0.18) !important;
            box-shadow: 0 8px 32px rgba(0,0,0,0.40),
                        0 0 0 1px rgba(255,255,255,0.10) inset !important;
            color: #ffffff !important;
        }
        /* Streamlit puts button text inside <p> — target it explicitly */
        .st-key-fab_menu_toggle button p,
        .st-key-fab_menu_toggle button span {
            color: #ffffff !important;
        }
        .st-key-fab_menu_toggle button:hover {
            background: rgba(40, 40, 40, 0.92) !important;
            transform: scale(1.06) !important;
            box-shadow: 0 12px 40px rgba(0,0,0,0.50) !important;
        }
        .st-key-fab_menu_toggle button:active {
            transform: scale(0.94) !important;
        }
        </style>""")

        # =========================================================
        # 0.1) Expanded FABs (only when menu is open)
        # =========================================================
        if st.session_state.fab_menu_open:
            # Product Adder FAB (position 5 — topmost)
            fab_product_container = st.container()
            with fab_product_container:
                if st.button("➕", key="smart_product_add_fab", help=t("new_order.add_products")):
                    st.session_state.product_adder_fullpage = True
                    st.session_state.show_micro = False
                    st.session_state.show_composer = False
                    st.session_state.show_draft_selector = False
                    st.session_state.fab_menu_open = False
                    st.rerun()

            fab_product_css = float_css_helper(
                right=FAB_RIGHT,
                bottom=f"calc({FAB_BASE} + {FAB_GAP} * 4)",
                width="auto",
                z_index="10000",
            )
            fab_product_css += FAB_GLASS_CSS
            fab_product_container.float(fab_product_css)

            # Mic FAB (position 3)
            fab_mic_container = st.container()
            with fab_mic_container:
                if st.button("🎙️", key="smart_add_fab"):
                    st.session_state.show_micro = True
                    st.session_state.show_composer = False
                    st.session_state.show_draft_selector = False
                    st.session_state.fab_menu_open = False
                    st.rerun()

            fab_mic_css = float_css_helper(
                right=FAB_RIGHT,
                bottom=f"calc({FAB_BASE} + {FAB_GAP} * 3)",
                width="auto",
                z_index="10000",
            )
            fab_mic_css += FAB_GLASS_CSS
            fab_mic_container.float(fab_mic_css)

            # Composer FAB (position 2)
            if not st.session_state.show_composer:
                fab_composer_container = st.container()
                with fab_composer_container:
                    if st.button("✏️", key="smart_composer_fab", help=t("new_order.write_order")):
                        st.session_state.show_composer = True
                        st.session_state.show_micro = False
                        st.session_state.show_draft_selector = False
                        st.session_state.fab_menu_open = False
                        st.rerun()

                fab_composer_css = float_css_helper(
                    right=FAB_RIGHT,
                    bottom=f"calc({FAB_BASE} + {FAB_GAP} * 2)",
                    width="auto",
                    z_index="10000",
                )
                fab_composer_css += FAB_GLASS_CSS
                fab_composer_container.float(fab_composer_css)

        # =========================================================
        # 1) MIC OVERLAY (audio only)
        # =========================================================
        if st.session_state.show_micro:
            mic_container = st.container()
            with mic_container:
                with st.container():
                    col_title, col_close = st.columns([4, 1])
                    with col_title:
                        st.markdown(f"**{t('new_order.audio_label')}**")
                    with col_close:
                        if st.button("✕", key=K("close_mic"), help=t("action.close")):
                            st.session_state.show_micro = False
                            st.rerun()

                audio_file = st.audio_input("", key=K("audio_msg"), label_visibility="collapsed")

                if st.button(t("action.send_audio"), key=K("btn_send_audio"), use_container_width=True, type="primary"):
                    if audio_file is not None:
                        st.session_state[S("audio_bytes")] = audio_file.read()
                    st.session_state.show_micro = False
                    st.rerun()

            mic_overlay_css = float_css_helper(
                left=SIDE_PAD,
                right=SIDE_PAD,
                bottom="10.0rem",
                width="auto",
                z_index="10001",
            )
            mic_overlay_css += """
            background: rgba(255,255,255,.98);
            backdrop-filter: saturate(180%) blur(16px);
            border: 1px solid rgba(148,163,184,.45);
            border-radius: 20px;
            padding: 14px 16px;
            box-shadow: 0 16px 48px rgba(2,6,23,.20), 0 0 0 1px rgba(255,255,255,.5) inset;
            max-width: 400px;
            margin: 0 auto;
            """
            mic_container.float(mic_overlay_css)

        # =========================================================
        # 4) FLOATING WHATSAPP COMPOSER (FAB-STYLE)
        # =========================================================
        typed = ""
        send_clicked = False

        if st.session_state.show_composer:
            wa_bar = st.container()
            with wa_bar:
                with st.container():
                    col_title, col_close = st.columns([4, 1])
                    with col_title:
                        st.markdown(f"**✏️ {t('new_order.write_order')}**")
                    with col_close:
                        if st.button("✕", key=K("close_composer"), help=t("action.close")):
                            st.session_state.show_composer = False
                            st.rerun()

                with st.form(key=K("wa_compose_form"), clear_on_submit=True):
                    with st.container():
                        c1, c2 = st.columns([4, 1])
                        with c1:
                            typed = st.text_input(
                                "",
                                placeholder=t("new_order.compose_placeholder"),
                                key=K("wa_text_input_field"),
                                label_visibility="collapsed",width=450
                            )
                        with c2:
                            send_clicked = st.form_submit_button("➤", use_container_width=True, key=K("btn_send_to_notes"),width=80)

            wa_css = float_css_helper(
                left=SIDE_PAD,
                right="1.10rem",
                bottom="8.0rem",
                width="auto",
                z_index="9999",
            )
            wa_css += """
            background: rgba(255,255,255,.98);
            backdrop-filter: saturate(180%) blur(16px);
            border: 1px solid rgba(148,163,184,.45);
            border-radius: 20px;
            padding: 14px 16px calc(14px + env(safe-area-inset-bottom));
            box-shadow: 0 16px 48px rgba(2,6,23,.20), 0 0 0 1px rgba(255,255,255,.5) inset;
            max-width: 600px;
            margin: 0 auto;
            """
            wa_css += """
            div[data-testid="column"] {
                flex-shrink: 1 !important;
                min-width: 0 !important;
            }
            div[data-testid="stHorizontalBlock"] {
                flex-wrap: nowrap !important;
                display: flex !important;
                gap: 8px !important;
            }
            div[data-testid="stTextInput"] {
                min-width: 0 !important;
                flex: 1 !important;
            }
            div[data-testid="stTextInput"] input {
                min-width: 0 !important;
                width: 100% !important;
                font-size: 0.85rem !important;
                padding: 8px 10px !important;
                height: auto !important;
            }
            div[data-testid="stTextInput"] input::placeholder {
                font-size: 0.82rem !important;
            }
            button[kind="formSubmit"] {
                min-width: 40px !important;
                max-width: 50px !important;
                white-space: nowrap !important;
                padding: 8px !important;
            }
            @media (max-width: 640px) {
                div[data-testid="stHorizontalBlock"] { gap: 6px !important; }
                button[kind="formSubmit"] {
                    min-width: 36px !important;
                    max-width: 44px !important;
                    padding: 6px !important;
                    font-size: 1.1rem !important;
                }
            }
            """
            wa_bar.float(wa_css)

        # =========================================================
        # 4.5) FLOATING DELETE BUTTON FAB (inside menu)
        # =========================================================
        clear_clicked = False
        if st.session_state.fab_menu_open:
            fab_delete_container = st.container()
            with fab_delete_container:
                clear_clicked = st.button("🗑️", key=K("btn_clear_notes"), help=t("new_order.clear_all"))

            fab_delete_css = float_css_helper(
                right=FAB_RIGHT,
                bottom=f"calc({FAB_BASE} + {FAB_GAP} * 1)",
                width="auto",
                z_index="10000",
            )
            fab_delete_css += FAB_GLASS_CSS
            fab_delete_container.float(fab_delete_css)

        # =========================================================
        # 🎯 FLOATING DRAFT SELECTOR FAB + PANEL
        # Always visible (left side, bottom bar level) when products exist
        # =========================================================
        st.session_state.setdefault("show_draft_selector", False)

        # Pill CSS for the draft button (not circular like FAB)
        DRAFT_PILL_CSS = """
        padding: 0;
        & button {
            height: 3.2rem !important;
            min-height: 3.2rem !important;
            border-radius: 1.6rem !important;
            padding: 0 1.1rem !important;
            background: rgba(255, 255, 255, 0.35) !important;
            backdrop-filter: saturate(180%) blur(20px) !important;
            -webkit-backdrop-filter: saturate(180%) blur(20px) !important;
            border: 1px solid rgba(255, 255, 255, 0.55) !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.10),
                        0 0 0 0.5px rgba(255, 255, 255, 0.4) inset !important;
            font-size: 0.9rem !important;
            white-space: nowrap !important;
            transition: transform 0.18s ease, box-shadow 0.18s ease !important;
        }
        & button:hover {
            transform: scale(1.04) !important;
            background: rgba(255, 255, 255, 0.52) !important;
            box-shadow: 0 6px 24px rgba(0, 0, 0, 0.14) !important;
        }
        & button:active {
            transform: scale(0.96) !important;
        }
        """

        if show_add_button:
            if not st.session_state.show_draft_selector:
                fab_draft_container = st.container()
                with fab_draft_container:
                    if st.button(t("new_order.draft_label"), key=K("fab_draft_select"), help=t("new_order.select_draft_help")):
                        st.session_state.show_draft_selector = True
                        st.session_state.show_micro = False
                        st.session_state.show_composer = False
                        st.session_state.fab_menu_open = False
                        st.rerun()

                fab_draft_css = float_css_helper(
                    left=SIDE_PAD,
                    bottom=FAB_BASE,
                    width="auto",
                    z_index="10000",
                )
                fab_draft_css += DRAFT_PILL_CSS
                fab_draft_container.float(fab_draft_css)

            if st.session_state.show_draft_selector:
                drafts_list = load_venue_drafts(venue_id, _refresh_token=get_drafts_refresh_token())

                draft_panel = st.container()
                with draft_panel:
                    col_title, col_close = st.columns([4, 1])
                    with col_title:
                        st.caption(t("msg.select_draft"))
                    with col_close:
                        if st.button("✕", key=K("close_draft_selector"), help=t("action.close")):
                            st.session_state.show_draft_selector = False
                            st.rerun()

                    if drafts_list:
                        draft_options = []
                        draft_ids = []
                        for draft in drafts_list:
                            title = draft.title or t("new_order.no_title")
                            date = draft.created_at.strftime('%d/%m %H:%M')
                            draft_options.append(f"#{draft.id} {title} · {date}")
                            draft_ids.append(int(draft.id))

                        chosen_idx = st.radio(
                            t("new_order.choose_colon"),
                            range(len(draft_options)),
                            format_func=lambda i: draft_options[i],
                            key=K("draft_quick_select"),
                            label_visibility="collapsed"
                        )

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(t("action.add_here"), key=K("confirm_quick"), use_container_width=True, type="primary"):
                                st.session_state[S("selected_draft_id")] = draft_ids[chosen_idx]
                                st.session_state[S("trigger_add")] = True
                                st.session_state.show_draft_selector = False
                                st.rerun()
                        with col2:
                            if st.button(t("new_order.new_label"), key=K("new_quick"), use_container_width=True):
                                st.session_state[S("selected_draft_id")] = -1
                                st.session_state[S("trigger_add")] = True
                                st.session_state.show_draft_selector = False
                                st.rerun()
                    else:
                        if st.button(t("action.new_draft"), key=K("new_quick"), use_container_width=True, type="primary"):
                            st.session_state[S("selected_draft_id")] = -1
                            st.session_state[S("trigger_add")] = True
                            st.session_state.show_draft_selector = False
                            st.rerun()

                draft_panel_css = float_css_helper(
                    left=SIDE_PAD,
                    right=SIDE_PAD,
                    bottom="5.75rem",
                    width="auto",
                    z_index="10001",
                )
                draft_panel_css += """
                background: rgba(255,255,255,.98);
                backdrop-filter: saturate(180%) blur(16px);
                border: 1px solid rgba(148,163,184,.45);
                border-radius: 20px;
                padding: 14px 16px;
                box-shadow: 0 16px 48px rgba(2,6,23,.20), 0 0 0 1px rgba(255,255,255,.5) inset;
                max-width: 400px;
                margin: 0 auto;
                """
                draft_panel.float(draft_panel_css)

        # Spacer so page content isn't hidden behind picker + add + composer
        # Spacer so page content isn't hidden behind bottom UI
        base = 20  # always keep a bit of room for the fixed bottom bar

        extra = 0
        if st.session_state.get("show_composer") or st.session_state.get("show_micro") or st.session_state.get("show_draft_selector"):
            extra = 260  # only when overlays/panels are open

        st.markdown(f"<div style='height:{base + extra}px'></div>", unsafe_allow_html=True)

        # =========================================================
        # 5) ACTIONS: Notas / preview pipeline
        # =========================================================
        if clear_clicked:
            reset_notes_only(clear_resolved_picks=False)
            st.session_state.show_composer = False
            st.session_state.fab_menu_open = False
            st.rerun()

        if send_clicked and typed and typed.strip():
            append_message("user", typed.strip())
            st.session_state.show_composer = False
            st.rerun()

    render_fabs(show_add_button)

    # =========================================================
    # 6) ADD TO BORRADOR LOGIC (triggered from draft selector popover)
    # =========================================================
    trigger_add = st.session_state.get(S("trigger_add"), False)
    if trigger_add:
        # Clear the trigger flag
        if st.session_state.get(S("trigger_add"), False):
            st.session_state[S("trigger_add")] = False

        parsed_df = st.session_state.get(S("parsed_df"))

        if not isinstance(parsed_df, pd.DataFrame) or parsed_df.empty:
            st.warning(t("msg.nothing_to_save_yet"))
            st.stop()

        # Filter out striked products (optimized vectorized approach)
        striked_products = st.session_state.get(S("striked_products"), set())
        df_filtered = parsed_df.copy()

        if striked_products and not df_filtered.empty:
            # Fully vectorized: create product keys without apply()
            names = df_filtered['matched_name'].fillna(df_filtered['spoken_name']).fillna('').astype(str).str.strip()
            quantities = df_filtered['quantity'].astype(str)
            indices = df_filtered.index.astype(str)

            # Create product keys efficiently
            product_keys = indices + '_' + names + '_' + quantities

            # Vectorized filter (much faster than iterrows)
            keep_mask = ~product_keys.isin(striked_products)
            df_filtered = df_filtered[keep_mask]

        if df_filtered.empty:
            st.warning(t("msg.all_items_crossed"))
            st.stop()

        df_to_use = apply_unit_choice(df_filtered)
        actor = current_actor()

        # Get all drafts (cached)
        drafts = load_venue_drafts(venue_id, _refresh_token=get_drafts_refresh_token())

        target_id: int = 0

        # Check if user already selected from popover
        selected_from_popover = st.session_state.get(S("selected_draft_id"))

        if selected_from_popover == -1:
            new_id = _create_draft_and_insert_lines(venue_id=venue_id, actor=actor, df=df_to_use)
            if new_id:
                target_id = int(new_id)
                st.session_state.pop(S("selected_draft_id"), None)

        elif selected_from_popover:
            target_id = int(selected_from_popover)
            st.session_state.pop(S("selected_draft_id"), None)

        elif active_draft_id:
            target_id = int(active_draft_id)

        elif len(drafts) == 1:
            target_id = int(drafts[0].id)
            _set_active_draft(target_id)

        else:
            new_id = _create_draft_and_insert_lines(venue_id=venue_id, actor=actor, df=df_to_use)
            if new_id:
                target_id = int(new_id)

        if target_id:
            _add_lines_to_existing_draft(
                venue_id=venue_id,
                order_id=int(target_id),
                actor=actor,
                df=df_to_use,
            )

            _set_active_draft(target_id)
            bump_orders_refresh_token()
            bump_drafts_refresh_token()  # Invalidate draft cache

            st.success(t("new_order.added_to_draft", n=str(target_id)))
            st.session_state[S("striked_products")] = set()
            reset_notes_only(do_rerun=False)
            time.sleep(0.3)
            _go_orders(target_id)

        # NOTE: You have a duplicated second "determine draft target" block in your original snippet.
        # Keep ONLY one of them to avoid double-execution and confusing state.



