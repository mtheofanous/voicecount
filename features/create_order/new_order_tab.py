from __future__ import annotations

import time
import hashlib
import re
from typing import Dict, List, Optional, Any

import pandas as pd
import streamlit as st
from sqlmodel import Session, select, and_
from datetime import datetime
from domain.models import Product, Order, OrderLine
from core.db import engine
from core.config import get_database_url
from features.utils.asr_google import asr_google

from features.manage_orders.orders import current_actor
from features.utils.voice_and_orders_utils import *

# ✅ single source of truth for normalization + unit synonyms
from core.normalization import normalize_text, normalize_unit, UNIT_SYNONYMS


def get_session() -> Session:
    return Session(engine)


def unit_dropdown_options() -> List[str]:
    """
    Canonical unit options derived from UNIT_SYNONYMS.
    Example output: ['box', 'g', 'kg', 'pack', 'unit']
    """
    return sorted(set(UNIT_SYNONYMS.values()))


def resolve_venue_id(passed_venue_id: Optional[int]) -> int:
    """
    Prefer session active venue (most accurate in your multi-venue UI),
    but fall back to the function argument if needed.
    """
    sid = st.session_state.get("active_venue_id")
    if sid:
        return int(sid)
    if passed_venue_id:
        return int(passed_venue_id)
    st.error("No active venue selected")
    st.stop()
    raise RuntimeError("Unreachable")


def add_provider_column(sess: Session, df: pd.DataFrame, *, venue_id: Optional[int] = None) -> pd.DataFrame:
    """
    Adds/updates a 'provider' column based on matched_name -> Product.provider_name.
    Accent/diacritics-insensitive via normalize_text.
    """
    out = df.copy()

    if "provider" not in out.columns:
        out["provider"] = ""

    q = select(Product)
    if venue_id is not None:
        q = q.where(Product.venue_id == int(venue_id))

    products = sess.exec(q).all()

    # normalized name -> provider_name
    name_to_provider: Dict[str, str] = {}
    for p in products:
        k = normalize_text(getattr(p, "name", "") or "")
        if not k:
            continue
        name_to_provider[k] = (getattr(p, "provider_name", "") or "")

    if "matched_name" not in out.columns:
        return out

    def _provider(matched_name: Any) -> str:
        k = normalize_text(matched_name)
        return name_to_provider.get(k, "")

    out["provider"] = out["matched_name"].apply(_provider).astype("string")
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


def _split_aliases_cell(cell: str) -> List[str]:
    # Admin uses " | " join; be tolerant to commas/semicolons/newlines too.
    raw = str(cell or "").replace("\n", " ")
    parts: List[str] = []
    for chunk in raw.split("|"):
        for sub in re.split(r"[,;]+", chunk):
            v = normalize_text(sub)
            if v:
                parts.append(v)
    # stable dedupe
    seen = set()
    out: List[str] = []
    for a in parts:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def build_alias_indexes(products: List[Product]) -> Dict[str, Any]:
    """Build alias indexes from Product.aliases (ADMIN3 enrichment).

    Returns:
        {
          'alias_to_products': dict(alias_norm -> [product_name...]),
          'token_to_products': dict(token -> set(product_name...)),
        }
    """
    alias_to_products: Dict[str, List[str]] = {}
    token_to_products: Dict[str, set] = {}

    for p in products:
        names_for_p = [p.name]
        # include product name itself as an alias
        aliases = _split_aliases_cell(getattr(p, "aliases", "") or "")
        aliases.append(normalize_text(p.name))
        # also include provider name tokens (optional)
        prov = normalize_text(getattr(p, "provider_name", "") or "")
        if prov:
            aliases.append(prov)

        for a in aliases:
            if not a:
                continue
            alias_to_products.setdefault(a, [])
            if p.name not in alias_to_products[a]:
                alias_to_products[a].append(p.name)

            for tok in a.split():
                if len(tok) < 2:
                    continue
                token_to_products.setdefault(tok, set()).add(p.name)

    return {"alias_to_products": alias_to_products, "token_to_products": token_to_products}


def alias_suggestions(
    query_norm: str,
    *,
    alias_to_products: Dict[str, List[str]],
    token_to_products: Dict[str, set],
    limit: int = 20,
) -> List[str]:
    """Return candidate product *names* using alias indexes.

    Ranking:
      1) exact alias matches (all products linked to that alias)
      2) token hits scored by overlap count
    """
    q = normalize_text(query_norm)
    if not q:
        return []

    # 1) exact alias
    exact = alias_to_products.get(q, [])
    if exact:
        return exact[:limit]

    # 2) token overlap
    toks = [t for t in q.split() if len(t) >= 2]
    if not toks:
        return []

    scores: Dict[str, int] = {}
    for t in toks:
        for pname in token_to_products.get(t, set()):
            scores[pname] = scores.get(pname, 0) + 1

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0].lower()))
    return [p for (p, _s) in ranked[:limit]]


def new_order_tab(venue_id: int, role: str | None = None) -> None:
    """
    Voice + typed new order tab (WhatsApp style).
    - Chat capture (typed + audio ASR)
    - Parse into structured rows
    - Edit rows (with Unit dropdown derived from UNIT_SYNONYMS)
    - Save as new draft OR add to existing draft
    - After save/add: resets this tab ("clean again")
    """
    # ---------------------------
    # Venue + namespacing (prevents key collisions)
    # ---------------------------
    venue_id = resolve_venue_id(venue_id)

    # # Debug (you can remove later)
    # st.write("Nuevo pedido venue:", venue_id)
    # st.write("DB URL:", get_database_url())
    # st.write("Active venue:", st.session_state.get("active_venue_id"))

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
    st.session_state.setdefault(S("adding_to_draft_mode"), False)
    st.session_state.setdefault(S("audio_widget_key"), f"{NS}asr_audio_in_{int(time.time())}")
    st.session_state.setdefault(S("order_chat"), [])  # list[dict]: {"role":"user"|"asr", "text": str}
    st.session_state.setdefault(S("audiorecorder_key"), K("audiorecorder_v1"))

    # ---------------------------
    # Helpers
    # ---------------------------
    def reset_new_order(used_component_recorder: bool) -> None:
        st.session_state[S("transcript_area")] = ""
        st.session_state[S("last_audio_hash")] = None
        st.session_state.pop(S("parsed_df"), None)
        st.session_state.pop(S("parse_candidates_df"), None)
        st.session_state.pop(S("finalize_parse_pending"), None)
        st.session_state[S("adding_to_draft_mode")] = False
        st.session_state[S("order_chat")] = []

        st.session_state.pop(K("parse_editor"), None)
        st.session_state.pop(K("choose_draft_select"), None)

        st.session_state[S("audio_widget_key")] = f"{NS}asr_audio_in_{int(time.time())}"
        st.session_state[S("audiorecorder_key")] = f"{K('audiorecorder')}_{int(time.time())}"

        st.rerun()


    def bump_orders_refresh_token() -> None:
        """Notify the Orders tab that DB data changed (draft updated/created).
        Orders uses a per-venue refresh token key to clear edit buffers and reload DB truth.
        """
        k = f"orders_refresh_token_{venue_id}"
        st.session_state[k] = int(st.session_state.get(k, 0)) + 1


    
    def finalize_candidates_to_df(
        candidates_df: pd.DataFrame,
        *,
        name_to_product: Dict[str, Any],
        venue_id: int,
    ) -> pd.DataFrame:
        """
        Aggregate parse candidates (sum quantities of same product), normalize units,
        add provider column, and return final df ready for editor / saving.
        """
        df_in = candidates_df.copy()

        merged: Dict[Any, Dict[str, Any]] = {}
        for _, r in df_in.iterrows():
            matched_name = safe_str(r.get("matched_name")).strip()
            qty = float(r.get("quantity") or 0.0)
            unit_val = normalize_unit(r.get("unit")) or "unit"
            conf = float(r.get("confidence") or 0.0)
            spoken = safe_str(r.get("spoken_name")).strip()
            unit_custom = safe_str(r.get("unit_custom"))

            if not matched_name and qty == 0.0 and not unit_val:
                continue

            prod_obj = name_to_product.get(matched_name) if matched_name else None
            pid = prod_obj.id if prod_obj else None
            key = ("pid", pid) if pid is not None else ("spoken", normalize_text(spoken))

            if key not in merged:
                merged[key] = {
                    "spoken_name": spoken,
                    "matched_name": matched_name or None,
                    "confidence": conf,
                    "quantity": qty,
                    "unit": unit_val,
                    "unit_custom": unit_custom,
                    "status": "OK" if matched_name else "Revisar",
                }
            else:
                merged[key]["quantity"] = float(merged[key]["quantity"] or 0.0) + qty
                merged[key]["confidence"] = max(float(merged[key]["confidence"] or 0.0), conf)

                if not safe_str(merged[key].get("unit_custom")) and unit_custom:
                    merged[key]["unit_custom"] = unit_custom

                if spoken and spoken not in (merged[key].get("spoken_name") or ""):
                    merged[key]["spoken_name"] = (merged[key]["spoken_name"] + " | " + spoken).strip(" | ")

        df = pd.DataFrame(list(merged.values()))
        if "unit_custom" not in df.columns:
            df["unit_custom"] = ""

        with get_session() as s:
            df = add_provider_column(s, df, venue_id=venue_id)

        return df

    def append_message(role_: str, text_: str) -> None:
        text_ = (text_ or "").strip()
        if not text_:
            return

        st.session_state[S("order_chat")].append({"role": role_, "text": text_})

        prev = (st.session_state.get(S("transcript_area")) or "").strip()
        st.session_state[S("transcript_area")] = (prev + "\n" + text_).strip() if prev else text_

    # ---------------------------
    # Sidebar options
    # ---------------------------
    with st.sidebar.expander("⚙️ Opciones de transcripción", expanded=False):
        asr_backend = st.selectbox(
            "Backend ASR",
            ["OpenAI Whisper API", "Google Speech-to-Text","Faster-Whisper (local)"],
            key=K("asr_backend"),
        )
        lang_code = st.selectbox(
            "Idioma",
            ["auto", "es", "el", "en"],
            index=0,
            key=K("lang_code"),
        )
        samplerate = st.selectbox(
            "Samplerate",
            [16000, 22050, 24000],
            index=0,
            key=K("samplerate"),
        )

    # ---------------------------
    # Load catalog
    # ---------------------------
    with get_session() as s:
        products = s.exec(
            select(Product)
            .where(Product.venue_id == venue_id)
            .order_by(Product.name.asc(), Product.provider_name.asc())
        ).all()

    if not products:
        st.warning("Primero crea tu catálogo en la pestaña 'Catálogo'.")
        return

    # ✅ catalog matching uses normalized text (accent-insensitive)
    catalog_names = [normalize_text(p.name) for p in products]
    catalog_norm_to_product = {normalize_text(p.name): p for p in products}
    name_to_product = {p.name: p for p in products}
    
    def build_google_phrases(products: List[Product]) -> List[str]:
        phrases: List[str] = []
        for p in products:
            # product name
            if getattr(p, "name", None):
                phrases.append(str(p.name).strip())

            # aliases stored like: "alias1 | alias2 | alias3"
            raw_aliases = (getattr(p, "aliases", "") or "")
            for a in raw_aliases.split("|"):
                a = a.strip()
                if a:
                    phrases.append(a)

        # stable dedupe (case-insensitive)
        seen = set()
        out: List[str] = []
        for x in phrases:
            k = x.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(x.strip())
        return out

    catalog_prompt_names = build_google_phrases(products)


    # ---------------------------
    # Alias indexes (from ADMIN3 enrichment)
    # ---------------------------
    alias_indexes = build_alias_indexes(products)
    alias_to_products = alias_indexes["alias_to_products"]
    token_to_products = alias_indexes["token_to_products"]



    # ---------------------------
    # CSS
    # ---------------------------
    st.markdown(
        """
        <style>
        .chat-bubble {
            display: inline-block;
            padding: 10px 12px;
            border-radius: 14px;
            margin: 4px 0;
            max-width: 86%;
            line-height: 1.35;
            font-size: 0.95rem;
            word-wrap: break-word;
            box-shadow: 0 1px 2px rgba(0,0,0,0.06);
        }
        .bubble-user { background: #DCF8C6; border-top-right-radius: 7px; }
        .bubble-asr  { background: #FFFFFF; border-top-left-radius: 7px; }
        [data-testid="stChatMessage"] { padding: 0.2rem 0.2rem; }
        [data-testid="stChatInput"] textarea { border-radius: 16px; }

        button[data-testid="stAudioRecorderStartButton"],
        button[data-testid="stAudioRecorderStopButton"] {
            font-size: 1.2rem !important;
            padding: 0.9rem 1.3rem !important;
            border-radius: 14px !important;
            min-width: 56px !important;
            height: 56px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================
    # HEADER
    # =========================================================
    # `vertical_alignment` exists only in newer Streamlit versions.
    # Avoid it for compatibility.
    hL, hR = st.columns([3, 1])
    with hL:
        st.subheader("🧾 Nuevo pedido")
    with hR:
        if st.button("🧹 Limpiar", key=K("btn_clear_all_chat")):
            reset_new_order(used_component_recorder=False)

    # =========================================================
    # CHAT PANEL
    # =========================================================
    with st.container(border=True):
        if not st.session_state[S("order_chat")]:
            st.info("Empieza escribiendo abajo o graba un audio 👇")

        for msg in st.session_state[S("order_chat")]:
            role_msg = msg.get("role")
            txt = msg.get("text", "")
            if role_msg == "user":
                with st.chat_message("user"):
                    st.markdown(f'<div class="chat-bubble bubble-user">{txt}</div>', unsafe_allow_html=True)
            else:
                with st.chat_message("assistant"):
                    st.markdown(f'<div class="chat-bubble bubble-asr">{txt}</div>', unsafe_allow_html=True)

    # =========================================================
    # INPUT ROW: MIC + TYPING
    # =========================================================
    mic_col, type_col = st.columns([2, 12])

    audio_bytes = b""
    used_component_recorder = False

    with mic_col:
        try:
            from audiorecorder import audiorecorder
            used_component_recorder = True

            audio_seg = audiorecorder(
                start_prompt="🎤",
                stop_prompt="⏹️",
                key=st.session_state[S("audiorecorder_key")],
            )
            if audio_seg is not None and len(audio_seg) > 0:
                audio_bytes = audio_seg.export(format="wav").read()

        except Exception:
            audio = mic_or_upload_audio(
                "🎤",
                key=st.session_state[S("audio_widget_key")],
                sample_rate=samplerate,
            )
            if audio is not None:
                audio_bytes = read_audio_bytes(audio)

    # Auto-transcribe when new audio arrives
    audio_hash = hashlib.sha1(audio_bytes).hexdigest() if audio_bytes else None
    if audio_bytes and audio_hash and audio_hash != st.session_state.get(S("last_audio_hash")):
        try:
            with st.spinner("Transcribiendo…"):
                if asr_backend == "OpenAI Whisper API":
                    transcript = asr_openai_whisper(audio_bytes, catalog_prompt_names, language=lang_code)

                elif asr_backend == "Faster-Whisper (local)":
                    transcript = asr_faster_whisper(audio_bytes, catalog_prompt_names, language=lang_code)

                elif asr_backend == "Google Speech-to-Text":
                    transcript = asr_google(audio_bytes, catalog_prompt_names, language=lang_code)

                else:
                    transcript = ""


            st.session_state[S("last_audio_hash")] = audio_hash
            transcript = cleanup_asr_transcript(transcript)
            append_message("asr", transcript)
            st.rerun()
        except Exception as e:
            st.error(f"Error transcribiendo: {e}")

    with type_col:
        typed = st.chat_input(
            "Escribe un ítem del pedido… (ej: 3 cajas cerveza)",
            key=K("chat_input"),
        )
        if typed:
            append_message("user", typed)
            st.rerun()

    # =========================================================
    # PARSE
    # =========================================================
    chat_msgs = [
        (m.get("text") or "").strip()
        for m in st.session_state.get(S("order_chat"), [])
        if (m.get("text") or "").strip()
    ]
    raw_text = (st.session_state.get(S("transcript_area")) or "").strip()
    has_any_text = bool(chat_msgs) or bool(raw_text)

    parse_clicked = st.button(
        "✨ Parsear pedido",
        type="primary",
        disabled=not has_any_text,
        key=K("btn_parse_order"),
    )

    # =========================
    # PARSE
    # =========================
    
    if parse_clicked:
        items: List[str] = []
        if chat_msgs:
            for msg_text in chat_msgs:
                items.extend(tokenize_items(msg_text))
        else:
            items = tokenize_items(raw_text)
            
            
        # ✅ second-pass: split fragments that contain multiple catalog aliases
        items2 = []
        for frag in items:
            items2.extend(split_fragment_by_catalog_aliases(frag, alias_to_products))

        items = items2

        parsed_rows: List[Dict[str, Any]] = []
        for frag in items:
            parsed = parse_item(frag)
            if not parsed:
                parsed_rows.append({
                    "spoken_name": frag,
                    "matched_name": None,
                    "confidence": 0.0,
                    "quantity": None,
                    "unit": "unit",
                    "unit_custom": "",
                    "suggestions": [],  # ✅ internal only
                    "status": "No interpretado",
                })
                continue

            name, qty, unit_raw = parsed
            name_norm = normalize_text(name)

            # 1) Try direct fuzzy match
            # 1) Alias-first match (ADMIN3 enriched aliases)
            suggestions_list: List[str] = []
            prod: Optional[Product] = None
            score: Optional[float] = None

            # Try exact alias (also on simple singular variants)
            alias_keys = [name_norm]
            name_sing_es = singularize_es(name_norm)
            if name_sing_es and name_sing_es != name_norm:
                alias_keys.append(name_sing_es)
            name_sing_el = singularize_el(name_norm)
            if name_sing_el and name_sing_el != name_norm:
                alias_keys.append(name_sing_el)

            exact_hits: List[str] = []
            for k in alias_keys:
                hits = alias_to_products.get(k, [])
                if hits:
                    exact_hits = hits
                    break

            if exact_hits:
                if len(exact_hits) == 1:
                    prod = name_to_product.get(exact_hits[0])
                    score = 100.0 if prod else None
                else:
                    # ambiguous alias: force user to choose
                    suggestions_list = exact_hits[:20]

            # 2) If still no product, fuzzy match against catalog names
            if prod is None and not suggestions_list:
                match_norm, score = fuzzy_match(name_norm, catalog_names)
                prod = catalog_norm_to_product.get(match_norm) if match_norm else None

                # fallback: try singular ES
                if prod is None:
                    if name_sing_es and name_sing_es != name_norm:
                        match_norm2, score2 = fuzzy_match(name_sing_es, catalog_names)
                        prod2 = catalog_norm_to_product.get(match_norm2) if match_norm2 else None
                        if prod2:
                            prod = prod2
                            match_norm, score = match_norm2, score2

                # fallback: try singular EL
                if prod is None:
                    if name_sing_el and name_sing_el != name_norm:
                        match_norm3, score3 = fuzzy_match(name_sing_el, catalog_names)
                        prod3 = catalog_norm_to_product.get(match_norm3) if match_norm3 else None
                        if prod3:
                            prod = prod3
                            match_norm, score = match_norm3, score3

            # unit normalization
            unit_val = normalize_unit(unit_raw) if unit_raw else ""
            if prod and getattr(prod, "unit", None):
                unit_val = normalize_unit(prod.unit)

            # 3) If no product, generate suggestions (alias token hits first, then fuzzy)
            if prod is None and not suggestions_list:
                alias_hits = alias_suggestions(
                    name_norm,
                    alias_to_products=alias_to_products,
                    token_to_products=token_to_products,
                    limit=20,
                )
                if alias_hits:
                    suggestions_list = alias_hits
                else:
                    top = suggest_matches(name_norm, catalog_names, limit=10)
                    suggestions_list = [cand for (cand, _sc) in top]

            parsed_rows.append({
                "spoken_name": name,
                "matched_name": prod.name if prod else None,
                "confidence": round(score or 0.0, 1),
                "quantity": qty,
                "unit": unit_val or "unit",
                "unit_custom": "",
                "suggestions": suggestions_list,  # ✅ internal only
                "status": "OK" if prod else ("Elegir" if suggestions_list else "Revisar"),
            })

        # ✅ Store candidates and resolve ambiguous items BEFORE building final df
        st.session_state[S("parse_candidates_df")] = pd.DataFrame(parsed_rows)
        st.session_state.pop(S("parsed_df"), None)
        st.session_state[S("finalize_parse_pending")] = True
        st.rerun()

    # =========================================================
    # Resolve ambiguous items (expander BEFORE final df)
    # =========================================================
    candidates_df = st.session_state.get(S("parse_candidates_df"))
    finalize_pending = st.session_state.get(S("finalize_parse_pending"), False)

    if finalize_pending and isinstance(candidates_df, pd.DataFrame) and not candidates_df.empty:
        def _needs_choice(row: pd.Series) -> bool:
            return (
                safe_str(row.get("status")) == "Elegir"
                and isinstance(row.get("suggestions"), list)
                and len(row.get("suggestions")) > 0
            )

        needs_mask = candidates_df.apply(_needs_choice, axis=1)
        needs_df = candidates_df[needs_mask].copy()

        # If no ambiguous rows, auto-finalize
        if needs_df.empty:
            st.session_state[S("finalize_parse_pending")] = False
            st.session_state[S("parsed_df")] = finalize_candidates_to_df(
                candidates_df,
                name_to_product=name_to_product,
                venue_id=venue_id,
            )
            st.rerun()

        # Otherwise show expander for user resolution
        with st.expander("🔎 Productos con varias opciones (elige una)", expanded=True):
            st.caption("Selecciona una opción por cada producto ambiguo y pulsa **OK** para finalizar el parse.")

            picks: Dict[int, str] = {}
            for idx, row in needs_df.iterrows():
                spoken = safe_str(row.get("spoken_name"))
                qty = row.get("quantity")
                unit = safe_str(row.get("unit") or "unit")
                opts: List[str] = row.get("suggestions") or []

                label = f"{spoken} — {qty or ''} {unit}".strip()
                pick = st.selectbox(
                    label,
                    options=opts,
                    index=0,
                    key=K(f"resolve_pick_{idx}"),
                )
                picks[int(idx)] = pick

            ok_clicked = st.button("✅ OK (finalizar parse)", type="primary", key=K("btn_finalize_parse"))
            if ok_clicked:
                df2 = candidates_df.copy()
                for idx, pick in picks.items():
                    df2.at[idx, "matched_name"] = pick
                    df2.at[idx, "status"] = "OK"
                    df2.at[idx, "confidence"] = max(float(df2.at[idx, "confidence"] or 0.0), 99.0)

                st.session_state[S("parse_candidates_df")] = df2
                st.session_state[S("finalize_parse_pending")] = False
                st.session_state[S("parsed_df")] = finalize_candidates_to_df(
                    df2,
                    name_to_product=name_to_product,
                    venue_id=venue_id,
                )
                st.rerun()

# =========================

    # =========================
    # RESULT EDITOR
    # =========================
    parsed_df = st.session_state.get(S("parsed_df"))
    if isinstance(parsed_df, pd.DataFrame) and not parsed_df.empty:
        unit_options = unit_dropdown_options()
        OTHER = "Other…"
        if OTHER not in unit_options:
            unit_options = unit_options + [OTHER]

        df_editor = parsed_df.copy()

        # Make sure unit values are canonical or "Other…"
        def _editor_unit(u: Any) -> str:
            u_norm = normalize_unit(u)
            return u_norm if u_norm in unit_options else OTHER

        df_editor["unit"] = df_editor["unit"].apply(_editor_unit)
        if "unit_custom" not in df_editor.columns:
            df_editor["unit_custom"] = ""

        edited = st.data_editor(
            df_editor,
            width="stretch",
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "confidence": st.column_config.NumberColumn("Confianza", help="0-100"),
                "quantity": st.column_config.NumberColumn("Cantidad"),
                "unit": st.column_config.SelectboxColumn("Unidad", options=unit_options),
                "unit_custom": st.column_config.TextColumn("Unidad (custom)", help="Usa esto si Unidad = Other…"),
                "matched_name": st.column_config.TextColumn("Producto (catálogo)"),
                "status": st.column_config.TextColumn("Estado", disabled=True),
                "provider": st.column_config.TextColumn("Proveedor (catálogo)", disabled=True),
            },
            key=K("parse_editor"),
        )

        # Apply unit choice + normalization
        edited = apply_unit_choice(edited)

        with get_session() as s:
            edited = add_provider_column(s, edited, venue_id=venue_id)

        st.session_state[S("parsed_df")] = edited

# =========================================================
    # Save actions
    # =========================================================
    has_parsed = (
        S("parsed_df") in st.session_state
        and isinstance(st.session_state[S("parsed_df")], pd.DataFrame)
        and not st.session_state[S("parsed_df")].empty
    )
    if not has_parsed:
        return

    df_to_use = st.session_state[S("parsed_df")].copy()
    df_to_use = apply_unit_choice(df_to_use)  # ✅ ensure canonical units right before saving

    b1, b2 = st.columns(2)
    save_draft_clicked = b1.button("💾 Nuevo borrador", type="primary", key=K("btn_save_new_draft"))
    add_to_existing_clicked = b2.button("➕ Añadir a borrador", key=K("btn_add_to_existing"))

    # ---------------- SAVE AS NEW DRAFT ----------------
    if save_draft_clicked:
        with get_session() as s:
            actor = current_actor()
            new_order = Order(
                status="draft",
                title=None,
                venue_id=venue_id,
                created_by=actor,
            )
            s.add(new_order)
            s.commit()
            s.refresh(new_order)
            oid = new_order.id

            merged_db: Dict[Any, Dict[str, Any]] = {}
            for _, row in df_to_use.iterrows():
                matched_name = safe_str(row.get("matched_name")).strip()
                qty = float(row.get("quantity") or 0.0)
                conf = float(row.get("confidence") or 0.0)

                unit_val = normalize_unit(row.get("unit")) or "unit"

                spoken_norm = normalize_text(row.get("spoken_name") or "")

                if not matched_name and qty == 0.0 and not unit_val:
                    continue

                prod_obj = (
                    s.exec(select(Product).where(Product.venue_id == venue_id, Product.name == matched_name)).first()
                    if matched_name else None
                )
                pid = prod_obj.id if prod_obj else None

                # Prefer product unit if present (normalized)
                if prod_obj and getattr(prod_obj, "unit", None):
                    unit_val = normalize_unit(prod_obj.unit) or unit_val

                key = ("pid", pid) if pid is not None else ("spoken", normalize_text(row.get("spoken_name") or ""))
                if key in merged_db:
                    merged_db[key]["quantity"] = float(merged_db[key]["quantity"] or 0.0) + qty
                    merged_db[key]["confidence"] = max(float(merged_db[key]["confidence"] or 0.0), conf)
                else:
                    merged_db[key] = dict(
                        venue_id=venue_id,
                        order_id=oid,
                        product_id=pid,
                        spoken_name=spoken_norm,
                        matched_name=(matched_name if matched_name else None),
                        confidence=conf,
                        quantity=qty,
                        unit=unit_val,
                    )

            for payload in merged_db.values():
                s.add(OrderLine(**payload))
            s.commit()

        st.success(f"Borrador guardado ✅ (ID {oid})")
        bump_orders_refresh_token()
        # Open this draft automatically in Orders tab
        st.session_state["orders_active_order_id"] = int(oid)

        reset_new_order(used_component_recorder=used_component_recorder)

    # ---------------- ADD TO EXISTING DRAFT ----------------
    # ---------------- ADD TO EXISTING DRAFT ----------------
    with get_session() as s:
        drafts = s.exec(
            select(Order)
            .where(Order.venue_id == venue_id, Order.status == "draft")
            .order_by(Order.created_at.desc())
        ).all()

    if add_to_existing_clicked:
        if not drafts:
            st.warning("No hay borradores disponibles. Crea uno nuevo primero.")
        else:
            st.session_state[S("adding_to_draft_mode")] = True

    if st.session_state.get(S("adding_to_draft_mode"), False) and drafts:
        draft_options = [(o.id, f"#{o.id} — {o.title or o.created_at.strftime('%Y-%m-%d %H:%M')}") for o in drafts]
        labels = [lbl for _, lbl in draft_options]
        ids = [oid for oid, _ in draft_options]

        chosen_label = st.selectbox("Borrador", options=labels, key=K("choose_draft_select"))
        selected_draft_id = ids[labels.index(chosen_label)]

        cA, cB = st.columns([1, 1])
        confirm_add = cA.button("✅ Confirmar", type="primary", key=K("confirm_add_lines"))
        cancel_add = cB.button("❌ Cancelar", key=K("cancel_add_lines"))

        if cancel_add:
            st.session_state[S("adding_to_draft_mode")] = False
            st.rerun()

        if confirm_add:
            oid = int(selected_draft_id)
            actor = current_actor()

            with get_session() as s:
                # ✅ Ensure order exists & belongs to this venue and is still a draft
                order = s.exec(
                    select(Order).where(
                        Order.id == oid,
                        Order.venue_id == venue_id,
                        Order.status == "draft",
                    )
                ).first()

                if not order:
                    st.error("El borrador seleccionado no existe o ya no es un borrador.")
                    st.session_state[S("adding_to_draft_mode")] = False
                    st.stop()

                for _, row in df_to_use.iterrows():
                    matched_name = safe_str(row.get("matched_name")).strip()
                    spoken_norm = normalize_text(row.get("spoken_name") or "")  # ✅ FIX: define per row

                    qty_to_add = float(row.get("quantity") or 0.0)
                    conf = float(row.get("confidence") or 0.0)
                    unit_val = normalize_unit(row.get("unit")) or "unit"

                    # ✅ Skip empty / zero qty
                    if qty_to_add <= 0.0 and not matched_name and not spoken_norm:
                        continue
                    if qty_to_add <= 0.0:
                        continue

                    prod_obj = (
                        s.exec(
                            select(Product).where(
                                Product.venue_id == venue_id,
                                Product.name == matched_name,
                            )
                        ).first()
                        if matched_name else None
                    )
                    pid = int(prod_obj.id) if (prod_obj and prod_obj.id is not None) else None

                    # Prefer product unit if present
                    if prod_obj and getattr(prod_obj, "unit", None):
                        unit_val = normalize_unit(prod_obj.unit) or unit_val

                    existing_line = None

                    # ✅ FIX: always scope by venue_id too
                    if pid is not None:
                        existing_line = s.exec(
                            select(OrderLine).where(
                                and_(
                                    OrderLine.venue_id == venue_id,
                                    OrderLine.order_id == oid,
                                    OrderLine.product_id == pid,
                                    # Merge only if unit matches (prevents accidental merges)
                                    OrderLine.unit == unit_val,
                                )
                            )
                        ).first()

                    if existing_line is None and matched_name:
                        existing_line = s.exec(
                            select(OrderLine).where(
                                and_(
                                    OrderLine.venue_id == venue_id,
                                    OrderLine.order_id == oid,
                                    OrderLine.matched_name == matched_name,
                                    OrderLine.unit == unit_val,
                                )
                            )
                        ).first()

                    if existing_line is None and (not matched_name) and spoken_norm:
                        existing_line = s.exec(
                            select(OrderLine).where(
                                and_(
                                    OrderLine.venue_id == venue_id,
                                    OrderLine.order_id == oid,
                                    OrderLine.product_id.is_(None),
                                    OrderLine.matched_name.is_(None),
                                    OrderLine.spoken_name == spoken_norm,
                                )
                            )
                        ).first()

                    if existing_line:
                        existing_line.quantity = float(existing_line.quantity or 0.0) + qty_to_add
                        existing_line.unit = normalize_unit(existing_line.unit) or unit_val
                        existing_line.confidence = max(float(existing_line.confidence or 0.0), conf)
                        existing_line.updated_at = datetime.utcnow()
                        existing_line.updated_by = actor
                        s.add(existing_line)
                    else:
                        s.add(
                            OrderLine(
                                venue_id=venue_id,
                                order_id=oid,
                                product_id=pid,
                                spoken_name=spoken_norm,
                                matched_name=(matched_name if matched_name else None),
                                provider=(getattr(prod_obj, "provider_name", None) or None) if prod_obj else None,
                                confidence=conf,
                                quantity=qty_to_add,
                                unit=unit_val,
                                updated_at=datetime.utcnow(),
                                updated_by=actor,
                            )
                        )

                # ✅ Update parent order audit (recommended)
                order.updated_at = datetime.utcnow()
                order.updated_by = actor
                if not order.created_by:
                    order.created_by = actor
                s.add(order)

                s.commit()

            st.success(f"Líneas añadidas ✅ (ID {oid}).")
            bump_orders_refresh_token()
            # Open this draft automatically in Orders tab
            st.session_state["orders_active_order_id"] = int(oid)
            st.session_state[S("adding_to_draft_mode")] = False

            reset_new_order(used_component_recorder=used_component_recorder)

