"""
features/manage_orders/orders.py

Refactored Orders tab (Streamlit) — simple, robust, production-oriented.

Goals
-----
- Clear UX: filter -> pick order -> do the next action (edit / send / receive).
- Robust DB access: small helper functions, safe defaults, minimal global state.
- Works with Auth → Venue email templates & invoice-style header.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlmodel import select

from core.db import get_session
from domain.models import Order, OrderLine, Product, Provider, ProviderReceipt

# Auth DB (venue templates + invoice header fields live there)
try:
    from features.auth_and_manage.auth_multi_tenant import get_auth_session, Venue  # type: ignore
except Exception:  # pragma: no cover
    get_auth_session = None  # type: ignore
    Venue = None  # type: ignore


# =============================================================================
# Backwards compatibility
# =============================================================================

def current_actor() -> str:
    """Return the identifier to stamp in audit fields.

    Older parts of the app (e.g. create_order/new_order_tab.py) import
    `current_actor` from this module.

    We prefer the authenticated user's email (multi-tenant auth), but fall back
    gracefully if auth isn't initialized.
    """
    try:
        from features.auth_and_manage.auth_multi_tenant import current_user  # type: ignore

        u = current_user() or {}
        email = (u.get("email") or "").strip()
        if email:
            return email
        full_name = (u.get("full_name") or u.get("name") or "").strip()
        if full_name:
            return full_name
        uid = u.get("id")
        if uid is not None:
            return f"user:{uid}"
    except Exception:
        pass

    # Last-resort fallback
    return (st.session_state.get("actor") or "system")


# =============================================================================
# Small utils
# =============================================================================

def _now() -> datetime:
    return datetime.utcnow()


def _safe_str(x: Any) -> str:
    return ("" if x is None else str(x)).strip()


def _split_first_pipe(value: Optional[str]) -> str:
    """Providers allow 'a@x.com | b@y.com'. Return first non-empty."""
    raw = _safe_str(value)
    if not raw:
        return ""
    for part in raw.split("|"):
        p = part.strip()
        if p:
            return p
    return ""


def _norm_provider(name: Optional[str]) -> str:
    return (_safe_str(name) or "(Sin proveedor)")


def _order_label(o: Order) -> str:
    title = _safe_str(getattr(o, "title", "")) or ""
    when = getattr(o, "created_at", None)
    when_s = when.strftime("%Y-%m-%d %H:%M") if when else ""
    base = f"#{int(o.id)}" if getattr(o, "id", None) is not None else "#—"
    if title:
        return f"{base} — {title}"
    return f"{base} — {when_s}" if when_s else base


def _status_label(status: str) -> str:
    m = {
        "draft": "Borrador",
        "ready_to_send": "Listo",
        "pending_receive": "Pendiente",
        "final": "Historial",
    }
    return m.get((status or "").strip().lower(), status or "—")


def _status_chip(status: str) -> str:
    s = (status or "").strip().lower()
    if s == "draft":
        return "📝 Borrador"
    if s == "ready_to_send":
        return "📤 Listo"
    if s == "pending_receive":
        return "📦 Pendiente"
    if s == "final":
        return "✅ Historial"
    return s or "—"


def _segmented_status_filter(key: str) -> str:
    options = ["draft", "ready_to_send", "pending_receive", "final"]
    labels = {
        "draft": "Borradores",
        "ready_to_send": "Listo",
        "pending_receive": "Pendiente",
        "final": "Historial",
    }
    return st.radio(
        "Estado",
        options,
        horizontal=True,
        key=key,
        format_func=lambda x: labels.get(x, x),
        label_visibility="collapsed",
    )


def _normalize_phone(raw: str, country_code: str) -> str:
    raw = _safe_str(raw)
    if not raw:
        return ""
    digits = "".join(ch for ch in raw if ch.isdigit() or ch == "+")
    if not digits:
        return ""
    if digits.startswith("+"):
        return digits
    digits2 = digits.lstrip("0")
    cc = _safe_str(country_code) or "+34"
    if not cc.startswith("+"):
        cc = "+" + cc
    return f"{cc}{digits2}"


# =============================================================================
# Data access (cached)
# =============================================================================

@st.cache_data(ttl=10, show_spinner=False, hash_funcs={type(lambda: None): lambda _: "session_fn"})
def _list_orders_cached(_get_session_fn, venue_id: int, refresh_token: int) -> List[Order]:
    _ = refresh_token
    with _get_session_fn() as s:
        return list(
            s.exec(
                select(Order)
                .where(Order.venue_id == venue_id)
                .order_by(Order.created_at.desc())
            ).all()
        )


@st.cache_data(ttl=60, show_spinner=False, hash_funcs={type(lambda: None): lambda _: "session_fn"})
def _products_cached(_get_session_fn, venue_id: int) -> List[Product]:
    with _get_session_fn() as s:
        return list(
            s.exec(
                select(Product)
                .where(Product.venue_id == venue_id)
                .order_by(Product.name.asc(), Product.provider_name.asc())
            ).all()
        )


@st.cache_data(ttl=10, show_spinner=False, hash_funcs={type(lambda: None): lambda _: "session_fn"})
def _order_lines_cached(_get_session_fn, order_id: int, refresh_token: int) -> List[OrderLine]:
    _ = refresh_token
    with _get_session_fn() as s:
        return list(
            s.exec(
                select(OrderLine)
                .where(OrderLine.order_id == order_id)
                .order_by(OrderLine.id.asc())
            ).all()
        )


@st.cache_data(ttl=60, show_spinner=False, hash_funcs={type(lambda: None): lambda _: "session_fn"})
def _providers_cached(_get_session_fn, venue_id: int) -> Dict[str, Provider]:
    """Map provider name -> Provider row."""
    with _get_session_fn() as s:
        rows = list(
            s.exec(select(Provider).where(Provider.venue_id == venue_id)).all()
        )
    out: Dict[str, Provider] = {}
    for p in rows:
        out[_norm_provider(getattr(p, "name", ""))] = p
    return out


@st.cache_data(ttl=10, show_spinner=False, hash_funcs={type(lambda: None): lambda _: "session_fn"})
def _provider_receipts_cached(_get_session_fn, order_id: int, refresh_token: int) -> Dict[str, ProviderReceipt]:
    _ = refresh_token
    with _get_session_fn() as s:
        rows = list(s.exec(select(ProviderReceipt).where(ProviderReceipt.order_id == order_id)).all())
    out: Dict[str, ProviderReceipt] = {}
    for r in rows:
        out[_norm_provider(getattr(r, "provider_name", ""))] = r
    return out


def _bump_refresh(venue_id: int) -> None:
    k = f"orders_refresh_token_{venue_id}"
    st.session_state[k] = int(st.session_state.get(k, 0)) + 1


def _refresh_token(venue_id: int) -> int:
    return int(st.session_state.get(f"orders_refresh_token_{venue_id}", 0))


# =============================================================================
# Venue templates (Auth DB)
# =============================================================================

@dataclass
class VenueTemplates:
    venue_name: str = ""
    owner_name: str = ""
    address: str = ""
    tax_number: str = ""
    email: str = ""
    phone: str = ""
    subject_tpl: str = ""
    opening_tpl: str = ""
    closing_tpl: str = ""


@st.cache_data(ttl=120, show_spinner=False)
def _load_venue_templates(venue_id: int, refresh_token: int) -> VenueTemplates:
    """
    Read venue metadata + templates from auth DB.
    Robust to missing auth module (returns empty templates).
    """
    _ = refresh_token
    if get_auth_session is None or Venue is None:
        return VenueTemplates()

    try:
        with get_auth_session() as s:  # type: ignore
            v = s.exec(select(Venue).where(Venue.id == venue_id)).first()
    except Exception:
        v = None

    if not v:
        return VenueTemplates()

    return VenueTemplates(
        venue_name=_safe_str(getattr(v, "name", "")),
        owner_name=_safe_str(getattr(v, "owner_name", "")),
        address=_safe_str(getattr(v, "address", "")),
        tax_number=_safe_str(getattr(v, "tax_number", "")),
        email=_safe_str(getattr(v, "email", "")),
        phone=_safe_str(getattr(v, "phone", "")),
        subject_tpl=_safe_str(getattr(v, "email_subject_tpl", "")),
        opening_tpl=_safe_str(getattr(v, "email_opening_tpl", "")),
        closing_tpl=_safe_str(getattr(v, "email_closing_tpl", "")),
    )


def _render_tpl(tpl: str, *, order_id: int, date_str: str, venue_name: str) -> str:
    if not _safe_str(tpl):
        return ""
    try:
        return tpl.format(order_id=order_id, date=date_str, venue_name=venue_name)
    except Exception:
        return tpl


def _invoice_header_text(v: VenueTemplates) -> str:
    """Plain-text invoice-style header (Cabecera)."""
    parts: List[str] = []
    if v.venue_name:
        parts.append(v.venue_name)
    if v.owner_name:
        parts.append(f"Titular: {v.owner_name}")
    if v.address:
        parts.append(f"Dirección: {v.address}")
    if v.tax_number:
        parts.append(f"NIF/CIF: {v.tax_number}")
    if v.email:
        parts.append(f"Email: {v.email}")
    if v.phone:
        parts.append(f"Tel: {v.phone}")
    return "\n".join(parts).strip()


def _venue_missing_required(v: VenueTemplates) -> List[str]:
    missing = []
    if not v.tax_number:
        missing.append("NIF/CIF (tax number)")
    if not v.address:
        missing.append("Dirección (address)")
    return missing


# =============================================================================
# Mutations (DB writes)
# =============================================================================

def _set_order_status(order_id: int, status: str, actor: str) -> None:
    with get_session() as s:
        o = s.exec(select(Order).where(Order.id == order_id)).first()
        if not o:
            return
        o.status = status
        o.updated_at = _now()
        o.updated_by = actor
        if status == "final":
            o.verified_at = _now()
            o.verified_by = actor
        s.add(o)
        s.commit()


def _delete_order(order_id: int) -> None:
    with get_session() as s:
        for ln in s.exec(select(OrderLine).where(OrderLine.order_id == order_id)).all():
            s.delete(ln)
        o = s.exec(select(Order).where(Order.id == order_id)).first()
        if o:
            s.delete(o)
        s.commit()


def _create_empty_draft(venue_id: int, actor: str) -> int:
    with get_session() as s:
        o = Order(
            venue_id=venue_id,
            status="draft",
            created_at=_now(),
            updated_at=_now(),
            created_by=actor,
            updated_by=actor,
            title=None,
            note=None,
        )
        s.add(o)
        s.commit()
        s.refresh(o)
        return int(o.id)


def _save_lines_from_editor(
    *,
    venue_id: int,
    order_id: int,
    actor: str,
    df: pd.DataFrame,
    products_by_id: Dict[int, Product],
) -> None:
    """
    Accepts an editor DF with columns:
      line_id (nullable), product_id, quantity, unit, delete
    """
    df2 = df.copy()

    def _pid_to_int(pid: Any) -> Optional[int]:
        """Robustly coerce data_editor product_id values (can be list/NA/str/float)."""
        if pid is None:
            return None
        if isinstance(pid, (list, tuple)):
            if not pid:
                return None
            pid = pid[0]
        try:
            if pd.isna(pid):
                return None
        except Exception:
            pass
        if isinstance(pid, int):
            return pid
        if isinstance(pid, float):
            if pd.isna(pid):
                return None
            return int(pid)
        s = str(pid).strip()
        if not s:
            return None
        try:
            return int(float(s))
        except Exception:
            return None


    def norm_unit(u: Any) -> str:
        u2 = _safe_str(u).lower()
        return u2 or "unidad"

    # sanitize
    df2["quantity"] = pd.to_numeric(df2.get("quantity"), errors="coerce").fillna(0.0)
    df2["unit"] = df2.get("unit").apply(norm_unit) if "unit" in df2.columns else "unidad"
    if "delete" not in df2.columns:
        df2["delete"] = False

    del_ids = [int(x) for x in df2.loc[df2["delete"] == True, "line_id"].dropna().tolist()]  # noqa: E712

    keep = df2.loc[df2["delete"] != True].copy()  # noqa: E712
    keep = keep.loc[keep["quantity"] > 0].copy()

    now = _now()
    with get_session() as s:
        if del_ids:
            for ln in s.exec(select(OrderLine).where(OrderLine.id.in_(del_ids))).all():
                s.delete(ln)

        for _, r in keep.iterrows():
            line_id = r.get("line_id")
            pid = _pid_to_int(r.get("product_id")) or 0
            if not pid:
                continue
            qty = float(r.get("quantity") or 0.0)
            unit = norm_unit(r.get("unit"))
            prod = products_by_id.get(pid)
            provider_name = _safe_str(getattr(prod, "provider_name", "")) if prod else ""

            if pd.notna(line_id):
                ln = s.exec(select(OrderLine).where(OrderLine.id == int(line_id))).first()
                if ln:
                    ln.product_id = pid
                    ln.quantity = qty
                    ln.unit = unit
                    ln.provider = provider_name or ln.provider
                    ln.updated_at = now
                    ln.updated_by = actor
                    s.add(ln)
            else:
                s.add(
                    OrderLine(
                        venue_id=venue_id,
                        order_id=order_id,
                        product_id=pid,
                        quantity=qty,
                        unit=unit,
                        provider=provider_name or None,
                        updated_at=now,
                        updated_by=actor,
                    )
                )

        o = s.exec(select(Order).where(Order.id == order_id)).first()
        if o:
            o.updated_at = now
            o.updated_by = actor
            if not o.created_by:
                o.created_by = actor
            s.add(o)

        s.commit()


def _upsert_provider_receipt(
    *,
    venue_id: int,
    order_id: int,
    provider_name: str,
    received: bool,
    actor: str,
    note: str = "",
) -> None:
    prov = _norm_provider(provider_name)
    now = _now()
    with get_session() as s:
        row = s.exec(
            select(ProviderReceipt).where(
                ProviderReceipt.order_id == order_id,
                ProviderReceipt.provider_name == prov,
            )
        ).first()

        if row:
            row.received = bool(received)
            row.received_at = now if received else None
            row.received_by = actor if received else None
            row.note = _safe_str(note) or None
            row.updated_at = now
            s.add(row)
        else:
            s.add(
                ProviderReceipt(
                    venue_id=venue_id,
                    order_id=order_id,
                    provider_name=prov,
                    received=bool(received),
                    received_at=now if received else None,
                    received_by=actor if received else None,
                    note=_safe_str(note) or None,
                    created_at=now,
                    updated_at=now,
                )
            )
        s.commit()


# =============================================================================
# UI helpers
# =============================================================================

def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .voi-card{border:1px solid rgba(49,51,63,.12); border-radius:16px; padding:14px 14px; margin:10px 0; background:rgba(255,255,255,.03);}
        .voi-row{display:flex; gap:10px; align-items:center;}
        .voi-muted{opacity:.7; font-size:.9rem;}
        .voi-chip{display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid rgba(49,51,63,.18); font-size:.85rem; opacity:.9;}
        .voi-divider{height:1px; background:rgba(49,51,63,.12); margin:12px 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _group_lines_by_provider(lines: List[OrderLine], products_by_id: Dict[int, Product]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns {provider_name: [ {name, qty, unit, line_id}, ... ]}
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for ln in lines:
        pid = getattr(ln, "product_id", None)
        prod = products_by_id.get(int(pid)) if pid is not None else None
        prov = _norm_provider(getattr(prod, "provider_name", None) if prod else getattr(ln, "provider", None))
        name = _safe_str(getattr(prod, "name", None) if prod else getattr(ln, "matched_name", None) or getattr(ln, "spoken_name", "Producto"))
        unit = _safe_str(getattr(prod, "unit", None) if prod else getattr(ln, "unit", None)) or "unidad"
        qty = float(getattr(ln, "quantity", 0.0) or 0.0)
        out.setdefault(prov, []).append(
            {"line_id": int(getattr(ln, "id", 0) or 0), "name": name, "qty": qty, "unit": unit}
        )
    # stable order
    for prov in out:
        out[prov] = sorted(out[prov], key=lambda x: (x["name"] or "").lower())
    return dict(sorted(out.items(), key=lambda kv: kv[0].lower()))


def _editor_df_from_lines(lines: List[OrderLine]) -> pd.DataFrame:
    rows = []
    for ln in lines:
        rows.append(
            {
                "line_id": getattr(ln, "id", None),
                "product_id": getattr(ln, "product_id", None),
                "quantity": float(getattr(ln, "quantity", 0.0) or 0.0),
                "unit": _safe_str(getattr(ln, "unit", "")) or "unidad",
                "delete": False,
            }
        )
    if not rows:
        rows = [{"line_id": pd.NA, "product_id": pd.NA, "quantity": 1.0, "unit": "unidad", "delete": False}]
    return pd.DataFrame(rows)


# =============================================================================
# Sections
# =============================================================================

def _render_header(order: Order) -> None:
    c1, c2 = st.columns([2.2, 1], vertical_alignment="center")
    with c1:
        st.markdown(f"### {_order_label(order)}")
        st.caption(
            f"{_status_chip(order.status)} · "
            f"Creado: {getattr(order, 'created_at', None).strftime('%Y-%m-%d %H:%M') if getattr(order,'created_at',None) else '—'} · "
            f"Por: {_safe_str(getattr(order,'created_by', '—')) or '—'}"
        )
    with c2:
        st.markdown(f"<div class='voi-chip'>{_status_label(order.status)}</div>", unsafe_allow_html=True)


def _render_lines_editor(
    *,
    venue_id: int,
    order: Order,
    actor: str,
    products: List[Product],
    lines: List[OrderLine],
) -> None:
    products_by_id = {int(p.id): p for p in products if p.id is not None}

    # --- UI helpers (keep local: avoids leaking into other pages) ---
    import re
    import unicodedata

    def _strip_accents(s: str) -> str:
        s = unicodedata.normalize("NFD", s or "")
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
        return unicodedata.normalize("NFC", s)

    def _norm_words(s: str) -> List[str]:
        # Unicode-safe "words" (Greek included) + numbers
        raw = re.findall(r"[0-9]+|[^\W_]+", (s or ""), flags=re.UNICODE)
        return [_strip_accents(w).lower() for w in raw if w.strip()]

    def _remove_name_words_from_description(name: str, desc: str) -> str:
        name_set = set(_norm_words(name))
        if not name_set:
            return (desc or "").strip()

        desc_raw = re.findall(r"[0-9]+|[^\W_]+", (desc or ""), flags=re.UNICODE)
        kept: List[str] = []
        for w in desc_raw:
            if _strip_accents(w).lower() not in name_set:
                kept.append(w)
        return " ".join(kept).strip()

    def _pid_to_int(pid: Any) -> Optional[int]:
        """Robustly coerce data_editor product_id values (can be list/NA/str/float)."""
        if pid is None:
            return None
        # data_editor sometimes returns list-like values; take the first
        if isinstance(pid, (list, tuple)):
            if not pid:
                return None
            pid = pid[0]
        try:
            if pd.isna(pid):
                return None
        except Exception:
            pass
        if isinstance(pid, int):
            return pid
        if isinstance(pid, float):
            if pd.isna(pid):
                return None
            return int(pid)
        s = str(pid).strip()
        if not s:
            return None
        try:
            return int(float(s))
        except Exception:
            return None

    # --- Make Producto labels: name — (description without name-words) — proveedor ---
    label_by_id: Dict[int, str] = {}
    for p in products:
        if p.id is None:
            continue
        name = _safe_str(getattr(p, "name", ""))
        desc = _safe_str(getattr(p, "description", ""))
        prov = _safe_str(getattr(p, "provider_name", "")) or "(Sin proveedor)"
        desc_clean = _remove_name_words_from_description(name, desc)

        parts = [name]
        if desc_clean:
            parts.append(desc_clean)
        parts.append(prov)

        label_by_id[int(p.id)] = " — ".join([x for x in parts if _safe_str(x)])

    st.markdown("### 🧾 Líneas del pedido")

    # Make long labels readable (wrap)
    st.markdown(
        """
        <style>
        [data-testid="stDataEditor"] td {
            white-space: normal !important;
            line-height: 1.25;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    df = _editor_df_from_lines(lines)

    edited = st.data_editor(
        df,
        hide_index=True,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "line_id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
            "product_id": st.column_config.SelectboxColumn(
                "Producto",
                options=sorted(label_by_id.keys()),
                format_func=lambda pid: label_by_id.get(_pid_to_int(pid) or -1, str(pid)),
                required=True,
                width="large",
            ),
            "quantity": st.column_config.NumberColumn("Cantidad", min_value=0.0, step=0.5, width="small"),
            "unit": st.column_config.TextColumn("Unidad", disabled=True, width="small"),
            "delete": st.column_config.CheckboxColumn("🗑️", help="Marca para eliminar", width="small"),
        },
        key=f"order_editor_{int(order.id)}",
        disabled=(order.status != "draft"),
    )

    # ✅ Keep unit synced to the selected product (robust to pid being list/NA)
    def _unit_for_pid(pid: Any) -> str:
        pid_i = _pid_to_int(pid)
        if pid_i is None:
            return "unidad"
        p = products_by_id.get(pid_i)
        return (_safe_str(getattr(p, "unit", "")) or "unidad").lower() if p else "unidad"

    if "product_id" in edited.columns:
        edited["unit"] = edited["product_id"].map(_unit_for_pid)

    if order.status != "draft":
        st.info("Este pedido ya no es un borrador. Para editar líneas, vuelve a Borradores.")
        return

    c1, c2, c3 = st.columns([1.3, 1.3, 2.4], vertical_alignment="center")
    with c1:
        if st.button("💾 Guardar cambios", type="primary", use_container_width=True, key=f"save_{int(order.id)}"):
            _save_lines_from_editor(
                venue_id=venue_id,
                order_id=int(order.id),
                actor=actor,
                df=edited,
                products_by_id=products_by_id,
            )
            _bump_refresh(venue_id)
            st.success("Guardado ✓")
            st.rerun()
    with c2:
        if st.button("↩️ Descartar", use_container_width=True, key=f"discard_{int(order.id)}"):
            st.rerun()
    with c3:
        st.caption("Tip: añade filas, cambia productos y cantidades, y guarda.")


def _render_workflow_actions(
    *,
    venue_id: int,
    order: Order,
    role: Optional[str],
    actor: str,
) -> None:
    st.markdown('<div class="voi-divider"></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.6, 1.6, 1.2], vertical_alignment="center")

    status = (order.status or "draft").strip().lower()
    can_manage = (role or "").lower() in {"owner", "manager"}

    with c1:
        if status == "draft":
            if st.button("✅ Pasar a Listo", type="primary", use_container_width=True, key=f"to_ready_{int(order.id)}"):
                _set_order_status(int(order.id), "ready_to_send", actor)
                _bump_refresh(venue_id)
                st.rerun()
        elif status == "ready_to_send":
            if st.button("↩️ Volver a Borrador", use_container_width=True, key=f"to_draft_{int(order.id)}"):
                _set_order_status(int(order.id), "draft", actor)
                _bump_refresh(venue_id)
                st.rerun()
        else:
            st.caption("")

    with c2:
        if status == "pending_receive":
            if st.button("✅ Cerrar (a Historial)", type="primary", use_container_width=True, key=f"to_final_{int(order.id)}"):
                _set_order_status(int(order.id), "final", actor)
                _bump_refresh(venue_id)
                st.rerun()
        else:
            st.caption("")

    with c3:
        if status == "draft" and can_manage:
            if st.button("🗑️ Eliminar", use_container_width=True, key=f"del_{int(order.id)}"):
                _delete_order(int(order.id))
                _bump_refresh(venue_id)
                st.session_state.pop(f"orders_active_order_id_{venue_id}", None)
                st.success("Pedido eliminado.")
                st.rerun()


def _render_send_section(
    *,
    venue_id: int,
    order: Order,
    products: List[Product],
    lines: List[OrderLine],
) -> None:
    st.markdown("### 📤 Enviar (WhatsApp / Email)")

    if (order.status or "").strip().lower() != "ready_to_send":
        st.info("Esta sección solo está disponible en estado **Listo**.")
        return

    v = _load_venue_templates(venue_id, _refresh_token(venue_id))

    missing_required = _venue_missing_required(v)
    if missing_required:
        st.error(
            "Faltan campos obligatorios del local para enviar:\n"
            + "\n".join([f"- {x}" for x in missing_required])
            + "\n\nVe a **Administración → Gestionar organización** y complétalos."
        )
        return

    if not (v.subject_tpl and v.opening_tpl and v.closing_tpl):
        st.warning(
            "Plantillas no configuradas. Ve a **Administración → Configure emails** y completa:\n"
            "- Asunto\n- Cabecera\n- Cierre"
        )
        return

    products_by_id = {int(p.id): p for p in products if p.id is not None}
    grouped = _group_lines_by_provider(lines, products_by_id)
    if not grouped:
        st.info("No hay líneas para enviar.")
        return

    # =============================================================================
    # Factura-style summary (per provider + totals, optional IVA)
    # =============================================================================

    # Wrap long product labels in the editor / tables
    st.markdown(
        """
        <style>
        [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {white-space: normal !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    csum1, csum2, csum3 = st.columns([1.4, 1.2, 1.4], vertical_alignment="center")
    with csum1:
        show_prices = st.toggle(
            "Mostrar precios",
            value=True,
            key=f"sum_show_prices_{int(order.id)}",
            help="Si no hay precios en el catálogo, los importes saldrán a 0.",
        )
    with csum2:
        include_iva = st.toggle(
            "Incluir IVA",
            value=False,
            key=f"sum_include_iva_{int(order.id)}",
        )
    with csum3:
        iva_pct = st.number_input(
            "IVA %",
            min_value=0.0,
            max_value=100.0,
            value=21.0,
            step=0.5,
            disabled=not include_iva,
            key=f"sum_iva_pct_{int(order.id)}",
        )

    def _price_for_pid(pid: Optional[int]) -> float:
        if pid is None:
            return 0.0
        p = products_by_id.get(int(pid))
        try:
            return float(getattr(p, "price", 0.0) or 0.0) if p else 0.0
        except Exception:
            return 0.0

    def _provider_summary_df(provider_name: str, prov_lines: List[Dict[str, Any]]) -> "pd.DataFrame":
        rows: List[Dict[str, Any]] = []
        for ln in prov_lines:
            line_id = ln.get("line_id")
            # Find product_id by looking up the actual OrderLine when possible
            pid: Optional[int] = None
            if line_id:
                try:
                    # lines is in scope; cheap linear scan (small list)
                    for _ol in lines:
                        if int(getattr(_ol, "id", 0) or 0) == int(line_id):
                            pid = int(getattr(_ol, "product_id", 0) or 0) or None
                            break
                except Exception:
                    pid = None

            qty = float(ln.get("qty") or 0.0)
            unit = _safe_str(ln.get("unit") or "unidad")
            name = _safe_str(ln.get("name") or "")
            unit_price = _price_for_pid(pid) if show_prices else 0.0
            amount = qty * unit_price
            rows.append(
                {
                    "Producto": name,
                    "Cantidad": qty,
                    "Unidad": unit,
                    "Precio": unit_price,
                    "Importe": amount,
                }
            )
        dfp = pd.DataFrame(rows)
        # Nice formatting
        if not dfp.empty:
            dfp["Cantidad"] = pd.to_numeric(dfp["Cantidad"], errors="coerce").fillna(0.0)
            dfp["Precio"] = pd.to_numeric(dfp["Precio"], errors="coerce").fillna(0.0)
            dfp["Importe"] = pd.to_numeric(dfp["Importe"], errors="coerce").fillna(0.0)
        return dfp

    any_price = any((_price_for_pid(getattr(ol, "product_id", None)) > 0.0) for ol in lines)
    if show_prices and not any_price:
        st.info("ℹ️ No se han encontrado precios en el catálogo para este pedido. Los importes saldrán a 0.")

    with st.expander("🧾 Resumen (factura)", expanded=True):
        grand_subtotal = 0.0
        provider_totals: List[Dict[str, Any]] = []

        for prov, prov_lines in grouped.items():
            dfp = _provider_summary_df(prov, prov_lines)
            subtotal = float(dfp["Importe"].sum()) if (show_prices and not dfp.empty) else 0.0
            grand_subtotal += subtotal
            iva_amount = subtotal * (float(iva_pct) / 100.0) if include_iva else 0.0
            total = subtotal + iva_amount

            st.markdown(f"**Proveedor:** {prov}")
            if not dfp.empty:
                # Show without index
                st.dataframe(
                    dfp,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Producto": st.column_config.TextColumn(width="large"),
                        "Cantidad": st.column_config.NumberColumn(format="%.2f", width="small"),
                        "Unidad": st.column_config.TextColumn(width="small"),
                        "Precio": st.column_config.NumberColumn(format="%.2f", width="small"),
                        "Importe": st.column_config.NumberColumn(format="%.2f", width="small"),
                    },
                )
            else:
                st.caption("Sin líneas")

            # Provider totals line
            if show_prices:
                if include_iva:
                    st.markdown(
                        f"**Subtotal:** {subtotal:,.2f}  ·  **IVA ({iva_pct:g}%):** {iva_amount:,.2f}  ·  **Total:** {total:,.2f}"
                    )
                else:
                    st.markdown(f"**Subtotal:** {subtotal:,.2f}")
            st.markdown("---")

            provider_totals.append(
                {
                    "Proveedor": prov,
                    "Subtotal": subtotal,
                    "IVA": iva_amount,
                    "Total": total,
                }
            )

        # Grand totals
        if show_prices:
            grand_iva = grand_subtotal * (float(iva_pct) / 100.0) if include_iva else 0.0
            grand_total = grand_subtotal + grand_iva

            st.markdown("### Totales")
            if include_iva:
                st.markdown(
                    f"**Subtotal:** {grand_subtotal:,.2f}  ·  **IVA ({iva_pct:g}%):** {grand_iva:,.2f}  ·  **Total:** {grand_total:,.2f}"
                )
            else:
                st.markdown(f"**Total:** {grand_subtotal:,.2f}")
        else:
            st.caption("Activa **Mostrar precios** para ver importes.")

    provider_dir = _providers_cached(get_session, venue_id)

    with st.expander("📤 Generar envíos", expanded=True):
        top1, top2, top3 = st.columns([1, 1, 2], vertical_alignment="center")
        with top1:
            send_wa = st.toggle("WhatsApp", value=True, key=f"send_wa_{int(order.id)}")
        with top2:
            send_email = st.toggle("Email", value=True, key=f"send_em_{int(order.id)}")
        with top3:
            move_to_pending = st.toggle("Marcar como pendiente después", value=True, key=f"send_move_{int(order.id)}")

        cc = st.text_input("Prefijo país WhatsApp", value="+34", key=f"send_cc_{int(order.id)}")

        # Preview box: final invoice-style header (as requested)
        with st.popover("👁️ Vista previa (Cabecera final)", use_container_width=True):
            st.code(_invoice_header_text(v) or "—", language=None)

        date_str = datetime.now().strftime("%Y-%m-%d")
        subject_base = _render_tpl(v.subject_tpl, order_id=int(order.id), date_str=date_str, venue_name=v.venue_name)
        subject = subject_base
        if v.venue_name and v.venue_name.lower() not in subject_base.lower():
            subject = f"{v.venue_name} — {subject_base}".strip(" —")

        opening = _render_tpl(v.opening_tpl, order_id=int(order.id), date_str=date_str, venue_name=v.venue_name)
        closing = _render_tpl(v.closing_tpl, order_id=int(order.id), date_str=date_str, venue_name=v.venue_name)
        header_txt = _invoice_header_text(v)

        generate_key = f"send_generated_{int(order.id)}"
        if st.button("🚀 Generar", type="primary", use_container_width=True, key=f"btn_gen_{int(order.id)}"):
            st.session_state[generate_key] = True
            if move_to_pending:
                _set_order_status(int(order.id), "pending_receive", actor=_safe_str(st.session_state.get("user_email") or st.session_state.get("actor") or ""))
                _bump_refresh(venue_id)
            st.rerun()

        if not st.session_state.get(generate_key, False):
            st.caption("Genera para ver botones de envío por proveedor.")
            return

        st.success("Envíos generados ✓")

        for prov, prov_lines in grouped.items():
            p = provider_dir.get(_norm_provider(prov))
            provider_email = ""
            provider_phone = ""
            if p:
                provider_email = _split_first_pipe(getattr(p, "order_email", None) or getattr(p, "emails", None))
                provider_phone = _split_first_pipe(getattr(p, "order_phone", None) or getattr(p, "phones", None))

            # Build body
            body_lines: List[str] = []
            if header_txt:
                body_lines.extend([header_txt, "", "--------------------", ""])
            if opening:
                body_lines.append(opening.strip())
                body_lines.append("")

            for ln in prov_lines:
                qty = float(ln.get("qty") or 0.0)
                if qty > 0:
                    body_lines.append(f"- {ln.get('name','')} — {qty:g} {ln.get('unit') or 'unidad'}")

            if closing:
                body_lines.extend(["", closing.strip()])

            body_text = "\n".join([x for x in body_lines if x is not None]).strip()

            st.markdown(
                f"<div class='voi-card'>"
                f"<div class='voi-row' style='justify-content:space-between;'>"
                f"<div><div style='font-weight:700;'>{prov}</div>"
                f"<div class='voi-muted'>{len(prov_lines)} productos</div></div>"
                f"<div class='voi-chip'>Enviar</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            a1, a2, a3 = st.columns([1, 1, 1.2], vertical_alignment="center")
            with a1:
                if send_email and provider_email:
                    import urllib.parse as up
                    mailto = f"mailto:{up.quote(provider_email)}?subject={up.quote(subject)}&body={up.quote(body_text)}"
                    st.link_button("📧 Email", mailto, use_container_width=True)
                else:
                    st.caption("Sin email")
            with a2:
                if send_wa and provider_phone:
                    import urllib.parse as up
                    phone_norm = _normalize_phone(provider_phone, cc)
                    if phone_norm:
                        wa = f"https://wa.me/{phone_norm.replace('+','')}?text={up.quote(body_text)}"
                        st.link_button("📲 WhatsApp", wa, use_container_width=True)
                    else:
                        st.caption("Tel inválido")
                else:
                    st.caption("Sin teléfono")
            with a3:
                st.download_button(
                    "📄 TXT",
                    data=body_text.encode("utf-8"),
                    file_name=f"pedido_{_safe_str(prov).replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key=f"dl_{int(order.id)}_{prov}",
                )

            with st.expander("👁️ Vista previa mensaje", expanded=False):
                st.code(body_text, language=None)

        st.markdown('<div class="voi-divider"></div>', unsafe_allow_html=True)
        if st.button("🔄 Regenerar envíos", use_container_width=True, key=f"regen_{int(order.id)}"):
            st.session_state.pop(generate_key, None)
            st.rerun()


def _render_receive_section(
    *,
    venue_id: int,
    order: Order,
    products: List[Product],
    lines: List[OrderLine],
    actor: str,
) -> None:
    st.markdown("### 📦 Recepción por proveedor")

    if (order.status or "").strip().lower() != "pending_receive":
        st.info("Esta sección solo está disponible en estado **Pendiente**.")
        return

    products_by_id = {int(p.id): p for p in products if p.id is not None}
    grouped = _group_lines_by_provider(lines, products_by_id)
    receipts = _provider_receipts_cached(get_session, int(order.id), _refresh_token(venue_id))

    if not grouped:
        st.info("No hay líneas en este pedido.")
        return

    providers = list(grouped.keys())
    received_count = 0
    for prov in providers:
        r = receipts.get(_norm_provider(prov))
        if r and bool(getattr(r, "received", False)):
            received_count += 1

    st.progress(received_count / max(1, len(providers)))
    st.caption(f"{received_count}/{len(providers)} proveedores marcados como recibidos")

    for prov in providers:
        prov_key = _norm_provider(prov)
        r = receipts.get(prov_key)
        is_received = bool(r and getattr(r, "received", False))
        badge = "✅ Recibido" if is_received else "⏳ Pendiente"

        st.markdown(
            f"<div class='voi-card'>"
            f"<div class='voi-row' style='justify-content:space-between;'>"
            f"<div><div style='font-weight:700;'>{prov_key}</div>"
            f"<div class='voi-muted'>{len(grouped[prov])} líneas</div></div>"
            f"<div class='voi-chip'>{badge}</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        with st.expander("Abrir recepción", expanded=not is_received):
            note = st.text_area("Nota (opcional)", value=_safe_str(getattr(r, "note", "")) if r else "", key=f"recv_note_{int(order.id)}_{prov_key}")
            cols = st.columns([1, 1], vertical_alignment="center")
            with cols[0]:
                if st.button("✅ Marcar recibido", type="primary", use_container_width=True, key=f"recv_yes_{int(order.id)}_{prov_key}"):
                    _upsert_provider_receipt(
                        venue_id=venue_id,
                        order_id=int(order.id),
                        provider_name=prov_key,
                        received=True,
                        actor=actor,
                        note=note,
                    )
                    _bump_refresh(venue_id)
                    st.rerun()
            with cols[1]:
                if st.button("↩️ Marcar pendiente", use_container_width=True, key=f"recv_no_{int(order.id)}_{prov_key}"):
                    _upsert_provider_receipt(
                        venue_id=venue_id,
                        order_id=int(order.id),
                        provider_name=prov_key,
                        received=False,
                        actor=actor,
                        note=note,
                    )
                    _bump_refresh(venue_id)
                    st.rerun()


# =============================================================================
# Public entrypoint used by app.py
# =============================================================================

def orders_tab(venue_id: int, venue_role: Optional[str]) -> None:
    """
    Main Orders tab entrypoint.
    app.py calls: orders_tab(venue_id, venue_role)
    """
    _inject_css()

    # Determine actor identity (best-effort)
    actor = _safe_str(
        st.session_state.get("user_email")
        or st.session_state.get("actor")
        or st.session_state.get("email")
        or "user"
    )

    st.markdown("## 🧾 Pedidos")

    # Session state key for currently selected order (used across reruns)
    active_key = f"orders_active_order_id_{venue_id}"

    # Top toolbar (create columns first)
    t1, t2 = st.columns([4.9, 1.4], vertical_alignment="center")
    with t1:
        status_filter = _segmented_status_filter(key=f"orders_status_{venue_id}")
    with t2:
        # Allow creating a draft even when the current filter has no orders.
        if status_filter == "draft":
            if st.button(
                "➕ Nuevo borrador",
                type="primary",
                use_container_width=True,
                key=f"new_draft_top_{venue_id}",
            ):
                oid = _create_empty_draft(venue_id, actor)
                _bump_refresh(venue_id)
                st.session_state[active_key] = int(oid)
                st.rerun()

    # Load orders list
    orders_all = _list_orders_cached(get_session, venue_id, _refresh_token(venue_id))

    def matches(o: Order) -> bool:
        stt = (getattr(o, "status", "draft") or "draft").strip().lower()
        if status_filter != "all" and stt != status_filter:
            return False
        return True

    orders = [o for o in orders_all if getattr(o, "id", None) is not None and matches(o)]

    if not orders:
        st.info("No hay pedidos para este filtro.")
        return

    # Order picker
    ids = [int(o.id) for o in orders if o.id is not None]
    labels = {int(o.id): _order_label(o) for o in orders if o.id is not None}
    default_oid = int(st.session_state.get(active_key) or ids[0])

    if default_oid not in ids:
        default_oid = ids[0]

    picked = st.selectbox(
        "Pedido",
        options=ids,
        index=ids.index(default_oid),
        format_func=lambda oid: labels.get(int(oid), str(oid)),
        key=f"orders_picker_{venue_id}",
    )
    st.session_state[active_key] = int(picked)

    # Reload selected order from DB (source of truth)
    with get_session() as s:
        order = s.exec(
            select(Order).where(Order.id == int(picked), Order.venue_id == venue_id)
        ).first()

    if not order:
        st.error("Pedido no encontrado.")
        return

    # Load shared data
    products = _products_cached(get_session, venue_id)
    lines = _order_lines_cached(get_session, int(order.id), _refresh_token(venue_id))

    _render_header(order)
    _render_workflow_actions(venue_id=venue_id, order=order, role=venue_role, actor=actor)

    st.markdown('<div class="voi-divider"></div>', unsafe_allow_html=True)

    # Sections (simple)
    if (order.status or "").strip().lower() == "draft":
        _render_lines_editor(venue_id=venue_id, order=order, actor=actor, products=products, lines=lines)

    if (order.status or "").strip().lower() == "ready_to_send":
        st.markdown('<div class="voi-divider"></div>', unsafe_allow_html=True)
        _render_send_section(venue_id=venue_id, order=order, products=products, lines=lines)

    if (order.status or "").strip().lower() == "pending_receive":
        st.markdown('<div class="voi-divider"></div>', unsafe_allow_html=True)
        _render_receive_section(venue_id=venue_id, order=order, products=products, lines=lines, actor=actor)

