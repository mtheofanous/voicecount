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
import hashlib
import pandas as pd
import streamlit as st
from sqlmodel import select
import re
import unicodedata
from core.db import get_session
from domain.models import *
from core.public_links import build_seguimiento_url, ROLE_SUPPLIER, ROLE_VENUE, norm_provider
from core.mailer import send_smtp_email
# Auth DB (venue templates + invoice header fields live there)
try:
    from features.auth_and_manage.auth_multi_tenant import get_auth_session, Venue, User  # type: ignore
except Exception:  # pragma: no cover
    get_auth_session = None  # type: ignore
    Venue = None  # type: ignore
    User = None


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

def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def _vat_pct_for_product(prod, fallback_vat_pct: float) -> float:
    """
    Returns VAT % to use for this product:
    - If prod.iva is present and > 0, use it
    - else fallback_vat_pct (from UI toggle)
    """
    if prod is None:
        return float(fallback_vat_pct)

    v = getattr(prod, "iva", None)
    v = _safe_float(v, default=float(fallback_vat_pct))
    # treat 0 or negative as "missing"
    if v <= 0:
        return float(fallback_vat_pct)
    return float(v)

def _df_fingerprint(df: pd.DataFrame) -> str:
    """
    Stable fingerprint for detecting meaningful df changes
    (row order independent, ignores index).
    """
    if df is None or df.empty:
        return ""
    cols = ["line_id", "product_id", "quantity", "unit", "delete"]
    cols = [c for c in cols if c in df.columns]
    df_norm = df[cols].copy().sort_values(cols).reset_index(drop=True)
    return hashlib.md5(pd.util.hash_pandas_object(df_norm, index=False).values).hexdigest()

def _get_send_status_map(*, order_id: int) -> dict[str, ProviderSendStatus]:
    with get_session() as s:
        rows = s.exec(
            select(ProviderSendStatus).where(ProviderSendStatus.order_id == int(order_id))
        ).all()
    return {norm_provider(r.provider_name): r for r in rows}


def _touch_send_status(
    *,
    venue_id: int,
    order_id: int,
    provider_name: str,
    actor: str,
    channel: str,  # "email" | "whatsapp" | "txt"
    ok: bool,
    error: str | None = None,
) -> None:
    prov = norm_provider(provider_name)
    now = datetime.utcnow()

    with get_session() as s:
        obj = s.exec(
            select(ProviderSendStatus).where(
                ProviderSendStatus.order_id == int(order_id),
                ProviderSendStatus.provider_name == prov,
            )
        ).first()

        if not obj:
            obj = ProviderSendStatus(
                venue_id=int(venue_id),
                order_id=int(order_id),
                provider_name=prov,
            )

        obj.send_attempts = int(getattr(obj, "send_attempts", 0) or 0) + 1
        obj.updated_at = now

        if ok:
            obj.sent = True
            obj.sent_at = now
            obj.sent_by = (actor or "").strip() or None
            obj.last_error = None

            if channel == "email":
                obj.sent_email = True
            elif channel == "whatsapp":
                obj.sent_whatsapp = True
            elif channel == "txt":
                obj.sent_txt = True
        else:
            # keep last error for debugging / support
            obj.last_error = (error or "").strip()[:500] or "unknown error"

        s.add(obj)
        s.commit()


def _all_providers_sent(*, order_id: int, provider_names: list[str]) -> bool:
    provs = {norm_provider(p) for p in provider_names}
    m = _get_send_status_map(order_id=order_id)
    sent_provs = {p for p, row in m.items() if bool(getattr(row, "sent", False))}
    return provs.issubset(sent_provs)


def _reset_send_status(*, order_id: int, provider_name: str | None = None) -> None:
    with get_session() as s:
        q = select(ProviderSendStatus).where(ProviderSendStatus.order_id == int(order_id))
        if provider_name:
            q = q.where(ProviderSendStatus.provider_name == norm_provider(provider_name))

        rows = s.exec(q).all()
        for r in rows:
            r.sent = False
            r.sent_email = False
            r.sent_whatsapp = False
            r.sent_txt = False
            r.sent_at = None
            r.sent_by = None
            r.send_attempts = 0
            r.last_error = None
            r.updated_at = datetime.utcnow()
            s.add(r)
        s.commit()

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
    email_lang: str = "es"
    email_cc: str = ""
    email_bcc: str = ""
    opening_tpl_es: str = ""
    opening_tpl_en: str = ""
    opening_tpl_gr: str = ""


@st.cache_data(ttl=120, show_spinner=False)
def _load_venue_templates(venue_id: int, refresh_token: int) -> VenueTemplates:
    """
    Read venue metadata + email settings from auth DB.

    Updated for:
    - standardized subject/closing (so we don't rely on subject_tpl/closing_tpl)
    - auto-language openings (ES/EN/GR)
    - per-supplier CC/BCC rules
    - legal footer uses company/address/VAT from venue fields

    Still robust if auth module is missing.
    """
    _ = refresh_token
    if get_auth_session is None or Venue is None or User is None:
        return VenueTemplates()

    v = None
    owner_name = ""
    try:
        with get_auth_session() as s:
            v = s.exec(select(Venue).where(Venue.id == venue_id)).first()
            if v:
                owner_name = s.exec(
                    select(User.full_name)
                    .where(
                        User.account_id == v.account_id,
                        User.account_role == "owner",
                        User.is_active == True,
                    )
                    .order_by(User.created_at.asc())
                ).first() or ""
    except Exception:
        pass

    if not v:
        return VenueTemplates()


    # New fields (safe getattr so code doesn't crash if DB isn't migrated yet)
    email_lang = _safe_str(getattr(v, "email_lang", "")).strip().lower() or "es"
    if email_lang not in {"es", "en", "gr"}:
        email_lang = "es"

    email_cc = _safe_str(getattr(v, "email_cc", "")).strip()
    email_bcc = _safe_str(getattr(v, "email_bcc", "")).strip()

    # Prefer per-language openings; fallback to legacy email_opening_tpl; then empty.
    open_es = _safe_str(getattr(v, "email_opening_tpl_es", "")).strip()
    open_en = _safe_str(getattr(v, "email_opening_tpl_en", "")).strip()
    open_gr = _safe_str(getattr(v, "email_opening_tpl_gr", "")).strip()
    legacy_open = _safe_str(getattr(v, "email_opening_tpl", "")).strip()

    # Choose a "current opening_tpl" based on email_lang, but keep all three for UI/preview/send logic.
    opening_by_lang = {
        "es": open_es or (legacy_open if email_lang == "es" else ""),
        "en": open_en or (legacy_open if email_lang == "en" else ""),
        "gr": open_gr or (legacy_open if email_lang == "gr" else ""),
    }
    chosen_opening = opening_by_lang.get(email_lang, "") or legacy_open

    return VenueTemplates(
        venue_name=_safe_str(getattr(v, "name", "")),
        owner_name=_safe_str(owner_name), 
        address=_safe_str(getattr(v, "address", "")),
        tax_number=_safe_str(getattr(v, "tax_number", "")),
        email=_safe_str(getattr(v, "email", "")),
        phone=_safe_str(getattr(v, "phone", "")),

        # Legacy fields kept for compatibility with existing code paths:
        # subject_tpl/closing_tpl are intentionally blank because you standardize them now.
        subject_tpl="",
        opening_tpl=chosen_opening,
        closing_tpl="",

        # New fields for send logic / preview:
        email_lang=email_lang,
        email_cc=email_cc,
        email_bcc=email_bcc,
        opening_tpl_es=opening_by_lang.get("es", "") or open_es,
        opening_tpl_en=opening_by_lang.get("en", "") or open_en,
        opening_tpl_gr=opening_by_lang.get("gr", "") or open_gr,
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
    invoice_number: str = "",
    supplier_declaration: str = "",
    supplier_comment: str = "",
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
            # invoice number (optional)
            inv = _safe_str(invoice_number)
            if inv:
                row.invoice_number = inv
                row.invoice_number_set_at = now
                row.invoice_number_set_by = actor

            # supplier declaration (optional)
            decl = (_safe_str(supplier_declaration) or '').strip().lower()
            if decl in {'full','partial','none'}:
                row.supplier_declaration = decl
                row.supplier_declared_at = now
                row.supplier_declared_by = actor
                cmt = _safe_str(supplier_comment)
                row.supplier_declared_comment = cmt or None

            row.updated_at = now
            s.add(row)
        else:
            s.add(
                ProviderReceipt(
                    venue_id=venue_id,
                    order_id=order_id,
                    provider_name=prov,

                    # invoice number (optional)
                    invoice_number=(_safe_str(invoice_number) or None),
                    invoice_number_set_at=(now if _safe_str(invoice_number) else None),
                    invoice_number_set_by=(actor if _safe_str(invoice_number) else None),

                    # supplier declaration (optional)
                    supplier_declaration=((_safe_str(supplier_declaration) or '').strip().lower() if (_safe_str(supplier_declaration) or '').strip().lower() in {'full','partial','none'} else None),
                    supplier_declared_at=(now if (_safe_str(supplier_declaration) or '').strip().lower() in {'full','partial','none'} else None),
                    supplier_declared_by=(actor if (_safe_str(supplier_declaration) or '').strip().lower() in {'full','partial','none'} else None),
                    supplier_declared_comment=(_safe_str(supplier_comment) or None),

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

import urllib.parse as up

def _split_emails(raw: str) -> list[str]:
    raw = _safe_str(raw)
    if not raw:
        return []
    # allow "a@x.com | b@y.com" or comma-separated
    raw = raw.replace(",", "|")
    return [p.strip() for p in raw.split("|") if p.strip()]

def _email_preview_html(*, venue_name: str, provider_name: str, order_id: int, link_url: str, intro: str) -> str:
    intro_html = (
        f"<p style='margin:0 0 14px 0;font-size:15px;line-height:1.4'>{intro}</p>"
        if intro.strip()
        else "<p style='margin:0 0 14px 0;font-size:15px;line-height:1.4'>Hola,</p>"
    )

    return f"""
    <div style="font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
                background:#f6f7fb;padding:22px">
      <div style="max-width:620px;margin:0 auto;background:#ffffff;border-radius:16px;
                  padding:22px;border:1px solid rgba(0,0,0,.06)">
        <div style="display:flex;justify-content:space-between;gap:10px;align-items:center">
          <div style="font-weight:800;font-size:16px">{venue_name or "Pedido"}</div>
          <div style="font-size:12px;opacity:.7">Pedido #{order_id}</div>
        </div>

        <div style="margin-top:12px">
          {intro_html}
          <p style="margin:0 0 10px 0;font-size:14px;opacity:.85">
            Por favor, abre el link para añadir el <b>Nº de factura</b> y marcar incidencias (OK / Partial / Missing).
            Una vez guardado, el link queda bloqueado.
          </p>
        </div>

        <div style="margin-top:16px;margin-bottom:6px">
          <a href="{link_url}" style="display:inline-block;background:#111827;color:#fff;
                                    padding:12px 16px;border-radius:12px;text-decoration:none;
                                    font-weight:700;font-size:14px">
            Abrir link de recepción
          </a>
        </div>

        <div style="margin-top:14px;font-size:12px;opacity:.75">
          Proveedor: <b>{provider_name}</b>
        </div>

        <hr style="margin:18px 0;border:none;border-top:1px solid rgba(0,0,0,.08)" />

        <div style="font-size:12px;opacity:.70">
          Si no puedes abrir el botón, copia y pega este enlace:<br/>
          <span style="word-break:break-all">{link_url}</span>
        </div>
      </div>
    </div>
    """

def _email_preview_text(*, provider_name: str, order_id: int, link_url: str, intro: str) -> str:
    intro_txt = intro.strip() or "Hola,"
    return (
        f"{intro_txt}\n\n"
        f"Pedido #{order_id} · Proveedor: {provider_name}\n\n"
        "Abre el link para añadir el Nº de factura y marcar incidencias (OK / Partial / Missing).\n"
        "Una vez guardado, el link queda bloqueado.\n\n"
        f"{link_url}\n"
    )

def _render_resend_dialog(
    *,
    venue_id: int,
    order: Order,
    provider_name: str,
    provider_row: Provider | None,
    templates: VenueTemplates,
    send_status: ProviderSendStatus | None,
    actor: str,
) -> None:
    """
    Premium modal:
      - editable To/CC/BCC/Subject/Intro
      - preview TEXT + HTML
      - send SMTP + update ProviderSendStatus via _touch_send_status
    """
    order_id = int(order.id)
    link_url = build_seguimiento_url(order_id=order_id, provider_name=provider_name, role=ROLE_SUPPLIER, page_path="seguimiento")

    # defaults from Provider row + templates
    to_default = _split_first_pipe(getattr(provider_row, "email", "") if provider_row else "")
    cc_default = _safe_str(getattr(templates, "email_cc", ""))
    bcc_default = _safe_str(getattr(templates, "email_bcc", ""))

    subj_default = f"Link de recepción · Pedido #{order_id}"
    intro_default = "Hola, te reenvío el link de recepción."

    # Streamlit modal (if supported)
    try:
        dialog = st.dialog  # type: ignore[attr-defined]
        use_dialog = True
    except Exception:
        use_dialog = False

    def _content():
        st.markdown("### 🔁 Reenviar link (premium)")
        st.caption("Edita destinatarios, revisa la vista previa y envía. Queda registrado en el historial de envío.")

        c1, c2 = st.columns([1.2, 1.0])
        with c1:
            st.text_input("Proveedor", value=provider_name, disabled=True)
        with c2:
            st.link_button("🔗 Abrir link", link_url, use_container_width=True)

        to_val = st.text_input("Para (To)", value=to_default, placeholder="proveedor@email.com")
        cc_val = st.text_input("CC", value=cc_default, placeholder="cc@tuempresa.com | otra@tuempresa.com")
        bcc_val = st.text_input("BCC", value=bcc_default, placeholder="bcc@tuempresa.com")

        subject = st.text_input("Asunto", value=subj_default)
        intro = st.text_area("Texto de introducción", value=intro_default, height=90)

        # Quick actions
        cA, cB = st.columns(2)
        with cA:
            mailto = (
                f"mailto:{up.quote(to_val)}?"
                + up.urlencode([("subject", subject), ("body", _email_preview_text(provider_name=provider_name, order_id=order_id, link_url=link_url, intro=intro))])
                if to_val.strip()
                else ""
            )
            st.link_button("📨 Abrir en cliente (mailto)", mailto or "#", disabled=(not bool(to_val.strip())), use_container_width=True)
        with cB:
            st.code(link_url, language=None)

        st.divider()
        tabs = st.tabs(["👀 Vista previa (HTML)", "📝 Vista previa (texto)"])
        with tabs[0]:
            html = _email_preview_html(
                venue_name=_safe_str(getattr(templates, "venue_name", "")),
                provider_name=provider_name,
                order_id=order_id,
                link_url=link_url,
                intro=intro,
            )
            st.components.v1.html(html, height=420, scrolling=True)  # type: ignore

        with tabs[1]:
            st.text(_email_preview_text(provider_name=provider_name, order_id=order_id, link_url=link_url, intro=intro))

        st.divider()

        # Send
        can_send = bool(to_val.strip())
        if st.button("✅ Enviar ahora (SMTP)", type="primary", use_container_width=True, disabled=(not can_send)):
            try:
                send_smtp_email(
                    to=_split_emails(to_val),
                    cc=_split_emails(cc_val),
                    bcc=_split_emails(bcc_val),
                    subject=subject,
                    text_body=_email_preview_text(provider_name=provider_name, order_id=order_id, link_url=link_url, intro=intro),
                    html_body=_email_preview_html(
                        venue_name=_safe_str(getattr(templates, "venue_name", "")),
                        provider_name=provider_name,
                        order_id=order_id,
                        link_url=link_url,
                        intro=intro,
                    ),
                )
                _touch_send_status(
                    venue_id=int(venue_id),
                    order_id=order_id,
                    provider_name=provider_name,
                    actor=actor,
                    channel="email",
                    ok=True,
                    error=None,
                )
                st.success("Email enviado ✅")
                st.rerun()
            except Exception as e:
                _touch_send_status(
                    venue_id=int(venue_id),
                    order_id=order_id,
                    provider_name=provider_name,
                    actor=actor,
                    channel="email",
                    ok=False,
                    error=str(e),
                )
                st.error(f"No se pudo enviar: {e}")

    if use_dialog:
        # The dialog is created at call-time; inside we render content
        @dialog("Reenviar link")  # type: ignore[misc]
        def _dlg():
            _content()
        _dlg()
    else:
        # Fallback: expander (older Streamlit)
        with st.expander("🔁 Reenviar link (premium)", expanded=True):
            _content()


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .voi-card{border:1px solid rgba(49,51,63,.12); border-radius:16px; padding:14px 14px; margin:10px 0; background:rgba(255,255,255,.03);}
        .voi-row{display:flex; gap:10px; align-items:center;}
        .voi-muted{opacity:.7; font-size:.9rem;}
        .voi-chip{display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid rgba(49,51,63,.18); font-size:.85rem; opacity:.9;}
        .voi-chip-ok{background:rgba(46,160,67,.12); border-color:rgba(46,160,67,.35);}
        .voi-chip-warn{background:rgba(255,159,10,.12); border-color:rgba(255,159,10,.35);}
        .voi-chip-bad{background:rgba(255,69,58,.12); border-color:rgba(255,69,58,.35);}
        .voi-chip-muted{background:rgba(120,120,120,.10); border-color:rgba(120,120,120,.22);}
        .voi-chip-wrap{display:flex; flex-wrap:wrap; gap:8px; align-items:center;}
        .voi-divider{height:1px; background:rgba(49,51,63,.12); margin:12px 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )



def _provider_receipts_map(*, order_id: int) -> dict[str, ProviderReceipt]:
    """Return {provider_name: ProviderReceipt} for this order."""
    with get_session() as s:
        rows = list(
            s.exec(select(ProviderReceipt).where(ProviderReceipt.order_id == int(order_id))).all()
        )
    return { _norm_provider(getattr(r, "provider_name", "")): r for r in rows if r }

def _decl_chip_class(decl: str) -> str:
    d = (decl or "").strip().lower()
    if d == "full":
        return "voi-chip-ok"
    if d == "partial":
        return "voi-chip-warn"
    if d == "none":
        return "voi-chip-bad"
    return "voi-chip-muted"

def _decl_icon(decl: str) -> str:
    d = (decl or "").strip().lower()
    return {"full":"✅", "partial":"🟠", "none":"🔴"}.get(d, "—")

def _render_provider_chips(*, order: Order, lines: list[OrderLine], products: list[Product]) -> None:
    """Compact, mobile-friendly status chips per supplier."""
    products_by_id = {int(p.id): p for p in products if p.id is not None}
    grouped = _group_lines_by_provider(lines, products_by_id)
    if not grouped:
        return

    receipts = _provider_receipts_map(order_id=int(order.id))

    bits: list[str] = []
    for prov in grouped.keys():
        r = receipts.get(_norm_provider(prov))
        decl = _safe_str(getattr(r, "supplier_declaration", "")) if r else ""
        inv = _safe_str(getattr(r, "invoice_number", "")) if r else ""
        received = bool(getattr(r, "received", False)) if r else False

        parts = [prov, _decl_icon(decl)]
        parts.append(f"🧾 {inv}" if inv else "🧾 —")
        if received:
            parts.append("📥 Recibido")

        cls = _decl_chip_class(decl)
        label = " · ".join(parts)
        bits.append(f"<div class='voi-chip {cls}'>{label}</div>")

    st.markdown("<div class='voi-chip-wrap'>" + "".join(bits) + "</div>", unsafe_allow_html=True)


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

    def _strip_accents(s: str) -> str:
        s = unicodedata.normalize("NFD", s or "")
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
        return unicodedata.normalize("NFC", s)

    def _norm_words(s: str) -> List[str]:
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

    def _sanitize_editor_df(df0: pd.DataFrame) -> pd.DataFrame:
        """Force editor df into types data_editor + DB save can handle."""
        df1 = df0.copy()

        # Ensure required columns exist
        for col, default in [
            ("line_id", pd.NA),
            ("product_id", pd.NA),
            ("quantity", 0.0),
            ("unit", "unidad"),
            ("delete", False),
        ]:
            if col not in df1.columns:
                df1[col] = default

        # product_id must be scalar int/NA (NOT list) for SelectboxColumn
        df1["product_id"] = df1["product_id"].apply(_pid_to_int)
        df1["product_id"] = df1["product_id"].where(df1["product_id"].notna(), pd.NA)

        # quantity numeric
        df1["quantity"] = pd.to_numeric(df1["quantity"], errors="coerce").fillna(0.0)

        # unit string
        df1["unit"] = df1["unit"].astype(str).replace({"nan": "unidad"}).fillna("unidad")

        # delete bool
        df1["delete"] = df1["delete"].fillna(False).astype(bool)

        return df1

    def _products_in_order(df: pd.DataFrame) -> set[int]:
        df = _sanitize_editor_df(df)
        out: set[int] = set()
        for pid, deleted in zip(df["product_id"], df["delete"]):
            pid_i = _pid_to_int(pid)
            if pid_i is not None and not bool(deleted):
                out.add(pid_i)
        return out

    def _qty_by_product(df: pd.DataFrame) -> Dict[int, float]:
        df = _sanitize_editor_df(df)
        out: Dict[int, float] = {}
        for pid, qty, deleted in zip(df["product_id"], df["quantity"], df["delete"]):
            pid_i = _pid_to_int(pid)
            if pid_i is None or bool(deleted):
                continue
            out[pid_i] = out.get(pid_i, 0.0) + float(qty or 0.0)
        return out

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

    # -----------------------------
    # Session DF per order (single source of truth for editor + quick-add)
    # -----------------------------
    editor_key = f"order_editor_{int(order.id)}"
    df_state_key = f"{editor_key}__df"

    # If not draft, we should not keep stale draft editor state
    if order.status != "draft":
        st.session_state.pop(df_state_key, None)

    # Initialize once per order from DB lines
    if df_state_key not in st.session_state:
        st.session_state[df_state_key] = _sanitize_editor_df(_editor_df_from_lines(lines))
    else:
        # keep it clean on every run (prevents list-typed product_id crash loops)
        st.session_state[df_state_key] = _sanitize_editor_df(st.session_state[df_state_key])

    def _add_product_to_df(pid: int, qty: float) -> None:
        """Append a new line (or increment existing) in the session DF, then rerun."""
        try:
            qty_f = float(qty)
        except Exception:
            return
        if qty_f <= 0:
            return

        df0 = _sanitize_editor_df(st.session_state[df_state_key])

        mask_same = (df0["product_id"] == pid) & (df0["delete"] != True)  # noqa: E712
        if mask_same.any():
            idx = df0.index[mask_same][0]
            df0.at[idx, "quantity"] = float(df0.at[idx, "quantity"] or 0.0) + qty_f
        else:
            df0 = pd.concat(
                [
                    df0,
                    pd.DataFrame(
                        [{
                            "line_id": pd.NA,
                            "product_id": pid,
                            "quantity": qty_f,
                            "unit": (_safe_str(getattr(products_by_id.get(pid), "unit", "")) or "unidad").lower(),
                            "delete": False,
                        }]
                    ),
                ],
                ignore_index=True,
            )

        st.session_state[df_state_key] = _sanitize_editor_df(df0)
        st.rerun()

    # -----------------------------
    # Quick add expander
    # -----------------------------
    # -----------------------------
    # Quick add expander (cascading filters + paging + toggle)
    # -----------------------------

    # Build options from products
    all_categories = sorted({
        (_safe_str(getattr(p, "category", "")) or "").strip()
        for p in products
        if _safe_str(getattr(p, "category", "")).strip()
    })
    all_providers = sorted({
        (_safe_str(getattr(p, "provider_name", "")) or "(Sin proveedor)").strip()
        for p in products
    })

    # Maps
    cat_by_pid = {
        int(p.id): (_safe_str(getattr(p, "category", "")) or "").strip()
        for p in products
        if p.id is not None
    }
    prov_by_pid = {
        int(p.id): (_safe_str(getattr(p, "provider_name", "")) or "(Sin proveedor)").strip()
        for p in products
        if p.id is not None
    }

    with st.expander("➕ Añadir productos", expanded=True):
        reset_flag = f"{editor_key}__qa_reset_qty"
        if st.session_state.get(reset_flag):
            qty_prefix = f"{editor_key}__qa_qty_"

            # remove all qty widget states BEFORE instantiating the widgets
            for k in list(st.session_state.keys()):
                if isinstance(k, str) and k.startswith(qty_prefix):
                    st.session_state.pop(k, None)

            st.session_state.pop(reset_flag, None)


        # ---------- ONE-LINE FILTER BAR ----------
        f1, f2, f3, f4 = st.columns(
            [2.6, 1.6, 1.6, 1.2],
            vertical_alignment="center",
        )

        with f1:
            q = st.text_input(
                "Buscar",
                key=f"{editor_key}__qa_search",
                placeholder="Producto…",
            ).strip().lower()

        # Prepare order state for dependent options
        df_current = _sanitize_editor_df(st.session_state[df_state_key])
        qty_by_pid = _qty_by_product(df_current)

        base_pids = sorted(label_by_id.keys())
        if q:
            base_pids = [pid for pid in base_pids if q in label_by_id.get(pid, "").lower()]

        # Session keys
        cat_key = f"{editor_key}__qa_cat"
        prov_key = f"{editor_key}__qa_prov"
        hide_key = f"{editor_key}__qa_hide"

        st.session_state.setdefault(cat_key, "Todas")
        st.session_state.setdefault(prov_key, "Todos")
        st.session_state.setdefault(hide_key, False)

        current_cat = st.session_state[cat_key]
        current_prov = st.session_state[prov_key]

        # --- Compute cascading options ---
        cats_for_prov = (
            sorted({cat_by_pid.get(pid, "") for pid in base_pids if prov_by_pid.get(pid, "") == current_prov})
            if current_prov != "Todos"
            else sorted({cat_by_pid.get(pid, "") for pid in base_pids})
        )
        cats_for_prov = [c for c in cats_for_prov if c]

        provs_for_cat = (
            sorted({prov_by_pid.get(pid, "") for pid in base_pids if cat_by_pid.get(pid, "") == current_cat})
            if current_cat != "Todas"
            else sorted({prov_by_pid.get(pid, "") for pid in base_pids})
        )
        provs_for_cat = [p for p in provs_for_cat if p]

        # Reset invalid selections
        if current_cat != "Todas" and current_cat not in cats_for_prov:
            st.session_state[cat_key] = "Todas"
            current_cat = "Todas"

        if current_prov != "Todos" and current_prov not in provs_for_cat:
            st.session_state[prov_key] = "Todos"
            current_prov = "Todos"

        with f2:
            selected_cat = st.selectbox(
                "Categoría",
                options=["Todas"] + cats_for_prov,
                index=(["Todas"] + cats_for_prov).index(current_cat),
                key=cat_key,
            )

        with f3:
            # Re-evaluate providers after category selection
            prov_opts = (
                sorted({prov_by_pid.get(pid, "") for pid in base_pids if cat_by_pid.get(pid, "") == selected_cat})
                if selected_cat != "Todas"
                else sorted({prov_by_pid.get(pid, "") for pid in base_pids})
            )
            prov_opts = [p for p in prov_opts if p]

            selected_prov = st.selectbox(
                "Proveedor",
                options=["Todos"] + prov_opts,
                index=(["Todos"] + prov_opts).index(st.session_state[prov_key]),
                key=prov_key,
            )

        with f4:
            hide_in_order = st.toggle(
                "Ocultar en pedido",
                key=hide_key,
            )

        # ---------- Reset paging when filters change ----------
        filters_sig = (q, selected_cat, selected_prov, bool(hide_in_order))
        sig_key = f"{editor_key}__qa_filters_sig"
        page_key = f"{editor_key}__qa_page"

        if st.session_state.get(sig_key) != filters_sig:
            st.session_state[sig_key] = filters_sig
            st.session_state[page_key] = 1

        # ---------- Apply filters ----------
        pids = base_pids

        if selected_cat != "Todas":
            pids = [pid for pid in pids if cat_by_pid.get(pid, "") == selected_cat]

        if selected_prov != "Todos":
            pids = [pid for pid in pids if prov_by_pid.get(pid, "") == selected_prov]

        if hide_in_order:
            pids = [pid for pid in pids if float(qty_by_pid.get(pid, 0.0) or 0.0) <= 0.0]

        total = len(pids)

        # ---------- Paging ----------
        p1, p2, p3, p4 = st.columns([1.2, 1.2, 1.6, 2.0], vertical_alignment="center")

        page_size = p3.selectbox(
            "Por página",
            options=[30, 60, 90, 120],
            index=1,
            key=f"{editor_key}__qa_page_size",
        )

        st.session_state.setdefault(page_key, 1)
        total_pages = max(1, (total + page_size - 1) // page_size)
        st.session_state[page_key] = min(st.session_state[page_key], total_pages)

        with p1:
            if st.button("⬅️", disabled=st.session_state[page_key] <= 1):
                st.session_state[page_key] -= 1
                st.rerun()

        with p2:
            if st.button("➡️", disabled=st.session_state[page_key] >= total_pages):
                st.session_state[page_key] += 1
                st.rerun()

        with p4:
            st.caption(f"{total} resultados · Página {st.session_state[page_key]} / {total_pages}")

        # ---------- Grid ----------
        start = (st.session_state[page_key] - 1) * page_size
        end = start + page_size
        pids_page = pids[start:end]

        with st.container(height=400):
            cols = st.columns(3, gap="small")
            for i, pid in enumerate(pids_page):
                col = cols[i % 3]
                p = products_by_id.get(pid)
                unit_txt = (_safe_str(getattr(p, "unit", "")) or "unidad").lower()

                label = label_by_id.get(pid, str(pid))
                parts = label.split(" — ", 1)
                name = parts[0]
                rest = parts[1] if len(parts) > 1 else ""

                with col:
                    existing_qty = float(qty_by_pid.get(pid, 0.0) or 0.0)
                    in_order = existing_qty > 0
                    qty_txt = f"{existing_qty:g}"

                    bg = "rgba(33,150,243,0.08)" if in_order else "transparent"
                    border = "rgba(33,150,243,0.5)" if in_order else "rgba(49,51,63,.2)"

                    unit_row = (
                        f"""
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-top:6px">
                            <span style="font-size:0.95rem;font-weight:700">{unit_txt}</span>
                            <span style="font-size:0.9rem;font-weight:800;
                                        background:rgba(33,150,243,.95);
                                        color:white;padding:4px 10px;
                                        border-radius:999px">
                                ✓ En pedido · {qty_txt}
                            </span>
                        </div>
                        """
                        if in_order
                        else f"""
                        <div style="margin-top:6px">
                            <span style="font-size:0.95rem;font-weight:700">{unit_txt}</span>
                        </div>
                        """
                    )

                    st.markdown(
                        f"""
                        <div style="
                            padding:.6rem;
                            border:1px solid {border};
                            border-left:4px solid {'#2196F3' if in_order else border};
                            border-radius:.6rem;
                            background:{bg};
                            line-height:1.25
                        ">
                            <div style="font-weight:700">{name}</div>
                            {'<div style="opacity:0.55;font-size:0.9em">' + rest + '</div>' if rest else ''}
                            {unit_row}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    form_key = f"{editor_key}__qa_form_{pid}"
                    with st.form(key=form_key, clear_on_submit=False):
                        qty_val = st.number_input(
                            "Qty",
                            min_value=0,
                            step=1,
                            value=0,
                            key=f"{editor_key}__qa_qty_{pid}",
                            label_visibility="collapsed",
                        )
                        submitted = st.form_submit_button(
                            "Sumar" if in_order else "Añadir",
                            use_container_width=True,
                        )

                    if submitted:
                        _add_product_to_df(pid, qty_val)




    # -----------------------------
    # Lines editor
    # -----------------------------
    st.markdown("### 🧾 Líneas del pedido")

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

    # IMPORTANT: always sanitize before feeding to data_editor
    df_for_editor = _sanitize_editor_df(st.session_state[df_state_key])

    # ✅ If Streamlit previously stored list-typed product_id in the widget key state,
    # clear it so the editor doesn't crash.
    if "product_id" in df_for_editor.columns:
        had_list = df_for_editor["product_id"].apply(lambda x: isinstance(x, (list, tuple))).any()
        if had_list:
            st.session_state.pop(editor_key, None)

    edited = st.data_editor(
        df_for_editor,
        hide_index=True,
        num_rows="dynamic",
        width='stretch',
        column_config={
            "line_id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
            "product_id": st.column_config.SelectboxColumn(
                "Producto",
                options=sorted(label_by_id.keys()),
                format_func=lambda pid: label_by_id.get(_pid_to_int(pid) or -1, str(pid)),
                required=True,
                width="large",
            ),
            "quantity": st.column_config.NumberColumn("Qty", min_value=0.0, step=0.5, width="small"),
            "unit": st.column_config.TextColumn("Unidad", disabled=True, width="small"),
            "delete": st.column_config.CheckboxColumn("🗑️", help="Marca para eliminar", width="small"),
        },
        key=editor_key,
        disabled=(order.status != "draft"),
    )

    # Persist edits back to session df (clean types)
    edited = _sanitize_editor_df(edited)
    st.session_state[df_state_key] = edited.copy()

    # ✅ Keep unit synced to selected product
    def _unit_for_pid(pid: Any) -> str:
        pid_i = _pid_to_int(pid)
        if pid_i is None:
            return "unidad"
        p = products_by_id.get(pid_i)
        return (_safe_str(getattr(p, "unit", "")) or "unidad").lower() if p else "unidad"

    if "product_id" in edited.columns:
        edited["unit"] = edited["product_id"].map(_unit_for_pid)
        st.session_state[df_state_key] = _sanitize_editor_df(edited)

    # If not draft -> no editing
    if order.status != "draft":
        st.info("Este pedido ya no es un borrador. Para editar líneas, vuelve a Borradores.")
        return

    # -----------------------------
    # Actions
    # -----------------------------
    c1, c2, c3 = st.columns([1.3, 1.3, 2.4], vertical_alignment="center")

    with c1:
        if st.button("💾 Guardar cambios", type="primary", width="stretch", key=f"save_{int(order.id)}"):
            df_to_save = _sanitize_editor_df(st.session_state[df_state_key])

            _save_lines_from_editor(
                venue_id=venue_id,
                order_id=int(order.id),
                actor=actor,
                df=df_to_save,
                products_by_id=products_by_id,
            )

            # ✅ tell next run to reset quick-add qty widgets
            st.session_state[f"{editor_key}__qa_reset_qty"] = True

            # Clear editor widget + df state so next run reloads clean from DB
            st.session_state.pop(df_state_key, None)
            st.session_state.pop(editor_key, None)

            _bump_refresh(venue_id)
            st.success("Guardado ✓")
            st.rerun()



    with c2:
        if st.button("↩️ Descartar", width='stretch', key=f"discard_{int(order.id)}"):
            st.session_state.pop(df_state_key, None)
            st.session_state.pop(editor_key, None)
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
            if st.button("✅ Pasar a Listo", type="primary", width='stretch', key=f"to_ready_{int(order.id)}"):
                _set_order_status(int(order.id), "ready_to_send", actor)
                _bump_refresh(venue_id)
                st.rerun()
        elif status == "ready_to_send":
            if st.button("↩️ Volver a Borrador", width='stretch', key=f"to_draft_{int(order.id)}"):
                _set_order_status(int(order.id), "draft", actor)
                _bump_refresh(venue_id)
                st.rerun()
        else:
            st.caption("")

    with c2:
        if status == "pending_receive":
            if st.button("✅ Cerrar (a Historial)", type="primary", width='stretch', key=f"to_final_{int(order.id)}"):
                _set_order_status(int(order.id), "final", actor)
                _bump_refresh(venue_id)
                st.rerun()
        else:
            st.caption("")

    with c3:
        if status == "draft" and can_manage:
            if st.button("🗑️ Eliminar", width='stretch', key=f"del_{int(order.id)}"):
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
    actor: str,
) -> None:
    st.markdown("### Summary")

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

    products_by_id: Dict[int, Product] = {int(p.id): p for p in products if p.id is not None}

    # =============================================================================
    # Helpers: robust parsing
    # =============================================================================
    def _pid_to_int(x) -> Optional[int]:
        """Streamlit can sometimes return list/tuple; make this safe."""
        try:
            if x is None:
                return None
            try:
                if pd.isna(x):
                    return None
            except Exception:
                pass
            if isinstance(x, (list, tuple)):
                if not x:
                    return None
                x = x[0]
            return int(x)
        except Exception:
            return None

    def _safe_float(x, default: float = 0.0) -> float:
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    # =============================================================================
    # Discount rules (ProviderDiscountRule): line + prev-month
    # =============================================================================
    from datetime import timedelta
    from sqlalchemy import func  # make sure sqlalchemy is installed
    # Provider map: normalized provider name -> Provider row
    providers_by_name = _providers_cached(get_session, venue_id)

    @st.cache_data(ttl=60, show_spinner=False, hash_funcs={type(lambda: None): lambda _: "session_fn"})
    def _provider_rules_cached(_get_session_fn, venue_id: int, provider_id: int) -> List[ProviderDiscountRule]:
        with _get_session_fn() as s:
            return list(
                s.exec(
                    select(ProviderDiscountRule)
                    .where(
                        ProviderDiscountRule.venue_id == venue_id,
                        ProviderDiscountRule.provider_id == provider_id,
                        ProviderDiscountRule.is_active == True,  # noqa: E712
                    )
                    .order_by(
                        # product rules first, then global
                        ProviderDiscountRule.product_id.is_(None).asc(),
                        # prev_month rules first (we'll sort again in Python too)
                        ProviderDiscountRule.rule_kind.asc(),
                        ProviderDiscountRule.min_qty.desc(),
                    )
                ).all()
            )

    _rules_by_provider_norm: Dict[str, List[ProviderDiscountRule]] = {}

    def _rules_for_provider(provider_name: str) -> List[ProviderDiscountRule]:
        pnorm = _norm_provider(provider_name)
        if pnorm in _rules_by_provider_norm:
            return _rules_by_provider_norm[pnorm]

        prow = providers_by_name.get(pnorm)
        if not prow or prow.id is None:
            _rules_by_provider_norm[pnorm] = []
            return []

        rules = _provider_rules_cached(get_session, venue_id, int(prow.id))
        _rules_by_provider_norm[pnorm] = rules
        return rules

    def _prev_month_window(now: datetime) -> tuple[datetime, datetime]:
        """Previous calendar month [start, end)."""
        first_this_month = datetime(now.year, now.month, 1)
        last_prev_month = first_this_month - timedelta(days=1)
        start_prev_month = datetime(last_prev_month.year, last_prev_month.month, 1)
        end_prev_month = first_this_month
        return start_prev_month, end_prev_month

    @st.cache_data(ttl=60, show_spinner=False, hash_funcs={type(lambda: None): lambda _: "session_fn"})
    def _prev_month_qty_cached(
        _get_session_fn,
        *,
        venue_id: int,
        provider_norm: str,
        product_id: Optional[int],
    ) -> float:
        """
        Qty bought in previous calendar month, considering ONLY finalized orders.
        - If product_id is not None -> sum qty for that product
        - Else -> sum qty for provider (all its products) using OrderLine.provider == provider_norm
        """
        start_dt, end_dt = _prev_month_window(datetime.utcnow())

        with _get_session_fn() as s:
            q = (
                select(func.coalesce(func.sum(OrderLine.quantity), 0.0))
                .join(Order, Order.id == OrderLine.order_id)
                .where(
                    Order.venue_id == venue_id,
                    Order.status == "final",
                    Order.created_at >= start_dt,
                    Order.created_at < end_dt,
                )
            )

            if product_id is not None:
                q = q.where(OrderLine.product_id == int(product_id))
            else:
                q = q.where(OrderLine.provider == provider_norm)

            val = s.exec(q).one()
            try:
                return float(val or 0.0)
            except Exception:
                return 0.0

    def _match_scope_ok(r: ProviderDiscountRule, pid: Optional[int]) -> bool:
        """True if rule applies to this product (product-specific or global)."""
        if r.product_id is None:
            return True
        if pid is None:
            return False
        return int(r.product_id) == int(pid)

    def _rule_priority_key(r: ProviderDiscountRule) -> tuple:
        """
        Sorting priority:
          1) product-specific first
          2) prev_month rules before line rules
          3) higher threshold wins (prev_month_min_qty or min_qty)
          4) stronger effect tie-breaker
        """
        is_product = 1 if r.product_id is not None else 0
        rk = (r.rule_kind or "line_pct").strip()
        is_prev = 1 if rk.startswith("prev_month") else 0

        threshold = 0.0
        if is_prev:
            threshold = float(getattr(r, "prev_month_min_qty", 0.0) or 0.0)
        else:
            threshold = float(r.min_qty or 0.0)

        # effect strength (rough)
        strength = 0.0
        if rk.endswith("_net_price"):
            strength = float(getattr(r, "price_override", 0.0) or 0.0)  # lower is "stronger" but only used for tiebreak
        else:
            strength = float(r.discount_percent or 0.0)

        return (is_product, is_prev, threshold, strength)

    def _pricing_for_line(provider_name: str, pid: Optional[int], qty: float, gross_unit: float) -> Dict[str, Any]:
        """
        Decide net unit price for this line based on:
          - prev_month_* rules (eligibility based on previous calendar month)
          - line_* rules (eligibility based on this line qty)
        Returns:
          { net_unit, discount_pct, rule_kind, rule_note, rule_id, applied }
        """
        if qty <= 0 or gross_unit <= 0:
            return {
                "net_unit": gross_unit,
                "discount_pct": 0.0,
                "rule_kind": "",
                "rule_note": "",
                "rule_id": None,
                "applied": False,
            }

        prov_norm = _norm_provider(provider_name)
        rules = _rules_for_provider(prov_norm)
        if not rules:
            return {
                "net_unit": gross_unit,
                "discount_pct": 0.0,
                "rule_kind": "",
                "rule_note": "",
                "rule_id": None,
                "applied": False,
            }

        # Build candidates that are eligible
        candidates: List[ProviderDiscountRule] = []
        for r in rules:
            rk = (r.rule_kind or "line_pct").strip()

            if not _match_scope_ok(r, pid):
                continue

            if rk.startswith("prev_month"):
                th = float(getattr(r, "prev_month_min_qty", 0.0) or 0.0)
                if th <= 0:
                    continue

                # product-specific prev-month uses product qty; global uses provider qty
                qty_last_month = _prev_month_qty_cached(
                    get_session,
                    venue_id=venue_id,
                    provider_norm=prov_norm,
                    product_id=(int(pid) if (pid is not None and r.product_id is not None) else None),
                )
                if qty_last_month >= th:
                    candidates.append(r)

            else:
                # line-based
                th = float(r.min_qty or 0.0)
                if qty >= th:
                    candidates.append(r)

        if not candidates:
            return {
                "net_unit": gross_unit,
                "discount_pct": 0.0,
                "rule_kind": "",
                "rule_note": "",
                "rule_id": None,
                "applied": False,
            }

        # choose best
        candidates.sort(key=_rule_priority_key, reverse=True)
        chosen = candidates[0]
        rk = (chosen.rule_kind or "line_pct").strip()

        if rk.endswith("_net_price"):
            net_unit = float(getattr(chosen, "price_override", None) or 0.0)
            if net_unit <= 0:
                # safety fallback
                return {
                    "net_unit": gross_unit,
                    "discount_pct": 0.0,
                    "rule_kind": "",
                    "rule_note": "",
                    "rule_id": None,
                    "applied": False,
                }
            disc_pct = (1.0 - (net_unit / gross_unit)) * 100.0 if gross_unit > 0 else 0.0
            disc_pct = max(0.0, min(100.0, disc_pct))
            return {
                "net_unit": net_unit,
                "discount_pct": float(disc_pct),
                "rule_kind": rk,
                "rule_note": _safe_str(getattr(chosen, "note", "")),
                "rule_id": int(chosen.id) if chosen.id is not None else None,
                "applied": True,
            }

        # pct rule
        disc = float(chosen.discount_percent or 0.0)
        disc = max(0.0, min(100.0, disc))
        net_unit = gross_unit * (1.0 - disc / 100.0)
        return {
            "net_unit": float(net_unit),
            "discount_pct": float(disc),
            "rule_kind": rk,
            "rule_note": _safe_str(getattr(chosen, "note", "")),
            "rule_id": int(chosen.id) if chosen.id is not None else None,
            "applied": disc > 0.0,
        }

    # =============================================================================
    # ✅ Producto label: "name — desc_clean" (reuse everywhere: summary + emails)
    # =============================================================================
    def _strip_accents(s: str) -> str:
        s = unicodedata.normalize("NFD", s or "")
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
        return unicodedata.normalize("NFC", s)

    def _norm_words(s: str) -> List[str]:
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

    summary_label_by_id: Dict[int, str] = {}
    for p in products:
        if p.id is None:
            continue
        name = _safe_str(getattr(p, "name", ""))
        desc = _safe_str(getattr(p, "description", ""))
        desc_clean = _remove_name_words_from_description(name, desc)
        parts = [name] + ([desc_clean] if desc_clean else [])
        summary_label_by_id[int(p.id)] = " — ".join([x for x in parts if _safe_str(x)])

    def _label_for_pid(pid: Optional[int], fallback_name: str = "Producto") -> str:
        if pid is None:
            return _safe_str(fallback_name) or "Producto"
        return summary_label_by_id.get(int(pid), _safe_str(fallback_name) or "Producto")

    # =============================================================================
    # Grouping
    # =============================================================================
    grouped = _group_lines_by_provider(lines, products_by_id)
    if not grouped:
        st.info("No hay líneas para enviar.")
        return

    # =============================================================================
    # Factura-style summary (per provider + totals, optional IVA per product)
    # =============================================================================
    st.markdown(
        """
        <style>
        [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {white-space: normal !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    csum1, csum2, csum4 = st.columns([1.4, 1.4, 1.6], vertical_alignment="center")
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
    with csum4:
        summary_mode = st.radio(
            "Resumen",
            options=["por_proveedor", "total"],
            format_func=lambda x: "Por proveedor" if x == "por_proveedor" else "Total",
            horizontal=True,
            label_visibility="collapsed",
            key=f"sum_mode_{int(order.id)}",
        )

    def _price_for_pid(pid: Optional[int]) -> float:
        if pid is None:
            return 0.0
        p = products_by_id.get(pid)
        return _safe_float(getattr(p, "price", 0.0) if p else 0.0, 0.0)

    def _vat_pct_for_pid(pid: Optional[int], fallback_vat_pct: float) -> float:
        if pid is None:
            return float(fallback_vat_pct)
        p = products_by_id.get(pid)
        if not p:
            return float(fallback_vat_pct)
        v_pct = _safe_float(getattr(p, "iva", None), float(fallback_vat_pct))
        return float(fallback_vat_pct) if v_pct <= 0 else float(v_pct)

    # fast lookup: line_id -> product_id
    line_pid_map: Dict[int, Optional[int]] = {}
    for _ol in (lines or []):
        lid = _pid_to_int(getattr(_ol, "id", None))
        pid = _pid_to_int(getattr(_ol, "product_id", None))
        if lid is not None:
            line_pid_map[lid] = pid

    def _rule_label(rule_kind: str) -> str:
        rk = (rule_kind or "").strip()
        m = {
            "line_pct": "Line % (min qty)",
            "line_net_price": "Line NET (min qty)",
            "prev_month_pct": "Prev month % (threshold)",
            "prev_month_net_price": "Prev month NET (threshold)",
        }
        return m.get(rk, rk or "")

    def _provider_summary_df(provider_name: str, prov_lines: List[Dict[str, Any]]) -> "pd.DataFrame":
        rows: List[Dict[str, Any]] = []

        for ln in prov_lines:
            line_id = _pid_to_int(ln.get("line_id"))
            pid = line_pid_map.get(line_id) if line_id is not None else _pid_to_int(ln.get("product_id"))

            qty = _safe_float(ln.get("qty"), 0.0)
            if qty <= 0:
                continue

            unit = _safe_str(ln.get("unit") or "unidad")
            name = _label_for_pid(pid, ln.get("name") or "Producto")

            gross_unit = _price_for_pid(pid) if show_prices else 0.0

            pricing = _pricing_for_line(provider_name, pid, qty, gross_unit) if show_prices else {
                "net_unit": 0.0, "discount_pct": 0.0, "rule_kind": "", "rule_note": "", "rule_id": None, "applied": False
            }

            net_unit = float(pricing.get("net_unit", gross_unit) or 0.0)
            disc_pct = float(pricing.get("discount_pct", 0.0) or 0.0)
            rule_kind = _safe_str(pricing.get("rule_kind", ""))
            rule_txt = _rule_label(rule_kind) if pricing.get("applied") else ""

            amount = qty * net_unit if show_prices else 0.0
            ahorro = qty * max(0.0, (gross_unit - net_unit)) if show_prices else 0.0

            row: Dict[str, Any] = {
                "Producto": name,
                "Qty": qty,
                "Unidad": unit,
            }

            if show_prices:
                row["Regla"] = rule_txt
                row["Precio base"] = gross_unit
                row["Desc.%"] = disc_pct if disc_pct > 0 else 0.0
                row["Precio"] = net_unit
                row["Importe"] = amount
                row["Ahorro"] = ahorro

                if include_iva:
                    vat_pct_line = _vat_pct_for_pid(pid, 21.00)
                    iva_eur = amount * (vat_pct_line / 100.0)
                    row["% IVA"] = vat_pct_line
                    row["IVA (€)"] = iva_eur
                    row["Total"] = amount + iva_eur

            rows.append(row)

        dfp = pd.DataFrame(rows)

        if not dfp.empty:
            dfp["Qty"] = pd.to_numeric(dfp.get("Qty"), errors="coerce").fillna(0.0)
            if show_prices:
                for c in ["Precio base", "Desc.%", "Precio", "Importe", "Ahorro"]:
                    if c in dfp.columns:
                        dfp[c] = pd.to_numeric(dfp.get(c), errors="coerce").fillna(0.0)

                if include_iva:
                    dfp["% IVA"] = pd.to_numeric(dfp.get("% IVA"), errors="coerce").fillna(21.00)
                    dfp["IVA (€)"] = pd.to_numeric(dfp.get("IVA (€)"), errors="coerce").fillna(0.0)
                    dfp["Total"] = pd.to_numeric(dfp.get("Total"), errors="coerce").fillna(0.0)

            if show_prices and include_iva:
                desired = ["Producto", "Qty", "Unidad", "Regla", "Precio base", "Desc.%", "Precio", "Importe", "Ahorro", "% IVA", "IVA (€)", "Total"]
            elif show_prices:
                desired = ["Producto", "Qty", "Unidad", "Regla", "Precio base", "Desc.%", "Precio", "Importe", "Ahorro"]
            else:
                desired = ["Producto", "Qty", "Unidad"]

            cols = [c for c in desired if c in dfp.columns] + [c for c in dfp.columns if c not in desired]
            dfp = dfp[cols]

        return dfp

    any_price = any((_price_for_pid(_pid_to_int(getattr(ol, "product_id", None))) > 0.0) for ol in (lines or []))
    if show_prices and not any_price:
        st.info("ℹ️ No se han encontrado precios en el catálogo para este pedido. Los importes saldrán a 0.")

    # =============================================================================
    # Render
    # =============================================================================
    if summary_mode == "total":
        dfs: List[pd.DataFrame] = []
        for prov, prov_lines in grouped.items():
            dfp = _provider_summary_df(prov, prov_lines)
            if not dfp.empty:
                dfp.insert(0, "Proveedor", prov)
                dfs.append(dfp)

        df_total = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        if not df_total.empty:
            if show_prices and include_iva:
                desired = ["Proveedor", "Producto", "Qty", "Unidad", "Regla", "Precio base", "Desc.%", "Precio", "Importe", "Ahorro", "% IVA", "IVA (€)", "Total"]
            elif show_prices:
                desired = ["Proveedor", "Producto", "Qty", "Unidad", "Regla", "Precio base", "Desc.%", "Precio", "Importe", "Ahorro"]
            else:
                desired = ["Proveedor", "Producto", "Qty", "Unidad"]

            cols = [c for c in desired if c in df_total.columns] + [c for c in df_total.columns if c not in desired]
            df_total = df_total[cols]
            st.dataframe(df_total, width="stretch", hide_index=True)
        else:
            st.info("No hay líneas para resumir.")

        if show_prices and not df_total.empty:
            subtotal = float(pd.to_numeric(df_total.get("Importe", 0), errors="coerce").fillna(0.0).sum())
            ahorro_total = float(pd.to_numeric(df_total.get("Ahorro", 0), errors="coerce").fillna(0.0).sum())
            iva_amount = (
                float(pd.to_numeric(df_total.get("IVA (€)", 0), errors="coerce").fillna(0.0).sum())
                if include_iva
                else 0.0
            )
            total = subtotal + iva_amount

            st.markdown("### Totales")
            if include_iva:
                st.markdown(
                    f"**Subtotal (neto):** {subtotal:,.2f}  ·  **Ahorro:** {ahorro_total:,.2f}  ·  "
                    f"**IVA:** {iva_amount:,.2f}  ·  **Total:** {total:,.2f}"
                )
            else:
                st.markdown(f"**Total (neto):** {subtotal:,.2f}  ·  **Ahorro:** {ahorro_total:,.2f}")
        elif not show_prices:
            st.caption("Activa **Mostrar precios** para ver importes.")

    else:
        grand_subtotal = 0.0
        grand_iva = 0.0
        grand_ahorro = 0.0

        for prov, prov_lines in grouped.items():
            with st.container(border=True):
                dfp = _provider_summary_df(prov, prov_lines)

                subtotal = float(dfp["Importe"].sum()) if (show_prices and "Importe" in dfp.columns and not dfp.empty) else 0.0
                ahorro = float(dfp["Ahorro"].sum()) if (show_prices and "Ahorro" in dfp.columns and not dfp.empty) else 0.0
                iva_amount = float(dfp["IVA (€)"].sum()) if (show_prices and include_iva and "IVA (€)" in dfp.columns and not dfp.empty) else 0.0
                total = subtotal + iva_amount

                grand_subtotal += subtotal
                grand_iva += iva_amount
                grand_ahorro += ahorro

                st.markdown(f"**Proveedor:** {prov}")
                if not dfp.empty:
                    st.dataframe(dfp, width="stretch", hide_index=True)
                else:
                    st.caption("Sin líneas.")

                if show_prices:
                    if include_iva:
                        st.markdown(
                            f"**Subtotal (neto):** {subtotal:,.2f}  ·  **Ahorro:** {ahorro:,.2f}  ·  "
                            f"**IVA:** {iva_amount:,.2f}  ·  **Total:** {total:,.2f}"
                        )
                    else:
                        st.markdown(f"**Subtotal (neto):** {subtotal:,.2f}  ·  **Ahorro:** {ahorro:,.2f}")

        if show_prices:
            grand_total = grand_subtotal + grand_iva
            st.markdown("### Totales")
            if include_iva:
                st.markdown(
                    f"**Subtotal (neto):** {grand_subtotal:,.2f}  ·  **Ahorro:** {grand_ahorro:,.2f}  ·  "
                    f"**IVA:** {grand_iva:,.2f}  ·  **Total:** {grand_total:,.2f}"
                )
            else:
                st.markdown(f"**Total (neto):** {grand_subtotal:,.2f}  ·  **Ahorro:** {grand_ahorro:,.2f}")
        else:
            st.caption("Activa **Mostrar precios** para ver importes.")


    # =============================================================================
    # Send section
    # =============================================================================
    provider_dir = _providers_cached(get_session, venue_id)

    with st.expander("📤 Generar envíos", expanded=True):
        top1, top2, top3 = st.columns([1, 1, 2], vertical_alignment="center")
        with top1:
            send_wa = st.toggle("WhatsApp", value=True, key=f"send_wa_{int(order.id)}")
        with top2:
            send_email = st.toggle("Email", value=True, key=f"send_em_{int(order.id)}")
        with top3:
            move_to_pending = st.toggle("Marcar como pendiente después", value=False, key=f"send_move_{int(order.id)}")

        wa_cc = st.text_input("Prefijo país WhatsApp", value="+34", key=f"send_cc_{int(order.id)}")

        import urllib.parse as up
        from datetime import datetime

        # ----------------------------
        # Templates helpers
        # ----------------------------
        def _render_tpl_safe(tpl: str, *, order_id: int, date_str: str, venue_name: str) -> str:
            try:
                return (tpl or "").format(order_id=order_id, date=date_str, venue_name=venue_name)
            except Exception:
                return (tpl or "").strip()

        def _build_subject(*, order_id: int, venue_name: str, date_str: str) -> str:
            return f"Pedido #{order_id} — {venue_name} — {date_str}"

        def _build_closing() -> str:
            owner = _safe_str(getattr(v, "owner_name", "")).strip()
            return ("Gracias.\n\n" f"Atentamente,\n{owner}").strip()

        DEFAULT_OPEN = {
            "es": (
                "Hola,\n\n"
                "Adjunto el pedido actualizado. Por favor, confirma disponibilidad y plazos de entrega.\n"
            ),
            "en": (
                "Hello,\n\n"
                "Please find the updated order attached. Kindly confirm availability and delivery lead times.\n"
            ),
            "gr": (
                "Γεια σας,\n\n"
                "Σας επισυνάπτω την ενημερωμένη παραγγελία. Παρακαλώ επιβεβαιώστε διαθεσιμότητα και χρόνο παράδοσης.\n"
            ),
        }

        def _build_legal_footer(v_any, lang: str) -> str:
            company = (_safe_str(getattr(v_any, "venue_name", None)) or _safe_str(getattr(v_any, "name", None))).strip()
            owner = _safe_str(getattr(v_any, "owner_name", None)).strip()
            address = _safe_str(getattr(v_any, "address", None)).strip()
            vat = _safe_str(getattr(v_any, "tax_number", None)).strip()
            email = _safe_str(getattr(v_any, "email", None)).strip()
            phone = _safe_str(getattr(v_any, "phone", None)).strip()

            missing = []
            if not company: missing.append("company/name")
            if not owner: missing.append("owner")
            if not address: missing.append("address")
            if not vat: missing.append("VAT/Tax ID")
            if not email: missing.append("email")
            if not phone: missing.append("phone")

            if lang == "en":
                labels = {
                    "owner": "Attn",
                    "address": "Address",
                    "vat": "VAT / Tax ID",
                    "email": "Email",
                    "phone": "Phone",
                    "missing": "Missing fields",
                    "notice": "This email (and any attachments) may contain confidential information.",
                }
            elif lang == "gr":
                labels = {
                    "owner": "Υπόψη",
                    "address": "Διεύθυνση",
                    "vat": "ΑΦΜ",
                    "email": "Email",
                    "phone": "Τηλέφωνο",
                    "missing": "Ελλιπή στοιχεία",
                    "notice": "Αυτό το email (και τυχόν συνημμένα) μπορεί να περιέχει εμπιστευτικές πληροφορίες.",
                }
            else:
                labels = {
                    "owner": "A la atención de",
                    "address": "Dirección",
                    "vat": "CIF/NIF",
                    "email": "Email",
                    "phone": "Teléfono",
                    "missing": "Campos pendientes",
                    "notice": "Este email (y cualquier adjunto) puede contener información confidencial.",
                }

            out = []
            out.append(company or "—")
            out.append(f"{labels['owner']}: {owner or '—'}")
            out.append(f"{labels['address']}: {address or '—'}")
            out.append(f"{labels['vat']}: {vat or '—'}")
            out.append(f"{labels['email']}: {email or '—'}")
            out.append(f"{labels['phone']}: {phone or '—'}")

            if missing:
                out.append(f"{labels['missing']}: " + ", ".join(missing))

            out.append(labels["notice"])
            return "\n".join(out).strip()

        def _plain_to_html(text: str) -> str:
            esc = (
                (text or "")
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            return esc.replace("\n", "<br>")

        # ----------------------------
        # Venue email settings
        # ----------------------------
        date_str = datetime.now().strftime("%Y-%m-%d")
        venue_name = _safe_str(getattr(v, "venue_name", None) or getattr(v, "name", None) or "").strip()

        lang = (v.email_lang or "es").strip().lower()

        if lang not in {"es", "en", "gr"}:
            lang = "es"

        cc_rule = _safe_str(getattr(v, "email_cc", None) or "").strip()
        bcc_rule = _safe_str(getattr(v, "email_bcc", None) or "").strip()

        open_es = _safe_str(getattr(v, "email_opening_tpl_es", None) or "").strip()
        open_en = _safe_str(getattr(v, "email_opening_tpl_en", None) or "").strip()
        open_gr = _safe_str(getattr(v, "email_opening_tpl_gr", None) or "").strip()

        opening_by_lang = {
            "es": open_es or DEFAULT_OPEN["es"],
            "en": open_en or DEFAULT_OPEN["en"],
            "gr": open_gr or DEFAULT_OPEN["gr"],
        }

        opening_raw = opening_by_lang.get(lang, DEFAULT_OPEN["es"])
        opening = _render_tpl_safe(opening_raw, order_id=int(order.id), date_str=date_str, venue_name=venue_name)

        subject = _build_subject(order_id=int(order.id), venue_name=venue_name, date_str=date_str)
        closing = _build_closing()
        footer = _build_legal_footer(v, lang)
        header_txt = _invoice_header_text(v)

        # ----------------------------
        # ✅ Body builder (provider): uses summary_label_by_id labels
        # ----------------------------
        def _build_provider_body_text(prov_lines: List[dict]) -> str:
            COL_PRODUCT = 44
            COL_QTY = 10
            COL_UNIT = 10

            def _fmt_qty(x: float) -> str:
                q = _safe_float(x, 0.0)
                return str(int(q)) if q.is_integer() else f"{q:g}"

            body_lines: List[str] = []

            if opening:
                body_lines += [opening.strip(), ""]

            header = f"{'Producto':<{COL_PRODUCT}}  {'Cantidad':>{COL_QTY}}  {'Unidad':<{COL_UNIT}}"
            sep = "-" * len(header)
            body_lines += [header, sep]

            for ln in prov_lines:
                qty = _safe_float(ln.get("qty"), 0.0)
                if qty <= 0:
                    continue

                pid = _pid_to_int(ln.get("product_id"))
                name = _label_for_pid(pid, ln.get("name") or "Producto")
                unit = _safe_str(ln.get("unit") or "unidad")

                name_cell = (name[: COL_PRODUCT - 1] + "…") if len(name) > COL_PRODUCT else name
                body_lines.append(f"{name_cell:<{COL_PRODUCT}}  {_fmt_qty(qty):>{COL_QTY}}  {unit:<{COL_UNIT}}")

            body_lines += ["", closing.strip(), "", "—", footer.strip()]
            return "\n".join(body_lines).strip()
        
        def _append_tracking_link(body: str, *, order_id: int, provider_name: str) -> str:
            url = build_seguimiento_url(
                order_id=int(order_id),
                provider_name=norm_provider(provider_name),
                role=ROLE_SUPPLIER,
                page_path="seguimiento",  # must match your Streamlit Page route
            )

            block = (
                "\n\n"
                "—\n"
                "✅ Confirmación de envío\n"
                "Este enlace es solo para este pedido y proveedor.\n"
                "Por favor ábrelo para confirmar lo enviado (OK / parcial / falta) y añadir comentarios:\n"
                f"{url}\n"
            )

            return (body or "").rstrip() + block


        st.caption(f"Email language: {lang.upper()} · CC: {cc_rule or '—'} · BCC: {bcc_rule or '—'}")

        # ----------------------------
        # ✅ Preview (one expander per provider, inside st.code)
        # ----------------------------
        with st.expander("👁️ Vista previa mensaje (por proveedor)", expanded=False):
            from_display = _safe_str(st.session_state.get("user_email") or st.session_state.get("actor") or "") or "(tu email)"

            def _provider_email(prov_name: str) -> str:
                p = provider_dir.get(_norm_provider(prov_name))
                if not p:
                    return ""
                return _split_first_pipe(getattr(p, "order_email", None) or getattr(p, "emails", None))

            def _build_preview_text(prov: str, prov_lines: List[dict]) -> str:
                to_email = _provider_email(prov) or "—"
                cc = cc_rule or "—"
                bcc = bcc_rule or "—"

                body = _append_tracking_link(
                    _build_provider_body_text(prov_lines),
                    order_id=int(order.id),
                    provider_name=prov,
                )


                return (
                    f"TO: {to_email}\n"
                    f"FROM: {from_display}\n"
                    f"SUBJECT: {subject}\n"
                    f"CC/BCC: {cc} / {bcc}\n\n"
                    f"{body}\n"
                )

            for prov, prov_lines in grouped.items():
                count_ok = sum(1 for ln in prov_lines if _safe_float(ln.get("qty"), 0.0) > 0.0)
                with st.expander(f"📦 {prov} · {count_ok} línea(s)", expanded=False):
                    st.code(_build_preview_text(prov, prov_lines), language="text")

        # ----------------------------
        # ✅ Generate
        # ----------------------------
        # ----------------------------
        # ✅ Generate
        # ----------------------------
        generate_key = f"send_generated_{int(order.id)}"
        pending_after_key = f"send_pending_after_{int(order.id)}"

        def _split_emails(s: str) -> list[str]:
            s = (s or "").strip()
            if not s:
                return []
            s = s.replace("|", ",")
            return [x.strip() for x in s.split(",") if x.strip()]

        if st.button("🚀 Generar", type="primary", width="stretch", key=f"btn_gen_{int(order.id)}"):
            st.session_state[generate_key] = True
            st.session_state[pending_after_key] = bool(move_to_pending)
            st.rerun()

        if not st.session_state.get(generate_key, False):
            st.caption("Genera para ver botones de envío por proveedor.")
            return

        st.success("Envíos generados ✓")

        # ---- Sent overview (PRO) ----
        sent_map = _get_send_status_map(order_id=int(order.id))
        provider_keys = [norm_provider(p) for p in grouped.keys()]
        sent_count = sum(
            1 for pk in provider_keys
            if sent_map.get(pk) and bool(getattr(sent_map[pk], "sent", False))
        )

        st.progress(sent_count / max(1, len(provider_keys)))
        st.caption(f"📧 Estado de envío: {sent_count}/{len(provider_keys)} proveedores enviados")

        actor_email = _safe_str(st.session_state.get("user_email") or st.session_state.get("actor") or actor)



        # ---- Send all + Reset (dev) ----
        c_all1, c_all2 = st.columns([1.6, 1.0], vertical_alignment="center")

        with c_all1:
            if send_email and st.button(
                "📤 Enviar a todos (SMTP)",
                type="primary",
                width="stretch",
                key=f"send_all_{int(order.id)}",
            ):
                ok, fail = 0, 0

                for prov, prov_lines in grouped.items():
                    prov_key = norm_provider(prov)
                    p = provider_dir.get(prov_key)
                    provider_email = _split_first_pipe(getattr(p, "order_email", None) or getattr(p, "emails", None)) if p else ""

                    body_text = _append_tracking_link(
                        _build_provider_body_text(prov_lines),
                        order_id=int(order.id),
                        provider_name=prov,
                    )

                    if not provider_email:
                        fail += 1
                        _touch_send_status(
                            venue_id=venue_id,
                            order_id=int(order.id),
                            provider_name=prov_key,
                            actor=actor_email,
                            channel="email",
                            ok=False,
                            error="missing provider email",
                        )
                        continue

                    try:
                        to_list = _split_emails(provider_email)
                        cc_list = _split_emails(cc_rule)
                        bcc_list = _split_emails(bcc_rule)

                        send_smtp_email(
                            to=to_list,
                            subject=subject,
                            text_body=body_text,
                            cc=cc_list or None,
                            bcc=bcc_list or None,
                        )

                        _touch_send_status(
                            venue_id=venue_id,
                            order_id=int(order.id),
                            provider_name=prov_key,
                            actor=actor_email,
                            channel="email",
                            ok=True,
                        )
                        ok += 1

                    except Exception as e:
                        fail += 1
                        _touch_send_status(
                            venue_id=venue_id,
                            order_id=int(order.id),
                            provider_name=prov_key,
                            actor=actor_email,
                            channel="email",
                            ok=False,
                            error=str(e),
                        )

                # ✅ auto-move ONLY when all providers sent
                if _all_providers_sent(order_id=int(order.id), provider_names=list(grouped.keys())):
                    _set_order_status(int(order.id), "pending_receive", actor=actor_email)
                    _bump_refresh(venue_id)
                    st.success(f"✅ Enviados {ok} · ❌ Fallos {fail} · 📦 Pedido ahora en Pendiente")
                    # clean state
                    st.session_state.pop(generate_key, None)
                    st.session_state.pop(pending_after_key, None)
                    st.rerun()
                else:
                    st.success(f"✅ Enviados {ok} · ❌ Fallos {fail}")
                    st.rerun()

        with c_all2:
            if st.button("🧹 Reset enviados (dev)", width="stretch", key=f"reset_sent_{int(order.id)}"):
                _reset_send_status(order_id=int(order.id))
                st.rerun()


        # ----------------------------
        # Send buttons (per provider)
        # ----------------------------
        for prov, prov_lines in grouped.items():
            prov_key = norm_provider(prov)

            p = provider_dir.get(prov_key)
            provider_email = _split_first_pipe(getattr(p, "order_email", None) or getattr(p, "emails", None)) if p else ""
            provider_phone = _split_first_pipe(getattr(p, "order_phone", None) or getattr(p, "phones", None)) if p else ""

            body_text = _append_tracking_link(
                _build_provider_body_text(prov_lines),
                order_id=int(order.id),
                provider_name=prov,
            )

            # ---- Sent badge (PRO) ----
            sent_row = sent_map.get(prov_key)
            is_sent = bool(sent_row and getattr(sent_row, "sent", False))
            attempts = int(getattr(sent_row, "send_attempts", 0) or 0) if sent_row else 0
            last_error = _safe_str(getattr(sent_row, "last_error", "")) if sent_row else ""
            channels = []
            if sent_row and getattr(sent_row, "sent_email", False): channels.append("email")
            if sent_row and getattr(sent_row, "sent_whatsapp", False): channels.append("wa")
            if sent_row and getattr(sent_row, "sent_txt", False): channels.append("txt")
            channels_str = " · ".join(channels) if channels else "—"

            chip = "📧 Enviado" if is_sent else "⏳ Pendiente"

            st.markdown(
                f"<div class='voi-card'>"
                f"<div class='voi-row' style='justify-content:space-between;'>"
                f"<div><div style='font-weight:700;'>{prov}</div>"
                f"<div class='voi-muted'>{len(prov_lines)} productos</div></div>"
                f"<div class='voi-chip'>{chip}</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Canales: {channels_str} · Intentos: {attempts}" + (f" · ⚠️ {last_error}" if last_error else ""))

            a1, a2, a3 = st.columns([1, 1, 1.2], vertical_alignment="center")

            # ---- Email (mailto + SMTP) ----
            with a1:
                if send_email and provider_email:
                    q = [("subject", subject), ("body", body_text)]
                    if cc_rule:
                        q.append(("cc", cc_rule))
                    if bcc_rule:
                        q.append(("bcc", bcc_rule))
                    mailto = f"mailto:{up.quote(provider_email)}?{up.urlencode(q)}"
                    st.link_button("📧 Abrir (mailto)", mailto, width="stretch")
                    st.caption("o")

                    btn_label = "🔁 Reenviar (SMTP)" if is_sent else "✅ Enviar ahora (SMTP)"
                    if st.button(btn_label, type="primary", width="stretch", key=f"smtp_send_{int(order.id)}_{prov_key}"):
                        try:
                            to_list = _split_emails(provider_email)
                            cc_list = _split_emails(cc_rule)
                            bcc_list = _split_emails(bcc_rule)

                            send_smtp_email(
                                to=to_list,
                                subject=subject,
                                text_body=body_text,
                                cc=cc_list or None,
                                bcc=bcc_list or None,
                            )

                            _touch_send_status(
                                venue_id=venue_id,
                                order_id=int(order.id),
                                provider_name=prov_key,
                                actor=actor_email,
                                channel="email",
                                ok=True,
                            )

                            st.success("Email enviado ✅")

                            # Refresh / auto-move
                            if _all_providers_sent(order_id=int(order.id), provider_names=list(grouped.keys())):
                                _set_order_status(int(order.id), "pending_receive", actor=actor_email)
                                _bump_refresh(venue_id)
                                st.success("📦 Todos enviados → Pedido ahora en Pendiente")
                                st.session_state.pop(generate_key, None)
                                st.session_state.pop(pending_after_key, None)
                                st.rerun()
                            else:
                                st.rerun()

                        except Exception as e:
                            _touch_send_status(
                                venue_id=venue_id,
                                order_id=int(order.id),
                                provider_name=prov_key,
                                actor=actor_email,
                                channel="email",
                                ok=False,
                                error=str(e),
                            )
                            st.error(f"No se pudo enviar: {e}")
                else:
                    st.caption("Sin email")

            # ---- WhatsApp (link + optional mark) ----
            with a2:
                if send_wa and provider_phone:
                    phone_norm = _normalize_phone(provider_phone, wa_cc)
                    if phone_norm:
                        wa = f"https://wa.me/{phone_norm.replace('+','')}?text={up.quote(body_text)}"
                        st.link_button("📲 WhatsApp", wa, width="stretch")

                        # PRO: manual mark (because link click ≠ sent)
                        if st.button("✅ Marcar WA enviado", width="stretch", key=f"mark_wa_{int(order.id)}_{prov_key}"):
                            _touch_send_status(
                                venue_id=venue_id,
                                order_id=int(order.id),
                                provider_name=prov_key,
                                actor=actor_email,
                                channel="whatsapp",
                                ok=True,
                            )
                            st.rerun()
                    else:
                        st.caption("Tel inválido")
                else:
                    st.caption("Sin teléfono")

            # ---- TXT (download + optional mark) ----
            with a3:
                st.download_button(
                    "📄 TXT",
                    data=body_text.encode("utf-8"),
                    file_name=f"pedido_{_safe_str(prov).replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    width="stretch",
                    key=f"dl_{int(order.id)}_{prov_key}",
                )
                if st.button("✅ Marcar TXT enviado", width="stretch", key=f"mark_txt_{int(order.id)}_{prov_key}"):
                    _touch_send_status(
                        venue_id=venue_id,
                        order_id=int(order.id),
                        provider_name=prov_key,
                        actor=actor_email,
                        channel="txt",
                        ok=True,
                    )
                    st.rerun()

        # ---- Footer actions ----
        st.markdown('<div class="voi-divider"></div>', unsafe_allow_html=True)

        if st.button("🔄 Regenerar envíos", width="stretch", key=f"regen_{int(order.id)}"):
            st.session_state.pop(generate_key, None)
            st.rerun()

        # Fallback manual move (only if toggle was on)
        if st.session_state.get(pending_after_key, False):
            if st.button("📦 Pasar a Pendiente", type="primary", width="stretch", key=f"to_pending_{int(order.id)}"):
                _set_order_status(int(order.id), "pending_receive", actor=actor_email)
                _bump_refresh(venue_id)
                st.session_state.pop(generate_key, None)
                st.session_state.pop(pending_after_key, None)
                st.rerun()




def _render_receive_section(
    *,
    venue_id: int,
    order: Order,
    products: List[Product],
    lines: List[OrderLine],
    actor: str,
) -> None:
    """
    UX premium — Recepción & Seguimiento (Pendiente):
    - Lo pedido vs Esperado (proveedor) vs Recibido (local) vs Pendiente
    - Señala si lo faltante está en factura o no (via tickets)
    - Tickets: ver/resolver
    - Reenviar link premium al proveedor (modal + tracking ProviderSendStatus)
    """


    # ----------------------------
    # Guards
    # ----------------------------
    if (order.status or "").strip().lower() != "pending_receive":
        st.info("Esta sección solo está disponible en estado **Pendiente**.")
        return

    # ----------------------------
    # Small robust helpers (quantity/provider)
    # ----------------------------
    def _line_qty(ln: OrderLine) -> float:
        # Your schema uses OrderLine.quantity :contentReference[oaicite:7]{index=7}
        v = getattr(ln, "quantity", None)
        if v is None:
            v = getattr(ln, "qty", None)  # fallback
        try:
            return float(v or 0.0)
        except Exception:
            return 0.0

    def _line_provider(ln: OrderLine, prod: Product | None) -> str:
        # Provider can come from Product.provider_name or OrderLine.provider 
        if prod is not None and getattr(prod, "provider_name", None):
            return norm_provider(getattr(prod, "provider_name"))
        return norm_provider(getattr(ln, "provider", None))

    # ----------------------------
    # Build maps
    # ----------------------------
    products_by_id = {int(p.id): p for p in products if p.id is not None}

    # Which providers are in this order
    provider_names: list[str] = []
    for ln in lines:
        pid = getattr(ln, "product_id", None)
        prod = products_by_id.get(int(pid)) if pid is not None else None
        prov = _line_provider(ln, prod)
        if prov and prov not in provider_names:
            provider_names.append(prov)
    provider_names = sorted(provider_names, key=lambda x: x.lower())

    # ----------------------------
    # Load followups / receipts / send status / provider directory / tickets
    # ----------------------------
    with get_session() as s:
        followups = s.exec(
            select(ProviderLineFollowUp).where(
                ProviderLineFollowUp.venue_id == int(venue_id),
                ProviderLineFollowUp.order_id == int(order.id),
            )
        ).all()
        followup_by_line = {int(getattr(fu, "order_line_id")): fu for fu in followups}

        receipts = s.exec(
            select(ProviderReceipt).where(
                ProviderReceipt.venue_id == int(venue_id),
                ProviderReceipt.order_id == int(order.id),
            )
        ).all()
        receipt_by_provider = {norm_provider(getattr(r, "provider_name", "")): r for r in receipts}

        # Send status map (already exists in your codebase) :contentReference[oaicite:9]{index=9}
        send_map = _get_send_status_map(order_id=int(order.id))

        # Provider directory used in send section (cached) :contentReference[oaicite:10]{index=10}
        provider_dir = _providers_cached(get_session, int(venue_id))  # name->Provider

        tickets = s.exec(
            select(SeguimientoTicket).where(
                SeguimientoTicket.venue_id == int(venue_id),
                SeguimientoTicket.order_id == int(order.id),
            )
        ).all()

    tickets_by_line: dict[int, list] = {}
    for t in tickets:
        tickets_by_line.setdefault(int(getattr(t, "order_line_id", 0) or 0), []).append(t)

    # ----------------------------
    # Line metrics using followups (source of truth)
    # - expected: followup.supplier_qty (if supplier declared) else quantity
    # - received: followup.venue_qty (what local recorded) else 0
    # ----------------------------
    def expected_qty(ln: OrderLine) -> float:
        fu = followup_by_line.get(int(getattr(ln, "id", 0) or 0))
        if fu is not None and getattr(fu, "supplier_qty", None) is not None:
            try:
                return float(getattr(fu, "supplier_qty") or 0.0)
            except Exception:
                pass
        return _line_qty(ln)

    def received_qty(ln: OrderLine) -> float:
        fu = followup_by_line.get(int(getattr(ln, "id", 0) or 0))
        if fu is not None and getattr(fu, "venue_qty", None) is not None:
            try:
                return float(getattr(fu, "venue_qty") or 0.0)
            except Exception:
                pass
        return 0.0

    def invoice_signal_for_line(line_id: int) -> tuple[str, str]:
        """
        Returns (badge_html, label)
        - "En factura" if any ticket has invoice_number or qty_invoiced
        - else "No en factura"
        """
        ts = tickets_by_line.get(int(line_id), [])
        for tt in ts:
            inv_no = getattr(tt, "invoice_number", None)
            qty_inv = getattr(tt, "qty_invoiced", None)
            if (inv_no and str(inv_no).strip()) or (qty_inv is not None):
                return ("<span class='rx-badge rx-info'>🧾 En factura</span>", "En factura")
        return ("<span class='rx-badge'>🧾 No en factura</span>", "No en factura")

    # ----------------------------
    # CSS (premium)
    # ----------------------------
    st.markdown(
        """
        <style>
        .rx-wrap{max-width:1200px;margin:0 auto;}
        .rx-title{font-size:1.25rem;font-weight:850;margin:8px 0 2px;}
        .rx-sub{opacity:.75;margin-bottom:10px}
        .rx-grid{display:grid;grid-template-columns: 1.6fr 1fr 1fr 1fr;gap:10px;margin:10px 0 14px;}
        .rx-card{border:1px solid rgba(49,51,63,.14);border-radius:16px;padding:12px;background:rgba(255,255,255,.55);}
        .rx-card h4{margin:0 0 6px 0;font-size:1rem}
        .rx-kpi{font-size:1.35rem;font-weight:850}
        .rx-muted{opacity:.7;font-size:.9rem}
        .rx-badge{display:inline-block;padding:4px 10px;border-radius:999px;border:1px solid rgba(49,51,63,.18);font-size:.85rem}
        .rx-ok{background:rgba(46,204,113,.12);border-color:rgba(46,204,113,.35)}
        .rx-warn{background:rgba(241,196,15,.14);border-color:rgba(241,196,15,.35)}
        .rx-bad{background:rgba(231,76,60,.12);border-color:rgba(231,76,60,.35)}
        .rx-info{background:rgba(33,150,243,.10);border-color:rgba(33,150,243,.35)}
        .rx-row{border:1px solid rgba(49,51,63,.14);border-radius:16px;padding:12px;background:rgba(255,255,255,.55);margin:10px 0;}
        .rx-row h5{margin:0 0 6px 0;font-size:1rem}
        .rx-chip{display:inline-flex;align-items:center;gap:8px;padding:6px 10px;border-radius:999px;
                 border:1px solid rgba(49,51,63,.18); background:rgba(255,255,255,.35); font-size:.9rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ----------------------------
    # Compute KPIs
    # ----------------------------
    total_ordered = sum(_line_qty(l) for l in lines)
    total_expected = sum(expected_qty(l) for l in lines)
    total_received = sum(received_qty(l) for l in lines)
    total_waiting = sum(max(0.0, expected_qty(l) - received_qty(l)) for l in lines)

    open_tickets = [t for t in tickets if (getattr(t, "state", "open") == "open")]

    # ----------------------------
    # Header + KPIs
    # ----------------------------
    st.markdown('<div class="rx-wrap">', unsafe_allow_html=True)
    st.markdown("<div class='rx-title'>📦 Recepción & Seguimiento (Premium)</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='rx-sub'>Ver lo pedido, lo esperado, lo pendiente y si lo faltante está en factura. "
        "Además puedes reenviar el link al proveedor con tracking.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="rx-grid">
          <div class="rx-card">
            <h4>Pedido</h4>
            <div class="rx-kpi">{total_ordered:g}</div>
            <div class="rx-muted">suma de cantidades</div>
          </div>
          <div class="rx-card">
            <h4>Esperado (proveedor)</h4>
            <div class="rx-kpi">{total_expected:g}</div>
            <div class="rx-muted">si no declaró → = pedido</div>
          </div>
          <div class="rx-card">
            <h4>Recibido (local)</h4>
            <div class="rx-kpi">{total_received:g}</div>
            <div class="rx-muted">según seguimiento</div>
          </div>
          <div class="rx-card">
            <h4>Pendiente</h4>
            <div class="rx-kpi">{total_waiting:g}</div>
            <div class="rx-muted">esperado − recibido</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_pend, tab_lines, tab_tickets = st.tabs(["⏳ Pendientes", "📦 Detalle líneas", "🎫 Tickets"])

    # ----------------------------
    # Build pending rows grouped by provider
    # ----------------------------
    pending_by_provider: dict[str, list[dict]] = {}
    all_rows: list[dict] = []

    for ln in lines:
        line_id = int(getattr(ln, "id", 0) or 0)
        pid = getattr(ln, "product_id", None)
        prod = products_by_id.get(int(pid)) if pid is not None else None

        prov = _line_provider(ln, prod)
        name = _safe_str(getattr(prod, "name", None) if prod else getattr(ln, "matched_name", None) or getattr(ln, "spoken_name", None) or "Producto")
        unit = _safe_str(getattr(prod, "unit", None) if prod else getattr(ln, "unit", None)) or "unidad"

        ordered = _line_qty(ln)
        exp = expected_qty(ln)
        rec = received_qty(ln)
        wait = max(0.0, exp - rec)

        inv_badge_html, inv_label = invoice_signal_for_line(line_id)

        row = {
            "Proveedor": prov,
            "Producto": name,
            "Unidad": unit,
            "Pedido": ordered,
            "Esperado": exp,
            "Recibido": rec,
            "Pendiente": wait,
            "Factura": inv_label,
            "_inv_badge_html": inv_badge_html,
            "line_id": line_id,
        }
        all_rows.append(row)
        if wait > 0.000001:
            pending_by_provider.setdefault(prov, []).append(row)

    # ----------------------------
    # ⏳ Pendientes tab
    # ----------------------------
    with tab_pend:
        if not pending_by_provider:
            st.success("No hay pendientes. ✅")
        else:
            # Load templates once for premium resend dialog (auth template helper exists in your file)
            templates = _load_venue_templates(int(venue_id), _refresh_token(int(venue_id)))

            for prov in sorted(pending_by_provider.keys(), key=lambda x: x.lower()):
                rows = pending_by_provider[prov]

                rcp = receipt_by_provider.get(norm_provider(prov))
                inv_no = _safe_str(getattr(rcp, "invoice_number", "")) if rcp else ""
                decl = _safe_str(getattr(rcp, "supplier_declaration", "")) if rcp else ""
                decl_chip = (
                    "<span class='rx-badge rx-ok'>Proveedor: full</span>"
                    if decl == "full"
                    else ("<span class='rx-badge rx-warn'>Proveedor: partial</span>" if decl == "partial" else ("<span class='rx-badge rx-bad'>Proveedor: none</span>" if decl == "none" else "<span class='rx-badge'>Proveedor: —</span>"))
                )

                # Send status chip (sent/attempts/error) :contentReference[oaicite:11]{index=11}
                srow = send_map.get(norm_provider(prov))
                sent = bool(getattr(srow, "sent", False)) if srow else False
                attempts = int(getattr(srow, "send_attempts", 0) or 0) if srow else 0
                last_err = _safe_str(getattr(srow, "last_error", "")) if srow else ""
                sent_at = getattr(srow, "sent_at", None) if srow else None

                send_chip = (
                    "<span class='rx-badge rx-ok'>📧 Enviado</span>"
                    if sent
                    else ("<span class='rx-badge rx-bad'>⚠️ Error</span>" if last_err else "<span class='rx-badge rx-warn'>⏳ No enviado</span>")
                )

                st.markdown(
                    f"""
                    <div class="rx-row">
                      <h5>{prov}</h5>
                      <div class="rx-muted">
                        {decl_chip} &nbsp; {send_chip} &nbsp;
                        🧾 Factura: <b>{inv_no or "—"}</b>
                        &nbsp; · &nbsp; Intentos: <b>{attempts}</b>
                        {" &nbsp;·&nbsp; Último envío: <b>"+sent_at.strftime("%Y-%m-%d %H:%M")+"</b>" if sent_at else ""}
                        {" &nbsp;·&nbsp; Último error: <b>"+last_err+"</b>" if last_err else ""}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Premium actions row (open link + resend modal)
                c1, c2, c3 = st.columns([1.0, 1.2, 1.0], vertical_alignment="center")
                link_url = build_seguimiento_url(
                    order_id=int(order.id),
                    provider_name=prov,
                    role=ROLE_SUPPLIER,
                    page_path="seguimiento",
                )

                with c1:
                    st.link_button("🔗 Abrir link", link_url, use_container_width=True)

                with c2:
                    st.code(link_url, language=None)

                with c3:
                    p_row = provider_dir.get(norm_provider(prov))
                    if st.button("🔁 Reenviar (premium)", key=f"rx_resend_premium_{int(order.id)}_{norm_provider(prov)}", use_container_width=True):
                        # Uses your premium modal helper :contentReference[oaicite:12]{index=12}
                        _render_resend_dialog(
                            venue_id=int(venue_id),
                            order=order,
                            provider_name=norm_provider(prov),
                            provider_row=p_row,
                            templates=templates,
                            send_status=srow,
                            actor=actor,
                        )

                # Pending items list
                for r in rows:
                    st.markdown(
                        f"""
                        <div class="rx-row">
                          <h5>{r["Producto"]} {r["_inv_badge_html"]}</h5>
                          <div class="rx-muted">
                            Pedido: <b>{r["Pedido"]:g}</b> {r["Unidad"]} · Esperado: <b>{r["Esperado"]:g}</b> {r["Unidad"]} · Recibido: <b>{r["Recibido"]:g}</b> {r["Unidad"]}
                          </div>
                          <div style="margin-top:8px">
                            <span class="rx-badge rx-bad">⏳ Pendiente: {r["Pendiente"]:g} {r["Unidad"]}</span>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # ----------------------------
    # 📦 Detalle líneas tab
    # ----------------------------
    with tab_lines:
        st.dataframe(
            [{k: v for k, v in r.items() if not k.startswith("_")} for r in all_rows],
            use_container_width=True,
            hide_index=True,
        )

    # ----------------------------
    # 🎫 Tickets tab
    # ----------------------------
    with tab_tickets:

        show_resolved = st.toggle("Mostrar resueltos", value=False, key=f"rx_show_res_{int(order.id)}")
        view = tickets if show_resolved else [t for t in tickets if getattr(t, "state", "open") == "open"]

        if not view:
            st.info("No hay tickets para este pedido.")
        else:
            # grouped by provider
            by_prov: dict[str, list] = {}
            for t in view:
                by_prov.setdefault(norm_provider(getattr(t, "provider_name", "")), []).append(t)

            for prov in sorted(by_prov.keys(), key=lambda x: x.lower()):
                st.markdown(
                    f"""
                    <div class="rx-row">
                        <h5>🎫 Tickets · {prov} · {len(by_prov[prov])} {'(incl. resueltos)' if show_resolved else 'abiertos'}</h5>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                for t in by_prov[prov]:
                    kind = _safe_str(getattr(t, "kind", "ticket"))
                    kind_label = {
                        "supplier_short": "Proveedor declara falta",
                        "delivery_mismatch": "Recepción no cuadra",
                        "invoice_mismatch": "Factura no cuadra",
                    }.get(kind, kind)

                    state = _safe_str(getattr(t, "state", "open"))
                    state_badge = (
                        "<span class='rx-badge rx-bad'>ABIERTO</span>"
                        if state == "open"
                        else "<span class='rx-badge rx-ok'>RESUELTO</span>"
                    )

                    inv_no = _safe_str(getattr(t, "invoice_number", ""))
                    inv_qty = getattr(t, "qty_invoiced", None)

                    st.markdown(
                        f"""
                        <div class="rx-row">
                            <h5>{kind_label} {state_badge}</h5>
                            <div class="rx-muted">
                            { _safe_str(getattr(t,'product_name','')) }
                            · Pedido: <b>{float(getattr(t,'qty_ordered',0.0) or 0.0):g}</b>
                            · Esperado: <b>{float(getattr(t,'qty_expected',0.0) or 0.0):g}</b>
                            · Recibido: <b>{float(getattr(t,'qty_received',0.0) or 0.0):g}</b>
                            {(" · 🧾 Factura: <b>"+inv_no+"</b>" if inv_no else "")}
                            {(" · Qty facturada: <b>"+str(inv_qty)+"</b>" if inv_qty is not None else "")}
                            </div>
                            {("<div class='rx-muted' style='margin-top:6px'>📝 "+_safe_str(getattr(t,'note',''))+"</div>" if _safe_str(getattr(t,'note','')) else "")}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if state == "open":
                        res = st.text_input(
                            "Resolución (opcional)",
                            key=f"rx_res_note_{int(getattr(t,'id'))}",
                            placeholder="Ej. Reposición mañana / Nota de crédito / Ajuste contable",
                        )
                        if st.button(
                            "✅ Resolver ticket",
                            key=f"rx_resolve_{int(getattr(t,'id'))}",
                            type="primary",
                            use_container_width=True,
                        ):
                            # reuse resolver from seguimiento module if present
                            try:
                                from features.seguimiento.seguimiento import _resolve_ticket  # type: ignore
                                _resolve_ticket(int(t.id), res)
                                st.success("Ticket resuelto ✅")
                                st.rerun()
                            except Exception as e:
                                st.error(f"No se pudo resolver el ticket: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

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
                width='stretch',
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
    _render_provider_chips(order=order, lines=lines, products=products)
    _render_workflow_actions(venue_id=venue_id, order=order, role=venue_role, actor=actor)

    st.markdown('<div class="voi-divider"></div>', unsafe_allow_html=True)

    # Sections (simple)
    if (order.status or "").strip().lower() == "draft":
        _render_lines_editor(venue_id=venue_id, order=order, actor=actor, products=products, lines=lines)

    if (order.status or "").strip().lower() == "ready_to_send":
        st.markdown('<div class="voi-divider"></div>', unsafe_allow_html=True)
        _render_send_section(venue_id=venue_id, order=order, products=products, lines=lines, actor=actor)

    if (order.status or "").strip().lower() == "pending_receive":
        st.markdown('<div class="voi-divider"></div>', unsafe_allow_html=True)
        _render_receive_section(venue_id=venue_id, order=order, products=products, lines=lines, actor=actor)

