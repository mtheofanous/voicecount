"""
Modern Orders UX (Streamlit)

This file replaces the old Orders tab with a simpler, modern flow:

DRAFT  -> build the order lines
READY  -> send per supplier (email/wa/txt) + creates workflow state ORDER_SENT per supplier
PENDING-> quick overview + open the seguimiento (venue) link per supplier
FINAL  -> read-only history

IMPORTANT:
- This file does NOT handle payments.
- It does NOT issue invoices / credit notes.
- It only sends links and tracks send attempts.
- Operational state machine lives in OrderWorkflow / OrderWorkflowEvent and the seguimiento page.

Source base: previous refactor file. fileciteturn2file0
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
import hashlib
import re
import unicodedata
import urllib.parse as up

import pandas as pd
import streamlit as st
from sqlmodel import select

from core.db import get_session
from core.public_links import build_seguimiento_url, ROLE_SUPPLIER, ROLE_VENUE, norm_provider
from core.mailer import send_smtp_email


# from features.manage_orders.receive_orders import _render_pending_panel, _render_incidences_tab
# try:
#     from features.manage_orders.receive_orders import render_receive_panel, render_incidences_tab
#     _render_pending_panel = render_receive_panel
#     _render_incidences_tab = render_incidences_tab
# except ImportError:
#     # Fallback σε απλούστερη έκδοση αν δεν υπάρχει το αρχείο
#     def _render_pending_panel(venue_id: int, order):
#         from features.seguimiento.seguimiento import modern_seguimiento_page
#         # Εμφάνιση του seguimiento tab για το order
#         modern_seguimiento_page(venue_id=venue_id, order_id=order.id)
    
    # def _render_incidences_tab(venue_id: int):
    #     st.info("Η ενότητα 'Incidencias' δεν είναι διαθέσιμη αυτήν τη στιγμή.")
        
from domain.models import (
    Order,
    OrderLine,
    Product,
    Provider,
    ProviderSendStatus,
    ProviderDiscountRule,
)

try:
    from domain.models import ProviderReceipt  # type: ignore
except Exception:  # pragma: no cover
    ProviderReceipt = None  # type: ignore

try:
    from domain.models import OrderWorkflow, OrderWorkflowEvent  # type: ignore
except Exception:  # pragma: no cover
    OrderWorkflow = None  # type: ignore
    OrderWorkflowEvent = None  # type: ignore

try:
    from features.auth_and_manage.auth_multi_tenant import get_auth_session, Venue, User  # type: ignore
except Exception:  # pragma: no cover
    get_auth_session = None  # type: ignore
    Venue = None  # type: ignore
    User = None  # type: ignore


def current_actor() -> str:
    """Compatibility: other modules import current_actor() from here."""
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

    return (st.session_state.get("actor") or "system")


def _now():
    from datetime import datetime as _dt

    return _dt.utcnow()


def _s(x: Any) -> str:
    return ("" if x is None else str(x)).strip()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _split_first_pipe(value: Optional[str]) -> str:
    raw = _s(value)
    if not raw:
        return ""
    for part in raw.split("|"):
        p = part.strip()
        if p:
            return p
    return ""


def _split_emails(raw: str) -> list[str]:
    raw = _s(raw)
    if not raw:
        return []
    raw = raw.replace(",", "|")
    return [p.strip() for p in raw.split("|") if p.strip()]


def _normalize_phone(raw: str, country_code: str) -> str:
    raw = _s(raw)
    if not raw:
        return ""
    digits = "".join(ch for ch in raw if ch.isdigit() or ch == "+")
    if not digits:
        return ""
    if digits.startswith("+"):
        return digits
    digits2 = digits.lstrip("0")
    cc = _s(country_code) or "+34"
    if not cc.startswith("+"):
        cc = "+" + cc
    return f"{cc}{digits2}"


def _order_label(o: Order) -> str:
    title = _s(getattr(o, "title", "")) or ""
    when = getattr(o, "created_at", None)
    when_s = when.strftime("%Y-%m-%d %H:%M") if when else ""
    base = f"#{int(o.id)}" if getattr(o, "id", None) is not None else "#—"
    return f"{base} — {title}" if title else (f"{base} — {when_s}" if when_s else base)


def _status_chip(status: str) -> str:
    s = (_s(status)).lower()
    if s == "draft":
        return "📝 Borrador"
    if s == "ready_to_send":
        return "📤 Listo"
    if s == "pending_receive":
        return "📦 Pendiente"
    if s == "final":
        return "✅ Historial"
    return s or "—"


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)


def _norm_words(s: str) -> list[str]:
    raw = re.findall(r"[0-9]+|[^\W_]+", (s or ""), flags=re.UNICODE)
    return [_strip_accents(w).lower() for w in raw if w.strip()]


def _remove_name_words_from_description(name: str, desc: str) -> str:
    name_set = set(_norm_words(name))
    if not name_set:
        return (desc or "").strip()
    desc_raw = re.findall(r"[0-9]+|[^\W_]+", (desc or ""), flags=re.UNICODE)
    kept: list[str] = []
    for w in desc_raw:
        if _strip_accents(w).lower() not in name_set:
            kept.append(w)
    return " ".join(kept).strip()


@st.cache_data(ttl=10, show_spinner=False, hash_funcs={type(lambda: None): lambda _: "session_fn"})
def _list_orders_cached(_get_session_fn, venue_id: int, refresh_token: int) -> list[Order]:
    _ = refresh_token
    with _get_session_fn() as s:
        return list(
            s.exec(select(Order).where(Order.venue_id == venue_id).order_by(Order.created_at.desc())).all()
        )


@st.cache_data(ttl=60, show_spinner=False, hash_funcs={type(lambda: None): lambda _: "session_fn"})
def _products_cached(_get_session_fn, venue_id: int) -> list[Product]:
    with _get_session_fn() as s:
        return list(
            s.exec(select(Product).where(Product.venue_id == venue_id).order_by(Product.name.asc(), Product.provider_name.asc())).all()
        )


@st.cache_data(ttl=10, show_spinner=False, hash_funcs={type(lambda: None): lambda _: "session_fn"})
def _order_lines_cached(_get_session_fn, order_id: int, refresh_token: int) -> list[OrderLine]:
    _ = refresh_token
    with _get_session_fn() as s:
        return list(s.exec(select(OrderLine).where(OrderLine.order_id == order_id).order_by(OrderLine.id.asc())).all())


@st.cache_data(ttl=60, show_spinner=False, hash_funcs={type(lambda: None): lambda _: "session_fn"})
def _providers_cached(_get_session_fn, venue_id: int) -> dict[str, Provider]:
    with _get_session_fn() as s:
        rows = list(s.exec(select(Provider).where(Provider.venue_id == venue_id)).all())
    out: dict[str, Provider] = {}
    for p in rows:
        out[norm_provider(_s(getattr(p, "name", "")))] = p
    return out


@st.cache_data(ttl=10, show_spinner=False, hash_funcs={type(lambda: None): lambda _: "session_fn"})
def _provider_receipts_cached(_get_session_fn, order_id: int, refresh_token: int) -> dict[str, Any]:
    _ = refresh_token
    if ProviderReceipt is None:
        return {}
    with _get_session_fn() as s:
        rows = list(s.exec(select(ProviderReceipt).where(ProviderReceipt.order_id == order_id)).all())
    return {norm_provider(_s(getattr(r, "provider_name", ""))): r for r in rows}


def _bump_refresh(venue_id: int) -> None:
    k = f"orders_refresh_token_{venue_id}"
    st.session_state[k] = int(st.session_state.get(k, 0)) + 1


def _refresh_token(venue_id: int) -> int:
    return int(st.session_state.get(f"orders_refresh_token_{venue_id}", 0))


@dataclass
class VenueTemplates:
    venue_name: str = ""
    owner_name: str = ""
    address: str = ""
    tax_number: str = ""
    email: str = ""
    phone: str = ""
    email_lang: str = "es"
    email_cc: str = ""
    email_bcc: str = ""


@st.cache_data(ttl=120, show_spinner=False)
def _load_venue_templates(venue_id: int, refresh_token: int) -> VenueTemplates:
    _ = refresh_token
    if get_auth_session is None or Venue is None or User is None:
        return VenueTemplates()
    v = None
    owner_name = ""
    try:
        with get_auth_session() as s:
            v = s.exec(select(Venue).where(Venue.id == venue_id)).first()
            if v:
                owner_name = (
                    s.exec(
                        select(User.full_name)
                        .where(User.account_id == v.account_id, User.account_role == "owner", User.is_active == True)  # noqa
                        .order_by(User.created_at.asc())
                    ).first()
                    or ""
                )
    except Exception:
        pass
    if not v:
        return VenueTemplates()
    email_lang = _s(getattr(v, "email_lang", "")).lower() or "es"
    if email_lang not in {"es", "en", "gr"}:
        email_lang = "es"
    return VenueTemplates(
        venue_name=_s(getattr(v, "name", "")),
        owner_name=_s(owner_name),
        address=_s(getattr(v, "address", "")),
        tax_number=_s(getattr(v, "tax_number", "")),
        email=_s(getattr(v, "email", "")),
        phone=_s(getattr(v, "phone", "")),
        email_lang=email_lang,
        email_cc=_s(getattr(v, "email_cc", "")),
        email_bcc=_s(getattr(v, "email_bcc", "")),
    )


def _venue_missing_required(v: VenueTemplates) -> list[str]:
    missing: list[str] = []
    if not v.tax_number:
        missing.append("NIF/CIF (tax number)")
    if not v.address:
        missing.append("Dirección (address)")
    return missing


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


def _delete_order(order_id: int) -> None:
    with get_session() as s:
        for ln in s.exec(select(OrderLine).where(OrderLine.order_id == int(order_id))).all():
            s.delete(ln)
        o = s.exec(select(Order).where(Order.id == order_id)).first()
        if o:
            s.delete(o)
        s.commit()


def _pid_to_int(pid: Any) -> Optional[int]:
    if pid is None:
        return None
    if isinstance(pid, (list, tuple)):
        pid = pid[0] if pid else None
    try:
        if pid is not None and pd.isna(pid):
            return None
    except Exception:
        pass
    if isinstance(pid, int):
        return pid
    if isinstance(pid, float):
        return int(pid)
    s = str(pid).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _sanitize_editor_df(df0: pd.DataFrame) -> pd.DataFrame:
    df1 = df0.copy()
    for col, default in [
        ("line_id", pd.NA),
        ("product_id", pd.NA),
        ("quantity", 0.0),
        ("unit", "unidad"),
        ("delete", False),
    ]:
        if col not in df1.columns:
            df1[col] = default
    df1["product_id"] = df1["product_id"].apply(_pid_to_int)
    df1["product_id"] = df1["product_id"].where(df1["product_id"].notna(), pd.NA)
    df1["quantity"] = pd.to_numeric(df1["quantity"], errors="coerce").fillna(0.0)
    df1["unit"] = df1["unit"].astype(str).replace({"nan": "unidad"}).fillna("unidad")
    df1["delete"] = df1["delete"].fillna(False).astype(bool)
    return df1


def _save_lines_from_editor(*, venue_id: int, order_id: int, actor: str, df: pd.DataFrame, products_by_id: dict[int, Product]) -> None:
    df2 = _sanitize_editor_df(df)
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
            prod = products_by_id.get(pid)
            provider_name = _s(getattr(prod, "provider_name", "")) if prod else ""
            if pd.notna(line_id):
                ln = s.exec(select(OrderLine).where(OrderLine.id == int(line_id))).first()
                if ln:
                    ln.product_id = pid
                    ln.quantity = qty
                    ln.unit = (_s(getattr(prod, "unit", "")) or _s(r.get("unit")) or "unidad").lower()
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
                        unit=(_s(getattr(prod, "unit", "")) or "unidad").lower(),
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


def _get_send_status_map(*, order_id: int) -> dict[str, ProviderSendStatus]:
    with get_session() as s:
        rows = s.exec(select(ProviderSendStatus).where(ProviderSendStatus.order_id == int(order_id))).all()
    return {norm_provider(_s(r.provider_name)): r for r in rows}


def _touch_send_status(*, venue_id: int, order_id: int, provider_name: str, actor: str, channel: str, ok: bool, error: str | None = None) -> None:
    prov = norm_provider(provider_name)
    now = _now()
    with get_session() as s:
        obj = s.exec(select(ProviderSendStatus).where(ProviderSendStatus.order_id == int(order_id), ProviderSendStatus.provider_name == prov)).first()
        if not obj:
            obj = ProviderSendStatus(venue_id=int(venue_id), order_id=int(order_id), provider_name=prov)
        obj.send_attempts = int(getattr(obj, "send_attempts", 0) or 0) + 1
        obj.updated_at = now
        if ok:
            obj.sent = True
            obj.sent_at = now
            obj.sent_by = actor or None
            obj.last_error = None
            if channel == "email":
                obj.sent_email = True
            elif channel == "whatsapp":
                obj.sent_whatsapp = True
            elif channel == "txt":
                obj.sent_txt = True
        else:
            obj.last_error = (_s(error) or "unknown error")[:500]
        s.add(obj)
        s.commit()


def _all_providers_sent(*, order_id: int, provider_names: list[str]) -> bool:
    provs = {norm_provider(p) for p in provider_names}
    m = _get_send_status_map(order_id=order_id)
    sent_provs = {p for p, row in m.items() if bool(getattr(row, "sent", False))}
    return provs.issubset(sent_provs)


def _reset_send_status(*, order_id: int) -> None:
    with get_session() as s:
        rows = s.exec(select(ProviderSendStatus).where(ProviderSendStatus.order_id == int(order_id))).all()
        for r in rows:
            r.sent = False
            r.sent_email = False
            r.sent_whatsapp = False
            r.sent_txt = False
            r.sent_at = None
            r.sent_by = None
            r.send_attempts = 0
            r.last_error = None
            r.updated_at = _now()
            s.add(r)
        s.commit()


def _ensure_workflow_order_sent(*, venue_id: int, order_id: int, provider_name: str, actor: str) -> None:
    if OrderWorkflow is None:
        return
    prov = norm_provider(provider_name)
    now = _now()
    with get_session() as s:
        wf = s.exec(select(OrderWorkflow).where(OrderWorkflow.venue_id == int(venue_id), OrderWorkflow.order_id == int(order_id), OrderWorkflow.provider_name == prov)).first()
        if not wf:
            wf = OrderWorkflow(
                venue_id=int(venue_id),
                order_id=int(order_id),
                provider_name=prov,
                state="ORDER_SENT",
                updated_at=now,
                updated_by_role="venue",
                updated_by=actor,
                note="Order link sent to supplier.",
            )
            s.add(wf)
            if OrderWorkflowEvent is not None:
                s.add(
                    OrderWorkflowEvent(
                        venue_id=int(venue_id),
                        order_id=int(order_id),
                        provider_name=prov,
                        from_state="ORDER_SENT",
                        to_state="ORDER_SENT",
                        actor_role="venue",
                        actor=actor,
                        at=now,
                        note="Order link sent to supplier.",
                    )
                )
        s.commit()


def _products_label_map(products: list[Product]) -> dict[int, str]:
    out: dict[int, str] = {}
    for p in products:
        if p.id is None:
            continue
        name = _s(getattr(p, "name", ""))
        desc = _s(getattr(p, "description", ""))
        desc_clean = _remove_name_words_from_description(name, desc)
        prov = _s(getattr(p, "provider_name", "")) or "(Sin proveedor)"
        parts = [name] + ([desc_clean] if desc_clean else []) + [prov]
        out[int(p.id)] = " — ".join([x for x in parts if _s(x)])
    return out


def _group_lines_by_provider(lines: list[OrderLine], products_by_id: dict[int, Product]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for ln in lines:
        pid = getattr(ln, "product_id", None)
        prod = products_by_id.get(int(pid)) if pid is not None else None
        prov = norm_provider(_s(getattr(prod, "provider_name", None) if prod else getattr(ln, "provider", None)))
        name = _s(getattr(prod, "name", None) if prod else getattr(ln, "spoken_name", None) or "Producto")
        unit = (_s(getattr(prod, "unit", None)) if prod else _s(getattr(ln, "unit", None))) or "unidad"
        qty = _safe_float(getattr(ln, "quantity", 0.0), 0.0)
        out.setdefault(prov, []).append({"line_id": int(getattr(ln, "id", 0) or 0), "product_id": pid, "name": name, "qty": qty, "unit": unit})
    for prov in out:
        out[prov] = sorted(out[prov], key=lambda x: (x["name"] or "").lower())
    return dict(sorted(out.items(), key=lambda kv: kv[0].lower()))


def _editor_df_from_lines(lines: list[OrderLine], products_by_id: dict[int, Product]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ln in lines:
        pid = getattr(ln, "product_id", None)
        prod = products_by_id.get(int(pid)) if pid is not None else None
        rows.append(
            {
                "line_id": getattr(ln, "id", None),
                "product_id": pid,
                "quantity": float(getattr(ln, "quantity", 0.0) or 0.0),
                "unit": (_s(getattr(prod, "unit", "")) or _s(getattr(ln, "unit", "")) or "unidad").lower(),
                "delete": False,
            }
        )
    if not rows:
        rows = [{"line_id": pd.NA, "product_id": pd.NA, "quantity": 1.0, "unit": "unidad", "delete": False}]
    return pd.DataFrame(rows)


def _build_supplier_message_text(*, templates: VenueTemplates, order_id: int, provider_name: str, prov_lines: list[dict[str, Any]], products_by_id: dict[int, Product]) -> str:
    venue = templates.venue_name or "Pedido"
    date_str = datetime.now().strftime("%Y-%m-%d")
    supplier_link = build_seguimiento_url(order_id=order_id, provider_name=provider_name, role=ROLE_SUPPLIER, page_path="seguimiento")
    txt: list[str] = []
    txt.append(f"{venue} — Pedido #{order_id} — {date_str}")
    txt.append("")
    txt.append("Hola,")
    txt.append("Te comparto el pedido y el link para confirmar el envío (full / partial / none).")
    txt.append("")
    txt.append("LINK (confirmación proveedor):")
    txt.append(supplier_link)
    txt.append("")
    txt.append("PEDIDO:")
    for it in prov_lines:
        qty = _safe_float(it.get("qty"), 0.0)
        if qty <= 0:
            continue
        pid = _pid_to_int(it.get("product_id"))
        prod = products_by_id.get(pid) if pid else None
        name = _s(getattr(prod, "name", None) if prod else it.get("name") or "Producto")
        unit = (_s(getattr(prod, "unit", "")) if prod else _s(it.get("unit", ""))) or "unidad"
        txt.append(f"- {name} · {qty:g} {unit}")
    return "\n".join(txt).strip()


def _build_subject(templates: VenueTemplates, order_id: int) -> str:
    venue = templates.venue_name or "Pedido"
    date_str = datetime.now().strftime("%Y-%m-%d")
    return f"Pedido #{order_id} — {venue} — {date_str}"


def _inject_css() -> None:
    st.markdown(
        """
        
<style>
.voi-pill{
  display:inline-block;
  padding:4px 8px;
  border-radius:999px;
  border:1px solid rgba(49,51,63,.2);
  background:rgba(49,51,63,.04);
  font-size:.85rem;
  white-space:nowrap;
}
.voi-pill--save{
  border-color: rgba(46,125,50,.35);
  background: rgba(46,125,50,.10);
}
.voi-pill--vat{
  border-color: rgba(245,124,0,.35);
  background: rgba(245,124,0,.10);
}
.voi-pill--total{
  border-color: rgba(25,118,210,.35);
  background: rgba(25,118,210,.10);
}
.voi-pill--total b{ font-weight: 900; }
.voi-chiprow{
  display:flex;
  gap:6px;
  flex-wrap:nowrap;
  overflow-x:auto;
  -webkit-overflow-scrolling: touch;
  margin-top:8px;
  padding-bottom:4px;
}
.voi-chiprow::-webkit-scrollbar{ height:6px; }
.voi-chiprow::-webkit-scrollbar-thumb{ background: rgba(49,51,63,.25); border-radius:999px; }
</style>

<style>
        .voi-card{border:1px solid rgba(49,51,63,.12);border-radius:18px;padding:14px 14px;margin:10px 0;background:rgba(255,255,255,.03);}
        .voi-muted{opacity:.72;font-size:.9rem;}
        .voi-chip{display:inline-block;padding:6px 12px;border-radius:999px;border:1px solid rgba(49,51,63,.18);font-size:.85rem;}
        .voi-chip-ok{background:rgba(46,160,67,.12);border-color:rgba(46,160,67,.35);}
        .voi-chip-warn{background:rgba(255,159,10,.12);border-color:rgba(255,159,10,.35);}
        .voi-chip-bad{background:rgba(255,69,58,.12);border-color:rgba(255,69,58,.35);}
        .voi-divider{height:1px;background:rgba(49,51,63,.12);margin:14px 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _chip_for_send(sent: bool, last_error: str) -> tuple[str, str]:
    if sent:
        return ("📧 Enviado", "voi-chip voi-chip-ok")
    if last_error:
        return ("⚠️ Error", "voi-chip voi-chip-bad")
    return ("⏳ Pendiente", "voi-chip voi-chip-warn")


def _render_header(order: Order) -> None:
    c1, c2 = st.columns([2.2, 1], vertical_alignment="center")
    with c1:
        st.caption(f"{_status_chip(order.status)} · Creado: {getattr(order,'created_at',None).strftime('%Y-%m-%d %H:%M') if getattr(order,'created_at',None) else '—'}")
    with c2:
        st.markdown(f"<div class='voi-chip'>{_status_chip(order.status)}</div>", unsafe_allow_html=True)


def _render_workflow_actions(*, venue_id: int, order: Order, role: Optional[str], actor: str) -> None:
    st.markdown("<div class='voi-divider'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.4, 1.4, 1.2], vertical_alignment="center")
    status = (_s(order.status)).lower()
    can_manage = (_s(role)).lower() in {"owner", "manager"}
    with c1:
        if status == "draft":
            if st.button("✅ Pasar a Listo", type="primary", use_container_width=True):
                _set_order_status(int(order.id), "ready_to_send", actor); _bump_refresh(venue_id); st.rerun()
        elif status == "ready_to_send":
            if st.button("↩️ Volver a Borrador", use_container_width=True):
                _set_order_status(int(order.id), "draft", actor); _bump_refresh(venue_id); st.rerun()
    with c2:
        if status == "pending_receive":
            if st.button("✅ Cerrar (Historial)", type="primary", use_container_width=True):
                _set_order_status(int(order.id), "final", actor); _bump_refresh(venue_id); st.rerun()
    with c3:
        if status == "draft" and can_manage:
            if st.button("🗑️ Eliminar", use_container_width=True):
                _delete_order(int(order.id)); _bump_refresh(venue_id); st.session_state.pop(f"orders_active_order_id_{venue_id}", None); st.rerun()


def _render_lines_editor(*, venue_id: int, order: Order, actor: str, products: list[Product], lines: list[OrderLine]) -> None:
    st.subheader("🧾 Líneas del pedido")
    products_by_id = {int(p.id): p for p in products if p.id is not None}
    label_by_id = _products_label_map(products)
    editor_key = f"order_editor_{int(order.id)}"
    df_state_key = f"{editor_key}__df"
    if df_state_key not in st.session_state:
        st.session_state[df_state_key] = _sanitize_editor_df(_editor_df_from_lines(lines, products_by_id))
    else:
        st.session_state[df_state_key] = _sanitize_editor_df(st.session_state[df_state_key])

    
    # -----------------------------
    # Quick add expander (cascading filters + paging + toggle)
    # -----------------------------

    # Build options from products
    all_categories = sorted({
        (_s(getattr(p, "category", "")) or "").strip()
        for p in products
        if _s(getattr(p, "category", "")).strip()
    })
    all_providers = sorted({
        (_s(getattr(p, "provider_name", "")) or "(Sin proveedor)").strip()
        for p in products
    })

    # Maps
    cat_by_pid = {
        int(p.id): (_s(getattr(p, "category", "")) or "").strip()
        for p in products
        if p.id is not None
    }
    prov_by_pid = {
        int(p.id): (_s(getattr(p, "provider_name", "")) or "(Sin proveedor)").strip()
        for p in products
        if p.id is not None
    }

    def _qty_by_product(df: pd.DataFrame) -> dict[int, float]:
        df = _sanitize_editor_df(df)
        df2 = df.loc[df["delete"] != True].copy()  # noqa: E712
        df2 = df2.loc[df2["product_id"].notna()].copy()
        out: dict[int, float] = {}
        for _, r in df2.iterrows():
            pid = _pid_to_int(r.get("product_id"))
            if pid is None:
                continue
            out[int(pid)] = float(out.get(int(pid), 0.0) or 0.0) + float(r.get("quantity") or 0.0)
        return out

    def _add_product_to_df(pid: int, qty_val: float) -> None:
        qty_val = float(qty_val or 0.0)
        if qty_val <= 0:
            return
        df = _sanitize_editor_df(st.session_state[df_state_key])
        mask_same = (df["product_id"] == int(pid)) & (df["delete"] != True)  # noqa: E712
        if mask_same.any():
            idx = df.index[mask_same][0]
            df.at[idx, "quantity"] = float(df.at[idx, "quantity"] or 0.0) + qty_val
        else:
            unit = (_s(getattr(products_by_id.get(int(pid)), "unit", "")) or "unidad").lower()
            df = pd.concat(
                [df, pd.DataFrame([{"line_id": pd.NA, "product_id": int(pid), "quantity": qty_val, "unit": unit, "delete": False}])],
                ignore_index=True,
            )
        st.session_state[df_state_key] = _sanitize_editor_df(df)
        st.session_state[f"{editor_key}__qa_reset_qty"] = True
        st.rerun()

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
        f1, f2, f3, f4 = st.columns([2.6, 1.6, 1.6, 1.2], vertical_alignment="center")

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
        start_i = (st.session_state[page_key] - 1) * page_size
        end_i = start_i + page_size
        pids_page = pids[start_i:end_i]

        with st.container(height=400):
            cols = st.columns(3, gap="small")
            for i, pid in enumerate(pids_page):
                col = cols[i % 3]
                p = products_by_id.get(pid)
                unit_txt = (_s(getattr(p, "unit", "")) or "unidad").lower()

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

    st.caption("Tip: marca 🗑️ para eliminar una línea.")
    df_for_editor = _sanitize_editor_df(st.session_state[df_state_key])
    edited = st.data_editor(
        df_for_editor,
        hide_index=True,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "line_id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
            "product_id": st.column_config.SelectboxColumn("Producto", options=sorted(label_by_id.keys()), format_func=lambda pid: label_by_id.get(_pid_to_int(pid) or -1, str(pid)), required=True, width="large"),
            "quantity": st.column_config.NumberColumn("Qty", min_value=0.0, step=0.5, width="small"),
            "unit": st.column_config.TextColumn("Unidad", disabled=True, width="small"),
            "delete": st.column_config.CheckboxColumn("🗑️", width="small"),
        },
        key=editor_key,
    )
    edited = _sanitize_editor_df(edited)
    def _unit_for_pid(pid: Any) -> str:
        pid_i = _pid_to_int(pid)
        p = products_by_id.get(pid_i) if pid_i is not None else None
        return (_s(getattr(p, "unit", "")) or "unidad").lower() if p else "unidad"
    edited["unit"] = edited["product_id"].map(_unit_for_pid)
    st.session_state[df_state_key] = _sanitize_editor_df(edited)

    c1, c2 = st.columns([1.2, 1.0], vertical_alignment="center")
    with c1:
        if st.button("💾 Guardar", type="primary", use_container_width=True):
            _save_lines_from_editor(venue_id=venue_id, order_id=int(order.id), actor=actor, df=st.session_state[df_state_key], products_by_id=products_by_id)
            st.session_state.pop(df_state_key, None); st.session_state.pop(editor_key, None)
            _bump_refresh(venue_id); st.success("Guardado ✓"); st.rerun()
    with c2:
        if st.button("↩️ Descartar cambios", use_container_width=True):
            st.session_state.pop(df_state_key, None); st.session_state.pop(editor_key, None); st.rerun()


def _render_send_section(*, venue_id: int, order: Order, products: list[Product], lines: list[OrderLine], actor: str) -> None:
    
    v = _load_venue_templates(venue_id, _refresh_token(venue_id))
    missing_required = _venue_missing_required(v)
    if missing_required:
        st.error("Faltan campos obligatorios del local para enviar:\n" + "\n".join([f"- {x}" for x in missing_required]))
        return

    products_by_id = {int(p.id): p for p in products if p.id is not None}
    grouped = _group_lines_by_provider(lines, products_by_id)
    if not grouped:
        st.info("No hay líneas para enviar.")
        return
    st.markdown("### Summary")

    cA, cB, cC, cD = st.columns([1.15, 1.1, 1.2, 1.6], vertical_alignment="center")
    with cA:
        show_prices = st.toggle(
            "Mostrar importes",
            value=False,
            key=f"sum_show_prices_{int(order.id)}",
            help="Estimación basada en precios del catálogo y reglas de descuento. No es una factura.",
        )
    with cB:
        include_iva = st.toggle(
            "IVA",
            value=False,
            key=f"sum_include_iva_{int(order.id)}",
            help="Muestra % IVA, IVA € y Total (estimación).",
            disabled=not show_prices,
        )
    with cC:
        compact = st.toggle(
            "Compacto",
            value=True,
            key=f"sum_compact_{int(order.id)}",
            help="Mejor en móvil: tarjetas.",
        )
    with cD:
        sum_mode = st.radio(
            "Resumen",
            options=["por_proveedor", "total"],
            format_func=lambda x: "Por proveedor" if x == "por_proveedor" else "Total",
            horizontal=True,
            label_visibility="collapsed",
            key=f"sum_mode_{int(order.id)}",
        )

    # ---- pricing helpers (optional) ----
    from datetime import timedelta
    from sqlalchemy import func

    providers_by_name = _providers_cached(get_session, venue_id)
    _rules_by_provider_norm: dict[str, list[ProviderDiscountRule]] = {}

    @st.cache_data(ttl=60, show_spinner=False, hash_funcs={type(lambda: None): lambda _: "session_fn"})
    def _provider_rules_cached(_get_session_fn, venue_id: int, provider_id: int) -> list[ProviderDiscountRule]:
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
                        ProviderDiscountRule.product_id.is_(None).asc(),  # product first
                        ProviderDiscountRule.rule_kind.asc(),
                        ProviderDiscountRule.min_qty.desc(),
                    )
                ).all()
            )

    def _rules_for_provider(provider_name: str) -> list[ProviderDiscountRule]:
        pnorm = norm_provider(provider_name)
        if pnorm in _rules_by_provider_norm:
            return _rules_by_provider_norm[pnorm]
        prow = providers_by_name.get(pnorm)
        if not prow or getattr(prow, "id", None) is None:
            _rules_by_provider_norm[pnorm] = []
            return []
        rules = _provider_rules_cached(get_session, venue_id, int(prow.id))
        _rules_by_provider_norm[pnorm] = rules
        return rules

    def _prev_month_window(now: datetime) -> tuple[datetime, datetime]:
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
        if getattr(r, "product_id", None) is None:
            return True
        if pid is None:
            return False
        return int(r.product_id) == int(pid)

    def _rule_priority_key(r: ProviderDiscountRule) -> tuple:
        is_product = 1 if getattr(r, "product_id", None) is not None else 0
        rk = (getattr(r, "rule_kind", "") or "line_pct").strip()
        is_prev = 1 if rk.startswith("prev_month") else 0
        threshold = float(getattr(r, "prev_month_min_qty", 0.0) or 0.0) if is_prev else float(getattr(r, "min_qty", 0.0) or 0.0)
        strength = float(getattr(r, "discount_percent", 0.0) or 0.0)
        return (is_product, is_prev, threshold, strength)

    def _pricing_for_line(provider_name: str, pid: Optional[int], qty: float, gross_unit: float) -> dict[str, Any]:
        if qty <= 0 or gross_unit <= 0:
            return {"net_unit": gross_unit, "discount_pct": 0.0, "rule_kind": "", "rule_id": None, "applied": False}
        prov_norm = norm_provider(provider_name)
        rules = _rules_for_provider(prov_norm)
        if not rules:
            return {"net_unit": gross_unit, "discount_pct": 0.0, "rule_kind": "", "rule_id": None, "applied": False}

        candidates: list[ProviderDiscountRule] = []
        for r in rules:
            rk = (getattr(r, "rule_kind", "") or "line_pct").strip()
            if not _match_scope_ok(r, pid):
                continue
            if rk.startswith("prev_month"):
                th = float(getattr(r, "prev_month_min_qty", 0.0) or 0.0)
                if th <= 0:
                    continue
                qty_last_month = _prev_month_qty_cached(
                    get_session,
                    venue_id=venue_id,
                    provider_norm=prov_norm,
                    product_id=(int(pid) if (pid is not None and getattr(r, "product_id", None) is not None) else None),
                )
                if qty_last_month >= th:
                    candidates.append(r)
            else:
                th = float(getattr(r, "min_qty", 0.0) or 0.0)
                if qty >= th:
                    candidates.append(r)

        if not candidates:
            return {"net_unit": gross_unit, "discount_pct": 0.0, "rule_kind": "", "rule_id": None, "applied": False}

        candidates.sort(key=_rule_priority_key, reverse=True)
        chosen = candidates[0]
        rk = (getattr(chosen, "rule_kind", "") or "line_pct").strip()

        if rk.endswith("_net_price"):
            net_unit = float(getattr(chosen, "price_override", 0.0) or 0.0)
            if net_unit <= 0:
                return {"net_unit": gross_unit, "discount_pct": 0.0, "rule_kind": "", "rule_id": None, "applied": False}
            disc_pct = (1.0 - (net_unit / gross_unit)) * 100.0 if gross_unit > 0 else 0.0
            disc_pct = max(0.0, min(100.0, disc_pct))
            return {"net_unit": net_unit, "discount_pct": float(disc_pct), "rule_kind": rk, "rule_id": int(chosen.id) if getattr(chosen, "id", None) is not None else None, "applied": True}

        disc = float(getattr(chosen, "discount_percent", 0.0) or 0.0)
        disc = max(0.0, min(100.0, disc))
        net_unit = gross_unit * (1.0 - disc / 100.0)
        return {"net_unit": float(net_unit), "discount_pct": float(disc), "rule_kind": rk, "rule_id": int(chosen.id) if getattr(chosen, "id", None) is not None else None, "applied": disc > 0.0}

    def _price_for_pid(pid: Optional[int]) -> float:
        if pid is None:
            return 0.0
        p = products_by_id.get(pid)
        return _safe_float(getattr(p, "price", 0.0) if p else 0.0, 0.0)

    def _iva_pct_for_pid(pid: Optional[int], default_pct: float = 21.0) -> float:
        if pid is None:
            return float(default_pct)
        p = products_by_id.get(pid)
        if not p:
            return float(default_pct)
        v = _safe_float(getattr(p, "iva", None), float(default_pct))
        return float(default_pct) if v <= 0 else float(v)
    # --- label helpers: show description without repeating name words ---
    def _strip_accents(s: str) -> str:
        s = unicodedata.normalize("NFD", s or "")
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
        return unicodedata.normalize("NFC", s)

    def _norm_words(s: str) -> list[str]:
        raw = re.findall(r"[0-9]+|[^\W_]+", (s or ""), flags=re.UNICODE)
        return [_strip_accents(w).lower() for w in raw if w.strip()]

    def _remove_name_words_from_description(name: str, desc: str) -> str:
        name_set = set(_norm_words(name))
        if not name_set:
            return (desc or "").strip()
        desc_raw = re.findall(r"[0-9]+|[^\W_]+", (desc or ""), flags=re.UNICODE)
        kept: list[str] = []
        for w in desc_raw:
            if _strip_accents(w).lower() not in name_set:
                kept.append(w)
        return " ".join(kept).strip()

    def _product_label(pid: Optional[int], fallback_name: str = "Producto") -> str:
        if pid is None:
            return _s(fallback_name) or "Producto"
        p = products_by_id.get(pid)
        if not p:
            return _s(fallback_name) or "Producto"
        name = _s(getattr(p, "name", "")) or "Producto"
        desc = _s(getattr(p, "description", "")) or ""
        desc_clean = _remove_name_words_from_description(name, desc)
        return f"{name} — {desc_clean}" if desc_clean else name


    def _summary_rows(provider_name: str, prov_lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for it in prov_lines:
            qty = _safe_float(it.get("qty"), 0.0)
            if qty <= 0:
                continue
            pid = _pid_to_int(it.get("product_id"))
            prod = products_by_id.get(pid) if pid else None
            unit = (_s(getattr(prod, "unit", "")) if prod else _s(it.get("unit", ""))) or "unidad"

            label = _product_label(pid, _s(it.get("name") or "Producto"))

            row: dict[str, Any] = {
                "Proveedor": provider_name,
                "Label": label,
                "Qty": qty,
                "Unidad": unit,
            }

            if show_prices:
                gross_unit = _price_for_pid(pid)
                pricing = (
                    _pricing_for_line(provider_name, pid, qty, gross_unit)
                    if gross_unit > 0
                    else {"net_unit": 0.0, "discount_pct": 0.0, "rule_kind": "", "applied": False}
                )
                net_unit = float(pricing.get("net_unit", gross_unit) or 0.0)
                disc_pct = float(pricing.get("discount_pct", 0.0) or 0.0)

                amount = qty * net_unit
                ahorro = qty * max(0.0, (gross_unit - net_unit))
                row.update({
                    "Precio": net_unit,
                    "Desc.%": disc_pct if disc_pct > 0 else 0.0,
                    "Importe": amount,
                    "Ahorro": ahorro,
                })

                if include_iva:
                    iva_pct = _iva_pct_for_pid(pid, 21.0)
                    iva_eur = amount * (iva_pct / 100.0)
                    row.update({
                        "% IVA": iva_pct,
                        "IVA (€)": iva_eur,
                        "Total": amount + iva_eur,
                    })

            rows.append(row)
        return rows

    def _render_rows(rows: list[dict[str, Any]]) -> None:
        if not rows:
            st.info("No hay líneas para resumir.")
            return

        if compact:
            for r in rows:
                label = _s(r.get("Label"))
                qty_txt = f"{float(r.get('Qty') or 0.0):g} {_s(r.get('Unidad'))}"

                top = (
                    "<div style=\"display:flex;gap:10px;justify-content:space-between;align-items:baseline;flex-wrap:wrap\">"
                    f"<div style=\"font-weight:850;flex:1;min-width:220px\">{label}</div>"
                    f"<div style=\"font-weight:850;white-space:nowrap\">{qty_txt}</div>"
                    "</div>"
                )

                chips = ""
                if show_prices:
                    chips_items = [
                        ("Precio", float(r.get("Precio") or 0.0)),
                        ("Desc%", float(r.get("Desc.%") or 0.0)),
                        ("Importe", float(r.get("Importe") or 0.0)),
                        ("Ahorro", float(r.get("Ahorro") or 0.0)),
                    ]
                    if include_iva:
                        chips_items += [
                            ("IVA%", float(r.get("% IVA") or 0.0)),
                            ("IVA€", float(r.get("IVA (€)") or 0.0)),
                            ("Tot", float(r.get("Total") or 0.0)),
                        ]

                    def _chip(k: str, v: float) -> str:
                        if k in ("Desc%", "IVA%"):
                            vv = f"{v:g}%"
                        else:
                            vv = f"{v:,.2f}"

                        cls = "voi-pill"
                        if k == "Ahorro":
                            cls += " voi-pill--save"
                        elif k in ("IVA%", "IVA€"):
                            cls += " voi-pill--vat"
                        elif k == "Tot":
                            cls += " voi-pill--total"

                        return f"<span class='{cls}'>{k}: <b>{vv}</b></span>"

                    chips = "<div class='voi-chiprow'>" + "".join(_chip(k, v) for k, v in chips_items) + "</div>"

                st.markdown(
                    "<div class=\"voi-card\" style=\"padding:10px 12px\">"
                    f"<div class=\"voi-muted\">Proveedor: <b>{_s(r.get('Proveedor'))}</b></div>"
                    f"{top}"
                    f"{chips}"
                    "</div>",
                    unsafe_allow_html=True,
                )
        else:
            df = pd.DataFrame(rows)
            base_cols = ["Proveedor", "Label", "Qty", "Unidad"]
            if show_prices:
                base_cols += ["Precio", "Desc.%", "Importe", "Ahorro"]
                if include_iva:
                    base_cols += ["% IVA", "IVA (€)", "Total"]
            cols = [c for c in base_cols if c in df.columns] + [c for c in df.columns if c not in base_cols]
            df = df[cols]
            st.dataframe(df, use_container_width=True, hide_index=True)

    def _totals(rows: list[dict[str, Any]]) -> tuple[float, float, float, float]:
        subtotal = sum(float(r.get("Importe") or 0.0) for r in rows)
        ahorro = sum(float(r.get("Ahorro") or 0.0) for r in rows)
        iva_eur = sum(float(r.get("IVA (€)") or 0.0) for r in rows) if include_iva else 0.0
        total = sum(float(r.get("Total") or 0.0) for r in rows) if include_iva else subtotal
        return subtotal, ahorro, iva_eur, total

    if sum_mode == "total":
        rows_all: list[dict[str, Any]] = []
        for prov, prov_lines in grouped.items():
            rows_all.extend(_summary_rows(prov, prov_lines))
        _render_rows(rows_all)

        if show_prices and rows_all:
            subtotal, ahorro, iva_eur, total = _totals(rows_all)
            st.markdown("### Totales")
            if include_iva:
                st.markdown(f"**Subtotal (neto):** {subtotal:,.2f} · **Ahorro:** {ahorro:,.2f} · **IVA:** {iva_eur:,.2f} · **Total:** {total:,.2f}")
            else:
                st.markdown(f"**Total estimado (neto):** {subtotal:,.2f} · **Ahorro:** {ahorro:,.2f}")
            st.caption("Estimación: catálogo + reglas. La factura oficial del proveedor manda.")
        else:
            st.caption(f"{len(rows_all)} líneas en total.")

    else:
        for prov, prov_lines in grouped.items():
            rows = _summary_rows(prov, prov_lines)
            if not rows:
                continue
            with st.expander(f"{prov} · {len(rows)} líneas", expanded=False):
                _render_rows(rows)
                if show_prices:
                    subtotal, ahorro, iva_eur, total = _totals(rows)
                    if include_iva:
                        st.markdown(f"**Subtotal:** {subtotal:,.2f} · **Ahorro:** {ahorro:,.2f} · **IVA:** {iva_eur:,.2f} · **Total:** {total:,.2f}")
                    else:
                        st.markdown(f"**Subtotal:** {subtotal:,.2f} · **Ahorro:** {ahorro:,.2f}")


    st.markdown("<div class='voi-divider'></div>", unsafe_allow_html=True)

    provider_dir = _providers_cached(get_session, venue_id)
    send_map = _get_send_status_map(order_id=int(order.id))

    t1, t2, t3 = st.columns([1.0, 1.0, 1.2], vertical_alignment="center")
    with t1:
        use_email = st.toggle("Email", value=True)
    with t2:
        use_wa = st.toggle("WhatsApp", value=False)
    with t3:
        wa_cc = st.text_input("Prefijo país", value="+34")

    st.caption(f"CC: {v.email_cc or '—'} · BCC: {v.email_bcc or '—'}")

    sent_count = sum(1 for prov in grouped.keys() if bool(send_map.get(norm_provider(prov)) and getattr(send_map[norm_provider(prov)], "sent", False)))
    st.progress(sent_count / max(1, len(grouped)))
    st.caption(f"Enviados: {sent_count}/{len(grouped)}")

    c_all1, c_all2 = st.columns([1.5, 1.0], vertical_alignment="center")
    with c_all1:
        if st.button("🚀 Enviar a todos (SMTP)", type="primary", use_container_width=True, disabled=(not use_email)):
            ok, fail = 0, 0
            for prov, prov_lines in grouped.items():
                prov_norm = norm_provider(prov)
                p = provider_dir.get(prov_norm)
                to_email = _split_first_pipe(_s(getattr(p, "order_email", None) or getattr(p, "email", None) or getattr(p, "emails", None))) if p else ""
                if not to_email:
                    _touch_send_status(venue_id=venue_id, order_id=int(order.id), provider_name=prov_norm, actor=actor, channel="email", ok=False, error="missing provider email")
                    fail += 1
                    continue
                _ensure_workflow_order_sent(venue_id=venue_id, order_id=int(order.id), provider_name=prov_norm, actor=actor)
                subject = _build_subject(v, int(order.id))
                body_text = _build_supplier_message_text(templates=v, order_id=int(order.id), provider_name=prov_norm, prov_lines=prov_lines, products_by_id=products_by_id)
                try:
                    send_smtp_email(to=_split_emails(to_email), cc=_split_emails(v.email_cc), bcc=_split_emails(v.email_bcc), subject=subject, text_body=body_text)
                    _touch_send_status(venue_id=venue_id, order_id=int(order.id), provider_name=prov_norm, actor=actor, channel="email", ok=True)
                    ok += 1
                except Exception as e:
                    _touch_send_status(venue_id=venue_id, order_id=int(order.id), provider_name=prov_norm, actor=actor, channel="email", ok=False, error=str(e))
                    fail += 1
            if _all_providers_sent(order_id=int(order.id), provider_names=list(grouped.keys())):
                _set_order_status(int(order.id), "pending_receive", actor); _bump_refresh(venue_id)
                st.success(f"✅ Enviados {ok} · ❌ Fallos {fail} · Pedido → Pendiente"); st.rerun()
            else:
                st.success(f"✅ Enviados {ok} · ❌ Fallos {fail}"); st.rerun()

    with c_all2:
        if st.button("🧹 Reset enviados", use_container_width=True):
            _reset_send_status(order_id=int(order.id)); st.rerun()

    st.markdown("<div class='voi-divider'></div>", unsafe_allow_html=True)

    provider_dir = _providers_cached(get_session, venue_id)
    send_map = _get_send_status_map(order_id=int(order.id))

    # -----------------------------
    # Send settings (mobile-friendly)
    # -----------------------------
    with st.expander("⚙️ Envío: canales y CC", expanded=False):
        cset1, cset2 = st.columns([1.0, 1.0], vertical_alignment="center")
        with cset1:
            use_email = st.toggle("Email", value=True, key=f"use_email_{int(order.id)}")
        with cset2:
            use_wa = st.toggle("WhatsApp", value=False, key=f"use_wa_{int(order.id)}")

        wa_cc = st.text_input("Prefijo país (WhatsApp)", value="+34", key=f"wa_cc_{int(order.id)}")

        st.caption(f"CC: {v.email_cc or '—'} · BCC: {v.email_bcc or '—'}")

    # -----------------------------
    # Global actions
    # -----------------------------
    sent_count = sum(
        1
        for prov in grouped.keys()
        if bool(send_map.get(norm_provider(prov)) and getattr(send_map[norm_provider(prov)], "sent", False))
    )
    st.progress(sent_count / max(1, len(grouped)))
    st.caption(f"Enviados: {sent_count}/{len(grouped)}")

    g1, g2 = st.columns([1.6, 1.0], vertical_alignment="center")
    with g1:
        send_all_disabled = (not use_email)
        if st.button("🚀 Enviar a todos (Email)", type="primary", use_container_width=True, disabled=send_all_disabled):
            ok, fail = 0, 0
            for prov, prov_lines in grouped.items():
                prov_norm = norm_provider(prov)
                p = provider_dir.get(prov_norm)
                to_email = _split_first_pipe(_s(getattr(p, "order_email", None) or getattr(p, "email", None) or getattr(p, "emails", None))) if p else ""
                if not to_email:
                    _touch_send_status(
                        venue_id=venue_id,
                        order_id=int(order.id),
                        provider_name=prov_norm,
                        actor=actor,
                        channel="email",
                        ok=False,
                        error="missing provider email",
                    )
                    fail += 1
                    continue

                _ensure_workflow_order_sent(venue_id=venue_id, order_id=int(order.id), provider_name=prov_norm, actor=actor)
                subject = _build_subject(v, int(order.id))
                body_text = _build_supplier_message_text(
                    templates=v,
                    order_id=int(order.id),
                    provider_name=prov_norm,
                    prov_lines=prov_lines,
                    products_by_id=products_by_id,
                )

                try:
                    send_smtp_email(
                        to=_split_emails(to_email),
                        cc=_split_emails(v.email_cc),
                        bcc=_split_emails(v.email_bcc),
                        subject=subject,
                        text_body=body_text,
                    )
                    _touch_send_status(venue_id=venue_id, order_id=int(order.id), provider_name=prov_norm, actor=actor, channel="email", ok=True)
                    ok += 1
                except Exception as e:
                    _touch_send_status(venue_id=venue_id, order_id=int(order.id), provider_name=prov_norm, actor=actor, channel="email", ok=False, error=str(e))
                    fail += 1

            # move order to pending_receive only if every supplier has been sent at least once
            if _all_providers_sent(order_id=int(order.id), provider_names=list(grouped.keys())):
                _set_order_status(int(order.id), "pending_receive", actor)
                _bump_refresh(venue_id)
                st.success(f"✅ Enviados {ok} · ❌ Fallos {fail} · Pedido → Pendiente")
                st.rerun()
            else:
                st.success(f"✅ Enviados {ok} · ❌ Fallos {fail}")
                st.rerun()

    with g2:
        if st.button(
            "🧹 Reset enviados",
            use_container_width=True,
            key=f"reset_send_{int(order.id)}",
            ):
            _reset_send_status(order_id=int(order.id))
            st.rerun()

    st.markdown("<div class='voi-divider'></div>", unsafe_allow_html=True)

    # -----------------------------
    # Per-provider cards (mobile-first)
    # -----------------------------
    for prov, prov_lines in grouped.items():
        prov_norm = norm_provider(prov)
        p = provider_dir.get(prov_norm)

        to_email = _split_first_pipe(_s(getattr(p, "order_email", None) or getattr(p, "email", None) or getattr(p, "emails", None))) if p else ""
        phone = _split_first_pipe(_s(getattr(p, "order_phone", None) or getattr(p, "phone", None) or getattr(p, "phones", None))) if p else ""

        srow = send_map.get(prov_norm)
        sent = bool(getattr(srow, "sent", False)) if srow else False
        last_error = _s(getattr(srow, "last_error", "")) if srow else ""
        attempts = int(getattr(srow, "send_attempts", 0) or 0) if srow else 0
        chip_txt, chip_cls = _chip_for_send(sent, last_error)

        supplier_link = build_seguimiento_url(order_id=int(order.id), provider_name=prov_norm, role=ROLE_SUPPLIER, page_path="seguimiento")
        venue_link = build_seguimiento_url(order_id=int(order.id), provider_name=prov_norm, role=ROLE_VENUE, page_path="seguimiento")

        active_lines = len([x for x in prov_lines if _safe_float(x.get("qty"), 0) > 0])

        with st.container(border=True):
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;gap:10px;align-items:flex-start;'>"
                f"<div><div style='font-weight:900;font-size:1.05rem'>{prov_norm}</div>"
                f"<div class='voi-muted'>{active_lines} líneas · intentos: {attempts}</div></div>"
                f"<div class='{chip_cls}'>{chip_txt}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if last_error:
                st.caption(f"⚠️ {last_error}")

            # Primary actions
            b1, b2, b3 = st.columns([1.25, 1.05, 1.05], vertical_alignment="center")
            with b1:
                email_disabled = (not use_email) or (not to_email)
                email_label = "🔁 Reenviar email" if sent else "✅ Enviar email"
                if st.button(email_label, type="primary", use_container_width=True, disabled=email_disabled, key=f"send_email_{int(order.id)}_{prov_norm}"):
                    _ensure_workflow_order_sent(venue_id=venue_id, order_id=int(order.id), provider_name=prov_norm, actor=actor)
                    subject = _build_subject(v, int(order.id))
                    body_text = _build_supplier_message_text(
                        templates=v,
                        order_id=int(order.id),
                        provider_name=prov_norm,
                        prov_lines=prov_lines,
                        products_by_id=products_by_id,
                    )
                    try:
                        send_smtp_email(
                            to=_split_emails(to_email),
                            cc=_split_emails(v.email_cc),
                            bcc=_split_emails(v.email_bcc),
                            subject=subject,
                            text_body=body_text,
                        )
                        _touch_send_status(venue_id=venue_id, order_id=int(order.id), provider_name=prov_norm, actor=actor, channel="email", ok=True)
                        if _all_providers_sent(order_id=int(order.id), provider_names=list(grouped.keys())):
                            _set_order_status(int(order.id), "pending_receive", actor)
                            _bump_refresh(venue_id)
                        st.rerun()
                    except Exception as e:
                        _touch_send_status(venue_id=venue_id, order_id=int(order.id), provider_name=prov_norm, actor=actor, channel="email", ok=False, error=str(e))
                        st.error(f"No se pudo enviar: {e}")

            with b2:
                wa_disabled = (not use_wa) or (not phone)
                if wa_disabled:
                    st.button("📲 WhatsApp", use_container_width=True, disabled=True, key=f"wa_disabled_{int(order.id)}_{prov_norm}")
                else:
                    phone_norm = _normalize_phone(phone, wa_cc)
                    body_text = _build_supplier_message_text(
                        templates=v,
                        order_id=int(order.id),
                        provider_name=prov_norm,
                        prov_lines=prov_lines,
                        products_by_id=products_by_id,
                    )
                    if phone_norm:
                        wa = f"https://wa.me/{phone_norm.replace('+','')}?text={up.quote(body_text)}"
                        st.link_button("📲 WhatsApp", wa, use_container_width=True)
                    else:
                        st.button("📲 WhatsApp", use_container_width=True, disabled=True, key=f"wa_bad_{int(order.id)}_{prov_norm}")

            with b3:
                body_text = _build_supplier_message_text(
                    templates=v,
                    order_id=int(order.id),
                    provider_name=prov_norm,
                    prov_lines=prov_lines,
                    products_by_id=products_by_id,
                )
                st.download_button(
                    "📄 TXT",
                    data=body_text.encode("utf-8"),
                    file_name=f"pedido_{prov_norm.replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key=f"dl_{int(order.id)}_{prov_norm}",
                )

            with st.expander("Detalles", expanded=False):
                l1, l2 = st.columns([1.0, 1.0], vertical_alignment="center")
                with l1:
                    st.link_button("🔗 Link proveedor", supplier_link, use_container_width=True)
                with l2:
                    st.link_button("👀 Link local", venue_link, use_container_width=True)

                st.text_input("Link proveedor", value=supplier_link, disabled=True, label_visibility="collapsed", key=f"sl_{int(order.id)}_{prov_norm}")
                st.text_input("Link local", value=venue_link, disabled=True, label_visibility="collapsed", key=f"vl_{int(order.id)}_{prov_norm}")

                if use_wa and phone:
                    if st.button("✅ Marcar WA como enviado", use_container_width=True, key=f"mark_wa_{int(order.id)}_{prov_norm}"):
                        _touch_send_status(venue_id=venue_id, order_id=int(order.id), provider_name=prov_norm, actor=actor, channel="whatsapp", ok=True)
                        st.rerun()

                st.caption(f"Email proveedor: {to_email or '—'} · Tel: {phone or '—'}")

def _render_pending_section(*, venue_id: int, order: Order, products: list[Product], lines: list[OrderLine]) -> None:
    st.subheader("📦 Pendiente: recepción & seguimiento")
    products_by_id = {int(p.id): p for p in products if p.id is not None}
    grouped = _group_lines_by_provider(lines, products_by_id)
    if not grouped:
        st.info("No hay líneas.")
        return
    receipts = _provider_receipts_cached(get_session, int(order.id), _refresh_token(venue_id))
    send_map = _get_send_status_map(order_id=int(order.id))
    wf_state: dict[str, str] = {}
    if OrderWorkflow is not None:
        with get_session() as s:
            rows = s.exec(select(OrderWorkflow).where(OrderWorkflow.order_id == int(order.id), OrderWorkflow.venue_id == int(venue_id))).all()
        wf_state = {norm_provider(_s(r.provider_name)): _s(getattr(r, "state", "")) for r in rows}

    for prov in grouped.keys():
        prov_norm = norm_provider(prov)
        venue_link = build_seguimiento_url(order_id=int(order.id), provider_name=prov_norm, role=ROLE_VENUE, page_path="seguimiento")
        r = receipts.get(prov_norm)
        inv_no = _s(getattr(r, "invoice_number", "")) if r else ""
        decl = _s(getattr(r, "supplier_declaration", "")) if r else ""
        received = bool(getattr(r, "received", False)) if r else False
        st_row = send_map.get(prov_norm)
        sent = bool(getattr(st_row, "sent", False)) if st_row else False
        send_chip, send_cls = _chip_for_send(sent, _s(getattr(st_row, "last_error", "")) if st_row else "")
        wf = wf_state.get(prov_norm, "—")

        st.markdown(
            f"""
            <div class="voi-card">
              <div style="display:flex;justify-content:space-between;gap:10px;align-items:center;">
                <div>
                  <div style="font-weight:850">{prov_norm}</div>
                  <div class="voi-muted">Workflow: <b>{wf}</b> · Factura: <b>{inv_no or "—"}</b> · Proveedor: <b>{decl or "—"}</b> · Recibido: <b>{"sí" if received else "no"}</b></div>
                </div>
                <div class="{send_cls}">{send_chip}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns([1.2, 1.8], vertical_alignment="center")
        with c1:
            st.link_button("👀 Abrir seguimiento (local)", venue_link, use_container_width=True)
        with c2:
            st.code(venue_link, language=None)


def orders_tab(venue_id: int, venue_role: Optional[str]) -> None:
    _inject_css()
    actor = _s(st.session_state.get("user_email") or st.session_state.get("actor") or st.session_state.get("email") or current_actor())
    st.markdown("## 🧾 Pedidos")
    top1, top2 = st.columns([4.6, 1.4], vertical_alignment="center")
    with top1:
        status_filter = st.radio(
            "Estado",
            ["draft", "ready_to_send", "pending_receive", "incidences", "final"],
            horizontal=True,
            label_visibility="collapsed",
            format_func=lambda x: {"draft": "Borradores", "ready_to_send": "Listo", "pending_receive": "Pendiente", "incidences": "Incidencias", "final": "Historial"}.get(x, x),
            key=f"orders_status_{venue_id}",
        )
    with top2:
        if status_filter == "draft":
            if st.button("➕ Nuevo", type="primary", use_container_width=True):
                oid = _create_empty_draft(venue_id, actor)
                _bump_refresh(venue_id)
                st.session_state[f"orders_active_order_id_{venue_id}"] = int(oid)
                st.rerun()

    orders_all = _list_orders_cached(get_session, venue_id, _refresh_token(venue_id))

    # if status_filter == "incidences":
    #     _render_incidences_tab(venue_id=int(venue_id))
    #     return

    orders = [o for o in orders_all if _s(getattr(o, "status", "draft")).lower() == status_filter]
    if not orders:
        st.info("No hay pedidos para este filtro.")
        return

    active_key = f"orders_active_order_id_{venue_id}"
    ids = [int(o.id) for o in orders if o.id is not None]
    labels = {int(o.id): _order_label(o) for o in orders if o.id is not None}
    default_oid = int(st.session_state.get(active_key) or ids[0])
    if default_oid not in ids:
        default_oid = ids[0]
    picked = st.selectbox("Pedido", options=ids, index=ids.index(default_oid), format_func=lambda oid: labels.get(int(oid), str(oid)))
    st.session_state[active_key] = int(picked)

    with get_session() as s:
        order = s.exec(select(Order).where(Order.id == int(picked), Order.venue_id == int(venue_id))).first()
    if not order:
        st.error("Pedido no encontrado.")
        return

    products = _products_cached(get_session, venue_id)
    lines = _order_lines_cached(get_session, int(order.id), _refresh_token(venue_id))

    _render_header(order)
    _render_workflow_actions(venue_id=venue_id, order=order, role=venue_role, actor=actor)
    st.markdown("<div class='voi-divider'></div>", unsafe_allow_html=True)

    status = _s(order.status).lower()
    if status == "draft":
        _render_lines_editor(venue_id=venue_id, order=order, actor=actor, products=products, lines=lines)
    elif status == "ready_to_send":
        _render_send_section(venue_id=venue_id, order=order, products=products, lines=lines, actor=actor)
    # elif status == "pending_receive":
    
    elif status == "pending_receive":
        st.info("coming soon")

    else:
        st.info("Historial (solo lectura).")
