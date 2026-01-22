"""
orders.py — Modern pending + incidence follow-up panel

What you asked for
------------------
- A *single place* to see what is pending per supplier (and per product).
- No "local" link shown here (only supplier links).
- When supplier updates (OK / Parcial / Falta) those chips appear here.
- When the venue compares invoice vs received and detects "in invoice but not received",
  an *incidence ticket* is created with the real invoice number, supplier is notified,
  and accountant can see the expected resolution.

This file is designed to work with your existing tables:
- Order, OrderLine, Product
- OrderWorkflow / OrderWorkflowEvent
- ProviderSendStatus, Provider (directory)
- ProviderLineFollowUp (per-line supplier declaration)
- SeguimientoTicket (incidences)
- ProviderReceipt (invoice_number, received)

See domain models in domain/models.py. fileciteturn1file5
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Dict, List, Tuple
import os

import streamlit as st
from sqlmodel import select

from core.db import get_session
from core.public_links import build_seguimiento_url, ROLE_SUPPLIER, norm_provider
from core.mailer import send_smtp_email

from domain.models import (
    Order,
    OrderLine,
    Product,
    Provider,
    ProviderSendStatus,
    OrderWorkflow,
    SeguimientoTicket,
)

try:
    from domain.models import ProviderLineFollowUp  # type: ignore
except Exception:  # pragma: no cover
    ProviderLineFollowUp = None  # type: ignore

try:
    from domain.models import ProviderReceipt  # type: ignore
except Exception:  # pragma: no cover
    ProviderReceipt = None  # type: ignore


# =============================================================================
# Small UI system (chips + cards)
# =============================================================================

def _s(x: Any) -> str:
    return ("" if x is None else str(x)).strip()


def _now() -> datetime:
    return datetime.utcnow()


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .voi-card{border:1px solid rgba(49,51,63,.18); border-radius:16px; padding:12px 14px; background:rgba(255,255,255,.02);}
        .voi-muted{opacity:.65; font-size:.92rem;}
        .voi-row{display:flex; gap:10px; align-items:center; justify-content:space-between; flex-wrap:wrap;}
        .voi-chiprow{display:flex; gap:8px; flex-wrap:wrap; margin-top:8px;}
        .voi-chip{display:inline-flex; gap:6px; align-items:center; padding:6px 10px; border-radius:999px; border:1px solid rgba(49,51,63,.18); font-size:.88rem;}
        .voi-chip b{font-weight:800;}
        .voi-chip--ok{background:rgba(33,150,83,.10); border-color:rgba(33,150,83,.35);}
        .voi-chip--warn{background:rgba(255,193,7,.12); border-color:rgba(255,193,7,.35);}
        .voi-chip--bad{background:rgba(244,67,54,.10); border-color:rgba(244,67,54,.35);}
        .voi-chip--info{background:rgba(33,150,243,.08); border-color:rgba(33,150,243,.35);}
        .voi-divider{height:10px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _chip(text: str, kind: str = "info") -> str:
    cls = {"ok": "voi-chip voi-chip--ok", "warn": "voi-chip voi-chip--warn", "bad": "voi-chip voi-chip--bad", "info": "voi-chip voi-chip--info"}.get(kind, "voi-chip")
    return f"<span class='{cls}'>{text}</span>"


# =============================================================================
# Data fetch helpers
# =============================================================================

def _list_orders(venue_id: int, status: str) -> List[Order]:
    with get_session() as s:
        return list(
            s.exec(
                select(Order)
                .where(Order.venue_id == int(venue_id), Order.status == status)
                .order_by(Order.created_at.desc())
            ).all()
        )


def _order_lines(order_id: int) -> List[OrderLine]:
    with get_session() as s:
        return list(s.exec(select(OrderLine).where(OrderLine.order_id == int(order_id))).all())


def _products_map(venue_id: int) -> Dict[int, Product]:
    with get_session() as s:
        rows = list(s.exec(select(Product).where(Product.venue_id == int(venue_id))).all())
    return {int(p.id): p for p in rows if p.id is not None}


def _provider_dir(venue_id: int) -> Dict[str, Provider]:
    with get_session() as s:
        rows = list(s.exec(select(Provider).where(Provider.venue_id == int(venue_id))).all())
    return {norm_provider(_s(p.name)): p for p in rows}


def _send_status_map(venue_id: int, order_id: int) -> Dict[str, ProviderSendStatus]:
    with get_session() as s:
        rows = list(
            s.exec(
                select(ProviderSendStatus).where(
                    ProviderSendStatus.venue_id == int(venue_id),
                    ProviderSendStatus.order_id == int(order_id),
                )
            ).all()
        )
    return {norm_provider(_s(r.provider_name)): r for r in rows}


def _workflow_map(venue_id: int, order_id: int) -> Dict[str, str]:
    with get_session() as s:
        rows = list(
            s.exec(
                select(OrderWorkflow).where(
                    OrderWorkflow.venue_id == int(venue_id),
                    OrderWorkflow.order_id == int(order_id),
                )
            ).all()
        )
    return {norm_provider(_s(r.provider_name)): _s(r.state) or "—" for r in rows}


def _receipt_map(venue_id: int, order_id: int) -> Dict[str, Any]:
    if ProviderReceipt is None:
        return {}
    with get_session() as s:
        rows = list(
            s.exec(
                select(ProviderReceipt).where(
                    ProviderReceipt.venue_id == int(venue_id),
                    ProviderReceipt.order_id == int(order_id),
                )
            ).all()
        )
    return {norm_provider(_s(r.provider_name)): r for r in rows}


def _followup_map(venue_id: int, order_id: int) -> Dict[Tuple[str, int], Any]:
    """
    (provider_name, order_line_id) -> ProviderLineFollowUp row
    """
    if ProviderLineFollowUp is None:
        return {}
    with get_session() as s:
        rows = list(
            s.exec(
                select(ProviderLineFollowUp).where(
                    ProviderLineFollowUp.venue_id == int(venue_id),
                    ProviderLineFollowUp.order_id == int(order_id),
                )
            ).all()
        )
    out: Dict[Tuple[str, int], Any] = {}
    for r in rows:
        out[(norm_provider(_s(r.provider_name)), int(r.order_line_id))] = r
    return out


def _open_tickets(venue_id: int, only_open: bool = True) -> List[SeguimientoTicket]:
    with get_session() as s:
        q = select(SeguimientoTicket).where(SeguimientoTicket.venue_id == int(venue_id))
        if only_open:
            q = q.where(SeguimientoTicket.state == "open")
        return list(s.exec(q.order_by(SeguimientoTicket.created_at.desc())).all())


# =============================================================================
# Grouping logic
# =============================================================================

def _line_provider(ln: OrderLine, products: Dict[int, Product]) -> str:
    p = products.get(int(ln.product_id)) if getattr(ln, "product_id", None) is not None else None
    return norm_provider(_s(getattr(p, "provider_name", None) if p else getattr(ln, "provider", None)))


def _product_label(ln: OrderLine, products: Dict[int, Product]) -> str:
    p = products.get(int(ln.product_id)) if getattr(ln, "product_id", None) is not None else None
    return _s(getattr(p, "name", None) if p else getattr(ln, "spoken_name", None)) or "Producto"


def _unit(ln: OrderLine, products: Dict[int, Product]) -> str:
    p = products.get(int(ln.product_id)) if getattr(ln, "product_id", None) is not None else None
    return (_s(getattr(p, "unit", None)) if p else _s(getattr(ln, "unit", None)) or "unidad").lower()


def _qty(ln: OrderLine) -> float:
    try:
        return float(getattr(ln, "quantity", 0.0) or 0.0)
    except Exception:
        return 0.0


# =============================================================================
# Actions: reminder email (supplier link)
# =============================================================================

def _split_first_email(raw: str) -> str:
    raw = _s(raw).replace(",", "|")
    for p in raw.split("|"):
        p = p.strip()
        if p:
            return p
    return ""


def _provider_email(provider_dir: Dict[str, Provider], prov_name: str) -> str:
    p = provider_dir.get(norm_provider(prov_name))
    if not p:
        return ""
    return _split_first_email(_s(getattr(p, "order_email", None) or getattr(p, "emails", None) or ""))


def _send_supplier_reminder(*, to_email: str, order_id: int, provider_name: str, items: List[str], note: str = "") -> None:
    supplier_link = build_seguimiento_url(order_id=int(order_id), provider_name=norm_provider(provider_name), role=ROLE_SUPPLIER, page_path="seguimiento")
    subject = f"Pedido #{int(order_id)} · Confirmación de envío"
    body = (
        f"Hola {norm_provider(provider_name)},\n\n"
        f"Por favor confirma el estado del envío (OK/Parcial/Falta) en este enlace:\n{supplier_link}\n\n"
        f"Resumen:\n" + "\n".join(f"- {x}" for x in items) + "\n\n"
        f"Nota: {_s(note) or '—'}\n\n"
        f"Gracias."
    )
    send_smtp_email(to=[to_email], cc=[], bcc=[], subject=subject, text_body=body)


# =============================================================================
# Pending panel (the main thing you asked for)
# =============================================================================

def _render_pending_panel(*, venue_id: int, order: Order) -> None:
    products = _products_map(venue_id)
    provider_dir = _provider_dir(venue_id)
    lines = _order_lines(int(order.id))
    lines = [ln for ln in lines if _qty(ln) > 0]

    if not lines:
        st.info("Este pedido no tiene líneas.")
        return

    grouped: Dict[str, List[OrderLine]] = {}
    for ln in lines:
        prov = _line_provider(ln, products)
        grouped.setdefault(prov, []).append(ln)

    provs = sorted(grouped.keys())
    colf1, colf2 = st.columns([1.6, 1.4], vertical_alignment="center")
    with colf1:
        prov_filter = st.selectbox("Proveedor", options=["(Todos)"] + provs, index=0)
    with colf2:
        hide_ok = st.checkbox("Ocultar OK", value=False)

    send_map = _send_status_map(venue_id, int(order.id))
    wf_map = _workflow_map(venue_id, int(order.id))
    rec_map = _receipt_map(venue_id, int(order.id))
    fu_map = _followup_map(venue_id, int(order.id))

    # quick KPI row
    open_t = [t for t in _open_tickets(venue_id, only_open=True) if int(t.order_id) == int(order.id)]
    st.markdown(
        "<div class='voi-chiprow'>"
        + _chip(f"Pedido #{int(order.id)}", "info")
        + _chip(f"Proveedores: <b>{len(provs)}</b>", "info")
        + _chip(f"Incidencias abiertas: <b>{len(open_t)}</b>", "warn" if open_t else "ok")
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='voi-divider'></div>", unsafe_allow_html=True)

    for prov in provs:
        if prov_filter != "(Todos)" and prov != prov_filter:
            continue

        supplier_link = build_seguimiento_url(order_id=int(order.id), provider_name=prov, role=ROLE_SUPPLIER, page_path="seguimiento")

        wf_state = _s(wf_map.get(prov, "—"))
        wf_chip_kind = "info"
        if "CONFIRMED_FULL" in wf_state or wf_state in {"MATCHED_WITH_INVOICE", "CLOSED"}:
            wf_chip_kind = "ok"
        elif "CONFIRMED_PARTIAL" in wf_state or "WAITING" in wf_state:
            wf_chip_kind = "warn"
        elif "CONFIRMED_NONE" in wf_state or "DISCREPANCY" in wf_state:
            wf_chip_kind = "bad"

        receipt = rec_map.get(prov)
        inv_no = _s(getattr(receipt, "invoice_number", "")) if receipt else ""
        received = bool(getattr(receipt, "received", False)) if receipt else False

        # supplier declared status by line
        prov_lines = grouped.get(prov, [])
        rows: List[str] = []
        all_ok = True

        for ln in prov_lines:
            lid = int(getattr(ln, "id", 0) or 0)
            label = _product_label(ln, products)
            unit = _unit(ln, products)
            qty = _qty(ln)

            fu = fu_map.get((prov, lid))
            stt = _s(getattr(fu, "supplier_status", "")) if fu else ""
            sqty = getattr(fu, "supplier_qty", None) if fu else None

            # decide chip for line
            if stt == "missing":
                all_ok = False
                line_chip = _chip("🔴 Falta", "bad")
            elif stt == "partial":
                all_ok = False
                qtxt = f"{float(sqty or 0.0):g}/{qty:g}"
                line_chip = _chip(f"🟡 Parcial <b>{qtxt}</b>", "warn")
            elif stt == "ok":
                line_chip = _chip("✅ OK", "ok")
            else:
                all_ok = False
                line_chip = _chip("⏳ Sin respuesta", "info")

            if hide_ok and stt == "ok":
                continue

            rows.append(f"{line_chip} <b>{label}</b> · {qty:g} {unit}")

        if hide_ok and not rows:
            # everything was OK and hidden
            continue

        # incidence count for this provider
        prov_tickets = [t for t in open_t if norm_provider(_s(t.provider_name)) == prov]
        t_chip = _chip(f"Incidencias: <b>{len(prov_tickets)}</b>", "warn" if prov_tickets else "ok")

        # send status
        srow = send_map.get(prov)
        sent = bool(getattr(srow, "sent", False)) if srow else False
        send_chip = _chip("Enviado" if sent else "No enviado", "ok" if sent else "warn")

        receipt_chip = _chip(f"Factura: <b>{inv_no or '—'}</b>", "info") + _chip(f"Recibido: <b>{'sí' if received else 'no'}</b>", "info")

        st.markdown(
            f"""
            <div class="voi-card">
              <div class="voi-row">
                <div>
                  <div style="font-weight:900;font-size:1.05rem">{prov}</div>
                  <div class="voi-muted">Estado workflow: <b>{wf_state}</b></div>
                </div>
                <div class="voi-chiprow">
                  {send_chip}
                  {_chip(f"Workflow: <b>{wf_state}</b>", wf_chip_kind)}
                  {t_chip}
                </div>
              </div>
              <div class="voi-chiprow">{receipt_chip}</div>
              <div class="voi-divider"></div>
              <div class="voi-muted" style="margin-bottom:6px">Productos</div>
              <div class="voi-chiprow" style="gap:10px; align-items:flex-start; flex-direction:column;">
                {"".join(f"<div>{r}</div>" for r in rows) if rows else "<div class='voi-muted'>Nada para mostrar.</div>"}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Actions row
        a1, a2 = st.columns([1.15, 1.0], vertical_alignment="center")
        with a1:
            st.link_button("🔗 Abrir link proveedor", supplier_link, use_container_width=True)
        with a2:
            # optional reminder email
            to_email = _provider_email(provider_dir, prov)
            disabled = not bool(to_email)
            if st.button("📩 Recordatorio", use_container_width=True, disabled=disabled, key=f"rem_{int(order.id)}_{prov}"):
                items = [f"{_product_label(ln, products)} · {_qty(ln):g} {_unit(ln, products)}" for ln in prov_lines]
                _send_supplier_reminder(to_email=to_email, order_id=int(order.id), provider_name=prov, items=items)
                st.success(f"Enviado a {to_email}")

        st.markdown("<div class='voi-divider'></div>", unsafe_allow_html=True)


# =============================================================================
# Incidences dashboard (for accountant/overview)
# =============================================================================

def _render_incidences_tab(*, venue_id: int) -> None:
    st.markdown("## 🚨 Incidencias")
    st.caption("Todas las incidencias abiertas (facturado > recibido). El proveedor debe elegir: abono o entrega complementaria.")
    tickets = _open_tickets(venue_id, only_open=True)
    if not tickets:
        st.success("No hay incidencias abiertas 🎉")
        return

    # Filters
    provs = sorted({norm_provider(_s(t.provider_name)) for t in tickets})
    invs = sorted({(_s(t.invoice_number) or "—") for t in tickets})
    f1, f2 = st.columns([1.2, 1.2], vertical_alignment="center")
    with f1:
        prov_filter = st.selectbox("Proveedor", ["(Todos)"] + provs, index=0)
    with f2:
        inv_filter = st.selectbox("Factura", ["(Todas)"] + invs, index=0)

    for t in tickets:
        prov = norm_provider(_s(t.provider_name))
        inv = _s(t.invoice_number) or "—"
        if prov_filter != "(Todos)" and prov != prov_filter:
            continue
        if inv_filter != "(Todas)" and inv != inv_filter:
            continue

        supplier_link = build_seguimiento_url(order_id=int(t.order_id), provider_name=prov, role=ROLE_SUPPLIER, page_path="seguimiento")

        st.markdown(
            f"""
            <div class="voi-card">
              <div class="voi-row">
                <div>
                  <div style="font-weight:900">{prov}</div>
                  <div class="voi-muted">Pedido #{int(t.order_id)} · Factura: <b>{inv}</b></div>
                </div>
                <div class="voi-chiprow">
                  {_chip("Abierta", "warn")}
                  {_chip(f"Qty facturada: <b>{float(t.qty_invoiced or t.qty_expected or 0.0):g}</b>", "info")}
                  {_chip(f"Qty recibida: <b>{float(t.qty_received or 0.0):g}</b>", "info")}
                </div>
              </div>
              <div style="margin-top:10px"><b>{_s(t.product_name) or 'Producto'}</b> · {_s(t.unit) or 'unidad'}</div>
              <div class="voi-muted">{_s(t.note) or ''}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.link_button("🔗 Abrir link proveedor (resolver)", supplier_link, use_container_width=True)
        st.markdown("<div class='voi-divider'></div>", unsafe_allow_html=True)


# =============================================================================
# Main entry
# =============================================================================

def orders_tab(venue_id: int, venue_role: Optional[str]) -> None:
    _inject_css()

    tabs = st.tabs(["📦 Pedidos (pendiente)", "🚨 Incidencias"])
    with tabs[0]:
        st.markdown("## 📦 Panel pendiente")
        st.caption("Filtra por proveedor y revisa estado por producto (OK/Parcial/Falta). Solo se muestran links para proveedores.")

        orders = _list_orders(venue_id, "pending_receive")
        if not orders:
            st.info("No hay pedidos en estado pendiente (pending_receive).")
            return

        labels = {int(o.id): f"#{int(o.id)} · {_s(o.title) or 'Pedido'} · {o.created_at.strftime('%Y-%m-%d')}" for o in orders if o.id is not None}
        picked = st.selectbox("Pedido", options=[int(o.id) for o in orders if o.id is not None], format_func=lambda oid: labels.get(int(oid), str(oid)))

        with get_session() as s:
            order = s.exec(select(Order).where(Order.id == int(picked), Order.venue_id == int(venue_id))).first()
        if not order:
            st.error("Pedido no encontrado.")
            return

        _render_pending_panel(venue_id=venue_id, order=order)

    with tabs[1]:
        _render_incidences_tab(venue_id=venue_id)
