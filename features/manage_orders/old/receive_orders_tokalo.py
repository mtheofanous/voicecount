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
import urllib.parse

import streamlit as st
from sqlmodel import select

from core.db import get_session
from core.public_links import build_seguimiento_url, ROLE_SUPPLIER, ROLE_VENUE, norm_provider
try:
    from core.mailer import send_smtp_email  # type: ignore
except Exception:  # pragma: no cover
    send_smtp_email = None  # type: ignore

from domain.models import (
    Order,
    OrderLine,
    Product,
    Provider,
    ProviderSendStatus,
    OrderWorkflow,
    SeguimientoTicket,
    OrderWorkflowEvent 
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
    if send_smtp_email is None:
        return

    supplier_link = build_seguimiento_url(order_id=int(order_id), provider_name=norm_provider(provider_name), role=ROLE_SUPPLIER, page_path="seguimiento")
    venue_name = (os.getenv('VOICECOUNT_VENUE_NAME') or '').strip()
    if venue_name:
        supplier_link = supplier_link + '&venue_name=' + urllib.parse.quote(venue_name)
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
# Local verification + tickets + timeline (embedded in Pendiente panel)
# =============================================================================

def _load_workflow_row(*, venue_id: int, order_id: int, provider_name: str) -> Optional[OrderWorkflow]:
    with get_session() as s:
        return s.exec(
            select(OrderWorkflow).where(
                OrderWorkflow.venue_id == int(venue_id),
                OrderWorkflow.order_id == int(order_id),
                OrderWorkflow.provider_name == norm_provider(provider_name),
            )
        ).first()


def _transition(
    *,
    wf: OrderWorkflow,
    to_state: str,
    actor_role: str,
    actor: str,
    note: Optional[str] = None,
) -> OrderWorkflow:
    """Minimal transition helper (keeps OrderWorkflow + append-only OrderWorkflowEvent)."""
    from_state = _s(getattr(wf, 'state', '') or '—')
    to_state = _s(to_state)
    with get_session() as s:
        wf_db = s.exec(select(OrderWorkflow).where(OrderWorkflow.id == int(wf.id))).first()
        if not wf_db:
            return wf
        wf_db.state = to_state
        wf_db.updated_at = _now()
        wf_db.updated_by_role = _s(actor_role) or None
        wf_db.updated_by = _s(actor) or None
        wf_db.note = _s(note) or None
        s.add(wf_db)

        ev = OrderWorkflowEvent(
            venue_id=int(wf_db.venue_id),
            order_id=int(wf_db.order_id),
            provider_name=norm_provider(_s(wf_db.provider_name)),
            from_state=_s(from_state) or '—',
            to_state=to_state,
            actor_role=_s(actor_role) or 'system',
            actor=_s(actor) or None,
            at=_now(),
            note=_s(note) or None,
        )
        s.add(ev)
        s.commit()
        s.refresh(wf_db)
        return wf_db


def _upsert_receipt_header(
    *,
    venue_id: int,
    order_id: int,
    provider_name: str,
    received: bool,
    invoice_number: Optional[str] = None,
    note: Optional[str] = None,
) -> None:
    if ProviderReceipt is None:
        return
    prov = norm_provider(provider_name)
    with get_session() as s:
        row = s.exec(
            select(ProviderReceipt).where(
                ProviderReceipt.venue_id == int(venue_id),
                ProviderReceipt.order_id == int(order_id),
                ProviderReceipt.provider_name == prov,
            )
        ).first()
        if not row:
            row = ProviderReceipt(
                venue_id=int(venue_id),
                order_id=int(order_id),
                provider_name=prov,
                received=bool(received),
                invoice_number=_s(invoice_number) or None,
                note=_s(note) or None,
                updated_at=_now(),
                updated_by='venue',
            )
        else:
            row.received = bool(received)
            if invoice_number is not None:
                row.invoice_number = _s(invoice_number) or None
            if note is not None:
                row.note = _s(note) or None
            row.updated_at = _now()
            row.updated_by = 'venue'
        s.add(row)
        s.commit()



def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _upsert_followups_venue(
    *,
    venue_id: int,
    order_id: int,
    provider_name: str,
    per_line: Dict[int, Dict[str, Any]],
    comment: Optional[str] = None,
) -> None:
    """
    Upsert de los datos introducidos por el LOCAL (bar) por línea:
      - qty_received -> ProviderLineFollowUp.venue_qty
      - listed       -> ProviderLineFollowUp.invoice_listed
      - qty_invoiced -> ProviderLineFollowUp.qty_invoiced
      - comment      -> ProviderLineFollowUp.venue_comment (global, opcional)

    per_line esperado:
      {
        123: {"qty_received": 3, "listed": True, "qty_invoiced": 5},
        124: {"qty_received": 0, "listed": False, "qty_invoiced": None},
      }
    """
    if ProviderLineFollowUp is None:
        return

    prov = norm_provider(provider_name)
    now = _now()
    global_comment = _s(comment) or None

    with get_session() as s:
        for lid, payload in (per_line or {}).items():
            lid_int = int(lid)

            row = s.exec(
                select(ProviderLineFollowUp).where(
                    ProviderLineFollowUp.venue_id == int(venue_id),
                    ProviderLineFollowUp.order_id == int(order_id),
                    ProviderLineFollowUp.provider_name == prov,
                    ProviderLineFollowUp.order_line_id == lid_int,
                )
            ).first()

            if row is None:
                row = ProviderLineFollowUp(
                    venue_id=int(venue_id),
                    order_id=int(order_id),
                    provider_name=prov,
                    order_line_id=lid_int,
                )
                s.add(row)

            # --- LOCAL (bar) ---
            row.venue_qty = _to_float(payload.get("qty_received"), default=0.0)

            # Comentario global (si viene) o mantener el existente si no viene nada
            if global_comment is not None:
                row.venue_comment = global_comment

            # --- FACTURA (lo que aparece en la factura) ---
            # None => aún no revisado (si prefieres), pero aquí usamos bool cuando viene
            listed = payload.get("listed", None)
            row.invoice_listed = None if listed is None else bool(listed)

            # qty_invoiced: si no está listado o vacío -> None o 0 según tu preferencia
            qi = payload.get("qty_invoiced", None)
            if qi in (None, ""):
                row.qty_invoiced = None
            else:
                row.qty_invoiced = _to_float(qi, default=0.0)

            # --- AUDIT ---
            row.updated_at = now
            row.updated_by = "venue"

        s.commit()



def _ensure_ticket(
    *,
    venue_id: int,
    order_id: int,
    provider_name: str,
    order_line_id: int,
    kind: str,
    product_name: str,
    unit: str,
    qty_ordered: float,
    qty_expected: float,
    qty_received: float,
    invoice_number: Optional[str] = None,
    qty_invoiced: Optional[float] = None,
    note: Optional[str] = None,
) -> None:
    """Create (or keep) a single OPEN ticket per (order_line_id, kind)."""
    prov = norm_provider(provider_name)
    with get_session() as s:
        existing = s.exec(
            select(SeguimientoTicket).where(
                SeguimientoTicket.venue_id == int(venue_id),
                SeguimientoTicket.order_id == int(order_id),
                SeguimientoTicket.provider_name == prov,
                SeguimientoTicket.order_line_id == int(order_line_id),
                SeguimientoTicket.kind == _s(kind),
                SeguimientoTicket.state == 'open',
            )
        ).first()
        if existing:
            # refresh key fields (invoice number can arrive later)
            existing.updated_at = _now()
            if invoice_number is not None:
                existing.invoice_number = _s(invoice_number) or None
            if qty_invoiced is not None:
                existing.qty_invoiced = float(qty_invoiced)
            existing.qty_received = float(qty_received or 0.0)
            existing.note = _s(note) or existing.note
            s.add(existing)
            s.commit()
            return

        t = SeguimientoTicket(
            venue_id=int(venue_id),
            order_id=int(order_id),
            provider_name=prov,
            order_line_id=int(order_line_id),
            kind=_s(kind),
            state='open',
            product_name=_s(product_name) or 'Producto',
            unit=_s(unit) or 'unidad',
            qty_ordered=float(qty_ordered or 0.0),
            qty_expected=float(qty_expected or 0.0),
            qty_received=float(qty_received or 0.0),
            invoice_number=_s(invoice_number) or None,
            qty_invoiced=float(qty_invoiced) if qty_invoiced is not None else None,
            note=_s(note) or None,
            created_at=_now(),
            updated_at=_now(),
        )
        s.add(t)
        s.commit()


def _render_local_verification_panel(
    *,
    venue_id: int,
    order: Order,
    provider_name: str,
    wf_state: str,
    lines: List[OrderLine],
    products: Dict[int, Product],
    followup_map: Dict[Tuple[str, int], Any],
    receipt: Any,
) -> None:
    """Embedded local receiving + invoice matching + auto-incidences."""
    prov = norm_provider(provider_name)
    base = f"loc_{int(order.id)}_{prov}"

    default_received = bool(getattr(receipt, 'received', False)) if receipt else False
    default_inv = _s(getattr(receipt, 'invoice_number', '')) if receipt else ''

    c0, c1 = st.columns([1.0, 1.4], vertical_alignment='center')
    with c0:
        received = st.checkbox('Pedido recibido', value=default_received, key=f"{base}_received")
    with c1:
        invoice_number = st.text_input('Nº de factura', value=default_inv, placeholder='Ej.: F-2026-00123', key=f"{base}_inv")

    note = st.text_area('Nota (opcional)', value='', placeholder='Ej.: faltan cajas, producto equivocado…', key=f"{base}_note")

    st.markdown('#### Productos (pedido · proveedor · factura · recibido)')

    per_line: Dict[int, Dict[str, Any]] = {}
    any_not_listed = False
    any_discrepancy = False

    for ln in lines:
        lid = int(getattr(ln, 'id', 0) or 0)
        label = _product_label(ln, products)
        unit = _unit(ln, products)
        qty_ordered = float(_qty(ln))

        fu = followup_map.get((prov, lid))
        supplier_status = _s(getattr(fu, 'supplier_status', '')) if fu else ''
        supplier_qty = getattr(fu, 'supplier_qty', None) if fu else None
        venue_qty_existing = getattr(fu, 'venue_qty', None) if fu else None
        inv_listed_existing = getattr(fu, 'invoice_listed', None) if fu else None
        inv_qty_existing = getattr(fu, 'invoice_qty', None) if fu else None

        # sensible defaults
        listed_default = True if inv_listed_existing is None else bool(inv_listed_existing)
        inv_qty_default = float(inv_qty_existing) if inv_qty_existing is not None else float(qty_ordered)
        recv_qty_default = float(venue_qty_existing) if venue_qty_existing is not None else (float(qty_ordered) if received else 0.0)

        # supplier chip
        if supplier_status == 'missing':
            supp_txt = '🔴 Falta'
        elif supplier_status == 'partial':
            qtxt = f"{float(supplier_qty or 0.0):g}/{qty_ordered:g}"
            supp_txt = f"🟡 Parcial {qtxt}"
        elif supplier_status == 'ok':
            supp_txt = '✅ OK'
        else:
            supp_txt = '⏳ Sin respuesta'

        with st.container(border=True):
            st.markdown(f"**{label}** · pedido: {qty_ordered:g} {unit}")
            st.caption(f"Proveedor: {supp_txt}")

            a, b, c = st.columns([1.0, 1.2, 1.2], vertical_alignment='center')
            with a:
                listed = st.checkbox('En factura', value=listed_default, key=f"{base}_listed_{lid}")
            with b:
                qty_invoiced = st.number_input(
                    'Qty facturada',
                    min_value=0.0,
                    step=1.0,
                    value=float(inv_qty_default),
                    key=f"{base}_invqty_{lid}",
                    disabled=not listed,
                )
            with c:
                qty_received = st.number_input(
                    'Qty recibida',
                    min_value=0.0,
                    step=1.0,
                    value=float(recv_qty_default),
                    key=f"{base}_recvqty_{lid}",
                )

            if not listed:
                any_not_listed = True
            if listed and float(qty_invoiced) > float(qty_received) + 1e-9:
                any_discrepancy = True
                st.warning('Discrepancia: facturado > recibido', icon='⚠️')

        per_line[lid] = {
            'qty_ordered': qty_ordered,
            'listed': bool(listed),
            'qty_invoiced': float(qty_invoiced) if listed else 0.0,
            'qty_received': float(qty_received),
            'unit': unit,
            'label': label,
        }

    st.markdown('---')

    # CTA row
    b1, b2 = st.columns([1.2, 1.0], vertical_alignment='center')
    with b1:
        primary_label = '💾 Guardar recepción / factura'
        if st.button(primary_label, type='primary', use_container_width=True, key=f"{base}_save"):
            # Persist header + per-line venue quantities
            _upsert_receipt_header(
                venue_id=int(venue_id),
                order_id=int(order.id),
                provider_name=prov,
                received=bool(received),
                invoice_number=_s(invoice_number) or None,
                note=_s(note) or None,
            )
            _upsert_followups_venue(
                venue_id=int(venue_id),
                order_id=int(order.id),
                provider_name=prov,
                per_line=per_line,
                comment=_s(note) or None,
            )

            # Decide workflow transitions + tickets
            wf = _load_workflow_row(venue_id=int(venue_id), order_id=int(order.id), provider_name=prov)
            if wf and received and wf.state not in {'RECEIVED', 'MATCHED_WITH_INVOICE', 'INVOICE_DISCREPANCY', 'WAITING_SUPPLIER_ACTION', 'CLOSED'}:
                wf = _transition(wf=wf, to_state='RECEIVED', actor_role='receiving_employee', actor='venue', note=_s(note) or None)

            if wf and received:
                if any_not_listed:
                    # operational missing (manager decision)
                    for lid, v in per_line.items():
                        if not v.get('listed'):
                            _ensure_ticket(
                                venue_id=int(venue_id),
                                order_id=int(order.id),
                                provider_name=prov,
                                order_line_id=int(lid),
                                kind='operational_missing',
                                product_name=_s(v.get('label')),
                                unit=_s(v.get('unit')),
                                qty_ordered=float(v.get('qty_ordered') or 0.0),
                                qty_expected=float(v.get('qty_ordered') or 0.0),
                                qty_received=float(v.get('qty_received') or 0.0),
                                invoice_number=_s(invoice_number) or None,
                                note='No está en la factura (revisar operativa)',
                            )
                    wf = _transition(wf=wf, to_state='OPERATIONAL_MISSING_PRODUCT', actor_role='receiving_employee', actor='venue', note=_s(note) or None)
                    wf = _transition(wf=wf, to_state='WAITING_MANAGER_DECISION', actor_role='system', actor='system', note='Auto: líneas no listadas en factura')
                    st.warning('Guardado: hay líneas NO listadas en factura → decisión del manager.', icon='🧠')
                elif any_discrepancy:
                    for lid, v in per_line.items():
                        if v.get('listed') and float(v.get('qty_invoiced') or 0.0) > float(v.get('qty_received') or 0.0) + 1e-9:
                            _ensure_ticket(
                                venue_id=int(venue_id),
                                order_id=int(order.id),
                                provider_name=prov,
                                order_line_id=int(lid),
                                kind='invoice_discrepancy',
                                product_name=_s(v.get('label')),
                                unit=_s(v.get('unit')),
                                qty_ordered=float(v.get('qty_ordered') or 0.0),
                                qty_expected=float(v.get('qty_invoiced') or 0.0),
                                qty_received=float(v.get('qty_received') or 0.0),
                                invoice_number=_s(invoice_number) or None,
                                qty_invoiced=float(v.get('qty_invoiced') or 0.0),
                                note='Factura indica entrega, pero local no recibió todo.',
                            )
                    wf = _transition(wf=wf, to_state='INVOICE_DISCREPANCY', actor_role='receiving_employee', actor='venue', note=_s(note) or None)
                    wf = _transition(wf=wf, to_state='WAITING_SUPPLIER_ACTION', actor_role='system', actor='system', note='Auto: discrepancia con factura')
                    st.error('Guardado: incidencia abierta (facturado > recibido).', icon='🚨')
                else:
                    wf = _transition(wf=wf, to_state='MATCHED_WITH_INVOICE', actor_role='receiving_employee', actor='venue', note=_s(note) or None)
                    wf = _transition(wf=wf, to_state='CLOSED', actor_role='system', actor='system', note='Auto: todo coincide')
                    st.success('Guardado: todo coincide → cerrado.', icon='✅')
            else:
                st.success('Guardado.', icon='💾')

            st.rerun()

    with b2:
        st.caption('Reglas')
        st.write('• **Incidencia** si *facturado > recibido*')
        st.write('• **Operativa** si *NO está en factura*')


def _render_provider_timeline(*, venue_id: int, order_id: int, provider_name: str, limit: int = 8) -> None:
    prov = norm_provider(provider_name)
    with get_session() as s:
        evs = list(
            s.exec(
                select(OrderWorkflowEvent)
                .where(
                    OrderWorkflowEvent.venue_id == int(venue_id),
                    OrderWorkflowEvent.order_id == int(order_id),
                    OrderWorkflowEvent.provider_name == prov,
                )
                .order_by(OrderWorkflowEvent.at.desc())
                .limit(int(limit))
            ).all()
        )
    if not evs:
        st.caption('Sin eventos aún.')
        return
    for ev in evs:
        at = ev.at.strftime('%Y-%m-%d %H:%M')
        st.write(f"{at} · **{_s(ev.from_state)} → {_s(ev.to_state)}** · {_s(ev.actor_role)}")
        if ev.note:
            st.caption(_s(ev.note))
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
        venue_name = (os.getenv('VOICECOUNT_VENUE_NAME') or '').strip()
        if venue_name:
            supplier_link = supplier_link + '&venue_name=' + urllib.parse.quote(venue_name)

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


        # Actions row (NO local link — everything happens here)
        a1, a2 = st.columns([1.0, 1.0], vertical_alignment="center")
        with a1:
            st.link_button("🔗 Link proveedor", supplier_link, use_container_width=True)
        with a2:
            # optional reminder email
            to_email = _provider_email(provider_dir, prov)
            disabled = not bool(to_email)
            if st.button("📩 Recordatorio", use_container_width=True, disabled=disabled, key=f"rem_{int(order.id)}_{prov}"):
                items = [f"{_product_label(ln, products)} · {_qty(ln):g} {_unit(ln, products)}" for ln in prov_lines]
                _send_supplier_reminder(to_email=to_email, order_id=int(order.id), provider_name=prov, items=items)
                st.success(f"Enviado a {to_email}")

        # Local verification panel (inside the pending panel)
        with st.expander("📦 Recepción (local) · Factura · Incidencias", expanded=False):
            _render_local_verification_panel(
                venue_id=int(venue_id),
                order=order,
                provider_name=prov,
                wf_state=wf_state,
                lines=prov_lines,
                products=products,
                followup_map=fu_map,
                receipt=receipt,
            )

        # Mini timeline (last changes)
        with st.expander("🕒 Últimos cambios", expanded=False):
            _render_provider_timeline(
                venue_id=int(venue_id),
                order_id=int(order.id),
                provider_name=prov,
                limit=8,
            )

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