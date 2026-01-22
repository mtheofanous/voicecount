# features/seguimiento/seguimiento.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Tuple, List, Dict

import os
import streamlit as st
from sqlmodel import select

from core.db import get_session
from core.public_links import ROLE_SUPPLIER, ROLE_VENUE, norm_provider, norm_role, verify_link, build_seguimiento_url

# Email is only used for supplier/accountant notifications (no payments)
try:
    from core.mailer import send_smtp_email  # type: ignore
except Exception:  # pragma: no cover
    send_smtp_email = None  # type: ignore

from domain.models import (
    Order,
    OrderLine,
    Product,
    OrderWorkflow,
    OrderWorkflowEvent,
    SeguimientoTicket,
    ORDER_STATES,
)

# Optional (only if present in your models)
try:
    from domain.models import ProviderLineFollowUp  # type: ignore
except Exception:  # pragma: no cover
    ProviderLineFollowUp = None  # type: ignore

try:
    from domain.models import ProviderReceipt  # type: ignore
except Exception:  # pragma: no cover
    ProviderReceipt = None  # type: ignore

try:
    from domain.models import Provider  # type: ignore
except Exception:  # pragma: no cover
    Provider = None  # type: ignore


# =============================================================================
# UX labels (no new states — just human-friendly names)
# =============================================================================

STATE_LABELS = {
    "ORDER_SENT": "Pedido enviado",
    "SUPPLIER_CONFIRMED_FULL": "Proveedor confirmó envío completo",
    "SUPPLIER_CONFIRMED_PARTIAL": "Proveedor confirmó envío parcial",
    "SUPPLIER_CONFIRMED_NONE": "Proveedor indicó sin envío",
    "RECEIVED": "Pedido recibido",
    "MATCHED_WITH_INVOICE": "Coincide con factura",
    "INVOICE_DISCREPANCY": "Discrepancia con factura",
    "WAITING_SUPPLIER_ACTION": "Esperando acción del proveedor",
    "OPERATIONAL_MISSING_PRODUCT": "Falta operativa",
    "WAITING_MANAGER_DECISION": "Esperando decisión del manager",
    "SUPPLIER_CREDIT_NOTE_ISSUED": "Proveedor emitirá abono",
    "SUPPLEMENTARY_DELIVERY_SENT": "Proveedor enviará entrega complementaria",
    "DECISION_REORDER_SAME": "Decisión: reordenar (mismo proveedor)",
    "DECISION_SWITCH_SUPPLIER": "Decisión: cambiar proveedor",
    "DECISION_NOT_NEEDED": "Decisión: ya no se necesita",
    "CLOSED": "Cerrado",
}


# =============================================================================
# Helpers
# =============================================================================

def _now() -> datetime:
    return datetime.utcnow()


def _s(x: Any) -> str:
    return ("" if x is None else str(x)).strip()


def _f(x: Any) -> Optional[float]:
    """Accepts either "123", 123, ["123"] (Streamlit query params), returns float or None."""
    try:
        if isinstance(x, list) and x:
            x = x[0]
        s = _s(x)
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _line_qty(ln: OrderLine) -> float:
    try:
        return float(getattr(ln, "quantity", None) or 0.0)
    except Exception:
        return 0.0


def _product_label(p: Optional[Product], fallback: str = "Producto") -> str:
    if not p:
        return _s(fallback) or "Producto"
    name = _s(getattr(p, "name", "")) or "Producto"
    return name


def _load_order_bundle(order_id: int) -> Tuple[Order, list[OrderLine], dict[int, Product]]:
    with get_session() as s:
        order = s.exec(select(Order).where(Order.id == order_id)).first()
        if not order:
            raise ValueError("Order not found")

        lines = list(s.exec(select(OrderLine).where(OrderLine.order_id == order_id)).all())

        prod_ids = sorted({int(l.product_id) for l in lines if l.product_id is not None})
        products = list(s.exec(select(Product).where(Product.id.in_(prod_ids))).all()) if prod_ids else []
        prod_map = {int(p.id): p for p in products if p.id is not None}
        return order, lines, prod_map


def _line_provider_name(ln: OrderLine, prod_map: dict[int, Product]) -> str:
    pid = getattr(ln, "product_id", None)
    p = prod_map.get(int(pid)) if pid is not None else None
    # Prefer Product.provider_name; fallback to OrderLine.provider
    return norm_provider(getattr(p, "provider_name", None) if p else getattr(ln, "provider", None))


def _get_workflow(*, venue_id: int, order_id: int, provider_name: str) -> OrderWorkflow:
    """Seguimiento does NOT create workflows. ORDER_SENT must exist (created when order is sent)."""
    with get_session() as s:
        wf = s.exec(
            select(OrderWorkflow).where(
                OrderWorkflow.venue_id == venue_id,
                OrderWorkflow.order_id == order_id,
                OrderWorkflow.provider_name == provider_name,
            )
        ).first()

        if not wf:
            raise ValueError("Workflow not initialized. This order was not sent yet.")

        return wf


def _load_events(*, venue_id: int, order_id: int, provider_name: str) -> List[OrderWorkflowEvent]:
    with get_session() as s:
        return list(
            s.exec(
                select(OrderWorkflowEvent)
                .where(
                    OrderWorkflowEvent.venue_id == venue_id,
                    OrderWorkflowEvent.order_id == order_id,
                    OrderWorkflowEvent.provider_name == provider_name,
                )
                .order_by(OrderWorkflowEvent.at.asc())
            ).all()
        )


def _state_name(state: str) -> str:
    return STATE_LABELS.get(state, state)


def _accountant_emails() -> list[str]:
    """
    Optional accountant notification list.
    Configure in env/secrets:
      ACCOUNTANT_EMAILS="acc1@x.com | acc2@y.com"
    """
    raw = (os.getenv("ACCOUNTANT_EMAILS") or "").strip()
    if not raw and hasattr(st, "secrets"):
        try:
            raw = (st.secrets.get("ACCOUNTANT_EMAILS") or "").strip()
        except Exception:
            raw = ""
    raw = raw.replace(",", "|")
    return [p.strip() for p in raw.split("|") if p.strip()]


def _provider_email(*, venue_id: int, provider_name: str) -> Optional[str]:
    """
    Try to find a provider email from Provider directory (best) else from any Product row.
    """
    prov_n = norm_provider(provider_name)

    # Provider directory
    if Provider is not None:
        with get_session() as s:
            p = s.exec(select(Provider).where(Provider.venue_id == int(venue_id), Provider.name == prov_n)).first()
        if p:
            for candidate in [
                getattr(p, "order_email", None),
                getattr(p, "emails", None),
            ]:
                c = _s(candidate)
                if c:
                    first = c.replace(",", "|").split("|")[0].strip()
                    if first:
                        return first

    # Fallback: any product provider_email
    with get_session() as s:
        pr = s.exec(
            select(Product)
            .where(Product.venue_id == int(venue_id), Product.provider_name == prov_n)
            .limit(1)
        ).first()
    if pr:
        e = _s(getattr(pr, "provider_email", None))
        if e:
            return e.split("|")[0].strip() if "|" in e else e
    return None


def _upsert_provider_receipt(
    *,
    venue_id: int,
    order_id: int,
    provider_name: str,
    invoice_number: Optional[str] = None,
    received: Optional[bool] = None,
    note: Optional[str] = None,
    actor: Optional[str] = None,
) -> None:
    """
    Keep ProviderReceipt in sync so Orders can show invoice number + received flag.
    Safe no-op if ProviderReceipt model is not available.
    """
    if ProviderReceipt is None:
        return

    prov_n = norm_provider(provider_name)

    with get_session() as s:
        row = s.exec(
            select(ProviderReceipt).where(
                ProviderReceipt.venue_id == int(venue_id),
                ProviderReceipt.order_id == int(order_id),
                ProviderReceipt.provider_name == prov_n,
            )
        ).first()

        if not row:
            row = ProviderReceipt(
                venue_id=int(venue_id),
                order_id=int(order_id),
                provider_name=prov_n,
                created_at=_now(),
                updated_at=_now(),
                updated_by=_s(actor) or None,
            )

        if invoice_number is not None:
            inv = _s(invoice_number) or None
            row.invoice_number = inv
            row.invoice_number_set_at = _now()
            row.invoice_number_set_by = _s(actor) or None

        if received is not None:
            row.received = bool(received)
            row.received_at = _now() if bool(received) else None
            row.received_by = _s(actor) or None

        if note is not None:
            row.note = _s(note) or None

        row.updated_at = _now()
        row.updated_by = _s(actor) or None

        s.add(row)
        s.commit()


def _save_followups_bulk(
    *,
    venue_id: int,
    order_id: int,
    provider_name: str,
    per_line: Dict[int, Dict[str, Any]],
    actor: str,
) -> None:
    """
    Persist supplier per-line declaration (OK/partial/missing + qty + reason) for the dashboard.

    Safe no-op if ProviderLineFollowUp model is not available.
    """
    if ProviderLineFollowUp is None:
        return

    prov_n = norm_provider(provider_name)

    with get_session() as s:
        for lid, payload in (per_line or {}).items():
            lid_i = int(lid)
            row = s.exec(
                select(ProviderLineFollowUp).where(
                    ProviderLineFollowUp.venue_id == int(venue_id),
                    ProviderLineFollowUp.order_id == int(order_id),
                    ProviderLineFollowUp.provider_name == prov_n,
                    ProviderLineFollowUp.order_line_id == lid_i,
                )
            ).first()

            if not row:
                row = ProviderLineFollowUp(
                    venue_id=int(venue_id),
                    order_id=int(order_id),
                    provider_name=prov_n,
                    order_line_id=lid_i,
                    qty_ordered=float(payload.get("qty_ordered") or 0.0),
                    updated_at=_now(),
                    updated_by=_s(actor) or None,
                )

            row.qty_ordered = float(payload.get("qty_ordered") or row.qty_ordered or 0.0)
            row.supplier_status = _s(payload.get("supplier_status") or "ok") or "ok"
            row.supplier_qty = float(payload.get("supplier_qty") or 0.0) if row.supplier_status != "ok" else float(payload.get("qty_ordered") or row.qty_ordered or 0.0)
            row.supplier_reason = _s(payload.get("supplier_reason") or None) or None
            row.updated_at = _now()
            row.updated_by = _s(actor) or None

            s.add(row)

        s.commit()


def _email_notify_supplier_incidence(
    *,
    venue_id: int,
    order_id: int,
    provider_name: str,
    invoice_number: Optional[str],
    tickets: List[SeguimientoTicket],
    note: Optional[str],
) -> None:
    """
    Send an incidence email to the supplier with the signed supplier link.
    Safe no-op if email is not configured.
    """
    if send_smtp_email is None:
        return

    to_email = _provider_email(venue_id=venue_id, provider_name=provider_name)
    if not to_email:
        return

    supplier_link = build_seguimiento_url(
        order_id=int(order_id),
        provider_name=norm_provider(provider_name),
        role=ROLE_SUPPLIER,
        page_path="seguimiento",
    )

    lines_txt = []
    for t in tickets:
        q_inv = float(t.qty_invoiced or t.qty_expected or 0.0)
        lines_txt.append(f"- {t.product_name}: facturado {q_inv:g} {t.unit} · recibido {float(t.qty_received):g} {t.unit}")

    subject = f"[Incidencia] Pedido #{int(order_id)} · Factura {(_s(invoice_number) or '—')}"
    body = (
        f"Hola {norm_provider(provider_name)},\n\n"
        f"Se ha abierto una incidencia porque la factura indica cantidades entregadas que el local no ha recibido.\n\n"
        f"Pedido: #{int(order_id)}\n"
        f"Factura: {_s(invoice_number) or '—'}\n\n"
        f"Líneas afectadas:\n" + "\n".join(lines_txt) + "\n\n"
        f"Nota: {_s(note) or '—'}\n\n"
        f"Por favor, entra al enlace para indicar la solución (abono o entrega complementaria):\n{supplier_link}\n\n"
        f"Gracias."
    )

    try:
        send_smtp_email(to=[to_email], cc=[], bcc=[], subject=subject, text_body=body)
    except Exception:
        # do not block UX on email problems
        pass


def _email_notify_accountant(
    *,
    venue_id: int,
    order_id: int,
    provider_name: str,
    invoice_number: Optional[str],
    resolution: str,
    resolution_note: Optional[str],
    tickets: List[SeguimientoTicket],
) -> None:
    """
    Notify accountant(s) so they know invoice incidence + expected solution.
    Safe no-op if ACCOUNTANT_EMAILS is not set.
    """
    if send_smtp_email is None:
        return

    tos = _accountant_emails()
    if not tos:
        return

    lines_txt = []
    for t in tickets:
        q_inv = float(t.qty_invoiced or t.qty_expected or 0.0)
        lines_txt.append(f"- {t.product_name}: facturado {q_inv:g} {t.unit} · recibido {float(t.qty_received):g} {t.unit}")

    subject = f"[Incidencia resuelta] Pedido #{int(order_id)} · {norm_provider(provider_name)} · Factura {(_s(invoice_number) or '—')}"
    body = (
        f"Incidencia de factura actualizada.\n\n"
        f"Pedido: #{int(order_id)}\n"
        f"Proveedor: {norm_provider(provider_name)}\n"
        f"Factura: {_s(invoice_number) or '—'}\n"
        f"Solución esperada: {resolution}\n"
        f"Nota solución: {_s(resolution_note) or '—'}\n\n"
        f"Detalle líneas:\n" + "\n".join(lines_txt)
    )

    try:
        send_smtp_email(to=tos, cc=[], bcc=[], subject=subject, text_body=body)
    except Exception:
        pass


def _transition(
    *,
    wf: OrderWorkflow,
    to_state: str,
    actor_role: str,
    actor: Optional[str] = None,
    note: Optional[str] = None,
) -> OrderWorkflow:
    """Enforce transitions exactly as the flow definition."""
    if to_state not in ORDER_STATES:
        raise ValueError(f"Invalid state: {to_state}")

    from_state = wf.state

    allowed: dict[str, set[str]] = {
        "ORDER_SENT": {"SUPPLIER_CONFIRMED_FULL", "SUPPLIER_CONFIRMED_PARTIAL", "SUPPLIER_CONFIRMED_NONE"},
        "SUPPLIER_CONFIRMED_FULL": {"RECEIVED"},
        "SUPPLIER_CONFIRMED_PARTIAL": {"RECEIVED"},
        "SUPPLIER_CONFIRMED_NONE": {"RECEIVED"},
        "RECEIVED": {"MATCHED_WITH_INVOICE", "INVOICE_DISCREPANCY", "OPERATIONAL_MISSING_PRODUCT"},
        "MATCHED_WITH_INVOICE": {"CLOSED"},
        "INVOICE_DISCREPANCY": {"WAITING_SUPPLIER_ACTION"},
        "WAITING_SUPPLIER_ACTION": {"SUPPLIER_CREDIT_NOTE_ISSUED", "SUPPLEMENTARY_DELIVERY_SENT"},
        "SUPPLIER_CREDIT_NOTE_ISSUED": {"CLOSED"},
        "SUPPLEMENTARY_DELIVERY_SENT": {"CLOSED"},
        "OPERATIONAL_MISSING_PRODUCT": {"WAITING_MANAGER_DECISION"},
        "WAITING_MANAGER_DECISION": {"DECISION_REORDER_SAME", "DECISION_SWITCH_SUPPLIER", "DECISION_NOT_NEEDED"},
        "DECISION_REORDER_SAME": {"CLOSED"},
        "DECISION_SWITCH_SUPPLIER": {"CLOSED"},
        "DECISION_NOT_NEEDED": {"CLOSED"},
        "CLOSED": set(),
    }

    if from_state not in allowed or to_state not in allowed[from_state]:
        raise ValueError(f"Transition not allowed: {from_state} -> {to_state}")

    with get_session() as s:
        wf_db = s.exec(select(OrderWorkflow).where(OrderWorkflow.id == wf.id)).first()
        if not wf_db:
            raise ValueError("Workflow missing")

        ev = OrderWorkflowEvent(
            venue_id=wf_db.venue_id,
            order_id=wf_db.order_id,
            provider_name=wf_db.provider_name,
            from_state=wf_db.state,
            to_state=to_state,
            actor_role=actor_role,
            actor=actor,
            at=_now(),
            note=note,
        )
        s.add(ev)

        wf_db.state = to_state
        wf_db.updated_at = _now()
        wf_db.updated_by_role = actor_role
        wf_db.updated_by = actor
        wf_db.note = note

        s.add(wf_db)
        s.commit()
        s.refresh(wf_db)
        return wf_db


# =============================================================================
# Tickets (incidencias)
# =============================================================================

def _load_open_tickets(*, venue_id: int, order_id: int, provider_name: str) -> List[SeguimientoTicket]:
    with get_session() as s:
        return list(
            s.exec(
                select(SeguimientoTicket)
                .where(
                    SeguimientoTicket.venue_id == venue_id,
                    SeguimientoTicket.order_id == order_id,
                    SeguimientoTicket.provider_name == provider_name,
                    SeguimientoTicket.state == "open",
                )
                .order_by(SeguimientoTicket.created_at.asc())
            ).all()
        )


def _create_invoice_discrepancy_tickets(
    *,
    venue_id: int,
    order_id: int,
    provider_name: str,
    invoice_number: Optional[str],
    per_line: Dict[int, Dict[str, Any]],
    prod_map: Dict[int, Product],
    lines: List[OrderLine],
    note: Optional[str],
) -> int:
    """
    Creates one ticket per affected line when:
      qty_invoiced > qty_received (supplier invoiced as sent, venue did not receive all)
    per_line[lid] expects:
      qty_invoiced, qty_received, qty_ordered
    """
    line_by_id = {int(getattr(l, "id", 0) or 0): l for l in lines}
    created = 0

    with get_session() as s:
        for lid, payload in per_line.items():
            try:
                q_inv = float(payload.get("qty_invoiced") or 0.0)
                q_rec = float(payload.get("qty_received") or 0.0)
                q_ord = float(payload.get("qty_ordered") or 0.0)
            except Exception:
                continue

            if q_inv <= q_rec:
                continue

            ln = line_by_id.get(int(lid))
            if not ln:
                continue

            p = prod_map.get(int(ln.product_id)) if ln.product_id is not None else None
            unit = (_s(getattr(p, "unit", "")) if p else _s(getattr(ln, "unit", "")) or "unidad").lower()
            product_name = _product_label(p, fallback=getattr(ln, "spoken_name", None) or "Producto")

            t = SeguimientoTicket(
                venue_id=venue_id,
                order_id=order_id,
                provider_name=provider_name,
                order_line_id=int(lid),
                kind="invoice_claimed_not_received",
                state="open",
                product_name=product_name,
                unit=unit,
                qty_ordered=float(q_ord),
                qty_expected=float(q_inv),   # what invoice claims
                qty_received=float(q_rec),   # what venue received
                invoice_number=_s(invoice_number) or None,
                qty_invoiced=float(q_inv),
                note=_s(note) or None,
                created_at=_now(),
                updated_at=_now(),
            )
            s.add(t)
            created += 1

        s.commit()

    return created


def _resolve_open_tickets(
    *,
    venue_id: int,
    order_id: int,
    provider_name: str,
    resolution_kind: str,   # "credit_note" | "supplementary_delivery"
    resolution_note: Optional[str],
) -> List[SeguimientoTicket]:
    with get_session() as s:
        tickets = list(
            s.exec(
                select(SeguimientoTicket).where(
                    SeguimientoTicket.venue_id == venue_id,
                    SeguimientoTicket.order_id == order_id,
                    SeguimientoTicket.provider_name == provider_name,
                    SeguimientoTicket.state == "open",
                )
            ).all()
        )
        for t in tickets:
            t.state = "resolved_" + resolution_kind
            t.resolution_note = _s(resolution_note) or None
            t.updated_at = _now()
            t.resolved_at = _now()
            s.add(t)
        s.commit()
        return tickets


# =============================================================================
# UI helpers
# =============================================================================

def _header(*, title: str, wf: OrderWorkflow, order: Order, provider_name: str) -> None:
    st.title(title)
    st.caption(f"Pedido #{int(order.id)} · Proveedor: **{provider_name}**")
    st.info(f"Estado actual: **{_state_name(wf.state)}**", icon="🧭")


def _timeline(*, wf: OrderWorkflow) -> None:
    with st.expander("📜 Historial del flujo", expanded=False):
        try:
            evs = _load_events(
                venue_id=int(wf.venue_id),
                order_id=int(wf.order_id),
                provider_name=_s(wf.provider_name),
            )
        except Exception as e:
            st.error(f"No se pudo cargar el historial: {e}")
            return

        if not evs:
            st.caption("Sin eventos aún.")
            return

        for ev in evs:
            at = ev.at.strftime("%Y-%m-%d %H:%M")
            st.write(f"{at} · **{_state_name(ev.from_state)} → {_state_name(ev.to_state)}** · {ev.actor_role}")
            if ev.note:
                st.caption(ev.note)


def _render_order_lines_cards(
    *,
    lines: list[OrderLine],
    prod_map: dict[int, Product],
    editable: bool = False,
    base_key: str = "",
) -> tuple[dict[int, dict[str, Any]], str]:
    """
    Render order lines (no prices).

    If editable=True (supplier), shows per-line status (OK/Parcial/Falta) + qty/reason and returns:
      (lines_payload, inferred_to_state)
    """
    lines_payload: Dict[int, Dict[str, Any]] = {}

    ok_count = 0
    missing_count = 0
    total = 0

    for ln in lines:
        total += 1
        lid = int(getattr(ln, "id", 0) or 0)

        p = prod_map.get(int(ln.product_id)) if ln.product_id is not None else None
        label = _product_label(p, fallback=getattr(ln, "spoken_name", None) or "Producto")
        qty_ordered = float(_line_qty(ln))
        unit = (_s(getattr(p, "unit", "")) if p else _s(getattr(ln, "unit", "")) or "unidad").lower()

        if not editable:
            st.markdown(f"• **{label}** — {qty_ordered:g} {unit}")
            continue

        with st.container(border=True):
            st.markdown(
                f"""
                <div style="display:flex;justify-content:space-between;gap:10px;align-items:baseline;flex-wrap:wrap">
                  <div style="font-weight:850">{label}</div>
                  <div style="font-weight:850;white-space:nowrap">{qty_ordered:g} {unit}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns([1.1, 0.9, 2.0], vertical_alignment="center")

            with c1:
                status = st.selectbox(
                    "Estado",
                    options=["ok", "partial", "missing"],
                    index=0,
                    format_func=lambda x: {"ok": "✅ OK", "partial": "🟡 Parcial", "missing": "🔴 Falta"}[x],
                    key=f"{base_key}_st_{lid}",
                    label_visibility="collapsed",
                )

            with c2:
                if status == "partial":
                    qty_send = st.number_input(
                        "Qty",
                        min_value=0.0,
                        max_value=float(qty_ordered),
                        step=1.0,
                        value=float(qty_ordered),
                        key=f"{base_key}_qty_{lid}",
                        label_visibility="collapsed",
                    )
                elif status == "missing":
                    qty_send = 0.0
                    st.caption("0")
                else:
                    qty_send = float(qty_ordered)
                    st.caption("—")

            with c3:
                reason = ""
                if status in {"partial", "missing"}:
                    reason = st.text_input(
                        "Motivo",
                        value="",
                        placeholder="Opcional…",
                        key=f"{base_key}_rsn_{lid}",
                        label_visibility="collapsed",
                    )

            if status == "ok":
                ok_count += 1
            elif status == "missing":
                missing_count += 1

            lines_payload[lid] = {
                "qty_ordered": float(qty_ordered),
                "supplier_status": status,
                "supplier_qty": float(qty_send),
                "supplier_reason": _s(reason) or None,
            }

    if total > 0 and ok_count == total:
        inferred = "SUPPLIER_CONFIRMED_FULL"
    elif total > 0 and missing_count == total:
        inferred = "SUPPLIER_CONFIRMED_NONE"
    else:
        inferred = "SUPPLIER_CONFIRMED_PARTIAL"

    return lines_payload, inferred


# =============================================================================
# UI: Supplier
# =============================================================================

def _supplier_ui(*, order: Order, provider_name: str, wf: OrderWorkflow, lines: list[OrderLine], prod_map: dict[int, Product]) -> None:
    _header(title="Confirmación del proveedor", wf=wf, order=order, provider_name=provider_name)

    if wf.state == "CLOSED":
        st.success("Este flujo está cerrado. No se requieren más acciones.")
        _timeline(wf=wf)
        return

    if wf.state == "ORDER_SENT":
        st.subheader("Confirmar estado del envío")
        st.caption("Marca cada línea como **OK**, **Parcial** o **Falta**. (Sin precios)")
        base_key = f"supp_{int(order.id)}_{provider_name}"

        lines_payload, to_state = _render_order_lines_cards(
            lines=lines,
            prod_map=prod_map,
            editable=True,
            base_key=base_key,
        )

        comment = st.text_area("Nota (opcional)", placeholder="Ej.: llega mañana / falta X / etc.", key=f"{base_key}_note")

        if st.button("Guardar", type="primary", use_container_width=True, key=f"{base_key}_save"):
            # Persist per-line status for dashboards
            _save_followups_bulk(
                venue_id=int(order.venue_id),
                order_id=int(order.id),
                provider_name=provider_name,
                per_line=lines_payload,
                actor="supplier",
            )

            # Workflow transition
            note = _s(comment) or None
            wf2 = _transition(
                wf=wf,
                to_state=to_state,
                actor_role="supplier",
                actor="supplier",
                note=note,
            )
            st.success(f"Guardado. Nuevo estado: {_state_name(wf2.state)}")
            st.rerun()

    elif wf.state == "WAITING_SUPPLIER_ACTION":
        st.subheader("Incidencia abierta: discrepancia con factura")
        st.info("La factura indica que se entregó algo que el local NO recibió. Debes indicar cómo lo resolvéis.", icon="🧾")

        tickets = _load_open_tickets(
            venue_id=int(order.venue_id),
            order_id=int(order.id),
            provider_name=provider_name,
        )

        if tickets:
            st.markdown("#### Líneas afectadas")
            for t in tickets:
                st.markdown(
                    f"- **{t.product_name}** · facturado: {float(t.qty_invoiced or t.qty_expected):g} {t.unit} · recibido: {float(t.qty_received):g} {t.unit}"
                )
                if t.invoice_number:
                    st.caption(f"Factura: {t.invoice_number}")
                if t.note:
                    st.caption(t.note)

        action = st.radio(
            "Solución",
            ["Emitir abono (credit note)", "Enviar entrega complementaria"],
            horizontal=False,
            key="supp_action",
        )
        resolution_kind = "credit_note" if action.startswith("Emitir") else "supplementary_delivery"
        to_state = "SUPPLIER_CREDIT_NOTE_ISSUED" if resolution_kind == "credit_note" else "SUPPLEMENTARY_DELIVERY_SENT"

        comment = st.text_area("Nota (opcional)", placeholder="Ej.: número de abono / fecha estimada / etc.", key="supp_action_note")

        if st.button("Confirmar acción", type="primary", use_container_width=True, key="supp_action_confirm"):
            resolved_tickets = _resolve_open_tickets(
                venue_id=int(order.venue_id),
                order_id=int(order.id),
                provider_name=provider_name,
                resolution_kind=resolution_kind,
                resolution_note=_s(comment) or None,
            )

            wf2 = _transition(
                wf=wf,
                to_state=to_state,
                actor_role="supplier",
                actor="supplier",
                note=_s(comment) or None,
            )
            wf3 = _transition(
                wf=wf2,
                to_state="CLOSED",
                actor_role="system",
                actor="system",
                note="Auto-close after supplier action.",
            )

            # Accountant notification (optional)
            invoice_no = resolved_tickets[0].invoice_number if resolved_tickets else None
            _email_notify_accountant(
                venue_id=int(order.venue_id),
                order_id=int(order.id),
                provider_name=provider_name,
                invoice_number=invoice_no,
                resolution="Abono" if resolution_kind == "credit_note" else "Entrega complementaria",
                resolution_note=_s(comment) or None,
                tickets=resolved_tickets,
            )

            st.success(f"Confirmado. Estado final: {_state_name(wf3.state)}")
            st.rerun()

    else:
        st.info("No hay acciones pendientes para el proveedor en este estado.")

    _timeline(wf=wf)


# =============================================================================
# UI: Venue (Receiving + Manager)
# =============================================================================

def _venue_ui(
    *,
    order: Order,
    provider_name: str,
    wf: OrderWorkflow,
    lines: list[OrderLine],
    prod_map: dict[int, Product],
) -> None:
    _header(title="Seguimiento (local)", wf=wf, order=order, provider_name=provider_name)

    if wf.state == "CLOSED":
        st.success("Este flujo está cerrado. No se requieren más acciones.")
        _timeline(wf=wf)
        return

    with st.expander("🧾 Líneas del pedido", expanded=True):
        for ln in lines:
            p = prod_map.get(int(ln.product_id)) if ln.product_id is not None else None
            name = _product_label(p, fallback=getattr(ln, "spoken_name", None) or "Producto")
            unit = (getattr(p, "unit", "") if p else getattr(ln, "unit", "") or "") or ""
            st.write(f"- {name} · pedido: {_line_qty(ln):g} {unit}")

    st.divider()

    if wf.state in {"SUPPLIER_CONFIRMED_FULL", "SUPPLIER_CONFIRMED_PARTIAL", "SUPPLIER_CONFIRMED_NONE"}:
        st.subheader("Recepción")
        st.info("Empleado de recepción: confirma que la entrega ocurrió.", icon="📦")
        note = st.text_area("Nota de recepción (opcional)")
        if st.button("Marcar como RECIBIDO", type="primary", use_container_width=True):
            wf2 = _transition(
                wf=wf,
                to_state="RECEIVED",
                actor_role="receiving_employee",
                actor="venue",
                note=_s(note) or None,
            )

            # Keep receipt marker
            _upsert_provider_receipt(
                venue_id=int(order.venue_id),
                order_id=int(order.id),
                provider_name=provider_name,
                received=True,
                note=_s(note) or None,
                actor="venue",
            )

            st.success(f"Nuevo estado: {_state_name(wf2.state)}")
            st.rerun()

    elif wf.state == "RECEIVED":
        st.subheader("Comparar con la factura (y abrir incidencia si hay problema)")

        invoice_number = st.text_input("Nº de factura (obligatorio si hay incidencia)", placeholder="Ej.: F-2026-00123")
        note = st.text_area("Nota general (opcional)", placeholder="Ej.: faltan 2 cajas / producto equivocado / etc.")

        st.markdown("#### Cantidades por línea")
        st.caption("Regla: **Discrepancia con factura** = la factura indica que se entregó (qty facturada) pero el local NO recibió todo.")

        per_line: Dict[int, Dict[str, Any]] = {}
        any_not_listed = False
        any_discrepancy = False

        for ln in lines:
            lid = int(getattr(ln, "id", 0) or 0)
            p = prod_map.get(int(ln.product_id)) if ln.product_id is not None else None
            name = _product_label(p, fallback=getattr(ln, "spoken_name", None) or "Producto")
            unit = (_s(getattr(p, "unit", "")) if p else _s(getattr(ln, "unit", "")) or "unidad").lower()
            qty_ordered = float(_line_qty(ln))

            with st.container(border=True):
                st.markdown(f"**{name}** · pedido: {qty_ordered:g} {unit}")

                c1, c2, c3 = st.columns([1.1, 1.1, 1.3], vertical_alignment="center")

                with c1:
                    listed = st.checkbox("Está en la factura", value=True, key=f"inv_listed_{lid}")
                with c2:
                    qty_invoiced = st.number_input(
                        "Qty facturada",
                        min_value=0.0,
                        step=1.0,
                        value=float(qty_ordered),
                        key=f"inv_qty_{lid}",
                        disabled=not listed,
                    )
                with c3:
                    qty_received = st.number_input(
                        "Qty recibida",
                        min_value=0.0,
                        step=1.0,
                        value=float(qty_ordered),
                        key=f"recv_qty_{lid}",
                    )

                if not listed:
                    any_not_listed = True

                if listed and float(qty_invoiced) > float(qty_received) + 1e-9:
                    any_discrepancy = True
                    st.warning("Discrepancia: facturado > recibido", icon="⚠️")

                per_line[lid] = {
                    "qty_ordered": qty_ordered,
                    "qty_invoiced": float(qty_invoiced) if listed else 0.0,
                    "qty_received": float(qty_received),
                    "listed": bool(listed),
                }

        col_a, col_b = st.columns([1, 1])

        with col_a:
            if st.button("✅ Todo coincide → Cerrar", type="primary", use_container_width=True, key="close_match"):
                if any_not_listed or any_discrepancy:
                    st.error("Hay líneas marcadas como NO listadas o con discrepancia. No puedes cerrar como 'todo coincide'.")
                else:
                    # Store invoice number (even if no incidence)
                    _upsert_provider_receipt(
                        venue_id=int(order.venue_id),
                        order_id=int(order.id),
                        provider_name=provider_name,
                        invoice_number=_s(invoice_number) or None,
                        actor="venue",
                    )

                    wf2 = _transition(wf=wf, to_state="MATCHED_WITH_INVOICE", actor_role="receiving_employee", actor="venue")
                    wf3 = _transition(wf=wf2, to_state="CLOSED", actor_role="system", actor="system")
                    st.success(f"Estado final: {_state_name(wf3.state)}")
                    st.rerun()

        with col_b:
            if st.button("⚠️ Hay problema → Abrir incidencia", type="secondary", use_container_width=True, key="open_issue"):
                if not _s(invoice_number):
                    st.error("Para abrir una incidencia necesitas el Nº real de la factura.")
                    st.stop()

                # Always store invoice number
                _upsert_provider_receipt(
                    venue_id=int(order.venue_id),
                    order_id=int(order.id),
                    provider_name=provider_name,
                    invoice_number=_s(invoice_number) or None,
                    actor="venue",
                )

                # Priority: if NOT listed on invoice => operational missing (manager)
                if any_not_listed:
                    wf2 = _transition(
                        wf=wf,
                        to_state="OPERATIONAL_MISSING_PRODUCT",
                        actor_role="receiving_employee",
                        actor="venue",
                        note=_s(note) or None,
                    )
                    wf3 = _transition(wf=wf2, to_state="WAITING_MANAGER_DECISION", actor_role="system", actor="system")
                    st.success(f"Nuevo estado: {_state_name(wf3.state)}")
                    st.rerun()

                if not any_discrepancy:
                    st.error("Para 'Discrepancia con factura' debe existir al menos una línea con qty facturada > qty recibida.")
                    st.stop()

                affected = {
                    lid: v
                    for lid, v in per_line.items()
                    if v.get("listed") and float(v.get("qty_invoiced") or 0.0) > float(v.get("qty_received") or 0.0) + 1e-9
                }
                created = _create_invoice_discrepancy_tickets(
                    venue_id=int(order.venue_id),
                    order_id=int(order.id),
                    provider_name=provider_name,
                    invoice_number=_s(invoice_number) or None,
                    per_line=affected,
                    prod_map=prod_map,
                    lines=lines,
                    note=_s(note) or None,
                )

                wf2 = _transition(
                    wf=wf,
                    to_state="INVOICE_DISCREPANCY",
                    actor_role="receiving_employee",
                    actor="venue",
                    note=_s(note) or None,
                )
                wf3 = _transition(wf=wf2, to_state="WAITING_SUPPLIER_ACTION", actor_role="system", actor="system")

                # Notify supplier (email) with the signed supplier link
                open_tickets = _load_open_tickets(
                    venue_id=int(order.venue_id),
                    order_id=int(order.id),
                    provider_name=provider_name,
                )
                _email_notify_supplier_incidence(
                    venue_id=int(order.venue_id),
                    order_id=int(order.id),
                    provider_name=provider_name,
                    invoice_number=_s(invoice_number) or None,
                    tickets=open_tickets,
                    note=_s(note) or None,
                )

                st.success(f"Incidencia abierta ({created} ticket/s). Nuevo estado: {_state_name(wf3.state)}")
                st.rerun()

    elif wf.state == "WAITING_MANAGER_DECISION":
        st.subheader("Decisión del manager")
        choice = st.radio(
            "Decisión",
            ["Reordenar al mismo proveedor", "Cambiar de proveedor", "Ya no se necesita"],
        )
        to_state = {
            "Reordenar al mismo proveedor": "DECISION_REORDER_SAME",
            "Cambiar de proveedor": "DECISION_SWITCH_SUPPLIER",
            "Ya no se necesita": "DECISION_NOT_NEEDED",
        }[choice]

        note = st.text_area("Nota (opcional)")
        if st.button("Guardar decisión → Cerrar", type="primary", use_container_width=True):
            wf2 = _transition(wf=wf, to_state=to_state, actor_role="manager", actor="venue", note=_s(note) or None)
            wf3 = _transition(wf=wf2, to_state="CLOSED", actor_role="system", actor="system")
            st.success(f"Estado final: {_state_name(wf3.state)}")
            st.rerun()

    else:
        st.info("No hay acciones pendientes para el local en este estado.")

    _timeline(wf=wf)


# =============================================================================
# Entry point (public link)
# =============================================================================

def seguimiento_page() -> None:
    st.set_page_config(page_title="Seguimiento", layout="centered")

    qp = st.query_params
    order_id = _f(qp.get("order_id"))
    provider = _s(qp.get("provider"))
    role = _s(qp.get("role"))
    sig = _s(qp.get("sig"))

    if order_id is None or not provider or not role or not sig:
        st.error("Missing link parameters.")
        return

    order_id_i = int(order_id)
    provider_n = norm_provider(provider)
    role_n = norm_role(role)

    if role_n not in (ROLE_SUPPLIER, ROLE_VENUE):
        st.error("Invalid role.")
        return

    if not verify_link(order_id=order_id_i, provider_name=provider_n, role=role_n, sig=sig):
        st.error("Invalid or tampered link.")
        return

    try:
        order, lines_all, prod_map = _load_order_bundle(order_id_i)
    except Exception as e:
        st.error(f"Could not load order: {e}")
        return

    lines = [ln for ln in lines_all if _line_provider_name(ln, prod_map) == provider_n]
    if not lines:
        st.warning("No lines for this supplier.")
        return

    try:
        wf = _get_workflow(
            venue_id=int(order.venue_id),
            order_id=int(order.id),
            provider_name=provider_n,
        )
    except Exception as e:
        st.error(str(e))
        st.caption("Abre el pedido en Pedidos → Listo/Enviar para inicializar el flujo (ORDER_SENT).")
        return

    if role_n == ROLE_SUPPLIER:
        _supplier_ui(order=order, provider_name=provider_n, wf=wf, lines=lines, prod_map=prod_map)
    else:
        _venue_ui(order=order, provider_name=provider_n, wf=wf, lines=lines, prod_map=prod_map)


if __name__ == "__main__":
    seguimiento_page()
