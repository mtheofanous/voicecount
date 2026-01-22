# features/seguimiento/seguimiento.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Tuple, List, Dict

import streamlit as st
from sqlmodel import select

from core.db import get_session
from core.public_links import ROLE_SUPPLIER, ROLE_VENUE, norm_provider, norm_role, verify_link
from domain.models import (
    Order,
    OrderLine,
    Product,
    OrderWorkflow,
    OrderWorkflowEvent,
    ORDER_STATES,
)

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
    """
    Accepts either "123", 123, ["123"] (Streamlit query params),
    returns float or None.
    """
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


def _render_order_lines_cards(
    *,
    lines: list[OrderLine],
    prod_map: dict[int, Product],
    editable: bool = False,
    base_key: str = "",
) -> tuple[dict[int, dict[str, Any]], str]:
    """Render order lines (no prices).

    If editable=True, shows per-line status (OK/Parcial/Falta) + qty/reason and returns:
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
        label = _product_label(p, fallback="Producto")
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
                    qty_recv = st.number_input(
                        "Qty",
                        min_value=0.0,
                        max_value=float(qty_ordered),
                        step=1.0,
                        value=float(qty_ordered),
                        key=f"{base_key}_qty_{lid}",
                        label_visibility="collapsed",
                    )
                elif status == "missing":
                    qty_recv = 0.0
                    st.caption("0")
                else:
                    qty_recv = float(qty_ordered)
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
                "product_id": int(ln.product_id) if ln.product_id is not None else None,
                "ordered_qty": float(qty_ordered),
                "supplier_status": status,
                "supplier_qty": float(qty_recv),
                "supplier_reason": _s(reason) or None,
            }

    if total > 0 and ok_count == total:
        inferred = "SUPPLIER_CONFIRMED_FULL"
    elif total > 0 and missing_count == total:
        inferred = "SUPPLIER_CONFIRMED_NONE"
    else:
        inferred = "SUPPLIER_CONFIRMED_PARTIAL"

    return lines_payload, inferred

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
    """
    Seguimiento does NOT create workflows.
    ORDER_SENT must be created when the order is sent (Orders tab).
    """
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


def _transition(
    *,
    wf: OrderWorkflow,
    to_state: str,
    actor_role: str,
    actor: Optional[str] = None,
    note: Optional[str] = None,
) -> OrderWorkflow:
    """
    Enforce transitions exactly as the flow definition.
    (No invented roles/actions/states.)
    """
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
# UI helpers
# =============================================================================

def _header(*, title: str, wf: OrderWorkflow, order: Order, provider_name: str) -> None:
    st.title(title)
    st.caption(f"Pedido #{int(order.id)} · Proveedor: **{provider_name}**")
    st.info(f"Estado actual: **{_state_name(wf.state)}**", icon="🧭")


def _timeline(*, wf: OrderWorkflow) -> None:
    """
    Read-only visibility for accountant/owner/manager.
    No actions here.
    """
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

    comment = st.text_area("Nota (opcional)", placeholder="Ej.: llega mañana / falta X / etc.")

    if st.button("Guardar", type="primary", use_container_width=True):
        # Persist line-level info inside the workflow event note (lightweight, no extra tables)
        note = _s(comment) or ""
        if lines_payload:
            note = (note + " " if note else "") + "LINE_STATUS_JSON=" + str(lines_payload)

            wf2 = _transition(
                wf=wf,
                to_state=to_state,
                actor_role="supplier",
                actor="supplier",
                note=note or None,
            )
        st.success(f"Guardado. Nuevo estado: {_state_name(wf2.state)}")
        st.rerun()

    elif wf.state == "WAITING_SUPPLIER_ACTION":
        st.subheader("Acción del proveedor requerida")
        st.info("Se detectó una discrepancia que está listada en la factura. Elige una acción.", icon="🧾")

        action = st.radio(
            "Acción",
            ["Emitir abono (credit note)", "Enviar entrega complementaria"],
            horizontal=False,
        )
        to_state = "SUPPLIER_CREDIT_NOTE_ISSUED" if action.startswith("Emitir") else "SUPPLEMENTARY_DELIVERY_SENT"

        comment = st.text_area("Nota (opcional)", placeholder="Ej.: fecha estimada / referencia / etc.")
        if st.button("Confirmar acción", type="primary", use_container_width=True):
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
            name = p.name if p else "Producto"
            unit = (p.unit if p else "") or ""
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
            st.success(f"Nuevo estado: {_state_name(wf2.state)}")
            st.rerun()

    elif wf.state == "RECEIVED":
        st.subheader("Comparar con la factura")
        decision = st.radio(
            "Resultado",
            ["Todo coincide con la factura", "Hay una discrepancia"],
        )

        if decision == "Todo coincide con la factura":
            if st.button("Confirmar → Cerrar", type="primary", use_container_width=True):
                wf2 = _transition(wf=wf, to_state="MATCHED_WITH_INVOICE", actor_role="receiving_employee", actor="venue")
                wf3 = _transition(wf=wf2, to_state="CLOSED", actor_role="system", actor="system")
                st.success(f"Estado final: {_state_name(wf3.state)}")
                st.rerun()

        else:
            listed = st.radio(
                "¿El producto faltante/incorrecto está listado en la factura?",
                ["Sí, está listado en la factura", "No, NO está listado en la factura"],
            )
            note = st.text_area("Nota de discrepancia (opcional)")

            if listed.startswith("Sí"):
                if st.button("Marcar discrepancia → Esperar proveedor", type="primary", use_container_width=True):
                    wf2 = _transition(
                        wf=wf,
                        to_state="INVOICE_DISCREPANCY",
                        actor_role="receiving_employee",
                        actor="venue",
                        note=_s(note) or None,
                    )
                    wf3 = _transition(wf=wf2, to_state="WAITING_SUPPLIER_ACTION", actor_role="system", actor="system")
                    st.success(f"Nuevo estado: {_state_name(wf3.state)}")
                    st.rerun()
            else:
                if st.button("Marcar falta operativa → Esperar manager", type="primary", use_container_width=True):
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
