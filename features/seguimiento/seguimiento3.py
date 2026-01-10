# features/seguimiento/seguimiento.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlmodel import SQLModel, select

from core.db import get_session
from core.public_links import ROLE_SUPPLIER, ROLE_VENUE, norm_provider, norm_role, verify_link
from domain.models import Order, OrderLine, Product, ProviderLineFollowUp

STATUS_OPTIONS = ["ok", "partial", "missing"]


# =============================================================================
# Small helpers
# =============================================================================

def _safe_str(x: Any) -> str:
    return ("" if x is None else str(x)).strip()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "" or x is pd.NA:
            return default
        return float(x)
    except Exception:
        return default


def _status_clean(x: Any) -> str:
    v = (str(x) if x is not None else "ok").strip().lower()
    return v if v in set(STATUS_OPTIONS) else "ok"


def _product_label(p: Optional[Product]) -> Tuple[str, str, str]:
    """Returns (name, description, unit) with safe defaults."""
    if not p:
        return ("Producto", "", "unidad")
    name = _safe_str(getattr(p, "name", "")) or "Producto"
    desc = _safe_str(getattr(p, "description", ""))
    unit = _safe_str(getattr(p, "unit", "")) or "unidad"
    return (name, desc, unit)


def _ensure_tables() -> None:
    # Creates all SQLModel tables (including ProviderLineFollowUp) for fresh DB
    with get_session() as s:
        SQLModel.metadata.create_all(s.get_bind())


def _get_or_create_followup(
    *,
    venue_id: int,
    order_id: int,
    provider_name: str,
    order_line_id: int,
    qty_ordered: float,
) -> ProviderLineFollowUp:
    prov = norm_provider(provider_name)

    with get_session() as s:
        obj = s.exec(
            select(ProviderLineFollowUp).where(
                ProviderLineFollowUp.order_id == int(order_id),
                ProviderLineFollowUp.provider_name == prov,
                ProviderLineFollowUp.order_line_id == int(order_line_id),
            )
        ).first()

        if obj:
            # Keep qty_ordered in sync if it was never set
            if getattr(obj, "qty_ordered", None) is None:
                obj.qty_ordered = float(qty_ordered)  # type: ignore[attr-defined]
                obj.updated_at = datetime.utcnow()  # type: ignore[attr-defined]
                s.add(obj)
                s.commit()
            return obj

        obj = ProviderLineFollowUp(
            venue_id=int(venue_id),
            order_id=int(order_id),
            provider_name=prov,
            order_line_id=int(order_line_id),
            qty_ordered=float(qty_ordered),
            updated_at=datetime.utcnow(),
        )
        s.add(obj)
        s.commit()
        s.refresh(obj)
        return obj


def _save_followup_updates(
    *,
    followup_id: int,
    role: str,
    supplier_payload: Dict[str, Any],
    venue_payload: Dict[str, Any],
) -> None:
    role = norm_role(role)
    now = datetime.utcnow()

    with get_session() as s:
        obj = s.get(ProviderLineFollowUp, int(followup_id))
        if not obj:
            return

        obj.updated_at = now  # type: ignore[attr-defined]

        if role == ROLE_SUPPLIER:
            obj.supplier_status = _status_clean(supplier_payload.get("supplier_status"))  # type: ignore[attr-defined]
            obj.supplier_qty = _safe_float(supplier_payload.get("supplier_qty"), default=0.0)  # type: ignore[attr-defined]
            obj.supplier_reason = _safe_str(supplier_payload.get("supplier_reason")) or None  # type: ignore[attr-defined]
            obj.supplier_comment = _safe_str(supplier_payload.get("supplier_comment")) or None  # type: ignore[attr-defined]
        else:
            obj.venue_qty = _safe_float(venue_payload.get("venue_qty"), default=0.0)  # type: ignore[attr-defined]
            obj.venue_comment = _safe_str(venue_payload.get("venue_comment")) or None  # type: ignore[attr-defined]

        s.add(obj)
        s.commit()


# =============================================================================
# Page
# =============================================================================

def seguimiento_page() -> None:
    st.set_page_config(page_title="Seguimiento", layout="wide")
    _ensure_tables()

    qp = st.query_params

    # Required query params
    try:
        order_id = int(_safe_str(qp.get("order_id", "0")))
    except Exception:
        order_id = 0

    provider_name = norm_provider(_safe_str(qp.get("provider", "")))
    role = norm_role(_safe_str(qp.get("role", ROLE_SUPPLIER)))
    sig = _safe_str(qp.get("sig", ""))

    if order_id <= 0 or not provider_name or not sig:
        st.error("Link inválido (faltan parámetros).")
        st.stop()

    if not verify_link(order_id=order_id, provider_name=provider_name, role=role, sig=sig):
        st.error("Link inválido.")
        st.stop()

    # Load order + lines + products
    with get_session() as s:
        order = s.exec(select(Order).where(Order.id == int(order_id))).first()
        if not order:
            st.error("Pedido no encontrado.")
            st.stop()

        lines = s.exec(select(OrderLine).where(OrderLine.order_id == int(order_id))).all()
        products = s.exec(select(Product).where(Product.venue_id == int(order.venue_id))).all()

    products_by_id: Dict[int, Product] = {int(p.id): p for p in products if p.id is not None}

    # Filter lines by provider
    provider_lines: List[OrderLine] = []
    for ln in lines:
        pid = getattr(ln, "product_id", None)
        p = products_by_id.get(int(pid)) if pid is not None else None
        prov = norm_provider(_safe_str(getattr(p, "provider_name", "")) if p else _safe_str(getattr(ln, "provider", "")))
        if prov == provider_name:
            provider_lines.append(ln)

    st.title("📦 Seguimiento")
    st.caption(
        f"Pedido #{int(order.id)} · Proveedor: **{provider_name}** · Vista: **{('Proveedor' if role == ROLE_SUPPLIER else 'Local')}**"
    )

    order_status = (_safe_str(getattr(order, "status", "")) or "").strip().lower()
    is_final = order_status == "final"
    if is_final:
        st.warning("Este pedido ya está en **Final**. Solo lectura.")

    if not provider_lines:
        st.info("No hay líneas para este proveedor en este pedido.")
        st.stop()

    # Minimal guidance
    if role == ROLE_SUPPLIER:
        st.markdown("Marca cada producto como **OK / Parcial / Falta** y añade motivo/comentario si aplica.")
    else:
        st.markdown("Introduce lo **recibido** y añade tu comentario si hace falta.")

    # Build rows
    rows_ui: List[Dict[str, Any]] = []
    for ln in provider_lines:
        line_id = int(getattr(ln, "id", 0) or 0)
        pid = getattr(ln, "product_id", None)
        p = products_by_id.get(int(pid)) if pid is not None else None

        name, desc, unit = _product_label(p)
        qty_ordered = _safe_float(getattr(ln, "quantity", 0.0), default=0.0)

        fu = _get_or_create_followup(
            venue_id=int(order.venue_id),
            order_id=int(order.id),
            provider_name=provider_name,
            order_line_id=line_id,
            qty_ordered=qty_ordered,
        )

        supplier_status = _status_clean(getattr(fu, "supplier_status", "ok"))
        supplier_qty_raw = getattr(fu, "supplier_qty", None)
        supplier_qty = _safe_float(supplier_qty_raw, default=qty_ordered) if supplier_qty_raw is not None else qty_ordered

        venue_qty_raw = getattr(fu, "venue_qty", None)
        # UX default: if venue hasn't filled anything yet, prefill with supplier_qty (or ordered).
        if venue_qty_raw is None:
            venue_default = supplier_qty if supplier_qty is not None else qty_ordered
        else:
            venue_default = _safe_float(venue_qty_raw, default=0.0)

        rows_ui.append(
            {
                "followup_id": int(getattr(fu, "id")),
                "order_line_id": line_id,
                "name": name,
                "desc": desc,
                "unit": unit,
                "qty_ordered": qty_ordered,
                # supplier fields
                "supplier_status": supplier_status,
                "supplier_qty": supplier_qty,
                "supplier_reason": _safe_str(getattr(fu, "supplier_reason", "")),
                "supplier_comment": _safe_str(getattr(fu, "supplier_comment", "")),
                # venue fields
                "venue_qty_default": venue_default,
                "venue_comment": _safe_str(getattr(fu, "venue_comment", "")),
                "updated_at": getattr(fu, "updated_at", None),
            }
        )

    saved_payload: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []

    with st.form("seguimiento_form", clear_on_submit=False):
        for r in rows_ui:
            with st.container(border=True):
                top = st.columns([4.2, 1.0, 1.2], vertical_alignment="center")

                with top[0]:
                    st.markdown(f"**{r['name']}**")
                    if r["desc"]:
                        st.caption(r["desc"])
                with top[1]:
                    st.metric("Pedido", f"{r['qty_ordered']} {r['unit']}")
                with top[2]:
                    ua = r.get("updated_at")
                    if ua:
                        st.caption(f"Última act.: {ua}")

                cA, cB = st.columns(2, gap="large")

                # Supplier section
                with cA:
                    st.markdown("#### Proveedor")
                    disabled_supplier = (role != ROLE_SUPPLIER) or is_final

                    supplier_status = st.selectbox(
                        "Estado",
                        options=STATUS_OPTIONS,
                        index=STATUS_OPTIONS.index(_status_clean(r["supplier_status"])),
                        disabled=disabled_supplier,
                        key=f"s_status_{r['order_line_id']}",
                    )

                    # If OK, default shipped qty to ordered qty (still editable)
                    default_ship = r["qty_ordered"] if supplier_status == "ok" else _safe_float(r["supplier_qty"], default=0.0)
                    supplier_qty = st.number_input(
                        "Cantidad enviada",
                        min_value=0.0,
                        step=1.0,
                        value=float(default_ship),
                        disabled=disabled_supplier,
                        key=f"s_qty_{r['order_line_id']}",
                    )

                    supplier_reason = st.text_input(
                        "Motivo (si parcial/falta)",
                        value=_safe_str(r["supplier_reason"]),
                        disabled=disabled_supplier,
                        key=f"s_reason_{r['order_line_id']}",
                        placeholder="Ej: sin stock, llega mañana...",
                    )
                    supplier_comment = st.text_area(
                        "Comentario",
                        value=_safe_str(r["supplier_comment"]),
                        height=80,
                        disabled=disabled_supplier,
                        key=f"s_comment_{r['order_line_id']}",
                        placeholder="Añade detalles útiles (lote, sustitución, hora de salida, etc.)",
                    )

                # Venue section
                with cB:
                    st.markdown("#### Local")
                    disabled_venue = (role != ROLE_VENUE) or is_final

                    venue_qty = st.number_input(
                        "Cantidad recibida",
                        min_value=0.0,
                        step=1.0,
                        value=float(_safe_float(r["venue_qty_default"], default=0.0)),
                        disabled=disabled_venue,
                        key=f"v_qty_{r['order_line_id']}",
                    )
                    venue_comment = st.text_area(
                        "Comentario",
                        value=_safe_str(r["venue_comment"]),
                        height=140,
                        disabled=disabled_venue,
                        key=f"v_comment_{r['order_line_id']}",
                        placeholder="Ej: faltó 1, llegó dañado, sustitución no aceptada...",
                    )

                supplier_payload = {
                    "supplier_status": supplier_status,
                    "supplier_qty": supplier_qty,
                    "supplier_reason": supplier_reason,
                    "supplier_comment": supplier_comment,
                }
                venue_payload = {
                    "venue_qty": venue_qty,
                    "venue_comment": venue_comment,
                }
                saved_payload.append((int(r["followup_id"]), supplier_payload, venue_payload))

        can_save = not is_final
        label = "💾 Guardar"
        submit = st.form_submit_button(label, type="primary", use_container_width=True, disabled=not can_save)

    if submit and not is_final:
        for followup_id, sp, vp in saved_payload:
            _save_followup_updates(
                followup_id=followup_id,
                role=role,
                supplier_payload=sp,
                venue_payload=vp,
            )
        st.success("Guardado ✅ Ya puedes cerrar esta página.")


if __name__ == "__main__":
    seguimiento_page()
