# features/seguimiento/seguimiento.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlmodel import SQLModel, Field, select

from core.db import get_session
from core.public_links import ROLE_SUPPLIER, ROLE_VENUE, norm_provider, norm_role, verify_link
from domain.models import Order, OrderLine, Product, ProviderLineFollowUp, SeguimientoTicket

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
STATUS_OPTIONS = ["ok", "partial", "missing"]
TICKET_KINDS = [
    "supplier_short",    # supplier declared partial/missing or qty < ordered
    "delivery_mismatch", # received != expected (expected defaults to supplier_qty when present, else ordered)
    "invoice_mismatch",  # invoice qty/price mismatch vs received/catalog/expected
]
TICKET_STATES = ["open", "resolved"]


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


# def _ensure_tables() -> None:
#     # Creates all SQLModel tables (including SeguimientoTicket) for fresh DB
#     with get_session() as s:
#         SQLModel.metadata.create_all(s.get_bind())


# =============================================================================
# Followup helpers (ProviderLineFollowUp)
# =============================================================================
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
            # keep qty_ordered in sync if never set
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


def _save_followup_updates_bulk(
    *,
    role: str,
    updates: List[Dict[str, Any]],
) -> None:
    """
    updates: [{followup_id, supplier_status, supplier_qty, supplier_reason, supplier_comment, venue_qty, venue_comment}]
    """
    role = norm_role(role)
    now = datetime.utcnow()

    with get_session() as s:
        for u in updates:
            fid = int(u.get("followup_id") or 0)
            if fid <= 0:
                continue

            obj = s.get(ProviderLineFollowUp, fid)
            if not obj:
                continue

            obj.updated_at = now  # type: ignore[attr-defined]

            if role == ROLE_SUPPLIER:
                obj.supplier_status = _status_clean(u.get("supplier_status"))  # type: ignore[attr-defined]
                obj.supplier_qty = _safe_float(u.get("supplier_qty"), default=0.0)  # type: ignore[attr-defined]
                obj.supplier_reason = _safe_str(u.get("supplier_reason")) or None  # type: ignore[attr-defined]
                obj.supplier_comment = _safe_str(u.get("supplier_comment")) or None  # type: ignore[attr-defined]
            else:
                obj.venue_qty = _safe_float(u.get("venue_qty"), default=0.0)  # type: ignore[attr-defined]
                obj.venue_comment = _safe_str(u.get("venue_comment")) or None  # type: ignore[attr-defined]

            s.add(obj)

        s.commit()


# =============================================================================
# Ticket helpers
# =============================================================================
def _get_ticket(
    *,
    order_id: int,
    provider_name: str,
    order_line_id: int,
    kind: str,
) -> Optional[SeguimientoTicket]:
    prov = norm_provider(provider_name)
    with get_session() as s:
        return s.exec(
            select(SeguimientoTicket).where(
                SeguimientoTicket.order_id == int(order_id),
                SeguimientoTicket.provider_name == prov,
                SeguimientoTicket.order_line_id == int(order_line_id),
                SeguimientoTicket.kind == kind,
            )
        ).first()


def _upsert_ticket_open(
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
    note: str,
    invoice_number: Optional[str] = None,
    qty_invoiced: Optional[float] = None,
    unit_price_expected: Optional[float] = None,
    unit_price_invoiced: Optional[float] = None,
) -> None:
    prov = norm_provider(provider_name)
    now = datetime.utcnow()

    with get_session() as s:
        row = s.exec(
            select(SeguimientoTicket).where(
                SeguimientoTicket.order_id == int(order_id),
                SeguimientoTicket.provider_name == prov,
                SeguimientoTicket.order_line_id == int(order_line_id),
                SeguimientoTicket.kind == kind,
            )
        ).first()

        if not row:
            row = SeguimientoTicket(
                venue_id=int(venue_id),
                order_id=int(order_id),
                provider_name=prov,
                order_line_id=int(order_line_id),
                kind=kind,
                state="open",
                created_at=now,
            )

        row.state = "open"
        row.updated_at = now
        row.resolved_at = None  # reopen

        row.product_name = _safe_str(product_name)
        row.unit = _safe_str(unit) or "unidad"

        row.qty_ordered = float(qty_ordered or 0.0)
        row.qty_expected = float(qty_expected or 0.0)
        row.qty_received = float(qty_received or 0.0)

        row.note = _safe_str(note) or None

        # invoice optional
        row.invoice_number = _safe_str(invoice_number) or None
        row.qty_invoiced = (None if qty_invoiced is None else float(qty_invoiced))
        row.unit_price_expected = (None if unit_price_expected is None else float(unit_price_expected))
        row.unit_price_invoiced = (None if unit_price_invoiced is None else float(unit_price_invoiced))

        s.add(row)
        s.commit()


def _resolve_ticket(
    *,
    ticket_id: int,
    resolution_note: str,
) -> None:
    now = datetime.utcnow()
    with get_session() as s:
        t = s.get(SeguimientoTicket, int(ticket_id))
        if not t:
            return
        t.state = "resolved"
        t.resolution_note = _safe_str(resolution_note) or None
        t.resolved_at = now
        t.updated_at = now
        s.add(t)
        s.commit()


def _load_tickets(*, order_id: int, provider_name: str) -> List[SeguimientoTicket]:
    prov = norm_provider(provider_name)
    with get_session() as s:
        return list(
            s.exec(
                select(SeguimientoTicket).where(
                    SeguimientoTicket.order_id == int(order_id),
                    SeguimientoTicket.provider_name == prov,
                ).order_by(SeguimientoTicket.state.asc(), SeguimientoTicket.updated_at.desc())
            ).all()
        )


# =============================================================================
# Page
# =============================================================================
def seguimiento_page() -> None:
    st.set_page_config(page_title="Seguimiento", layout="wide")
    # _ensure_tables()

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

        lines = list(s.exec(select(OrderLine).where(OrderLine.order_id == int(order_id))).all())
        products = list(s.exec(select(Product).where(Product.venue_id == int(order.venue_id))).all())

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

    # -------------------------------------------------------------------------
    # Build followups (and keep them created)
    # -------------------------------------------------------------------------
    rows: List[Dict[str, Any]] = []
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
        supplier_qty = (
            _safe_float(supplier_qty_raw, default=qty_ordered)
            if supplier_qty_raw is not None
            else qty_ordered
        )

        venue_qty_raw = getattr(fu, "venue_qty", None)
        # default venue qty: if venue hasn't filled, assume supplier_qty (or ordered)
        venue_qty = (
            supplier_qty if venue_qty_raw is None else _safe_float(venue_qty_raw, default=0.0)
        )

        rows.append(
            {
                "followup_id": int(getattr(fu, "id")),
                "order_line_id": line_id,
                "product_id": int(pid) if pid is not None else None,
                "Producto": name,
                "Descripción": desc,
                "Unidad": unit,
                "Pedido": float(qty_ordered),

                # Supplier
                "Estado proveedor": supplier_status,
                "Enviado": float(supplier_qty),
                "Motivo": _safe_str(getattr(fu, "supplier_reason", "")),
                "Comentario proveedor": _safe_str(getattr(fu, "supplier_comment", "")),

                # Venue
                "Recibido": float(venue_qty),
                "Comentario local": _safe_str(getattr(fu, "venue_comment", "")),

                # Invoice fields (venue only; stored in tickets)
                "Factura Nº": "",
                "Facturado qty": pd.NA,
                "Precio facturado": pd.NA,
            }
        )

    df = pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # Top guidance + mode switches
    # -------------------------------------------------------------------------
    is_supplier = role == ROLE_SUPPLIER
    is_venue = role == ROLE_VENUE

    topL, topM, topR = st.columns([1.4, 1.4, 2.2], vertical_alignment="center")

    with topL:
        show_desc = st.toggle("Mostrar descripción", value=False, disabled=is_supplier, key="seg_show_desc")
    with topM:
        exceptions_only = st.toggle("Solo incidencias", value=False, key="seg_ex_only")
    with topR:
        invoice_mode = st.toggle(
            "Modo factura (reconciliar)",
            value=False,
            disabled=(not is_venue),
            help="Introduce qty/precio facturado para crear incidencias de factura.",
            key="seg_invoice_mode",
        )

    # Supplier: default all OK -> only touch exceptions
    if is_supplier:
        st.info("✅ Marca SOLO lo que no está OK. Por defecto todo se considera OK con la cantidad pedida.")

    # Venue: receive + track tickets
    if is_venue:
        st.info("📦 Rellena lo recibido. Si activas Modo factura, podrás crear incidencias si la factura no cuadra.")

    # -------------------------------------------------------------------------
    # Build "incidence" mask for filtering
    # -------------------------------------------------------------------------
    def _is_incident_row(r: pd.Series) -> bool:
        ordered = _safe_float(r.get("Pedido"), 0.0)
        s_status = _status_clean(r.get("Estado proveedor"))
        s_qty = _safe_float(r.get("Enviado"), ordered)
        v_qty = _safe_float(r.get("Recibido"), 0.0)

        # incident if supplier declared partial/missing OR sent != ordered OR received != expected
        expected = s_qty if (s_qty is not None) else ordered
        if s_status != "ok":
            return True
        if abs(s_qty - ordered) > 1e-9:
            return True
        if abs(v_qty - expected) > 1e-9:
            return True
        return False

    if exceptions_only:
        mask = df.apply(_is_incident_row, axis=1)
        df_view = df.loc[mask].copy()
        if df_view.empty:
            st.success("🟢 No hay incidencias detectadas en estas líneas.")
            # still show save button so supplier can confirm quickly if they want
            df_view = df.copy()
    else:
        df_view = df.copy()

    # -------------------------------------------------------------------------
    # Columns per role (lean defaults)
    # -------------------------------------------------------------------------
    base_cols = ["Producto", "Unidad", "Pedido"]
    if show_desc and "Descripción" in df_view.columns:
        base_cols.insert(1, "Descripción")

    supplier_cols = base_cols + ["Estado proveedor", "Enviado", "Motivo", "Comentario proveedor"]
    venue_cols = base_cols + ["Estado proveedor", "Enviado", "Recibido", "Comentario local"]

    if invoice_mode and is_venue:
        venue_cols += ["Factura Nº", "Facturado qty", "Precio facturado"]

    # Hidden technical cols always present for persistence
    hidden_cols = ["followup_id", "order_line_id", "product_id"]

    # -------------------------------------------------------------------------
    # Fast actions bar
    # -------------------------------------------------------------------------
    a1, a2, a3, a4 = st.columns([1.2, 1.2, 1.4, 2.2], vertical_alignment="center")

    with a1:
        if is_supplier and st.button("✅ Todo OK", use_container_width=True, disabled=is_final):
            # set all to OK and shipped qty = ordered
            df["Estado proveedor"] = "ok"
            df["Enviado"] = df["Pedido"]
            df["Motivo"] = ""
            df["Comentario proveedor"] = ""
            st.session_state["seg_df_override"] = df
            st.rerun()

    with a2:
        if is_venue and st.button("📦 Igualar recibido = enviado", use_container_width=True, disabled=is_final):
            df["Recibido"] = df["Enviado"]
            st.session_state["seg_df_override"] = df
            st.rerun()

    with a3:
        st.caption(f"{len(provider_lines)} líneas")

    with a4:
        st.caption("Tip: activa “Solo incidencias” para tocar únicamente lo que no cuadra.")

    if "seg_df_override" in st.session_state:
        try:
            df = st.session_state["seg_df_override"].copy()
            # refresh the view after override
            if exceptions_only:
                mask = df.apply(_is_incident_row, axis=1)
                df_view = df.loc[mask].copy()
                if df_view.empty:
                    df_view = df.copy()
            else:
                df_view = df.copy()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Data editor (single table)
    # -------------------------------------------------------------------------
    # Determine which columns are editable depending on role
    editable = set()
    if is_supplier:
        editable = {"Estado proveedor", "Enviado", "Motivo", "Comentario proveedor"}
    if is_venue:
        editable = {"Recibido", "Comentario local"}
        if invoice_mode:
            editable |= {"Factura Nº", "Facturado qty", "Precio facturado"}

    display_cols = hidden_cols + (supplier_cols if is_supplier else venue_cols)

    # Build column config
    colcfg: Dict[str, Any] = {
        "followup_id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
        "order_line_id": st.column_config.NumberColumn("Line", disabled=True, width="small"),
        "product_id": st.column_config.NumberColumn("PID", disabled=True, width="small"),

        "Producto": st.column_config.TextColumn("Producto", width="large"),
        "Descripción": st.column_config.TextColumn("Descripción", width="large"),
        "Unidad": st.column_config.TextColumn("Unidad", width="small"),
        "Pedido": st.column_config.NumberColumn("Pedido", width="small"),

        "Estado proveedor": st.column_config.SelectboxColumn(
            "Estado proveedor",
            options=STATUS_OPTIONS,
            width="small",
        ),
        "Enviado": st.column_config.NumberColumn("Enviado", min_value=0.0, step=1.0, width="small"),
        "Motivo": st.column_config.TextColumn("Motivo", width="medium"),
        "Comentario proveedor": st.column_config.TextColumn("Comentario proveedor", width="large"),

        "Recibido": st.column_config.NumberColumn("Recibido", min_value=0.0, step=1.0, width="small"),
        "Comentario local": st.column_config.TextColumn("Comentario local", width="large"),

        "Factura Nº": st.column_config.TextColumn("Factura Nº", width="small"),
        "Facturado qty": st.column_config.NumberColumn("Facturado qty", min_value=0.0, step=1.0, width="small"),
        "Precio facturado": st.column_config.NumberColumn("Precio facturado", min_value=0.0, step=0.01, width="small"),
    }

    # Disable non-editable columns
    disabled_cols = [c for c in display_cols if c not in editable]

    edited = st.data_editor(
        df_view[display_cols],
        hide_index=True,
        width="stretch",
        disabled=(is_final or disabled_cols),
        column_config=colcfg,
        key="seg_editor",
    )

    # -------------------------------------------------------------------------
    # Save & ticket generation
    # -------------------------------------------------------------------------
    can_save = not is_final

    save_col1, save_col2 = st.columns([1.4, 3.6], vertical_alignment="center")
    with save_col1:
        save = st.button("💾 Guardar", type="primary", use_container_width=True, disabled=not can_save)
    with save_col2:
        st.caption("Guarda una vez. Las incidencias se crean/actualizan automáticamente como tickets.")

    if save and can_save:
        # merge edits back into full df (so filter view edits persist)
        full = df.copy()
        edited_idx = set(edited["order_line_id"].astype(int).tolist())
        for i, r in edited.iterrows():
            lid = int(r.get("order_line_id") or 0)
            if lid <= 0:
                continue
            # update matching row in full
            m = full["order_line_id"].astype(int) == lid
            for col in edited.columns:
                full.loc[m, col] = r[col]

        # Prepare followup updates
        updates: List[Dict[str, Any]] = []
        for _, r in full.iterrows():
            u: Dict[str, Any] = {"followup_id": int(r.get("followup_id") or 0)}
            if is_supplier:
                u.update(
                    {
                        "supplier_status": _status_clean(r.get("Estado proveedor")),
                        "supplier_qty": _safe_float(r.get("Enviado"), default=_safe_float(r.get("Pedido"), 0.0)),
                        "supplier_reason": _safe_str(r.get("Motivo")),
                        "supplier_comment": _safe_str(r.get("Comentario proveedor")),
                    }
                )
            if is_venue:
                u.update(
                    {
                        "venue_qty": _safe_float(r.get("Recibido"), default=0.0),
                        "venue_comment": _safe_str(r.get("Comentario local")),
                    }
                )
            updates.append(u)

        _save_followup_updates_bulk(role=role, updates=updates)

        # Ticket creation/update based on mismatches
        for _, r in full.iterrows():
            line_id = int(r.get("order_line_id") or 0)
            if line_id <= 0:
                continue

            name = _safe_str(r.get("Producto"))
            unit = _safe_str(r.get("Unidad")) or "unidad"

            qty_ordered = _safe_float(r.get("Pedido"), 0.0)

            s_status = _status_clean(r.get("Estado proveedor"))
            s_qty = _safe_float(r.get("Enviado"), qty_ordered)

            v_qty = _safe_float(r.get("Recibido"), 0.0)

            # "expected" is supplier_qty when supplier has confirmed, else ordered
            qty_expected = s_qty if s_qty is not None else qty_ordered

            # 1) supplier_short ticket (created when supplier indicates issue or sent != ordered)
            if is_supplier:
                if s_status != "ok" or abs(s_qty - qty_ordered) > 1e-9:
                    note = f"Supplier: {s_status.upper()} · {s_qty:g}/{qty_ordered:g}"
                    reason = _safe_str(r.get("Motivo"))
                    if reason:
                        note += f" · {reason}"
                    _upsert_ticket_open(
                        venue_id=int(order.venue_id),
                        order_id=int(order.id),
                        provider_name=provider_name,
                        order_line_id=line_id,
                        kind="supplier_short",
                        product_name=name,
                        unit=unit,
                        qty_ordered=qty_ordered,
                        qty_expected=qty_expected,
                        qty_received=v_qty,
                        note=note,
                    )

            # 2) delivery_mismatch ticket (venue confirms mismatch vs expected)
            if is_venue:
                if abs(v_qty - qty_expected) > 1e-9 or s_status != "ok":
                    note = f"Delivery mismatch · expected {qty_expected:g} · received {v_qty:g}"
                    if s_status != "ok":
                        note = f"Supplier {s_status.upper()} · expected {qty_expected:g} · received {v_qty:g}"
                    _upsert_ticket_open(
                        venue_id=int(order.venue_id),
                        order_id=int(order.id),
                        provider_name=provider_name,
                        order_line_id=line_id,
                        kind="delivery_mismatch",
                        product_name=name,
                        unit=unit,
                        qty_ordered=qty_ordered,
                        qty_expected=qty_expected,
                        qty_received=v_qty,
                        note=note,
                    )

                # 3) invoice_mismatch ticket (only if invoice fields filled)
                if invoice_mode:
                    inv_no = _safe_str(r.get("Factura Nº"))
                    inv_qty_raw = r.get("Facturado qty")
                    inv_price_raw = r.get("Precio facturado")

                    inv_qty = None
                    if inv_qty_raw is not None and inv_qty_raw is not pd.NA and _safe_str(inv_qty_raw) != "":
                        inv_qty = _safe_float(inv_qty_raw, default=0.0)

                    inv_price = None
                    if inv_price_raw is not None and inv_price_raw is not pd.NA and _safe_str(inv_price_raw) != "":
                        inv_price = _safe_float(inv_price_raw, default=0.0)

                    # expected unit price: try product catalog price if exists
                    p = None
                    pid = r.get("product_id")
                    if pid is not None and pid is not pd.NA:
                        try:
                            p = products_by_id.get(int(pid))
                        except Exception:
                            p = None
                    expected_price = None
                    if p is not None:
                        expected_price = _safe_float(getattr(p, "price", None), default=0.0)
                        if expected_price <= 0:
                            expected_price = None

                    mismatch = False
                    reasons = []

                    if inv_qty is not None and abs(inv_qty - v_qty) > 1e-9:
                        mismatch = True
                        reasons.append(f"qty invoice {inv_qty:g} vs received {v_qty:g}")

                    if inv_price is not None and expected_price is not None and abs(inv_price - expected_price) > 1e-9:
                        mismatch = True
                        reasons.append(f"price invoice {inv_price:g} vs expected {expected_price:g}")

                    # If invoice mode is active and any invoice fields entered, allow mismatch ticket creation
                    any_invoice_entered = bool(inv_no or inv_qty is not None or inv_price is not None)

                    if any_invoice_entered and mismatch:
                        note = "Invoice mismatch · " + " · ".join(reasons)
                        _upsert_ticket_open(
                            venue_id=int(order.venue_id),
                            order_id=int(order.id),
                            provider_name=provider_name,
                            order_line_id=line_id,
                            kind="invoice_mismatch",
                            product_name=name,
                            unit=unit,
                            qty_ordered=qty_ordered,
                            qty_expected=qty_expected,
                            qty_received=v_qty,
                            note=note,
                            invoice_number=inv_no or None,
                            qty_invoiced=inv_qty,
                            unit_price_expected=expected_price,
                            unit_price_invoiced=inv_price,
                        )

        st.success("Guardado ✅ (incidencias actualizadas)")
        st.session_state.pop("seg_df_override", None)
        st.rerun()

    # -------------------------------------------------------------------------
    # Ticket board (venue only) - resolve across days
    # -------------------------------------------------------------------------
    if is_venue:
        st.markdown("---")
        st.subheader("🎫 Incidencias (tickets)")

        tickets = _load_tickets(order_id=int(order.id), provider_name=provider_name)

        open_t = [t for t in tickets if (t.state or "open") == "open"]
        res_t = [t for t in tickets if (t.state or "open") == "resolved"]

        k1, k2, k3 = st.columns([1.2, 1.2, 2.6], vertical_alignment="center")
        with k1:
            st.metric("Abiertas", len(open_t))
        with k2:
            st.metric("Resueltas", len(res_t))
        with k3:
            show_resolved = st.toggle("Mostrar resueltas", value=False, key="seg_show_resolved")

        show = tickets if show_resolved else open_t

        if not show:
            st.success("🟢 No hay incidencias abiertas para este proveedor.")
            return

        # Render tickets in a compact editor-like list
        for t in show:
            state_chip = "🟠 Abierta" if t.state == "open" else "✅ Resuelta"
            with st.container(border=True):
                top = st.columns([3.4, 1.0, 1.6], vertical_alignment="center")
                with top[0]:
                    st.markdown(f"**{t.product_name}**  ·  `{t.kind}`")
                    if t.note:
                        st.caption(t.note)
                with top[1]:
                    st.caption(state_chip)
                with top[2]:
                    st.caption(f"Act.: {t.updated_at.strftime('%Y-%m-%d %H:%M')}")

                # Compact facts line
                facts = f"Pedido {t.qty_ordered:g} · Esperado {t.qty_expected:g} · Recibido {t.qty_received:g}"
                if t.kind == "invoice_mismatch":
                    inv_bits = []
                    if t.invoice_number:
                        inv_bits.append(f"Factura {t.invoice_number}")
                    if t.qty_invoiced is not None:
                        inv_bits.append(f"Facturado qty {t.qty_invoiced:g}")
                    if t.unit_price_invoiced is not None:
                        inv_bits.append(f"Precio fact. {t.unit_price_invoiced:g}")
                    if t.unit_price_expected is not None:
                        inv_bits.append(f"Precio esp. {t.unit_price_expected:g}")
                    if inv_bits:
                        facts += " · " + " · ".join(inv_bits)
                st.caption(facts)

                if t.state == "open":
                    res_note = st.text_area(
                        "Nota de resolución",
                        value=_safe_str(t.resolution_note),
                        height=80,
                        key=f"res_note_{t.id}",
                        placeholder="Ej: nota de crédito emitida, reposición mañana, ajuste en factura…",
                    )
                    b1, b2 = st.columns([1.2, 3.8], vertical_alignment="center")
                    with b1:
                        if st.button("✅ Resolver", key=f"resolve_{t.id}", type="primary", use_container_width=True):
                            _resolve_ticket(ticket_id=int(t.id), resolution_note=res_note)
                            st.success("Ticket resuelto ✅")
                            st.rerun()
                    with b2:
                        st.caption("Resolver = queda en histórico, pero sigue visible si activas “Mostrar resueltas”.")
                else:
                    if t.resolution_note:
                        st.caption(f"✅ Resolución: {t.resolution_note}")


if __name__ == "__main__":
    seguimiento_page()
