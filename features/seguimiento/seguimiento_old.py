from __future__ import annotations

"""seguimiento_clear.py

Public tracking link UI (supplier + venue).

Goals
- extremely clear: "where we are" + "what you need to do now".
- supplier never sees irrelevant actions.
- supplier actions are limited to:
  (1) confirm delivery capability (full/partial/none)
  (2) when asked, choose resolution for invoice discrepancy

All closures happen on the venue side.
"""

from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple
import json
import streamlit as st
from sqlmodel import select
import re
from core.db import get_session
from core.public_links import ROLE_SUPPLIER, ROLE_VENUE, norm_provider, norm_role, verify_link
from domain.models import (
    Order,
    OrderLine,
    Product,
    Provider,
    OrderWorkflow,
    OrderWorkflowEvent,
    SeguimientoTicket,
    ProviderLineFollowUp,
    ProviderReceipt,
)

# -----------------------------
# Delivery schedule helpers
# -----------------------------

_TIME_RANGE_RE = re.compile(r"^\d{2}:\d{2}-\d{2}:\d{2}$")


def _parse_delivery_schedule_json(raw: str) -> Dict[str, List[str]]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    out: Dict[str, List[str]] = {}
    for day, slots in (data or {}).items():
        d = (str(day or "").strip().lower())
        if d not in {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}:
            continue
        clean: List[str] = []
        for s in (slots or []):
            s = str(s or "").strip()
            if _TIME_RANGE_RE.match(s) and s not in clean:
                clean.append(s)
        if clean:
            out[d] = clean
    return out


def _day_key_for_date(dt: datetime) -> str:
    # Monday=0 ... Sunday=6
    keys = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    return keys[int(dt.weekday())]

# -----------------------------
# UI
# -----------------------------

def _css() -> None:
    st.markdown(
        """
<style>
:root{
  --bg:#0b1220;
  --card:#ffffff;
  --text:#0f172a;
  --muted:#64748b;
  --line:#e2e8f0;
  --primary:#2563eb;
  --ok:#16a34a;
  --warn:#f59e0b;
  --bad:#ef4444;
}
.main{max-width:900px;margin:0 auto;padding:1rem 0.25rem;}
.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:16px;margin:12px 0;box-shadow:0 6px 18px rgba(2,6,23,.06);}
.h1{font-size:1.25rem;font-weight:900;margin:0 0 6px 0;color:var(--text);}
.muted{color:var(--muted);font-size:.92rem;}
.badge{display:inline-flex;align-items:center;gap:6px;padding:6px 10px;border-radius:999px;border:1px solid var(--line);font-weight:700;font-size:.82rem;}
.badge.ok{background:#ecfdf5;border-color:#bbf7d0;color:#166534;}
.badge.warn{background:#fffbeb;border-color:#fde68a;color:#92400e;}
.badge.bad{background:#fef2f2;border-color:#fecaca;color:#991b1b;}
.badge.info{background:#eff6ff;border-color:#bfdbfe;color:#1e3a8a;}
.stepper{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;}
.step{padding:6px 10px;border-radius:999px;border:1px solid var(--line);font-weight:800;font-size:.78rem;color:var(--muted);background:#f8fafc;}
.step.active{background:#eff6ff;border-color:#93c5fd;color:#1e3a8a;}
.step.done{background:#ecfdf5;border-color:#86efac;color:#166534;}
.hr{height:1px;background:var(--line);margin:12px 0;}
.pillrow{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;}
.pill{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;border:1px solid var(--line);background:#f8fafc;font-weight:800;font-size:.78rem;color:var(--text);}

</style>
""",
        unsafe_allow_html=True,
    )


def _badge(text: str, kind: str = "info") -> str:
    return f"<span class='badge {kind}'>{text}</span>"


def _now() -> datetime:
    return datetime.utcnow()


FLOW_STEPS = [
    ("ORDER_SENT", "1. Order sent"),
    ("SUPPLIER_CONFIRMED", "2. Supplier confirmed"),
    ("RECEIVED", "3. Venue received"),
    ("RESOLUTION", "4. Resolve issues"),
    ("CLOSED", "5. Closed"),
]


def _step_index(state: str) -> int:
    s = (state or "").upper()
    if s == "CLOSED":
        return 4
    if s in ("SUPPLIER_CREDIT_NOTE_ISSUED", "SUPPLEMENTARY_DELIVERY_SENT"):
        return 3
    if s in ("WAITING_SUPPLIER_ACTION", "INVOICE_DISCREPANCY"):
        return 3
    if s in ("MATCHED_WITH_INVOICE", "RECEIVED", "PARTIALLY_RECEIVED", "NOT_RECEIVED"):
        return 2
    if s.startswith("SUPPLIER_CONFIRMED"):
        return 1
    return 0


def _render_stepper(state: str) -> None:
    idx = _step_index(state)
    chips = []
    for i, (_, label) in enumerate(FLOW_STEPS):
        cls = "step"
        if i < idx:
            cls += " done"
        elif i == idx:
            cls += " active"
        chips.append(f"<span class='{cls}'>{label}</span>")
    st.markdown("<div class='stepper'>" + "".join(chips) + "</div>", unsafe_allow_html=True)


# -----------------------------
# Data
# -----------------------------

def load_context(order_id: int, provider_name: str, raw_role: str, token: str) -> Dict[str, Any]:
    role = norm_role(raw_role)
    provider_name = norm_provider(provider_name)

    if not verify_link(order_id=order_id, provider_name=provider_name, role=role, sig=token):
        raise ValueError("Invalid or expired link")


    with get_session() as s:
        order = s.exec(select(Order).where(Order.id == order_id)).first()
        if not order:
            raise ValueError("Order not found")

        wf = s.exec(
            select(OrderWorkflow).where(
                OrderWorkflow.order_id == order_id,
                OrderWorkflow.provider_name == provider_name,
            )
        ).first()
        if not wf:
            raise ValueError("Workflow not found")

        lines = list(
            s.exec(
                select(OrderLine)
                .join(Product, Product.id == OrderLine.product_id, isouter=True)
                .where(OrderLine.order_id == order_id)
                .where(
                    (Product.provider_name == provider_name) | (OrderLine.provider == provider_name)
                )
            ).all()
        )

        product_ids = [l.product_id for l in lines if l.product_id]
        products = {}
        if product_ids:
            ps = list(s.exec(select(Product).where(Product.id.in_(product_ids))).all())
            products = {p.id: p for p in ps}

        # Load all incidences for this supplier (invoice discrepancy, damaged/wrong, operational missing, etc.)
        tickets = list(
            s.exec(
                select(SeguimientoTicket).where(
                    SeguimientoTicket.order_id == order_id,
                    SeguimientoTicket.provider_name == provider_name,
                )
            ).all()
        )

        receipt = s.exec(
            select(ProviderReceipt).where(
                ProviderReceipt.order_id == order_id,
                ProviderReceipt.provider_name == provider_name,
            )
        ).first()

        # Provider directory entry (optional). Used for delivery schedule suggestions.
        provider_obj = s.exec(
            select(Provider).where(
                Provider.venue_id == order.venue_id,
                Provider.name == provider_name,
            )
        ).first()

        followups = list(
            s.exec(
                select(ProviderLineFollowUp).where(
                    ProviderLineFollowUp.order_id == order_id,
                    ProviderLineFollowUp.provider_name == provider_name,
                )
            ).all()
        )

    return {
        "role": role,
        "provider_name": provider_name,
        "provider_obj": provider_obj,
        "order": order,
        "workflow": wf,
        "lines": lines,
        "products": products,
        "tickets": tickets,
        "receipt": receipt,
        "followups": {fu.order_line_id: fu for fu in followups},
    }


# -----------------------------
# Mutations
# -----------------------------

def _add_event(s, venue_id: int, order_id: int, provider_name: str, frm: str, to: str, actor_role: str, actor: str, note: str = ""):
    s.add(
        OrderWorkflowEvent(
            venue_id=venue_id,
            order_id=order_id,
            provider_name=provider_name,
            from_state=frm,
            to_state=to,
            actor_role=actor_role,
            actor=actor,
            at=_now(),
            note=note,
        )
    )


def save_supplier_confirmation(ctx: Dict[str, Any], declaration: str, comment: str) -> None:
    wf: OrderWorkflow = ctx["workflow"]
    order: Order = ctx["order"]

    decl_map = {
        "full": "SUPPLIER_CONFIRMED_FULL",
        "partial": "SUPPLIER_CONFIRMED_PARTIAL",
        "none": "SUPPLIER_CONFIRMED_NONE",
    }
    to_state = decl_map[declaration]

    with get_session() as s:
        wf2 = s.exec(select(OrderWorkflow).where(OrderWorkflow.id == wf.id)).first()
        prev = wf2.state
        wf2.state = to_state
        wf2.updated_at = _now()
        wf2.updated_by_role = ROLE_SUPPLIER
        wf2.updated_by = "supplier"
        wf2.note = comment or ""
        s.add(wf2)

        receipt = s.exec(
            select(ProviderReceipt).where(
                ProviderReceipt.order_id == order.id,
                ProviderReceipt.provider_name == wf2.provider_name,
            )
        ).first()
        if not receipt:
            receipt = ProviderReceipt(
                venue_id=order.venue_id,
                order_id=order.id,
                provider_name=wf2.provider_name,
            )
        receipt.supplier_declaration = declaration
        receipt.supplier_declared_at = _now()
        receipt.supplier_declared_by = "supplier"
        receipt.supplier_declared_comment = comment or None
        receipt.updated_at = _now()
        receipt.updated_by = "supplier"
        s.add(receipt)

        _add_event(s, order.venue_id, order.id, wf2.provider_name, prev, to_state, ROLE_SUPPLIER, "supplier", comment or "")
        s.commit()


def save_supplier_invoice_number(ctx: Dict[str, Any], invoice_number: str) -> None:
    """Supplier fills the real invoice number (shown in venue Receive + Incidences)."""
    wf: OrderWorkflow = ctx["workflow"]
    order: Order = ctx["order"]
    inv = (invoice_number or "").strip()

    with get_session() as s:
        rec = s.exec(
            select(ProviderReceipt).where(
                ProviderReceipt.order_id == order.id,
                ProviderReceipt.provider_name == wf.provider_name,
            )
        ).first()

        if not rec:
            rec = ProviderReceipt(
                venue_id=order.venue_id,
                order_id=order.id,
                provider_name=wf.provider_name,
            )

        rec.invoice_number = inv or None
        rec.invoice_number_set_at = _now()
        rec.invoice_number_set_by = "supplier"
        rec.updated_at = _now()
        rec.updated_by = "supplier"
        s.add(rec)

        # Audit trail (no state change)
        _add_event(
            s,
            order.venue_id,
            order.id,
            wf.provider_name,
            wf.state,
            wf.state,
            ROLE_SUPPLIER,
            "supplier",
            f"Invoice number set: {inv}" if inv else "Invoice number cleared",
        )
        s.commit()

def _derive_declaration_from_lines(line_payload: Dict[int, Dict[str, Any]]) -> str:
    """Derive overall declaration (full/partial/none) from per-line choices."""
    choices = [((v or {}).get("choice") or "full") for v in (line_payload or {}).values()]
    if choices and all(c == "full" for c in choices):
        return "full"
    if choices and all(c == "none" for c in choices):
        return "none"
    return "partial"


def save_supplier_confirmation_per_line(
    ctx: Dict[str, Any],
    line_payload: Dict[int, Dict[str, Any]],
    comment: str,
) -> None:
    """Persist supplier confirmation per order line.

    - Upserts ProviderLineFollowUp per order line (supplier_status + supplier_qty)
    - Updates OrderWorkflow to SUPPLIER_CONFIRMED_* based on aggregate
    - Updates ProviderReceipt.supplier_declaration
    """
    wf: OrderWorkflow = ctx["workflow"]
    order: Order = ctx["order"]

    declaration = _derive_declaration_from_lines(line_payload)

    decl_map = {
        "full": "SUPPLIER_CONFIRMED_FULL",
        "partial": "SUPPLIER_CONFIRMED_PARTIAL",
        "none": "SUPPLIER_CONFIRMED_NONE",
    }
    to_state = decl_map[declaration]

    status_map = {
        "full": "ok",
        "partial": "partial",
        "none": "missing",
    }

    with get_session() as s:
        # --- workflow ---
        wf2 = s.exec(select(OrderWorkflow).where(OrderWorkflow.id == wf.id)).first()
        prev = wf2.state
        wf2.state = to_state
        wf2.updated_at = _now()
        wf2.updated_by_role = ROLE_SUPPLIER
        wf2.updated_by = "supplier"
        wf2.note = comment or ""
        s.add(wf2)

        # --- receipt ---
        receipt = s.exec(
            select(ProviderReceipt).where(
                ProviderReceipt.order_id == order.id,
                ProviderReceipt.provider_name == wf2.provider_name,
            )
        ).first()
        if not receipt:
            receipt = ProviderReceipt(
                venue_id=order.venue_id,
                order_id=order.id,
                provider_name=wf2.provider_name,
            )
        receipt.supplier_declaration = declaration
        receipt.supplier_declared_at = _now()
        receipt.supplier_declared_by = "supplier"
        receipt.supplier_declared_comment = comment or None
        receipt.updated_at = _now()
        receipt.updated_by = "supplier"
        s.add(receipt)

        # --- per-line followups ---
        for line in ctx.get("lines", []) or []:
            if not getattr(line, "id", None):
                continue
            lid = int(line.id)
            payload = (line_payload or {}).get(lid) or {}
            choice = (payload.get("choice") or "full").strip().lower()
            if choice not in ("full", "partial", "none"):
                choice = "full"

            ordered = float(getattr(line, "quantity", 0) or 0)
            if choice == "full":
                qty_send = ordered
            elif choice == "none":
                qty_send = 0.0
            else:
                qty_send = float(payload.get("qty") or 0.0)
                qty_send = max(0.0, min(ordered, qty_send))

            fu = s.exec(
                select(ProviderLineFollowUp).where(
                    ProviderLineFollowUp.order_id == order.id,
                    ProviderLineFollowUp.provider_name == wf2.provider_name,
                    ProviderLineFollowUp.order_line_id == lid,
                )
            ).first()

            if not fu:
                fu = ProviderLineFollowUp(
                    venue_id=order.venue_id,
                    order_id=order.id,
                    provider_name=wf2.provider_name,
                    order_line_id=lid,
                    qty_ordered=ordered,
                )
            fu.qty_ordered = ordered
            fu.supplier_status = status_map[choice]   # ok/partial/missing
            fu.supplier_qty = qty_send
            fu.updated_at = _now()
            fu.updated_by = "supplier"
            s.add(fu)

        _add_event(s, order.venue_id, order.id, wf2.provider_name, prev, to_state, ROLE_SUPPLIER, "supplier", comment or "")
        s.commit()

def save_supplier_resolution(
    ctx: Dict[str, Any],
    resolution: str,
    # ref: str,
    comment: str,
    credit_note_invoice_number: Optional[str] = None,
    redelivery_eta: Optional[str] = None,
) -> None:
    wf: OrderWorkflow = ctx["workflow"]
    order: Order = ctx["order"]

    res_map = {
        "credit_note": "SUPPLIER_CREDIT_NOTE_ISSUED",
        "supplementary_delivery": "SUPPLEMENTARY_DELIVERY_SENT",
    }
    to_state = res_map[resolution]

    # Create a structured note that the venue can easily verify.
    meta_parts: List[str] = [resolution]
    # if ref:
    #     meta_parts.append(f"ref={ref}")
    if credit_note_invoice_number:
        meta_parts.append(f"credit_note_invoice={credit_note_invoice_number}")
    if redelivery_eta:
        meta_parts.append(f"eta={redelivery_eta}")
    meta = " | ".join(meta_parts)

    with get_session() as s:
        wf2 = s.exec(select(OrderWorkflow).where(OrderWorkflow.id == wf.id)).first()
        prev = wf2.state
        wf2.state = to_state
        wf2.updated_at = _now()
        wf2.updated_by_role = ROLE_SUPPLIER
        wf2.updated_by = "supplier"
        wf2.note = meta
        s.add(wf2)

        tickets = list(
            s.exec(
                select(SeguimientoTicket).where(
                    SeguimientoTicket.order_id == order.id,
                    SeguimientoTicket.provider_name == wf2.provider_name,
                )
            ).all()
        )
        for t in tickets:
            if t.state in ("open", "SUPPLIER_ACTION_DONE"):
                t.state = "SUPPLIER_ACTION_DONE"
                note = f"[SUPPLIER] {meta}"
                if comment:
                    note += f"\n{comment}"
                t.resolution_note = note
                t.updated_at = _now()
                s.add(t)

        _add_event(
            s,
            order.venue_id,
            order.id,
            wf2.provider_name,
            prev,
            to_state,
            ROLE_SUPPLIER,
            "supplier",
            meta + (f"\n{comment}" if comment else ""),
        )
        s.commit()


# -----------------------------
# Screens
# -----------------------------

def _header(ctx: Dict[str, Any]) -> None:
    order: Order = ctx["order"]
    provider = ctx["provider_name"]
    receipt: Optional[ProviderReceipt] = ctx.get("receipt")
    inv = (getattr(receipt, "invoice_number", None) or "").strip() or "—"

    lines: List[OrderLine] = list(ctx.get("lines") or [])
    products: Dict[int, Product] = ctx.get("products") or {}
    tickets: List[SeguimientoTicket] = list(ctx.get("tickets") or [])

    def _pname(line: OrderLine) -> str:
        p = products.get(getattr(line, "product_id", None)) if getattr(line, "product_id", None) else None
        return (getattr(p, "name", None) or getattr(line, "spoken_name", None) or "Product").strip()

    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='card'>"
        f"<div class='h1'>Order #{int(order.id)} · {provider} · Invoice: {inv}</div>"
        f"<div class='muted'>Public tracking link</div>"
        f"<div style='margin-top:10px'>{_badge(ctx['role'].upper(), 'info')}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # # Show a clear summary of what has an issue (supplier needs to know exactly what/why)
    # open_like = [t for t in tickets if (t.state or "").lower() != "resolved"]
    # if open_like:
    #     with st.container(border=True):
    #         st.markdown("### ⚠️ Items with issues")
    #         for t in open_like:
    #             why = (t.kind or "").replace("_", " ")
    #             extra = (t.note or "").strip()
    #             txt = f"- **{t.product_name}** · {why}"
    #             if extra:
    #                 txt += f"  \n  <span class='muted'>{extra}</span>"
    #             st.markdown(txt, unsafe_allow_html=True)

    # Full order overview (always useful for supplier to double-check)
    if lines:
        with st.expander("📦 Full order (all items)", True):
            for line in sorted(lines, key=lambda l: _pname(l).lower()):
                p = products.get(getattr(line, "product_id", None)) if getattr(line, "product_id", None) else None
                name = (getattr(p, "name", None) or getattr(line, "spoken_name", None) or "Product").strip()
                unit = (getattr(p, "unit", None) or getattr(line, "unit", None) or "unit")
                ordered = float(getattr(line, "quantity", 0) or 0)
                st.markdown(f"- **{name}** · {ordered:g} {unit}")


def _render_supplier_confirmation(ctx: Dict[str, Any]) -> None:
    st.markdown(
        "<div class='card'>"
        "<div class='h1'>Step 2 · Confirm what you will deliver</div>"
        "<div class='muted'>For each product, choose Full / Partial / None. If Partial, specify how many you will send.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    lines: List[OrderLine] = list(ctx.get("lines") or [])
    products: Dict[int, Product] = ctx.get("products") or {}
    followups: Dict[int, ProviderLineFollowUp] = ctx.get("followups") or {}
    receipt: Optional[ProviderReceipt] = ctx.get("receipt")

    if not lines:
        st.info("No products found for this supplier.")
        return

    # Invoice number (supplier can fill). Avoid Streamlit "default+session_state" warnings by
    # initializing session_state once and not passing value=.
    st.markdown("#### Invoice number")
    inv_key = f"supplier_invoice_{int(ctx['order'].id)}_{ctx['provider_name']}"
    if inv_key not in st.session_state:
        st.session_state[inv_key] = (getattr(receipt, "invoice_number", None) or "")

    inv_input = st.text_input(
        "Invoice number (as written on the invoice)",
        key=inv_key,
        placeholder="e.g. INV-2026-00123",
    )
    if st.button("Save invoice number", use_container_width=True, key=f"save_inv_{inv_key}"):
        save_supplier_invoice_number(ctx, inv_input)
        st.success("Invoice number saved")
        st.rerun()

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    payload: Dict[int, Dict[str, Any]] = {}

    def _product_name(line: OrderLine) -> str:
        p = products.get(line.product_id) if getattr(line, "product_id", None) else None
        return (p.name if p else (line.spoken_name or "")).strip()

    for line in sorted(lines, key=lambda l: _product_name(l).lower()):
        if not getattr(line, "id", None):
            continue

        lid = int(line.id)
        p = products.get(line.product_id) if getattr(line, "product_id", None) else None

        name = (p.name if p else None) or (line.spoken_name or "Product")
        unit = (getattr(p, "unit", None) if p else None) or (getattr(line, "unit", None) or "unit")
        ordered = float(getattr(line, "quantity", 0) or 0)

        # preload previous supplier response (if any)
        fu = followups.get(lid)
        default_choice = "full"
        default_qty = ordered
        if fu:
            if (fu.supplier_status or "").lower() == "partial":
                default_choice = "partial"
                default_qty = float(fu.supplier_qty or 0.0)
            elif (fu.supplier_status or "").lower() == "missing":
                default_choice = "none"
                default_qty = 0.0
            else:
                default_choice = "full"
                default_qty = float(fu.supplier_qty) if fu.supplier_qty is not None else ordered

        with st.container(border=True):
            st.markdown(f"**{name}**")
            st.markdown(
                f"<div class='pillrow'><span class='pill'>Ordered: {ordered:g} {unit}</span></div>",
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns([1.1, 1])
            with c1:
                choice = st.selectbox(
                    "Status",
                    ["full", "partial", "none"],
                    index={"full": 0, "partial": 1, "none": 2}[default_choice],
                    format_func=lambda x: {
                        "full": "✅ Full",
                        "partial": "🟡 Partial",
                        "none": "❌ None",
                    }[x],
                    key=f"sup_conf_choice_{lid}",
                    label_visibility="collapsed",
                )

            qty_send = ordered if choice == "full" else (0.0 if choice == "none" else None)

            with c2:
                if choice == "partial":
                    qty_send = st.number_input(
                        "Qty to send",
                        min_value=1.0,
                        max_value=float(ordered) - 1.0,
                        value=float(ordered) - 1.0,
                        step=1.0,
                        key=f"sup_conf_qty_{lid}",
                        label_visibility="collapsed",
                    )
                    st.caption(f"Will send: {qty_send:g} {unit}")
                else:
                    st.caption(f"Will send: {qty_send:g} {unit}")

        payload[lid] = {"choice": choice, "qty": float(qty_send or 0.0)}

    comment = st.text_area("Comment (optional)")

    if st.button("Save confirmation", type="primary", use_container_width=True):
        save_supplier_confirmation_per_line(ctx, payload, comment)
        st.success("Saved")
        st.rerun()


def _render_supplier_resolution(ctx: Dict[str, Any]) -> None:
    tickets: List[SeguimientoTicket] = list(ctx.get("tickets") or [])
    open_t = [
        t for t in tickets
        if (t.state or "").lower() != "resolved"
        and (getattr(t, "kind", "") or "").lower() != "operational_missing"
    ]

    receipt: Optional[ProviderReceipt] = ctx.get("receipt")
    inv = (getattr(receipt, "invoice_number", None) or "").strip() or "—"

    st.markdown(
        "<div class='card'>"
        f"<div class='h1'>Step 4 · Resolve issues (Invoice: {inv})</div>"
        "<div class='muted'>Choose one option. The venue will verify and then close the incident.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    
    if not open_t:
        st.info("No supplier action is required for this order right now.")
        _render_readonly(ctx)
        return

    
    if open_t:

        lines: List[OrderLine] = list(ctx.get("lines") or [])
        products: Dict[int, Product] = dict(ctx.get("products") or {})
        followups: Dict[int, ProviderLineFollowUp] = dict(ctx.get("followups") or {})
        line_by_id = {int(l.id): l for l in lines if getattr(l, "id", None) is not None}

        def _pname(line: Optional[OrderLine], fallback: str) -> str:
            if (fallback or "").strip():
                return (fallback or "").strip()
            if not line:
                return "Product"
            p = products.get(getattr(line, "product_id", None)) if getattr(line, "product_id", None) else None
            return (getattr(p, "name", None) or getattr(line, "spoken_name", None) or "Product").strip()

        def _punit(line: Optional[OrderLine], fallback: str) -> str:
            if (fallback or "").strip():
                return (fallback or "").strip()
            if not line:
                return "unit"
            p = products.get(getattr(line, "product_id", None)) if getattr(line, "product_id", None) else None
            return (getattr(p, "unit", None) or getattr(line, "unit", None) or "unit").strip()

        def _expected_qty(line: Optional[OrderLine], fu: Optional[ProviderLineFollowUp]) -> float:
            """Expected (invoiced) qty.
            If supplier confirmed partial/missing we use that; otherwise fall back to ordered.
            """
            ordered = float(getattr(line, "quantity", 0) or 0) if line else 0.0
            if not fu:
                return ordered
            stt = (getattr(fu, "supplier_status", "") or "").lower()
            sq = getattr(fu, "supplier_qty", None)
            if stt == "missing":
                return 0.0
            if stt == "partial":
                return float(sq or 0.0)
            if stt == "ok":
                return float(sq) if sq is not None else ordered
            return ordered

        st.markdown("**Products with issues (what you are resolving):**")
        for t in open_t:
            lid = int(getattr(t, "order_line_id", 0) or 0)
            line = line_by_id.get(lid)
            fu = followups.get(lid)

            name = _pname(line, getattr(t, "product_name", "") or "")
            unit = _punit(line, getattr(t, "unit", "") or "")

            expected = _expected_qty(line, fu)
            issue_qty = float(getattr(t, "qty_invoiced", 0) or 0)
            # Prefer the venue-entered received qty; if it's missing, fall back to expected-issue.
            if fu and getattr(fu, "venue_qty", None) is not None:
                received = float(getattr(fu, "venue_qty", 0) or 0)
            else:
                received = max(0.0, float(expected) - float(issue_qty))

            kind = (getattr(t, "kind", "") or "").replace("_", " ")
            if (getattr(t, "kind", "") or "").lower() in {"invoice_discrepancy", "operational_missing"}:
                st.markdown(
                    f"- **{name}** · {kind} · invoiced **{expected:g}** · received **{received:g}** · missing **{issue_qty:g} {unit}**"
                )
            elif (getattr(t, "kind", "") or "").lower() == "damaged_wrong":
                st.markdown(
                    f"- **{name}** · damaged / wrong · invoiced **{expected:g}** · received **{received:g}** · damaged **{issue_qty:g} {unit}**"
                )
            else:
                st.markdown(
                    f"- **{name}** · {kind} · invoiced **{expected:g}** · received **{received:g}** · issue **{issue_qty:g} {unit}**"
                )

            if (getattr(t, "note", None) or "").strip():
                st.caption((getattr(t, "note", None) or "").strip())

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    resolution = st.radio(
        "Resolution",
        ["credit_note", "supplementary_delivery"],
        format_func=lambda x: {
            "credit_note": "📝 Issue credit note",
            "supplementary_delivery": "🚚 Send missing items in another delivery",
        }[x],
    )

    # ref = st.text_input("Reference (optional)", placeholder="e.g. Delivery #123")

    credit_note_no = ""
    eta_date = None
    eta_slot = ""
    if resolution == "credit_note":
        credit_note_no = st.text_input(
            "Credit note invoice number (required)",
            placeholder="e.g. CN-2026-001",
        )
        st.caption("Please enter the credit note number so the venue/accountant can match the document.")
    else:
        # Re-delivery: date + time window in hours (e.g. 08:00-14:00)
        eta_date = st.date_input("Expected delivery date (required)")

        # Suggest time slots from the provider directory if available
        provider_obj: Optional[Provider] = ctx.get("provider_obj")
        schedule = _parse_delivery_schedule_json(getattr(provider_obj, "delivery_schedule_json", None) or "") if provider_obj else {}

        preset_slots = ["08:00-14:00", "16:00-20:00"]
        weekday_key = _day_key_for_date(datetime.combine(eta_date, datetime.min.time())) if eta_date else ""
        suggested = schedule.get(weekday_key, []) if weekday_key else []
        slot_options = suggested or preset_slots

        eta_slot = st.selectbox(
            "Delivery window (required)",
            options=slot_options,
            help="Choose the expected delivery time window (hours).",
        )

        custom_slot = st.text_input(
            "Custom window (optional)",
            placeholder="e.g. 06:30-10:30",
            help="Format must be HH:MM-HH:MM",
        ).strip()
        if custom_slot:
            if _TIME_RANGE_RE.match(custom_slot):
                eta_slot = custom_slot
                st.caption(f"✅ Using custom window: {custom_slot}")
            else:
                st.error("Invalid time window. Use HH:MM-HH:MM (e.g. 06:30-10:30).")

        st.caption("Please tell us when the missing items will arrive (date + time window).")

    comment = st.text_area("Comment (optional)")

    if st.button("Submit resolution", type="primary", use_container_width=True):
        if resolution == "credit_note" and not (credit_note_no or "").strip():
            st.error("Please enter the credit note invoice number.")
            return
        if resolution == "supplementary_delivery" and not eta_date:
            st.error("Please select an expected delivery date.")
            return
        if resolution == "supplementary_delivery" and not (eta_slot or "").strip():
            st.error("Please select an expected delivery time window.")
            return

        save_supplier_resolution(
            ctx,
            resolution=resolution,
            # ref=ref,
            comment=comment,
            credit_note_invoice_number=(credit_note_no.strip() or None),
            redelivery_eta=(f"{eta_date.isoformat()} {eta_slot}" if eta_date and eta_slot else None),
        )
        st.success("Submitted. Waiting for venue verification.")
        st.rerun()


def _render_readonly(ctx: Dict[str, Any]) -> None:
    wf: OrderWorkflow = ctx["workflow"]
    state = wf.state

    st.markdown(
        f"<div class='card'>"
        f"<div class='h1'>Current status</div>"
        f"<div style='margin-top:8px'>{_badge(state, 'info')}</div>"
        f"<div class='hr'></div>"
        f"<div class='muted'>No action required right now.</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def seguimiento_app(order_id: int, provider_name: str, role: str, token: str) -> None:
    _css()

    try:
        ctx = load_context(order_id, provider_name, role, token)
    except Exception as e:
        st.error(str(e))
        return

    _header(ctx)

    wf: OrderWorkflow = ctx["workflow"]
    # st.markdown(
    #     f"<div class='card'>"
    #     f"<div class='h1'>Progress</div>"
    #     f"<div class='muted'>One universal flow: sent → confirmed → received → resolved → closed</div>"
    #     f"</div>",
    #     unsafe_allow_html=True,
    # )
    _render_stepper(wf.state)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    if ctx["role"] == ROLE_SUPPLIER:
        # Supplier actions depend on state
        if wf.state == "ORDER_SENT":
            _render_supplier_confirmation(ctx)
        elif wf.state in ("WAITING_SUPPLIER_ACTION", "INVOICE_DISCREPANCY"):
            _render_supplier_resolution(ctx)
        elif wf.state in ("SUPPLIER_CREDIT_NOTE_ISSUED", "SUPPLEMENTARY_DELIVERY_SENT"):
            st.info("✅ Thanks. Waiting for venue verification.")
            _render_readonly(ctx)
        else:
            _render_readonly(ctx)

    else:
        # Venue view is read-only in public link
        _render_readonly(ctx)

    st.markdown("</div>", unsafe_allow_html=True)


# Backwards compatible entrypoint used by your router

def run():
    params = st.query_params
    order_id = int(params.get("order_id", "0"))
    provider = params.get("provider", "")
    role = params.get("role", ROLE_SUPPLIER)
    token = params.get("token", "") or params.get("sig", "")
    seguimiento_app(order_id, provider, role, token)
