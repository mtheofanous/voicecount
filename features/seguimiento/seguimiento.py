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
import html
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
.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:16px;margin:12px 0;box-shadow:0 4px 12px rgba(2,6,23,.05);}
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

/* Product text hierarchy */
.product-name{font-weight:900;font-size:.95rem;color:var(--text);line-height:1.2;}
.product-desc{font-size:.78rem;color:var(--muted);opacity:.75;margin-top:2px;}
.row{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;}
.right{white-space:nowrap;font-weight:900;font-variant-numeric:tabular-nums;}
.pill-muted{background:#f1f5f9;color:var(--muted);border-color:var(--line);}
.num{font-variant-numeric:tabular-nums;}

/* Streamlit primary buttons: make them more visible */
div.stButton > button[kind="primary"]{
  height:44px;
  font-weight:900;
  border-radius:12px;
}

/* Venue / Supplier message bubble (minimal) */
.msgbox{border-left:4px solid #93c5fd;background:#eff6ff;}
.msgbox .label{font-weight:900;color:#1e3a8a;margin-bottom:6px;}
.msgbox .txt{white-space:pre-wrap;color:#0f172a;font-weight:700;line-height:1.25;}

</style>
""",
        unsafe_allow_html=True,
    )


def _badge(text: str, kind: str = "info") -> str:
    return f"<span class='badge {kind}'>{text}</span>"



def _product_html(name: str, desc: str = "") -> str:
    name = (name or "Product").strip()
    desc = (desc or "").strip()
    if desc:
        return (
            "<div>"
            f"<div class='product-name'>{name}</div>"
            f"<div class='product-desc'>{desc}</div>"
            "</div>"
        )
    return f"<div class='product-name'>{name}</div>"

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
    if s in ("SUPPLIER_CREDIT_NOTE_ISSUED", "SUPPLEMENTARY_DELIVERY_SENT", "SUPPLIER_CREDIT_NOTE_PENDING", "SUPPLIER_REJECTED"):
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



def save_supplier_resolution_per_ticket(
    ctx: Dict[str, Any],
    per_ticket: Dict[int, Dict[str, Any]],
    comment: str,
    credit_note_invoice_number: Optional[str] = None,
    shared_redelivery_eta: Optional[str] = None,
    shared_redelivery_invoice_number: Optional[str] = None,
) -> None:
    """Persist supplier resolutions per ticket.

    Rules:
    - operational_missing can only be supplementary_delivery (enforced in UI).
    - If at least one item is re-delivery -> workflow becomes SUPPLEMENTARY_DELIVERY_SENT
      else if at least one item is credit note -> SUPPLIER_CREDIT_NOTE_ISSUED.
    - Each ticket gets its own structured resolution_note, so the venue can verify line-by-line.
    """

    wf: OrderWorkflow = ctx["workflow"]
    order: Order = ctx["order"]

    # Determine workflow end state (single state for the provider)
    resolutions = [str(v.get("resolution") or "").strip() for v in (per_ticket or {}).values()]
    has_redelivery = any(r == "supplementary_delivery" for r in resolutions)
    has_credit_note = any(r == "credit_note" for r in resolutions)
    has_reject = any(r == "reject" for r in resolutions)

    # Priority: re-delivery > credit note (issued/pending) > reject
    if has_redelivery:
        to_state = "SUPPLEMENTARY_DELIVERY_SENT"
    elif has_credit_note:
        to_state = "SUPPLIER_CREDIT_NOTE_ISSUED" if (credit_note_invoice_number or "").strip() else "SUPPLIER_CREDIT_NOTE_PENDING"
    elif has_reject:
        to_state = "SUPPLIER_REJECTED"
    else:
        # No action selected: keep supplier-action state so venue can decide next.
        to_state = "SUPPLIER_REJECTED" if has_reject else "SUPPLIER_CREDIT_NOTE_PENDING"

    with get_session() as s:
        wf2 = s.exec(select(OrderWorkflow).where(OrderWorkflow.id == wf.id)).first()
        prev = wf2.state

        wf2.state = to_state
        wf2.updated_at = _now()
        wf2.updated_by_role = ROLE_SUPPLIER
        wf2.updated_by = "supplier"

        # Store an overall note summarizing the action (easy for venue/accounting)
        overall = []
        if has_credit_note:
            if (credit_note_invoice_number or '').strip():
                overall.append('credit_note')
                overall.append(f"credit_note_invoice={credit_note_invoice_number}")
            else:
                overall.append('credit_note_pending')
        if has_redelivery:
            overall.append('supplementary_delivery')
            if shared_redelivery_eta:
                overall.append(f"eta={shared_redelivery_eta}")
            if shared_redelivery_invoice_number:
                overall.append(f"invoice={shared_redelivery_invoice_number}")
        if has_reject:
            overall.append('reject')
        if not overall:
            overall.append('no_action')
        wf2.note = " | ".join(overall)
        s.add(wf2)

        # Update tickets
        tickets = list(
            s.exec(
                select(SeguimientoTicket).where(
                    SeguimientoTicket.order_id == order.id,
                    SeguimientoTicket.provider_name == wf2.provider_name,
                )
            ).all()
        )

        by_id = {int(getattr(t, "id", 0) or 0): t for t in tickets}

        for tid, payload in (per_ticket or {}).items():
            t = by_id.get(int(tid))
            if not t:
                continue

            res = str(payload.get("resolution") or "").strip()
            eta = payload.get("eta") or shared_redelivery_eta
            inv = payload.get("invoice") or shared_redelivery_invoice_number

            meta_parts = [res]
            if res == "credit_note" and credit_note_invoice_number:
                meta_parts.append(f"credit_note_invoice={credit_note_invoice_number}")
            if res == "supplementary_delivery" and eta:
                meta_parts.append(f"eta={eta}")
            if res == "supplementary_delivery" and inv:
                meta_parts.append(f"invoice={inv}")
            meta = " | ".join(meta_parts)

            # Only move actionable tickets
            if (t.state or "").lower() in {"open", "open_send", "supplier_action_done"}:
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
            wf2.note + (f"\n{comment}" if comment else ""),
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



    # Full order overview (always useful for supplier to double-check)
    if lines:
        with st.expander("📦 Full order (all items)", True):
            for line in sorted(lines, key=lambda l: _pname(l).lower()):
                p = products.get(getattr(line, "product_id", None)) if getattr(line, "product_id", None) else None
                name = (getattr(p, "name", None) or getattr(line, "spoken_name", None) or "Product").strip()
                unit = (getattr(p, "unit", None) or getattr(line, "unit", None) or "unit")
                ordered = float(getattr(line, "quantity", 0) or 0)
                desc = (getattr(p, "description", None) or "").strip() if p else ""
                st.markdown(
                    f"<div class='row'>{_product_html(name, desc)}<div class='right num'>{ordered:g} {unit}</div></div>",
                    unsafe_allow_html=True,
                )


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

    ic1, ic2 = st.columns([4, 1], vertical_alignment="bottom")
    with ic1:
        inv_input = st.text_input(
            "Invoice number (as written on the invoice)",
            key=inv_key,
            placeholder="e.g. INV-2026-00123",
        )
    with ic2:
        if st.button("Save", type="primary", use_container_width=True, key=f"save_inv_{inv_key}"):
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
            desc = (getattr(p, "description", None) or "").strip() if p else ""

            left, mid, right = st.columns([2.8, 3.0, 1.3], vertical_alignment="center")

            with left:
                st.markdown(_product_html(str(name), desc), unsafe_allow_html=True)
                st.markdown(
                    f"<div class='pillrow'><span class='pill'>Ordered: {ordered:g} {unit}</span></div>",
                    unsafe_allow_html=True,
                )

            with mid:
                m1, m2 = st.columns([1.15, 1.0], vertical_alignment="center")
                with m1:
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

                with m2:
                    if choice == "partial":
                        qty_send = st.number_input(
                            "Qty",
                            min_value=0.0,
                            max_value=float(ordered),
                            value=float(default_qty),
                            step=1.0,
                            key=f"sup_conf_qty_{lid}",
                            label_visibility="collapsed",
                        )
                    else:
                        # keep column height consistent
                        st.markdown("<div style='height:38px'></div>", unsafe_allow_html=True)

            with right:
                # Right-aligned "will send"
                qs = float(qty_send or 0.0)
                st.markdown(
                    f"<div class='right num'><div class='muted'>Will send</div>{qs:g} {unit}</div>",
                    unsafe_allow_html=True,
                )

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
        if (t.state or "").lower() not in ("resolved", "resolved_not_reordered", "urgent_requested")
        and getattr(t, "resolved_at", None) is None
    ]

    receipt: Optional[ProviderReceipt] = ctx.get("receipt")
    inv = (getattr(receipt, "invoice_number", None) or "").strip() or "—"

    # Invoice date: prefer invoice_number_set_at, fallback to order.created_at
    _inv_dt_raw = getattr(receipt, "invoice_number_set_at", None) if receipt else None
    if not isinstance(_inv_dt_raw, datetime):
        _inv_dt_raw = getattr(ctx.get("order"), "created_at", None)
    _inv_dt_str = _inv_dt_raw.strftime("%d %b %Y") if isinstance(_inv_dt_raw, datetime) else "—"

    # Saved credit note number from workflow note
    _wf_note = (getattr(ctx.get("workflow"), "note", None) or "")
    _saved_cn_no = ""
    for _p in _wf_note.replace("·", "|").split("|"):
        _p = _p.strip()
        if _p.startswith("credit_note_invoice="):
            _saved_cn_no = _p.split("=", 1)[1].strip()
            break

    _cn_line = ""
    if _saved_cn_no:
        _cn_line = f"<div class='muted' style='margin-top:4px;'>Credit note number: <b>{html.escape(_saved_cn_no)}</b></div>"

    # --- Step 4 header card ---
    st.markdown(
        "<div class='card'>"
        f"<div class='h1'>Step 4 · Resolve issues (Invoice: {html.escape(inv)})</div>"
        f"<div class='muted'>Invoice date: <b>{html.escape(_inv_dt_str)}</b></div>"
        f"{_cn_line}"
        "<div class='muted' style='margin-top:4px;'>Choose one option. The venue will verify and then close the incident.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # Venue → Supplier message (shown to supplier before resolving)
    # Stored inside OrderWorkflow.note like: "... Venue message: <text>"
    # ------------------------------------------------------------
    wf = ctx.get("workflow")
    _note = (getattr(wf, "note", None) or "").strip()

    venue_msg = ""
    marker = "Venue message:"
    if marker.lower() in _note.lower():
        # case-insensitive split
        idx = _note.lower().find(marker.lower())
        venue_msg = _note[idx + len(marker):].strip()

    if venue_msg:
        st.markdown(
            "<div class='card' style='border-left:6px solid #2563eb;'>"
            "<div style='font-weight:900;margin-bottom:4px;'>💬 Venue message</div>"
            f"<div class='muted' style='white-space:pre-wrap;'>{venue_msg}</div>"
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
            p = products.get(getattr(line, "product_id", None)) if (line and getattr(line, "product_id", None)) else None
            desc = (getattr(p, "description", None) or "").strip() if p else ""

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
                    f"<div class='row'>{_product_html(name, desc)}"
                    f"<div class='right num'>missing {issue_qty:g} {unit}</div></div>"
                    f"<div class='muted num' style='margin-top:2px;'>invoiced {expected:g} · received {received:g} · {kind}</div>",
                    unsafe_allow_html=True,
                )
            elif (getattr(t, "kind", "") or "").lower() in {"damaged", "wrong_item"}:
                kind_l = (getattr(t, "kind", "") or "").lower()
                kind_label = "damaged" if kind_l == "damaged" else "wrong item"
                st.markdown(
                    f"<div class='row'>{_product_html(name, desc)}"
                    f"<div class='right num'>issue {issue_qty:g} {unit}</div></div>"
                    f"<div class='muted num' style='margin-top:2px;'>invoiced {expected:g} · received {received:g} · {kind_label}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='row'>{_product_html(name, desc)}"
                    f"<div class='right num'>issue {issue_qty:g} {unit}</div></div>"
                    f"<div class='muted num' style='margin-top:2px;'>invoiced {expected:g} · received {received:g} · {kind}</div>",
                    unsafe_allow_html=True,
                )

            if (getattr(t, "note", None) or "").strip():
                st.caption((getattr(t, "note", None) or "").strip())

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # --- Per-product resolution (operational_missing can ONLY be re-delivery) ---

    st.markdown("### Choose a solution for each product")

    # Suggest time slots from the provider directory if available
    provider_obj: Optional[Provider] = ctx.get("provider_obj")
    schedule = _parse_delivery_schedule_json(getattr(provider_obj, "delivery_schedule_json", None) or "") if provider_obj else {}
    preset_slots = ["08:00-14:00", "16:00-20:00"]

    per_ticket: Dict[int, Dict[str, Any]] = {}

    for t in open_t:
        tid = int(getattr(t, "id", 0) or 0)
        if not tid:
            continue

        kind_raw = (getattr(t, "kind", "") or "").lower()
        key = f"res_{tid}"

        # Default + allowed options
        if kind_raw == "operational_missing":
            allowed = ["supplementary_delivery"]
            default = "supplementary_delivery"
        else:
            if kind_raw in {"damaged", "wrong_item"}:
                allowed = ["credit_note", "supplementary_delivery", "reject", "no_action"]
                default = "credit_note"
            else:
                allowed = ["credit_note", "supplementary_delivery", "no_action"]
                default = "credit_note"

        if key not in st.session_state:
            st.session_state[key] = default
        if st.session_state[key] not in allowed:
            st.session_state[key] = default

        # Compact row UI
        lid = int(getattr(t, "order_line_id", 0) or 0)
        line = (ctx.get("line_by_id") or {}).get(lid) if isinstance(ctx.get("line_by_id"), dict) else None

        name = (getattr(t, "product_name", None) or "Product").strip()
        desc = (getattr(t, "product_desc", None) or "").strip()

        unit = (getattr(t, "unit", None) or (getattr(line, "unit", None) if line else "") or "unit").strip()
        qty_issue = float(getattr(t, "qty_invoiced", 0) or 0)

        def _fmt_resolution(x: str) -> str:
            return {
                "credit_note": "📝 Credit note",
                "supplementary_delivery": "🚚 Re-delivery",
                "reject": "❌ Reject",
                "no_action": "⛔ No action",
            }[x]

        with st.container(border=True):
            c1, c2, c3 = st.columns([2.4, 1.6, 1.4])
            with c1:
                st.markdown(_product_html(name, desc), unsafe_allow_html=True)

            with c2:
                # Chips
                kind_label = (getattr(t, "kind", "") or "").replace("_", " ").strip() or "issue"
                st.markdown(
                    f"<div class='pillrow'>"
                    f"<span class='pill'>Qty: {qty_issue:g} {unit}</span>"
                    f"<span class='pill pill-muted'>{kind_label}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with c3:
                res = st.selectbox(
                    "Resolution",
                    options=allowed,
                    index=allowed.index(st.session_state[key]),
                    key=key,
                    format_func=_fmt_resolution,
                    label_visibility="collapsed",
                )

            per_ticket[tid] = {"resolution": res}

    credit_note_ids = [tid for tid, v in per_ticket.items() if (v.get("resolution") == "credit_note")]
    redelivery_ids = [tid for tid, v in per_ticket.items() if (v.get("resolution") == "supplementary_delivery")]

    # operational_missing items require a NEW invoice number for the re-delivery
    op_redelivery_ids = []
    for tid in redelivery_ids:
        t = next((x for x in open_t if int(getattr(x, 'id', 0) or 0) == int(tid)), None)
        if t and (getattr(t, 'kind', '') == 'operational_missing'):
            op_redelivery_ids.append(int(tid))

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # Shared credit note number
    credit_note_no: Optional[str] = None

    if credit_note_ids:
        cn_key = f"credit_note_no_{ctx['order'].id}_{ctx['provider_name']}"
        # Pre-fill from DB (ticket resolution_note) if not already in session
        if cn_key not in st.session_state:
            _saved_cn = ""
            for tid in credit_note_ids:
                _t = next((x for x in open_t if int(getattr(x, 'id', 0) or 0) == int(tid)), None)
                if _t:
                    _rn = (getattr(_t, "resolution_note", "") or "")
                    _m = _META_KV_RE and None  # parse inline
                    for _part in _rn.replace("·", "|").split("|"):
                        _part = _part.strip()
                        if _part.startswith("credit_note_invoice="):
                            _saved_cn = _part.split("=", 1)[1].strip()
                            break
                if _saved_cn:
                    break
            st.session_state[cn_key] = _saved_cn
        cc1, cc2 = st.columns([4, 1], vertical_alignment="bottom")
        with cc1:
            credit_note_input = st.text_input(
                "Credit note invoice number (applies to all credit note items)",
                key=cn_key,
                placeholder="e.g. CN-2026-001",
            )
        with cc2:
            if st.button("Save", type="primary", use_container_width=True, key=f"save_cn_{cn_key}"):
                _cn_to_save = (credit_note_input or "").strip() or None
                if _cn_to_save:
                    # Persist CN number to DB immediately (workflow note + ticket resolution_note)
                    _wf: OrderWorkflow = ctx["workflow"]
                    _order: Order = ctx["order"]
                    with get_session() as _s_db:
                        _wf2 = _s_db.exec(select(OrderWorkflow).where(OrderWorkflow.id == _wf.id)).first()
                        # Update workflow note: replace/add credit_note_invoice
                        _note_parts = [p.strip() for p in (_wf2.note or "").replace("·", "|").split("|") if p.strip()]
                        _note_parts = [p for p in _note_parts if not p.startswith("credit_note_invoice=") and p != "credit_note_pending"]
                        if "credit_note" not in _note_parts:
                            _note_parts = ["credit_note"] + [p for p in _note_parts if p != "credit_note"]
                        _note_parts.append(f"credit_note_invoice={_cn_to_save}")
                        _wf2.note = " | ".join(_note_parts)
                        # Update state from PENDING to ISSUED if applicable
                        if _wf2.state == "SUPPLIER_CREDIT_NOTE_PENDING":
                            _wf2.state = "SUPPLIER_CREDIT_NOTE_ISSUED"
                        _wf2.updated_at = _now()
                        _s_db.add(_wf2)
                        # Update each credit_note ticket's resolution_note
                        _tickets = list(_s_db.exec(
                            select(SeguimientoTicket).where(
                                SeguimientoTicket.order_id == _order.id,
                                SeguimientoTicket.provider_name == _wf2.provider_name,
                            )
                        ).all())
                        for _tk in _tickets:
                            _rn = _tk.resolution_note or ""
                            if "credit_note" in _rn.lower() and "credit_note_invoice=" not in _rn:
                                # Add CN number to existing resolution_note
                                _first_line = _rn.split("\n")[0]
                                _rest = _rn[len(_first_line):]
                                _tk.resolution_note = f"{_first_line} | credit_note_invoice={_cn_to_save}{_rest}"
                                _tk.updated_at = _now()
                                _s_db.add(_tk)
                            elif "credit_note_invoice=" in _rn:
                                # Replace existing CN number
                                import re as _re
                                _tk.resolution_note = _re.sub(
                                    r"credit_note_invoice=[^|\n]*",
                                    f"credit_note_invoice={_cn_to_save}",
                                    _rn,
                                )
                                _tk.updated_at = _now()
                                _s_db.add(_tk)
                        _s_db.commit()
                    st.success("Credit note number saved ✅")
                    st.rerun()
                else:
                    st.warning("Please enter a credit note number first.")
        credit_note_no = (credit_note_input or "").strip() or None
        st.caption("All products marked as *Credit note* will share the same credit note number.")


    # Re-delivery grouping
    same_eta: Optional[str] = None
    shared_selected = False
    same_invoice: Optional[str] = None

    if redelivery_ids:
        same_delivery = st.radio(
            "Re-delivery grouping",
            options=["same", "separate"],
            format_func=lambda x: {
                "same": "All re-delivery items come in the SAME delivery",
                "separate": "Re-delivery items come in DIFFERENT deliveries",
            }[x],
            horizontal=False,
        )

        def _eta_picker(prefix: str) -> Optional[str]:
            eta_date = st.date_input(f"Expected delivery date ({prefix})", key=f"eta_date_{prefix}")
            weekday_key = _day_key_for_date(datetime.combine(eta_date, datetime.min.time())) if eta_date else ""
            suggested = schedule.get(weekday_key, []) if weekday_key else []
            slot_options = suggested or preset_slots
            eta_slot = st.selectbox(
                f"Delivery window ({prefix})",
                options=slot_options,
                key=f"eta_slot_{prefix}",
            )
            custom_slot = st.text_input(
                f"Custom window (optional) ({prefix})",
                placeholder="e.g. 06:30-10:30",
                key=f"eta_custom_{prefix}",
            ).strip()
            if custom_slot:
                if _TIME_RANGE_RE.match(custom_slot):
                    eta_slot = custom_slot
                    st.caption(f"✅ Using custom window: {custom_slot}")
                else:
                    st.error("Invalid time window. Use HH:MM-HH:MM (e.g. 06:30-10:30).")
            if not eta_date or not (eta_slot or "").strip():
                return None
            return f"{eta_date.isoformat()} {eta_slot}".strip()

        shared_selected = (same_delivery == "same")
        if shared_selected:
            st.markdown("#### Re-delivery details (shared)")
            same_eta = _eta_picker("shared")
            st.caption("All products marked as *Re-delivery* will share this same date + time window.")

            if op_redelivery_ids:
                same_invoice = st.text_input("New invoice number (operational missing re-delivery)", placeholder="e.g. INV-2026-104").strip() or None
                st.caption("Operational missing items are re-delivered with a new invoice number.")
        else:
            st.markdown("#### Re-delivery details (per product)")
            for tid in redelivery_ids:
                t = next((x for x in open_t if int(getattr(x, "id", 0) or 0) == tid), None)
                if not t:
                    continue
                with st.container(border=True):
                    st.markdown(f"**{getattr(t, 'product_name', 'Product')}**")
                    eta = _eta_picker(str(tid))
                    per_ticket[tid]["eta"] = eta

                    if tid in op_redelivery_ids:
                        inv = st.text_input("New invoice number", key=f"op_inv_{tid}", placeholder="e.g. INV-2026-104").strip() or None
                        per_ticket[tid]["invoice"] = inv

    comment = st.text_area("Comment (optional)")

    if st.button("Submit resolution", type="primary", use_container_width=True):
        # Validation
        # In Greece, credit notes are often issued later.
        # Allow submit without number, but it will be marked as *pending* for the venue/accountant.
        # (Venue should NOT close until the number is provided.)
        if credit_note_ids and not (credit_note_no or ""):
            st.warning("Credit note number is missing — we will mark this as *Credit note pending*.")

        if redelivery_ids:
            if shared_selected:
                if not same_eta:
                    st.error("Please enter the re-delivery date + time window for the shared delivery.")
                    return
                if op_redelivery_ids and not (same_invoice or ''):
                    st.error('Please enter the NEW invoice number for operational-missing re-delivery (shared).')
                    return
            else:
                for tid in redelivery_ids:
                    if not (per_ticket.get(tid, {}) or {}).get("eta"):
                        st.error("Please enter a re-delivery date + window for each re-delivery item.")
                        return
                    # For operational_missing, invoice number is required
                    if tid in op_redelivery_ids and not (per_ticket.get(tid, {}) or {}).get('invoice'):
                        st.error('Please enter the NEW invoice number for each operational-missing re-delivery item.')
                        return

        save_supplier_resolution_per_ticket(
            ctx,
            per_ticket=per_ticket,
            comment=comment,
            credit_note_invoice_number=credit_note_no,
            shared_redelivery_eta=(same_eta if (redelivery_ids and shared_selected) else None),
            shared_redelivery_invoice_number=(same_invoice if (op_redelivery_ids and redelivery_ids and shared_selected) else None),
        )
        st.success("Submitted. Waiting for venue verification.")
        st.rerun()


def _render_readonly(ctx: Dict[str, Any]) -> None:
    wf: OrderWorkflow = ctx["workflow"]
    state = wf.state

    # Extract credit note number from workflow note if present
    _cn_display = ""
    for _p in (wf.note or "").replace("·", "|").split("|"):
        _p = _p.strip()
        if _p.startswith("credit_note_invoice="):
            _cn_display = _p.split("=", 1)[1].strip()
            break

    _cn_html = ""
    if _cn_display:
        _cn_html = f"<div style='margin-top:8px'><b>Credit note number:</b> {html.escape(_cn_display)}</div>"

    st.markdown(
        f"<div class='card'>"
        f"<div class='h1'>Current status</div>"
        f"<div style='margin-top:8px'>{_badge(state, 'info')}</div>"
        f"{_cn_html}"
        f"<div class='hr'></div>"
        f"<div class='muted'>No action required right now.</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_pending_cn_update(ctx: Dict[str, Any]) -> None:
    """Allow supplier to add/update credit note number when state is SUPPLIER_CREDIT_NOTE_PENDING."""
    wf: OrderWorkflow = ctx["workflow"]
    order: Order = ctx["order"]

    cn_key = f"pending_cn_{order.id}_{ctx['provider_name']}"
    if cn_key not in st.session_state:
        st.session_state[cn_key] = ""

    cc1, cc2 = st.columns([4, 1], vertical_alignment="bottom")
    with cc1:
        cn_input = st.text_input(
            "Credit note invoice number",
            key=cn_key,
            placeholder="e.g. CN-2026-001",
        )
    with cc2:
        if st.button("Save", type="primary", use_container_width=True, key=f"save_{cn_key}"):
            _cn_val = (cn_input or "").strip()
            if not _cn_val:
                st.warning("Please enter a credit note number.")
            else:
                with get_session() as _s_db:
                    _wf2 = _s_db.exec(select(OrderWorkflow).where(OrderWorkflow.id == wf.id)).first()
                    # Update workflow note
                    _note_parts = [p.strip() for p in (_wf2.note or "").replace("·", "|").split("|") if p.strip()]
                    _note_parts = [p for p in _note_parts if not p.startswith("credit_note_invoice=") and p != "credit_note_pending"]
                    if "credit_note" not in _note_parts:
                        _note_parts = ["credit_note"] + [p for p in _note_parts if p != "credit_note"]
                    _note_parts.append(f"credit_note_invoice={_cn_val}")
                    _wf2.note = " | ".join(_note_parts)
                    _wf2.state = "SUPPLIER_CREDIT_NOTE_ISSUED"
                    _wf2.updated_at = _now()
                    _s_db.add(_wf2)
                    # Update ticket resolution_notes
                    _tickets = list(_s_db.exec(
                        select(SeguimientoTicket).where(
                            SeguimientoTicket.order_id == order.id,
                            SeguimientoTicket.provider_name == _wf2.provider_name,
                        )
                    ).all())
                    import re as _re
                    for _tk in _tickets:
                        _rn = _tk.resolution_note or ""
                        if "credit_note" not in _rn.lower():
                            continue
                        if "credit_note_invoice=" in _rn:
                            _tk.resolution_note = _re.sub(
                                r"credit_note_invoice=[^|\n]*",
                                f"credit_note_invoice={_cn_val}",
                                _rn,
                            )
                        else:
                            _first_line = _rn.split("\n")[0]
                            _rest = _rn[len(_first_line):]
                            _tk.resolution_note = f"{_first_line} | credit_note_invoice={_cn_val}{_rest}"
                        _tk.updated_at = _now()
                        _s_db.add(_tk)
                    _s_db.commit()
                st.success("Credit note number saved ✅")
                st.rerun()


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
        elif wf.state == "SUPPLIER_CREDIT_NOTE_PENDING":
            # Supplier submitted without CN number — let them add it now
            st.info("⏳ Credit note number pending. Please provide it below.")
            _render_readonly(ctx)
            _render_pending_cn_update(ctx)
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
