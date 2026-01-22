from __future__ import annotations

"""features.manage_orders.receive_orders

Venue-side dashboard to:
- review supplier confirmation (expected products)
- record what the venue received (per line)
- auto-create incidence tickets
- request supplier resolution (credit note / re-delivery)
- verify supplier resolution and close
- show timeline / audit trail per supplier

UX additions implemented:
1) Lock invoice number once invoice is verified (workflow CLOSED)
2) Show badge "Set by supplier / Set by venue" next to invoice input
3) Show invoice number pill next to supplier name in Incidences

This module does NOT handle payments.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import requests
import json
import math
import re
import pandas as pd
import streamlit as st
from sqlmodel import select
import html
from core.db import get_session
from difflib import SequenceMatcher

from core.mailer import send_smtp_email
from core.public_links import ROLE_SUPPLIER, build_seguimiento_url, norm_provider
from core.url_nav import set_query_params, qp_int, qp_str
from datetime import datetime, timedelta
from sqlalchemy import func
from domain.models import (
    Order,
    OrderLine,
    OrderWorkflow,
    OrderWorkflowEvent,
    Product,
    Provider,
    ProviderLineFollowUp,
    ProviderReceipt,
    SeguimientoTicket, ProviderDiscountRule,ProviderSendStatus,
    UrgentReorderRequest

)


# =============================
# UI: styles
# =============================

def _inject_css() -> None:
    st.markdown(
        """
<style>
:root{--card:#ffffff;--text:#0f172a;--muted:#64748b;--border:#e2e8f0;--ok:#16a34a;--warn:#f59e0b;--bad:#ef4444;--info:#2563eb;}
.block-container{padding-top:1.1rem;max-width:1200px;}

.voi-card{background:var(--card);border:1px solid var(--border);border-radius:18px;padding:14px 14px;margin:12px 0;box-shadow:0 4px 16px rgba(2,6,23,.04);} 
.voi-title{font-weight:900;font-size:1.05rem;color:var(--text);}
.voi-muted{color:var(--muted);font-size:.92rem;}

.voi-badge{display:inline-flex;align-items:center;gap:6px;padding:6px 10px;border-radius:999px;border:1px solid var(--border);font-weight:800;font-size:.78rem;background:#f8fafc;color:var(--text);} 
.voi-badge.ok{background:#ecfdf5;border-color:#bbf7d0;color:#166534;}
.voi-badge.warn{background:#fffbeb;border-color:#fde68a;color:#92400e;}
.voi-badge.bad{background:#fef2f2;border-color:#fecaca;color:#991b1b;}
.voi-badge.info{background:#eff6ff;border-color:#bfdbfe;color:#1e3a8a;}

.voi-hr{height:1px;background:var(--border);margin:12px 0;}

.voi-kpi{display:flex;gap:10px;flex-wrap:wrap;margin:8px 0;}
.voi-kpi .k{flex:1;min-width:160px;border:1px solid var(--border);border-radius:16px;padding:10px;background:#ffffff;}
.voi-kpi .k .t{font-weight:900;}
.voi-kpi .k .v{font-weight:950;font-size:1.25rem;margin-top:4px;}

.line{border:1px solid var(--border);border-radius:16px;padding:12px;margin:10px 0;background:#fff;}
.line.bad{border-color:#fecaca;background:#fef2f2;}
.line .top{display:flex;justify-content:space-between;gap:10px;align-items:baseline;flex-wrap:wrap;}
.line .name{font-weight:900;}
.line .desc{color:var(--muted);opacity:.8;font-size:.9rem;margin-top:2px;}
.line .qty{font-weight:900;white-space:nowrap;}

.pillrow{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;}
.pill{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;border:1px solid var(--border);background:#f1f5f9;font-weight:800;font-size:.78rem;color:var(--text);}
.pill--ok{background:#ecfdf5;border-color:#bbf7d0;color:#166534;}
.pill--warn{background:#fffbeb;border-color:#fde68a;color:#92400e;}
.pill--bad{background:#fef2f2;border-color:#fecaca;color:#991b1b;}
.pill--info{background:#eff6ff;border-color:#bfdbfe;color:#1e3a8a;}


.small{font-size:.85rem;color:var(--muted);}
</style>
""",
        unsafe_allow_html=True,
    )


# =============================
# Helpers
# =============================

def _now() -> datetime:
    return datetime.utcnow()


def _s(v: Any) -> str:
    return ("" if v is None else str(v)).strip()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)




def _order_created_dt(o: Any) -> Optional[datetime]:
    """Best-effort order datetime (created_at / created)."""
    d = getattr(o, "created_at", None) or getattr(o, "created", None)
    if isinstance(d, datetime):
        return d
    # sometimes stored as date
    try:
        from datetime import date as _date
        if isinstance(d, _date):
            return datetime(d.year, d.month, d.day)
    except Exception:
        pass
    return None


def _fmt_order_date(o: Any) -> str:
    dt = _order_created_dt(o)
    return dt.strftime("%d %b %Y") if dt else "—"


def _days_ago_badge(o: Any, now: Optional[datetime] = None) -> str:
    """Returns a short '⏱️ 3d ago' style label (or empty if no date)."""
    dt = _order_created_dt(o)
    if not dt:
        return ""
    now = now or _now()
    # use full days, clamp to 0
    days = max(0, int((now.date() - dt.date()).days))
    if days == 0:
        return "⏱️ today"
    if days == 1:
        return "⏱️ 1d ago"
    return f"⏱️ {days}d ago"


def _render_global_filters(
    *,
    key_prefix: str,
    orders: List[Any],
    provider_options: List[str],
    show_date: bool = True,
    show_provider: bool = True,
) -> Dict[str, Any]:
    """
    Sticky (session_state-backed) filters shared by global tabs.
    Returns dict with keys: providers_set (normalized), date_from (date|None), date_to (date|None)
    """
    from datetime import date as _date

    out: Dict[str, Any] = {"providers_set": None, "date_from": None, "date_to": None}

    cols = []
    if show_provider and show_date:
        cols = st.columns([1.6, 1.6, 0.8], vertical_alignment="center")
    elif show_provider:
        cols = st.columns([2.4, 0.8], vertical_alignment="center")
    elif show_date:
        cols = st.columns([2.4, 0.8], vertical_alignment="center")
    else:
        cols = [st.container(), st.container()]

    idx = 0

    # Provider filter
    if show_provider:
        k_prov = f"{key_prefix}_prov_filter"
        default_sel = st.session_state.get(k_prov)
        if default_sel is None:
            default_sel = []  # means 'all'
        with cols[idx]:
            prov_sel = st.multiselect(
                "Provider",
                options=provider_options,
                default=default_sel,
                placeholder="All providers",
                key=k_prov,
            )
        idx += 1
        out["providers_set"] = {norm_provider(p) for p in prov_sel} if prov_sel else None

    # Date range filter
    if show_date:
        k_date = f"{key_prefix}_date_filter"
        # derive range from orders
        dts = [d for d in (_order_created_dt(o) for o in orders) if d is not None]
        min_d = min(dts).date() if dts else _now().date()
        max_d = max(dts).date() if dts else _now().date()

        default_range = st.session_state.get(k_date)
        if not default_range:
            default_range = (min_d, max_d)

        with cols[idx]:
            dr = st.date_input(
                "Date range",
                value=default_range,
                min_value=min_d,
                max_value=max_d,
                key=k_date,
            )
        idx += 1

        date_from: Optional[_date] = None
        date_to: Optional[_date] = None
        if isinstance(dr, tuple) and len(dr) == 2:
            date_from, date_to = dr[0], dr[1]
        elif isinstance(dr, list) and len(dr) == 2:
            date_from, date_to = dr[0], dr[1]
        elif isinstance(dr, _date):
            date_from, date_to = dr, dr

        out["date_from"] = date_from
        out["date_to"] = date_to

    # Clear filters
    with cols[-1]:
        if st.button("Clear filters", use_container_width=True, key=f"{key_prefix}_clear_filters"):
            if show_provider:
                st.session_state[f"{key_prefix}_prov_filter"] = []
            if show_date:
                # reset to full range
                dts = [d for d in (_order_created_dt(o) for o in orders) if d is not None]
                min_d = min(dts).date() if dts else _now().date()
                max_d = max(dts).date() if dts else _now().date()
                st.session_state[f"{key_prefix}_date_filter"] = (min_d, max_d)
            st.rerun()

    return out
# =============================
# Operational missing — urgent re-order helpers
# =============================

_DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def _safe_json_loads(raw: str) -> dict:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _parse_time_range_start(slot: str) -> Optional[tuple[int, int]]:
    """Parse "HH:MM-HH:MM" and return (hour, minute) for the start."""
    slot = (slot or "").strip()
    if len(slot) < 5 or "-" not in slot:
        return None
    start = slot.split("-", 1)[0].strip()
    if ":" not in start:
        return None
    hh, mm = start.split(":", 1)
    try:
        h = int(hh)
        m = int(mm)
        if 0 <= h <= 23 and 0 <= m <= 59:
            return (h, m)
    except Exception:
        return None
    return None


def _next_delivery_dt(provider: Optional[Provider], now: Optional[datetime] = None) -> Optional[datetime]:
    """Best-effort next delivery datetime based on Provider.delivery_schedule_json."""
    if not provider:
        return None
    sched = _safe_json_loads(_s(getattr(provider, "delivery_schedule_json", "")))
    if not sched:
        return None
    now = now or _now()
    # look up to 14 days ahead
    for add_days in range(0, 14):
        day_dt = now + timedelta(days=add_days)
        day_key = _DAYS[int(day_dt.weekday())]
        slots = sched.get(day_key) or []
        if not isinstance(slots, list):
            continue

        candidates: list[datetime] = []
        for s in slots:
            hm = _parse_time_range_start(str(s))
            if not hm:
                continue
            dt = datetime(day_dt.year, day_dt.month, day_dt.day, hm[0], hm[1])
            # if today, ensure it's in the future
            if add_days == 0 and dt <= now:
                continue
            candidates.append(dt)

        if candidates:
            return min(candidates)

    return None


def _norm_words(s: str) -> set[str]:
    import re

    words = re.findall(r"[0-9]+|[^\W_]+", (s or ""), flags=re.UNICODE)
    return {w.lower() for w in words if w and len(w) >= 3}


def _relevance(a: str, b: str) -> float:
    """Jaccard similarity on token sets."""
    A = _norm_words(a)
    B = _norm_words(b)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    uni = len(A | B)
    return float(inter) / float(uni) if uni else 0.0


def find_alternative_providers(*, ctx: Any, ticket: Any, line: Optional[OrderLine]) -> list[dict[str, Any]]:
    """Return product suggestions across providers.

    Output dict keys:
      - provider: provider name
      - product_id: product id
      - name/desc/unit
      - next_delivery_dt
      - relevance
    """
    products_by_id: dict[int, Product] = getattr(ctx, "products_by_id", {}) or {}
    providers_by_name: dict[str, Provider] = getattr(ctx, "providers_by_name", {}) or {}

    target_name = _s(getattr(ticket, "product_name", ""))
    if line is not None:
        # Prefer catalog name when available
        pid = getattr(line, "product_id", None)
        if pid is not None and int(pid) in products_by_id:
            target_name = _s(getattr(products_by_id[int(pid)], "name", "")) or target_name

    # candidate products: same venue catalog
    cands: list[dict[str, Any]] = []
    for pid, p in (products_by_id or {}).items():
        prov = norm_provider(_s(getattr(p, "provider_name", "")))
        pname = _s(getattr(p, "name", ""))
        if not pname:
            continue
        rel = _relevance(target_name, pname)
        if rel <= 0:
            continue
        prov_obj = providers_by_name.get(prov)
        nd = _next_delivery_dt(prov_obj)
        cands.append(
            {
                "provider": prov,
                "product_id": int(pid),
                "name": pname,
                "desc": _s(getattr(p, "description", "")),
                "unit": _s(getattr(p, "unit", "")) or "unit",
                "next_delivery_dt": nd,
                "relevance": float(rel),
            }
        )

    # Always include actual provider + original line product (even if no match)
    actual_provider = norm_provider(_s(getattr(ticket, "provider_name", "")))
    if not actual_provider:
        actual_provider = norm_provider(_s(getattr(line, "provider", "")) if line is not None else "")
    if actual_provider:
        actual_pid = None
        if line is not None and getattr(line, "product_id", None) is not None:
            actual_pid = int(getattr(line, "product_id"))
        if actual_pid is not None and actual_pid in products_by_id:
            p = products_by_id[actual_pid]
            prov_obj = providers_by_name.get(actual_provider)
            cands.append(
                {
                    "provider": actual_provider,
                    "product_id": int(actual_pid),
                    "name": _s(getattr(p, "name", "")) or target_name,
                    "desc": _s(getattr(p, "description", "")),
                    "unit": _s(getattr(p, "unit", "")) or "unit",
                    "next_delivery_dt": _next_delivery_dt(prov_obj),
                    "relevance": 1.0,
                }
            )
        else:
            prov_obj = providers_by_name.get(actual_provider)
            cands.append(
                {
                    "provider": actual_provider,
                    "product_id": None,
                    "name": target_name or "(product)",
                    "desc": "",
                    "unit": _s(getattr(ticket, "unit", "")) or "unit",
                    "next_delivery_dt": _next_delivery_dt(prov_obj),
                    "relevance": 0.5,
                }
            )

    # De-dup by (provider, product_id)
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for it in cands:
        key = (norm_provider(_s(it.get("provider"))), str(it.get("product_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)

    def _sort_key(it: dict[str, Any]) -> tuple:
        nd = it.get("next_delivery_dt")
        nd_ts = nd.timestamp() if isinstance(nd, datetime) else float("inf")
        # relevance desc
        rel = float(it.get("relevance") or 0.0)
        return (nd_ts, -rel)

    out.sort(key=_sort_key)
    return out


def _get_or_create_draft_for_op_missing(*, venue_id: int, actor: str) -> int:
    """Return an existing draft order id or create a new one."""
    with get_session() as s:
        o = s.exec(
            select(Order)
            .where(Order.venue_id == int(venue_id), Order.status == "draft")
            .order_by(Order.created_at.desc())
        ).first()
        if o and getattr(o, "id", None) is not None:
            return int(o.id)

        new_o = Order(
            venue_id=int(venue_id),
            status="draft",
            created_at=_now(),
            updated_at=_now(),
            created_by=actor,
            updated_by=actor,
            title="Operational missing (draft)",
            note="Auto-created draft for non-urgent operational missing.",
        )
        s.add(new_o)
        s.commit()
        s.refresh(new_o)
        return int(new_o.id)


def _add_product_to_order(*, venue_id: int, order_id: int, actor: str, product_id: Optional[int], qty: float, unit: str, provider_name: str, spoken_name: str = "") -> None:
    """Add (or increment) a product line in an order.

    Simplicity rules:
    - If same product_id already exists in order, increment qty.
    - We never create invoices here (new order => new invoice later).
    """
    qty = _safe_float(qty, 0.0)
    if qty <= 0:
        return
    now = _now()
    provider_name = norm_provider(provider_name)
    with get_session() as s:
        ln = None
        if product_id is not None:
            ln = s.exec(
                select(OrderLine).where(OrderLine.order_id == int(order_id), OrderLine.product_id == int(product_id))
            ).first()
        if ln:
            ln.quantity = float(_safe_float(getattr(ln, "quantity", 0.0), 0.0) + qty)
            ln.updated_at = now
            ln.updated_by = actor
            s.add(ln)
        else:
            s.add(
                OrderLine(
                    venue_id=int(venue_id),
                    order_id=int(order_id),
                    product_id=(int(product_id) if product_id is not None else None),
                    spoken_name=_s(spoken_name),
                    quantity=float(qty),
                    unit=(_s(unit) or "unit"),
                    provider=(provider_name or None),
                    updated_at=now,
                    updated_by=actor,
                )
            )

        o = s.exec(select(Order).where(Order.id == int(order_id))).first()
        if o:
            o.updated_at = now
            o.updated_by = actor
            s.add(o)
        s.commit()


def _create_urgent_order_from_cart(
    *,
    venue_id: int,
    actor: str,
    cart: list[dict[str, Any]],
    source_order_id: Optional[int] = None,
) -> Optional[int]:
    items = [it for it in (cart or []) if _safe_float(it.get("quantity"), 0.0) > 0]
    if not items:
        return None

    now = _now()

    # Normalize providers once
    providers_in_cart = sorted(
        {norm_provider(_s(it.get("provider_name"))) for it in items if _s(it.get("provider_name"))},
        key=lambda x: x.lower(),
    )

    with get_session() as s:
        src_txt = f" (from order #{int(source_order_id)})" if source_order_id else ""
        o = Order(
            venue_id=int(venue_id),
            status="pending_receive",  # ✅ NOT draft
            created_at=now,
            updated_at=now,
            created_by=actor,
            updated_by=actor,
            title=f"URGENT re-order{src_txt}",
            note=(
                "Urgent re-order created from incidences."
                + (f" Source order: #{int(source_order_id)}." if source_order_id else "")
                + " New order → new invoice number."
            ),
        )
        s.add(o)
        s.commit()
        s.refresh(o)
        oid = int(o.id)

        # --- Create lines ---
        for it in items:
            pid = it.get("product_id")
            pid_i: Optional[int] = None
            try:
                if pid is not None:
                    pid_i = int(pid)
            except Exception:
                pid_i = None

            qty = float(it.get("quantity") or 0.0)
            unit = _s(it.get("unit")) or "unit"
            provider_name = norm_provider(_s(it.get("provider_name")))
            spoken_name = _s(it.get("spoken_name")) or "Urgent item"

            s.add(
                OrderLine(
                    venue_id=int(venue_id),
                    order_id=oid,
                    product_id=pid_i,
                    spoken_name=f"{spoken_name} [URGENT]",
                    quantity=qty,
                    unit=unit,
                    provider=provider_name or None,
                    updated_at=now,
                    updated_by=actor,
                )
            )

        # ✅ CRITICAL: Create workflow rows per provider
        for prov in providers_in_cart:
            if not prov:
                continue
            existing = s.exec(
                select(OrderWorkflow).where(
                    OrderWorkflow.order_id == int(oid),
                    OrderWorkflow.provider_name == prov,
                )
            ).first()

            if not existing:
                wf = OrderWorkflow(
                    venue_id=int(venue_id),
                    order_id=int(oid),
                    provider_name=prov,
                    state="ORDER_SENT",
                    updated_at=now,
                    updated_by=actor,
                )
                # Best-effort extra fields if your model has them
                # Best-effort extra fields if your model has them
                for attr, val in [
                    ("updated_by_role", "venue"),
                    ("note", "Urgent order created and sent to supplier."),
                ]:
                    try:
                        setattr(wf, attr, val)
                    except Exception:
                        pass
                s.add(wf)


        s.commit()
        return oid




# =============================
# Urgent reorder requests
# =============================

_WEEKDAY_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

def _parse_delivery_schedule(raw: Any) -> Dict[str, List[str]]:
    """Parse Provider.delivery_schedule_json into normalized dict."""
    if not raw:
        return {}
    try:
        if isinstance(raw, str):
            data = json.loads(raw)
        elif isinstance(raw, dict):
            data = raw
        else:
            return {}
        out: Dict[str, List[str]] = {}
        for k, v in (data or {}).items():
            kk = _s(k).strip().lower()[:3]
            if kk not in _WEEKDAY_KEYS:
                continue
            slots = []
            for it in (v or []):
                s = _s(it).strip()
                if s:
                    slots.append(s)
            if slots:
                out[kk] = slots
        return out
    except Exception:
        return {}

def _next_delivery_datetime(provider: Optional[Provider], now: Optional[datetime] = None) -> Optional[datetime]:
    """Return the next delivery datetime based on provider.delivery_schedule_json."""
    if now is None:
        now = _now()
    if not provider:
        return None
    sched = _parse_delivery_schedule(getattr(provider, "delivery_schedule_json", None))
    if not sched:
        return None

    def _slot_start(slot: str) -> Optional[Tuple[int, int]]:
        m = re.match(r"^\s*(\d{1,2}):(\d{2})\s*-", slot or "")
        if not m:
            return None
        return int(m.group(1)), int(m.group(2))

    best: Optional[datetime] = None
    for add_days in range(0, 8):
        d = now.date() + timedelta(days=add_days)
        key = _WEEKDAY_KEYS[(now.weekday() + add_days) % 7]
        slots = sched.get(key) or []
        for slot in slots:
            hm = _slot_start(slot)
            if not hm:
                continue
            cand = datetime.combine(d, datetime.min.time()).replace(hour=hm[0], minute=hm[1])
            if cand < now:
                continue
            if best is None or cand < best:
                best = cand
    return best

def _fmt_eta(now: datetime, dt: Optional[datetime]) -> str:
    if not dt:
        return "—"
    delta = dt - now
    mins = int(delta.total_seconds() // 60)
    if mins < 0:
        mins = 0
    hours = mins // 60
    rem = mins % 60
    if hours >= 48:
        return f"in {hours//24}d"
    if hours >= 1:
        return f"in {hours}h {rem:02d}m"
    return f"in {rem}m"

def _upsert_urgent_reorder_request(
    *,
    ticket_id: int,
    provider_name: str,
    product_name: str,
    qty: float,
    unit: str,
    actor: str,
) -> int:
    """Create (or update) an urgent reorder request for a given ticket/incidence.

    This implementation assumes UrgentReorderRequest does NOT have ticket_id/venue_id/product_id fields.
    We store the ticket/incidence id into `incidence_id`.
    """
    now = _now()
    pname = _s(product_name)
    pnorm = re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9]+", " ", pname.lower())).strip()

    with get_session() as s:
        # "Upsert" by (incidence_id, product_normalized, original_provider_name) for open requests
        row = s.exec(
            select(UrgentReorderRequest).where(
                UrgentReorderRequest.incidence_id == int(ticket_id),
                UrgentReorderRequest.product_normalized == pnorm,
                UrgentReorderRequest.original_provider_name == _s(provider_name) or None,
                UrgentReorderRequest.status.in_(["pending", "sent"]),
            )
        ).first()

        if row:
            row.quantity = float(qty)
            row.unit = _s(unit) or row.unit
            row.updated_at = now
            s.add(row)
            s.commit()
            s.refresh(row)
            return int(row.id)

        req = UrgentReorderRequest(
            incidence_id=int(ticket_id),
            product_name=pname,
            product_normalized=pnorm,
            quantity=float(qty),
            unit=_s(unit) or "",
            original_provider_name=_s(provider_name) or None,
            status="pending",
            created_at=now,
            updated_at=now,
        )
        s.add(req)
        s.commit()
        s.refresh(req)
        return int(req.id)


def _list_open_urgent_requests() -> List[UrgentReorderRequest]:
    """List open urgent requests (DB-clear friendly)."""
    with get_session() as s:
        rows = list(
            s.exec(
                select(UrgentReorderRequest).where(
                    UrgentReorderRequest.status == "pending"
                ).order_by(UrgentReorderRequest.created_at.desc())
            ).all()
        )
    return rows


def _similarity(a: str, b: str) -> float:
    a = _s(a).strip().lower()
    b = _s(b).strip().lower()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()





def _suggest_providers_for_request(
    *,
    venue_id: int,
    req: Any,
    products: Dict[int, Product],
    providers_by_name: Dict[str, Provider],
    limit: int = 8,
    now_dt: Optional[datetime] = None,
    min_sim: float = 0.62,   # similarity threshold (0..1)
    min_rel: float = 0.10,   # keep very low relevance too
) -> List[Dict[str, Any]]:
    """
    Returns suggestions (fastest delivery first), including similar products.

    Each suggestion dict:
      provider_name, product_id, product_name, unit, delivery_dt, eta_txt,
      relevance, similarity, score
    """
    now_dt = now_dt or _now()

    target_name = _s(getattr(req, "product_name", "")) or ""
    target_unit = _s(getattr(req, "unit", "")) or "unit"
    original_provider = norm_provider(_s(getattr(req, "original_provider_name", "")))

    # normalize once
    tn = _s(target_name)

    raw: List[Dict[str, Any]] = []

    for pid, p in (products or {}).items():
        pname = _s(getattr(p, "name", ""))
        if not pname:
            continue

        prov = norm_provider(_s(getattr(p, "provider_name", "")))
        if not prov:
            continue

        # 1) your existing relevance (exact-ish / contains / token match)
        rel = float(_relevance(tn, pname) or 0.0)

        # 2) similarity (fuzzy) — you likely already have _similarity()
        # If you don't, tell me and I’ll swap in a local SequenceMatcher.
        sim = float(_similarity(tn, pname) or 0.0)

        # keep if either relevance OR similarity is good
        if rel < min_rel and sim < min_sim:
            continue

        prov_obj = providers_by_name.get(prov)
        dt = _next_delivery_dt(prov_obj, now_dt)

        # combined score: prioritize similarity, but keep relevance influence
        score = 0.65 * sim + 0.35 * rel

        raw.append(
            {
                "provider_name": prov,
                "product_id": int(pid),
                "product_name": pname,
                "unit": _s(getattr(p, "unit", "")) or target_unit,
                "delivery_dt": dt,
                "eta_txt": _fmt_eta(now_dt, dt),
                "relevance": rel,
                "similarity": sim,
                "score": score,
            }
        )

    # ✅ keep best candidate per provider:
    # best = earliest delivery; if tie, highest score
    best_by_provider: Dict[str, Dict[str, Any]] = {}
    for it in raw:
        prov = norm_provider(_s(it.get("provider_name")))
        if not prov:
            continue

        curr = best_by_provider.get(prov)
        if curr is None:
            best_by_provider[prov] = it
            continue

        a_dt = it.get("delivery_dt")
        b_dt = curr.get("delivery_dt")
        a_ts = a_dt.timestamp() if isinstance(a_dt, datetime) else float("inf")
        b_ts = b_dt.timestamp() if isinstance(b_dt, datetime) else float("inf")

        if (a_ts < b_ts) or (a_ts == b_ts and float(it.get("score") or 0.0) > float(curr.get("score") or 0.0)):
            best_by_provider[prov] = it

    out: List[Dict[str, Any]] = list(best_by_provider.values())

    # ✅ include original provider only if missing (prevents duplicates)
    if original_provider:
        already = any(norm_provider(_s(x.get("provider_name"))) == original_provider for x in out)
        if not already:
            prov_obj = providers_by_name.get(original_provider)
            dt = _next_delivery_dt(prov_obj, now_dt)
            out.append(
                {
                    "provider_name": original_provider,
                    "product_id": None,
                    "product_name": target_name or "(product)",
                    "unit": target_unit,
                    "delivery_dt": dt,
                    "eta_txt": _fmt_eta(now_dt, dt),
                    "relevance": 1.0,
                    "similarity": 1.0,
                    "score": 1.0,
                }
            )

    # sort: fastest delivery first, then best score
    def _sort_key(it: Dict[str, Any]) -> tuple:
        dt = it.get("delivery_dt")
        dt_ts = dt.timestamp() if isinstance(dt, datetime) else float("inf")
        return (dt_ts, -float(it.get("score") or 0.0))

    out.sort(key=_sort_key)
    return out[: int(limit)]




def _send_urgent_request_email(
    *,
    venue_id: int,
    order_id: int,
    provider_name: str,
    items: List[Dict[str, Any]],
    source_order_id: Optional[int] = None,
) -> Tuple[bool, str]:
    prov = norm_provider(provider_name)
    with get_session() as s:
        p = s.exec(select(Provider).where(Provider.venue_id == int(venue_id), Provider.name == prov)).first()
        if not p:
            p = s.exec(select(Provider).where(Provider.venue_id == int(venue_id), Provider.name == provider_name)).first()
        emails: List[str] = []
        if p and getattr(p, "order_email", None):
            emails = [x.strip() for x in (p.order_email or "").split("|") if x.strip()]
        if not emails and p and getattr(p, "emails", None):
            emails = [x.strip() for x in (p.emails or "").split("|") if x.strip()]
        if not emails:
            return False, "No supplier email configured."

    # ✅ Supplier link (same pattern you already use elsewhere)
    link = build_seguimiento_url(order_id=int(order_id), provider_name=prov, role=ROLE_SUPPLIER, page_path="seguimiento")

    lines = [f"- {it.get('name','')} · {it.get('qty','')} {it.get('unit','')}" for it in items]
    src_txt = f" (from order #{int(source_order_id)})" if source_order_id else ""

    subject = f"URGENT — re-order request (order #{int(order_id)}){src_txt}"
    body = (
        f"URGENT re-order request\n"
        f"New order: #{int(order_id)}{src_txt}\n"
        f"Provider: {prov}\n\n"
        + "\n".join(lines)
        + "\n\n"
        "Please open the link to view/confirm and coordinate delivery:\n"
        f"{link}\n"
    )

    try:
        send_smtp_email(to=emails, subject=subject, text_body=body)
    except Exception as e:
        return False, f"Email failed: {e}"
    return True, link


def _build_wa_link_for_urgent(
    *,
    venue_id: int,
    provider_name: str,
    items: List[Dict[str, Any]],
    wa_cc: str = "+34",
) -> Tuple[str, str]:
    import urllib.parse as up
    prov = norm_provider(provider_name)
    with get_session() as s:
        p = s.exec(select(Provider).where(Provider.venue_id == int(venue_id), Provider.name == prov)).first()
        if not p:
            p = s.exec(select(Provider).where(Provider.venue_id == int(venue_id), Provider.name == provider_name)).first()
        phone = _s(getattr(p, "order_phone", None) or getattr(p, "phone", None) or getattr(p, "phones", None)) if p else ""
        phone = (phone.split("|")[0].strip() if phone else "")

    def _normalize_phone(phone_raw: str, country_code: str) -> str:
        pr = re.sub(r"\D+", "", phone_raw or "")
        cc = re.sub(r"\D+", "", country_code or "")
        if not pr:
            return ""
        if (phone_raw or "").strip().startswith("+"):
            return "+" + pr
        if cc and not pr.startswith(cc):
            return "+" + cc + pr
        return "+" + pr

    phone_norm = _normalize_phone(phone, wa_cc)
    if not phone_norm:
        return "missing_phone", ""

    msg_lines = ["URGENT re-order request:"]
    for it in items:
        msg_lines.append(f"- {it.get('name','')} · {it.get('qty','')} {it.get('unit','')}")
    msg_lines.append("Please confirm availability and next delivery slot.")
    txt = "\n".join(msg_lines)
    url = f"https://wa.me/{phone_norm.replace('+','')}?text={up.quote(txt)}"
    return "ok", url


def _upsert_provider_send_status(
    *, venue_id: int, order_id: int, provider_name: str, sent_email: bool, sent_whatsapp: bool
) -> None:
    prov = norm_provider(provider_name)
    now = _now()
    with get_session() as s:
        row = s.exec(
            select(ProviderSendStatus).where(
                ProviderSendStatus.order_id == int(order_id),
                ProviderSendStatus.provider_name == prov,
            )
        ).first()
        if not row:
            row = ProviderSendStatus(
                venue_id=int(venue_id),
                order_id=int(order_id),
                provider_name=prov,
            )
        # best-effort fields (your model may have different ones)
        for attr, val in [
            ("sent", True),
            ("sent_email", bool(sent_email)),
            ("sent_whatsapp", bool(sent_whatsapp)),
            ("sent_at", now),
            ("updated_at", now),
        ]:
            try:
                setattr(row, attr, val)
            except Exception:
                pass
        s.add(row)
        s.commit()


def _move_urgent_items_into_order(
    *, venue_id: int, order_id: int, provider_name: str, items: List[Dict[str, Any]], actor: str
) -> None:
    """Create OrderLines in this same order, so they appear in Receive."""
    prov = norm_provider(provider_name)
    now = _now()

    with get_session() as s:
        for it in (items or []):
            pid = it.get("product_id")
            pid_i = None
            try:
                if pid is not None:
                    pid_i = int(pid)
            except Exception:
                pid_i = None

            qty = float(it.get("qty") or 0.0)
            if qty <= 0:
                continue

            unit = _s(it.get("unit")) or "unit"
            name = _s(it.get("name")) or "Urgent item"

            # If same product already exists for same provider in this order, increment qty
            existing = None
            if pid_i is not None:
                existing = s.exec(
                    select(OrderLine).where(
                        OrderLine.order_id == int(order_id),
                        OrderLine.product_id == int(pid_i),
                        OrderLine.provider == prov,
                    )
                ).first()

            if existing:
                existing.quantity = float(_safe_float(getattr(existing, "quantity", 0.0), 0.0) + qty)
                existing.updated_at = now
                existing.updated_by = actor
                s.add(existing)
            else:
                s.add(
                    OrderLine(
                        venue_id=int(venue_id),
                        order_id=int(order_id),
                        product_id=pid_i,
                        provider=prov,
                        spoken_name=f"{name} [URGENT]",
                        quantity=float(qty),
                        unit=unit,
                        updated_at=now,
                        updated_by=actor,
                    )
                )

        # ensure workflow exists (Receive uses it)
        wf = s.exec(
            select(OrderWorkflow).where(
                OrderWorkflow.order_id == int(order_id),
                OrderWorkflow.provider_name == prov,
            )
        ).first()
        if not wf:
            wf = OrderWorkflow(
                venue_id=int(venue_id),
                order_id=int(order_id),
                provider_name=prov,
                state="ORDER_SENT",
                updated_at=now,
                updated_by=actor,
            )
            s.add(wf)

        s.commit()

def _ensure_order_pending_receive(*, order_id: int, actor: str) -> None:
    """Make sure the order is visible in Receive/Track Order."""
    now = _now()
    with get_session() as s:
        o = s.exec(select(Order).where(Order.id == int(order_id))).first()
        if not o:
            return

        # Only bump from draft -> pending_receive (do not downgrade other states)
        if (_s(getattr(o, "status", "")).lower() in {"draft", "borrador"}):
            o.status = "pending_receive"
            o.updated_at = now
            o.updated_by = actor or "venue"
            s.add(o)
            s.commit()

def _render_urgent_tab(ctx: 'OrderContext') -> None:
    st.markdown("### ⚡ Urgent reorders")
    reqs = _list_open_urgent_requests()
    if not reqs:
        st.info("No urgent reorder requests yet. Use **Order urgent** in Incidences and click **Save & request decision**.")
        return

    providers_by_name = ctx.providers_by_name or {}
    products_by_id = ctx.products_by_id or {}

    with st.expander("⚙️ Send settings", expanded=False):
        c1, c2 = st.columns([1.0, 1.0], vertical_alignment="center")
        with c1:
            use_email = st.toggle("Email", value=True, key=f"urg_use_email_{int(ctx.order.id)}")
        with c2:
            use_wa = st.toggle("WhatsApp", value=False, key=f"urg_use_wa_{int(ctx.order.id)}")
        wa_cc = st.text_input("Prefijo país (WhatsApp)", value="+34", key=f"urg_wa_cc_{int(ctx.order.id)}")

    pending_send: Dict[str, List[Dict[str, Any]]] = {}

    for r in reqs:
        rid = int(getattr(r, "id", 0) or 0)
        pname = _s(getattr(r, "product_name", "")) or "—"
        qty = float(_safe_float(getattr(r, "quantity", 0.0), 0.0))
        srcp = norm_provider(_s(getattr(r, "original_provider_name", "")))
        unit = _s(getattr(r, "unit", "")) or "unit"


        with st.container(border=True):
            st.markdown(f"**{pname}**")
            st.caption(f"Qty: {qty:g} {unit} · From: {srcp or '—'}")

            suggestions = _suggest_providers_for_request(
                venue_id=int(ctx.order.venue_id),
                req=r,
                products=products_by_id,
                providers_by_name=providers_by_name,
                limit=8,
            )
            if not suggestions:
                st.warning("No provider suggestions found.")
                continue

            def _lbl(sug: Dict[str, Any]) -> str:
                dt = sug.get("delivery_dt")
                dt_txt = dt.strftime("%a %d %b %H:%M") if isinstance(dt, datetime) else "—"
                return f"{sug['provider_name']} · {sug.get('eta_txt','—')} · {dt_txt}"

            idxs = list(range(len(suggestions)))
            chosen_idx = st.radio(
                "Suggestions (fastest first)",
                options=idxs,
                index=0,
                format_func=lambda i: _lbl(suggestions[int(i)]),
                key=f"urg_pick_{int(ctx.order.id)}_{rid}",
            )
            chosen = suggestions[int(chosen_idx)]

            qty_send = st.number_input(
                "Qty to order",
                min_value=0.0,
                value=float(qty),
                step=1.0,
                key=f"urg_qty_{int(ctx.order.id)}_{rid}",
            )
            if qty_send > 0:
                prov = chosen["provider_name"]
                pending_send.setdefault(prov, []).append(
                    {
                        "req_id": rid,
                        "provider_name": prov,
                        "product_id": chosen.get("product_id"),
                        "name": chosen.get("product_name") or pname,
                        "qty": float(qty_send),
                        "unit": chosen.get("unit") or unit,
                    }
                )


    st.markdown("#### Send grouped messages")
    if not pending_send:
        st.info("Select a qty > 0 to prepare messages.")
        return

    prov_list = sorted(pending_send.keys(), key=lambda x: x.lower())
    prov_selected = st.multiselect(
        "Providers to contact now",
        options=prov_list,
        default=prov_list,
        key=f"urg_send_sel_{int(ctx.order.id)}",
    )

    if st.button("🚀 Send urgent requests", type="primary", use_container_width=True, key=f"urg_send_btn_{int(ctx.order.id)}"):
        actor = _s(st.session_state.get("user_email") or st.session_state.get("actor") or "venue")
        any_fail = False

        # ✅ Build ONE cart for the NEW urgent order (all selected providers)
        cart: list[dict[str, Any]] = []
        for prov in prov_selected:
            for it in (pending_send.get(prov) or []):
                cart.append(
                    {
                        "product_id": it.get("product_id"),
                        "quantity": float(it.get("qty") or 0.0),
                        "unit": it.get("unit") or "unit",
                        "provider_name": prov,
                        "spoken_name": it.get("name") or "Urgent item",
                    }
                )

        urgent_order_id = _create_urgent_order_from_cart(
            venue_id=int(ctx.order.venue_id),
            actor=actor,
            cart=cart,
            source_order_id=int(ctx.order.id),  # ✅ relationship
        )
        if not urgent_order_id:
            st.error("Could not create urgent order (empty cart).")
            st.stop()

        # ✅ Now send per provider using the NEW urgent order id
        for prov in prov_selected:
            items = pending_send.get(prov) or []
            if not items:
                continue

            email_ok = False
            wa_ok = False

            # 1) Email send (now references urgent_order_id + includes link)
            if use_email:
                ok, msg_or_link = _send_urgent_request_email(
                    venue_id=int(ctx.order.venue_id),
                    order_id=int(urgent_order_id),   # ✅ NEW ORDER
                    provider_name=prov,
                    items=items,
                    source_order_id=int(ctx.order.id),
                )
                if ok:
                    email_ok = True
                    st.success(f"Email sent to {prov}")
                else:
                    any_fail = True
                    st.error(f"{prov}: {msg_or_link}")

            # 2) WhatsApp link (still optional)
            if use_wa:
                status, url = _build_wa_link_for_urgent(
                    venue_id=int(ctx.order.venue_id),
                    provider_name=prov,
                    items=items,
                    wa_cc=wa_cc,
                )
                if status == "ok" and url:
                    wa_ok = True
                    st.link_button(f"Open WhatsApp for {prov}", url, use_container_width=True)
                else:
                    any_fail = True
                    st.error(f"{prov}: WhatsApp phone missing/invalid")

            # ✅ Mark provider as sent for the NEW urgent order
            if email_ok or wa_ok:
                _upsert_provider_send_status(
                    venue_id=int(ctx.order.venue_id),
                    order_id=int(urgent_order_id),   # ✅ NEW ORDER
                    provider_name=prov,
                    sent_email=bool(email_ok),
                    sent_whatsapp=False,
                )

        # ✅ Update urgent requests so they disappear
        with get_session() as s:
            for prov in prov_selected:
                for it in (pending_send.get(prov) or []):
                    rid = int(it.get("req_id") or 0)
                    rr = s.exec(select(UrgentReorderRequest).where(UrgentReorderRequest.id == rid)).first()
                    if rr:
                        rr.status = "sent"
                        for attr, val in [
                            ("selected_provider_name", prov),
                            ("updated_at", _now()),
                        ]:
                            try:
                                setattr(rr, attr, val)
                            except Exception:
                                pass
                        s.add(rr)
            s.commit()

        if not any_fail:
            st.success(f"Done ✓ Created urgent order #{int(urgent_order_id)}")

        # ✅ Jump user to the NEW urgent order in Track Order
        set_query_params(page="tracking", order_id=str(int(urgent_order_id)))
        st.rerun()



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


def _rules_for_provider(venue_id: int, providers_by_name: dict[str, Provider], provider_name: str) -> list[ProviderDiscountRule]:
    pnorm = norm_provider(provider_name)
    prow = providers_by_name.get(pnorm)
    if not prow or getattr(prow, "id", None) is None:
        return []
    return _provider_rules_cached(get_session, venue_id, int(prow.id))


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


def _pricing_for_line(
    *,
    venue_id: int,
    providers_by_name: dict[str, Provider],
    provider_name: str,
    pid: Optional[int],
    qty: float,
    gross_unit: float,
) -> dict[str, Any]:
    if qty <= 0 or gross_unit <= 0:
        return {"net_unit": gross_unit, "discount_pct": 0.0, "rule_kind": "", "rule_id": None, "applied": False}

    prov_norm = norm_provider(provider_name)
    rules = _rules_for_provider(venue_id, providers_by_name, prov_norm)
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

    # net price override rule
    if rk.endswith("_net_price"):
        net_unit = float(getattr(chosen, "price_override", 0.0) or 0.0)
        if net_unit <= 0:
            return {"net_unit": gross_unit, "discount_pct": 0.0, "rule_kind": "", "rule_id": None, "applied": False}
        disc_pct = (1.0 - (net_unit / gross_unit)) * 100.0 if gross_unit > 0 else 0.0
        disc_pct = max(0.0, min(100.0, disc_pct))
        return {
            "net_unit": net_unit,
            "discount_pct": float(disc_pct),
            "rule_kind": rk,
            "rule_id": int(chosen.id) if getattr(chosen, "id", None) is not None else None,
            "applied": True,
        }

    # pct discount rule
    disc = float(getattr(chosen, "discount_percent", 0.0) or 0.0)
    disc = max(0.0, min(100.0, disc))
    net_unit = gross_unit * (1.0 - disc / 100.0)
    return {
        "net_unit": float(net_unit),
        "discount_pct": float(disc),
        "rule_kind": rk,
        "rule_id": int(chosen.id) if getattr(chosen, "id", None) is not None else None,
        "applied": disc > 0.0,
    }


def _price_for_pid(products_by_id: dict[int, Product], pid: Optional[int]) -> float:
    if pid is None:
        return 0.0
    p = products_by_id.get(int(pid))
    return float(getattr(p, "price", 0.0) or 0.0) if p else 0.0


def _iva_pct_for_pid(products_by_id: dict[int, Product], pid: Optional[int], default_pct: float = 21.0) -> float:
    if pid is None:
        return float(default_pct)
    p = products_by_id.get(int(pid))
    if not p:
        return float(default_pct)
    v = float(getattr(p, "iva", 0.0) or 0.0)
    return float(default_pct) if v <= 0 else float(v)

def _badge(text: str, kind: str) -> str:
    kind = kind if kind in {"ok", "warn", "bad", "info"} else "info"
    return f"<span class='voi-badge {kind}'>{text}</span>"


def _state_badge(state: str) -> Tuple[str, str]:
    s = (state or "").upper()
    if s == "CLOSED":
        return "Closed", "ok"
    if s in {"INVOICE_DISCREPANCY", "WAITING_SUPPLIER_ACTION"}:
        return "Needs resolution", "bad"
    if s == "SUPPLIER_CREDIT_NOTE_PENDING":
        return "Credit note pending", "warn"
    if s == "SUPPLIER_REJECTED":
        return "Supplier rejected", "warn"
    if s in {"SUPPLIER_CREDIT_NOTE_ISSUED", "SUPPLEMENTARY_DELIVERY_SENT"}:
        return "Supplier acted", "info"
    if s in {"OPERATIONAL_MISSING_PRODUCT"}:
        return "Operational", "warn"
    if "RECEIV" in s:
        return "Received", "ok"
    if "SUPPLIER_CONFIRMED" in s:
        return "Supplier confirmed", "info"
    if s == "ORDER_SENT":
        return "Sent", "info"
    return s or "—", "info"


def _invoice_set_badge(receipt: Optional[ProviderReceipt]) -> Tuple[str, str]:
    if not receipt or not _s(getattr(receipt, "invoice_number", None)):
        return "Invoice # missing", "warn"
    by = _s(getattr(receipt, "invoice_number_set_by", None)).lower()
    if by.startswith("supplier"):
        return "Set by supplier", "info"
    if by.startswith("venue"):
        return "Set by venue", "info"
    return "Invoice # set", "info"


@dataclass
class OrderContext:
    order: Order
    providers_by_name: Dict[str, Provider]
    workflows_by_provider: Dict[str, OrderWorkflow]
    tickets_by_provider: Dict[str, List[SeguimientoTicket]]
    products_by_id: Dict[int, Product]
    lines_by_provider: Dict[str, List[OrderLine]]
    followups_by_key: Dict[Tuple[str, int], ProviderLineFollowUp]
    receipts_by_provider: Dict[str, ProviderReceipt]


# =============================
# Data loading
# =============================

def _get_active_orders(venue_id: int) -> List[Order]:
    with get_session() as s:
        return list(
            s.exec(
                select(Order)
                .where(Order.venue_id == int(venue_id))
                .order_by(Order.created_at.desc())
            ).all()
        )


def _get_history_orders(venue_id: int) -> List[Order]:
    """Orders that have at least one CLOSED provider workflow.

    We treat "finished" per provider, because the order is split per supplier.
    """
    with get_session() as s:
        q = (
            select(Order)
            .join(OrderWorkflow, OrderWorkflow.order_id == Order.id)
            .where(Order.venue_id == int(venue_id))
            .where(OrderWorkflow.state == "CLOSED")
            .order_by(Order.created_at.desc())
            .distinct()
        )
        return list(s.exec(q).all())

def _load_order_context(venue_id: int, order_id: int) -> OrderContext:
    with get_session() as s:
        order = s.exec(select(Order).where(Order.id == int(order_id))).first()
        if not order:
            raise ValueError("Order not found")

        workflows = list(
            s.exec(select(OrderWorkflow).where(OrderWorkflow.order_id == int(order_id))).all()
        )
        workflows_by_provider = {norm_provider(w.provider_name): w for w in workflows}

        tickets = list(
            s.exec(select(SeguimientoTicket).where(SeguimientoTicket.order_id == int(order_id))).all()
        )
        tickets_by_provider: Dict[str, List[SeguimientoTicket]] = {}
        for t in tickets:
            tickets_by_provider.setdefault(norm_provider(t.provider_name), []).append(t)

        lines = list(s.exec(select(OrderLine).where(OrderLine.order_id == int(order_id))).all())

        product_ids = [l.product_id for l in lines if getattr(l, "product_id", None)]
        products: Dict[int, Product] = {}
        if product_ids:
            ps = list(s.exec(select(Product).where(Product.id.in_(product_ids))).all())
            products = {int(p.id): p for p in ps if getattr(p, "id", None) is not None}

        # group lines by provider
        lines_by_provider: Dict[str, List[OrderLine]] = {}
        for l in lines:
            prov = ""
            p = products.get(int(l.product_id)) if getattr(l, "product_id", None) else None
            if p and getattr(p, "provider_name", None):
                prov = p.provider_name
            else:
                prov = _s(getattr(l, "provider", None))
            prov = norm_provider(prov)
            if prov:
                lines_by_provider.setdefault(prov, []).append(l)

        followups = list(
            s.exec(
                select(ProviderLineFollowUp)
                .where(ProviderLineFollowUp.order_id == int(order_id))
            ).all()
        )
        followups_by_key: Dict[Tuple[str, int], ProviderLineFollowUp] = {}
        for fu in followups:
            followups_by_key[(norm_provider(fu.provider_name), int(fu.order_line_id))] = fu

        receipts = list(
            s.exec(
                select(ProviderReceipt)
                .where(ProviderReceipt.order_id == int(order_id))
            ).all()
        )
        receipts_by_provider = {norm_provider(r.provider_name): r for r in receipts}

        providers = list(s.exec(select(Provider).where(Provider.venue_id == int(venue_id))).all())
        providers_by_name = {norm_provider(p.name): p for p in providers}

    return OrderContext(
        order=order,
        providers_by_name=providers_by_name,
        workflows_by_provider=workflows_by_provider,
        tickets_by_provider=tickets_by_provider,
        products_by_id=products,
        lines_by_provider=lines_by_provider,
        followups_by_key=followups_by_key,
        receipts_by_provider=receipts_by_provider,
    )


# =============================
# Timeline + workflow updates
# =============================

def _add_event(
    s,
    venue_id: int,
    order_id: int,
    provider: str,
    prev_state: str,
    new_state: str,
    actor_role: str,
    actor: str,
    note: str = "",
) -> None:
    s.add(
        OrderWorkflowEvent(
            venue_id=int(venue_id),
            order_id=int(order_id),
            provider_name=norm_provider(provider),
            from_state=_s(prev_state),
            to_state=_s(new_state),
            actor_role=_s(actor_role),
            actor=_s(actor),
            at=_now(),
            note=_s(note) or None,
        )
    )


def _set_workflow_state(
    *,
    venue_id: int,
    order_id: int,
    provider: str,
    to_state: str,
    actor_role: str,
    actor: str,
    note: str = "",
) -> None:
    prov = norm_provider(provider)
    with get_session() as s:
        wf = s.exec(
            select(OrderWorkflow).where(
                OrderWorkflow.order_id == int(order_id),
                OrderWorkflow.provider_name == prov,
            )
        ).first()
        if not wf:
            wf = OrderWorkflow(
                venue_id=int(venue_id),
                order_id=int(order_id),
                provider_name=prov,
                state="ORDER_SENT",
            )

        prev = _s(wf.state)
        wf.state = _s(to_state)
        wf.updated_at = _now()
        wf.updated_by_role = actor_role
        wf.updated_by = actor
        if note:
            wf.note = note

        s.add(wf)
        _add_event(s, int(venue_id), int(order_id), prov, prev, wf.state, actor_role, actor, note)
        s.commit()


def _load_timeline(order_id: int, provider: str) -> List[OrderWorkflowEvent]:
    prov = norm_provider(provider)
    with get_session() as s:
        return list(
            s.exec(
                select(OrderWorkflowEvent)
                .where(OrderWorkflowEvent.order_id == int(order_id))
                .where(OrderWorkflowEvent.provider_name == prov)
                .order_by(OrderWorkflowEvent.at.desc())
            ).all()
        )


def _render_timeline(ctx: OrderContext, provider: str) -> None:
    events = _load_timeline(int(ctx.order.id), provider)
    if not events:
        st.caption("No timeline yet.")
        return
    for e in events:
        at = e.at.strftime("%Y-%m-%d %H:%M") if getattr(e, "at", None) else ""
        note = _s(getattr(e, "note", None))
        frm = _s(getattr(e, "from_state", None))
        to = _s(getattr(e, "to_state", None))
        who = f"{_s(getattr(e, 'actor_role', None))}:{_s(getattr(e, 'actor', None))}".strip(":")
        st.markdown(f"- **{at}** · `{who}` · {frm} → {to}" + (f" · {note}" if note else ""))


# =============================
# Invoice number (shared)
# =============================

def upsert_provider_invoice_number(
    *,
    venue_id: int,
    order_id: int,
    provider_name: str,
    invoice_number: str,
    actor_role: str,
    actor: str,
) -> Tuple[bool, str]:
    inv = _s(invoice_number)
    if not inv:
        return False, "Invoice number cannot be empty."

    prov = norm_provider(provider_name)
    with get_session() as s:
        r = s.exec(
            select(ProviderReceipt).where(
                ProviderReceipt.venue_id == int(venue_id),
                ProviderReceipt.order_id == int(order_id),
                ProviderReceipt.provider_name == prov,
            )
        ).first()
        if not r:
            r = ProviderReceipt(
                venue_id=int(venue_id),
                order_id=int(order_id),
                provider_name=prov,
            )

        r.invoice_number = inv
        r.invoice_number_set_at = _now()
        r.invoice_number_set_by = f"{actor_role}:{actor}".strip(":")
        r.updated_at = _now()
        r.updated_by = actor_role
        s.add(r)

        _add_event(
            s,
            int(venue_id),
            int(order_id),
            prov,
            prev_state="",
            new_state="",
            actor_role=actor_role,
            actor=actor,
            note=f"Invoice number set: {inv}",
        )

        s.commit()

    return True, "ok"


# =============================
# Rendering helpers
# =============================

def _line_name(line: OrderLine, products: Dict[int, Product]) -> str:
    p = products.get(int(line.product_id)) if getattr(line, "product_id", None) else None
    return (p.name if p else (line.spoken_name or "Product")).strip()


def _line_unit(line: OrderLine, products: Dict[int, Product]) -> str:
    p = products.get(int(line.product_id)) if getattr(line, "product_id", None) else None
    return (getattr(p, "unit", None) if p else None) or (getattr(line, "unit", None) or "unit")


def _line_desc(line: OrderLine, products: Dict[int, Product]) -> str:
    p = products.get(int(line.product_id)) if getattr(line, "product_id", None) else None
    return ((getattr(p, "description", None) if p else None) or "").strip()


def _derive_expected_qty(line: OrderLine, fu: Optional[ProviderLineFollowUp]) -> float:
    ordered = _safe_float(getattr(line, "quantity", 0.0), 0.0)
    if not fu:
        return ordered
    stt = (_s(getattr(fu, "supplier_status", None))).lower()
    sq = getattr(fu, "supplier_qty", None)
    if stt == "missing":
        return 0.0
    if stt == "partial":
        return _safe_float(sq, 0.0)
    if stt == "ok":
        return _safe_float(sq, ordered) if sq is not None else ordered
    return ordered


def _render_expected_lines(
    ctx: OrderContext,
    provider: str,
    *,
    show_prices: bool = False,
    include_iva: bool = False
) -> None:
    prov = norm_provider(provider)
    lines = ctx.lines_by_provider.get(prov, []) or []
    if not lines:
        st.info("No products.")
        return

    st.markdown("<div class='voi-muted'><b>Supplier confirmation</b></div>", unsafe_allow_html=True)

    for ln in sorted(lines, key=lambda x: _line_name(x, ctx.products_by_id).lower()):
        lid = int(ln.id)
        name = _line_name(ln, ctx.products_by_id)
        unit = _line_unit(ln, ctx.products_by_id)
        desc = _line_desc(ln, ctx.products_by_id)
        ordered = _safe_float(getattr(ln, "quantity", 0.0), 0.0)

        fu = ctx.followups_by_key.get((prov, lid))
        stt = (_s(getattr(fu, "supplier_status", None))).lower() if fu else "unknown"
        sqty = getattr(fu, "supplier_qty", None) if fu else None

        # supplier confirmed qty (expected)
        if stt == "missing":
            tag = ("❌ Not sending", "bad")
            expected_qty = 0.0
            send_txt = f"0 {unit}"
            is_bad = True
        elif stt == "partial":
            tag = ("🟡 Partial", "warn")
            expected_qty = _safe_float(sqty, 0.0)
            send_txt = f"{expected_qty:g} {unit}"
            is_bad = expected_qty < ordered
        elif stt == "ok":
            tag = ("✅ Full", "ok")
            expected_qty = _safe_float(sqty, ordered) if sqty is not None else ordered
            send_txt = f"{expected_qty:g} {unit}"
            is_bad = False
        else:
            tag = ("⚪ Not specified", "warn")
            expected_qty = ordered  # best guess
            send_txt = "(not specified)"
            is_bad = False

        # ---- pricing pills (optional) ----
        price_html = ""
        if show_prices:
            pid_raw = getattr(ln, "product_id", None)
            pid = int(pid_raw) if pid_raw not in (None, "", 0, "0") else None

            gross_unit = _price_for_pid(ctx.products_by_id, pid)  # catalog/unit price (gross/base)
            if gross_unit > 0 and expected_qty > 0:
                pricing = _pricing_for_line(
                    venue_id=int(ctx.order.venue_id),
                    providers_by_name=ctx.providers_by_name,
                    provider_name=provider,
                    pid=pid,
                    qty=float(expected_qty),
                    gross_unit=float(gross_unit),
                )
                net_unit = float(pricing.get("net_unit", gross_unit) or gross_unit)
                disc_pct = float(pricing.get("discount_pct", 0.0) or 0.0)

                subtotal = float(expected_qty) * net_unit

                pills = [
                    f"<span class='pill'>€/{unit}: {net_unit:,.2f}</span>",
                ]
                if disc_pct > 0:
                    pills.append(f"<span class='pill pill--warn'>Disc: {disc_pct:g}%</span>")
                pills.append(f"<span class='pill pill--ok'>Subtotal: {subtotal:,.2f}</span>")

                if include_iva:
                    iva_pct = _iva_pct_for_pid(ctx.products_by_id, pid, 21.0)
                    iva_eur = subtotal * (iva_pct / 100.0)
                    total = subtotal + iva_eur
                    pills.append(f"<span class='pill'>IVA {iva_pct:g}%: {iva_eur:,.2f}</span>")
                    pills.append(f"<span class='pill pill--ok'>Total: {total:,.2f}</span>")

                price_html = "<div class='pillrow'>" + "".join(pills) + "</div>"

        cls = "line bad" if is_bad else "line"
        st.markdown(
            f"<div class='{cls}'>"
            f"<div class='top'><div><div class='name'>{name}</div>"
            + (f"<div class='desc'>{desc}</div>" if desc else "")
            + f"</div><div class='qty'>{send_txt}</div></div>"
            f"<div class='pillrow'>"
            f"<span class='pill'>Ordered: {ordered:g} {unit}</span>"
            f"<span class='pill pill--{tag[1]}'>{tag[0]}</span>"
            f"</div>"
            f"{price_html}"
            f"</div>",
            unsafe_allow_html=True,
        )



def _render_receive_form(ctx: OrderContext, provider: str) -> None:
    prov = norm_provider(provider)
    order = ctx.order
    lines = ctx.lines_by_provider.get(prov, []) or []
    if not lines:
        return

    st.markdown("<div class='voi-muted'><b>Venue received</b></div>", unsafe_allow_html=True)

    STATUS_OPTIONS = ["OK", "Missing", "Damaged", "Wrong item"]  # split for better accountability
    INVOICE_OPTIONS = ["In invoice", "Not in invoice"]  # no "Unknown"

    # --- callback: sync dependent fields when Status changes (works ONLY outside st.form) ---
    def _on_status_change(status_key: str, issue_key: str, invoice_key: str, qty_expected_key: str) -> None:
        status_now = _s(st.session_state.get(status_key, "OK"))
        qty_expected = float(st.session_state.get(qty_expected_key, 0.0) or 0.0)

        if status_now == "OK":
            st.session_state[issue_key] = 0.0
            st.session_state[invoice_key] = "Not in invoice"

        elif status_now == "Missing":
            # Default issue qty to expected ONLY if user hasn't already put a value
            cur = _safe_float(st.session_state.get(issue_key, 0.0), 0.0)
            if cur <= 0:
                st.session_state[issue_key] = qty_expected

            # Missing => invoice listed relevant; keep existing if valid else default
            inv = _s(st.session_state.get(invoice_key, "Not in invoice"))
            if inv not in INVOICE_OPTIONS:
                st.session_state[invoice_key] = "Not in invoice"

        elif status_now in {"Damaged", "Wrong item"}:
            cur = _safe_float(st.session_state.get(issue_key, 0.0), 0.0)
            if cur <= 0:
                st.session_state[issue_key] = qty_expected

            # Invoice selector is disabled for Damaged / Wrong item (needs_invoice=False),
            # but we still set a deterministic value
            st.session_state[invoice_key] = "In invoice"

    for ln in sorted(lines, key=lambda x: _line_name(x, ctx.products_by_id).lower()):
        lid = int(ln.id)
        name = _line_name(ln, ctx.products_by_id)
        unit = _line_unit(ln, ctx.products_by_id)
        qty_ordered = _safe_float(getattr(ln, "quantity", 0.0), 0.0)
        fu = ctx.followups_by_key.get((prov, lid))
        qty_expected = _derive_expected_qty(ln, fu)

        base = f"recv_{int(order.id)}_{prov}_{lid}_"
        status_key = base + "status"
        issue_key = base + "issue_qty"
        invoice_key = base + "invoice"

        # helper key to pass expected qty into callback safely
        qty_expected_key = base + "qty_expected"

        # defaults from existing followup
        default_status = "OK"
        default_issue_qty = 0.0
        default_invoice = "Not in invoice"

        if fu:
            vc = _s(getattr(fu, "venue_comment", None)).lower()
            if "[venue_status=missing]" in vc:
                default_status = "Missing"
            elif "[venue_status=damaged]" in vc:
                default_status = "Damaged"
            elif "[venue_status=wrong_item]" in vc:
                default_status = "Wrong item"
            elif "[venue_status=ok]" in vc:
                default_status = "OK"

            default_issue_qty = _safe_float(getattr(fu, "qty_invoiced", None), 0.0)
            inv_db = getattr(fu, "invoice_listed", None)
            if inv_db is True:
                default_invoice = "In invoice"
            elif inv_db is False:
                default_invoice = "Not in invoice"
            else:
                default_invoice = "Not in invoice"

        # Initialize state
        if status_key not in st.session_state:
            st.session_state[status_key] = default_status
        if issue_key not in st.session_state:
            st.session_state[issue_key] = float(default_issue_qty)
        if invoice_key not in st.session_state:
            st.session_state[invoice_key] = default_invoice

        # Always refresh expected qty for callback (it can change based on followups)
        st.session_state[qty_expected_key] = float(qty_expected)

        # Normalize old values
        if st.session_state[status_key] not in STATUS_OPTIONS:
            st.session_state[status_key] = default_status
        if st.session_state[invoice_key] not in INVOICE_OPTIONS:
            st.session_state[invoice_key] = default_invoice

        status_now = _s(st.session_state.get(status_key, default_status))
        needs_issue_qty = status_now in {"Missing", "Damaged", "Wrong item"}
        needs_invoice = status_now == "Missing"

        c0, c1, c2, c3 = st.columns([2.4, 1.25, 1.1, 1.2], vertical_alignment="center")

        with c0:
            st.markdown(f"**{name}**")
            st.caption(f"Expected: {qty_expected:g} {unit} · Ordered: {qty_ordered:g} {unit}")

        # Status selectbox WITH callback (now allowed, since we're not in st.form)
        with c1:
            st.selectbox(
                "Status",
                STATUS_OPTIONS,
                index=STATUS_OPTIONS.index(st.session_state[status_key]),
                key=status_key,
                label_visibility="collapsed",
                on_change=_on_status_change,
                kwargs=dict(
                    status_key=status_key,
                    issue_key=issue_key,
                    invoice_key=invoice_key,
                    qty_expected_key=qty_expected_key,
                ),
            )

        # Re-read after widget (callback may have changed state)
        status_now = _s(st.session_state.get(status_key, default_status))
        needs_issue_qty = status_now in {"Missing", "Damaged", "Wrong item"}
        needs_invoice = status_now == "Missing"

        # Safety: if user flips to OK, keep issue qty 0 even if they previously had a value
        if status_now == "OK":
            st.session_state[issue_key] = 0.0
            
        # Enforce bounds based on expected qty
        if needs_issue_qty:
            # clamp to [1, qty_expected]
            cur = _safe_float(st.session_state.get(issue_key, 0.0), 0.0)
            if cur < 1:
                st.session_state[issue_key] = 1.0
            elif cur > float(qty_expected):
                st.session_state[issue_key] = float(qty_expected)
        else:
            # OK status => no issue
            st.session_state[issue_key] = 0.0

        with c2:
            st.number_input(
                "Issue qty",
                min_value=1.0 if needs_issue_qty else 0.0,
                max_value=float(qty_expected) if needs_issue_qty else 0.0,
                value=float(st.session_state.get(issue_key, 0.0)),
                step=1.0,
                disabled=not needs_issue_qty,
                key=issue_key,
                label_visibility="collapsed",
            )

        with c3:
            st.selectbox(
                "Invoice listed",
                INVOICE_OPTIONS,
                index=INVOICE_OPTIONS.index(st.session_state[invoice_key]),
                disabled=not needs_invoice,
                key=invoice_key,
                label_visibility="collapsed",
            )




# =============================
# Tickets + Save-all
# =============================

def _upsert_followup_and_ticket(
    *,
    ctx: OrderContext,
    provider: str,
    line: OrderLine,
    issue_status: str,
    issue_qty: float,
    invoice_listed: Optional[bool],
    venue_qty: float,
    invoice_number: Optional[str],
    unit: str,
    product_name: str,
) -> Tuple[bool, bool]:
    """Returns flags: (invoice_discrepancy, operational_missing)."""
    prov = norm_provider(provider)
    order = ctx.order
    lid = int(line.id)
    qty_ordered = _safe_float(getattr(line, "quantity", 0.0), 0.0)

    any_invoice_discrepancy = False
    any_operational_missing = False

    # tag in venue_comment
    tag = f"[VENUE_STATUS={issue_status}]"

    with get_session() as s:
        fu = s.exec(
            select(ProviderLineFollowUp).where(
                ProviderLineFollowUp.order_id == int(order.id),
                ProviderLineFollowUp.provider_name == prov,
                ProviderLineFollowUp.order_line_id == lid,
            )
        ).first()
        if not fu:
            fu = ProviderLineFollowUp(
                venue_id=int(order.venue_id),
                order_id=int(order.id),
                provider_name=prov,
                order_line_id=lid,
                qty_ordered=qty_ordered,
            )

        fu.venue_qty = float(venue_qty)
        fu.qty_invoiced = float(issue_qty)  # reuse field to store issue qty
        fu.invoice_listed = invoice_listed
        fu.venue_comment = (fu.venue_comment or "").split("[VENUE_STATUS=")[0].strip() + " " + tag
        fu.updated_at = _now()
        fu.updated_by = "venue"
        s.add(fu)

        # Ticket logic
        kind: Optional[str] = None
        if issue_status == "missing":
            if invoice_listed is False:
                kind = "operational_missing"
                any_operational_missing = True
            else:
                kind = "invoice_discrepancy"
                any_invoice_discrepancy = True
        elif issue_status == "damaged":
            kind = "damaged"
            any_invoice_discrepancy = True  # requires supplier action
        elif issue_status == "wrong_item":
            kind = "wrong_item"
            any_invoice_discrepancy = True  # requires supplier action

        if kind:
            t = s.exec(
                select(SeguimientoTicket).where(
                    SeguimientoTicket.order_id == int(order.id),
                    SeguimientoTicket.provider_name == prov,
                    SeguimientoTicket.order_line_id == lid,
                    SeguimientoTicket.kind == kind,
                )
            ).first()
            if not t:
                initial_state = "open_internal" if kind == "operational_missing" else "open"
                t = SeguimientoTicket(
                    venue_id=int(order.venue_id),
                    order_id=int(order.id),
                    provider_name=prov,
                    order_line_id=lid,
                    kind=kind,
                    state=initial_state,
                )


            t.product_name = product_name
            t.unit = unit
            t.qty_invoiced = float(issue_qty)
            t.invoice_number = invoice_number
            t.updated_at = _now()
            s.add(t)

        s.commit()

    return any_invoice_discrepancy, any_operational_missing


def _clear_receive_form_state(order_id: int, provider: str) -> None:
    prefix = f"recv_{int(order_id)}_{norm_provider(provider)}_"
    for k in list(st.session_state.keys()):
        if k.startswith(prefix):
            del st.session_state[k]


def save_all_received_for_provider(*, ctx: OrderContext, provider: str) -> Tuple[bool, str]:
    prov = norm_provider(provider)
    order = ctx.order
    lines = ctx.lines_by_provider.get(prov, []) or []
    if not lines:
        return False, "No lines for provider"

    any_invoice_discrepancy = False
    any_operational_missing = False

    receipt = ctx.receipts_by_provider.get(prov)
    invoice_number = _s(getattr(receipt, "invoice_number", None)) if receipt else None

    for ln in lines:
        if getattr(ln, "id", None) is None:
            continue
        lid = int(ln.id)
        unit = _line_unit(ln, ctx.products_by_id)
        name = _line_name(ln, ctx.products_by_id)

        fu_existing = ctx.followups_by_key.get((prov, lid))
        qty_expected = _derive_expected_qty(ln, fu_existing)
        qty_ordered = _safe_float(getattr(ln, "quantity", 0.0), 0.0)

        base = f"recv_{int(order.id)}_{prov}_{lid}_"
        status_ui = (st.session_state.get(base + "status") or "OK").strip()
        issue_qty = _safe_float(st.session_state.get(base + "issue_qty"), 0.0)
        inv_ui = st.session_state.get(base + "invoice")

        status_map = {"OK": "ok", "Missing": "missing", "Damaged": "damaged", "Wrong item": "wrong_item"}
        issue_status = status_map.get(status_ui, "ok")

        if issue_status == "ok":
            venue_qty = qty_expected
            issue_qty = 0.0
            invoice_listed = None
        else:
            max_base = qty_expected if qty_expected > 0 else qty_ordered
            issue_qty = max(0.0, min(max_base, issue_qty))
            venue_qty = max(0.0, max_base - issue_qty)
            if issue_status == "missing":
                if inv_ui == "In invoice":
                    invoice_listed = True
                elif inv_ui == "Not in invoice":
                    invoice_listed = False
                else:
                    invoice_listed = None
            else:
                invoice_listed = None

        inv_disc, op_miss = _upsert_followup_and_ticket(
            ctx=ctx,
            provider=prov,
            line=ln,
            issue_status=issue_status,
            issue_qty=issue_qty,
            invoice_listed=invoice_listed,
            venue_qty=venue_qty,
            invoice_number=invoice_number,
            unit=unit,
            product_name=name,
        )
        any_invoice_discrepancy = any_invoice_discrepancy or inv_disc
        any_operational_missing = any_operational_missing or op_miss

    # update workflow state
    if any_invoice_discrepancy:
        _set_workflow_state(
            venue_id=int(order.venue_id),
            order_id=int(order.id),
            provider=prov,
            to_state="INVOICE_DISCREPANCY",
            actor_role="venue",
            actor="venue",
            note="Venue saved receiving: discrepancies detected.",
        )
    elif any_operational_missing:
        _set_workflow_state(
            venue_id=int(order.venue_id),
            order_id=int(order.id),
            provider=prov,
            to_state="OPERATIONAL_MISSING_PRODUCT",
            actor_role="venue",
            actor="venue",
            note="Venue saved receiving: operational missing product.",
        )
    else:
        _set_workflow_state(
            venue_id=int(order.venue_id),
            order_id=int(order.id),
            provider=prov,
            to_state="RECEIVED",
            actor_role="venue",
            actor="venue",
            note="Venue saved receiving: all OK.",
        )

    _clear_receive_form_state(int(order.id), prov)

    # reset provider selector
    idx_key = f"recv_current_provider_idx_{int(order.id)}"
    sel_key = f"recv_provider_sel_{int(order.id)}"
    st.session_state[idx_key] = 0
    if sel_key in st.session_state:
        del st.session_state[sel_key]

    return True, "Saved"

def _normalize_solution_meta(sol: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize supplier solution metadata (defensive / backward-compatible).

        Why:
        - Some venues were entering 'supplementary_delivery' / 're_delivery' into the *ref* field
            while choosing 'credit_note' (because the UI had only a free text input).
            That makes the workflow look like credit note even though the intent is re-delivery.
        - We also want to support 'mixed' item lists where some items are CN and others are re-delivery,
            based on per-item 'reason' markers.

        This function only normalizes fields for rendering; DB remains unchanged.
        """
        sol = dict(sol or {})
        res = _s(sol.get("resolution")).strip().lower()
        ref = _s(sol.get("ref")).strip().lower()
        cn_no = _s(sol.get("credit_note_invoice")).strip()

        # normalize aliases
        if res in {"creditnote", "credit-note"}:
            res = "credit_note"
        if res in {"redelivery", "re-delivery", "re_delivery"}:
            res = "re_delivery"

        # Back-compat heuristic: decision-type accidentally entered in "ref"
        if res == "credit_note" and (not cn_no) and ref in {"supplementary_delivery", "re_delivery", "re-delivery", "redelivery"}:
            res = "re_delivery" if "re" in ref else "supplementary_delivery"

        sol["resolution"] = res
        return sol

def _is_redelivery_item(it: Dict[str, Any]) -> bool:
    r = _s(it.get("reason")).strip().lower()
    return r in {"supplementary_delivery", "re_delivery", "re-delivery", "redelivery", "supplementary", "redelivery_item"}
# =============================
# Supplier resolution + verify
# =============================

def request_supplier_resolution(venue_id: int, order_id: int, provider: str) -> Tuple[bool, str]:
    """Send email to supplier with tracking link."""
    prov = norm_provider(provider)
    with get_session() as s:
        p = s.exec(select(Provider).where(Provider.venue_id == int(venue_id), Provider.name == prov)).first()
        if not p:
            p = s.exec(select(Provider).where(Provider.venue_id == int(venue_id), Provider.name == provider)).first()
        emails = []
        if p and getattr(p, "order_email", None):
            emails = [x.strip() for x in (p.order_email or "").split("|") if x.strip()]
        if not emails and p and getattr(p, "emails", None):
            emails = [x.strip() for x in (p.emails or "").split("|") if x.strip()]
        if not emails:
            return False, "No supplier email configured."

    link = build_seguimiento_url(order_id=int(order_id), provider_name=prov, role=ROLE_SUPPLIER, page_path="seguimiento")
    subject = f"Action required — Invoice discrepancy ({prov})"
    body = f"Please open the link and choose a resolution (credit note or re-delivery):\n\n{link}\n"

    try:
        send_smtp_email(to=emails, subject=subject, text_body=body)
    except Exception as e:
        return False, f"Email failed: {e}"

    _set_workflow_state(
        venue_id=int(venue_id),
        order_id=int(order_id),
        provider=prov,
        to_state="WAITING_SUPPLIER_ACTION",
        actor_role="venue",
        actor="venue",
        note="Requested supplier decision via link.",
    )

    return True, link


def supplier_decision_from_venue(ctx: OrderContext, provider: str, decision: str, ref: str, comment: str) -> str:
    """Venue manually records supplier decision."""
    prov = norm_provider(provider)
    decision = (decision or "").strip().lower()
    if decision not in {"credit_note", "re_delivery"}:
        return "Invalid decision"

    to_state = "SUPPLIER_CREDIT_NOTE_ISSUED" if decision == "credit_note" else "SUPPLEMENTARY_DELIVERY_SENT"
    note = " | ".join([decision, _s(ref)]).strip(" |")
    if comment:
        note = (note + " · " + comment).strip()

    _set_workflow_state(
        venue_id=int(ctx.order.venue_id),
        order_id=int(ctx.order.id),
        provider=prov,
        to_state=to_state,
        actor_role="venue",
        actor="venue",
        note=note,
    )

    # mark discrepancy tickets as supplier-action-done
    with get_session() as s:
        tickets = list(
            s.exec(
                select(SeguimientoTicket).where(
                    SeguimientoTicket.order_id == int(ctx.order.id),
                    SeguimientoTicket.provider_name == prov,
                    SeguimientoTicket.kind.in_(["invoice_discrepancy", "damaged", "wrong_item"]),
                )
            ).all()
        )
        for t in tickets:
            if t.state == "open":
                t.state = "SUPPLIER_ACTION_DONE"
                t.resolution_note = note or None
                t.updated_at = _now()
                s.add(t)
        s.commit()

    return "ok"


def venue_verify_and_close(*, ctx: OrderContext, provider: str, mode: str) -> str:
    """Venue-side verification step.

    Why this exists:
    - After supplier answers, we show *two* independent previews (credit note + re-delivery).
    - The venue may need to verify each outcome separately in mixed scenarios.
    - Therefore we only close the subset of tickets that match the requested `mode`,
      and we close the workflow only when nothing verification-relevant remains.
    """

    prov = norm_provider(provider)
    mode = (mode or "").strip().lower()
    if mode not in {"credit_note", "supplementary", "reject"}:
        return "Invalid mode"

    order = ctx.order

    def _ticket_mode(t: SeguimientoTicket) -> str:
        meta = _parse_ticket_resolution_note(_s(getattr(t, "resolution_note", None)))
        r = _s(meta.get("resolution")).strip().lower()
        if r == "credit_note":
            return "credit_note"
        if r in {"supplementary_delivery", "re_delivery", "re-delivery", "redelivery"}:
            return "supplementary"
        if r == "reject":
            return "reject"
        # Fallback: if supplier didn't write a per-ticket resolution, we treat it as current mode
        return mode

    resolution_state = (
        "resolved_credit_note_verified"
        if mode == "credit_note"
        else ("resolved_supplementary_received" if mode == "supplementary" else "resolved_reject_accepted")
    )

    # 1) Update only the relevant tickets
    with get_session() as s:
        tickets = list(
            s.exec(
                select(SeguimientoTicket).where(
                    SeguimientoTicket.order_id == int(order.id),
                    SeguimientoTicket.provider_name == prov,
                    SeguimientoTicket.kind.in_(["invoice_discrepancy", "damaged", "wrong_item"]),
                )
            ).all()
        )

        touched = 0
        for t in tickets:
            if t.state not in {"open", "SUPPLIER_ACTION_DONE"}:
                continue
            if _ticket_mode(t) != mode:
                continue

            t.state = resolution_state
            t.resolved_at = _now()
            t.resolution_note = (t.resolution_note or "") + f"\n[VENUE] Verified {mode}."
            t.updated_at = _now()
            s.add(t)
            touched += 1

        if touched == 0:
            # Avoid surprising "Closed" when there is nothing to verify for that mode.
            s.commit()
            return "Nothing to verify for this mode"

        # Mark receipt as verified only when *all* verification-relevant tickets are resolved
        # (keeps accounting flow sane for mixed outcomes).
        remaining = [
            t
            for t in tickets
            if t.state in {"open", "SUPPLIER_ACTION_DONE"}
        ]

        if not remaining:
            receipt = s.exec(
                select(ProviderReceipt).where(
                    ProviderReceipt.order_id == int(order.id),
                    ProviderReceipt.provider_name == prov,
                )
            ).first()
            if receipt:
                receipt.received = True
                receipt.received_at = _now()
                receipt.received_by = "venue"
                receipt.updated_at = _now()
                receipt.updated_by = "venue"
                s.add(receipt)

        s.commit()

    # 2) Move workflow forward:
    # - If nothing left → CLOSED
    # - Else keep it in a supplier-acted state that matches what's still pending verification
    ctx2 = _load_order_context(int(order.venue_id), int(order.id))
    wf2 = ctx2.workflows_by_provider.get(prov)
    remaining_modes: set[str] = set()

    if wf2:
        with get_session() as s:
            rem = list(
                s.exec(
                    select(SeguimientoTicket).where(
                        SeguimientoTicket.order_id == int(order.id),
                        SeguimientoTicket.provider_name == prov,
                        SeguimientoTicket.kind.in_(["invoice_discrepancy", "damaged", "wrong_item"]),
                        SeguimientoTicket.state.in_(["open", "SUPPLIER_ACTION_DONE"]),
                    )
                ).all()
            )
        for t in rem:
            remaining_modes.add(_ticket_mode(t))

    if not remaining_modes:
        _set_workflow_state(
            venue_id=int(order.venue_id),
            order_id=int(order.id),
            provider=prov,
            to_state="CLOSED",
            actor_role="venue",
            actor="venue",
            note=f"Venue verified supplier resolution ({mode}) and closed.",
        )
    else:
        # Best-effort: pick a representative state to keep UI accurate
        if "credit_note" in remaining_modes:
            to_state = "SUPPLIER_CREDIT_NOTE_ISSUED"
        elif "supplementary" in remaining_modes:
            to_state = "SUPPLEMENTARY_DELIVERY_SENT"
        else:
            to_state = "SUPPLIER_REJECTED"

        _set_workflow_state(
            venue_id=int(order.venue_id),
            order_id=int(order.id),
            provider=prov,
            to_state=to_state,
            actor_role="venue",
            actor="venue",
            note=f"Venue verified {mode}; remaining pending verification: {', '.join(sorted(remaining_modes))}.",
        )

    return "ok"
def resolve_operational_missing(*, ctx: OrderContext, provider: str) -> str:
    prov = norm_provider(provider)
    order = ctx.order

    with get_session() as s:
        tickets = list(
            s.exec(
                select(SeguimientoTicket).where(
                    SeguimientoTicket.order_id == int(order.id),
                    SeguimientoTicket.provider_name == prov,
                    SeguimientoTicket.kind == "operational_missing",
                )
            ).all()
        )
        for t in tickets:
            if t.state == "open":
                t.state = "resolved_internal"
                t.resolved_at = _now()
                t.resolution_note = "[VENUE] Resolved internally."
                t.updated_at = _now()
                s.add(t)
        s.commit()

    _set_workflow_state(
        venue_id=int(order.venue_id),
        order_id=int(order.id),
        provider=prov,
        to_state="CLOSED",
        actor_role="venue",
        actor="venue",
        note="Operational missing resolved; closed.",
    )

    return "ok"


# =============================
# Operational-missing helpers
# =============================

def _close_ticket(*, ticket_id: int, new_state: str, note: str, actor: str = "venue") -> None:
    """Close/resolve a SeguimientoTicket safely."""
    tid = int(ticket_id)
    with get_session() as s:
        t = s.exec(select(SeguimientoTicket).where(SeguimientoTicket.id == tid)).first()
        if not t:
            return
        t.state = _s(new_state) or t.state
        t.resolved_at = _now()
        t.updated_at = _now()
        prev = (t.resolution_note or "").strip()
        add = f"[VENUE] {note}" if note else "[VENUE] Closed"
        t.resolution_note = (prev + "\n" + add).strip() if prev else add
        s.add(t)
        s.commit()


def _get_or_create_draft_order(*, venue_id: int, actor: str = "venue") -> int:
    """Return a draft order id for the venue (creates one if missing)."""
    vid = int(venue_id)
    with get_session() as s:
        # best-effort: find latest draft
        q = (
            select(Order)
            .where(Order.venue_id == vid)
            .where((Order.status == "draft") | (Order.status == "borrador"))
            .order_by(Order.created_at.desc())
        )
        o = s.exec(q).first()
        if o and getattr(o, "id", None) is not None:
            return int(o.id)

        # create new draft
        o = Order(
            venue_id=vid,
            status="draft",
            title="Borrador",
            created_at=_now(),
            created_by=actor,
        )
        s.add(o)
        s.commit()
        s.refresh(o)
        return int(o.id)


def _create_new_draft_order(*, venue_id: int, actor: str, title: str) -> int:
    vid = int(venue_id)
    with get_session() as s:
        o = Order(
            venue_id=vid,
            status="draft",
            title=title or "Nuevo pedido",
            created_at=_now(),
            created_by=actor,
        )
        s.add(o)
        s.commit()
        s.refresh(o)
        return int(o.id)


def _add_line_to_order(
    *, venue_id: int, order_id: int, product_id: int | None,
    provider: str, name: str, qty: float, unit: str, actor: str = "venue",
    chip: str = ""
) -> None:
    with get_session() as s:
        chip_txt = f" [{chip}]" if chip else ""
        ln = OrderLine(
            venue_id=int(venue_id),
            order_id=int(order_id),
            product_id=int(product_id) if product_id else None,
            provider=_s(provider) or None,
            spoken_name=(_s(name) + chip_txt).strip() or None,
            quantity=float(qty or 0.0),
            unit=_s(unit) or None,
            updated_at=_now(),
            updated_by=actor,
        )
        s.add(ln)
        s.commit()



def _search_similar_products(*, venue_id: int, query_name: str, limit: int = 50) -> list[Product]:
    """Best-effort similarity search by name (simple contains tokens)."""
    qn = (_s(query_name) or "").strip().lower()
    if not qn:
        return []
    tokens = [t for t in re.split(r"\W+", qn) if len(t) >= 3][:4]
    if not tokens:
        tokens = [qn[:6]]
    with get_session() as s:
        # Start broad: venue products
        ps = list(s.exec(select(Product).where(Product.venue_id == int(venue_id))).all())
    def score(p: Product) -> float:
        nm = (_s(getattr(p, 'name', ''))).lower()
        if not nm:
            return 0.0
        hits = sum(1 for t in tokens if t in nm)
        return hits / max(1, len(tokens))
    ranked = [(score(p), p) for p in ps]
    ranked = [rp for rp in ranked if rp[0] > 0]
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in ranked[:limit]]


# =============================
# Incidences: open filters
# =============================

def _provider_open_tickets(ctx: OrderContext, provider: str) -> List[SeguimientoTicket]:
    prov = norm_provider(provider)
    out = []
    for t in ctx.tickets_by_provider.get(prov, []) or []:
        if getattr(t, "resolved_at", None) is not None:
            continue
        if _s(getattr(t, "state", None)).lower().startswith("resolved"):
            continue
        # open or waiting verification
        out.append(t)
    return out


def _parse_supplier_solution_meta(note: str) -> Dict[str, Any]:
    """Parse workflow.note written by supplier/venue.

    Expected patterns (flexible):
      - credit_note | ref=... | credit_note_invoice=... | items=...
      - supplementary_delivery | ref=... | eta=YYYY-MM-DD 08:00-14:00 | items=...

    Returns:
      {
        "resolution": "credit_note"|"supplementary_delivery"|"re_delivery"|"",
        "ref": str,
        "credit_note_invoice": str,
        "eta": str,
        "items": [{"name":...,"qty":...,"unit":...,"reason":...}, ...]
      }
    """
    raw = _s(note)
    parts = [p.strip() for p in raw.replace("·", "|").split("|") if p.strip()]
    out: Dict[str, Any] = {"resolution": "", "ref": "", "credit_note_invoice": "", "eta": "", "items": []}
    if not parts:
        return out

    out["resolution"] = parts[0].strip()

    def _parse_items(val: str) -> List[Dict[str, str]]:
        # NAME:QTYUNIT(reason)/NAME2:QTYUNIT(reason)
        # Older versions may use '|'
        items: List[Dict[str, str]] = []
        chunks = []
        for sep in ["/", "|"]:
            if sep in (val or ""):
                chunks = [c for c in (val or "").split(sep) if c.strip()]
                break
        if not chunks:
            chunks = [val] if (val or "").strip() else []

        for chunk in chunks:
            c = chunk.strip()
            name, rest = (c.split(":", 1) + [""])[:2]
            name = name.strip()
            reason = ""
            if "(" in rest and rest.endswith(")"):
                before, reason = rest.rsplit("(", 1)
                reason = reason[:-1].strip()
            else:
                before = rest

            # split qty+unit: e.g. 1.5KG or 1TEM
            before = before.strip()
            qty_str = ""
            unit_str = ""
            for i, ch in enumerate(before):
                if not (ch.isdigit() or ch in ".,"):
                    qty_str = before[:i].strip().replace(",", ".")
                    unit_str = before[i:].strip()
                    break
            if not qty_str:
                qty_str = before.replace(",", ".")

            try:
                qty = float(qty_str) if qty_str else 0.0
            except Exception:
                qty = 0.0

            if name:
                items.append(
                    {
                        "name": name,
                        "qty": f"{qty:g}" if qty else (qty_str or ""),
                        "unit": unit_str,
                        "reason": reason,
                    }
                )
        return items

    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            k = k.strip().lower()
            v = v.strip()
            if k == "ref":
                out["ref"] = v
            elif k in {"credit_note_invoice", "credit_note_no"}:
                out["credit_note_invoice"] = v
            elif k == "eta":
                out["eta"] = v
            elif k == "items":
                out["items"] = _parse_items(v)
        else:
            # if venue manually recorded: note = "credit_note | <ref>"
            if not out.get("ref") and out.get("resolution"):
                out["ref"] = p

    return out



# =========================================================
# Ticket-level supplier resolution parsing (source of truth)
# =========================================================
_META_KV_RE = re.compile(r"^\s*([a-zA-Z_]+)\s*=\s*(.+?)\s*$")

def _parse_ticket_resolution_note(note: str) -> Dict[str, Any]:
    """Parse SeguimientoTicket.resolution_note written by supplier.

    Format written in seguimiento.py:
      "[SUPPLIER] <resolution> | key=value | key=value"
    Examples:
      "[SUPPLIER] credit_note | credit_note_invoice=CN-123 | invoice=INV-1"
      "[SUPPLIER] supplementary_delivery | eta=2026-01-22 08:00-14:00 | invoice=INV-1"
      "[SUPPLIER] re_delivery | eta=..."
    """
    txt = _s(note)
    if not txt:
        return {"resolution": "", "credit_note_invoice": "", "eta": "", "invoice": ""}

    if txt.startswith("[SUPPLIER]"):
        txt = txt[len("[SUPPLIER]"):].strip()

    # allow either " | " or newlines (comment is stored after newline)
    first_line = txt.splitlines()[0].strip()
    parts = [p.strip() for p in first_line.replace("·", "|").split("|") if p.strip()]
    if not parts:
        return {"resolution": "", "credit_note_invoice": "", "eta": "", "invoice": ""}

    out: Dict[str, Any] = {"resolution": parts[0].strip(), "credit_note_invoice": "", "eta": "", "invoice": ""}

    for p in parts[1:]:
        m = _META_KV_RE.match(p)
        if not m:
            continue
        k = (m.group(1) or "").strip().lower()
        v = (m.group(2) or "").strip()
        if k in {"credit_note_invoice", "credit_note"}:
            out["credit_note_invoice"] = v
        elif k == "eta":
            out["eta"] = v
        elif k in {"invoice", "invoice_number"}:
            out["invoice"] = v

    return out

def _render_incidences_cards(
    ctx: OrderContext,
    providers: List[str],
    show_prices: bool = False,
    include_iva: bool = False,
) -> None:
    order = ctx.order
    actor = "venue"
    open_any = False
    

    def _invoice_date_for_provider(provn: str) -> datetime:
        """Best-effort invoice date for display (Greece reality: CN may arrive later).

        We don't store a dedicated invoice date, so we use:
        1) invoice_number_set_at (if invoice # was entered)
        2) order.created_at
        """
        receipt = ctx.receipts_by_provider.get(provn)
        dt = getattr(receipt, "invoice_number_set_at", None) if receipt else None
        if isinstance(dt, datetime):
            return dt
        when = getattr(order, "created_at", None)
        return when if isinstance(when, datetime) else _now()

    def _fmt_dt(dt: Optional[datetime]) -> str:
        if not isinstance(dt, datetime):
            return "—"
        return dt.strftime("%Y-%m-%d")

    def _credit_reason(kind: str) -> str:
        k = (kind or "").strip().lower()
        if k == "invoice_discrepancy":
            return "Invoice correction"
        if k == "missing":
            return "Missing items"
        if k == "damaged":
            return "Damaged"
        if k == "wrong_item":
            return "Wrong item"
        return k.replace("_", " ") or "Correction"

    


    def _credit_note_preview(prov: str, provn: str, open_t: List[SeguimientoTicket], sol: Dict[str, Any]) -> None:
        """Render an *expected* credit note (preview) based on tickets AND supplier note items.

        Key rule:
        - If supplier note has explicit items, we only show the items that are meant for credit note
            (i.e., not flagged as re-delivery items).
        """
        wf = ctx.workflows_by_provider.get(provn)
        sol = _normalize_solution_meta(sol)
        resolution = _s(sol.get("resolution", "")).lower()

        # Show CN preview only when:
        # - supplier explicitly chose credit_note, OR
        # - supplier note items contain credit-note items (mixed-mode support)
        items = sol.get("items") or []
        has_cn_items = any((isinstance(it, dict) and not _is_redelivery_item(it)) for it in (items or []))
        if resolution != "credit_note" and not has_cn_items:
            return

        receipt = ctx.receipts_by_provider.get(provn)
        inv_no = _s(getattr(receipt, "invoice_number", None)) or "—"
        inv_dt = _invoice_date_for_provider(provn)
        cn_no = _s(sol.get("credit_note_invoice")) or "—"

        # Build best-effort mapping name->item metadata from supplier note
        sol_items: List[Dict[str, Any]] = [it for it in (items or []) if isinstance(it, dict)]
        sol_by_name = {_s(it.get("name")).strip().lower(): it for it in sol_items if _s(it.get("name")).strip()}
        credit_candidates: List[SeguimientoTicket] = []

        if sol_by_name:
            # Only tickets that match supplier items, and those items are not re-delivery flagged
            for t in open_t:
                nm = _s(getattr(t, "product_name", None)).strip().lower()
                it = sol_by_name.get(nm)
                if it and not _is_redelivery_item(it):
                    credit_candidates.append(t)
        else:
            # Fallback: legacy behavior (credit kinds)
            credit_kinds = {"invoice_discrepancy", "damaged", "wrong_item"}
            credit_candidates = [t for t in open_t if (_s(getattr(t, "kind", None)).lower() in credit_kinds)]

        if not credit_candidates:
            st.markdown(
                "<div class='voi-card' style='border-color:#fde68a;background:#fffbeb'>"
                "<div class='voi-title'>🧾 Expected credit note (preview)</div>"
                "<div class='voi-muted'>No eligible items found yet.</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            return

        rows: List[Dict[str, Any]] = []
        total_net = 0.0
        total_vat = 0.0

        line_by_id_local: Dict[int, OrderLine] = {
            int(getattr(ln, "id", 0) or 0): ln
            for ln in (ctx.lines_by_provider.get(provn, []) or [])
            if int(getattr(ln, "id", 0) or 0)
        }

        for t in credit_candidates:
            
            lid = int(getattr(t, "order_line_id", 0) or 0)
            ln = line_by_id_local.get(lid)

            name = _s(getattr(t, "product_name", None)) or (_line_name(ln, ctx.products_by_id) if ln else "Product")
            unit = _s(getattr(t, "unit", None)) or (_line_unit(ln, ctx.products_by_id) if ln else "unit")

            # qty: prefer supplier note qty (if present), else ticket qty_invoiced
            qty = None
            it = sol_by_name.get(_s(getattr(t, "product_name", None)).strip().lower()) if sol_by_name else None
            if it:
                try:
                    qty = float(it.get("qty"))
                except Exception:
                    qty = None
            if qty is None:
                qty = _safe_float(getattr(t, "qty_invoiced", None), 0.0)

            if qty <= 0:
                continue

            pid = None
            if ln is not None:
                pid_raw = getattr(ln, "product_id", None)
                if pid_raw not in (None, "", 0, "0"):
                    try:
                        pid = int(pid_raw)
                    except Exception:
                        pid = None

            gross_unit = _price_for_pid(ctx.products_by_id, pid)
            pricing = (
                _pricing_for_line(
                    venue_id=int(ctx.order.venue_id),
                    providers_by_name=ctx.providers_by_name,
                    provider_name=prov,
                    pid=pid,
                    qty=float(qty),
                    gross_unit=float(gross_unit) if gross_unit > 0 else 0.0,
                )
                if (gross_unit > 0 and qty > 0)
                else {"net_unit": 0.0, "discount_pct": 0.0, "applied": False}
            )

            net_unit = float(pricing.get("net_unit", 0.0) or 0.0)
            disc_pct = float(pricing.get("discount_pct", 0.0) or 0.0)

            net_amount = qty * net_unit
            vat_pct = _iva_pct_for_pid(ctx.products_by_id, pid, 21.0) if include_iva else 0.0
            vat_eur = net_amount * (vat_pct / 100.0) if include_iva else 0.0
            total = net_amount + vat_eur

            total_net += net_amount
            total_vat += vat_eur

            rows.append(
                {
                    "Product": name,
                    "Qty": float(qty),
                    "Unit": unit,
                    "Net €/unit": float(net_unit),
                    "Disc%": float(disc_pct) if disc_pct > 0 else 0.0,
                    "Net amount": float(net_amount),
                    "VAT%": float(vat_pct) if include_iva else 0.0,
                    "VAT €": float(vat_eur) if include_iva else 0.0,
                    "Total": float(total),
                    "Reason": _credit_reason(_s(getattr(t, "kind", None))),
                }
            )

        st.markdown(
            "<div class='voi-card' style='border-color:#fde68a;background:#fffbeb'>"
            "<div class='voi-title'>🧾 Expected credit note (preview)</div>"
            f"<div class='voi-muted'>Reference invoice: <b>{html.escape(inv_no)}</b> · Date: <b>{html.escape(_fmt_dt(inv_dt))}</b></div>"
            f"<div class='voi-muted'>Credit note number: <b>{html.escape(cn_no)}</b></div>"
            "</div>",
            unsafe_allow_html=True,
        )

        reasons = sorted({_credit_reason(_s(getattr(t, "kind", None))) for t in credit_candidates})
        reason_txt = ", ".join(reasons) if reasons else "Correction"
        st.markdown(
            "- **Reference to original invoice:** "
            f"`{inv_no}` · {_fmt_dt(inv_dt)}\n"
            "- **Reason for issuance:** "
            f"{reason_txt}\n"
            "- **Products/quantities being credited:** see table below\n"
            + (
                "- **VAT + total credit amount:** see totals below"
                if include_iva
                else "- **Total credit amount (net):** see totals below"
            )
        )

        if rows:
            df = pd.DataFrame(rows)
            cols = ["Product", "Qty", "Unit", "Net €/unit", "Disc%", "Net amount"]
            if include_iva:
                cols += ["VAT%", "VAT €", "Total"]
            cols += ["Reason"]
            df = df[[c for c in cols if c in df.columns]]
            st.dataframe(df, use_container_width=True, hide_index=True)

            if include_iva:
                st.markdown(
                    f"**Total credit (net):** {total_net:,.2f} · **VAT:** {total_vat:,.2f} · **Total credit:** {(total_net + total_vat):,.2f}"
                )
            else:
                st.markdown(f"**Total credit (net):** {total_net:,.2f}")
            st.caption("Preview only: based on catalog + discount rules. Supplier credit note is the official document.")

    def _redelivery_preview(
        ctx: OrderContext,
        prov: str,
        provn: str,
        open_t: List[SeguimientoTicket],
        sol: Dict[str, Any],
    ) -> None:
        """Render an *expected* re-delivery preview.

        Source of truth preference:
        1) If tickets contain per-line supplier decisions (resolution_note), the caller should pass only those tickets.
        2) Otherwise we fall back to workflow note items/resolution.
        """
        sol = _normalize_solution_meta(sol or {})
        resolution = _s(sol.get("resolution", "")).strip().lower()

        # Determine whether we should show this block at all
        items = sol.get("items") or []
        has_rd_items = any(isinstance(it, dict) and _is_redelivery_item(it) for it in items)
        is_rd = resolution in {"supplementary_delivery", "re_delivery"}

        if not has_rd_items and not is_rd:
            # If caller passed ticket subset, still allow rendering (mixed-mode)
            # by checking ticket-level resolution_note.
            any_ticket_rd = False
            for t in (open_t or []):
                meta = _parse_ticket_resolution_note(_s(getattr(t, "resolution_note", None)))
                r = _s(meta.get("resolution")).strip().lower()
                if r in {"supplementary_delivery", "re_delivery"}:
                    any_ticket_rd = True
                    break
            if not any_ticket_rd:
                return

        # Pull shared eta/invoice from first redelivery ticket if available
        eta = _s(sol.get("eta"))
        inv_ref = ""
        for t in (open_t or []):
            meta = _parse_ticket_resolution_note(_s(getattr(t, "resolution_note", None)))
            r = _s(meta.get("resolution")).strip().lower()
            if r in {"supplementary_delivery", "re_delivery"}:
                if not eta:
                    eta = _s(meta.get("eta"))
                inv_ref = _s(meta.get("invoice"))
                break

        # Provider details (best-effort)
        p = ctx.providers_by_name.get(provn) or ctx.providers_by_name.get(norm_provider(prov))
        emails: List[str] = []
        if p and getattr(p, "order_email", None):
            emails += [x.strip() for x in (p.order_email or "").split("|") if x.strip()]
        if p and getattr(p, "emails", None):
            emails += [x.strip() for x in (p.emails or "").split("|") if x.strip()]
        emails = list(dict.fromkeys([e for e in emails if e]))  # de-dupe keep order

        phone = _s(getattr(p, "phone", None)) if p else ""
        address = _s(getattr(p, "address", None)) if p else ""

        st.markdown(
            "<div class='voi-card' style='border-color:#bfdbfe;background:#eff6ff'>"
            "<div class='voi-title'>🚚 Expected re-delivery (preview)</div>"
            f"<div class='voi-muted'>Supplier: <b>{html.escape(prov)}</b></div>"
            + (f"<div class='voi-muted'>Expected: <b>{html.escape(eta)}</b></div>" if eta else "")
            + (f"<div class='voi-muted'>Reference invoice: <b>{html.escape(inv_ref)}</b></div>" if inv_ref else "")
            + "</div>",
            unsafe_allow_html=True,
        )

        if emails or phone or address:
            st.markdown("**Supplier details**")
            if emails:
                st.markdown(f"- **Email:** `{emails[0]}`" + (f" (+{len(emails)-1} more)" if len(emails) > 1 else ""))
            if phone:
                st.markdown(f"- **Phone:** `{phone}`")
            if address:
                st.markdown(f"- **Address:** `{address}`")

        # Items: prefer explicit workflow-note items (flagged as redelivery), else list from tickets.
        redel_items: List[Dict[str, Any]] = []
        if items:
            redel_items = [it for it in items if isinstance(it, dict) and _is_redelivery_item(it)]

        if redel_items:
            st.markdown("**Products/quantities being re-delivered:**")
            for it in redel_items:
                nm = _s(it.get("name"))
                qty = _s(it.get("qty"))
                unit = _s(it.get("unit"))
                why = (_s(it.get("reason"))).replace("_", " ")
                st.markdown(f"- **{nm}** · {qty} {unit} · {why}")
        else:
            st.markdown("**Products/quantities being re-delivered:**")
            for t in (open_t or []):
                meta = _parse_ticket_resolution_note(_s(getattr(t, "resolution_note", None)))
                r = _s(meta.get("resolution")).strip().lower()
                if r not in {"supplementary_delivery", "re_delivery"} and (has_rd_items or is_rd):
                    # If caller didn't pass subset, keep only rd-labeled tickets
                    continue
                lid = int(getattr(t, "order_line_id", 0) or 0)
                ln = None
                try:
                    ln = next((x for x in (ctx.lines_by_provider.get(provn, []) or []) if int(getattr(x, "id", 0) or 0) == lid), None)
                except Exception:
                    ln = None

                nm = _s(getattr(t, "product_name", None)) or (_line_name(ln, ctx.products_by_id) if ln else "Product")
                unit = _s(getattr(t, "unit", None)) or (_line_unit(ln, ctx.products_by_id) if ln else "unit")
                qty = _safe_float(getattr(t, "qty_invoiced", None), 0.0)
                if qty <= 0:
                    qty = _safe_float(getattr(t, "qty_expected", None), 0.0)
                why = (_s(getattr(t, "kind", None)) or "re_delivery").replace("_", " ")
                st.markdown(f"- **{nm}** · {qty:g} {unit} · {why}")


    for prov in providers:
        provn = norm_provider(prov)
        open_t = _provider_open_tickets(ctx, prov)
        if not open_t:
            continue
        open_any = True

        wf = ctx.workflows_by_provider.get(provn)
        state = _s(getattr(wf, "state", None)) if wf else ""
        
        
        oid = int(getattr(order, "id", 0) or 0)
        
        # ---------------------------------------------------------
        # UI lock: hide issue lines after "Save & request decision"
        # ---------------------------------------------------------
        decisions_key = f"inc_decisions_done_{oid}_{provn}"
        if decisions_key not in st.session_state:
            st.session_state[decisions_key] = False

        # ✅ Robust fallback: if workflow missing/empty, derive from ticket kinds
        if not state:
            open_kinds = {(_s(getattr(t, "kind", "")).lower()) for t in open_t}
            if open_kinds & {"invoice_discrepancy", "damaged", "wrong_item"}:
                state = "INVOICE_DISCREPANCY"
            elif "operational_missing" in open_kinds:
                state = "OPERATIONAL_MISSING_PRODUCT"

        badge_txt, badge_kind = _state_badge(state)

        receipt = ctx.receipts_by_provider.get(provn)
        inv_no = _s(getattr(receipt, "invoice_number", None))
        inv_pill = f"<span class='pill'>🧾 {inv_no}</span>" if inv_no else ""

        st.markdown(
            f"<div class='voi-card'>"
            f"<div class='voi-title'>{prov}{inv_pill}</div>"
            f"<div class='voi-muted'>Workflow: {_badge(badge_txt, badge_kind)}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Build quick lookups for prettier line rendering
        prov_lines = ctx.lines_by_provider.get(provn, []) or []
        line_by_id: Dict[int, OrderLine] = {}
        for ln in prov_lines:
            lid = int(getattr(ln, "id", 0) or 0)
            if lid:
                line_by_id[lid] = ln

        
        # --- Supplier solution ---
        # Workflow note can represent a *mixed* outcome (some items credit note, others re-delivery).
        # Tickets store the per-line truth in `resolution_note`. Prefer that when present.
        sol_wf = _normalize_solution_meta(_parse_supplier_solution_meta(_s(getattr(wf, "note", None)) if wf else ""))
        
        # ---------------------------------------------------------
        # Auto-lock issue lines when supplier already acted
        # (so lines don't show again after refresh / revisit)
        # ---------------------------------------------------------
        state_u = (state or "").upper()
        wf_res = (sol_wf.get("resolution") or "").strip().lower()

        supplier_acted_states = {
            "SUPPLIER_CREDIT_NOTE_ISSUED",
            "SUPPLEMENTARY_DELIVERY_SENT",
            "SUPPLIER_REJECTED",
            "CLOSED",
        }

        # If supplier decision exists (or we are past decision stage), lock the UI
        if (state_u in supplier_acted_states) or (wf_res in {"credit_note", "supplementary_delivery", "re_delivery"}):
            st.session_state[decisions_key] = True

                
        
        
        
        sol = sol_wf  # keep `sol` name for downstream chips/matching code

        ticket_meta_by_id: Dict[int, Dict[str, Any]] = {}
        credit_t: List[SeguimientoTicket] = []
        redel_t: List[SeguimientoTicket] = []
        undecided_t: List[SeguimientoTicket] = []

        for t in open_t:
            tid_local = int(getattr(t, "id", 0) or 0)
            meta = _parse_ticket_resolution_note(_s(getattr(t, "resolution_note", None)))
            ticket_meta_by_id[tid_local] = meta

            r = (_s(meta.get("resolution"))).strip().lower()
            if r == "credit_note":
                credit_t.append(t)
            elif r in {"supplementary_delivery", "re_delivery", "re-delivery", "redelivery"}:
                redel_t.append(t)
            else:
                undecided_t.append(t)

        # --- Expected previews ---
        if credit_t or redel_t:
            if credit_t:
                # Prefer per-ticket CN number when available
                cn_no = ""
                for t in credit_t:
                    meta = _parse_ticket_resolution_note(_s(getattr(t, "resolution_note", None)))
                    if _s(meta.get("credit_note_invoice")):
                        cn_no = _s(meta.get("credit_note_invoice"))
                        break
                sol_cn = dict(sol_wf)
                sol_cn["resolution"] = "credit_note"
                if cn_no:
                    sol_cn["credit_note_invoice"] = cn_no
                with st.expander("🧾 Expected credit note (preview)", expanded=False):
                    _credit_note_preview(prov, provn, credit_t, sol_cn)
                    if st.button("✅ Verify & close (credit note)", use_container_width=True, key=f"inc_cn_verify_{int(order.id)}_{provn}"):
                        res = venue_verify_and_close(ctx=ctx, provider=provn, mode="credit_note")
                        if res == "ok":
                            st.success("Closed")
                            st.rerun()
                        else:
                            st.error(res)

            if redel_t:
                # Prefer eta/invoice from first redelivery ticket
                meta0 = _parse_ticket_resolution_note(_s(getattr(redel_t[0], "resolution_note", None)))
                sol_rd = dict(sol_wf)
                sol_rd["resolution"] = (_s(meta0.get("resolution")) or "re_delivery").strip().lower()
                if _s(meta0.get("eta")):
                    sol_rd["eta"] = _s(meta0.get("eta"))
                if _s(meta0.get("invoice")):
                    sol_rd["invoice"] = _s(meta0.get("invoice"))
                with st.expander("🚚 Expected re-delivery (preview)", expanded=False):
                    _redelivery_preview(ctx, prov, provn, redel_t, sol_rd)
                    if st.button("✅ Verify & close (re-delivery)", use_container_width=True, key=f"inc_rd_verify_{int(order.id)}_{provn}"):
                        res = venue_verify_and_close(ctx=ctx, provider=provn, mode="supplementary")
                        if res == "ok":
                            st.success("Closed")
                            st.rerun()
                        else:
                            st.error(res)

        else:
            # Fallback: no per-ticket decisions recorded yet → use workflow note items/resolution.
            res = (sol_wf.get("resolution") or "").strip().lower()
            items = sol_wf.get("items") or []
            has_rd_items = any(isinstance(it, dict) and _is_redelivery_item(it) for it in items)
            has_cn_items = any(isinstance(it, dict) and (not _is_redelivery_item(it)) for it in items)

            if has_cn_items:
                with st.expander("🧾 Expected credit note (preview)", expanded=False):
                    _credit_note_preview(prov, provn, open_t, sol_wf)
                    if st.button("✅ Verify & close (credit note)", use_container_width=True, key=f"inc_cn_verify_{int(order.id)}_{provn}_fallback"):
                        res = venue_verify_and_close(ctx=ctx, provider=provn, mode="credit_note")
                        if res == "ok":
                            st.success("Closed")
                            st.rerun()
                        else:
                            st.error(res)
            if has_rd_items:
                with st.expander("🚚 Expected re-delivery (preview)", expanded=False):
                    _redelivery_preview(ctx, prov, provn, open_t, sol_wf)
                    if st.button("✅ Verify & close (re-delivery)", use_container_width=True, key=f"inc_rd_verify_{int(order.id)}_{provn}_fallback"):
                        res = venue_verify_and_close(ctx=ctx, provider=provn, mode="supplementary")
                        if res == "ok":
                            st.success("Closed")
                            st.rerun()
                        else:
                            st.error(res)

            if not items:
                if res == "credit_note":
                    _credit_note_preview(prov, provn, open_t, sol_wf)
                elif res in {"supplementary_delivery", "re_delivery"}:
                    _redelivery_preview(ctx, prov, provn, open_t, sol_wf)
                else:
                    _credit_note_preview(prov, provn, open_t, sol_wf)


# Best-effort match supplier note items by product name
        sol_items: List[Dict[str, Any]] = list((sol.get("items") or []) if isinstance(sol, dict) else [])
        sol_by_name: Dict[str, Dict[str, Any]] = {}
        for itx in sol_items:
            nm = _s(itx.get("name")).lower()
            if nm:
                sol_by_name[nm] = itx

        def _match_sol_item(ticket: SeguimientoTicket) -> Optional[Dict[str, Any]]:
            nm = _s(getattr(ticket, "product_name", None)).lower()
            if nm and nm in sol_by_name:
                return sol_by_name[nm]
            for k2, v2 in sol_by_name.items():
                if nm and (nm in k2 or k2 in nm):
                    return v2
            return None




        # ---------------------------------------------------------
        # Helper: apply inline reorder decisions for this provider
        # ---------------------------------------------------------
        def _apply_inline_reorders_for_provider(*, close_non_urgent_op_missing: bool = False) -> None:
            for tt in open_t:
                kind = (_s(getattr(tt, "kind", ""))).lower()
                if kind not in {"operational_missing", "invoice_discrepancy", "damaged", "wrong_item"}:
                    continue

                tid = int(getattr(tt, "id", 0) or 0)
                key_base = f"inc_{oid}_{provn}_{tid}"
                reorder_key = key_base + "_reorder"
                done_key = key_base + "_reorder_done"

                toggled = bool(st.session_state.get(reorder_key, False))

                # ✅ operational_missing + NOT urgent => auto-close as "not re-ordered"
                if kind == "operational_missing" and (not toggled) and close_non_urgent_op_missing:
                    if st.session_state.get(done_key, False):
                        continue

                    _close_ticket(
                        ticket_id=tid,
                        new_state="resolved_not_reordered",
                        note="Operational missing: not reordered (urgent toggle OFF).",
                        actor=_s(st.session_state.get("user_email") or st.session_state.get("actor") or "venue"),
                    )
                    st.session_state[done_key] = True
                    continue

                # Existing behavior: only act if toggle is ON
                if not toggled:
                    continue

                if st.session_state.get(done_key, False):
                    continue

                lid = int(getattr(tt, "order_line_id", 0) or 0)
                ln = line_by_id.get(lid)

                rq = 0.0

                # ✅ For operational_missing, qty_invoiced is where you stored the missing qty (issue_qty)
                if kind == "operational_missing":
                    rq = _safe_float(getattr(tt, "qty_invoiced", None), 0.0)

                # Existing: for invoice discrepancy / damaged, also use qty_invoiced
                elif kind in {"invoice_discrepancy", "damaged"}:
                    rq = _safe_float(getattr(tt, "qty_invoiced", None), 0.0)

                # Fallbacks (only if rq is still 0)
                if rq <= 0:
                    rq = _safe_float(getattr(tt, "qty_expected", None), 0.0)

                if rq <= 0 and ln is not None:
                    fu = ctx.followups_by_key.get((provn, int(getattr(ln, "id", 0) or 0)))
                    rq = _derive_expected_qty(ln, fu)

                if rq <= 0:
                    rq = 1.0


                try:
                    _upsert_urgent_reorder_request(
                        ticket_id=tid,
                        provider_name=provn,
                        product_name=_line_name(ln, ctx.products_by_id) if ln else _s(getattr(tt, "product_name", "")),
                        qty=float(rq),
                        unit=_line_unit(ln, ctx.products_by_id) if ln else (_s(getattr(tt, "unit", "")) or "unit"),
                        actor=_s(st.session_state.get("user_email") or st.session_state.get("actor") or "venue"),
                    )
                except Exception:
                    pass

                st.session_state[done_key] = True

        # ---------------------------------------------------------
        # action panel  ✅ (THIS is what you were missing)
        # ---------------------------------------------------------
        a1, a2, a3 = st.columns([1.05, 1.05, 1.2], vertical_alignment="center")
        supplier_link = build_seguimiento_url(
            order_id=int(order.id),
            provider_name=provn,
            role=ROLE_SUPPLIER,
            page_path="seguimiento",
        )

        if state.upper() in {"SUPPLIER_CREDIT_NOTE_ISSUED", "SUPPLEMENTARY_DELIVERY_SENT"}:
            # If we have mixed outcomes, verification is handled inside the two independent previews above.
            if (len(credit_t) > 0) and (len(redel_t) > 0):
                st.info("Mixed outcome: verify each preview above using its **Verify & close** button.")
            else:
                label = "✅ Verify credit note & close" if state.upper() == "SUPPLIER_CREDIT_NOTE_ISSUED" else "✅ Verify delivery & close"
                mode = "credit_note" if state.upper() == "SUPPLIER_CREDIT_NOTE_ISSUED" else "supplementary"
                if a1.button(label, use_container_width=True, key=f"inc_verify_{order.id}_{provn}"):
                    res = venue_verify_and_close(ctx=ctx, provider=provn, mode=mode)
                    if res == "ok":
                        st.success("Closed")
                        st.rerun()
                    else:
                        st.error(res)

        elif state.upper() == "SUPPLIER_REJECTED":
            st.warning("Supplier rejected the claim (typically used for *Wrong item* / *Damaged* disputes).")
            if a1.button("✅ Accept reject & close", use_container_width=True, key=f"inc_rej_{order.id}_{provn}"):
                res = venue_verify_and_close(ctx=ctx, provider=provn, mode="reject")
                if res == "ok":
                    st.success("Closed")
                    st.rerun()
                else:
                    st.error(res)

        # ✅ RESTORED: send incidences to supplier + create urgent requests
        elif state.upper() == "INVOICE_DISCREPANCY":
            if a1.button("💾 Save & request decision", use_container_width=True, key=f"inc_req_{order.id}_{provn}"):
                _apply_inline_reorders_for_provider(close_non_urgent_op_missing=True)

                ok, msg = request_supplier_resolution(int(order.venue_id), int(order.id), provn)
                if ok:
                    st.session_state[decisions_key] = True   # ✅ NEW: lock UI
                    st.success("Link sent")
                    st.link_button("Open supplier link", msg, use_container_width=True)
                    st.rerun()
                else:
                    st.error(msg)


        elif state.upper() == "OPERATIONAL_MISSING_PRODUCT":
            if a1.button("💾 Save & request decision", use_container_width=True, key=f"inc_op_save_{order.id}_{provn}"):
                _apply_inline_reorders_for_provider(close_non_urgent_op_missing=True)

                ctx2 = _load_order_context(int(ctx.order.venue_id), int(ctx.order.id))
                if len(_provider_open_tickets(ctx2, provn)) == 0:
                    _set_workflow_state(
                        venue_id=int(order.venue_id),
                        order_id=int(order.id),
                        provider=provn,
                        to_state="CLOSED",
                        actor_role="venue",
                        actor="venue",
                        note="Operational missing: non-urgent items ignored (not reordered).",
                    )

                st.session_state[decisions_key] = True   # ✅ NEW: lock UI
                st.success("Saved ✓")
                st.rerun()

        elif state.upper() == "WAITING_SUPPLIER_ACTION":
            ref = st.text_input("Reference", key=f"dec_ref_{order.id}_{provn}")
            comment = st.text_input("Comment", key=f"dec_c_{order.id}_{provn}")
            if a1.button("📝 Credit note", use_container_width=True, key=f"dec_cn_{order.id}_{provn}"):
                res = supplier_decision_from_venue(ctx=ctx, provider=provn, decision="credit_note", ref=ref, comment=comment)
                if res == "ok":
                    st.success("Recorded")
                    st.rerun()
                else:
                    st.error(res)
            if a2.button("🚚 Re-delivery", use_container_width=True, key=f"dec_rd_{order.id}_{provn}"):
                res = supplier_decision_from_venue(ctx=ctx, provider=provn, decision="re_delivery", ref=ref, comment=comment)
                if res == "ok":
                    st.success("Recorded")
                    st.rerun()
                else:
                    st.error(res)
            a3.link_button("🔗 Supplier link", supplier_link, use_container_width=True)

        elif state.upper() == "SUPPLIER_CREDIT_NOTE_PENDING":
            st.warning("⏳ Supplier chose credit note, but the credit note number is still missing.")
            st.caption("Ask the supplier to open the link again and add the credit note number.")
            a3.link_button("🔗 Supplier link", supplier_link, use_container_width=True)


        
        # ---------------------------------------------------------
        # Render issue lines (pretty + per-line reorder control)
        # Only show while venue is still working on incidences
        # ---------------------------------------------------------
        editable_states = {
            "INVOICE_DISCREPANCY",
            "OPERATIONAL_MISSING_PRODUCT",
            "WAITING_SUPPLIER_ACTION",
            "SUPPLIER_CREDIT_NOTE_PENDING",
        }
        
        show_issue_lines = ((state_u in editable_states) and (not st.session_state.get(decisions_key, False)))

        if show_issue_lines:
            for t in open_t:
                tid = int(getattr(t, "id", 0) or 0)
                lid = int(getattr(t, "order_line_id", 0) or 0)
                ln = line_by_id.get(lid)

                pname_raw = _line_name(ln, ctx.products_by_id) if ln else _s(getattr(t, "product_name", ""))
                pdesc_raw = _line_desc(ln, ctx.products_by_id) if ln else ""
                unit_raw = _s(getattr(t, "unit", None)) or (_line_unit(ln, ctx.products_by_id) if ln else "unit")

                pname = html.escape(pname_raw or "")
                pdesc = html.escape(pdesc_raw or "")
                unit = html.escape(unit_raw or "")

                expected_qty = _safe_float(getattr(t, "qty_expected", None), 0.0)
                if expected_qty <= 0 and ln is not None:
                    fu = ctx.followups_by_key.get((provn, int(getattr(ln, "id", 0) or 0)))
                    expected_qty = _derive_expected_qty(ln, fu)

                sol_mode = (_s(sol.get("resolution")) if isinstance(sol, dict) else "").strip().lower()
                sol_item = _match_sol_item(t)

                sol_qty = None
                if sol_item is not None:
                    try:
                        sol_qty = float(sol_item.get("qty"))
                    except Exception:
                        sol_qty = None

                kind = (_s(getattr(t, "kind", ""))).lower()
                if sol_qty is None:
                    if kind in {"invoice_discrepancy", "damaged"}:
                        sol_qty = _safe_float(getattr(t, "qty_invoiced", None), 0.0)
                    else:
                        sol_qty = max(0.0, expected_qty - _safe_float(getattr(t, "qty_received", None), 0.0))

                kind_txt_raw = (_s(getattr(t, "kind", "")) or "issue").replace("_", " ")
                kind_txt = html.escape(kind_txt_raw)

                sol_chip = ""
                if sol_mode in {"credit_note", "creditnote"}:
                    cn = _s(sol.get("credit_note_invoice")) if isinstance(sol, dict) else ""
                    cn = html.escape(cn or "")
                    cn_txt = f" · #{cn}" if cn else ""
                    sol_chip = f"<span class='pill pill--warn'>📝 CN{cn_txt}: {sol_qty:g} {unit}</span>"

                elif sol_mode in {"supplementary_delivery", "re_delivery", "re-delivery", "redelivery"}:
                    eta = _s(sol.get("eta")) if isinstance(sol, dict) else ""
                    eta = html.escape(eta or "")
                    eta_txt = f" · {eta}" if eta else ""
                    sol_chip = f"<span class='pill pill--ok'>🚚 Re-delivery{eta_txt}: {sol_qty:g} {unit}</span>"

                    if kind == "operational_missing":
                        old_inv = html.escape(_s(getattr(t, "invoice_no", "")) or "")
                        new_inv = html.escape(_s(sol.get("new_invoice")) if isinstance(sol, dict) else "")
                        if old_inv:
                            sol_chip += f" <span class='pill'>missing from invoice #{old_inv}</span>"
                        if new_inv:
                            sol_chip += f" <span class='pill pill--info'>new invoice #{new_inv}</span>"

                # Issue qty pill
                if kind == "operational_missing":
                    inc_qty = _safe_float(getattr(t, "qty_invoiced", None), 0.0)
                    if inc_qty <= 0:
                        inc_qty = max(0.0, float(expected_qty or 0.0) - _safe_float(getattr(t, "qty_received", None), 0.0))
                else:
                    inc_qty = float(sol_qty or 0.0)
                    if inc_qty <= 0:
                        inc_qty = _safe_float(getattr(t, "qty_invoiced", None), 0.0)
                    if inc_qty <= 0:
                        inc_qty = _safe_float(getattr(t, "qty_expected", None), 0.0)
                    if inc_qty <= 0:
                        inc_qty = float(expected_qty or 0.0)

                issue_cls = {
                    "operational_missing": "info",
                    "invoice_discrepancy": "warn",
                    "damaged": "bad",
                    "wrong_item": "bad",
                }.get(kind, "warn")
                inc_chip = f"<span class='pill pill--{issue_cls}'>Issue: {inc_qty:g} {unit}</span>"

                key_base = f"inc_{oid}_{provn}_{tid}"

                left, right = st.columns([0.84, 0.16], vertical_alignment="center")

                kind_cls = {
                    "operational_missing": "info",
                    "invoice_discrepancy": "warn",
                    "damaged": "bad",
                    "wrong_item": "bad",
                }.get(kind, "")
                kind_chip = (
                    f"<span class='pill pill--{kind_cls}'>{kind_txt}</span>"
                    if kind_cls
                    else f"<span class='pill'>{kind_txt}</span>"
                )

                with left:
                    st.markdown(
                        (
                            "<div class='line'>"
                            "  <div class='top'>"
                            "    <div>"
                            f"      <div class='name'>{pname}</div>"
                            f"      {('<div class=' + 'desc' + '>' + pdesc + '</div>') if pdesc else ''}"
                            "    </div>"
                            "    <div class='pillrow'>"
                            f"      <span class='pill'>Expected: {expected_qty:g} {unit}</span>"
                            f"      {inc_chip}"
                            f"      {kind_chip}"
                            f"      {sol_chip}"
                            "    </div>"
                            "  </div>"
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )

                with right:
                    reorder_kinds = {"operational_missing", "invoice_discrepancy", "damaged", "wrong_item"}
                    if kind in reorder_kinds:
                        reorder_key = key_base + "_reorder"
                        done_key = key_base + "_reorder_done"

                        val = st.toggle("Order urgent", key=reorder_key, value=False)

                        # If user turns the toggle OFF again, allow future processing
                        if not val:
                            st.session_state.pop(done_key, None)
                    else:
                        st.markdown("")
        else:
             st.success("✔ Decisions requested.")            

    if not open_any:
        st.success("✅ No incidences requiring action.")


# =============================
# History
# =============================

def _provider_closed(ctx: OrderContext, provider: str) -> bool:
    wf = ctx.workflows_by_provider.get(norm_provider(provider))
    return (_s(getattr(wf, "state", None)).upper() == "CLOSED")


def _tickets_for_provider(ctx: OrderContext, provider: str) -> List[SeguimientoTicket]:
    return list(ctx.tickets_by_provider.get(norm_provider(provider), []) or [])


def _provider_resolution_summary(ctx: OrderContext, provider: str) -> Dict[str, Any]:
    """Summarize how the provider was resolved (for History).

    We close per provider.

    Returns:
      mode:
        - credit_note: supplier issued a credit note and venue verified it
        - ok_redelivery: supplier re-delivered and venue verified it (still OK, but we show details)
        - ok_internal: operational missing resolved internally (OK)
        - closed: closed but could not infer resolution

      solution:
        Parsed supplier solution note (kind/ref/eta/credit_note_invoice/items/raw)

      verified_at:
        When the venue verified the resolution (best-effort from ticket.resolved_at)
    """
    prov = norm_provider(provider)
    ts = _tickets_for_provider(ctx, prov)

    # Look for any supplier note on tickets
    supplier_note = ""
    for t in ts:
        n = _s(getattr(t, "resolution_note", None))
        if "[SUPPLIER]" in n:
            supplier_note = n
            break

    sol = (
        _parse_supplier_solution_meta(supplier_note)
        if supplier_note
        else {"kind": "", "ref": "", "eta": "", "credit_note_invoice": "", "items": [], "raw": ""}
    )

    states = [(_s(getattr(t, "state", None)) or "").lower() for t in ts]

    has_cn_verified = any(s.startswith("resolved_credit_note") for s in states)
    has_redel_verified = any(s.startswith("resolved_supplementary") for s in states)
    has_internal = any(s.startswith("resolved_internal") for s in states)

    # Best-effort: when venue verified (resolved_at on the relevant tickets)
    def _max_resolved_at(prefix: str) -> Optional[datetime]:
        dts = []
        for t in ts:
            stt = (_s(getattr(t, "state", None)) or "").lower()
            if stt.startswith(prefix):
                ra = getattr(t, "resolved_at", None)
                if isinstance(ra, datetime):
                    dts.append(ra)
        return max(dts) if dts else None

    if has_cn_verified:
        return {
            "mode": "credit_note",
            "solution": sol,
            "verified_at": _max_resolved_at("resolved_credit_note"),
        }

    if has_redel_verified:
        return {
            "mode": "ok_redelivery",
            "solution": sol,
            "verified_at": _max_resolved_at("resolved_supplementary"),
        }

    if has_internal:
        return {
            "mode": "ok_internal",
            "solution": sol,
            "verified_at": _max_resolved_at("resolved_internal"),
        }

    # closed but no resolved state detected (fallback)
    return {"mode": "closed", "solution": sol, "verified_at": None}


def _render_history_tab(venue_id: int, *, deep_provider: Optional[str] = None):
    orders = _get_history_orders(int(venue_id))
    if not orders:
        st.info("No closed supplier history yet.")
        return

    def _order_label_with_invoices(o: Order) -> str:
        base = f"#{int(o.id)}" if getattr(o, "id", None) is not None else "#—"
        when = getattr(o, "created_at", None)
        when_s = when.strftime("%Y-%m-%d %H:%M") if when else ""

        # build invoice summary per provider (only if exists)
        try:
            ctx = _load_order_context(int(venue_id), int(o.id))
            parts = []
            for prov, rec in (ctx.receipts_by_provider or {}).items():
                inv = _s(getattr(rec, "invoice_number", None))
                if inv:
                    parts.append(f"{prov}·{inv}")
            inv_txt = (" | ".join(parts)) if parts else ""
        except Exception:
            inv_txt = ""

        head = f"{base} — {when_s}" if when_s else base
        return head + (f" — {inv_txt}" if inv_txt else "")

    selected_id = st.selectbox(
        "Order",
        options=[int(o.id) for o in orders if o.id is not None],
        format_func=lambda oid: _order_label_with_invoices(next(o for o in orders if int(o.id) == int(oid))),
        key=f"hist_sel_{int(venue_id)}",
    )

    ctx = _load_order_context(int(venue_id), int(selected_id))
    providers = sorted(ctx.lines_by_provider.keys(), key=lambda x: x.lower())
    
    # ✅ Only keep providers that are 📧 Enviado (ProviderSendStatus.sent == True)
    with get_session() as s:
        rows = list(
            s.exec(
                select(ProviderSendStatus).where(ProviderSendStatus.order_id == int(ctx.order.id))
            ).all()
        )

    sent_providers = {
        norm_provider(r.provider_name)
        for r in rows
        if bool(getattr(r, "sent", False))
    }

    providers = [p for p in providers if norm_provider(p) in sent_providers]

    # Deep-link: provider (best-effort)
    if deep_provider:
        target = norm_provider(deep_provider)
        providers_norm = [norm_provider(p) for p in providers]
        if target in providers_norm:
            idx = providers_norm.index(target)
            idx_key = f"recv_current_provider_idx_{int(selected_id)}"
            sel_key = f"recv_provider_sel_{int(selected_id)}"
            st.session_state[idx_key] = idx
            st.session_state[sel_key] = providers[idx]

    # -----------------------------
    # Deep-link: provider (best-effort)
    # -----------------------------
    if deep_provider:
        target = norm_provider(deep_provider)
        providers_norm = [norm_provider(p) for p in providers]
        if target in providers_norm:
            idx = providers_norm.index(target)
            st.session_state[f"recv_current_provider_idx_{int(selected_id)}"] = idx
            st.session_state[f"recv_provider_sel_{int(selected_id)}"] = providers[idx]

    
    if not providers:
        st.info("No sent suppliers (📧 Enviado) for this order yet.")
        return

    
    closed_providers = [p for p in providers if _provider_closed(ctx, p)]
    if not closed_providers:
        st.info("This order has no closed suppliers yet.")
        return

    st.markdown("### Closed suppliers")

    for prov in closed_providers:
        provn = norm_provider(prov)
        receipt = ctx.receipts_by_provider.get(provn)
        inv_no = _s(getattr(receipt, "invoice_number", None)) or "—"

        summ = _provider_resolution_summary(ctx, provn)
        mode = summ.get("mode")
        sol = summ.get("solution") or {}

        verified_at = summ.get("verified_at")

        if mode == "credit_note":
            title = f"{prov} · 🧾 Invoice: {inv_no} · Credit note"
            badge = _badge("Credit note", "warn")
        elif mode == "ok_redelivery":
            eta = _s(sol.get("eta"))
            eta_txt = f" · ETA: {eta}" if eta else ""
            when_txt = f" · Verified: {verified_at.strftime('%Y-%m-%d %H:%M')}" if isinstance(verified_at, datetime) else ""
            title = f"{prov} · 🧾 Invoice: {inv_no} · OK (Re-delivery)" + eta_txt + when_txt
            badge = _badge("OK", "ok") + " " + _badge("Re-delivery", "info")
        elif mode in {"ok_internal", "ok"}:
            when_txt = f" · Verified: {verified_at.strftime('%Y-%m-%d %H:%M')}" if isinstance(verified_at, datetime) else ""
            title = f"{prov} · 🧾 Invoice: {inv_no} · OK" + when_txt
            badge = _badge("OK", "ok")
        else:
            title = f"{prov} · 🧾 Invoice: {inv_no} · Closed"
            badge = _badge("Closed", "info")

        st.markdown(
            f"<div class='voi-card'>"
            f"<div class='voi-title'>{title}</div>"
            f"<div class='voi-muted'>{badge}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Show the full order analytically for this supplier
        lines = ctx.lines_by_provider.get(provn, []) or []
        rows: List[Dict[str, Any]] = []
        for ln in lines:
            lid = int(getattr(ln, "id", 0) or 0)
            fu = ctx.followups_by_key.get((provn, lid))
            ordered = _safe_float(getattr(ln, "quantity", 0.0), 0.0)
            expected = _derive_expected_qty(ln, fu)
            received = _safe_float(getattr(fu, "venue_qty", expected) if fu else expected, expected)

            # when supplementary delivery verified, treat as normal (no issue)
            issue_qty = (
                _safe_float(getattr(fu, "qty_invoiced", 0.0) if fu else 0.0, 0.0)
                if mode in {"credit_note", "ok_redelivery"}
                else 0.0
            )

            rows.append(
                {
                    "Product": _line_name(ln, ctx.products_by_id),
                    "Unit": _line_unit(ln, ctx.products_by_id),
                    "Ordered": float(ordered),
                    "Expected": float(expected),
                    "Received": float(received),
                    "Issue qty": float(issue_qty),
                }
            )

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Credit note details (products, CN number, related invoice)
        if mode == "credit_note":
            cn = _s(sol.get("credit_note_invoice")) or "—"
            st.markdown(f"**Credit note number:** `{cn}`")
            st.markdown(f"**Related invoice number:** `{inv_no}`")

            items = sol.get("items") or []
            if items:
                st.markdown("**Products in credit note:**")
                for it in items:
                    nm = _s(it.get("name"))
                    qty = _s(it.get("qty"))
                    why = _s(it.get("why"))
                    why_txt = why.replace("_", " ") if why else ""
                    st.markdown(f"- **{nm}** · {qty}" + (f" · {why_txt}" if why_txt else ""))

        with st.expander("🕒 Timeline / audit trail", expanded=False):
            _render_timeline(ctx, provn)
# =============================
# Main dashboard
# =============================
# =============================
# Global dashboards (no implicit order)
# =============================

def _list_pending_receive_items(venue_id: int) -> List[Dict[str, Any]]:
    """Flat list of (order, provider) that still needs Receive action."""
    DONE_STATES = {
        "RECEIVED",
        "INVOICE_DISCREPANCY",
        "OPERATIONAL_MISSING_PRODUCT",
        "WAITING_SUPPLIER_ACTION",
        "SUPPLIER_CREDIT_NOTE_ISSUED",
        "SUPPLIER_CREDIT_NOTE_PENDING",
        "SUPPLEMENTARY_DELIVERY_SENT",
        "SUPPLIER_REJECTED",
        "CLOSED",
    }

    orders = _get_active_orders(int(venue_id))
    if not orders:
        return []

    order_ids = [int(o.id) for o in orders if getattr(o, "id", None) is not None]
    if not order_ids:
        return []

    with get_session() as s:
        send_rows = list(
            s.exec(select(ProviderSendStatus).where(ProviderSendStatus.order_id.in_(order_ids))).all()
        )
        sent_pairs: set[tuple[int, str]] = set()
        for r in send_rows:
            if (
                bool(getattr(r, "sent", False))
                or bool(getattr(r, "sent_email", False))
                or bool(getattr(r, "sent_whatsapp", False))
            ):
                sent_pairs.add((int(r.order_id), norm_provider(getattr(r, "provider_name", "") or "")))

        wf_rows = list(s.exec(select(OrderWorkflow).where(OrderWorkflow.order_id.in_(order_ids))).all())
        wf_by_pair = {(int(w.order_id), norm_provider(w.provider_name)): w for w in wf_rows}

        rc_rows = list(s.exec(select(ProviderReceipt).where(ProviderReceipt.order_id.in_(order_ids))).all())
        rc_by_pair = {(int(r.order_id), norm_provider(r.provider_name)): r for r in rc_rows}

    out: List[Dict[str, Any]] = []
    for o in orders:
        oid = int(getattr(o, "id", 0) or 0)
        if not oid:
            continue

        provs = sorted([p for (oo, p) in sent_pairs if oo == oid and p], key=lambda x: x.lower())
        for provn in provs:
            wf = wf_by_pair.get((oid, provn))
            state = (_s(getattr(wf, "state", None)) or "ORDER_SENT").upper()
            if state in DONE_STATES:
                continue

            rec = rc_by_pair.get((oid, provn))
            inv = _s(getattr(rec, "invoice_number", None))
            out.append(
                {
                    "order": o,
                    "order_id": oid,
                    "provider_norm": provn,
                    "provider_display": provn,
                    "invoice_number": inv,
                    "state": state,
                }
            )

    out.sort(key=lambda r: (-int(r["order_id"]), (r["provider_display"] or "").lower()))
    return out


def _render_receive_provider_panel(ctx: OrderContext, provider: str) -> None:
    """Render the Receive UI block for a single provider (reused in global Receive list)."""
    current_provider = provider

    wf = ctx.workflows_by_provider.get(norm_provider(current_provider))
    state = _s(getattr(wf, "state", None)) if wf else "ORDER_SENT"
    badge_txt, badge_kind = _state_badge(state)

    st.markdown(
        f"<div class='voi-card'><div class='voi-title'>{current_provider}</div>"
        f"<div class='voi-muted'>Workflow: {_badge(badge_txt, badge_kind)}</div></div>",
        unsafe_allow_html=True,
    )

    receipt = ctx.receipts_by_provider.get(norm_provider(current_provider))

    inv_key = f"recv_inv_{int(ctx.order.id)}_{norm_provider(current_provider)}"
    db_inv = _s(getattr(receipt, "invoice_number", None))

    # Session-state sync (only when empty)
    if inv_key not in st.session_state:
        st.session_state[inv_key] = db_inv
    else:
        if (not (st.session_state[inv_key] or "").strip()) and db_inv:
            st.session_state[inv_key] = db_inv

    invoice_locked = (state or "").upper() == "CLOSED"

    btxt, bkind = _invoice_set_badge(receipt)
    if invoice_locked:
        btxt = "Verified (locked)"
        bkind = "ok"

    # Optional view toggles (update immediately)
    cP1, cP2 = st.columns([1.0, 1.0], vertical_alignment="center")
    with cP1:
        show_prices = st.toggle(
            "Show expected prices",
            value=False,
            key=f"recv_show_prices_{int(ctx.order.id)}_{norm_provider(current_provider)}",
            help="Estimated from catalog price + discount rules. Not an official invoice.",
        )
    with cP2:
        include_iva = st.toggle(
            "IVA",
            value=False,
            key=f"recv_show_iva_{int(ctx.order.id)}_{norm_provider(current_provider)}",
            disabled=not show_prices,
        )

    # Invoice input + badge
    c_inv1, c_inv2 = st.columns([2.0, 1.0], vertical_alignment="center")
    with c_inv1:
        inv_val = st.text_input(
            "Invoice number",
            key=inv_key,
            placeholder="e.g. 2026-001234",
            disabled=invoice_locked,
        )
    with c_inv2:
        st.markdown(_badge(btxt, bkind), unsafe_allow_html=True)

    _render_expected_lines(ctx, current_provider, show_prices=show_prices, include_iva=include_iva)

    st.markdown("<div class='voi-hr'></div>", unsafe_allow_html=True)

    # IMPORTANT: _render_receive_form is outside st.form, so its callbacks work.
    _render_receive_form(ctx, current_provider)

    # Buttons
    b1, b2, b3 = st.columns([1.0, 1.0, 1.0], vertical_alignment="center")
    with b1:
        save_invoice = st.button(
            "Save invoice #",
            use_container_width=True,
            disabled=invoice_locked,
            key=f"btn_save_invoice_{int(ctx.order.id)}_{norm_provider(current_provider)}",
        )
    with b2:
        save_all = st.button(
            "💾 Save all",
            type="primary",
            use_container_width=True,
            key=f"btn_save_all_{int(ctx.order.id)}_{norm_provider(current_provider)}",
        )
    with b3:
        st.caption(" ")

    if save_invoice:
        ok2, msg2 = upsert_provider_invoice_number(
            venue_id=int(ctx.order.venue_id),
            order_id=int(ctx.order.id),
            provider_name=current_provider,
            invoice_number=inv_val,
            actor_role="venue",
            actor="venue",
        )
        if ok2:
            st.success("Invoice saved")
            st.rerun()
        else:
            st.error(msg2)

    if save_all:
        ok, msg = save_all_received_for_provider(ctx=ctx, provider=current_provider)
        if ok:
            st.success("Saved ✓")
            st.rerun()
        else:
            st.error(msg)

    with st.expander("🕒 Timeline", expanded=False):
        _render_timeline(ctx, current_provider)


def _list_open_incidences_items(venue_id: int) -> List[Dict[str, Any]]:
    """Flat list of (order, provider) that has at least one open SeguimientoTicket."""
    orders = _get_active_orders(int(venue_id))
    if not orders:
        return []

    order_ids = [int(o.id) for o in orders if getattr(o, "id", None) is not None]
    if not order_ids:
        return []

    with get_session() as s:
        send_rows = list(
            s.exec(select(ProviderSendStatus).where(ProviderSendStatus.order_id.in_(order_ids))).all()
        )
        sent_pairs: set[tuple[int, str]] = set()
        for r in send_rows:
            if (
                bool(getattr(r, "sent", False))
                or bool(getattr(r, "sent_email", False))
                or bool(getattr(r, "sent_whatsapp", False))
            ):
                sent_pairs.add((int(r.order_id), norm_provider(getattr(r, "provider_name", "") or "")))

    out: List[Dict[str, Any]] = []
    for o in orders:
        oid = int(getattr(o, "id", 0) or 0)
        if not oid:
            continue

        ctx = _load_order_context(int(venue_id), oid)

        providers = sorted(ctx.lines_by_provider.keys(), key=lambda x: x.lower())
        # only providers actually sent
        providers = [p for p in providers if (oid, norm_provider(p)) in sent_pairs]

        for prov in providers:
            open_t = _provider_open_tickets(ctx, prov)
            if not open_t:
                continue
            receipt = ctx.receipts_by_provider.get(norm_provider(prov))
            inv = _s(getattr(receipt, "invoice_number", None))
            wf = ctx.workflows_by_provider.get(norm_provider(prov))
            wf_state = (_s(getattr(wf, "state", None)) or "ORDER_SENT").upper()
            
            # flags for section-level filtering in global Incidences UI
            sol_wf = _normalize_solution_meta(_parse_supplier_solution_meta(_s(getattr(wf, "note", None)) if wf else ""))
            res = (_s(sol_wf.get("resolution")) or "").strip().lower()
            items = sol_wf.get("items") or []
            has_rd_items = any(isinstance(it, dict) and _is_redelivery_item(it) for it in (items or []))
            has_cn_items = any(isinstance(it, dict) and (not _is_redelivery_item(it)) for it in (items or []))

            has_cn_ticket = False
            has_rd_ticket = False
            for tix in (open_t or []):
                meta = _parse_ticket_resolution_note(_s(getattr(tix, "resolution_note", None)))
                r = _s(meta.get("resolution")).strip().lower()
                if r == "credit_note":
                    has_cn_ticket = True
                if r in {"supplementary_delivery", "re_delivery"}:
                    has_rd_ticket = True

            has_credit_note = bool(has_cn_items or has_cn_ticket or res == "credit_note")
            has_redelivery = bool(has_rd_items or has_rd_ticket or res in {"supplementary_delivery", "re_delivery"})
            out.append(
                {
                    "order": o,
                    "order_id": oid,
                    "provider": prov,
                    "invoice_number": inv,
                    "open_count": int(len(open_t)),
                    "workflow_state": wf_state,
                    "has_credit_note": has_credit_note,
                    "has_redelivery": has_redelivery,
                    "has_open": True,
                }
            )

    out.sort(key=lambda r: (-int(r["open_count"]), -int(r["order_id"]), (r["provider"] or "").lower()))
    return out


def _list_open_urgent_requests_grouped() -> Dict[int, List[UrgentReorderRequest]]:
    """Group pending urgent requests by the *source order id* (via incidence SeguimientoTicket)."""
    reqs = _list_open_urgent_requests()
    if not reqs:
        return {}

    inc_ids = sorted({int(getattr(r, "incidence_id", 0) or 0) for r in reqs if int(getattr(r, "incidence_id", 0) or 0)})
    if not inc_ids:
        return {}

    with get_session() as s:
        tickets = list(s.exec(select(SeguimientoTicket).where(SeguimientoTicket.id.in_(inc_ids))).all())
    ticket_order_by_id = {int(t.id): int(getattr(t, "order_id", 0) or 0) for t in tickets if getattr(t, "id", None) is not None}

    grouped: Dict[int, List[UrgentReorderRequest]] = {}
    for r in reqs:
        inc = int(getattr(r, "incidence_id", 0) or 0)
        oid = int(ticket_order_by_id.get(inc, 0) or 0)
        if not oid:
            continue
        grouped.setdefault(oid, []).append(r)

    # newest requests first per group
    for oid in list(grouped.keys()):
        grouped[oid].sort(key=lambda x: getattr(x, "created_at", None) or _now(), reverse=True)

    return grouped


def _render_urgent_requests_for_order(ctx: OrderContext, reqs: List[UrgentReorderRequest], *, key_prefix: str) -> None:
    """Same UX as _render_urgent_tab, but scoped to one source order and a provided req list."""
    st.markdown("### ⚡ Urgent reorders")

    if not reqs:
        st.info("No urgent reorder requests for this order.")
        return

    providers_by_name = ctx.providers_by_name or {}
    products_by_id = ctx.products_by_id or {}

    with st.expander("⚙️ Send settings", expanded=False):
        c1, c2 = st.columns([1.0, 1.0], vertical_alignment="center")
        with c1:
            use_email = st.toggle("Email", value=True, key=f"{key_prefix}_use_email")
        with c2:
            use_wa = st.toggle("WhatsApp", value=False, key=f"{key_prefix}_use_wa")
        wa_cc = st.text_input("Prefijo país (WhatsApp)", value="+34", key=f"{key_prefix}_wa_cc")

    pending_send: Dict[str, List[Dict[str, Any]]] = {}

    for r in reqs:
        rid = int(getattr(r, "id", 0) or 0)
        pname = _s(getattr(r, "product_name", "")) or "—"
        qty = float(_safe_float(getattr(r, "quantity", 0.0), 0.0))
        srcp = norm_provider(_s(getattr(r, "original_provider_name", "")))
        unit = _s(getattr(r, "unit", "")) or "unit"

        with st.container(border=True):
            st.markdown(f"**{pname}**")
            st.caption(f"Qty: {qty:g} {unit} · From: {srcp or '—'}")

            suggestions = _suggest_providers_for_request(
                venue_id=int(ctx.order.venue_id),
                req=r,
                products=products_by_id,
                providers_by_name=providers_by_name,
                limit=8,
            )
            if not suggestions:
                st.warning("No provider suggestions found.")
                continue

            labels = []
            for sgg in suggestions:
                prov = _s(sgg.get("provider_name"))
                prod_name = _s(sgg.get("product_name"))
                eta = _s(sgg.get("eta_txt")) or "—"
                score = float(sgg.get("score") or 0.0)
                labels.append(f"{prov} — {prod_name} · ETA {eta} · score {score:.2f}")

            chosen = st.selectbox(
                "Choose provider/product",
                options=list(range(len(suggestions))),
                format_func=lambda i: labels[int(i)],
                key=f"{key_prefix}_pick_{rid}",
            )
            sel = suggestions[int(chosen)]

            if st.button("➕ Add to send list", use_container_width=True, key=f"{key_prefix}_add_{rid}"):
                prov = _s(sel.get("provider_name")) or "—"
                pending_send.setdefault(prov, []).append(
                    {
                        "req_id": rid,
                        "product_id": sel.get("product_id"),
                        "name": _s(sel.get("product_name")) or pname,
                        "qty": qty,
                        "unit": _s(sel.get("unit")) or unit,
                    }
                )
                st.success(f"Added to send list for {prov}")

    st.divider()

    if not pending_send:
        st.info("Select at least one suggestion and click **Add to send list**.")
        return

    prov_list = sorted(pending_send.keys(), key=lambda x: x.lower())
    prov_selected = st.multiselect(
        "Providers to contact now",
        options=prov_list,
        default=prov_list,
        key=f"{key_prefix}_send_sel",
    )

    if st.button("🚀 Send urgent requests", type="primary", use_container_width=True, key=f"{key_prefix}_send_btn"):
        actor = _s(st.session_state.get("user_email") or st.session_state.get("actor") or "venue")
        any_fail = False

        # ✅ Build ONE cart for the NEW urgent order (all selected providers)
        cart: List[Dict[str, Any]] = []
        for prov in prov_selected:
            for it in (pending_send.get(prov) or []):
                cart.append(
                    {
                        "product_id": it.get("product_id"),
                        "quantity": float(it.get("qty") or 0.0),
                        "unit": it.get("unit") or "unit",
                        "provider_name": prov,
                        "spoken_name": it.get("name") or "Urgent item",
                    }
                )

        urgent_order_id = _create_urgent_order_from_cart(
            venue_id=int(ctx.order.venue_id),
            actor=actor,
            cart=cart,
            source_order_id=int(ctx.order.id),  # ✅ correct relationship
        )
        if not urgent_order_id:
            st.error("Could not create urgent order (empty cart).")
            st.stop()

        # ✅ Now send per provider using the NEW urgent order id
        for prov in prov_selected:
            items = pending_send.get(prov) or []
            if not items:
                continue

            email_ok = False
            wa_ok = False

            # 1) Email send (references urgent_order_id + includes link)
            if use_email:
                ok, msg_or_link = _send_urgent_request_email(
                    venue_id=int(ctx.order.venue_id),
                    order_id=int(urgent_order_id),   # ✅ NEW ORDER
                    provider_name=prov,
                    items=items,
                    source_order_id=int(ctx.order.id),
                )
                if ok:
                    email_ok = True
                    st.success(f"Email sent to {prov}")
                else:
                    any_fail = True
                    st.error(f"{prov}: {msg_or_link}")

            # 2) WhatsApp link (still optional)
            if use_wa:
                status, url = _build_wa_link_for_urgent(
                    venue_id=int(ctx.order.venue_id),
                    provider_name=prov,
                    items=items,
                    wa_cc=wa_cc,
                )
                if status == "ok" and url:
                    wa_ok = True
                    st.link_button(f"Open WhatsApp for {prov}", url, use_container_width=True)
                else:
                    any_fail = True
                    st.error(f"{prov}: WhatsApp phone missing/invalid")

            # ✅ Mark provider as sent for the NEW urgent order
            if email_ok or wa_ok:
                _upsert_provider_send_status(
                    venue_id=int(ctx.order.venue_id),
                    order_id=int(urgent_order_id),   # ✅ NEW ORDER
                    provider_name=prov,
                    sent_email=bool(email_ok),
                    sent_whatsapp=False,
                )

        # ✅ Update urgent requests so they disappear
        with get_session() as s:
            for prov in prov_selected:
                for it in (pending_send.get(prov) or []):
                    rid = int(it.get("req_id") or 0)
                    rr = s.exec(select(UrgentReorderRequest).where(UrgentReorderRequest.id == rid)).first()
                    if rr:
                        rr.status = "sent"
                        rr.updated_at = _now()
                        s.add(rr)
            s.commit()

        if any_fail:
            st.warning("Some providers could not be contacted. Fix issues above and retry.")
        else:
            st.success("Urgent requests sent ✓")
        st.rerun()


# =============================
# Main dashboard (global)
# =============================

def tracking_dashboard(
    venue_id: int,
    *,
    deep_order_id: Optional[int] = None,
    deep_provider: Optional[str] = None,
) -> None:
    _inject_css()

    st.markdown("# Track order")

    orders = _get_active_orders(int(venue_id))
    if not orders:
        st.info("No orders yet.")
        return

    tab_labels = ["📦 Receive", "🚨 Incidences", "⚡ Urgent", "📚 History"]
    tab_key = f"tracking_global_tab_{int(venue_id)}"
    st.session_state.setdefault(tab_key, tab_labels[0])
    if st.session_state[tab_key] not in tab_labels:
        st.session_state[tab_key] = tab_labels[0]

    selected_tab = st.radio(
        "Tracking navigation",
        tab_labels,
        horizontal=True,
        key=tab_key,
        label_visibility="collapsed",
    )

    # -----------------------------
    # 📦 Receive (global)
    # -----------------------------
    if selected_tab == "📦 Receive":
        tasks = _list_pending_receive_items(int(venue_id))
        if not tasks:
            st.success("✅ Nothing pending to receive right now.")
            return

        # Sticky filters
        provider_options = sorted({(_s(t.get("provider_display") or "")) for t in tasks if _s(t.get("provider_display") or "")}, key=lambda x: x.lower())
        filters = _render_global_filters(
            key_prefix=f"recv_global_{int(venue_id)}",
            orders=[t.get("order") for t in tasks if t.get("order") is not None],
            provider_options=provider_options,
            show_date=True,
            show_provider=True,
        )
        prov_set = filters.get("providers_set")
        date_from = filters.get("date_from")
        date_to = filters.get("date_to")

        f1, f2 = st.columns([2.2, 1.0], vertical_alignment="center")
        with f1:
            q = st.text_input("Search provider / invoice / order", placeholder="e.g. makro, 2026-, #12").strip().lower()
        with f2:
            expand_all = st.toggle("Expand all", value=False)
            
            
        def _matches(t: Dict[str, Any]) -> bool:
            """Filter pending receive tasks by provider/date/search."""
            prov_raw = _s(t.get("provider_display") or "")
            prov_n = norm_provider(prov_raw) if prov_raw else ""

            if prov_set is not None and prov_n not in prov_set:
                return False

            dt = _order_created_dt(t.get("order"))
            if (date_from or date_to) and dt:
                d = dt.date()
                if date_from and d < date_from:
                    return False
                if date_to and d > date_to:
                    return False

            if not q:
                return True
            prov = prov_raw.lower()
            inv = (_s(t.get("invoice_number") or "")).lower()
            oid = str(t.get("order_id") or "")
            return (q in prov) or (q in inv) or (q in oid) or (q in f"#{oid}")

        tasks2 = [t for t in tasks if _matches(t)]
        if not tasks2:
            st.info("No matches.")
            return

        for t in tasks2:
            oid = int(t["order_id"])
            prov = t["provider_display"] or "—"
            inv = t["invoice_number"] or "—"
            order_date = _fmt_order_date(t.get("order"))
            ago = _days_ago_badge(t.get("order"))
            title = f"{prov} · 🧾 {inv} · #{oid} · 📅 {order_date}" + (f" · {ago}" if ago else "")

            expanded = bool(expand_all) or (deep_order_id is not None and int(deep_order_id) == oid)
            with st.expander(title, expanded=expanded):
                # URL sync (optional)
                if qp_int("order_id") != oid:
                    set_query_params(page="tracking", order_id=str(oid), provider=norm_provider(prov))

                ctx = _load_order_context(int(venue_id), oid)
                _render_receive_provider_panel(ctx, prov)
        return

    # -----------------------------
    # 🚨 Incidences (global)
    # -----------------------------
    if selected_tab == "🚨 Incidences":
        items = _list_open_incidences_items(int(venue_id))
        if not items:
            st.success("✅ No open incidences.")
            return

        # Sticky filters
        provider_options = sorted({(_s(t.get("provider") or "")) for t in items if _s(t.get("provider") or "")}, key=lambda x: x.lower())
        filters = _render_global_filters(
            key_prefix=f"inc_global_{int(venue_id)}",
            orders=[t.get("order") for t in items if t.get("order") is not None],
            provider_options=provider_options,
            show_date=True,
            show_provider=True,
        )
        prov_set = filters.get("providers_set")
        date_from = filters.get("date_from")
        date_to = filters.get("date_to")

        
        # Incidence type filter (what the expander contains)
        type_key = f"inc_global_{int(venue_id)}_type_filter"
        type_options = [
            "Open incidences",
            "Credit note",
            "Re-delivery",
        ]
        if st.session_state.get(type_key) is None:
            st.session_state[type_key] = type_options[:]  # all by default

        sel_types = st.multiselect(
            "Show",
            options=type_options,
            default=st.session_state.get(type_key, type_options[:]),
            key=type_key,
            help="Filter by the kind of incidence you want to work on.",
        )

        def _type_match(t: Dict[str, Any]) -> bool:
            if not sel_types:
                return True
            if ("Open incidences" in sel_types) and bool(t.get("has_open", False)):
                return True
            if ("Credit note" in sel_types) and bool(t.get("has_credit_note", False)):
                return True
            if ("Re-delivery" in sel_types) and bool(t.get("has_redelivery", False)):
                return True
            return False

# Incidence stage filter (by provider workflow state)
        stage_key = f"inc_global_{int(venue_id)}_stage_filter"
        stage_options = [
            "Open (not requested)",
            "Awaiting supplier",
            "Credit note",
            "Re-delivery",
        ]
        default_stage = st.session_state.get(stage_key)
        if default_stage is None:
            default_stage = stage_options[:]  # all
            st.session_state[stage_key] = default_stage

        sel_stage = st.multiselect(
            "Incidence stage",
            options=stage_options,
            default=st.session_state.get(stage_key, stage_options[:]),
            key=stage_key,
            help="Filter by where the incidence is in the workflow.",
        )

        def _stage_match(wf_state: str) -> bool:
            if not sel_stage:
                return True
            wf_state = (wf_state or "").upper()
            allowed: set[str] = set()

            if "Open (not requested)" in sel_stage:
                allowed |= {"INVOICE_DISCREPANCY"}

            if "Awaiting supplier" in sel_stage:
                allowed |= {"WAITING_SUPPLIER_ACTION"}

            if "Credit note" in sel_stage:
                allowed |= {"SUPPLIER_CREDIT_NOTE_PENDING", "SUPPLIER_CREDIT_NOTE_ISSUED"}

            if "Re-delivery" in sel_stage:
                allowed |= {"SUPPLEMENTARY_DELIVERY_SENT"}

            # If we don't recognize state, treat as not matching (keeps filter strict)
            return wf_state in allowed

        f1, f2 = st.columns([2.2, 1.0], vertical_alignment="center")
        with f1:
            q = st.text_input("Search provider / invoice / order", key="inc_global_search", placeholder="e.g. invoice, #34").strip().lower()
        with f2:
            expand_all = st.toggle("Expand all", value=False, key="inc_global_expand_all")

        def _matches_inc(t: Dict[str, Any]) -> bool:
            prov_raw = _s(t.get("provider") or "")
            prov_n = norm_provider(prov_raw) if prov_raw else ""
            if prov_set is not None and prov_n not in prov_set:
                return False

            if not _type_match(t):
                return False

            dt = _order_created_dt(t.get("order"))
            if (date_from or date_to) and dt:
                d = dt.date()
                if date_from and d < date_from:
                    return False
                if date_to and d > date_to:
                    return False

            wf_state = _s(t.get("workflow_state") or "")
            if wf_state and not _stage_match(wf_state):
                return False

            if not q:
                return True
            prov = prov_raw.lower()
            inv = (_s(t.get("invoice_number") or "")).lower()
            oid = str(t.get("order_id") or "")
            return (q in prov) or (q in inv) or (q in oid) or (q in f"#{oid}")

        items2 = [t for t in items if _matches_inc(t)]
        if not items2:
            st.info("No matches.")
            return

        for t in items2:
            oid = int(t["order_id"])
            prov = _s(t.get("provider") or "—")
            inv = _s(t.get("invoice_number") or "—")
            n = int(t.get("open_count") or 0)
            order_date = _fmt_order_date(t.get("order"))
            ago = _days_ago_badge(t.get("order"))
            title = f"{prov} · 🧾 {inv} · #{oid} · 📅 {order_date}" + (f" · {ago}" if ago else "") + f" · {n} open"

            expanded = bool(expand_all) or (deep_order_id is not None and int(deep_order_id) == oid)
            with st.expander(title, expanded=expanded):
                if qp_int("order_id") != oid:
                    set_query_params(page="tracking", order_id=str(oid), provider=norm_provider(prov))

                ctx = _load_order_context(int(venue_id), oid)
                _render_incidences_cards(ctx, [prov])
        return

    # -----------------------------
    # ⚡ Urgent (global)
    # -----------------------------
    if selected_tab == "⚡ Urgent":
        grouped = _list_open_urgent_requests_grouped()
        if not grouped:
            st.info("No urgent reorder requests yet. Use **Order urgent** in Incidences and click **Save & request decision**.")
            return

        # Sticky filters
        all_reqs = [r for lst in grouped.values() for r in (lst or [])]
        provider_options = sorted(
            {
                _s(getattr(r, "original_provider_name", None) or getattr(r, "provider_name", None) or "")
                for r in all_reqs
                if _s(getattr(r, "original_provider_name", None) or getattr(r, "provider_name", None) or "")
            },
            key=lambda x: x.lower(),
        )

        order_by_id = {int(getattr(o, "id", 0) or 0): o for o in orders if getattr(o, "id", None) is not None}
        urgent_orders = [order_by_id.get(int(oid)) for oid in grouped.keys() if order_by_id.get(int(oid)) is not None]

        filters = _render_global_filters(
            key_prefix=f"urg_global_{int(venue_id)}",
            orders=urgent_orders,
            provider_options=provider_options,
            show_date=True,
            show_provider=True,
        )
        prov_set = filters.get("providers_set")
        date_from = filters.get("date_from")
        date_to = filters.get("date_to")

        # newest source orders first
        order_ids = sorted(grouped.keys(), reverse=True)
        for oid in order_ids:
            oid_i = int(oid)

            # provider filter on requests (before loading ctx)
            reqs = list(grouped.get(oid_i) or [])
            if prov_set is not None:
                reqs = [
                    r
                    for r in reqs
                    if norm_provider(
                        _s(getattr(r, "original_provider_name", None) or getattr(r, "provider_name", None) or "")
                    )
                    in prov_set
                ]
            if not reqs:
                continue

            order_obj = order_by_id.get(oid_i)
            dt = _order_created_dt(order_obj) if order_obj is not None else None
            if (date_from or date_to) and dt is not None:
                d = dt.date()
                if date_from and d < date_from:
                    continue
                if date_to and d > date_to:
                    continue

            ctx = _load_order_context(int(venue_id), oid_i)

            order_date = _fmt_order_date(ctx.order)
            ago = _days_ago_badge(ctx.order)

            title = f"#{oid_i} · 📅 {order_date}" + (f" · {ago}" if ago else "") + f" · {len(reqs)} requests"

            expanded = (deep_order_id is not None and int(deep_order_id) == oid_i)
            with st.expander(title, expanded=expanded):
                set_query_params(page="tracking", order_id=str(oid_i))
                _render_urgent_requests_for_order(ctx, reqs, key_prefix=f"urg_{oid_i}")
        return

    # -----------------------------
    # 📚 History (already global)
    # -----------------------------
    if selected_tab == "📚 History":
        _render_history_tab(int(venue_id), deep_provider=deep_provider)
        return