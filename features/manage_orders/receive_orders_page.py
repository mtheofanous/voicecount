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

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple
import json
import math
import re
import pandas as pd
import streamlit as st


def _orders_refresh_token(venue_id: int) -> int:
    """Session-state based cache buster for order-related caches."""
    return int(st.session_state.get(f"orders_refresh_token_{int(venue_id)}", 0) or 0)


def _bump_orders_refresh_token(venue_id: int) -> None:
    """Increment the venue refresh token to bust cached reads.

    Any write that affects orders / tickets / workflows should call this to force
    `_get_active_orders`, `_load_order_context`, `_load_dashboard_bundle`, etc.
    to recompute on the next rerun.
    """
    k = f"orders_refresh_token_{int(venue_id)}"
    st.session_state[k] = int(st.session_state.get(k, 0) or 0) + 1

from sqlmodel import select
import html
from core.db import get_session
from difflib import SequenceMatcher

from core.mailer import send_smtp_email
from core.public_links import ROLE_SUPPLIER, build_seguimiento_url, norm_provider
from core.url_nav import set_query_params, qp_int, qp_str
from datetime import datetime, timedelta

from features.manage_orders.orders import _load_venue_templates
from features.manage_orders.emails import build_resolution_email_full, build_urgent_email_full

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
    UrgentReorderRequest, ProviderResolution

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

/* ---------- Invoice-like mini table ---------- */
.inv{margin-top:6px;}
.inv-t{
  border-collapse:collapse;
  margin-left:auto;          /* align right */
  font-size:.78rem;
  color:var(--text);
}
.inv-t td{padding:1px 0;vertical-align:top;}
.inv-k{padding-right:10px;color:var(--muted);}
.inv-v{width:78px;text-align:right;padding-right:10px;color:var(--text);}
.inv-a{width:96px;text-align:right;font-variant-numeric:tabular-nums;color:var(--text);}

.inv-total td{
  padding-top:6px;
  border-top:1px solid var(--border);
  font-weight:900;
  font-size:.85rem;
  color:var(--text);
}

/* Optional: make discount slightly muted (invoice style) */
.inv-discount .inv-k,
.inv-discount .inv-v,
.inv-discount .inv-a{
  color:#92400e;
}

/* Optional: tax slightly muted */
.inv-tax .inv-k,
.inv-tax .inv-v,
.inv-tax .inv-a{
  color:var(--muted);
}
/* ---------- Invoice table (many products) ---------- */
.inv-wrap{margin-top:10px;}
.inv-table{
  width:100%;
  border-collapse:separate;
  border-spacing:0;
  border:1px solid var(--border);
  border-radius:14px;
  overflow:hidden;
  background:#fff;
  font-variant-numeric: tabular-nums;
}
.inv-table th, .inv-table td{
  padding:8px 10px;
  border-bottom:1px solid var(--border);
  font-size:.82rem;
}
.inv-table thead th{
  position:sticky; top:0; /* nice inside popovers */
  background:#f8fafc;
  color:var(--muted);
  font-weight:900;
  text-transform:uppercase;
  letter-spacing:.02em;
  font-size:.72rem;
}
.inv-table td.name{
  width:44%;
  font-weight:900;
  color:var(--text);
}
.inv-table td.num, .inv-table th.num{
  text-align:right;
  white-space:nowrap;
}
.inv-table td.muted{
  color:var(--muted);
  font-weight:800;
}
.inv-table tfoot td{
  background:#f8fafc;
  font-weight:950;
  border-bottom:none;
}
.inv-table tr:last-child td{border-bottom:none;}
.inv-neg{color:#b45309;font-weight:900;}
.inv-meta{margin-top:2px;color:var(--muted);font-size:.75rem;font-weight:800;opacity:.9;}
.inv-table td.name{width:32%;}
.inv-desc{margin-top:2px;color:var(--muted);font-size:.75rem;font-weight:700;opacity:.9;}
/* ---------- Invoice header ---------- */
.inv-head{
  display:flex;
  justify-content:space-between;
  align-items:baseline;
  gap:12px;
  margin:6px 4px 10px 4px;
  color:var(--muted);
  font-weight:900;
  font-size:.82rem;
}
.inv-head-left{
  white-space:nowrap;
}
.inv-head-right{
  white-space:nowrap;
}
.inv-table td.reason{color:var(--muted);font-weight:800;font-size:.78rem;white-space:nowrap;}
.inv-flag{
  margin-left:6px;
  padding:2px 8px;
  border-radius:999px;
  border:1px solid var(--border);
  background:#fff7ed;
  color:#9a3412;
  font-weight:900;
  font-size:.68rem;
}


.small{font-size:.85rem;color:var(--muted);}

/* ---------- Supplier/Venue comments (compact) ---------- */
.voi-commentline{
  margin:-4px 0 8px 0;
  padding:8px 12px;
  border:1px solid var(--border);
  border-radius:14px;
  background:#f8fafc;
  color:var(--text);
  font-weight:800;
  font-size:.85rem;
}
.voi-commentline .lbl{color:var(--muted);font-weight:900;margin-right:6px;}
.voi-commentline .txt{font-weight:800; color:var(--text); opacity:.92;}

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



_VENUE_STATUS_RE = re.compile(r"\[VENUE_STATUS=([a-zA-Z_]+)\]")

def _get_received_info(ctx: OrderContext, prov: str, lid: int) -> tuple[str, float, float, bool, str]:
    """
    Returns:
      (issue_status, received_qty, issue_qty, in_invoice, reason)

    - issue_status: "ok" | "missing" | "damaged" | "wrong_item" | "unknown"
    - received_qty: what venue received (fu.venue_qty) (may be 0)
    - issue_qty: what venue marked as missing/damaged/wrong qty (fu.qty_invoiced reused)
    - in_invoice: True/False depending on invoice_listed when Missing. For damaged/wrong we treat as True.
    - reason: human readable reason (status + optional comment)
    """
    fu = ctx.followups_by_key.get((prov, int(lid)))

    # Defaults
    issue_status = "unknown"
    received_qty = 0.0
    issue_qty = 0.0
    in_invoice = True
    reason = "—"

    if not fu:
        return issue_status, received_qty, issue_qty, in_invoice, reason

    # Quantities saved by "Venue received" flow
    received_qty = _safe_float(getattr(fu, "venue_qty", None), 0.0)
    issue_qty = _safe_float(getattr(fu, "qty_invoiced", None), 0.0)  # you reuse this as issue qty

    # Status is stored as a tag in venue_comment: "[VENUE_STATUS=missing]"
    vc = _s(getattr(fu, "venue_comment", None)).strip()
    m = _VENUE_STATUS_RE.search(vc)
    if m:
        issue_status = (m.group(1) or "").strip().lower()

    # invoice_listed only meaningful for Missing; can be True/False/None
    inv = getattr(fu, "invoice_listed", None)
    if issue_status == "missing":
        if inv is True:
            in_invoice = True
        elif inv is False:
            in_invoice = False
        else:
            in_invoice = True  # treat unknown as "in invoice" to avoid accidentally pricing it out
    elif issue_status in ("damaged", "wrong_item"):
        in_invoice = True  # your UI forces "In invoice" for these anyway
    else:
        in_invoice = True

    # Human reason: use status label + any free text (without the tag)
    status_label = {
        "ok": "OK",
        "missing": "Missing",
        "damaged": "Damaged",
        "wrong_item": "Wrong item",
        "unknown": "—",
    }.get(issue_status, issue_status.upper() if issue_status else "—")

    # remove the tag from the comment to keep it clean
    clean_comment = _VENUE_STATUS_RE.sub("", vc).strip()
    if clean_comment:
        reason = f"{status_label} · {clean_comment}"
    else:
        reason = status_label if status_label != "—" else "—"

    return issue_status, received_qty, issue_qty, in_invoice, reason




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


import unicodedata
import re

_STOP = {
    "de", "del", "la", "el", "los", "las", "con", "sin", "para", "por",
    "and", "or", "the", "a", "an",
    "kg", "kilo", "kilos", "gr", "g", "ml", "l", "lt", "litro", "litros",
    "uds", "ud", "unidad", "unidades", "pcs", "pc", "pack", "x",
}

_UNIT_ALIASES = {
    "kg": {"kg", "kilo", "kilos"},
    "g": {"g", "gr"},
    "l": {"l", "lt", "litro", "litros"},
    "ml": {"ml"},
    "ud": {"ud", "uds", "unidad", "unidades", "pcs", "pc"},
}

def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def _norm_words(s: str) -> set[str]:
    """
    Stronger tokenization for catalog matching:
    - lowercase + strip accents
    - split to alnum tokens
    - remove stop words / unit aliases normalized
    - light singularization
    """
    s = _strip_accents((s or "").lower())
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    raw = [t for t in s.split() if t.strip()]

    out: set[str] = set()
    for t in raw:
        if t in _STOP:
            continue

        # normalize units
        for canon, alts in _UNIT_ALIASES.items():
            if t in alts:
                out.add(canon)
                t = ""
                break
        if not t:
            continue

        # naive singularization
        if len(t) > 4 and t.endswith("s"):
            t = t[:-1]
        if len(t) > 5 and t.endswith("es"):
            t = t[:-2]

        if t and t not in _STOP:
            out.add(t)

    return out

def _extract_unit_tokens(tokens: set[str]) -> set[str]:
    return {t for t in tokens if t in {"kg", "g", "l", "ml", "ud"}}

def _relevance(a: str, b: str) -> float:
    """
    Improved relevance:
    - base: Jaccard
    - + containment bonus
    - + unit match bonus
    """
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return 0.0

    A = _norm_words(a)
    B = _norm_words(b)
    if not A or not B:
        return 0.0

    inter = len(A & B)
    uni = len(A | B)
    base = float(inter) / float(uni) if uni else 0.0

    contains_bonus = 0.0
    a_norm = " ".join(sorted(A))
    b_norm = " ".join(sorted(B))
    if a_norm and b_norm and (a_norm in b_norm or b_norm in a_norm):
        contains_bonus = 0.15

    unit_bonus = 0.0
    ua = _extract_unit_tokens(A)
    ub = _extract_unit_tokens(B)
    if ua and ub and ua == ub:
        unit_bonus = 0.10

    score = base + contains_bonus + unit_bonus
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0
    return float(score)


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


# def _get_or_create_draft_for_op_missing(*, venue_id: int, actor: str) -> int:
#     """Return an existing draft order id or create a new one."""
#     with get_session() as s:
#         o = s.exec(
#             select(Order)
#             .where(Order.venue_id == int(venue_id), Order.status == "draft")
#             .order_by(Order.created_at.desc())
#         ).first()
#         if o and getattr(o, "id", None) is not None:
#             return int(o.id)

#         new_o = Order(
#             venue_id=int(venue_id),
#             status="draft",
#             created_at=_now(),
#             updated_at=_now(),
#             created_by=actor,
#             updated_by=actor,
#             title="Operational missing (draft)",
#             note="Auto-created draft for non-urgent operational missing.",
#         )
#         s.add(new_o)
#         s.commit()
#         s.refresh(new_o)
#         return int(new_o.id)


# def _add_product_to_order(*, venue_id: int, order_id: int, actor: str, product_id: Optional[int], qty: float, unit: str, provider_name: str, spoken_name: str = "") -> None:
#     """Add (or increment) a product line in an order.

#     Simplicity rules:
#     - If same product_id already exists in order, increment qty.
#     - We never create invoices here (new order => new invoice later).
#     """
#     qty = _safe_float(qty, 0.0)
#     if qty <= 0:
#         return
#     now = _now()
#     provider_name = norm_provider(provider_name)
#     with get_session() as s:
#         ln = None
#         if product_id is not None:
#             ln = s.exec(
#                 select(OrderLine).where(OrderLine.order_id == int(order_id), OrderLine.product_id == int(product_id))
#             ).first()
#         if ln:
#             ln.quantity = float(_safe_float(getattr(ln, "quantity", 0.0), 0.0) + qty)
#             ln.updated_at = now
#             ln.updated_by = actor
#             s.add(ln)
#         else:
#             s.add(
#                 OrderLine(
#                     venue_id=int(venue_id),
#                     order_id=int(order_id),
#                     product_id=(int(product_id) if product_id is not None else None),
#                     spoken_name=_s(spoken_name),
#                     quantity=float(qty),
#                     unit=(_s(unit) or "unit"),
#                     provider=(provider_name or None),
#                     updated_at=now,
#                     updated_by=actor,
#                 )
#             )

#         o = s.exec(select(Order).where(Order.id == int(order_id))).first()
#         if o:
#             o.updated_at = now
#             o.updated_by = actor
#             s.add(o)
#         s.commit()


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

# def _next_delivery_datetime(provider: Optional[Provider], now: Optional[datetime] = None) -> Optional[datetime]:
#     """Return the next delivery datetime based on provider.delivery_schedule_json."""
#     if now is None:
#         now = _now()
#     if not provider:
#         return None
#     sched = _parse_delivery_schedule(getattr(provider, "delivery_schedule_json", None))
#     if not sched:
#         return None

#     def _slot_start(slot: str) -> Optional[Tuple[int, int]]:
#         m = re.match(r"^\s*(\d{1,2}):(\d{2})\s*-", slot or "")
#         if not m:
#             return None
#         return int(m.group(1)), int(m.group(2))

#     best: Optional[datetime] = None
#     for add_days in range(0, 8):
#         d = now.date() + timedelta(days=add_days)
#         key = _WEEKDAY_KEYS[(now.weekday() + add_days) % 7]
#         slots = sched.get(key) or []
#         for slot in slots:
#             hm = _slot_start(slot)
#             if not hm:
#                 continue
#             cand = datetime.combine(d, datetime.min.time()).replace(hour=hm[0], minute=hm[1])
#             if cand < now:
#                 continue
#             if best is None or cand < best:
#                 best = cand
#     return best

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

    # 1) Resolve supplier emails
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

    # 2) Load venue templates (for footer + language)
    templates = _load_venue_templates(int(venue_id), 0)
    lang = _s(getattr(templates, "email_lang", "en")) or "en"

    # 3) Supplier link
    link = build_seguimiento_url(
        order_id=int(order_id),
        provider_name=prov,
        role=ROLE_SUPPLIER,
        page_path="seguimiento",
    )

    # 4) Enrich items with description (best effort)
    # Expecting items like: {"product_id": 123, "name": "...", "qty": 2, "unit": "kg"}
    with get_session() as s:
        pids = [int(it["product_id"]) for it in items if it.get("product_id") is not None]
        products_by_id: Dict[int, Product] = {}
        if pids:
            prows = list(
                s.exec(
                    select(Product).where(
                        Product.venue_id == int(venue_id),
                        Product.id.in_(list(set(pids))),
                    )
                ).all()
            )
            products_by_id = {int(pp.id): pp for pp in prows if getattr(pp, "id", None) is not None}

    enriched: List[Dict[str, Any]] = []
    for it in items:
        pid = it.get("product_id")
        prod = products_by_id.get(int(pid)) if pid is not None else None

        enriched.append(
            {
                "name": _s(it.get("name") or getattr(prod, "name", "") or "Product"),
                "description": _s(getattr(prod, "description", "")) if prod else _s(it.get("description") or ""),
                "qty": it.get("qty"),
                "unit": it.get("unit"),
            }
        )

    # 5) Subject + body from centralized builder (includes footer + consistent format)
    subject, body = build_urgent_email_full(
        venue_ctx=templates,
        order_id=int(order_id),
        provider_name=prov,
        supplier_link=link,
        items=enriched,
        source_order_id=source_order_id,
        lang=lang,
    )

    # 6) Send
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


# def _move_urgent_items_into_order(
#     *, venue_id: int, order_id: int, provider_name: str, items: List[Dict[str, Any]], actor: str
# ) -> None:
#     """Create OrderLines in this same order, so they appear in Receive."""
#     prov = norm_provider(provider_name)
#     now = _now()

#     with get_session() as s:
#         for it in (items or []):
#             pid = it.get("product_id")
#             pid_i = None
#             try:
#                 if pid is not None:
#                     pid_i = int(pid)
#             except Exception:
#                 pid_i = None

#             qty = float(it.get("qty") or 0.0)
#             if qty <= 0:
#                 continue

#             unit = _s(it.get("unit")) or "unit"
#             name = _s(it.get("name")) or "Urgent item"

#             # If same product already exists for same provider in this order, increment qty
#             existing = None
#             if pid_i is not None:
#                 existing = s.exec(
#                     select(OrderLine).where(
#                         OrderLine.order_id == int(order_id),
#                         OrderLine.product_id == int(pid_i),
#                         OrderLine.provider == prov,
#                     )
#                 ).first()

#             if existing:
#                 existing.quantity = float(_safe_float(getattr(existing, "quantity", 0.0), 0.0) + qty)
#                 existing.updated_at = now
#                 existing.updated_by = actor
#                 s.add(existing)
#             else:
#                 s.add(
#                     OrderLine(
#                         venue_id=int(venue_id),
#                         order_id=int(order_id),
#                         product_id=pid_i,
#                         provider=prov,
#                         spoken_name=f"{name} [URGENT]",
#                         quantity=float(qty),
#                         unit=unit,
#                         updated_at=now,
#                         updated_by=actor,
#                     )
#                 )

#         # ensure workflow exists (Receive uses it)
#         wf = s.exec(
#             select(OrderWorkflow).where(
#                 OrderWorkflow.order_id == int(order_id),
#                 OrderWorkflow.provider_name == prov,
#             )
#         ).first()
#         if not wf:
#             wf = OrderWorkflow(
#                 venue_id=int(venue_id),
#                 order_id=int(order_id),
#                 provider_name=prov,
#                 state="ORDER_SENT",
#                 updated_at=now,
#                 updated_by=actor,
#             )
#             s.add(wf)

#         s.commit()

# def _ensure_order_pending_receive(*, order_id: int, actor: str) -> None:
#     """Make sure the order is visible in Receive/Track Order."""
#     now = _now()
#     with get_session() as s:
#         o = s.exec(select(Order).where(Order.id == int(order_id))).first()
#         if not o:
#             return

#         # Only bump from draft -> pending_receive (do not downgrade other states)
#         if (_s(getattr(o, "status", "")).lower() in {"draft", "borrador"}):
#             o.status = "pending_receive"
#             o.updated_at = now
#             o.updated_by = actor or "venue"
#             s.add(o)
#             s.commit()

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

    # ✅ NEW: used for the Suggested badge (delivery within 24h + relevance ≥ 0.45)
    now_dt = _now()

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

            # ✅ UPDATED label: adds "✅ Suggested" when fast + strong match
            def _lbl(sug: Dict[str, Any]) -> str:
                dt = sug.get("delivery_dt")
                dt_txt = dt.strftime("%a %d %b %H:%M") if isinstance(dt, datetime) else "—"

                rel = float(sug.get("relevance") or 0.0)
                badge = ""
                if isinstance(dt, datetime):
                    hrs = (dt - now_dt).total_seconds() / 3600.0
                    if 0 <= hrs <= 24 and rel >= 0.45:
                        badge = "✅ Suggested · "

                return f"{badge}{sug['provider_name']} · {sug.get('eta_txt','—')} · {dt_txt}"

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

    if st.button(
        "🚀 Send urgent requests",
        type="primary",
        use_container_width=True,
        key=f"urg_send_btn_{int(ctx.order.id)}",
    ):
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
                    order_id=int(urgent_order_id),  # ✅ NEW ORDER
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
                    order_id=int(urgent_order_id),  # ✅ NEW ORDER
                    provider_name=prov,
                    sent_email=bool(email_ok),
                    sent_whatsapp=False,
                )

        # ✅ Update urgent requests so they disappear
        with get_session() as s:
            for prov in prov_selected:
                for it in (pending_send.get(prov) or []):
                    rid2 = int(it.get("req_id") or 0)
                    rr = s.exec(select(UrgentReorderRequest).where(UrgentReorderRequest.id == rid2)).first()
                    if rr:
                        rr.status = "sent"
                        try:
                            rr.selected_provider_name = prov
                        except Exception:
                            pass
                        try:
                            rr.updated_at = _now()
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
    if not text:
        return ""
    kind = kind if kind in {"ok", "warn", "bad", "info"} else "info"
    return f"<span class='voi-badge {kind}'>{html.escape(text)}</span>"


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
    resolutions_by_provider: Dict[str, List[ProviderResolution]] = field(default_factory=dict)


# =============================
# Data loading
# =============================

@st.cache_data(show_spinner=False, ttl=15)
def _get_active_orders(venue_id: int, *, refresh_token: int = 0) -> List[Order]:
    """Fast: cached list of recent orders for the venue.

    refresh_token is only used to invalidate cache when orders change.
    """
    _ = int(refresh_token or 0)
    with get_session() as s:
        return list(
            s.exec(
                select(Order)
                .where(Order.venue_id == int(venue_id))
                .order_by(Order.created_at.desc())
                .limit(200)
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

@st.cache_data(show_spinner=False, ttl=15)
def _load_order_context(venue_id: int, order_id: int, *, refresh_token: int = 0) -> OrderContext:
    _ = int(refresh_token or 0)
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

        # ✅ NEW: load ProviderResolution rows and group by provider
        resolutions = list(
            s.exec(
                select(ProviderResolution)
                .where(ProviderResolution.order_id == int(order_id))
            ).all()
        )
        resolutions_by_provider: Dict[str, List[ProviderResolution]] = {}
        for r in resolutions:
            prov = norm_provider(getattr(r, "provider_name", "") or "")
            if prov:
                resolutions_by_provider.setdefault(prov, []).append(r)

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
        resolutions_by_provider=resolutions_by_provider,  # ✅ NEW
    )



@st.cache_data(show_spinner=False, ttl=20)
def _load_dashboard_bundle(
    venue_id: int,
    order_ids: tuple[int, ...],
    *,
    refresh_token: int = 0,
) -> dict[str, Any]:
    """Bulk-load everything the dashboard needs in a few queries.

    Goal: avoid calling _load_order_context() repeatedly inside dashboard loops.
    Cache is busted via refresh_token (orders_refresh_token_{venue_id}).
    """
    _ = int(refresh_token or 0)
    if not order_ids:
        return {"contexts": {}, "sent_providers_by_order": {}}

    # ---- bulk queries ----
    with get_session() as s:
        orders = list(s.exec(select(Order).where(Order.id.in_(list(order_ids)))).all())

        providers = list(s.exec(select(Provider).where(Provider.venue_id == int(venue_id))).all())
        providers_by_name_global = {norm_provider(p.name): p for p in providers if getattr(p, "name", None)}
        orders_by_id = {int(o.id): o for o in orders if getattr(o, "id", None) is not None}

        workflows = list(
            s.exec(select(OrderWorkflow).where(OrderWorkflow.order_id.in_(list(order_ids)))).all()
        )
        workflows_by_ok: dict[tuple[int, str], OrderWorkflow] = {}
        for w in workflows:
            oid = int(getattr(w, "order_id", 0) or 0)
            prov = norm_provider(getattr(w, "provider_name", "") or "")
            if oid and prov:
                workflows_by_ok[(oid, prov)] = w

        tickets = list(
            s.exec(select(SeguimientoTicket).where(SeguimientoTicket.order_id.in_(list(order_ids)))).all()
        )
        tickets_by_ok: Dict[Tuple[int, str], List[SeguimientoTicket]] = {}
        for t in tickets:
            oid = int(getattr(t, "order_id", 0) or 0)
            prov = norm_provider(getattr(t, "provider_name", "") or "")
            if oid and prov:
                tickets_by_ok.setdefault((oid, prov), []).append(t)

        lines = list(s.exec(select(OrderLine).where(OrderLine.order_id.in_(list(order_ids)))).all())

        # products
        product_ids = sorted({int(l.product_id) for l in lines if getattr(l, "product_id", None)})
        products: Dict[int, Product] = {}
        if product_ids:
            ps = list(s.exec(select(Product).where(Product.id.in_(product_ids))).all())
            products = {int(p.id): p for p in ps if getattr(p, "id", None) is not None}

        followups = list(
            s.exec(select(ProviderLineFollowUp).where(ProviderLineFollowUp.order_id.in_(list(order_ids)))).all()
        )
        followups_by_okl: Dict[Tuple[int, str, int], ProviderLineFollowUp] = {}
        for fu in followups:
            oid = int(getattr(fu, "order_id", 0) or 0)
            prov = norm_provider(getattr(fu, "provider_name", "") or "")
            lid = int(getattr(fu, "order_line_id", 0) or 0)
            if oid and prov and lid:
                followups_by_okl[(oid, prov, lid)] = fu

        receipts = list(
            s.exec(select(ProviderReceipt).where(ProviderReceipt.order_id.in_(list(order_ids)))).all()
        )
        receipts_by_ok: Dict[Tuple[int, str], ProviderReceipt] = {}
        for r in receipts:
            oid = int(getattr(r, "order_id", 0) or 0)
            prov = norm_provider(getattr(r, "provider_name", "") or "")
            if oid and prov:
                receipts_by_ok[(oid, prov)] = r

        # ProviderResolution is optional in some deployments
        resolutions_by_ok: Dict[Tuple[int, str], List[Any]] = {}
        try:
            resolutions = list(
                s.exec(select(ProviderResolution).where(ProviderResolution.order_id.in_(list(order_ids)))).all()
            )
            for rr in resolutions:
                oid = int(getattr(rr, "order_id", 0) or 0)
                prov = norm_provider(getattr(rr, "provider_name", "") or "")
                if oid and prov:
                    resolutions_by_ok.setdefault((oid, prov), []).append(rr)
        except Exception:
            resolutions = []

        send_rows = list(
            s.exec(select(ProviderSendStatus).where(ProviderSendStatus.order_id.in_(list(order_ids)))).all()
        )
        sent_providers_by_order: Dict[int, set[str]] = {}
        for r in send_rows:
            oid = int(getattr(r, "order_id", 0) or 0)
            if not oid:
                continue
            if bool(getattr(r, "sent", False)) or bool(getattr(r, "sent_email", False)) or bool(getattr(r, "sent_whatsapp", False)):
                sent_providers_by_order.setdefault(oid, set()).add(norm_provider(getattr(r, "provider_name", "") or ""))

    # ---- build OrderContext per order ----
    contexts: Dict[int, OrderContext] = {}

    # group lines by (order_id, provider)
    lines_by_ok: Dict[Tuple[int, str], List[OrderLine]] = {}
    for l in lines:
        oid = int(getattr(l, "order_id", 0) or 0)
        if not oid:
            continue

        prov = ""
        p = products.get(int(l.product_id)) if getattr(l, "product_id", None) else None
        if p and getattr(p, "provider_name", None):
            prov = p.provider_name
        else:
            prov = _s(getattr(l, "provider", None))
        prov = norm_provider(prov)
        if not prov:
            continue
        lines_by_ok.setdefault((oid, prov), []).append(l)

    for oid in order_ids:
        order = orders_by_id.get(int(oid))
        if not order:
            continue

        # per-provider maps
        lines_by_provider: Dict[str, List[OrderLine]] = {}
        tickets_by_provider: Dict[str, List[SeguimientoTicket]] = {}
        workflows_by_provider: Dict[str, OrderWorkflow] = {}
        receipts_by_provider: Dict[str, ProviderReceipt] = {}
        resolutions_by_provider: Dict[str, Any] = {}
        followups_by_key: Dict[Tuple[str, int], ProviderLineFollowUp] = {}

        # providers involved
        providers = {prov for (oo, prov) in lines_by_ok.keys() if int(oo) == int(oid)}
        for prov in providers:
            lns = lines_by_ok.get((int(oid), prov), [])
            if lns:
                lines_by_provider[prov] = lns
            tks = tickets_by_ok.get((int(oid), prov), [])
            if tks:
                tickets_by_provider[prov] = tks
            w = workflows_by_ok.get((int(oid), prov))
            if w:
                workflows_by_provider[prov] = w
            r = receipts_by_ok.get((int(oid), prov))
            if r:
                receipts_by_provider[prov] = r
            rr = resolutions_by_ok.get((int(oid), prov))
            if rr:
                resolutions_by_provider[prov] = rr

            # followups per line
            for ln in lns:
                lid = int(getattr(ln, "id", 0) or 0)
                if not lid:
                    continue
                fu = followups_by_okl.get((int(oid), prov, lid))
                if fu:
                    followups_by_key[(prov, lid)] = fu


        # Build the per-order context
        contexts[int(oid)] = OrderContext(
            order=order,
            providers_by_name=providers_by_name_global,
            workflows_by_provider=workflows_by_provider,
            tickets_by_provider=tickets_by_provider,
            products_by_id=products,
            lines_by_provider=lines_by_provider,
            followups_by_key=followups_by_key,
            receipts_by_provider=receipts_by_provider,
            resolutions_by_provider=resolutions_by_provider,
        )

    return {"contexts": contexts, "sent_providers_by_order": sent_providers_by_order}






# =============================
# Comments (supplier/venue)
# =============================

def _supplier_comment_for_provider(ctx: "OrderContext", prov_key: str) -> str:
    """Best-effort supplier comment for a provider.

    Primary source: ProviderReceipt.supplier_declared_comment (supplier confirmation).
    Fallback: OrderWorkflow.note when supplier confirmed and note looks like free text.
    """
    prov_key = norm_provider(prov_key)
    receipt = getattr(ctx, "receipts_by_provider", {}).get(prov_key)
    comment = _s(getattr(receipt, "supplier_declared_comment", None))
    if comment:
        return comment

    wf = getattr(ctx, "workflows_by_provider", {}).get(prov_key)
    state = _s(getattr(wf, "state", None)).upper()
    note = _s(getattr(wf, "note", None))
    if not note:
        return ""

    # Only use workflow.note as a fallback when we're in supplier confirmation states.
    if not state.startswith("SUPPLIER_CONFIRMED"):
        return ""

    # Avoid showing resolution meta / JSON-ish blobs.
    noisy_tokens = ("{", "}", "[", "]", "resolution", "credit", "re-delivery", "redelivery", "meta:")
    low = note.lower()
    if any(t in low for t in noisy_tokens) and len(note) > 80:
        return ""

    return note.strip()


def _render_commentline(*, label: str, text: str) -> None:
    txt = (text or "").strip()
    if not txt:
        return
    st.markdown(
        f"""<div class='voi-commentline'><span class='lbl'>{html.escape(label)}</span><span class='txt'>{html.escape(txt)}</span></div>""",
        unsafe_allow_html=True,
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


# def _render_timeline(ctx: OrderContext, provider: str) -> None:
#     events = _load_timeline(int(ctx.order.id), provider)
#     if not events:
#         st.caption("No timeline yet.")
#         return
#     for e in events:
#         at = e.at.strftime("%Y-%m-%d %H:%M") if getattr(e, "at", None) else ""
#         note = _s(getattr(e, "note", None))
#         frm = _s(getattr(e, "from_state", None))
#         to = _s(getattr(e, "to_state", None))
#         who = f"{_s(getattr(e, 'actor_role', None))}:{_s(getattr(e, 'actor', None))}".strip(":")
#         st.markdown(f"- **{at}** · `{who}` · {frm} → {to}" + (f" · {note}" if note else ""))


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
    include_iva: bool = False,
    supplier_resolution_by_line_id: Optional[Dict[int, str]] = None,
) -> None:
    prov = norm_provider(provider)
    lines = ctx.lines_by_provider.get(prov, []) or []
    if not lines:
        st.info("No products.")
        return

    # ---------- Precompute summary + per-line data ----------
    total_lines = 0
    missing_lines = 0
    partial_lines = 0
    mismatch_lines = 0  # expected < ordered

    table_subtotal = 0.0
    table_iva = 0.0
    table_total = 0.0

    computed: dict[int, dict] = {}

    for ln in lines:
        lid = int(ln.id)
        ordered = _safe_float(getattr(ln, "quantity", 0.0), 0.0)

        fu = ctx.followups_by_key.get((prov, lid))
        stt = (_s(getattr(fu, "supplier_status", None))).lower() if fu else "unknown"
        sqty = getattr(fu, "supplier_qty", None) if fu else None

        # Supplier "expected" qty logic
        if stt == "missing":
            expected_qty = 0.0
            missing_lines += 1
        elif stt == "partial":
            expected_qty = _safe_float(sqty, 0.0)
            partial_lines += 1
        elif stt == "ok":
            expected_qty = _safe_float(sqty, ordered) if sqty is not None else ordered
        else:
            expected_qty = ordered

        total_lines += 1
        if expected_qty < ordered:
            mismatch_lines += 1

        # Pricing lookup (units, discount) — amounts will be computed later per row
        if show_prices:
            pid_raw = getattr(ln, "product_id", None)
            pid = int(pid_raw) if pid_raw not in (None, "", 0, "0") else None
            gross_unit = _price_for_pid(ctx.products_by_id, pid)

            if gross_unit > 0:
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
                iva_pct = _iva_pct_for_pid(ctx.products_by_id, pid, 21.0) if include_iva else 0.0

                computed[lid] = {
                    "ordered": float(ordered),
                    "expected_qty": float(expected_qty),
                    "stt": stt,
                    "gross_unit": float(gross_unit),
                    "net_unit": float(net_unit),
                    "disc_pct": float(disc_pct),
                    "iva_pct": float(iva_pct),
                }
            else:
                computed[lid] = {
                    "ordered": float(ordered),
                    "expected_qty": float(expected_qty),
                    "stt": stt,
                }
        else:
            computed[lid] = {
                "ordered": float(ordered),
                "expected_qty": float(expected_qty),
                "stt": stt,
            }

    # ---------- Summary header ----------
    summary_left = f"📦 {total_lines} items"
    if missing_lines or partial_lines or mismatch_lines:
        summary_left += f" · ❌ {missing_lines} missing · 🟡 {partial_lines} partial · ⚠️ {mismatch_lines} mismatch"

    st.markdown(
        f"<div class='voi-muted'><b>Supplier confirmation</b> · {summary_left}</div>",
        unsafe_allow_html=True,
    )

    # ---------- Render invoice-like list (many products) ----------
    rows_html = ""
    table_subtotal = 0.0
    table_iva = 0.0
    table_total = 0.0

    for ln in sorted(lines, key=lambda x: _line_name(x, ctx.products_by_id).lower()):
        lid = int(ln.id)
        name = _line_name(ln, ctx.products_by_id)
        unit = _line_unit(ln, ctx.products_by_id)
        desc = _line_desc(ln, ctx.products_by_id)

        c = computed.get(lid, {})
        ordered_qty = float(c.get("ordered", 0.0) or 0.0)
        expected_qty = float(c.get("expected_qty", 0.0) or 0.0)

        # returns: (issue_status, received_qty, issue_qty, in_invoice, reason)
        issue_status, venue_received_qty, issue_qty, in_invoice, reason = _get_received_info(ctx, prov, lid)

        # RECEIVED + REASON
        if issue_status == "ok":
            received_qty = expected_qty
            reason_cell = "—"
        else:
            received_qty = float(venue_received_qty or 0.0)

            # ✅ BUSINESS RULE:
            # missing + present in invoice => invoice discrepancy
            if issue_status == "missing" and in_invoice is True:
                reason_cell = "Invoice discrepancy"
            else:
                reason_cell = (
                    reason
                    if reason and reason != "—"
                    else (issue_status.upper() if issue_status not in ("unknown", "") else "—")
                )

        # ✅ SOLUTION column (supplier-only)
        if issue_status == "ok":
            solution_cell = "—"
        else:
            solution_cell = ""  # empty until supplier resolution exists

        if supplier_resolution_by_line_id:
            r = (supplier_resolution_by_line_id.get(lid) or "").strip().lower()
            if r == "credit_note":
                solution_cell = "Credit note"
            elif r in {"supplementary_delivery", "re_delivery", "re-delivery", "redelivery"}:
                solution_cell = "Re-delivery"
            elif r == "reject":
                solution_cell = "Rejected"
            elif r == "partial_delivery":
                solution_cell = "Partial delivery"

        # PRICING quantity
        price_qty = expected_qty

        # If supplier decided re-delivery, invoice shouldn’t be credited for missing qty
        if supplier_resolution_by_line_id:
            r = (supplier_resolution_by_line_id.get(lid) or "").strip().lower()
            if r in {"supplementary_delivery", "re_delivery", "re-delivery", "redelivery"}:
                price_qty = received_qty

        # fallback legacy rule
        elif issue_status == "missing" and (in_invoice is False):
            price_qty = received_qty

        # Cells
        net_unit_cell = "—"
        base_cell = "—"
        disc_cell = "—"
        iva_cell = "—"
        total_cell = "—"

        if show_prices and ("net_unit" in c) and price_qty >= 0:
            unit_net = float(c.get("net_unit", 0.0) or 0.0)
            gross_unit = float(c.get("gross_unit", unit_net) or unit_net)
            disc_pct = float(c.get("disc_pct", 0.0) or 0.0)
            iva_pct = float(c.get("iva_pct", 0.0) or 0.0) if include_iva else 0.0

            net_unit_cell = f"€{unit_net:,.2f}"

            subtotal = price_qty * unit_net
            iva_eur = subtotal * (iva_pct / 100.0) if include_iva else 0.0
            total = subtotal + iva_eur

            base_before_disc = price_qty * gross_unit
            discount_eur = max(0.0, base_before_disc - subtotal)

            base_cell = f"€{subtotal:,.2f}"
            if disc_pct > 0.0001 or discount_eur > 0.004:
                disc_cell = f"<span class='inv-neg'>-€{discount_eur:,.2f}</span>"
            else:
                disc_cell = "—"

            iva_cell = f"€{iva_eur:,.2f}" if include_iva else "—"
            total_cell = f"€{total:,.2f}" if include_iva else f"€{subtotal:,.2f}"

            table_subtotal += subtotal
            table_iva += iva_eur
            table_total += total

        desc_html = f"<div class='inv-desc'>{desc}</div>" if desc else ""
        not_in_invoice_badge = "" if (in_invoice is not False) else " <span class='inv-flag'>NOT IN INVOICE</span>"

        rows_html += (
            "<tr>"
            f"<td class='name'>{name}{not_in_invoice_badge}{desc_html}</td>"
            f"<td class='num'>{ordered_qty:g} {unit}</td>"
            f"<td class='num'>{expected_qty:g} {unit}</td>"
            f"<td class='num'>{received_qty:g} {unit}</td>"
            f"<td class='reason'>{reason_cell}</td>"
            f"<td class='reason'>{solution_cell}</td>"
            f"<td class='num'>{net_unit_cell}</td>"
            f"<td class='num'>{base_cell}</td>"
            f"<td class='num'>{disc_cell}</td>"
            f"<td class='num'>{iva_cell}</td>"
            f"<td class='num'><b>{total_cell}</b></td>"
            "</tr>"
        )

    # Footer totals
    footer_subtotal = f"€{table_subtotal:,.2f}" if show_prices else "—"
    footer_iva = f"€{table_iva:,.2f}" if (show_prices and include_iva) else "—"
    footer_total = (
        f"€{table_total:,.2f}" if (show_prices and include_iva)
        else (f"€{table_subtotal:,.2f}" if show_prices else "—")
    )

    # Invoice header
    receipt = (ctx.receipts_by_provider or {}).get(prov)
    invoice_no = _s(getattr(receipt, "invoice_number", None)).strip() if receipt else ""
    invoice_label = f"Invoice Nº {invoice_no}" if invoice_no else ""

    summary_right = ""
    if show_prices:
        if include_iva:
            summary_right = f"Est. total (IVA incl): {footer_total}"
        else:
            summary_right = f"Est. subtotal: {footer_subtotal}"

    invoice_html = (
        "<div class='inv-wrap'>"
        "<div class='inv-head'>"
        f"<div class='inv-head-left'>{invoice_label}</div>"
        f"<div class='inv-head-right'>{summary_right}</div>"
        "</div>"
        "<table class='inv-table'>"
        "<thead>"
        "<tr>"
        "<th>Product</th>"
        "<th class='num'>Ordered</th>"
        "<th class='num'>Expected</th>"
        "<th class='num'>Received</th>"
        "<th>Reason</th>"
        "<th>Solution</th>"
        "<th class='num'>Net €/unit</th>"
        "<th class='num'>Disc%</th>"
        "<th class='num'>Net</th>"
        "<th class='num'>IVA</th>"
        "<th class='num'>Total</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        f"{rows_html}"
        "</tbody>"
        "<tfoot>"
        "<tr>"
        "<td class='muted'>TOTAL</td>"
        "<td class='num'></td>"
        "<td class='num'></td>"
        "<td class='num'></td>"
        "<td></td>"
        "<td></td>"
        "<td></td>"  # Net €/unit column
        f"<td class='num'>{footer_subtotal}</td>"
        "<td class='num'></td>"
        f"<td class='num'>{footer_iva}</td>"
        f"<td class='num'>{footer_total}</td>"
        "</tr>"
        "</tfoot>"
        "</table>"
        "</div>"
    )

    st.markdown(invoice_html, unsafe_allow_html=True)


def _render_receive_form(ctx: OrderContext, provider: str) -> None:
    prov = norm_provider(provider)
    order = ctx.order
    lines = ctx.lines_by_provider.get(prov, []) or []
    if not lines:
        return

    st.markdown("<div class='voi-muted'><b>Venue received</b></div>", unsafe_allow_html=True)

    STATUS_OPTIONS_ALL = ["OK", "Missing", "Damaged", "Wrong item"]
    INVOICE_OPTIONS_ALL = ["In invoice", "Not in invoice"]

    # --- callback: sync dependent fields when Status changes (works ONLY outside st.form) ---
    def _on_status_change(status_key: str, issue_key: str, invoice_key: str, qty_expected_key: str) -> None:
        status_now = _s(st.session_state.get(status_key, "OK"))
        qty_expected = float(st.session_state.get(qty_expected_key, 0.0) or 0.0)

        if status_now == "OK":
            st.session_state[issue_key] = 0.0
            st.session_state[invoice_key] = "Not in invoice"

        elif status_now == "Missing":
            cur = _safe_float(st.session_state.get(issue_key, 0.0), 0.0)
            if cur <= 0:
                st.session_state[issue_key] = qty_expected
            inv = _s(st.session_state.get(invoice_key, "Not in invoice"))
            if inv not in INVOICE_OPTIONS_ALL:
                st.session_state[invoice_key] = "Not in invoice"

        elif status_now in {"Damaged", "Wrong item"}:
            cur = _safe_float(st.session_state.get(issue_key, 0.0), 0.0)
            if cur <= 0:
                st.session_state[issue_key] = qty_expected
            st.session_state[invoice_key] = "In invoice"

    for ln in sorted(lines, key=lambda x: _line_name(x, ctx.products_by_id).lower()):
        lid = int(ln.id)
        name = _line_name(ln, ctx.products_by_id)
        unit = _line_unit(ln, ctx.products_by_id)
        qty_ordered = _safe_float(getattr(ln, "quantity", 0.0), 0.0)

        fu = ctx.followups_by_key.get((prov, lid))

        # supplier confirmed "not sending"
        supplier_status = (_s(getattr(fu, "supplier_status", None))).lower() if fu else ""
        supplier_confirmed_missing = (supplier_status == "missing")

        qty_expected = _derive_expected_qty(ln, fu)

        base = f"recv_{int(order.id)}_{prov}_{lid}_"
        status_key = base + "status"
        issue_key = base + "issue_qty"
        invoice_key = base + "invoice"
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

            # Note: qty_invoiced is used as "issue qty" in your UI
            default_issue_qty = _safe_float(getattr(fu, "qty_invoiced", None), 0.0)

            inv_db = getattr(fu, "invoice_listed", None)
            if inv_db is True:
                default_invoice = "In invoice"
            elif inv_db is False:
                default_invoice = "Not in invoice"
            else:
                default_invoice = "Not in invoice"

        # ✅ Hard rule: if supplier confirmed missing -> only Missing + Not in invoice
        # and issue qty should represent the ordered qty missing.
        if supplier_confirmed_missing:
            default_status = "Missing"
            default_invoice = "Not in invoice"
            default_issue_qty = float(qty_ordered) if qty_ordered > 0 else 0.0

        # Initialize state
        if status_key not in st.session_state:
            st.session_state[status_key] = default_status
        if issue_key not in st.session_state:
            st.session_state[issue_key] = float(default_issue_qty)
        if invoice_key not in st.session_state:
            st.session_state[invoice_key] = default_invoice

        # ✅ expected max base for issue qty:
        # - normal: qty_expected
        # - supplier missing: qty_ordered (so we can record the missing qty properly)
        issue_max_base = float(qty_ordered) if supplier_confirmed_missing else float(qty_expected)

        # refresh expected qty for callback + clamping
        st.session_state[qty_expected_key] = float(issue_max_base)

        # Per-line allowed options
        STATUS_OPTIONS = ["Missing"] if supplier_confirmed_missing else STATUS_OPTIONS_ALL
        INVOICE_OPTIONS = ["Not in invoice"] if supplier_confirmed_missing else INVOICE_OPTIONS_ALL

        # Normalize old values (but respect locked mode)
        if st.session_state[status_key] not in STATUS_OPTIONS:
            st.session_state[status_key] = STATUS_OPTIONS[0]
        if st.session_state[invoice_key] not in INVOICE_OPTIONS:
            st.session_state[invoice_key] = INVOICE_OPTIONS[0]

        # ✅ Force locked values every render (robust against old session state)
        if supplier_confirmed_missing:
            st.session_state[status_key] = "Missing"
            st.session_state[invoice_key] = "Not in invoice"
            st.session_state[issue_key] = float(qty_ordered) if qty_ordered > 0 else 0.0

        status_now = _s(st.session_state.get(status_key, default_status))
        needs_issue_qty = status_now in {"Missing", "Damaged", "Wrong item"}
        needs_invoice = status_now == "Missing"

        c0, c1, c2, c3 = st.columns([2.4, 1.25, 1.1, 1.2], vertical_alignment="center")

        with c0:
            st.markdown(f"**{name}**")
            if supplier_confirmed_missing:
                st.caption("🚫 Supplier confirmed: **Not sending**")
            # st.caption(f"Expected: {qty_expected:g} {unit} · Ordered: {qty_ordered:g} {unit}")

        # Status selectbox
        with c1:
            st.selectbox(
                "Status",
                STATUS_OPTIONS,
                index=0 if supplier_confirmed_missing else STATUS_OPTIONS.index(st.session_state[status_key]),
                key=status_key,
                label_visibility="collapsed",
                disabled=supplier_confirmed_missing,  # ✅ lock if supplier said not sending
                on_change=None if supplier_confirmed_missing else _on_status_change,
                kwargs=None if supplier_confirmed_missing else dict(
                    status_key=status_key,
                    issue_key=issue_key,
                    invoice_key=invoice_key,
                    qty_expected_key=qty_expected_key,
                ),
            )

        # Re-read after widget
        status_now = _s(st.session_state.get(status_key, default_status))
        needs_issue_qty = status_now in {"Missing", "Damaged", "Wrong item"}
        needs_invoice = status_now == "Missing"

        # Enforce issue qty rules + bounds
        if supplier_confirmed_missing:
            # locked to ordered qty missing, no editing
            st.session_state[issue_key] = float(qty_ordered) if qty_ordered > 0 else 0.0
        else:
            if status_now == "OK":
                st.session_state[issue_key] = 0.0
            elif needs_issue_qty:
                cur = _safe_float(st.session_state.get(issue_key, 0.0), 0.0)
                if cur < 1:
                    st.session_state[issue_key] = 1.0
                elif cur > float(issue_max_base):
                    st.session_state[issue_key] = float(issue_max_base)

        with c2:
            st.number_input(
                "Issue qty",
                min_value=1.0 if needs_issue_qty else 0.0,
                max_value=float(issue_max_base) if needs_issue_qty else 0.0,
                step=1.0,
                disabled=(not needs_issue_qty) or supplier_confirmed_missing,
                key=issue_key,
                label_visibility="collapsed",
            )

        # Invoice selector rules:
        # - Only relevant for Missing
        # - Locked to "Not in invoice" when supplier confirmed missing
        if supplier_confirmed_missing:
            st.session_state[invoice_key] = "Not in invoice"
        else:
            if status_now == "OK":
                st.session_state[invoice_key] = "Not in invoice"
            elif status_now in {"Damaged", "Wrong item"}:
                st.session_state[invoice_key] = "In invoice"

        with c3:
            st.selectbox(
                "Invoice listed",
                INVOICE_OPTIONS,
                index=0 if supplier_confirmed_missing else INVOICE_OPTIONS.index(st.session_state[invoice_key]),
                disabled=(not needs_invoice) or supplier_confirmed_missing,  # ✅ lock
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
    """Persist the whole Receive form for a provider in **one DB transaction**.

    This is a major performance hot path on Streamlit Cloud.
    Previously we opened a new DB session/commit per line; now we:
    - Upsert ProviderReceipt (invoice number)
    - Bulk upsert ProviderLineFollowUp
    - Bulk upsert SeguimientoTicket
    - Update OrderWorkflow + timeline event
    ...all in a single session/commit.
    """
    prov = norm_provider(provider)
    order = ctx.order
    lines = ctx.lines_by_provider.get(prov, []) or []
    if not lines:
        return False, "No lines for provider"

    # Invoice number REQUIRED (from UI state)
    inv_key = f"recv_inv_{int(order.id)}_{prov}"
    invoice_number_ui = (st.session_state.get(inv_key) or "").strip()
    if not invoice_number_ui:
        return False, "Invoice number is required before saving."

    now = _now()

    # Pre-compute line ids once
    line_ids: List[int] = []
    for ln in lines:
        if getattr(ln, "id", None) is None:
            continue
        try:
            line_ids.append(int(ln.id))
        except Exception:
            continue

    if not line_ids:
        return False, "No valid lines for provider"

    any_invoice_discrepancy = False
    any_operational_missing = False

    with get_session() as s:
        # ------------------------------------------------------------
        # 1) Upsert ProviderReceipt invoice number (once)
        # ------------------------------------------------------------
        receipt = s.exec(
            select(ProviderReceipt).where(
                ProviderReceipt.order_id == int(order.id),
                ProviderReceipt.provider_name == prov,
            )
        ).first()

        if not receipt:
            receipt = ProviderReceipt(
                venue_id=int(order.venue_id),
                order_id=int(order.id),
                provider_name=prov,
                created_at=now,
            )

        receipt.invoice_number = invoice_number_ui
        receipt.updated_at = now
        receipt.updated_by = "venue"
        s.add(receipt)

        # ------------------------------------------------------------
        # 2) Bulk prefetch followups + tickets for this provider/order
        # ------------------------------------------------------------
        fu_rows = list(
            s.exec(
                select(ProviderLineFollowUp).where(
                    ProviderLineFollowUp.order_id == int(order.id),
                    ProviderLineFollowUp.provider_name == prov,
                    ProviderLineFollowUp.order_line_id.in_(line_ids),
                )
            ).all()
        )
        fu_by_line: Dict[int, ProviderLineFollowUp] = {
            int(getattr(r, "order_line_id")): r for r in fu_rows if getattr(r, "order_line_id", None) is not None
        }

        ticket_kinds = ["invoice_discrepancy", "operational_missing", "damaged", "wrong_item"]
        t_rows = list(
            s.exec(
                select(SeguimientoTicket).where(
                    SeguimientoTicket.order_id == int(order.id),
                    SeguimientoTicket.provider_name == prov,
                    SeguimientoTicket.order_line_id.in_(line_ids),
                    SeguimientoTicket.kind.in_(ticket_kinds),
                )
            ).all()
        )
        t_by_key: Dict[Tuple[int, str], SeguimientoTicket] = {}
        for tr in t_rows:
            lid = getattr(tr, "order_line_id", None)
            kind = _s(getattr(tr, "kind", None))
            if lid is None or not kind:
                continue
            t_by_key[(int(lid), kind)] = tr

        # ------------------------------------------------------------
        # 3) Apply Receive form state (no DB calls in the loop)
        # ------------------------------------------------------------
        status_map = {"OK": "ok", "Missing": "missing", "Damaged": "damaged", "Wrong item": "wrong_item"}

        for ln in lines:
            if getattr(ln, "id", None) is None:
                continue
            lid = int(ln.id)

            unit = _line_unit(ln, ctx.products_by_id)
            name = _line_name(ln, ctx.products_by_id)

            # Use ctx for expected qty derivation (fast/in-memory)
            fu_existing_ctx = ctx.followups_by_key.get((prov, lid))
            qty_expected = _derive_expected_qty(ln, fu_existing_ctx)
            qty_ordered = _safe_float(getattr(ln, "quantity", 0.0), 0.0)

            base = f"recv_{int(order.id)}_{prov}_{lid}_"
            status_ui = (st.session_state.get(base + "status") or "OK").strip()
            issue_qty = _safe_float(st.session_state.get(base + "issue_qty"), 0.0)
            inv_ui = st.session_state.get(base + "invoice")

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

            # -------------------------
            # FollowUp upsert (batched)
            # -------------------------
            fu = fu_by_line.get(lid)
            if not fu:
                fu = ProviderLineFollowUp(
                    venue_id=int(order.venue_id),
                    order_id=int(order.id),
                    provider_name=prov,
                    order_line_id=lid,
                    qty_ordered=_safe_float(getattr(ln, "quantity", 0.0), 0.0),
                )
                fu_by_line[lid] = fu

            # tag in venue_comment
            tag = f"[VENUE_STATUS={issue_status}]"
            base_comment = (fu.venue_comment or "").split("[VENUE_STATUS=")[0].strip()
            fu.venue_comment = (base_comment + " " + tag).strip()

            fu.venue_qty = float(venue_qty)
            fu.qty_invoiced = float(issue_qty)  # reuse field to store issue qty
            fu.invoice_listed = invoice_listed
            fu.updated_at = now
            fu.updated_by = "venue"
            s.add(fu)

            # -------------------------
            # Ticket upsert (batched)
            # -------------------------
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
                any_invoice_discrepancy = True
            elif issue_status == "wrong_item":
                kind = "wrong_item"
                any_invoice_discrepancy = True

            if kind:
                t = t_by_key.get((lid, kind))
                if not t:
                    initial_state = "open_internal" if kind == "operational_missing" else "open"
                    t = SeguimientoTicket(
                        venue_id=int(order.venue_id),
                        order_id=int(order.id),
                        provider_name=prov,
                        order_line_id=lid,
                        kind=kind,
                        state=initial_state,
                        created_at=now,
                    )
                    t_by_key[(lid, kind)] = t

                t.product_name = name
                t.unit = unit
                t.qty_invoiced = float(issue_qty)
                t.invoice_number = invoice_number_ui
                t.updated_at = now
                s.add(t)

        # ------------------------------------------------------------
        # 4) Update workflow state + add timeline event (same txn)
        # ------------------------------------------------------------
        if any_invoice_discrepancy:
            to_state = "INVOICE_DISCREPANCY"
            note = "Venue saved receiving: discrepancies detected."
        elif any_operational_missing:
            to_state = "OPERATIONAL_MISSING_PRODUCT"
            note = "Venue saved receiving: operational missing product."
        else:
            to_state = "RECEIVED"
            note = "Venue saved receiving: all OK."

        wf = s.exec(
            select(OrderWorkflow).where(
                OrderWorkflow.order_id == int(order.id),
                OrderWorkflow.provider_name == prov,
            )
        ).first()

        if not wf:
            wf = OrderWorkflow(
                venue_id=int(order.venue_id),
                order_id=int(order.id),
                provider_name=prov,
                state="ORDER_SENT",
                created_at=now,
            )

        prev = _s(wf.state)
        wf.state = _s(to_state)
        wf.updated_at = now
        wf.updated_by_role = "venue"
        wf.updated_by = "venue"
        wf.note = note

        s.add(wf)
        _add_event(s, int(order.venue_id), int(order.id), prov, prev, wf.state, "venue", "venue", note)

        # Single commit for everything
        s.commit()
    # Clear UI state + reset selectors
    _clear_receive_form_state(int(order.id), prov)
    idx_key = f"recv_current_provider_idx_{int(order.id)}"
    sel_key = f"recv_provider_sel_{int(order.id)}"
    st.session_state[idx_key] = 0
    if sel_key in st.session_state:
        del st.session_state[sel_key]

    return True, "Saved"

# =============================
# Supplier resolution + verify
# =============================

def request_supplier_resolution(venue_id: int, order_id: int, provider: str, venue_comment: str = "") -> Tuple[bool, str]:
    """Send email to supplier with tracking link + item list + venue footer."""
    prov = norm_provider(provider)

    templates = _load_venue_templates(int(venue_id), 0)
    lang = _s(getattr(templates, "email_lang", "en")) or "en"

    with get_session() as s:
        p = s.exec(select(Provider).where(Provider.venue_id == int(venue_id), Provider.name == prov)).first()
        if not p:
            p = s.exec(select(Provider).where(Provider.venue_id == int(venue_id), Provider.name == provider)).first()

        emails: List[str] = []
        if p and getattr(p, "order_email", None):
            emails = [x.strip() for x in (p.order_email or "").split("|") if x.strip()]
        if not emails and p and getattr(p, "emails", None):
            emails = [x.strip() for x in (p.emails or "").split("|") if x.strip()]
        if not emails:
            return False, "No supplier email configured."

        # Pull open supplier-action tickets for this provider/order
        tickets = list(
            s.exec(
                select(SeguimientoTicket).where(
                    SeguimientoTicket.order_id == int(order_id),
                    SeguimientoTicket.provider_name == prov,
                    SeguimientoTicket.kind.in_(["invoice_discrepancy", "damaged", "wrong_item"]),
                    SeguimientoTicket.state.in_(["open", "open_internal"]),
                )
            ).all()
        )

        # Map order_line_id -> Product description (best effort)
        line_ids = [int(t.order_line_id) for t in tickets if getattr(t, "order_line_id", None) is not None]
        lines_by_id: Dict[int, OrderLine] = {}
        if line_ids:
            rows = list(s.exec(select(OrderLine).where(OrderLine.id.in_(line_ids))).all())
            lines_by_id = {int(r.id): r for r in rows if getattr(r, "id", None) is not None}

        prod_ids = []
        for ln in lines_by_id.values():
            pid = getattr(ln, "product_id", None)
            if pid is not None:
                prod_ids.append(int(pid))

        products_by_id: Dict[int, Product] = {}
        if prod_ids:
            prows = list(
                s.exec(select(Product).where(Product.venue_id == int(venue_id), Product.id.in_(list(set(prod_ids))))).all()
            )
            products_by_id = {int(pp.id): pp for pp in prows if getattr(pp, "id", None) is not None}

    # Supplier link
    link = build_seguimiento_url(order_id=int(order_id), provider_name=prov, role=ROLE_SUPPLIER, page_path="seguimiento")

    # Build item rows (name — description — qty)
    items: List[Dict[str, Any]] = []
    invoice_refs: List[str] = []

    for t in tickets:
        name = _s(getattr(t, "product_name", "")) or "Product"
        unit = _s(getattr(t, "unit", "")) or "unit"
        qty = float(getattr(t, "qty_invoiced", 0.0) or 0.0)

        inv = _s(getattr(t, "invoice_number", ""))  # might be empty
        if inv:
            invoice_refs.append(inv)

        desc = ""
        lid = getattr(t, "order_line_id", None)
        if lid is not None:
            ln = lines_by_id.get(int(lid))
            pid = getattr(ln, "product_id", None) if ln else None
            prod = products_by_id.get(int(pid)) if pid is not None else None
            desc = _s(getattr(prod, "description", "")) if prod else ""

        items.append({"name": name, "description": desc, "qty": qty, "unit": unit})

    # Prefer a single invoice ref if all same; otherwise omit from subject and keep body list only
    invoice_ref = ""
    uniq = sorted({x for x in invoice_refs if x})
    if len(uniq) == 1:
        invoice_ref = uniq[0]

    subject, body = build_resolution_email_full(
        venue_ctx=templates,
        order_id=int(order_id),
        provider_name=prov,
        supplier_link=link,
        items=items,
        invoice_ref=invoice_ref,
        lang=lang,
    )

    # Optional venue message for supplier
    vc = _s(venue_comment)
    if vc:
        body = (body.rstrip() + "\n\nVenue message:\n" + vc + "\n").rstrip() + "\n"

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
        note=("Requested supplier decision via link." + (f" Venue message: {_s(venue_comment)}" if _s(venue_comment) else "")),
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

# def _derive_workflow_state_from_open_resolutions(open_rows: list[ProviderResolution]) -> str:
#     """
#     Keep your existing workflow states, but derive them from what's still OPEN.
#     Priority: supplementary > credit_note > reject
#     """
#     types = {(_s(r.resolution_type).strip().lower()) for r in (open_rows or [])}

#     if "supplementary_delivery" in types:
#         return "SUPPLEMENTARY_DELIVERY_SENT"

#     if "credit_note" in types:
#         # if any open credit_note has invoice, treat as ISSUED else PENDING
#         has_invoice = any(_s(getattr(r, "credit_note_invoice", None)).strip() for r in open_rows if _s(getattr(r, "resolution_type", "")).lower() == "credit_note")
#         return "SUPPLIER_CREDIT_NOTE_ISSUED" if has_invoice else "SUPPLIER_CREDIT_NOTE_PENDING"

#     if "reject" in types:
#         return "SUPPLIER_REJECTED"

#     return "CLOSED"


def venue_verify_and_close(*, ctx: OrderContext, provider: str, mode: str, credit_note_invoice: Optional[str] = None) -> str:
    """
    Close ONE resolution track (credit note OR re-delivery OR reject) for a provider,
    without forcing the provider workflow to CLOSED unless nothing else remains open.
    """
    prov = norm_provider(provider)
    mode = (mode or "").strip().lower()
    if mode not in {"credit_note", "supplementary", "reject"}:
        return "Invalid mode"

    # map UI modes -> stored types
    rtype = (
        "credit_note"
        if mode == "credit_note"
        else ("supplementary_delivery" if mode == "supplementary" else "reject")
    )

    order = ctx.order
    now = _now()
    actor = _s(st.session_state.get("user_email") or st.session_state.get("actor") or "venue")

    # Resolve discrepancy-related tickets BUT ONLY those matching this mode
    resolution_state = (
        "resolved_credit_note_verified"
        if mode == "credit_note"
        else ("resolved_supplementary_received" if mode == "supplementary" else "resolved_reject_accepted")
    )

    # Helper: which ticket resolution strings map to re-delivery
    _RD = {"supplementary_delivery", "re_delivery", "re-delivery", "redelivery"}

    with get_session() as s:
        # -----------------------
        # 1) Close relevant tickets
        # -----------------------
        tickets = list(
            s.exec(
                select(SeguimientoTicket).where(
                    SeguimientoTicket.order_id == int(order.id),
                    SeguimientoTicket.provider_name == prov,
                    SeguimientoTicket.kind.in_(
                        ["invoice_discrepancy", "damaged", "wrong_item", "operational_missing"]
                    ),
                )
            ).all()
        )

        for t in tickets:
            if _s(getattr(t, "state", "")).strip() not in {"open", "SUPPLIER_ACTION_DONE"}:
                continue

            meta = _parse_ticket_resolution_note(_s(getattr(t, "resolution_note", None)))
            t_res = _s(meta.get("resolution")).strip().lower()

            # match ticket resolution to what we're closing
            if rtype == "credit_note" and t_res != "credit_note":
                continue
            if rtype == "supplementary_delivery" and t_res not in _RD:
                continue
            if rtype == "reject" and t_res != "reject":
                continue

            t.state = resolution_state
            t.resolved_at = now

            # If this is a credit-note closure, persist the CN number into the *first line*
            # so previews + parsing can pick it up later (even if supplier didn't provide it).
            cn_in = _s(credit_note_invoice).strip() if mode == "credit_note" else ""
            if cn_in:
                note0 = _s(getattr(t, "resolution_note", None))
                lines = note0.splitlines() if note0 else []
                first = lines[0] if lines else ""
                # strip any leading tag like [SUPPLIER] / [VENUE]
                first_clean = re.sub(r"^\s*\[[A-Z_]+\]\s*", "", first).strip()
                if not first_clean:
                    first_clean = "credit_note"
                # ensure resolution is "credit_note" for this track
                if "credit_note" not in first_clean.lower():
                    first_clean = "credit_note"
                # upsert credit_note_invoice=...
                if re.search(r"(?i)credit_note_invoice\s*=", first_clean):
                    first_clean = re.sub(r"(?i)(credit_note_invoice\s*=\s*)([^|]+)", r"\1" + cn_in, first_clean)
                else:
                    first_clean = first_clean + " | " + f"credit_note_invoice={cn_in}"
                if lines:
                    lines[0] = first_clean
                else:
                    lines = [first_clean]
                t.resolution_note = "\n".join(lines)

            # Append audit line
            extra = f" Credit note: {cn_in}." if cn_in else ""
            t.resolution_note = (t.resolution_note or "") + f"\n[VENUE] Verified {mode}." + extra
            t.updated_at = now
            s.add(t)

        # -----------------------
        # 2) Close the ProviderResolution row (this type only)
        # -----------------------
        pr = s.exec(
            select(ProviderResolution).where(
                ProviderResolution.order_id == int(order.id),
                ProviderResolution.provider_name == prov,
                ProviderResolution.resolution_type == rtype,
            )
        ).first()

        # If it doesn't exist (older orders), create it then close it (safe fallback)
        if not pr:
            pr = ProviderResolution(
                venue_id=int(order.venue_id),
                order_id=int(order.id),
                provider_name=prov,
                resolution_type=rtype,
                status="open",
                opened_at=now,
                opened_by=actor,
                created_at=now,
                updated_at=now,
                updated_by=actor,
            )

        # -----------------------
        # 2b) Credit note number (venue can enter if supplier didn't)
        # -----------------------
        # -----------------------
        # 2b) Credit note number is required (venue can enter if supplier didn't)
        # -----------------------
        if mode == "credit_note":
            cn_in = _s(credit_note_invoice).strip()
            if not cn_in:
                return "Credit note number is required to close."
        pr.status = "closed"
        pr.closed_at = now
        pr.closed_by = actor
        pr.updated_at = now
        pr.updated_by = actor
        s.add(pr)

        # -----------------------
        # 3) Update ProviderReceipt (optional but useful)
        # -----------------------
        receipt = s.exec(
            select(ProviderReceipt).where(
                ProviderReceipt.order_id == int(order.id),
                ProviderReceipt.provider_name == prov,
            )
        ).first()

        if receipt:
            # keep your existing logic: once you verify something, consider provider "received"
            receipt.received = True
            receipt.received_at = now
            receipt.received_by = actor
            receipt.updated_at = now
            receipt.updated_by = actor
            s.add(receipt)
        # -----------------------
        # 4) Determine next workflow state based on REMAINING OPEN TICKETS
        # (not ProviderResolution rows, which may be missing/incomplete)
        # -----------------------
        remaining = list(
            s.exec(
                select(SeguimientoTicket).where(
                    SeguimientoTicket.order_id == int(order.id),
                    SeguimientoTicket.provider_name == prov,
                    SeguimientoTicket.kind.in_(
                        ["invoice_discrepancy", "damaged", "wrong_item", "operational_missing"]
                    ),
                )
            ).all()
        )

        remaining_types: set[str] = set()
        has_cn_invoice = False

        _RD = {"supplementary_delivery", "re_delivery", "re-delivery", "redelivery"}

        for t2 in remaining:
            if _s(getattr(t2, "state", "")).strip() not in {"open", "SUPPLIER_ACTION_DONE"}:
                continue

            meta2 = _parse_ticket_resolution_note(_s(getattr(t2, "resolution_note", None)))
            r2 = _s(meta2.get("resolution")).strip().lower()

            if r2 == "credit_note":
                remaining_types.add("credit_note")
                if _s(meta2.get("credit_note_invoice")).strip():
                    has_cn_invoice = True
            elif r2 in _RD:
                remaining_types.add("supplementary_delivery")
            elif r2 == "reject":
                remaining_types.add("reject")

        # Mirror priority: supplementary > credit_note > reject
        if "supplementary_delivery" in remaining_types:
            next_state = "SUPPLEMENTARY_DELIVERY_SENT"
        elif "credit_note" in remaining_types:
            next_state = "SUPPLIER_CREDIT_NOTE_ISSUED" if has_cn_invoice else "SUPPLIER_CREDIT_NOTE_PENDING"
        elif "reject" in remaining_types:
            next_state = "SUPPLIER_REJECTED"
        else:
            next_state = "CLOSED"

        # -----------------------
        # 5) If fully done, set overall closure markers on receipt
        # -----------------------
        if next_state == "CLOSED" and receipt:
            receipt.all_resolutions_closed_at = now
            receipt.all_resolutions_closed_by = actor
            receipt.updated_at = now
            receipt.updated_by = actor
            s.add(receipt)

        s.commit()

    # -----------------------
    # 6) Persist provider workflow state for Track Order UI
    # -----------------------
    _set_workflow_state(
        venue_id=int(order.venue_id),
        order_id=int(order.id),
        provider=prov,
        to_state=next_state,
        actor_role="venue",
        actor=actor,
        note=f"Venue verified {mode}. Remaining open resolutions: {next_state}.",
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


# def _get_or_create_draft_order(*, venue_id: int, actor: str = "venue") -> int:
#     """Return a draft order id for the venue (creates one if missing)."""
#     vid = int(venue_id)
#     with get_session() as s:
#         # best-effort: find latest draft
#         q = (
#             select(Order)
#             .where(Order.venue_id == vid)
#             .where((Order.status == "draft") | (Order.status == "borrador"))
#             .order_by(Order.created_at.desc())
#         )
#         o = s.exec(q).first()
#         if o and getattr(o, "id", None) is not None:
#             return int(o.id)

#         # create new draft
#         o = Order(
#             venue_id=vid,
#             status="draft",
#             title="Borrador",
#             created_at=_now(),
#             created_by=actor,
#         )
#         s.add(o)
#         s.commit()
#         s.refresh(o)
#         return int(o.id)


# def _create_new_draft_order(*, venue_id: int, actor: str, title: str) -> int:
#     vid = int(venue_id)
#     with get_session() as s:
#         o = Order(
#             venue_id=vid,
#             status="draft",
#             title=title or "Nuevo pedido",
#             created_at=_now(),
#             created_by=actor,
#         )
#         s.add(o)
#         s.commit()
#         s.refresh(o)
#         return int(o.id)


# def _add_line_to_order(
#     *, venue_id: int, order_id: int, product_id: int | None,
#     provider: str, name: str, qty: float, unit: str, actor: str = "venue",
#     chip: str = ""
# ) -> None:
#     with get_session() as s:
#         chip_txt = f" [{chip}]" if chip else ""
#         ln = OrderLine(
#             venue_id=int(venue_id),
#             order_id=int(order_id),
#             product_id=int(product_id) if product_id else None,
#             provider=_s(provider) or None,
#             spoken_name=(_s(name) + chip_txt).strip() or None,
#             quantity=float(qty or 0.0),
#             unit=_s(unit) or None,
#             updated_at=_now(),
#             updated_by=actor,
#         )
#         s.add(ln)
#         s.commit()



# def _search_similar_products(*, venue_id: int, query_name: str, limit: int = 50) -> list[Product]:
#     """Best-effort similarity search by name (simple contains tokens)."""
#     qn = (_s(query_name) or "").strip().lower()
#     if not qn:
#         return []
#     tokens = [t for t in re.split(r"\W+", qn) if len(t) >= 3][:4]
#     if not tokens:
#         tokens = [qn[:6]]
#     with get_session() as s:
#         # Start broad: venue products
#         ps = list(s.exec(select(Product).where(Product.venue_id == int(venue_id))).all())
#     def score(p: Product) -> float:
#         nm = (_s(getattr(p, 'name', ''))).lower()
#         if not nm:
#             return 0.0
#         hits = sum(1 for t in tokens if t in nm)
#         return hits / max(1, len(tokens))
#     ranked = [(score(p), p) for p in ps]
#     ranked = [rp for rp in ranked if rp[0] > 0]
#     ranked.sort(key=lambda x: x[0], reverse=True)
#     return [p for _, p in ranked[:limit]]


# =============================
# Incidences: open filters
# =============================


@st.cache_data(ttl=20, show_spinner=False)
def _count_open_tickets_db(
    venue_id: int,
    order_id: int,
    provider: str,
    *,
    refresh_token: int = 0,
) -> int:
    """Fast path: count open tickets without loading full OrderContext.

    Used after venue actions to decide whether a workflow can be auto-closed.
    We keep a short TTL and also include refresh_token for deterministic busting.
    """
    _ = int(refresh_token or 0)
    prov = norm_provider(provider)

    with get_session() as s:
        stmt = (
            select(func.count(SeguimientoTicket.id))
            .where(SeguimientoTicket.venue_id == int(venue_id))
            .where(SeguimientoTicket.order_id == int(order_id))
            .where(SeguimientoTicket.provider_name == prov)
            .where(SeguimientoTicket.resolved_at.is_(None))
            .where(~func.lower(SeguimientoTicket.state).like("resolved%"))
        )
        try:
            return int(s.exec(stmt).one() or 0)
        except Exception:
            # Fallback for engines that return tuples
            row = s.exec(stmt).first()
            if isinstance(row, (tuple, list)) and row:
                return int(row[0] or 0)
            return int(row or 0)

def _provider_open_tickets(ctx: OrderContext, provider: str) -> List[SeguimientoTicket]:
    prov = norm_provider(provider)

    # ✅ NEW: if workflow is CLOSED, don't show anything in Incidences
    wf = ctx.workflows_by_provider.get(prov)
    if wf and _s(getattr(wf, "state", "")).upper() == "CLOSED":
        return []

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

    m_tag = re.match(r'^\s*\[[A-Z_]+\]\s*(.*)$', txt)
    if m_tag:
        txt = (m_tag.group(1) or '').strip()

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
    
    
    # --- Fuse header card + expander (same UX as Receive) ---
    st.markdown(
        """
        <style>
        .voi-card.voi-card--header{
            margin-bottom: 0.35rem;
            border-bottom-left-radius: 0 !important;
            border-bottom-right-radius: 0 !important;
        }
        div[data-testid="stExpander"]{
            border: 1px solid rgba(49, 51, 63, 0.12);
            border-top: none;
            border-bottom-left-radius: 12px;
            border-bottom-right-radius: 12px;
            padding: 0.25rem 0.25rem 0.5rem 0.25rem;
            margin-top: -10px;
            background: #fff;
        }
        div[data-testid="stExpander"] summary{
            padding: 0.25rem 0.5rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    

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

      
            # VAT is 0 unless include_iva=True
            vat_pct = _iva_pct_for_pid(ctx.products_by_id, pid, 21.0) if include_iva else 0.0
            vat_eur = net_amount * (vat_pct / 100.0) if include_iva else 0.0
            total = net_amount + vat_eur  # net-only when include_iva=False

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
                    "VAT%": float(vat_pct),
                    "VAT €": float(vat_eur),
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
            rows_html = ""

            for r in rows:
                nm = html.escape(_s(r.get("Product", "")))
                qty = float(r.get("Qty", 0) or 0)
                unit = html.escape(_s(r.get("Unit", "")))
                reason = html.escape(_s(r.get("Reason", "")))

                netu = float(r.get("Net €/unit", 0.0) or 0.0)
                disc = float(r.get("Disc%", 0.0) or 0.0)
                net_amount = float(r.get("Net amount", 0.0) or 0.0)

                vat_eur = float(r.get("VAT €", 0.0) or 0.0)
                vat_cell = f"€{vat_eur:,.2f}" if include_iva else "—"

                total = float(r.get("Total", net_amount) or net_amount)

                rows_html += (
                    "<tr>"
                    f"<td class='name'>{nm}</td>"
                    f"<td class='num'>{qty:g} {unit}</td>"
                    f"<td class='reason'>{reason}</td>"
                    f"<td class='num'>€{netu:,.2f}</td>"
                    f"<td class='num'>{disc:g}%</td>"
                    f"<td class='num'>€{net_amount:,.2f}</td>"
                    f"<td class='num'>{vat_cell}</td>"  # ✅ ALWAYS show IVA column, but dash when disabled
                    f"<td class='num'><b>€{total:,.2f}</b></td>"
                    "</tr>"
                )

            footer_net = f"€{total_net:,.2f}"
            footer_vat = f"€{total_vat:,.2f}" if include_iva else "—"
            footer_total = f"€{(total_net + total_vat):,.2f}" if include_iva else footer_net

            receipt = ctx.receipts_by_provider.get(provn)
            inv_no = _s(getattr(receipt, "invoice_number", None)) or "—"
            inv_dt = _invoice_date_for_provider(provn)
            cn_no = _s(sol.get("credit_note_invoice")) or "—"

            head_left = (
                "Credit note (expected)"
                f" · Ref invoice Nº {html.escape(inv_no)}"
                f" · {html.escape(_fmt_dt(inv_dt))}"
            )
            head_right = f"Est. total: {footer_total}"

            cn_html = (
                "<div class='inv-wrap'>"
                "<div class='inv-head'>"
                f"<div class='inv-head-left'>{head_left}</div>"
                f"<div class='inv-head-right'>{head_right}</div>"
                "</div>"
                "<table class='inv-table'>"
                "<thead><tr>"
                "<th>Product</th>"
                "<th class='num'>Credited</th>"
                "<th>Reason</th>"
                "<th class='num'>Net €/unit</th>"
                "<th class='num'>Disc%</th>"
                "<th class='num'>Net</th>"
                "<th class='num'>IVA</th>"  # ✅ ALWAYS show IVA column
                "<th class='num'>Total</th>"
                "</tr></thead>"
                f"<tbody>{rows_html}</tbody>"
                "<tfoot><tr>"
                "<td class='muted'>TOTAL</td>"
                "<td class='num'></td>"
                "<td></td>"
                "<td class='num'></td>"
                "<td class='num'></td>"
                f"<td class='num'>{footer_net}</td>"
                f"<td class='num'>{footer_vat}</td>"
                f"<td class='num'>{footer_total}</td>"
                "</tr></tfoot>"
                "</table>"
                f"<div class='voi-muted' style='margin-top:.35rem'>Credit note number: <b>{html.escape(cn_no)}</b></div>"
                "</div>"
            )

            st.markdown(cn_html, unsafe_allow_html=True)


    def _redelivery_preview(
        ctx: OrderContext,
        prov: str,
        provn: str,
        open_t: List[SeguimientoTicket],
        sol: Dict[str, Any],
    ) -> None:
        """Render an *expected* re-delivery preview.

        Production behavior:
        - If supplier set different delivery windows per product (ETA per ticket), show them grouped by ETA.
        - Prefer ticket truth (resolution_note) when available.
        - Fall back to workflow note items if tickets don't contain resolution_note data.
        - Sort ETA groups by parsed date+time (robust).
        """
        sol = _normalize_solution_meta(sol or {})
        resolution = _s(sol.get("resolution", "")).strip().lower()

        # Determine whether we should show this block at all
        items = sol.get("items") or []
        has_rd_items = any(isinstance(it, dict) and _is_redelivery_item(it) for it in items)
        is_rd = resolution in {"supplementary_delivery", "re_delivery"}

        if not has_rd_items and not is_rd:
            any_ticket_rd = False
            for t in (open_t or []):
                meta = _parse_ticket_resolution_note(_s(getattr(t, "resolution_note", None)))
                r = _s(meta.get("resolution")).strip().lower()
                if r in {"supplementary_delivery", "re_delivery"}:
                    any_ticket_rd = True
                    break
            if not any_ticket_rd:
                return

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

        # -----------------------------
        # ETA parsing for robust sorting
        # -----------------------------
        def _parse_eta_sort_key(eta_raw: str):
            """
            Returns a tuple suitable for sorting:
            (bucket_rank, dt, start_minutes, original)
            bucket_rank:
            0 => parsed successfully
            1 => date parsed but no time
            9 => unknown ('—' or empty) or unparseable (goes last)
            """
            import re
            from datetime import datetime

            eta = (eta_raw or "").strip()
            if not eta or eta == "—":
                return (9, datetime.max, 24 * 60 + 1, eta_raw)

            # Normalize separators
            s = eta.replace("T", " ").strip()

            # Try ISO date first: YYYY-MM-DD ...
            m_iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
            dt = None

            if m_iso:
                y, mo, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
                try:
                    dt = datetime(y, mo, d)
                except Exception:
                    dt = None
            else:
                # Try EU date: DD/MM/YYYY ...
                m_eu = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
                if m_eu:
                    d, mo, y = int(m_eu.group(1)), int(m_eu.group(2)), int(m_eu.group(3))
                    try:
                        dt = datetime(y, mo, d)
                    except Exception:
                        dt = None

            if not dt:
                return (9, datetime.max, 24 * 60 + 1, eta_raw)

            # Extract start time from patterns like "08:00-14:00" or "8:00 - 14:00"
            m_time = re.search(r"\b(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\b", s)
            if m_time:
                hh, mm = int(m_time.group(1)), int(m_time.group(2))
                if 0 <= hh <= 23 and 0 <= mm <= 59:
                    return (0, dt, hh * 60 + mm, eta_raw)

            # If there is a single time (rare): "2026-01-29 08:00"
            m_single = re.search(r"\b(\d{1,2}):(\d{2})\b", s)
            if m_single:
                hh, mm = int(m_single.group(1)), int(m_single.group(2))
                if 0 <= hh <= 23 and 0 <= mm <= 59:
                    return (0, dt, hh * 60 + mm, eta_raw)

            # Date known, time unknown
            return (1, dt, 24 * 60, eta_raw)

        # -----------------------------
        # Case A: workflow items exist (legacy / fallback)
        # -----------------------------
        redel_items: List[Dict[str, Any]] = []
        if items:
            redel_items = [it for it in items if isinstance(it, dict) and _is_redelivery_item(it)]

        # -----------------------------
        # Case B: ticket truth (preferred)
        # Group tickets by ETA (per-product schedule)
        # -----------------------------
        by_eta: Dict[str, Dict[str, Any]] = {}  # eta -> {"tickets":[...], "invoice":"..."}
        if not redel_items:
            for t in (open_t or []):
                meta = _parse_ticket_resolution_note(_s(getattr(t, "resolution_note", None)))
                r = _s(meta.get("resolution")).strip().lower()

                if r not in {"supplementary_delivery", "re_delivery"}:
                    continue

                eta = _s(meta.get("eta")).strip() or "—"
                inv = _s(meta.get("invoice")).strip() or ""

                if eta not in by_eta:
                    by_eta[eta] = {"tickets": [], "invoice": inv}
                else:
                    if not by_eta[eta].get("invoice") and inv:
                        by_eta[eta]["invoice"] = inv

                by_eta[eta]["tickets"].append(t)

        # Header ETA summary
        header_eta = ""
        if redel_items:
            header_eta = _s(sol.get("eta")).strip()
        else:
            if len(by_eta) == 1:
                header_eta = next(iter(by_eta.keys()))
                if header_eta == "—":
                    header_eta = ""
            elif len(by_eta) > 1:
                header_eta = "Multiple deliveries"

        # Render header card
        st.markdown(
            "<div class='voi-card' style='border-color:#bfdbfe;background:#eff6ff'>"
            "<div class='voi-title'>🚚 Expected re-delivery (preview)</div>"
            f"<div class='voi-muted'>Supplier: <b>{html.escape(prov)}</b></div>"
            + (f"<div class='voi-muted'>Expected: <b>{html.escape(header_eta)}</b></div>" if header_eta else "")
            + "</div>",
            unsafe_allow_html=True,
        )

        # Supplier details
        if emails or phone or address:
            st.markdown("**Supplier details**")
            if emails:
                st.markdown(
                    f"- **Email:** `{emails[0]}`" + (f" (+{len(emails)-1} more)" if len(emails) > 1 else "")
                )
            if phone:
                st.markdown(f"- **Phone:** `{phone}`")
            if address:
                st.markdown(f"- **Address:** `{address}`")

        # -----------------------------
        # Render content
        # -----------------------------
        if redel_items:
            # Legacy workflow-based list (single window)
            eta = _s(sol.get("eta")).strip()
            inv_ref = _s(sol.get("invoice")).strip()

            if eta or inv_ref:
                st.markdown("**Delivery window:**")
                if eta:
                    st.markdown(f"- **Expected:** {eta}")
                if inv_ref:
                    st.markdown(f"- **Reference invoice:** {inv_ref}")

            st.markdown("**Products/quantities being re-delivered:**")
            for it in redel_items:
                nm = _s(it.get("name"))
                qty = _s(it.get("qty"))
                unit = _s(it.get("unit"))
                why = (_s(it.get("reason"))).replace("_", " ")
                st.markdown(f"- **{nm}** · {qty} {unit} · {why}")
            return

        # Ticket-based grouped schedule
        st.markdown("**Re-delivery plan:**")

        # ✅ Robust sort: parse date + start time; unknown goes last
        eta_keys_sorted = sorted(by_eta.keys(), key=_parse_eta_sort_key)

        for eta in eta_keys_sorted:
            bucket = by_eta.get(eta) or {}
            inv_ref = _s(bucket.get("invoice") or "").strip()
            tickets_for_eta: List[SeguimientoTicket] = bucket.get("tickets") or []

            if eta and eta != "—":
                st.markdown(f"### 🚚 Delivery window: **{eta}**")
            else:
                st.markdown("### 🚚 Delivery window: **—**")

            if inv_ref:
                st.caption(f"Reference invoice: {inv_ref}")

            st.markdown("**Products/quantities being re-delivered:**")
            for t in tickets_for_eta:
                lid = int(getattr(t, "order_line_id", 0) or 0)

                # best-effort line lookup
                ln = None
                try:
                    ln = next(
                        (
                            x
                            for x in (ctx.lines_by_provider.get(provn, []) or [])
                            if int(getattr(x, "id", 0) or 0) == lid
                        ),
                        None,
                    )
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
        # ✅ do not render incidences for CLOSED workflows
        if _provider_closed(ctx, provn):
            continue
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

        # --- header meta (same as Receive) ---
        from collections import Counter


        # ---------------------------------------------------------
        # Header meta (same as Receive) + Sent + Supplier answered
        # ---------------------------------------------------------
        oid = int(getattr(order, "id", 0) or 0)

        created_at = getattr(order, "created_at", None)
        created_txt = created_at.strftime("%d %b %Y, %H:%M") if created_at else "—"

        creator_raw = getattr(order, "created_by", None) or getattr(order, "updated_by", None) or ""
        creator_raw = (creator_raw or "").strip()
        if creator_raw and "@" in creator_raw:
            short = creator_raw.split("@", 1)[0].replace(".", " ").replace("_", " ").strip()
            creator_txt = short.title() if short else creator_raw
        else:
            creator_txt = creator_raw or "Unknown user"

        receipt = ctx.receipts_by_provider.get(provn)
        inv_no = _s(getattr(receipt, "invoice_number", None)) or "—"

        # ---------------------------------------------------------
        # 📤 Sent to supplier (WAITING_SUPPLIER_ACTION)
        # ---------------------------------------------------------
        wf = (getattr(ctx, "workflows_by_provider", {}) or {}).get(provn)
        sent_part = ""
        if (state or "").upper() == "WAITING_SUPPLIER_ACTION":
            sent_at = getattr(wf, "updated_at", None) if wf else None
            if sent_at:
                sent_part = (
                    " &nbsp;·&nbsp; 📤 Sent: "
                    + html.escape(sent_at.strftime("%d %b %Y, %H:%M"))
                )

        # ---------------------------------------------------------
        # ✅ Supplier answered summary
        # ---------------------------------------------------------
        answered_part = ""
        sol_counts = Counter()
        answered_at = None

        tickets = list(ctx.tickets_by_provider.get(provn, []) or [])
        supplier_done = []

        for t in tickets:
            if (_s(getattr(t, "state", "")).lower() == "supplier_action_done"):
                supplier_done.append(t)

                note = _s(getattr(t, "resolution_note", "")).lower()
                if "supplementary_delivery" in note:
                    sol_counts["supplementary_delivery"] += 1
                elif "credit_note_pending" in note:
                    sol_counts["credit_note_pending"] += 1
                elif "credit_note" in note:
                    sol_counts["credit_note"] += 1
                elif "reject" in note:
                    sol_counts["reject"] += 1

        if supplier_done:
            answered_at = max(
                dt for dt in (getattr(t, "updated_at", None) for t in supplier_done) if dt
            )

            chips = []
            if sol_counts["credit_note"]:
                chips.append(f"🧾 {sol_counts['credit_note']}")
            if sol_counts["credit_note_pending"]:
                chips.append(f"⏳ {sol_counts['credit_note_pending']}")
            if sol_counts["supplementary_delivery"]:
                chips.append(f"🚚 {sol_counts['supplementary_delivery']}")
            if sol_counts["reject"]:
                chips.append(f"⛔ {sol_counts['reject']}")

            answered_part = (
                " &nbsp;·&nbsp; ✅ Answered: "
                + html.escape(answered_at.strftime("%d %b %Y, %H:%M"))
            )
            if chips:
                answered_part += " &nbsp;·&nbsp; " + " &nbsp;·&nbsp; ".join(
                    html.escape(c) for c in chips
                )

        # ---------------------------------------------------------
        # Header HTML
        # ---------------------------------------------------------
        header_html = (
            "<div class='voi-card voi-card--header' style='background:#f7f9fc; border-left:4px solid #ff4d4d;'>"
            "<div style='display:flex; justify-content:space-between; align-items:center; gap:8px;'>"
            "<div style='min-width:0'>"
            f"<div class='voi-title' style='margin:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{html.escape(prov)}</div>"
            "<div class='voi-muted' style='white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>"
            f"🧾 {html.escape(inv_no)}"
            f"&nbsp;·&nbsp; 🗓️ {html.escape(created_txt)}"
            f"&nbsp;·&nbsp; 👤 {html.escape(creator_txt)}"
            f"{sent_part}{answered_part}"
            "</div>"
            "</div>"
            "<div class='voi-badges' style='display:flex; gap:6px; flex-wrap:wrap; justify-content:flex-end;'>"
            f"{_badge(badge_txt, badge_kind)}"
            "<span class='voi-badge bad'>Incidences</span>"
            "</div>"
            "</div>"
            "</div>"
        )

        st.markdown(header_html, unsafe_allow_html=True)




        # ✅ Supplier comment (from supplier confirmation)
        _render_commentline(label="💬 Supplier message", text=_supplier_comment_for_provider(ctx, provn))

        expanded = bool(st.session_state.get("inc_global_expand_all", False))
        with st.expander("Details", expanded=expanded):

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
            sol_wf = _normalize_solution_meta(
                _parse_supplier_solution_meta(_s(getattr(wf, "note", None)) if wf else "")
            )

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

            sol = sol_wf  # keep `sol` name for downstream chips/matching code

            ticket_meta_by_id: Dict[int, Dict[str, Any]] = {}
            credit_t: List[SeguimientoTicket] = []
            redel_t: List[SeguimientoTicket] = []
            undecided_t: List[SeguimientoTicket] = []

            # ✅ NEW: per-line single source of truth for Solution column in tables
            resolution_by_line_id: Dict[int, str] = {}

            # --- Classify tickets + build per-line resolution map ---
            for t in open_t:
                tid_local = int(getattr(t, "id", 0) or 0)
                meta = _parse_ticket_resolution_note(_s(getattr(t, "resolution_note", None)))
                ticket_meta_by_id[tid_local] = meta

                r = (_s(meta.get("resolution"))).strip().lower()

                # ✅ fill map: order_line_id -> resolution
                lid = int(getattr(t, "order_line_id", 0) or 0)
                if lid and r:
                    resolution_by_line_id[lid] = r

                # buckets for UI
                if r == "credit_note":
                    credit_t.append(t)
                elif r in {"supplementary_delivery", "re_delivery", "re-delivery", "redelivery"}:
                    redel_t.append(t)
                else:
                    undecided_t.append(t)

            # # ✅ Production lock rule:
            # ✅ Production lock rule:
            # lock if workflow indicates supplier acted OR workflow resolution exists OR any ticket has a resolution
            supplier_acted = (state_u in supplier_acted_states) or (
                wf_res in {"credit_note", "supplementary_delivery", "re_delivery"}
            )
            if (not supplier_acted) and any(resolution_by_line_id.values()):
                supplier_acted = True

            if supplier_acted:
                st.session_state[decisions_key] = True

            # ---------------------------------------------------------
            # ✅ NEW: decide tabs based on BOTH ticket truth + workflow truth
            # (this fixes urgent orders missing tabs/buttons)
            # ---------------------------------------------------------
            res = (sol_wf.get("resolution") or "").strip().lower()
            wf_items = sol_wf.get("items") or []
            has_rd_items = any(isinstance(it, dict) and _is_redelivery_item(it) for it in wf_items)
            has_cn_items = any(isinstance(it, dict) and (not _is_redelivery_item(it)) for it in wf_items)

            # ticket-based buckets (your existing)
            has_credit_pending = len(credit_t) > 0
            has_redel_pending = len(redel_t) > 0

            # ✅ expand using workflow/meta truth too
            has_credit_pending = has_credit_pending or (res == "credit_note") or has_cn_items
            has_redel_pending  = has_redel_pending  or (res in {"supplementary_delivery", "re_delivery"}) or has_rd_items

            # ✅ Always show invoice tab when we have open tickets (incidence context exists)
            show_tabs = bool(open_t)

            if show_tabs:
                labels = ["Invoice / expected lines"]
                if has_credit_pending:
                    labels.append("🧾 Expected credit note")
                if has_redel_pending:
                    labels.append("🚚 Expected re-delivery")

                tabs = st.tabs(labels)

                # --- Tab 0: invoice / expected lines ---
                with tabs[0]:
                    _render_expected_lines(
                        ctx,
                        prov,
                        show_prices=True,
                        include_iva=True,
                        supplier_resolution_by_line_id=resolution_by_line_id,
                    )

                tab_idx = 1

                # --- Credit note tab (now shown also when workflow says CN/items exist) ---
                if has_credit_pending:
                    with tabs[tab_idx]:
                        cn_no = ""

                        # 1) Prefer ticket meta CN number (if present)
                        for t in credit_t:
                            meta = ticket_meta_by_id.get(int(getattr(t, "id", 0) or 0)) or {}
                            if _s(meta.get("credit_note_invoice")):
                                cn_no = _s(meta.get("credit_note_invoice"))
                                break

                        # 2) Fallback: workflow meta CN number (if supplier put it there)
                        if not cn_no:
                            cn_no = _s(sol_wf.get("credit_note_invoice") or sol_wf.get("credit_note_number") or "")

                        cn_input_key = f"inc_cn_no_{order.id}_{provn}"
                        cn_val = st.text_input(
                            "Credit note number",
                            value=cn_no,
                            placeholder="e.g. CN-123 / ΠΙΣ-45",
                            key=cn_input_key,
                        )

                        sol_cn = dict(sol_wf)
                        sol_cn["resolution"] = "credit_note"
                        if _s(cn_val).strip():
                            sol_cn["credit_note_invoice"] = _s(cn_val).strip()

                        # ✅ IMPORTANT: if we don't have credit_t (urgent edge case),
                        # pass open_t so preview can still use workflow items/kinds
                        _credit_note_preview(
                            prov,
                            provn,
                            credit_t if len(credit_t) > 0 else open_t,
                            sol_cn,
                        )

                        st.divider()

                        c1, c2 = st.columns([1.2, 1.0], vertical_alignment="center")
                        with c1:
                            st.caption("Required to close (supplier may leave it blank; venue fills it here).")
                        with c2:
                            if st.button(
                                "✅ Verify credit note & close",
                                use_container_width=True,
                                disabled=(not _s(cn_val).strip()),
                                key=f"inc_verify_cn_{order.id}_{provn}",
                            ):
                                res_close = venue_verify_and_close(
                                    ctx=ctx,
                                    provider=provn,
                                    mode="credit_note",
                                    credit_note_invoice=_s(cn_val).strip(),
                                )
                                if res_close == "ok":
                                    st.success("Credit note closed")
                                    st.rerun()
                                else:
                                    st.error(res_close)

                    tab_idx += 1

                # --- Re-delivery tab (now shown also when workflow says RD/items exist) ---
                if has_redel_pending:
                    with tabs[tab_idx]:
                        sol_rd = dict(sol_wf)
                        sol_rd["resolution"] = "re_delivery"

                        # ✅ IMPORTANT: if we don't have redel_t (urgent edge case),
                        # pass open_t so preview can still use workflow items/resolution_note
                        _redelivery_preview(
                            ctx,
                            prov,
                            provn,
                            redel_t if len(redel_t) > 0 else open_t,
                            sol_rd,
                        )

                        st.divider()

                        if st.button(
                            "✅ Verify delivery & close",
                            use_container_width=True,
                            key=f"inc_verify_rd_{order.id}_{provn}",
                        ):
                            res_close = venue_verify_and_close(ctx=ctx, provider=provn, mode="supplementary")
                            if res_close == "ok":
                                st.success("Re-delivery closed")
                                st.rerun()
                            else:
                                st.error(res_close)

            else:
                # This should rarely happen now, but keep safe fallback
                _credit_note_preview(prov, provn, open_t, sol_wf)
                _redelivery_preview(ctx, prov, provn, open_t, sol_wf)


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
                changed = False
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
                        changed = True
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
                        changed = True

                        # ✅ NEW: if it’s operational_missing and we created the urgent request,
                        # close the incidence so it disappears from "Open incidences"
                        if kind == "operational_missing":
                            _close_ticket(
                                ticket_id=tid,
                                new_state="urgent_requested",
                                note="Urgent request created; moved to Urgent tab.",
                                actor=_s(st.session_state.get("user_email") or st.session_state.get("actor") or "venue"),
                            )
                            changed = True

                    except Exception:
                        pass

                    st.session_state[done_key] = True

                if changed:
                    _bump_orders_refresh_token(int(order.venue_id))

            # ---------------------------------------------------------
            # action panel  ✅ (THIS is what you were missing)
            # # ---------------------------------------------------------
   
            a1, a2, a3 = st.columns([1.05, 1.05, 1.2], vertical_alignment="center")

            supplier_link = build_seguimiento_url(
                order_id=int(order.id),
                provider_name=provn,
                role=ROLE_SUPPLIER,
                page_path="seguimiento",
            )

            state_u = (state or "").upper()

            # States where venue is still working / can request supplier decision
            requestable_states = {
                "INVOICE_DISCREPANCY",
                "OPERATIONAL_MISSING_PRODUCT",
                "WAITING_SUPPLIER_ACTION",
                "SUPPLIER_CREDIT_NOTE_PENDING",
            }

            # If supplier already acted (or we already requested decision), lock the UI
            locked = bool(st.session_state.get(decisions_key, False))


            # -----------------------------
            # State-specific actions
            # -----------------------------
            if state_u == "SUPPLIER_REJECTED":
                st.warning("Supplier rejected the claim (typically used for *Wrong item* / *Damaged* disputes).")
                if a1.button("✅ Accept reject & close", use_container_width=True, key=f"inc_rej_{order.id}_{provn}"):
                    res = venue_verify_and_close(ctx=ctx, provider=provn, mode="reject")
                    if res == "ok":
                        _bump_orders_refresh_token(int(order.venue_id))
                        st.success("Closed")
                        st.rerun()
                    else:
                        st.error(res)

            elif state_u in {"INVOICE_DISCREPANCY", "OPERATIONAL_MISSING_PRODUCT", "WAITING_SUPPLIER_ACTION"}:
                # ✅ Show the Save button ONLY if not locked
                # This prevents the button persisting after request.
                if not locked:
                    venue_msg = st.text_area(
                        "Message to supplier (optional)",
                        value="",
                        placeholder="e.g. Please confirm ETA / credit note number. Any substitution acceptable?",
                        key=f"venue_msg_{order.id}_{provn}",
                        height=30,
                    )

                    if a1.button("💾 Save & request decision", use_container_width=True, key=f"inc_req_{order.id}_{provn}"):
                        # ✅ Lock immediately so next rerun hides the button (even if email is slow/edge cases)
                        st.session_state[decisions_key] = True

                        _apply_inline_reorders_for_provider(close_non_urgent_op_missing=True)

                        # OP missing has the extra auto-close logic
                        if state_u == "OPERATIONAL_MISSING_PRODUCT":
                            open_cnt = _count_open_tickets_db(
                                int(order.venue_id),
                                int(order.id),
                                provn,
                                refresh_token=_orders_refresh_token(int(order.venue_id)),
                            )
                            if int(open_cnt) == 0:
                                _set_workflow_state(
                                    venue_id=int(order.venue_id),
                                    order_id=int(order.id),
                                    provider=provn,
                                    to_state="CLOSED",
                                    actor_role="venue",
                                    actor="venue",
                                    note="Operational missing: non-urgent items ignored (not reordered).",
                                )
                                _bump_orders_refresh_token(int(order.venue_id))
                                st.success("Saved ✓")
                                st.rerun()

                        # Default path: request supplier resolution link
                        ok, msg = request_supplier_resolution(int(order.venue_id), int(order.id), provn, venue_comment=venue_msg)
                        if ok:
                            _bump_orders_refresh_token(int(order.venue_id))
                            st.success("Link sent")
                            st.link_button("Open supplier link", msg, use_container_width=True)
                            st.rerun()
                        else:
                            # If sending failed, unlock so user can try again
                            st.session_state[decisions_key] = False
                            st.error(msg)



            elif state_u == "SUPPLIER_CREDIT_NOTE_PENDING":
                st.warning("⏳ Supplier chose credit note, but the credit note number is still missing.")
                st.caption("Ask the supplier to open the link again and add the credit note number.")


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
            # else:
            #     st.success("✔ Decisions requested.")            

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

    orders = _get_active_orders(int(venue_id), refresh_token=_orders_refresh_token(int(venue_id)))
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
    
    st.markdown(
        """
        <style>
        /* Fuse header card + expander */
        .voi-card.voi-card--header{
            margin-bottom: 0.35rem;
            border-bottom-left-radius: 0 !important;
            border-bottom-right-radius: 0 !important;
        }

        /* Style the expander container to look like the same card */
        div[data-testid="stExpander"]{
            border: 1px solid rgba(49, 51, 63, 0.12);
            border-top: none;
            border-bottom-left-radius: 12px;
            border-bottom-right-radius: 12px;
            padding: 0.25rem 0.25rem 0.5rem 0.25rem;
            margin-top: -10px; /* pulls it up under the card */
            background: #fff;
        }

        /* Make expander header more compact */
        div[data-testid="stExpander"] summary{
            padding: 0.25rem 0.5rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    
    current_provider = provider
    prov_key = norm_provider(current_provider)

    wf = ctx.workflows_by_provider.get(prov_key)
    state = _s(getattr(wf, "state", None)) if wf else "ORDER_SENT"
    badge_txt, badge_kind = _state_badge(state)

    receipt = ctx.receipts_by_provider.get(prov_key)

    inv_key = f"recv_inv_{int(ctx.order.id)}_{prov_key}"
    db_inv = _s(getattr(receipt, "invoice_number", None))

    # Session-state sync (only when empty)
    if inv_key not in st.session_state:
        st.session_state[inv_key] = db_inv
    else:
        if (not (st.session_state[inv_key] or "").strip()) and db_inv:
            st.session_state[inv_key] = db_inv

    invoice_locked = (state or "").upper() == "CLOSED"

    inv_badge_txt, inv_badge_kind = _invoice_set_badge(receipt)
    if invoice_locked:
        inv_badge_txt, inv_badge_kind = "Verified (locked)", "ok"

    # --- Header metadata (invoice #, created_at, created_by) ---
    order = getattr(ctx, "order", None)
    created_at = getattr(order, "created_at", None)
    created_txt = created_at.strftime("%d %b %Y, %H:%M") if created_at else "—"

    creator_raw = getattr(order, "created_by", None) or getattr(order, "updated_by", None) or ""
    creator_raw = (creator_raw or "").strip()
    if creator_raw and "@" in creator_raw:
        short = creator_raw.split("@", 1)[0].replace(".", " ").replace("_", " ").strip()
        creator_txt = short.title() if short else creator_raw
    else:
        creator_txt = creator_raw or "Unknown user"

    # Prefer what user typed (session) > DB value > dash
    header_inv = (st.session_state.get(inv_key) or db_inv or "—").strip() or "—"

    # ---------- Compact header (now with meta line) ----------
    # ---------- Compact header (with meta line) ----------
    st.markdown(
        f"""
        <div class="voi-card voi-card--header"
            style="background:#f7f9fc; border-left:4px solid #4c78ff;">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:8px;">
            <div style="min-width:0">
            <div class="voi-title"
                style="margin:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                {current_provider}
            </div>
            <div class="voi-muted" style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                🧾 {header_inv} &nbsp;·&nbsp; 🗓️ {created_txt} &nbsp;·&nbsp; 👤 {creator_txt}
            </div>
            </div>
            <div style="display:flex; gap:6px; flex-wrap:wrap; justify-content:flex-end;">
            {_badge(badge_txt, badge_kind)}
            {_badge(inv_badge_txt, inv_badge_kind)}
            </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # ✅ Supplier comment (from supplier confirmation)
    _render_commentline(label="💬 Supplier message", text=_supplier_comment_for_provider(ctx, prov_key))

    with st.expander("Details"):
        # ---------- Work area ----------
        _render_expected_lines(
            ctx,
            current_provider,
            show_prices=True,
            include_iva=True,
        )
        _render_receive_form(ctx, current_provider)
                    


        # ---------- Action row (invoice + expected popover + save) ----------
        c1, c2 = st.columns(2, vertical_alignment="center")

        # Use a form so typing in inputs does not rerun the whole script on every keystroke.
        # This makes saving feel dramatically faster on Streamlit Cloud.
        with st.form(key=f"form_receive_{int(ctx.order.id)}_{prov_key}", clear_on_submit=False):
            with c1:
                inv_val = st.text_input(
                    "Invoice #",
                    key=inv_key,
                    placeholder="Invoice # (required)",
                    disabled=invoice_locked,
                    label_visibility="collapsed",
                )

                inv_required_missing = (not invoice_locked) and (not (inv_val or "").strip())

            with c2:
                save_all = st.form_submit_button(
                    "💾 Save",
                    type="primary",
                    use_container_width=True,
                    disabled=invoice_locked or inv_required_missing,
                )

            if save_all:
                ok, msg = save_all_received_for_provider(ctx=ctx, provider=current_provider)
                if ok:
                    # Bump refresh token so cached reads update without forcing an expensive full page rerun.
                    _bump_orders_refresh_token(int(ctx.order.venue_id))
                    st.success("Saved ✓")
                else:
                    st.error(msg)

        st.markdown("<div class='voi-hr'></div>", unsafe_allow_html=True)

@st.cache_data(show_spinner=False, ttl=15)
def _list_open_incidences_items(
    venue_id: int,
    order_ids: Optional[tuple[int, ...]] = None,
    *,
    refresh_token: int = 0,
) -> List[Dict[str, Any]]:
    """Flat list of (order, provider) that has at least one open SeguimientoTicket.

    Optimized:
    - Reuses the bulk dashboard bundle (contexts + sent providers) instead of re-loading
      per-order context or re-querying ProviderSendStatus on every call.
    - Cached and invalidated via refresh_token.
    """
    _ = int(refresh_token or 0)

    if order_ids is None:
        orders = _get_active_orders(int(venue_id), refresh_token=_orders_refresh_token(int(venue_id)))
        order_ids = tuple(int(o.id) for o in orders if getattr(o, "id", None) is not None)
        if not order_ids:
            return []
    else:
        order_ids = tuple(int(x) for x in order_ids if int(x) > 0)
        if not order_ids:
            return []

    bundle = _load_dashboard_bundle(int(venue_id), order_ids, refresh_token=_orders_refresh_token(int(venue_id)))
    contexts = bundle.get("contexts", {}) or {}
    sent_by_order = bundle.get("sent_providers_by_order", {}) or {}

    out: List[Dict[str, Any]] = []
    for oid in order_ids:
        ctx = contexts.get(int(oid))
        if not ctx:
            continue

        sent_set = set(sent_by_order.get(int(oid), set()) or set())
        if not sent_set:
            continue

        # only providers actually sent
        providers = [p for p in ctx.lines_by_provider.keys() if norm_provider(p) in sent_set]
        providers.sort(key=lambda x: x.lower())

        for prov in providers:
            provn = norm_provider(prov)
            if _provider_closed(ctx, provn):
                continue
            open_t = _provider_open_tickets(ctx, provn)
            if not open_t:
                continue
            receipt = ctx.receipts_by_provider.get(provn)
            inv = _s(getattr(receipt, "invoice_number", None))
            out.append(
                {
                    "order": ctx.order,
                    "order_id": int(oid),
                    "provider": prov,
                    "invoice_number": inv,
                    "open_count": int(len(open_t)),
                }
            )

    out.sort(key=lambda r: (-int(r["open_count"]), -int(r["order_id"]), (r["provider"] or "").lower()))
    return out


# def _list_open_urgent_requests_grouped() -> Dict[int, List[UrgentReorderRequest]]:
#     """Group pending urgent requests by the *source order id* (via incidence SeguimientoTicket)."""
#     reqs = _list_open_urgent_requests()
#     if not reqs:
#         return {}

#     inc_ids = sorted({int(getattr(r, "incidence_id", 0) or 0) for r in reqs if int(getattr(r, "incidence_id", 0) or 0)})
#     if not inc_ids:
#         return {}

#     with get_session() as s:
#         tickets = list(s.exec(select(SeguimientoTicket).where(SeguimientoTicket.id.in_(inc_ids))).all())
#     ticket_order_by_id = {int(t.id): int(getattr(t, "order_id", 0) or 0) for t in tickets if getattr(t, "id", None) is not None}

#     grouped: Dict[int, List[UrgentReorderRequest]] = {}
#     for r in reqs:
#         inc = int(getattr(r, "incidence_id", 0) or 0)
#         oid = int(ticket_order_by_id.get(inc, 0) or 0)
#         if not oid:
#             continue
#         grouped.setdefault(oid, []).append(r)

#     # newest requests first per group
#     for oid in list(grouped.keys()):
#         grouped[oid].sort(key=lambda x: getattr(x, "created_at", None) or _now(), reverse=True)

#     return grouped


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

    st.markdown("# Dashboard")

    orders = _get_active_orders(int(venue_id), refresh_token=_orders_refresh_token(int(venue_id)))
    if not orders:
        st.info("No orders yet.")
        return
    
    order_ids = [int(o.id) for o in orders if o.id is not None]
    bundle = _load_dashboard_bundle(int(venue_id), tuple(order_ids), refresh_token=_orders_refresh_token(int(venue_id)))
    # NOTE: avoid escaped quotes (was causing SyntaxError on Streamlit Cloud)
    contexts = bundle.get("contexts", {})
    sent_providers_by_order = bundle.get("sent_providers_by_order", {})
    selected_id = order_ids[0]   # or your deep link logic / expander selection
    
    ctx = contexts.get(int(selected_id)) or _load_order_context(int(venue_id), int(selected_id), refresh_token=_orders_refresh_token(int(venue_id)))
    providers = sorted(ctx.lines_by_provider.keys(), key=lambda x: x.lower())
    
    
    
    #======================PANEL
    
     # ✅ Only keep providers that are actually sent (email or whatsapp)
    sent_providers = sent_providers_by_order.get(int(ctx.order.id), set())
    providers = [p for p in providers if norm_provider(p) in sent_providers]

    # -----------------------------
    # Deep-link: provider (best-effort)
    # -----------------------------
    desired_norm = None
    if deep_provider:
        desired_norm = norm_provider(deep_provider)
        st.session_state[f"recv_desired_provider_{int(ctx.order.id)}"] = desired_norm
        
    if not providers:
        st.info("No suppliers have been sent yet (📧 Email / WhatsApp).")
        return


    # -----------------------------
    # KPIs (GLOBAL across all active orders)
    # -----------------------------
    try:
        # Reuse already-loaded bundle + contexts (avoid extra DB roundtrips)
        orders_all = orders
        order_ids_all = tuple(int(o.id) for o in orders_all if getattr(o, "id", None) is not None)

        # Sent providers per order (already computed in _load_dashboard_bundle)
        sent_by_order_all: Dict[int, set[str]] = sent_providers_by_order or {}
        providers_global = sum(len(v or set()) for v in sent_by_order_all.values())

        # Open incidences (cached + uses bundle)
        inc_items = _list_open_incidences_items(
            int(venue_id),
            order_ids_all,
            refresh_token=_orders_refresh_token(int(venue_id)),
        )
        open_total = sum(int(it.get("open_count") or 0) for it in inc_items)

        # Pending products (global): lines pending "Receive" across all active orders/providers
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

        pending_products = 0
        redeliveries_pending = 0
        credit_notes_pending = 0

        for oid in order_ids_all:
            ctx_o = contexts.get(int(oid))
            if not ctx_o:
                continue

            sent_set = set(sent_by_order_all.get(int(oid), set()) or set())
            if not sent_set:
                continue

            # pending products
            for prov in sent_set:
                wf = ctx_o.workflows_by_provider.get(norm_provider(prov))
                stt = _s(getattr(wf, "state", None)).upper() if wf else "ORDER_SENT"
                if stt in DONE_STATES:
                    continue

                provn = norm_provider(prov)
                for ln in (ctx_o.lines_by_provider.get(provn, []) or []):
                    lid = int(getattr(ln, "id", 0) or 0)
                    if not lid:
                        continue
                    fu = ctx_o.followups_by_key.get((provn, lid))
                    # If venue already entered qty, it's not pending
                    if fu is not None and getattr(fu, "venue_qty", None) is not None:
                        continue
                    if bool(getattr(ln, "received_ok", False)):
                        continue
                    pending_products += 1

            # re-deliveries / credit notes pending (ticket meta)
            for prov in sent_set:
                provn = norm_provider(prov)
                if _provider_closed(ctx_o, provn):
                    continue
                open_t = _provider_open_tickets(ctx_o, provn)
                if not open_t:
                    continue

                has_credit_pending = False
                has_redel_pending = False
                for t in open_t:
                    meta = _parse_ticket_resolution_note(_s(getattr(t, "resolution_note", None)))
                    r = _s(meta.get("resolution")).strip().lower()
                    if r == "credit_note":
                        has_credit_pending = True
                    elif r in {"supplementary_delivery", "re_delivery", "re-delivery", "redelivery"}:
                        has_redel_pending = True

                if has_redel_pending:
                    redeliveries_pending += 1
                if has_credit_pending:
                    credit_notes_pending += 1

        # Urgent: pending urgent requests
        try:
            urgent_requests_pending = len(_list_open_urgent_requests())
        except Exception:
            urgent_requests_pending = 0

    except Exception:
        providers_global = 0
        open_total = 0
        pending_products = 0
        redeliveries_pending = 0
        credit_notes_pending = 0
        urgent_requests_pending = 0

    st.markdown(
        "<div class='voi-kpi'>"
        # f"<div class='k'><div class='t'>Providers</div><div class='v'>{providers_global}</div></div>"
        f"<div class='k'><div class='t'>Pending products</div><div class='v'>{pending_products}</div></div>"
        f"<div class='k'><div class='t'>Open incidences</div><div class='v'>{open_total}</div></div>"
        f"<div class='k'><div class='t'>Re-deliveries</div><div class='v'>{redeliveries_pending}</div></div>"
        f"<div class='k'><div class='t'>Credit notes</div><div class='v'>{credit_notes_pending}</div></div>"
        f"<div class='k'><div class='t'>Urgent requests</div><div class='v'>{urgent_requests_pending}</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    tab_labels = ["📦 Receive", "🚨 Incidences", "⚡ Urgent"]
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

        f1, f2 = st.columns([2.2, 1.0], vertical_alignment="center")
        with f1:
            q = st.text_input("Search provider / invoice / order", placeholder="e.g. makro, 2026-, #12").strip().lower()
        with f2:
            expand_all = st.toggle("Expand all", value=False)

        def _matches(t: Dict[str, Any]) -> bool:
            if not q:
                return True
            prov = (t.get("provider_display") or "").lower()
            inv = (t.get("invoice_number") or "").lower()
            oid = str(t.get("order_id") or "")
            return (q in prov) or (q in inv) or (q in oid) or (q in f"#{oid}")

        tasks2 = [t for t in tasks if _matches(t)]
        if not tasks2:
            st.info("No matches.")
            return

        def _pretty_actor(actor: str) -> str:
            actor = (actor or "").strip()
            if not actor:
                return "Unknown user"
            # If it's an email, show a nicer label (before @), but keep full email if you prefer
            if "@" in actor:
                short = actor.split("@", 1)[0].replace(".", " ").replace("_", " ").strip()
                return short.title() if short else actor
            return actor

        for t in tasks2:
            oid = int(t["order_id"])
            prov = t["provider_display"] or "—"
            inv = t["invoice_number"] or "—"
            order = t.get("order")  # this exists because _list_pending_receive_items adds it

            created_at = getattr(order, "created_at", None)
            created_txt = created_at.strftime("%d %b %Y, %H:%M") if created_at else "—"

            # URL sync (optional)
            if qp_int("order_id") != oid:
                set_query_params(page="tracking", order_id=str(oid), provider=norm_provider(prov))

            ctx = contexts.get(int(oid)) or _load_order_context(int(venue_id), int(oid), refresh_token=_orders_refresh_token(int(venue_id)))
            # Render each provider panel as a fragment when available.
            # This keeps saves fast by rerunning only the provider section instead of the whole dashboard.
            if hasattr(st, 'fragment'):
                @st.fragment
                def _provider_panel(_ctx=ctx, _prov=prov):
                    _render_receive_provider_panel(_ctx, _prov)
                _provider_panel()
            else:
                _render_receive_provider_panel(ctx, prov)
            
            st.empty()

        return

    # -----------------------------
    # 🚨 Incidences (global)
    # -----------------------------
    if selected_tab == "🚨 Incidences":
        items = _list_open_incidences_items(int(venue_id))
        if not items:
            st.success("✅ No open incidences.")
            return

        f1, f2 = st.columns([2.2, 1.0], vertical_alignment="center")
        with f1:
            q = st.text_input("Search provider / invoice / order", key="inc_global_search", placeholder="e.g. invoice, #34").strip().lower()
        with f2:
            expand_all = st.toggle("Expand all", value=False, key="inc_global_expand_all")

        def _matches_inc(t: Dict[str, Any]) -> bool:
            if not q:
                return True
            prov = (t.get("provider") or "").lower()
            inv = (t.get("invoice_number") or "").lower()
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
            title = f"{prov} · 🧾 {inv} · #{oid} · {n} open"

            # expanded = bool(expand_all) or (deep_order_id is not None and int(deep_order_id) == oid)
            # with st.expander(title, expanded=expanded):
            if qp_int("order_id") != oid:
                set_query_params(page="tracking", order_id=str(oid), provider=norm_provider(prov))

            ctx = contexts.get(int(oid)) or _load_order_context(int(venue_id), int(oid), refresh_token=_orders_refresh_token(int(venue_id)))
            _render_incidences_cards(ctx, [prov], show_prices=True, include_iva=True)

        return

    # -----------------------------
    # ⚡ Urgent (global)
    # -----------------------------
    if selected_tab == "⚡ Urgent":
        _render_urgent_tab(ctx)
