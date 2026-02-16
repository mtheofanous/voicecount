"""
Modern Orders UX (Streamlit)

This file replaces the old Orders tab with a simpler, modern flow:

DRAFT  -> build the order lines
READY  -> send per supplier (email/wa/txt) + creates workflow state ORDER_SENT per supplier
PENDING-> quick overview + open the seguimiento (venue) link per supplier
FINAL  -> read-only history

IMPORTANT:
- This file does NOT handle payments.
- It does NOT issue invoices / credit notes.
- It only sends links and tracks send attempts.
- Operational state machine lives in OrderWorkflow / OrderWorkflowEvent and the seguimiento page.

Source base: previous refactor file. fileciteturn2file0
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Any, Optional

import re
import unicodedata
import urllib.parse as up

import pandas as pd
import streamlit as st
from sqlmodel import select
from sqlalchemy import func

from core.db import get_session
from core.public_links import build_seguimiento_url, ROLE_SUPPLIER, ROLE_VENUE, norm_provider
from core.mailer import send_smtp_email
from core.url_nav import set_query_params, qp_int, qp_str
from features.manage_orders.emails import build_order_email_full

from domain.models import (
    Order,
    OrderLine,
    Product,
    Provider,
    ProviderSendStatus,
    ProviderDiscountRule,
)

try:
    from domain.models import ProviderReceipt  # type: ignore
except Exception:  # pragma: no cover
    ProviderReceipt = None  # type: ignore

try:
    from domain.models import OrderWorkflow, OrderWorkflowEvent  # type: ignore
except Exception:  # pragma: no cover
    OrderWorkflow = None  # type: ignore
    OrderWorkflowEvent = None  # type: ignore

try:
    from features.auth_and_manage.auth_multi_tenant import get_auth_session, Venue, User  # type: ignore
except Exception:  # pragma: no cover
    get_auth_session = None  # type: ignore
    Venue = None  # type: ignore
    User = None  # type: ignore


@dataclass(frozen=True)
class OrderRow:
    id: int
    title: str
    status: str
    created_at: Optional[datetime] = None


def current_actor() -> str:
    """Compatibility: other modules import current_actor() from here."""
    try:
        from features.auth_and_manage.auth_multi_tenant import current_user  # type: ignore

        u = current_user() or {}
        email = (u.get("email") or "").strip()
        if email:
            return email
        full_name = (u.get("full_name") or u.get("name") or "").strip()
        if full_name:
            return full_name
        uid = u.get("id")
        if uid is not None:
            return f"user:{uid}"
    except Exception:
        pass

    return (st.session_state.get("actor") or "system")


def _now():
    from datetime import datetime as _dt

    return _dt.utcnow()


def _s(x: Any) -> str:
    return ("" if x is None else str(x)).strip()



def _fmt_pct(p: float) -> str:
    try:
        p = float(p or 0.0)
    except Exception:
        p = 0.0
    if abs(p) < 1e-9:
        p = 0.0
    return f"{p:.1f}%"

def _fmt_price(p: float) -> str:
    try:
        p = float(p or 0.0)
    except Exception:
        p = 0.0
    if abs(p) < 1e-12:
        p = 0.0
    return f"{p:.4f}"

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _split_first_pipe(value: Optional[str]) -> str:
    raw = _s(value)
    if not raw:
        return ""
    for part in raw.split("|"):
        p = part.strip()
        if p:
            return p
    return ""


def _split_emails(raw: str) -> list[str]:
    raw = _s(raw)
    if not raw:
        return []
    raw = raw.replace(",", "|")
    return [p.strip() for p in raw.split("|") if p.strip()]


def _normalize_phone(raw: str, country_code: str) -> str:
    raw = _s(raw)
    if not raw:
        return ""
    digits = "".join(ch for ch in raw if ch.isdigit() or ch == "+")
    if not digits:
        return ""
    if digits.startswith("+"):
        return digits
    digits2 = digits.lstrip("0")
    cc = _s(country_code) or "+34"
    if not cc.startswith("+"):
        cc = "+" + cc
    return f"{cc}{digits2}"


def _order_label(o: Order) -> str:
    title = _s(getattr(o, "title", "")) or ""
    when = getattr(o, "created_at", None)
    when_s = when.strftime("%Y-%m-%d %H:%M") if when else ""
    base = f"#{int(o.id)}" if getattr(o, "id", None) is not None else "#—"
    return f"{base} — {title}" if title else (f"{base} — {when_s}" if when_s else base)


def _status_chip(status: str) -> str:
    s = (_s(status)).lower()
    if s == "draft":
        return "📝 Borrador"
    if s == "ready_to_send":
        return "📤 Listo"
    if s == "pending_receive":
        return "📦 Pendiente"
    if s == "final":
        return "✅ Historial"
    return s or "—"


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)


def _norm_words(s: str) -> list[str]:
    raw = re.findall(r"[0-9]+|[^\W_]+", (s or ""), flags=re.UNICODE)
    return [_strip_accents(w).lower() for w in raw if w.strip()]


def _remove_name_words_from_description(name: str, desc: str) -> str:
    name_set = set(_norm_words(name))
    if not name_set:
        return (desc or "").strip()
    desc_raw = re.findall(r"[0-9]+|[^\W_]+", (desc or ""), flags=re.UNICODE)
    kept: list[str] = []
    for w in desc_raw:
        if _strip_accents(w).lower() not in name_set:
            kept.append(w)
    return " ".join(kept).strip()


@st.cache_data(ttl=10, show_spinner=False, hash_funcs={type(lambda: None): lambda _: "session_fn"})
def _list_orders_cached(_get_session_fn, venue_id: int, refresh_token: int) -> list[OrderRow]:
    _ = refresh_token
    with _get_session_fn() as s:
        rows = s.exec(
            select(Order.id, Order.title, Order.status, Order.created_at)
            .where(Order.venue_id == int(venue_id))
            .order_by(Order.created_at.desc())
        ).all()

    out: list[OrderRow] = []
    for oid, title, status, created_at in rows:
        if oid is None:
            continue
        out.append(
            OrderRow(
                id=int(oid),
                title=_s(title) or f"Pedido #{int(oid)}",
                status=_s(status) or "draft",
                created_at=created_at,
            )
        )
    return out


@st.cache_data(ttl=60, show_spinner=False, hash_funcs={type(lambda: None): lambda _: "session_fn"})
def _products_cached(_get_session_fn, venue_id: int) -> list[Product]:
    with _get_session_fn() as s:
        return list(
            s.exec(select(Product).where(Product.venue_id == venue_id).order_by(Product.name.asc(), Product.provider_name.asc())).all()
        )


@st.cache_data(ttl=120, show_spinner=False, hash_funcs={type(lambda: None): lambda _: "session_fn"})
def _product_ui_index_cached(_get_session_fn, venue_id: int):
    """Precomputed product index for fast reruns (search/filter/labels)."""
    products = _products_cached(_get_session_fn, int(venue_id))

    products_by_id: dict[int, Product] = {int(p.id): p for p in products if getattr(p, "id", None) is not None}

    label_by_id = _products_label_map(products)

    cat_by_pid: dict[int, str] = {
        int(p.id): (_s(getattr(p, "category", "")) or "").strip()
        for p in products
        if getattr(p, "id", None) is not None
    }
    prov_by_pid: dict[int, str] = {
        int(p.id): (_s(getattr(p, "provider_name", "")) or "(Sin proveedor)").strip()
        for p in products
        if getattr(p, "id", None) is not None
    }

    all_categories = sorted({c for c in cat_by_pid.values() if c})
    all_providers = sorted({p for p in prov_by_pid.values() if p})

    base_pids = sorted(label_by_id.keys())
    return (
        products,
        products_by_id,
        label_by_id,
        cat_by_pid,
        prov_by_pid,
        all_categories,
        all_providers,
        base_pids,
    )


@st.cache_data(ttl=10, show_spinner=False, hash_funcs={type(lambda: None): lambda _: "session_fn"})
def _order_lines_cached(_get_session_fn, order_id: int, refresh_token: int) -> list[OrderLine]:
    _ = refresh_token
    with _get_session_fn() as s:
        return list(s.exec(select(OrderLine).where(OrderLine.order_id == order_id).order_by(OrderLine.id.asc())).all())


@st.cache_data(ttl=60, show_spinner=False, hash_funcs={type(lambda: None): lambda _: "session_fn"})
def _providers_cached(_get_session_fn, venue_id: int) -> dict[str, Provider]:
    with _get_session_fn() as s:
        rows = list(s.exec(select(Provider).where(Provider.venue_id == venue_id)).all())
    out: dict[str, Provider] = {}
    for p in rows:
        out[norm_provider(_s(getattr(p, "name", "")))] = p
    return out


# -----------------------------------------------------------------------------
# Smart cesta (draft optimizer): suggest same/similar products from other providers
# using effective net unit price (base price minus provider discount rules).
# -----------------------------------------------------------------------------

_STOPWORDS_SMART = {
    "de","del","la","el","los","las","y","en","con","sin","para","por","a","al","un","una","unos","unas",
    "kg","g","gr","l","ml","ud","uds","unidad","unidades",
}

def _norm_tokens_smart(s: str) -> set[str]:
    toks = {_strip_accents(t).lower() for t in re.findall(r"[0-9]+|[^\W_]+", (s or ""), flags=re.UNICODE)}
    return {t for t in toks if t and t not in _STOPWORDS_SMART}

def _relevance_smart(a: str, b: str) -> float:
    A = _norm_tokens_smart(a)
    B = _norm_tokens_smart(b)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    j = inter / union if union else 0.0
    # small bonus if one name contains the other (after normalization)
    sa = " ".join(sorted(A))
    sb = " ".join(sorted(B))
    bonus = 0.10 if (sa and sb and (sa in sb or sb in sa)) else 0.0
    return float(min(1.0, j + bonus))



def _smart_product_label(name: str, desc: str) -> str:
    name = (_s(name) or "").strip()
    desc = (_s(desc) or "").strip()

    if not name and not desc:
        return "este producto"

    if not name:
        return desc
    if not desc:
        return name

    # normalize for "contains" check
    n = re.sub(r"\s+", " ", name).strip().lower()
    d = re.sub(r"\s+", " ", desc).strip().lower()

    # if desc already contains name (or is basically the same), don't repeat
    if n and (n in d or d in n):
        return desc  # desc already includes name context

    return f"{name} — {desc}"

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

def _rules_for_provider_global(
    *,
    venue_id: int,
    providers_by_name: dict[str, Provider],
    provider_name: str,
) -> list[ProviderDiscountRule]:
    pnorm = norm_provider(provider_name)
    prow = providers_by_name.get(pnorm)
    if not prow or getattr(prow, "id", None) is None:
        return []
    return _provider_rules_cached(get_session, int(venue_id), int(prow.id))

def _match_scope_ok_global(r: ProviderDiscountRule, pid: Optional[int]) -> bool:
    if getattr(r, "product_id", None) is None:
        return True
    if pid is None:
        return False
    return int(r.product_id) == int(pid)

def _rule_priority_key_global(r: ProviderDiscountRule) -> tuple:
    is_product = 1 if getattr(r, "product_id", None) is not None else 0
    rk = (getattr(r, "rule_kind", "") or "line_pct").strip()
    is_prev = 1 if rk.startswith("prev_month") else 0
    threshold = float(getattr(r, "prev_month_min_qty", 0.0) or 0.0) if is_prev else float(getattr(r, "min_qty", 0.0) or 0.0)
    strength = float(getattr(r, "discount_percent", 0.0) or 0.0)
    return (is_product, is_prev, threshold, strength)

def _pricing_for_line_global(
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
    rules = _rules_for_provider_global(venue_id=venue_id, providers_by_name=providers_by_name, provider_name=prov_norm)
    if not rules:
        return {"net_unit": gross_unit, "discount_pct": 0.0, "rule_kind": "", "rule_id": None, "applied": False}

    candidates: list[ProviderDiscountRule] = []
    for r in rules:
        rk = (getattr(r, "rule_kind", "") or "line_pct").strip()
        if not _match_scope_ok_global(r, pid):
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

    candidates.sort(key=_rule_priority_key_global, reverse=True)
    chosen = candidates[0]
    rk = (getattr(chosen, "rule_kind", "") or "line_pct").strip()

    if rk.endswith("_net_price"):
        net_unit = float(getattr(chosen, "price_override", 0.0) or 0.0)
        if net_unit <= 0:
            return {"net_unit": gross_unit, "discount_pct": 0.0, "rule_kind": "", "rule_id": None, "applied": False}
        disc_pct = (1.0 - (net_unit / gross_unit)) * 100.0 if gross_unit > 0 else 0.0
        disc_pct = max(0.0, min(100.0, disc_pct))
        return {
            "net_unit": float(net_unit),
            "discount_pct": float(disc_pct),
            "rule_kind": rk,
            "rule_id": int(chosen.id) if getattr(chosen, "id", None) is not None else None,
            "applied": True,
        }

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


def _smart_cesta_suggestions(
    *,
    venue_id: int,
    providers_by_name: dict[str, Provider],
    products_by_id: dict[int, Product],
    draft_df: pd.DataFrame,
    min_rel: float = 0.45,
    min_saving_pct: float = 0.02,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    # Index products by unit for faster candidate search
    by_unit: dict[str, list[tuple[int, Product]]] = {}
    for pid, p in (products_by_id or {}).items():
        unit = (_s(getattr(p, "unit", "")) or "unidad").strip().lower()
        by_unit.setdefault(unit, []).append((int(pid), p))

    out: list[dict[str, Any]] = []

    for row_idx, r in draft_df.iterrows():
        if bool(r.get("delete", False)):
            continue
        pid = _pid_to_int(r.get("product_id"))
        if pid is None:
            continue
        qty = float(_safe_float(r.get("quantity"), 0.0))
        if qty <= 0:
            continue

        cur_p = products_by_id.get(int(pid))
        if not cur_p:
            continue

        cur_name = _s(getattr(cur_p, "name", "")) or ""
        cur_desc = _s(getattr(cur_p, "description", "")) or ""
        cur_unit = (_s(getattr(cur_p, "unit", "")) or "unidad").strip().lower()
        cur_provider = norm_provider(_s(getattr(cur_p, "provider_name", "")) or "")
        cur_gross = float(_safe_float(getattr(cur_p, "price", 0.0), 0.0))

        cur_pr = _pricing_for_line_global(
            venue_id=int(venue_id),
            providers_by_name=providers_by_name,
            provider_name=cur_provider,
            pid=int(pid),
            qty=qty,
            gross_unit=cur_gross,
        )
        cur_net = float(cur_pr.get("net_unit") or cur_gross)

        cands: list[dict[str, Any]] = []
        for cand_pid, cand_p in by_unit.get(cur_unit, []):
            cand_provider = norm_provider(_s(getattr(cand_p, "provider_name", "")) or "")
            if not cand_provider or cand_provider == cur_provider:
                continue

            rel = float(_relevance_smart(cur_name, _s(getattr(cand_p, "name", "")) or "") or 0.0)
            if rel < float(min_rel):
                continue

            cand_gross = float(_safe_float(getattr(cand_p, "price", 0.0), 0.0))
            if cand_gross <= 0:
                continue

            cand_pr = _pricing_for_line_global(
                venue_id=int(venue_id),
                providers_by_name=providers_by_name,
                provider_name=cand_provider,
                pid=int(cand_pid),
                qty=qty,
                gross_unit=cand_gross,
            )
            cand_net = float(cand_pr.get("net_unit") or cand_gross)

            if cur_net > 0:
                saving_pct = (cur_net - cand_net) / cur_net
                if saving_pct < float(min_saving_pct):
                    continue

            cands.append(
                {
                    "provider": cand_provider,
                    "product_id": int(cand_pid),
                    "name": _s(getattr(cand_p, "name", "")) or "",
                    "description": _s(getattr(cand_p, "description", "")) or "",
                    "relevance": float(rel),
                    "gross_unit": float(cand_gross),
                    "net_unit": float(cand_net),
                    "discount_pct": float(cand_pr.get("discount_pct") or 0.0),
                    "rule_kind": _s(cand_pr.get("rule_kind") or ""),
                }
            )

        cands.sort(key=lambda x: (x["net_unit"], -x["relevance"]))
        best = cands[: max(1, int(top_k))]

        if best:
            out.append(
                {
                    "row_idx": int(row_idx),
                    "line_id": r.get("line_id", None),
                    "line_product_id": int(pid),
                    "line_name": cur_name,
                    "line_description": cur_desc,
                    "qty": float(qty),
                    "unit": cur_unit,
                    "current_provider": cur_provider,
                    "current_gross": float(cur_gross),
                    "current_net": float(cur_net),
                    "current_discount_pct": float(cur_pr.get("discount_pct") or 0.0),
                    "suggestions": best,
                }
            )

    return out


@st.cache_data(ttl=10, show_spinner=False, hash_funcs={type(lambda: None): lambda _: "session_fn"})
def _provider_receipts_cached(_get_session_fn, order_id: int, refresh_token: int) -> dict[str, Any]:
    _ = refresh_token
    if ProviderReceipt is None:
        return {}
    with _get_session_fn() as s:
        rows = list(s.exec(select(ProviderReceipt).where(ProviderReceipt.order_id == order_id)).all())
    return {norm_provider(_s(getattr(r, "provider_name", ""))): r for r in rows}


def _bump_refresh(venue_id: int) -> None:
    k = f"orders_refresh_token_{venue_id}"
    st.session_state[k] = int(st.session_state.get(k, 0)) + 1


def _refresh_token(venue_id: int) -> int:
    return int(st.session_state.get(f"orders_refresh_token_{venue_id}", 0))


@dataclass
class VenueTemplates:
    venue_name: str = ""
    owner_name: str = ""
    address: str = ""
    tax_number: str = ""
    email: str = ""
    phone: str = ""
    email_lang: str = "es"
    email_cc: str = ""
    email_bcc: str = ""


@st.cache_data(ttl=120, show_spinner=False)
def _load_venue_templates(venue_id: int, refresh_token: int) -> VenueTemplates:
    _ = refresh_token
    if get_auth_session is None or Venue is None or User is None:
        return VenueTemplates()
    v = None
    owner_name = ""
    try:
        with get_auth_session() as s:
            v = s.exec(select(Venue).where(Venue.id == venue_id)).first()
            if v:
                owner_name = (
                    s.exec(
                        select(User.full_name)
                        .where(User.account_id == v.account_id, User.account_role == "owner", User.is_active == True)  # noqa
                        .order_by(User.created_at.asc())
                    ).first()
                    or ""
                )
    except Exception:
        pass
    if not v:
        return VenueTemplates()
    email_lang = _s(getattr(v, "email_lang", "")).lower() or "es"
    if email_lang not in {"es", "en", "gr"}:
        email_lang = "es"
    return VenueTemplates(
        venue_name=_s(getattr(v, "name", "")),
        owner_name=_s(owner_name),
        address=_s(getattr(v, "address", "")),
        tax_number=_s(getattr(v, "tax_number", "")),
        email=_s(getattr(v, "email", "")),
        phone=_s(getattr(v, "phone", "")),
        email_lang=email_lang,
        email_cc=_s(getattr(v, "email_cc", "")),
        email_bcc=_s(getattr(v, "email_bcc", "")),
    )


def _venue_missing_required(v: VenueTemplates) -> list[str]:
    missing: list[str] = []
    if not v.tax_number:
        missing.append("NIF/CIF (tax number)")
    if not v.address:
        missing.append("Dirección (address)")
    return missing


def _set_order_status(order_id: int, status: str, actor: str) -> None:
    with get_session() as s:
        o = s.exec(select(Order).where(Order.id == order_id)).first()
        if not o:
            return
        o.status = status
        o.updated_at = _now()
        o.updated_by = actor
        if status == "final":
            o.verified_at = _now()
            o.verified_by = actor
        s.add(o)
        s.commit()


def _create_empty_draft(venue_id: int, actor: str) -> int:
    with get_session() as s:
        o = Order(
            venue_id=venue_id,
            status="draft",
            created_at=_now(),
            updated_at=_now(),
            created_by=actor,
            updated_by=actor,
            title=None,
            note=None,
        )
        s.add(o)
        s.commit()
        s.refresh(o)
        return int(o.id)


def _delete_order(order_id: int) -> None:
    with get_session() as s:
        for ln in s.exec(select(OrderLine).where(OrderLine.order_id == int(order_id))).all():
            s.delete(ln)
        o = s.exec(select(Order).where(Order.id == order_id)).first()
        if o:
            s.delete(o)
        s.commit()


def _pid_to_int(pid: Any) -> Optional[int]:
    if pid is None:
        return None
    if isinstance(pid, (list, tuple)):
        pid = pid[0] if pid else None
    try:
        if pid is not None and pd.isna(pid):
            return None
    except Exception:
        pass
    if isinstance(pid, int):
        return pid
    if isinstance(pid, float):
        return int(pid)
    s = str(pid).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _sanitize_editor_df(df0: pd.DataFrame) -> pd.DataFrame:
    df1 = df0.copy()
    for col, default in [
        ("line_id", pd.NA),
        ("product_id", pd.NA),
        ("quantity", 0.0),
        ("unit", "unidad"),
        ("delete", False),
    ]:
        if col not in df1.columns:
            df1[col] = default
    df1["product_id"] = df1["product_id"].apply(_pid_to_int)
    df1["product_id"] = df1["product_id"].where(df1["product_id"].notna(), pd.NA)
    df1["quantity"] = pd.to_numeric(df1["quantity"], errors="coerce").fillna(0.0)
    df1["unit"] = df1["unit"].astype(str).replace({"nan": "unidad"}).fillna("unidad")
    df1["delete"] = df1["delete"].fillna(False).astype(bool)
    return df1


def _save_lines_from_editor(*, venue_id: int, order_id: int, actor: str, df: pd.DataFrame, products_by_id: dict[int, Product]) -> None:
    df2 = _sanitize_editor_df(df)
    del_ids = [int(x) for x in df2.loc[df2["delete"] == True, "line_id"].dropna().tolist()]  # noqa: E712
    keep = df2.loc[df2["delete"] != True].copy()  # noqa: E712
    keep = keep.loc[keep["quantity"] > 0].copy()
    now = _now()
    with get_session() as s:
        if del_ids:
            for ln in s.exec(select(OrderLine).where(OrderLine.id.in_(del_ids))).all():
                s.delete(ln)
        for _, r in keep.iterrows():
            line_id = r.get("line_id")
            pid = _pid_to_int(r.get("product_id")) or 0
            if not pid:
                continue
            qty = float(r.get("quantity") or 0.0)
            prod = products_by_id.get(pid)
            provider_name = _s(getattr(prod, "provider_name", "")) if prod else ""
            if pd.notna(line_id):
                ln = s.exec(select(OrderLine).where(OrderLine.id == int(line_id))).first()
                if ln:
                    ln.product_id = pid
                    ln.quantity = qty
                    ln.unit = (_s(getattr(prod, "unit", "")) or _s(r.get("unit")) or "unidad").lower()
                    ln.provider = provider_name or ln.provider
                    ln.updated_at = now
                    ln.updated_by = actor
                    s.add(ln)
            else:
                s.add(
                    OrderLine(
                        venue_id=venue_id,
                        order_id=order_id,
                        product_id=pid,
                        quantity=qty,
                        unit=(_s(getattr(prod, "unit", "")) or "unidad").lower(),
                        provider=provider_name or None,
                        updated_at=now,
                        updated_by=actor,
                    )
                )
        o = s.exec(select(Order).where(Order.id == order_id)).first()
        if o:
            o.updated_at = now
            o.updated_by = actor
            if not o.created_by:
                o.created_by = actor
            s.add(o)
        s.commit()


def _get_send_status_map(*, order_id: int) -> dict[str, ProviderSendStatus]:
    with get_session() as s:
        rows = s.exec(select(ProviderSendStatus).where(ProviderSendStatus.order_id == int(order_id))).all()
    return {norm_provider(_s(r.provider_name)): r for r in rows}


def _touch_send_status(*, venue_id: int, order_id: int, provider_name: str, actor: str, channel: str, ok: bool, error: str | None = None) -> None:
    prov = norm_provider(provider_name)
    now = _now()
    with get_session() as s:
        obj = s.exec(select(ProviderSendStatus).where(ProviderSendStatus.order_id == int(order_id), ProviderSendStatus.provider_name == prov)).first()
        if not obj:
            obj = ProviderSendStatus(venue_id=int(venue_id), order_id=int(order_id), provider_name=prov)
        obj.send_attempts = int(getattr(obj, "send_attempts", 0) or 0) + 1
        obj.updated_at = now
        if ok:
            obj.sent = True
            obj.sent_at = now
            obj.sent_by = actor or None
            obj.last_error = None
            if channel == "email":
                obj.sent_email = True
            elif channel == "whatsapp":
                obj.sent_whatsapp = True
            elif channel == "txt":
                obj.sent_txt = True
        else:
            obj.last_error = (_s(error) or "unknown error")[:500]
        s.add(obj)
        s.commit()


def _all_providers_sent(*, order_id: int, provider_names: list[str]) -> bool:
    provs = {norm_provider(p) for p in provider_names}
    m = _get_send_status_map(order_id=order_id)
    sent_provs = {p for p, row in m.items() if bool(getattr(row, "sent", False))}
    return provs.issubset(sent_provs)


def _reset_send_status(*, order_id: int) -> None:
    with get_session() as s:
        rows = s.exec(select(ProviderSendStatus).where(ProviderSendStatus.order_id == int(order_id))).all()
        for r in rows:
            r.sent = False
            r.sent_email = False
            r.sent_whatsapp = False
            r.sent_txt = False
            r.sent_at = None
            r.sent_by = None
            r.send_attempts = 0
            r.last_error = None
            r.updated_at = _now()
            s.add(r)
        s.commit()


def _ensure_workflow_order_sent(*, venue_id: int, order_id: int, provider_name: str, actor: str) -> None:
    if OrderWorkflow is None:
        return
    prov = norm_provider(provider_name)
    now = _now()
    with get_session() as s:
        wf = s.exec(select(OrderWorkflow).where(OrderWorkflow.venue_id == int(venue_id), OrderWorkflow.order_id == int(order_id), OrderWorkflow.provider_name == prov)).first()
        if not wf:
            wf = OrderWorkflow(
                venue_id=int(venue_id),
                order_id=int(order_id),
                provider_name=prov,
                state="ORDER_SENT",
                updated_at=now,
                updated_by_role="venue",
                updated_by=actor,
                note="Order link sent to supplier.",
            )
            s.add(wf)
            if OrderWorkflowEvent is not None:
                s.add(
                    OrderWorkflowEvent(
                        venue_id=int(venue_id),
                        order_id=int(order_id),
                        provider_name=prov,
                        from_state="ORDER_SENT",
                        to_state="ORDER_SENT",
                        actor_role="venue",
                        actor=actor,
                        at=now,
                        note="Order link sent to supplier.",
                    )
                )
        s.commit()


def _products_label_map(products: list[Product]) -> dict[int, str]:
    out: dict[int, str] = {}
    for p in products:
        if p.id is None:
            continue
        name = _s(getattr(p, "name", ""))
        desc = _s(getattr(p, "description", ""))
        desc_clean = _remove_name_words_from_description(name, desc)
        prov = _s(getattr(p, "provider_name", "")) or "(Sin proveedor)"
        parts = [name] + ([desc_clean] if desc_clean else []) + [prov]
        out[int(p.id)] = " — ".join([x for x in parts if _s(x)])
    return out


def _group_lines_by_provider(lines: list[OrderLine], products_by_id: dict[int, Product]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for ln in lines:
        pid = getattr(ln, "product_id", None)
        prod = products_by_id.get(int(pid)) if pid is not None else None
        prov = norm_provider(_s(getattr(prod, "provider_name", None) if prod else getattr(ln, "provider", None)))
        name = _s(getattr(prod, "name", None) if prod else getattr(ln, "spoken_name", None) or "Producto")
        unit = (_s(getattr(prod, "unit", None)) if prod else _s(getattr(ln, "unit", None))) or "unidad"
        qty = _safe_float(getattr(ln, "quantity", 0.0), 0.0)
        out.setdefault(prov, []).append({"line_id": int(getattr(ln, "id", 0) or 0), "product_id": pid, "name": name, "qty": qty, "unit": unit})
    for prov in out:
        out[prov] = sorted(out[prov], key=lambda x: (x["name"] or "").lower())
    return dict(sorted(out.items(), key=lambda kv: kv[0].lower()))


def _editor_df_from_lines(lines: list[OrderLine], products_by_id: dict[int, Product]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ln in lines:
        pid = getattr(ln, "product_id", None)
        prod = products_by_id.get(int(pid)) if pid is not None else None
        rows.append(
            {
                "line_id": getattr(ln, "id", None),
                "product_id": pid,
                "quantity": float(getattr(ln, "quantity", 0.0) or 0.0),
                "unit": (_s(getattr(prod, "unit", "")) or _s(getattr(ln, "unit", "")) or "unidad").lower(),
                "delete": False,
            }
        )
    if not rows:
        rows = [{"line_id": pd.NA, "product_id": pd.NA, "quantity": 1.0, "unit": "unidad", "delete": False}]
    return pd.DataFrame(rows)


def _build_supplier_message_text(
    *,
    templates: VenueTemplates,
    order_id: int,
    provider_name: str,
    prov_lines: list[dict[str, Any]],
    products_by_id: dict[int, Product],
) -> str:
    supplier_link = build_seguimiento_url(
        order_id=order_id,
        provider_name=provider_name,
        role=ROLE_SUPPLIER,
        page_path="seguimiento",
    )

    items: list[dict[str, Any]] = []
    for it in prov_lines:
        qty = _safe_float(it.get("qty"), 0.0)
        if qty <= 0:
            continue

        pid = _pid_to_int(it.get("product_id"))
        prod = products_by_id.get(pid) if pid else None

        name = _s(getattr(prod, "name", None) if prod else it.get("name") or "Producto")
        desc = _s(getattr(prod, "description", None) if prod else it.get("description") or "")
        unit = (_s(getattr(prod, "unit", "")) if prod else _s(it.get("unit", ""))) or "unidad"

        items.append({"name": name, "description": desc, "qty": qty, "unit": unit})

    # subject+body consistency handled in emails.py; here we only return body text
    _subject, body = build_order_email_full(
        venue_ctx=templates,
        order_id=int(order_id),
        provider_name=provider_name,
        supplier_link=supplier_link,
        items=items,
        lang=_s(getattr(templates, "email_lang", "es")) or "es",
    )
    return body

def _build_subject(templates: VenueTemplates, order_id: int, provider_name: str) -> str:
    supplier_link = ""  # not needed for subject
    items: list[dict[str, Any]] = []
    subject, _body = build_order_email_full(
        venue_ctx=templates,
        order_id=int(order_id),
        provider_name=provider_name,
        supplier_link=supplier_link,
        items=items,
        lang=_s(getattr(templates, "email_lang", "es")) or "es",
    )
    return subject


def _inject_css() -> None:
    st.markdown(
        """
        
    <style>
    .voi-pill{
    display:inline-block;
    padding:4px 8px;
    border-radius:999px;
    border:1px solid rgba(49,51,63,.2);
    background:rgba(49,51,63,.04);
    font-size:.85rem;
    white-space:nowrap;
    }
    .voi-pill--save{
    border-color: rgba(46,125,50,.35);
    background: rgba(46,125,50,.10);
    }
    .voi-pill--vat{
    border-color: rgba(245,124,0,.35);
    background: rgba(245,124,0,.10);
    }
    .voi-pill--total{
    border-color: rgba(25,118,210,.35);
    background: rgba(25,118,210,.10);
    }
    .voi-pill--total b{ font-weight: 900; }
    .voi-chiprow{
    display:flex;
    gap:6px;
    flex-wrap:nowrap;
    overflow-x:auto;
    -webkit-overflow-scrolling: touch;
    margin-top:8px;
    padding-bottom:4px;
    }
    .voi-chiprow::-webkit-scrollbar{ height:6px; }
    .voi-chiprow::-webkit-scrollbar-thumb{ background: rgba(49,51,63,.25); border-radius:999px; }
    </style>
    

    <style>
        .voi-card{border:1px solid rgba(49,51,63,.12);border-radius:18px;padding:14px 14px;margin:10px 0;background:rgba(255,255,255,.03);}
        .voi-muted{opacity:.72;font-size:.9rem;}
        .voi-chip{display:inline-block;padding:6px 12px;border-radius:999px;border:1px solid rgba(49,51,63,.18);font-size:.85rem;}
        .voi-chip-ok{background:rgba(46,160,67,.12);border-color:rgba(46,160,67,.35);}
        .voi-chip-warn{background:rgba(255,159,10,.12);border-color:rgba(255,159,10,.35);}
        .voi-chip-bad{background:rgba(255,69,58,.12);border-color:rgba(255,69,58,.35);}
        .voi-divider{height:1px;background:rgba(49,51,63,.12);margin:14px 0;}
        .voi-alert{border:1px solid rgba(255,159,10,.35);background:rgba(255,159,10,.10);border-radius:14px;padding:8px 10px;margin-top:8px;display:flex;gap:10px;align-items:flex-start;}
        .voi-alert b{font-weight:800;}
        .voi-alert-icon{width:28px;height:28px;border-radius:999px;display:flex;align-items:center;justify-content:center;background:rgba(255,159,10,.18);border:1px solid rgba(255,159,10,.35);flex:0 0 28px;}
        .voi-alert-text{font-size:.9rem;line-height:1.25;}
    </style>
        """,
        unsafe_allow_html=True,
    )


def _chip_for_send(sent: bool, last_error: str) -> tuple[str, str]:
    if sent:
        return ("📧 Enviado", "voi-chip voi-chip-ok")
    if last_error:
        return ("⚠️ Error", "voi-chip voi-chip-bad")
    return ("⏳ Pendiente", "voi-chip voi-chip-warn")


def _render_header(order: Order) -> None:
    created_at = getattr(order, 'created_at', None)
    created_by = getattr(order, 'created_by', None) or '—'
    date_str = created_at.strftime('%Y-%m-%d %H:%M') if created_at else '—'
    st.caption(f"Creado: {date_str} · Por: {created_by}")
    
 


def _render_workflow_actions(*, venue_id: int, order: Order, role: Optional[str], actor: str) -> None:
    st.markdown("<div class='voi-divider'></div>", unsafe_allow_html=True)
    status = (_s(order.status)).lower()
    can_manage = (_s(role)).lower() in {"owner", "manager"}
    with st.container(horizontal=True):
        if status == "draft":
            if st.button("✅ Pasar a Cesta", type="primary", use_container_width=True):
                _set_order_status(int(order.id), "ready_to_send", actor)
                st.session_state.pop(f"orders_active_order_id_{venue_id}", None)
                st.session_state[f"orders_active_order_id_{venue_id}"] = int(order.id)
                _bump_refresh(venue_id)
                st.session_state["page"] = "orders"
                set_query_params(page="orders", order_id=str(int(order.id)))
                st.rerun()

        elif status == "ready_to_send":
            if st.button("↩️ Volver a Borrador", use_container_width=True):
                _set_order_status(int(order.id), "draft", actor)
                st.session_state.pop(f"orders_active_order_id_{venue_id}", None)
                st.session_state[f"borrador_active_order_id_{venue_id}"] = int(order.id)
                _bump_refresh(venue_id)
                st.session_state["page"] = "borrador"
                set_query_params(page="borrador", order_id=str(int(order.id)))
                st.rerun()

        if status == "pending_receive":
            if st.button("✅ Cerrar (Historial)", type="primary", use_container_width=True):
                _set_order_status(int(order.id), "final", actor); _bump_refresh(venue_id); st.rerun()


def _render_lines_editor(*, venue_id: int, order: Order, actor: str, products: list[Product], lines: list[OrderLine]) -> None:

    # Use cached product index (major speedup on reruns)
    products, products_by_id, label_by_id, cat_by_pid, prov_by_pid, all_categories, all_providers, base_pids = _product_ui_index_cached(get_session, venue_id)
    editor_key = f"order_editor_{int(order.id)}"
    df_state_key = f"{editor_key}__df"
    
    # -----------------------------
    # -----------------------------
    # ✅ External refresh detection
    # If DB changed outside this editor (e.g., new_order_tab added lines),
    # drop local editor buffer and rebuild from DB truth.
    # -----------------------------
    refresh_sig_key = f"{editor_key}__refresh_sig"
    cur_sig = (int(order.id), int(st.session_state.get(f"orders_refresh_token_{venue_id}", 0)))

    if st.session_state.get(refresh_sig_key) != cur_sig:
        st.session_state[refresh_sig_key] = cur_sig
        st.session_state.pop(df_state_key, None)   # ✅ this is the key one



    if df_state_key not in st.session_state:
        st.session_state[df_state_key] = _sanitize_editor_df(_editor_df_from_lines(lines, products_by_id))
    else:
        st.session_state[df_state_key] = _sanitize_editor_df(st.session_state[df_state_key])


    
#     # -----------------------------
#     # Quick add expander (cascading filters + paging + toggle)
#     # -----------------------------

#     # (categories/providers maps come from _product_ui_index_cached)

#     def _qty_by_product(df: pd.DataFrame) -> dict[int, float]:
#         df = _sanitize_editor_df(df)
#         df2 = df.loc[df["delete"] != True].copy()  # noqa: E712
#         df2 = df2.loc[df2["product_id"].notna()].copy()
#         out: dict[int, float] = {}
#         for _, r in df2.iterrows():
#             pid = _pid_to_int(r.get("product_id"))
#             if pid is None:
#                 continue
#             out[int(pid)] = float(out.get(int(pid), 0.0) or 0.0) + float(r.get("quantity") or 0.0)
#         return out

#     # NOTE: Resetting number_input inside st.form can be tricky because widget state
#     # is sticky across reruns. A reliable approach is to add a nonce to the widget
#     # key and bump it after submitting (forces Streamlit to recreate the widget).
#     qa_nonce_key = f"{editor_key}__qa_nonce"
#     st.session_state.setdefault(qa_nonce_key, 0)

#     def _add_product_to_df(pid: int, qty_val: float) -> None:
#         qty_val = float(qty_val or 0.0)
#         if qty_val <= 0:
#             return
#         df = _sanitize_editor_df(st.session_state[df_state_key])
#         mask_same = (df["product_id"] == int(pid)) & (df["delete"] != True)  # noqa: E712
#         if mask_same.any():
#             idx = df.index[mask_same][0]
#             df.at[idx, "quantity"] = float(df.at[idx, "quantity"] or 0.0) + qty_val
#         else:
#             unit = (_s(getattr(products_by_id.get(int(pid)), "unit", "")) or "unidad").lower()
#             df = pd.concat(
#                 [df, pd.DataFrame([{"line_id": pd.NA, "product_id": int(pid), "quantity": qty_val, "unit": unit, "delete": False}])],
#                 ignore_index=True,
#             )
#         st.session_state[df_state_key] = _sanitize_editor_df(df)
#         # Force qty inputs to reset to 0.0
#         st.session_state[qa_nonce_key] = int(st.session_state.get(qa_nonce_key, 0) or 0) + 1
#         st.rerun()
        

   
#     # Backwards compatible cleanup (older sessions may still carry legacy keys)
#     legacy_reset_flag = f"{editor_key}__qa_reset_qty"
#     if st.session_state.get(legacy_reset_flag):
#         qty_prefix = f"{editor_key}__qa_qty_"
#         for k in list(st.session_state.keys()):
#             if isinstance(k, str) and k.startswith(qty_prefix):
#                 st.session_state.pop(k, None)
#         st.session_state.pop(legacy_reset_flag, None)

#     q = st.text_input(
#             "Buscar",
#             key=f"{editor_key}__qa_search",
#             placeholder="Producto…",
#         ).strip().lower()
    
#     with st.container(horizontal=True):


#         # Prepare order state for dependent options
#         df_current = _sanitize_editor_df(st.session_state[df_state_key])
#         qty_by_pid = _qty_by_product(df_current)

#         base_pids = sorted(label_by_id.keys())
#         if q:
#             base_pids = [pid for pid in base_pids if q in label_by_id.get(pid, "").lower()]

#         # Session keys
#         cat_key = f"{editor_key}__qa_cat"
#         prov_key = f"{editor_key}__qa_prov"
#         hide_key = f"{editor_key}__qa_hide"

#         st.session_state.setdefault(cat_key, "Todas")
#         st.session_state.setdefault(prov_key, "Todos")
        
        
#         hide_in_order = bool(st.session_state.get(hide_key, False))

#         current_cat = st.session_state[cat_key]
#         current_prov = st.session_state[prov_key]

#         # --- Compute cascading options ---
#         cats_for_prov = (
#             sorted({cat_by_pid.get(pid, "") for pid in base_pids if prov_by_pid.get(pid, "") == current_prov})
#             if current_prov != "Todos"
#             else sorted({cat_by_pid.get(pid, "") for pid in base_pids})
#         )
#         cats_for_prov = [c for c in cats_for_prov if c]

#         provs_for_cat = (
#             sorted({prov_by_pid.get(pid, "") for pid in base_pids if cat_by_pid.get(pid, "") == current_cat})
#             if current_cat != "Todas"
#             else sorted({prov_by_pid.get(pid, "") for pid in base_pids})
#         )
#         provs_for_cat = [p for p in provs_for_cat if p]

#         # Reset invalid selections
#         if current_cat != "Todas" and current_cat not in cats_for_prov:
#             st.session_state[cat_key] = "Todas"
#             current_cat = "Todas"

#         if current_prov != "Todos" and current_prov not in provs_for_cat:
#             st.session_state[prov_key] = "Todos"
#             current_prov = "Todos"


#         cat_tabs = ["Todas"] + cats_for_prov
#         cat_tab_objs = st.tabs(cat_tabs)

#         selected_cat = "Todas"
#         for i, tab in enumerate(cat_tab_objs):
#             with tab:
#                 selected_cat = cat_tabs[i]
#                 break


    
#         # Re-evaluate providers after category selection
#         prov_opts = (
#             sorted({prov_by_pid.get(pid, "") for pid in base_pids if cat_by_pid.get(pid, "") == selected_cat})
#             if selected_cat != "Todas"
#             else sorted({prov_by_pid.get(pid, "") for pid in base_pids})
#         )
#         prov_opts = [p for p in prov_opts if p]

#         prov_tabs = ["Todos"] + prov_opts
#         prov_tab_objs = st.tabs(prov_tabs)

#         selected_prov = "Todos"
#         for i, tab in enumerate(prov_tab_objs):
#             with tab:
#                 selected_prov = prov_tabs[i]
#                 break


#         # ---------- Reset paging when filters change ----------
#         filters_sig = (q, selected_cat, selected_prov, bool(hide_in_order))
#         sig_key = f"{editor_key}__qa_filters_sig"
#         page_key = f"{editor_key}__qa_page"

#         if st.session_state.get(sig_key) != filters_sig:
#             st.session_state[sig_key] = filters_sig
#             st.session_state[page_key] = 1

#         # ---------- Apply filters ----------
#         pids = base_pids

#         if selected_cat != "Todas":
#             pids = [pid for pid in pids if cat_by_pid.get(pid, "") == selected_cat]

#         if selected_prov != "Todos":
#             pids = [pid for pid in pids if prov_by_pid.get(pid, "") == selected_prov]

#         if hide_in_order:
#             pids = [pid for pid in pids if float(qty_by_pid.get(pid, 0.0) or 0.0) <= 0.0]

#         total = len(pids)

#     # ---------- Paging ----------

#     page_size = 60

#     st.session_state.setdefault(page_key, 1)
#     total_pages = max(1, (total + page_size - 1) // page_size)
#     st.session_state[page_key] = min(st.session_state[page_key], total_pages)

#     with st.container(horizontal=True):
        
#         if st.button("⬅️", disabled=st.session_state[page_key] <= 1):
#             st.session_state[page_key] -= 1
#             st.rerun()


#         if st.button("➡️", disabled=st.session_state[page_key] >= total_pages):
#             st.session_state[page_key] += 1
#             st.rerun()


#         st.caption(f"{total} resultados · Página {st.session_state[page_key]} / {total_pages}")
        
#         hide_in_order = st.toggle(
#             "Ocultar en pedido",
#             key=hide_key,
#         )

# # ---------- Grid ----------
#     # ---------- Grid ----------
#     start_i = (st.session_state[page_key] - 1) * page_size
#     end_i = start_i + page_size
#     pids_page = pids[start_i:end_i]

#     # --- helper: session-state key for per-product draft qty ---
#     def ss_qty_key(editor_key, pid):
#         return f"{editor_key}__qa_qty_live_{pid}"

#     # --- Global CSS (once) ---
#     st.markdown("""
#     <style>
#     @media (max-width: 700px) {
#     .st-key-my_blue_container [data-testid="stHorizontalBlock"]{
#         display:flex !important;
#         flex-wrap: wrap !important;
#         gap: 12px !important;
#     }
#     .st-key-my_blue_container [data-testid="column"]{
#         flex: 0 0 calc(50% - 12px) !important;
#         width: calc(50% - 12px) !important;
#         max-width: calc(50% - 12px) !important;
#         min-width: 0 !important;
#     }

#     /* Make widgets shrink properly inside columns */
#     .st-key-my_blue_container [data-testid="stNumberInput"],
#     .st-key-my_blue_container [data-testid="stButton"]{
#         width: 100% !important;
#         min-width: 0 !important;
#     }
#     }
#     </style>
#     """, unsafe_allow_html=True)

#     def render_product_card(pid):
#         p = products_by_id.get(pid)
#         if not p:
#             return

#         label = label_by_id.get(pid, str(pid))
#         parts = label.split(" — ", 1)
#         name = parts[0].strip()

#         # ✅ new: description + unit (best-effort)
#         desc = (getattr(p, "description", "") or "").strip()
#         unit = (getattr(p, "unit", "") or "").strip()
#         provider = (getattr(p, "provider_name", "") or "").strip()


#         existing_qty = float(qty_by_pid.get(pid, 0.0) or 0.0)
#         in_order = existing_qty > 0
#         qty_txt = f"{existing_qty:g}"

#         k_qty = ss_qty_key(editor_key, pid)
#         st.session_state.setdefault(k_qty, 0)

#         card_key = f"my_product_{pid}"
#         pill_key = f"{card_key}_pill"

#         # ✅ tweak these to your taste
#         PILL_SIDE_PAD = 12
#         PILL_BOTTOM_PAD = 12
#         CARD_PAD_BOTTOM = 86  # space reserved so pill never overlaps content

#         st.markdown(
#             f"""
#     <style>
#     /* --- Card --- */
#     .st-key-{card_key} {{
#     background:#fff;
#     border:3px solid rgba(0,0,0,.90);
#     border-radius:40px;
#     padding: 6px 6px {CARD_PAD_BOTTOM}px 6px;  /* ✅ reserve space for pill INSIDE */
#     margin-bottom:18px;
#     position:relative;
#     overflow:hidden; /* ✅ keeps pill clipped inside rounded corners */
#     }}

#     /* Name */
#     .st-key-{card_key} .product-name {{
#     font-weight:900;
#     font-size:1.05rem;
#     line-height:1.2;
#     text-align:center;
#     padding: 24px 10px 6px 10px;
#     min-height: 92px;
#     display:flex;
#     align-items:center;
#     justify-content:center;
#     }}
    
#     /* Provider name (small, subtle) */
#     .st-key-{card_key} .product-provider{{
#     text-align: center;
#     font-size: .78rem;
#     font-weight: 800;
#     letter-spacing: .4px;
#     text-transform: uppercase;
#     opacity: .65;
#     margin-top: -6px;
#     margin-bottom: 6px;
#     }}


#     /* ✅ NEW: description + unit (small + muted) */
#     .st-key-{card_key} .product-meta {{
#     text-align:center;
#     margin-top:-4px;
#     padding: 0 14px 6px 14px;
#     }}
#     .st-key-{card_key} .product-desc {{
#     font-size:.78rem;
#     line-height:1.15;
#     opacity:.55;
#     font-weight:700;
#     }}
#     .st-key-{card_key} .product-unit {{
#     font-size:.74rem;
#     opacity:.45;
#     font-weight:800;
#     margin-top:2px;
#     }}

#     /* Badge */
#     .st-key-{card_key} .badge {{
#     position:absolute;
#     top:12px;
#     right:12px;
#     background: rgba(33,150,243,.95);
#     color:white;
#     font-weight:900;
#     font-size:.72rem;
#     padding: 2px 10px;
#     border-radius:999px;
#     z-index:3;
#     }}

#     /* --- Pill INSIDE card (absolute) --- */
#     .st-key-{pill_key} {{
#     position:absolute;
#     left:{PILL_SIDE_PAD}px;
#     right:{PILL_SIDE_PAD}px;
#     bottom:{PILL_BOTTOM_PAD}px;

#     background:#f3f4f6;
#     border:2px solid rgba(0,0,0,.18);
#     border-radius:45px;
#     padding: 10px 12px;
#     box-shadow: 0 10px 24px rgba(0,0,0,.10);
#     z-index:2;
#     }}

#     /* Make pill widgets align nicely */
#     .st-key-{pill_key} [data-testid="stHorizontalBlock"] {{
#     align-items:center;
#     gap:10px;
#     }}

#     /* Number input: keep compact */
#     .st-key-{pill_key} [data-testid="stNumberInput"] {{
#     max-width:110px;
#     }}
#     .st-key-{pill_key} input {{
#     text-align:center;
#     font-weight:900;
#     }}

#     /* Button: full height inside pill */
#     .st-key-{pill_key} [data-testid="stButton"] button {{
#     border-radius:14px;
#     font-weight:900;
#     }}
#     </style>
#     """,
#             unsafe_allow_html=True,
#         )

#         with st.container(key=card_key):
#             if in_order:
#                 st.markdown(f"<div class='badge'>✓ {qty_txt}</div>", unsafe_allow_html=True)

#             st.markdown(f"<div class='product-name'>{name}</div>", unsafe_allow_html=True)
            
#             if provider:
#                 st.markdown(f"<div class='product-provider'>{provider}</div>", unsafe_allow_html=True)

#             # ✅ NEW meta lines
#             meta_html = "<div class='product-meta'>"
#             if desc:
#                 meta_html += f"<div class='product-desc'>{desc}</div>"
#             if unit:
#                 meta_html += f"<div class='product-unit'>{unit}</div>"
#             meta_html += "</div>"
#             st.markdown(meta_html, unsafe_allow_html=True)

#             # Pill UI (inside card, anchored at bottom)
#             # Pill UI (inside card, anchored at bottom)
#             with st.container(key=pill_key, horizontal=True, gap="small"):

#             # - button (only changes draft qty in session, NO DB)
    
#                 if st.button("−", key=f"minus_{pid}", use_container_width=True):
#                     st.session_state[k_qty] = max(0, int(st.session_state[k_qty]) - 1)

#             # qty display (draft)
        
#                 st.markdown(
#                     f"<div style='text-align:center;font-weight:900;font-size:1.05rem;padding-top:8px;'>"
#                     f"{int(st.session_state[k_qty])}</div>",
#                     unsafe_allow_html=True
#                 )

#             # + button (only changes draft qty in session, NO DB)
        
#                 if st.button("+", key=f"plus_{pid}", use_container_width=True):
#                     st.session_state[k_qty] = int(st.session_state[k_qty]) + 1

#             # ✅ Commit button (THIS is the only place DB updates happen)
        
#                 action_label = "Sumar" if in_order else "Añadir"
#                 if st.button(action_label, key=f"add_{pid}", use_container_width=True, type="primary"):
#                     qty_val = int(st.session_state[k_qty])
#                     if qty_val > 0:
#                         _add_product_to_df(pid, qty_val)  # ✅ DB write ONLY here
#                         st.session_state[k_qty] = 0
#                         st.rerun()




#                 st.markdown("</div>", unsafe_allow_html=True)



#     st.markdown("""
#     <style>
#     /* Force 2 columns on mobile Safari inside this container */
#     @media (max-width: 900px) {

#     /* The row wrapper that holds columns */
#     .st-key-my_blue_container [data-testid="stHorizontalBlock"],
#     .st-key-my_blue_container div[data-testid="stHorizontalBlock"]{
#         display: flex !important;
#         flex-wrap: wrap !important;
#         gap: 10px !important;
#     }

#     /* Columns (Streamlit has used different testids across versions) */
#     .st-key-my_blue_container [data-testid="column"],
#     .st-key-my_blue_container [data-testid="stColumn"],
#     .st-key-my_blue_container div[data-testid="column"],
#     .st-key-my_blue_container div[data-testid="stColumn"]{
#         flex: 0 0 calc(50% - 10px) !important;
#         width: calc(50% - 10px) !important;
#         max-width: calc(50% - 10px) !important;
#         min-width: 0 !important;
#     }

#     /* Prevent widgets from imposing min-width that breaks the column */
#     .st-key-my_blue_container *{
#         min-width: 0 !important;
#     }
#     }
#     </style>
#     """, unsafe_allow_html=True)

#     with st.container(border=True, key="my_blue_container", height=800):
#         # 2 cards per row
#         for i in range(0, len(pids_page), 2):
#             col1, col2 = st.columns(2, gap="small")

#             pid1 = pids_page[i]
#             with col1:
#                 render_product_card(pid1)

#             if i + 1 < len(pids_page):
#                 pid2 = pids_page[i + 1]
#                 with col2:
#                     render_product_card(pid2)

 
    df_for_editor = _sanitize_editor_df(st.session_state[df_state_key])

    def _unit_for_pid(pid: Any) -> str:
        pid_i = _pid_to_int(pid)
        p = products_by_id.get(pid_i) if pid_i is not None else None
        return (_s(getattr(p, "unit", "")) or "unidad").lower() if p else "unidad"

    form_key = f"{editor_key}__form"

    with st.form(key=form_key, clear_on_submit=False):
        edited = st.data_editor(
            df_for_editor,
            hide_index=True,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "line_id": None,
                "product_id": st.column_config.SelectboxColumn(
                    "Producto",
                    options=sorted(label_by_id.keys()),
                    format_func=lambda pid: label_by_id.get(_pid_to_int(pid) or -1, str(pid)),
                    required=True,
                    disabled=True,
                    width="large",
                ),
                "quantity": st.column_config.NumberColumn("Qty", min_value=0, step=1, width="small"),
                "unit": st.column_config.TextColumn("Unidad", disabled=True, width="small"),
                "delete": st.column_config.CheckboxColumn("🗑️", width="small"),
            },
            key=editor_key,  # editor state lives here
        )


        with st.container(horizontal=True):
            guardar = st.form_submit_button("💾 Guardar", type="primary", use_container_width=True)

            eliminar = st.form_submit_button("🗑️ Eliminar", use_container_width=True)


    # -----------------------------
    # --- Handle submits (ONLY runs when one of the form buttons is clicked) ---
    if guardar or eliminar:
        if guardar:
            # Read whatever is currently in the editor, sanitize + recompute unit
            edited = _sanitize_editor_df(edited)
            edited["unit"] = edited["product_id"].map(_unit_for_pid)
            edited = _sanitize_editor_df(edited)

            # ✅ Only here we persist + save to DB
            st.session_state[df_state_key] = edited

            _save_lines_from_editor(
                venue_id=venue_id,
                order_id=int(order.id),
                actor=actor,
                df=st.session_state[df_state_key],
                products_by_id=products_by_id,
            )

            # Clean editor + quick-add widget state so the UI comes back "fresh".
            st.session_state.pop(df_state_key, None)
            st.session_state.pop(editor_key, None)

            qa_nonce_key = f"{editor_key}__qa_nonce"
            st.session_state[qa_nonce_key] = int(st.session_state.get(qa_nonce_key, 0) or 0) + 1

            _bump_refresh(venue_id)
            st.success("Guardado ✓")
            st.rerun()

        elif eliminar:
            _delete_order(int(order.id))
            _bump_refresh(venue_id)
            st.session_state.pop(f"borrador_active_order_id_{venue_id}", None)
            st.session_state.pop(df_state_key, None)
            st.session_state.pop(editor_key, None)
            st.rerun()


def _render_send_section(*, venue_id: int, order: Order, products: list[Product], lines: list[OrderLine], actor: str) -> None:
    v = _load_venue_templates(venue_id, _refresh_token(venue_id))
    missing_required = _venue_missing_required(v)
    if missing_required:
        st.error("Faltan campos obligatorios del local para enviar:\n" + "\n".join([f"- {x}" for x in missing_required]))
        return

    products_by_id = {int(p.id): p for p in products if p.id is not None}
    grouped = _group_lines_by_provider(lines, products_by_id)
    if not grouped:
        st.info("No hay líneas para enviar.")
        return

    st.markdown("### Summary")

    with st.container(horizontal=True, gap="small"):
        # cA, cC, cD = st.columns([1.15, 1.2, 1.6], vertical_alignment="center")
        # with cA:
        show_prices = st.toggle(
            "Mostrar importes",
            value=False,
            key=f"sum_show_prices_{int(order.id)}",
            help="Estimación basada en precios del catálogo y reglas de descuento. No es una factura.",
        )
        include_iva = show_prices
        # with cC:
        compact = st.toggle(
            "Compacto",
            value=True,
            key=f"sum_compact_{int(order.id)}",
            help="Mejor en móvil: tarjetas.",
        )
    # with cD:
        sum_mode = st.radio(
            "Resumen",
            options=["por_proveedor", "total"],
            format_func=lambda x: "Por proveedor" if x == "por_proveedor" else "Total",
            horizontal=True,
            label_visibility="collapsed",
            key=f"sum_mode_{int(order.id)}",
        )

        apply_smart_prices = st.toggle(
            "🧠 Aplicar precios inteligentes",
            value=False,
            key=f"sum_apply_smart_{int(order.id)}",
            disabled=not show_prices,
            help="Reasigna automáticamente cada línea al proveedor más barato (incluyendo descuentos) para enviar y para que el link del proveedor funcione sin pasos extra.",
        )

    # ---- pricing helpers (optional) ----
    providers_by_name = _providers_cached(get_session, venue_id)

    # ---- Smart price opportunities (best alternative per line) ----
    smart_best_by_pid: dict[int, dict[str, Any]] = {}
    smart_current_net_by_pid: dict[int, float] = {}

    if show_prices:
        df_smart = _sanitize_editor_df(_editor_df_from_lines(lines, products_by_id))

        try:
            smart_sugg = _smart_cesta_suggestions(
                venue_id=int(venue_id),
                providers_by_name=providers_by_name,
                products_by_id=products_by_id,
                draft_df=df_smart,
                min_rel=0.45,
                min_saving_pct=0.0,  # show any cheaper option
                top_k=1,
            )
        except Exception:
            smart_sugg = []

        for it in smart_sugg or []:
            cur_pid = int(it.get("line_product_id") or 0)
            if not cur_pid or not it.get("suggestions"):
                continue
            best = it["suggestions"][0]
            smart_best_by_pid[cur_pid] = best
            smart_current_net_by_pid[cur_pid] = float(it.get("current_net") or 0.0)

    # Shared discount engine (also used by Smart Cesta in Draft)
    def _pricing_for_line(provider_name: str, pid: Optional[int], qty: float, gross_unit: float) -> dict[str, Any]:
        return _pricing_for_line_global(
            venue_id=int(venue_id),
            providers_by_name=providers_by_name,
            provider_name=provider_name,
            pid=pid,
            qty=qty,
            gross_unit=gross_unit,
        )

    def _price_for_pid(pid: Optional[int]) -> float:
        if pid is None:
            return 0.0
        p = products_by_id.get(pid)
        return _safe_float(getattr(p, "price", 0.0) if p else 0.0, 0.0)

    def _iva_pct_for_pid(pid: Optional[int], default_pct: float = 21.0) -> float:
        if pid is None:
            return float(default_pct)
        p = products_by_id.get(pid)
        if not p:
            return float(default_pct)
        v2 = _safe_float(getattr(p, "iva", None), float(default_pct))
        return float(default_pct) if v2 <= 0 else float(v2)

    # --- label helpers: show description without repeating name words ---
    def _strip_accents(s: str) -> str:
        s = unicodedata.normalize("NFD", s or "")
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
        return unicodedata.normalize("NFC", s)

    def _norm_words(s: str) -> list[str]:
        raw = re.findall(r"[0-9]+|[^\W_]+", (s or ""), flags=re.UNICODE)
        return [_strip_accents(w).lower() for w in raw if w.strip()]

    def _remove_name_words_from_description(name: str, desc: str) -> str:
        name_set = set(_norm_words(name))
        if not name_set:
            return (desc or "").strip()
        desc_raw = re.findall(r"[0-9]+|[^\W_]+", (desc or ""), flags=re.UNICODE)
        kept: list[str] = []
        for w in desc_raw:
            if _strip_accents(w).lower() not in name_set:
                kept.append(w)
        return " ".join(kept).strip()

    def _product_label(pid: Optional[int], fallback_name: str = "Producto") -> str:
        if pid is None:
            return _s(fallback_name) or "Producto"
        p = products_by_id.get(pid)
        if not p:
            return _s(fallback_name) or "Producto"
        name = _s(getattr(p, "name", "")) or "Producto"
        desc = _s(getattr(p, "description", "")) or ""
        desc_clean = _remove_name_words_from_description(name, desc)
        return f"{name} — {desc_clean}" if desc_clean else name

    def _summary_rows(provider_name: str, prov_lines: list[dict[str, Any]], *, apply_smart: bool) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for it in prov_lines:
            qty = _safe_float(it.get("qty"), 0.0)
            if qty <= 0:
                continue
            pid = _pid_to_int(it.get("product_id"))
            prod = products_by_id.get(pid) if pid else None
            unit = (_s(getattr(prod, "unit", "")) if prod else _s(it.get("unit", ""))) or "unidad"

            label = _product_label(pid, _s(it.get("name") or "Producto"))

            row: dict[str, Any] = {
                "Proveedor": provider_name,
                "Label": label,
                "Qty": qty,
                "Unidad": unit,
            }

            if show_prices:
                gross_unit = _price_for_pid(pid)
                pricing = (
                    _pricing_for_line(provider_name, pid, qty, gross_unit)
                    if gross_unit > 0
                    else {"net_unit": 0.0, "discount_pct": 0.0, "rule_kind": "", "applied": False}
                )

                base_net_unit = float(pricing.get("net_unit", gross_unit) or 0.0)
                base_disc_pct = float(pricing.get("discount_pct", 0.0) or 0.0)

                net_unit = base_net_unit
                disc_pct = base_disc_pct
                gross_for_ahorro = gross_unit

                smart = smart_best_by_pid.get(int(pid)) if pid is not None else None
                smart_available = False
                smart_applied = False
                smart_saving = 0.0
                smart_to_provider = ""
                smart_to_pid: Optional[int] = None

                if pid is not None and smart:
                    best_net = float(smart.get("net_unit") or 0.0)
                    if best_net > 0 and best_net + 1e-9 < base_net_unit:
                        smart_available = True
                        smart_saving = qty * (base_net_unit - best_net)
                        smart_to_provider = _s(smart.get("provider") or "")
                        smart_to_pid = _pid_to_int(smart.get("product_id"))

                        if apply_smart:
                            smart_applied = True
                            net_unit = best_net
                            disc_pct = float(smart.get("discount_pct") or 0.0)
                            gross_for_ahorro = float(smart.get("gross_unit") or 0.0) or gross_unit

                            if smart_to_provider:
                                row["Proveedor"] = smart_to_provider
                            if smart_to_pid is not None:
                                row["Label"] = _product_label(smart_to_pid, row.get("Label") or "Producto")

                amount = qty * net_unit
                ahorro = qty * max(0.0, (gross_for_ahorro - net_unit))

                best = smart_best_by_pid.get(pid)
                smart_available = bool(best)

                sug_name = ""
                sug_desc = ""
                if smart_available:
                    tpid_raw = best.get("product_id")
                    try:
                        tpid = int(tpid_raw) if tpid_raw is not None else None
                    except Exception:
                        tpid = None
                    if tpid is not None:
                        p2 = products_by_id.get(tpid)
                        if p2 is not None:
                            sug_name = _s(getattr(p2, "name", "")) or ""
                            sug_desc = _s(getattr(p2, "description", "")) or ""

                row.update({
                    "Precio": net_unit,
                    "Desc.%": disc_pct if disc_pct > 0 else 0.0,
                    "Importe": amount,
                    "Ahorro": ahorro,

                    "Smart disponible": bool(smart_available),
                    "Smart aplicado": bool(smart_applied) if smart_available else False,
                    "Ahorro smart": float(smart_saving) if smart_available else 0.0,
                    "Smart proveedor": (_s(best.get("provider")) if smart_available else ""),

                    "Smart producto nombre": sug_name if smart_available else "",
                    "Smart producto descripción": sug_desc if smart_available else "",
                })

                if include_iva:
                    pid_for_iva = smart_to_pid if (smart_applied and smart_to_pid is not None) else pid
                    iva_pct = _iva_pct_for_pid(pid_for_iva, 21.0)
                    iva_eur = amount * (iva_pct / 100.0)
                    row.update({
                        "% IVA": iva_pct,
                        "IVA (€)": iva_eur,
                        "Total": amount + iva_eur,
                    })

            rows.append(row)

        return rows

    def _render_rows(rows: list[dict[str, Any]]) -> None:
        if not rows:
            st.info("No hay líneas para resumir.")
            return

        if compact:
            for r in rows:
                label = _s(r.get("Label"))
                provider = _s(r.get("Proveedor") or r.get("Provider"))
                qty_txt = f"{float(r.get('Qty') or 0.0):g} {_s(r.get('Unidad'))}"

                if " — " in label:
                    name_part, desc_part = label.split(" — ", 1)
                else:
                    name_part, desc_part = label, ""

                suffix_parts: list[str] = []
                if desc_part:
                    suffix_parts.append(f"<span style='font-size:.78rem;color:#64748b;font-weight:400'>{desc_part}</span>")
                if provider and sum_mode != "por_proveedor":
                    suffix_parts.append(f"<span style='font-size:.75rem;color:#94a3b8;font-weight:500'>{provider}</span>")
                suffix_html = (" <span style='color:#cbd5e1;font-size:.75rem'>·</span> ".join(suffix_parts))
                label_line = f"<span style='font-weight:850'>{name_part}</span>"
                if suffix_html:
                    label_line += f" <span style='color:#cbd5e1'>·</span> {suffix_html}"

                top = (
                    "<div style=\"display:flex;gap:10px;justify-content:space-between;align-items:baseline;flex-wrap:wrap\">"
                    f"<div style=\"flex:1;min-width:180px;line-height:1.4\">{label_line}</div>"
                    f"<div style=\"font-weight:850;white-space:nowrap\">{qty_txt}</div>"
                    "</div>"
                )

                chips = ""
                alert = ""

                # 🧠 Smart price alert
                if bool(r.get("Smart disponible")):
                    s_save = float(r.get("Ahorro smart") or 0.0)

                    if bool(r.get("Smart aplicado")):
                        alert = (
                            "<div class='voi-alert' style='border-color:rgba(46,160,67,.35);background:rgba(46,160,67,.10)'>"
                            "<div class='voi-alert-icon' style='background:rgba(46,160,67,.18);border-color:rgba(46,160,67,.35)'>🧠</div>"
                            "<div class='voi-alert-text'><b>Smart aplicado</b><br/>"
                            "Precio optimizado antes de enviar.</div>"
                            "</div>"
                        )
                    else:
                        if s_save > 0.005:
                            sp = _s(r.get("Smart proveedor") or "")
                            vv = f"{s_save:,.2f}"

                            prod_txt = _smart_product_label(
                                r.get("Smart producto nombre") or "",
                                r.get("Smart producto descripción") or "",
                            )

                            alert = (
                                "<div class='voi-alert'>"
                                "<div class='voi-alert-icon'>🏷️</div>"
                                "<div class='voi-alert-text'>"
                                f"<b>Ahorra {vv}€</b><br/>"
                                f"{prod_txt} · {sp}</div>"
                                "</div>"
                            )

                # Price chips
                if show_prices:
                    chips_items = [
                        ("Precio", float(r.get("Precio") or 0.0)),
                        ("Desc%", float(r.get("Desc.%") or 0.0)),
                        ("Importe", float(r.get("Importe") or 0.0)),
                        ("Ahorro", float(r.get("Ahorro") or 0.0)),
                    ]
                    if include_iva:
                        chips_items += [
                            ("IVA%", float(r.get("% IVA") or 0.0)),
                            ("IVA€", float(r.get("IVA (€)") or 0.0)),
                            ("Tot", float(r.get("Total") or 0.0)),
                        ]

                    def _chip(k: str, v: float) -> str:
                        if k in ("Desc%", "IVA%"):
                            vv = f"{v:g}%"
                        else:
                            vv = f"{v:,.2f}"

                        cls = "voi-pill"
                        if k == "Ahorro":
                            cls += " voi-pill--save"
                        elif k in ("IVA%", "IVA€"):
                            cls += " voi-pill--vat"
                        elif k == "Tot":
                            cls += " voi-pill--total"

                        return f"<span class='{cls}'>{k}: <b>{vv}</b></span>"

                    chips = "<div class='voi-chiprow'>" + "".join(_chip(k, v) for k, v in chips_items) + "</div>"

                st.markdown(
                    "<div class=\"voi-card\" style=\"padding:10px 12px\">"
                    f"{top}"
                    f"{alert}"
                    f"{chips}"
                    "</div>",
                    unsafe_allow_html=True,
                )
        else:
            df = pd.DataFrame(rows)
            base_cols = ["Proveedor", "Label", "Qty", "Unidad"]
            if show_prices:
                base_cols += ["Precio", "Desc.%", "Importe", "Ahorro"]
                if include_iva:
                    base_cols += ["% IVA", "IVA (€)", "Total"]
            cols = [c for c in base_cols if c in df.columns] + [c for c in df.columns if c not in base_cols]
            df = df[cols]
            st.dataframe(df, use_container_width=True, hide_index=True)

    def _totals(rows: list[dict[str, Any]]) -> tuple[float, float, float, float]:
        subtotal = sum(float(r.get("Importe") or 0.0) for r in rows)
        ahorro = sum(float(r.get("Ahorro") or 0.0) for r in rows)
        iva_eur = sum(float(r.get("IVA (€)") or 0.0) for r in rows) if include_iva else 0.0
        total = sum(float(r.get("Total") or 0.0) for r in rows) if include_iva else subtotal
        return subtotal, ahorro, iva_eur, total

    # Build baseline rows (current providers) and effective rows (optionally smart-applied)
    rows_all_base: list[dict[str, Any]] = []
    for prov, prov_lines in grouped.items():
        rows_all_base.extend(_summary_rows(prov, prov_lines, apply_smart=False))

    rows_all_eff: list[dict[str, Any]] = []
    for prov, prov_lines in grouped.items():
        rows_all_eff.extend(_summary_rows(prov, prov_lines, apply_smart=bool(apply_smart_prices)))

    def _money_saved(base_rows: list[dict[str, Any]], eff_rows: list[dict[str, Any]]) -> tuple[float, float]:
        base_sub = sum(float(r.get("Importe") or 0.0) for r in base_rows)
        eff_sub = sum(float(r.get("Importe") or 0.0) for r in eff_rows)
        base_tot = sum(float(r.get("Total") or 0.0) for r in base_rows) if include_iva else base_sub
        eff_tot = sum(float(r.get("Total") or 0.0) for r in eff_rows) if include_iva else eff_sub
        return max(0.0, base_sub - eff_sub), max(0.0, base_tot - eff_tot)

    if sum_mode == "total":
        _render_rows(rows_all_eff)

        if show_prices and rows_all_eff:
            subtotal, ahorro, iva_eur, total = _totals(rows_all_eff)
            saved_net, saved_total = _money_saved(rows_all_base, rows_all_eff)

            st.markdown("### Totales")
            if include_iva:
                st.markdown(
                    f"**Subtotal (neto):** {subtotal:,.2f} · **Ahorro (descuentos):** {ahorro:,.2f} · "
                    f"**IVA:** {iva_eur:,.2f} · **Total:** {total:,.2f}"
                )
                if apply_smart_prices and saved_total > 0:
                    st.markdown(f"🧠 **Ahorro smart aplicado:** **{saved_total:,.2f}€** (vs. precios actuales)")
                elif (not apply_smart_prices) and saved_total > 0:
                    st.markdown(f"⚡ **Ahorro smart potencial:** **{saved_total:,.2f}€** si aplicas precios inteligentes")
            else:
                st.markdown(f"**Total estimado (neto):** {subtotal:,.2f} · **Ahorro (descuentos):** {ahorro:,.2f}")
                if apply_smart_prices and saved_net > 0:
                    st.markdown(f"🧠 **Ahorro smart aplicado:** **{saved_net:,.2f}€** (vs. precios actuales)")
                elif (not apply_smart_prices) and saved_net > 0:
                    st.markdown(f"⚡ **Ahorro smart potencial:** **{saved_net:,.2f}€** si aplicas precios inteligentes")

            st.caption("Estimación: catálogo + reglas. La factura oficial del proveedor manda.")
        else:
            st.caption(f"{len(rows_all_eff)} líneas en total.")
    else:
        if apply_smart_prices:
            grouped_rows: dict[str, list[dict[str, Any]]] = {}
            for r in rows_all_eff:
                grouped_rows.setdefault(_s(r.get("Proveedor") or "—"), []).append(r)
        else:
            grouped_rows = {}
            for prov, prov_lines in grouped.items():
                grouped_rows[prov] = _summary_rows(prov, prov_lines, apply_smart=False)

        for prov in sorted(grouped_rows.keys()):
            rows = grouped_rows.get(prov) or []
            if not rows:
                continue
            with st.expander(f"{prov} · {len(rows)} líneas", expanded=False):
                _render_rows(rows)
                if show_prices:
                    subtotal, ahorro, iva_eur, total = _totals(rows)
                    if include_iva:
                        st.markdown(
                            f"**Subtotal:** {subtotal:,.2f} · **Ahorro:** {ahorro:,.2f} · "
                            f"**IVA:** {iva_eur:,.2f} · **Total:** {total:,.2f}"
                        )
                    else:
                        st.markdown(f"**Subtotal:** {subtotal:,.2f} · **Ahorro:** {ahorro:,.2f}")

    # -----------------------------
    # 🧠 Group lines for sending
    # -----------------------------
    def _group_lines_for_sending(*, apply_smart: bool) -> dict[str, list[dict[str, Any]]]:
        if not apply_smart:
            out: dict[str, list[dict[str, Any]]] = {}
            for prov, prov_lines in (grouped or {}).items():
                prov_norm = norm_provider(prov)
                out.setdefault(prov_norm, []).extend(prov_lines or [])
            for p in out:
                out[p] = sorted(out[p], key=lambda x: (_s(x.get("name")) or "").lower())
            return dict(sorted(out.items(), key=lambda kv: (kv[0] or "").lower()))

        out: dict[str, list[dict[str, Any]]] = {}

        for prov, prov_lines in (grouped or {}).items():
            for it in (prov_lines or []):
                qty = _safe_float(it.get("qty"), 0.0)
                if qty <= 0:
                    continue

                pid = _pid_to_int(it.get("product_id"))
                base_unit = (_s(it.get("unit")) or "unidad")
                base_name = _s(it.get("name") or "Producto")

                target_prov = prov
                target_pid = pid

                if pid is not None:
                    smart = smart_best_by_pid.get(int(pid))
                    base_net = float(smart_current_net_by_pid.get(int(pid)) or 0.0)
                    best_net = float(smart.get("net_unit") or 0.0) if smart else 0.0

                    if smart and base_net > 0 and best_net > 0 and (best_net + 1e-9) < base_net:
                        target_prov = _s(smart.get("provider") or prov) or prov
                        target_pid = _pid_to_int(smart.get("product_id")) or pid

                prod2 = products_by_id.get(int(target_pid)) if target_pid is not None else None
                name2 = _s(getattr(prod2, "name", "")) if prod2 is not None else base_name
                unit2 = (_s(getattr(prod2, "unit", "")) or base_unit) if prod2 is not None else base_unit

                prov_norm = norm_provider(target_prov)
                out.setdefault(prov_norm, []).append(
                    {
                        "line_id": int(it.get("line_id") or 0),
                        "product_id": target_pid,
                        "name": name2 or base_name,
                        "qty": float(qty),
                        "unit": (unit2 or "unidad"),
                    }
                )

        for p in out:
            out[p] = sorted(out[p], key=lambda x: (_s(x.get("name")) or "").lower())

        return dict(sorted(out.items(), key=lambda kv: (kv[0] or "").lower()))

    apply_smart_for_send = bool(apply_smart_prices)
    grouped_send = _group_lines_for_sending(apply_smart=apply_smart_for_send)

    # ✅ KEY FIX:
    # When smart sending is ON, we MUST materialize the smart provider/product
    # into OrderLine before sending so that the supplier link (seguimiento)
    # can actually find the products for that supplier.
    def _materialize_smart_to_db_for_send(*, provider_name: Optional[str] = None) -> int:
        if not apply_smart_for_send:
            return 0
        if not grouped_send:
            return 0

        target_map: dict[int, tuple[str, int, str]] = {}  # line_id -> (prov_norm, pid, unit)
        for prov_norm, prov_lines in grouped_send.items():
            if provider_name is not None and norm_provider(provider_name) != norm_provider(prov_norm):
                continue
            for it in (prov_lines or []):
                lid = int(it.get("line_id") or 0)
                pid = _pid_to_int(it.get("product_id"))
                if lid <= 0 or pid is None:
                    continue
                p2 = products_by_id.get(int(pid))
                unit2 = (_s(getattr(p2, "unit", "")) if p2 else _s(it.get("unit") or "")) or "unidad"
                target_map[lid] = (norm_provider(prov_norm), int(pid), unit2.lower())

        if not target_map:
            return 0

        applied = 0
        now = _now()
        with get_session() as s:
            for lid, (prov_norm, pid, unit2) in target_map.items():
                ol = s.get(OrderLine, int(lid))
                if not ol:
                    continue
                # update only if needed (minimizes writes)
                cur_pid = getattr(ol, "product_id", None)
                cur_prov = norm_provider(_s(getattr(ol, "provider", "")))
                cur_unit = (_s(getattr(ol, "unit", "")) or "").lower()

                needs = False
                if cur_pid is None or int(cur_pid) != int(pid):
                    needs = True
                if cur_prov != prov_norm:
                    needs = True
                if unit2 and cur_unit != unit2:
                    needs = True

                if not needs:
                    continue

                ol.product_id = int(pid)
                ol.provider = prov_norm
                ol.unit = unit2 or "unidad"
                ol.updated_at = now
                ol.updated_by = actor
                s.add(ol)
                applied += 1

            s.commit()

        return applied
    
    # ✅ Reload lines from DB after materializing smart changes
    def _reload_lines_from_db() -> list[OrderLine]:
        with get_session() as s:
            # Adjust field name if yours differs (most likely OrderLine.order_id)
            return (
                s.query(OrderLine)
                .filter(OrderLine.order_id == int(order.id))
                .all()
            )

    if apply_smart_for_send:
        st.info(
            "🧠 **Smart activo:** al enviar, el pedido se reasigna automáticamente (por línea) al proveedor más barato "
            "para que el email y el link del proveedor funcionen sin tocar nada en 'Cesta inteligente'."
        )

    st.markdown("<div class='voi-divider'></div>", unsafe_allow_html=True)

    provider_dir = _providers_cached(get_session, venue_id)
    send_map = _get_send_status_map(order_id=int(order.id))

    # Build once (used by sending message builder)
    products_by_id = {int(p.id): p for p in products if getattr(p, "id", None) is not None}


    # -----------------------------
    # Send settings (mobile-friendly)
    # -----------------------------
    with st.expander("⚙️ Envío: canales y CC", expanded=False):
        cset1, cset2 = st.columns([1.0, 1.0], vertical_alignment="center")
        with cset1:
            use_email = st.toggle("Email", value=True, key=f"use_email_{int(order.id)}")
        with cset2:
            use_wa = st.toggle("WhatsApp", value=False, key=f"use_wa_{int(order.id)}")

        wa_cc = st.text_input("Prefijo país (WhatsApp)", value="+34", key=f"wa_cc_{int(order.id)}")

        st.caption(f"CC: {v.email_cc or '—'} · BCC: {v.email_bcc or '—'}")

    # -----------------------------
    # Global actions
    # -----------------------------
    sent_count = sum(
        1
        for prov in grouped_send.keys()
        if bool(send_map.get(norm_provider(prov)) and getattr(send_map[norm_provider(prov)], "sent", False))
    )
    st.progress(sent_count / max(1, len(grouped_send)))
    st.caption(f"Enviados: {sent_count}/{len(grouped_send)}")

    g1, g2 = st.columns([1.6, 1.0], vertical_alignment="center")
    with g1:
        send_all_disabled = (not use_email)
        if st.button("🚀 Enviar a todos (Email)", type="primary", use_container_width=True, disabled=send_all_disabled):
            ok, fail = 0, 0

            # ✅ IMPORTANT: If smart is ON, persist to DB ONCE and rebuild sending groups
            if apply_smart_for_send:
                changed = _materialize_smart_to_db_for_send(provider_name=None)
                if changed:
                    lines = _reload_lines_from_db()
                    grouped = _group_lines_by_provider(lines, products_by_id)
                    grouped_send = _group_lines_for_sending(apply_smart=True)
                    _bump_refresh(venue_id)

            for prov, prov_lines in grouped_send.items():
                prov_norm = norm_provider(prov)
                p = provider_dir.get(prov_norm)
                to_email = _split_first_pipe(_s(getattr(p, "order_email", None) or getattr(p, "email", None) or getattr(p, "emails", None))) if p else ""
                if not to_email:
                    _touch_send_status(
                        venue_id=venue_id,
                        order_id=int(order.id),
                        provider_name=prov_norm,
                        actor=actor,
                        channel="email",
                        ok=False,
                        error="missing provider email",
                    )
                    fail += 1
                    continue

                _ensure_workflow_order_sent(venue_id=venue_id, order_id=int(order.id), provider_name=prov_norm, actor=actor)
                subject = _build_subject(v, int(order.id), prov_norm)
                body_text = _build_supplier_message_text(
                    templates=v,
                    order_id=int(order.id),
                    provider_name=prov_norm,
                    prov_lines=prov_lines,
                    products_by_id=products_by_id,
                )

                try:
                    send_smtp_email(
                        to=_split_emails(to_email),
                        cc=_split_emails(v.email_cc),
                        bcc=_split_emails(v.email_bcc),
                        subject=subject,
                        text_body=body_text,
                    )
                    _touch_send_status(venue_id=venue_id, order_id=int(order.id), provider_name=prov_norm, actor=actor, channel="email", ok=True)
                    ok += 1
                except Exception as e:
                    _touch_send_status(venue_id=venue_id, order_id=int(order.id), provider_name=prov_norm, actor=actor, channel="email", ok=False, error=str(e))
                    fail += 1


            # move order to pending_receive only if every supplier has been sent at least once
            if _all_providers_sent(order_id=int(order.id), provider_names=list(grouped_send.keys())):
                _set_order_status(int(order.id), "pending_receive", actor)
                _bump_refresh(venue_id)
                st.success(f"✅ Enviados {ok} · ❌ Fallos {fail} · Pedido → Pendiente")
                st.rerun()
            else:
                st.success(f"✅ Enviados {ok} · ❌ Fallos {fail}")
                st.rerun()

    with g2:
        if st.button(
            "🧹 Reset enviados",
            use_container_width=True,
            key=f"reset_send_{int(order.id)}",
            ):
            _reset_send_status(order_id=int(order.id))
            st.rerun()

    st.markdown("<div class='voi-divider'></div>", unsafe_allow_html=True)

    # -----------------------------
    # Per-provider cards (mobile-first)
    # -----------------------------
    for prov, prov_lines in grouped_send.items():
        prov_norm = norm_provider(prov)
        p = provider_dir.get(prov_norm)

        to_email = _split_first_pipe(_s(getattr(p, "order_email", None) or getattr(p, "email", None) or getattr(p, "emails", None))) if p else ""
        phone = _split_first_pipe(_s(getattr(p, "order_phone", None) or getattr(p, "phone", None) or getattr(p, "phones", None))) if p else ""

        srow = send_map.get(prov_norm)
        sent = bool(getattr(srow, "sent", False)) if srow else False
        last_error = _s(getattr(srow, "last_error", "")) if srow else ""
        attempts = int(getattr(srow, "send_attempts", 0) or 0) if srow else 0
        chip_txt, chip_cls = _chip_for_send(sent, last_error)

        supplier_link = build_seguimiento_url(order_id=int(order.id), provider_name=prov_norm, role=ROLE_SUPPLIER, page_path="seguimiento")
        venue_link = build_seguimiento_url(order_id=int(order.id), provider_name=prov_norm, role=ROLE_VENUE, page_path="seguimiento")

        active_lines = len([x for x in prov_lines if _safe_float(x.get("qty"), 0) > 0])

        with st.container(border=True):
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;gap:10px;align-items:flex-start;'>"
                f"<div><div style='font-weight:900;font-size:1.05rem'>{prov_norm}</div>"
                f"<div class='voi-muted'>{active_lines} líneas · intentos: {attempts}</div></div>"
                f"<div class='{chip_cls}'>{chip_txt}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if last_error:
                st.caption(f"⚠️ {last_error}")

            # Primary actions
            b1, b2 = st.columns(2, vertical_alignment="center")
            with b1:
                email_disabled = (not use_email) or (not to_email)
                email_label = "🔁 Reenviar email" if sent else "✅ Enviar email"
                if st.button(email_label, type="primary", use_container_width=True, disabled=email_disabled, key=f"send_email_{int(order.id)}_{prov_norm}"):
                    # ✅ IMPORTANT: If smart is ON, persist to DB and rebuild groups BEFORE sending
                    if apply_smart_for_send:
                        changed = _materialize_smart_to_db_for_send(provider_name=None)
                        if changed:
                            lines = _reload_lines_from_db()
                            grouped = _group_lines_by_provider(lines, products_by_id)
                            grouped_send = _group_lines_for_sending(apply_smart=True)
                            _bump_refresh(venue_id)

                            # refresh prov_lines so email matches DB + link
                            prov_lines = grouped_send.get(prov_norm, prov_lines)
                            
                    _ensure_workflow_order_sent(venue_id=venue_id, order_id=int(order.id), provider_name=prov_norm, actor=actor)
                    subject = _build_subject(v, int(order.id), prov_norm)
                    body_text = _build_supplier_message_text(
                        templates=v,
                        order_id=int(order.id),
                        provider_name=prov_norm,
                        prov_lines=prov_lines,
                        products_by_id=products_by_id,
                    )
                    try:
                        send_smtp_email(
                            to=_split_emails(to_email),
                            cc=_split_emails(v.email_cc),
                            bcc=_split_emails(v.email_bcc),
                            subject=subject,
                            text_body=body_text,
                        )
                        _touch_send_status(venue_id=venue_id, order_id=int(order.id), provider_name=prov_norm, actor=actor, channel="email", ok=True)
                        if _all_providers_sent(order_id=int(order.id), provider_names=list(grouped_send.keys())):
                            _set_order_status(int(order.id), "pending_receive", actor)
                            _bump_refresh(venue_id)
                        st.rerun()
                    except Exception as e:
                        _touch_send_status(venue_id=venue_id, order_id=int(order.id), provider_name=prov_norm, actor=actor, channel="email", ok=False, error=str(e))
                        st.error(f"No se pudo enviar: {e}")

            with b2:
                wa_disabled = (not use_wa) or (not phone)
                if wa_disabled:
                    st.button("📲 WhatsApp", use_container_width=True, disabled=True, key=f"wa_disabled_{int(order.id)}_{prov_norm}")
                else:
                    phone_norm = _normalize_phone(phone, wa_cc)
                    body_text = _build_supplier_message_text(
                        templates=v,
                        order_id=int(order.id),
                        provider_name=prov_norm,
                        prov_lines=prov_lines,
                        products_by_id=products_by_id,
                    )
                    if phone_norm:
                        wa = f"https://wa.me/{phone_norm.replace('+','')}?text={up.quote(body_text)}"
                        st.link_button("📲 WhatsApp", wa, use_container_width=True)
                    else:
                        st.button("📲 WhatsApp", use_container_width=True, disabled=True, key=f"wa_bad_{int(order.id)}_{prov_norm}")


def borrador_tab(
    venue_id: int,
    venue_role: Optional[str],
    *,
    deep_order_id: Optional[int] = None,
) -> None:
    """Page for draft (borrador) orders only."""

    _inject_css()
    actor = _s(
        st.session_state.get("user_email")
        or st.session_state.get("actor")
        or st.session_state.get("email")
        or current_actor()
    )

    st.markdown("#### 📝 Borradores")

    active_key = f"borrador_active_order_id_{venue_id}"

    # Load orders
    orders_all = _list_orders_cached(get_session, venue_id, _refresh_token(venue_id))

    # Deep-link: consume once
    deep_order_id_once_key = f"borrador_deep_order_consumed_{venue_id}"
    if deep_order_id is not None and not st.session_state.get(deep_order_id_once_key):
        st.session_state[deep_order_id_once_key] = True
        st.session_state[active_key] = int(deep_order_id)
    else:
        deep_order_id = None

    # Filter draft only
    orders = [o for o in orders_all if _s(getattr(o, "status", "draft")).lower() == "draft"]
    if not orders:
        st.info("No hay borradores.")
        return

    # Order picker
    ids = [int(o.id) for o in orders if o.id is not None]
    labels = {int(o.id): _order_label(o) for o in orders if o.id is not None}

    default_oid = int(st.session_state.get(active_key) or ids[0])
    if default_oid not in ids:
        default_oid = ids[0]

    picked = st.selectbox(
        "Pedido",
        options=ids,
        index=ids.index(default_oid),
        format_func=lambda oid: labels.get(int(oid), str(oid)),
        key=f"borrador_picker_{venue_id}",
    )
    st.session_state[active_key] = int(picked)

    # URL sync
    cur_oid = qp_int("order_id")
    if cur_oid != int(picked):
        set_query_params(page="borrador", order_id=str(int(picked)))

    # Load + render
    with get_session() as s:
        order = s.exec(
            select(Order).where(Order.id == int(picked), Order.venue_id == int(venue_id))
        ).first()

    if not order:
        st.error("Pedido no encontrado.")
        return

    products, products_by_id, label_by_id, cat_by_pid, prov_by_pid, all_categories, all_providers, base_pids = _product_ui_index_cached(get_session, venue_id)
    lines = _order_lines_cached(get_session, int(order.id), _refresh_token(venue_id))

    _render_header(order)
    _render_workflow_actions(venue_id=venue_id, order=order, role=venue_role, actor=actor)
    st.markdown("<div class='voi-divider'></div>", unsafe_allow_html=True)

    _render_lines_editor(venue_id=venue_id, order=order, actor=actor, products=products, lines=lines)


def orders_tab(
    venue_id: int,
    venue_role: Optional[str],
    *,
    deep_order_id: Optional[int] = None,
    deep_provider: Optional[str] = None,
    deep_status: Optional[str] = None,
) -> None:
    """Page for ready-to-send (listo) orders only."""

    _inject_css()
    actor = _s(
        st.session_state.get("user_email")
        or st.session_state.get("actor")
        or st.session_state.get("email")
        or current_actor()
    )

    st.markdown("## 🧾 Pedidos")

    active_key = f"orders_active_order_id_{venue_id}"

    # Load orders
    orders_all = _list_orders_cached(get_session, venue_id, _refresh_token(venue_id))

    # Deep-link: consume once
    deep_order_id_once_key = f"orders_deep_order_consumed_{venue_id}"
    if deep_order_id is not None and not st.session_state.get(deep_order_id_once_key):
        st.session_state[deep_order_id_once_key] = True
        st.session_state[active_key] = int(deep_order_id)
    else:
        deep_order_id = None

    # Filter ready_to_send only
    orders = [o for o in orders_all if _s(getattr(o, "status", "draft")).lower() == "ready_to_send"]
    if not orders:
        st.info("No hay pedidos listos para enviar.")
        return

    # Order picker
    ids = [int(o.id) for o in orders if o.id is not None]
    labels = {int(o.id): _order_label(o) for o in orders if o.id is not None}

    default_oid = int(st.session_state.get(active_key) or ids[0])
    if default_oid not in ids:
        default_oid = ids[0]

    picked = st.selectbox(
        "Pedido",
        options=ids,
        index=ids.index(default_oid),
        format_func=lambda oid: labels.get(int(oid), str(oid)),
    )
    st.session_state[active_key] = int(picked)

    # URL sync
    cur_oid = qp_int("order_id")
    if cur_oid != int(picked):
        set_query_params(page="orders", order_id=str(int(picked)))

    # Load + render
    with get_session() as s:
        order = s.exec(
            select(Order).where(Order.id == int(picked), Order.venue_id == int(venue_id))
        ).first()

    if not order:
        st.error("Pedido no encontrado.")
        return

    products, products_by_id, label_by_id, cat_by_pid, prov_by_pid, all_categories, all_providers, base_pids = _product_ui_index_cached(get_session, venue_id)
    lines = _order_lines_cached(get_session, int(order.id), _refresh_token(venue_id))

    _render_header(order)
    _render_workflow_actions(venue_id=venue_id, order=order, role=venue_role, actor=actor)
    st.markdown("<div class='voi-divider'></div>", unsafe_allow_html=True)

    _render_send_section(venue_id=venue_id, order=order, products=products, lines=lines, actor=actor)
