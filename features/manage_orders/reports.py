from __future__ import annotations

from datetime import datetime, timedelta, date
from typing import Any, Optional

import pandas as pd
import streamlit as st
from sqlmodel import select

from core.db import get_session
from core.public_links import norm_provider
from domain.models import ProviderSendStatus

# Reuse existing business logic + pricing rules from receive_orders
from features.manage_orders.receive_orders import (
    _load_order_context,
    _get_history_orders,
    _tickets_for_provider,
    _provider_closed,
    _provider_resolution_summary,
    _price_for_pid,
    _pricing_for_line,
    _iva_pct_for_pid,
    _safe_float,
    _s,
    _now,
    _get_received_info,
    _is_redelivery_item,
)

ROUND_N = 3

def _round_df(df: pd.DataFrame, digits: int = ROUND_N) -> pd.DataFrame:
    """Round all numeric columns to N decimals for consistent UI + exports."""
    out = df.copy()
    num_cols = out.select_dtypes(include=["number"]).columns
    if len(num_cols) > 0:
        out[num_cols] = out[num_cols].round(digits)
    return out


def _to_csv_excel_utf8(df: pd.DataFrame) -> bytes:
    """
    Excel-friendly CSV for Greek letters:
    UTF-8 with BOM so Excel detects encoding.
    """
    return df.to_csv(index=False).encode("utf-8-sig")


def _to_xlsx_bytes(df: pd.DataFrame, sheet_name: str = "Report") -> bytes:
    """Excel-native export, best for Greek letters."""
    import io
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    bio.seek(0)
    return bio.read()

def _money(x: float) -> str:
    try:
        return f"€{float(x):,.2f}"
    except Exception:
        return "€0.000"


def _month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def _invoice_dt_for(ctx, provn: str) -> datetime:
    receipt = (ctx.receipts_by_provider or {}).get(provn)
    dt = getattr(receipt, "invoice_number_set_at", None) if receipt else None
    if isinstance(dt, datetime):
        return dt
    when = getattr(ctx.order, "created_at", None)
    return when if isinstance(when, datetime) else _now()


def _expected_qty(ctx, prov: str, ln) -> float:
    ordered = _safe_float(getattr(ln, "quantity", 0.0), 0.0)
    fu = ctx.followups_by_key.get((prov, int(ln.id)))
    stt = (_s(getattr(fu, "supplier_status", None))).lower() if fu else "unknown"
    sqty = getattr(fu, "supplier_qty", None) if fu else None

    if stt == "missing":
        return 0.0
    if stt == "partial":
        return _safe_float(sqty, 0.0)
    if stt == "ok":
        return _safe_float(sqty, ordered) if sqty is not None else ordered
    return ordered


@st.cache_data(ttl=30, show_spinner=False)
def _provider_sent_pairs(order_ids: tuple[int, ...]) -> set[tuple[int, str]]:
    if not order_ids:
        return set()
    with get_session() as s:
        rows = list(
            s.exec(select(ProviderSendStatus).where(ProviderSendStatus.order_id.in_(order_ids))).all()
        )
    out: set[tuple[int, str]] = set()
    for r in rows:
        if (
            bool(getattr(r, "sent", False))
            or bool(getattr(r, "sent_email", False))
            or bool(getattr(r, "sent_whatsapp", False))
        ):
            out.add(
                (int(getattr(r, "order_id", 0) or 0), norm_provider(_s(getattr(r, "provider_name", ""))))
            )
    return out

def _credit_note_issue_date(ctx, provn: str) -> Optional[datetime]:
    """
    Returns the issue date of the credit note if known.
    Priority:
      1) ProviderResolution.created_at (credit_note)
      2) Parsed supplier solution note metadata (credit_note_date)
      3) None
    """
    # 1) ProviderResolution row (strongest signal)
    for r in (ctx.resolutions_by_provider.get(provn) or []):
        if (_s(getattr(r, "resolution_type", "")).strip().lower() == "credit_note"):
            dt = getattr(r, "created_at", None)
            if isinstance(dt, datetime):
                return dt

    # 2) Supplier solution note metadata
    try:
        summ = _provider_resolution_summary(ctx, provn) or {}
        sol = summ.get("solution") or {}
        raw = sol.get("credit_note_date")
        if raw:
            # Accept ISO or YYYY-MM-DD
            try:
                return datetime.fromisoformat(str(raw))
            except Exception:
                pass
    except Exception:
        pass

    return None

def _credit_note_state(ctx, provn: str) -> tuple[str, str]:
    """
    Returns (cn_number, cn_status):
      cn_status: "missing" | "open" | "closed" | ""
    """
    cn_no = ""
    cn_status = ""

    # ProviderResolution rows (strongest signal)
    for r in (ctx.resolutions_by_provider.get(provn) or []):
        rtype = _s(getattr(r, "resolution_type", "")).strip().lower()
        if rtype != "credit_note":
            continue
        status = _s(getattr(r, "status", "")).strip().lower()
        cn = _s(getattr(r, "credit_note_invoice", None)).strip()
        if cn and not cn_no:
            cn_no = cn
        if status in {"open", "pending"}:
            cn_status = "open"
        elif status == "closed":
            cn_status = "closed"

    # Fallback: provider summary note parsing
    if not cn_no or not cn_status:
        try:
            summ = _provider_resolution_summary(ctx, provn) or {}
            if (_s(summ.get("mode")) or "").strip().lower() == "credit_note":
                sol = summ.get("solution") or {}
                cn = _s(sol.get("credit_note_invoice")).strip()
                if cn and not cn_no:
                    cn_no = cn
                if not cn_status:
                    cn_status = "closed" if cn_no else "missing"
        except Exception:
            pass

    if cn_status == "" and cn_no:
        cn_status = "closed"
    return cn_no, cn_status


def _norm_name(x: str) -> str:
    x = (x or "").strip().lower()
    return " ".join(x.replace("/", " ").replace("-", " ").split())


def _credit_qty_map_from_supplier_note(ctx, provn: str) -> dict[str, float]:
    """
    Build {normalized_item_name -> credit_qty} from supplier solution note items
    (credit note items only; redelivery items ignored).
    """
    out: dict[str, float] = {}
    summ = _provider_resolution_summary(ctx, provn) or {}
    if (_s(summ.get("mode")) or "").strip().lower() != "credit_note":
        return out

    sol = summ.get("solution") or {}
    items = sol.get("items") or []
    if not isinstance(items, list):
        return out

    for it in items:
        if not isinstance(it, dict):
            continue
        if _is_redelivery_item(it):
            continue
        nm = _s(it.get("name")) or ""
        qty = _safe_float(it.get("qty"), 0.0)
        key = _norm_name(nm)
        if key and qty > 0:
            out[key] = out.get(key, 0.0) + float(qty)
    return out


@st.cache_data(show_spinner=False)
def _build_reports_df(venue_id: int) -> pd.DataFrame:
    """
    Line-level DF for all CLOSED provider workflows.
    Includes CN expected amounts (derived from supplier credit note items).
    """
    orders = _get_history_orders(int(venue_id))
    if not orders:
        return pd.DataFrame()

    order_ids = [int(o.id) for o in orders if getattr(o, "id", None) is not None]
    sent_pairs = _provider_sent_pairs(tuple(order_ids))

    rows: list[dict[str, Any]] = []

    for o in orders:
        oid = int(getattr(o, "id", 0) or 0)
        if not oid:
            continue

        ctx = _load_order_context(int(venue_id), oid)
        providers = sorted((ctx.lines_by_provider or {}).keys(), key=lambda x: x.lower())
        providers = [p for p in providers if (oid, norm_provider(p)) in sent_pairs]

        for prov in providers:
            provn = norm_provider(prov)
            if not _provider_closed(ctx, provn):
                continue

            inv_dt = _invoice_dt_for(ctx, provn)
            inv_date = inv_dt.date()
            inv_month = _month_key(inv_date)

            receipt = (ctx.receipts_by_provider or {}).get(provn)
            inv_no = _s(getattr(receipt, "invoice_number", None)).strip()

            summ = _provider_resolution_summary(ctx, provn) or {}
            mode = (_s(summ.get("mode")) or "").strip().lower()
            sol_key = mode
            if sol_key == "ok_redelivery":
                sol_key = "supplementary_delivery"
            elif sol_key == "ok_internal":
                sol_key = "ok_internal"

            cn_no, cn_status = _credit_note_state(ctx, provn) if sol_key == "credit_note" else ("", "")
            cn_issue_dt = _credit_note_issue_date(ctx, provn) if sol_key == "credit_note" else None
            credit_qty_map = _credit_qty_map_from_supplier_note(ctx, provn) if sol_key == "credit_note" else {}


            tickets = _tickets_for_provider(ctx, provn)
            has_incidence = bool(tickets)
            incidence_kinds = ",".join(
                sorted(
                    set(
                        [
                            (_s(getattr(t, "kind", "")) or "").strip().lower()
                            for t in tickets
                            if _s(getattr(t, "kind", ""))
                        ]
                    )
                )
            )

            lines = (ctx.lines_by_provider.get(provn) or [])
            for ln in lines:
                lid = int(getattr(ln, "id", 0) or 0)
                pid_raw = getattr(ln, "product_id", None)
                pid = int(pid_raw) if pid_raw not in (None, "", 0, "0") else None

                p = (ctx.products_by_id or {}).get(pid) if pid else None
                product_name = _s(getattr(p, "name", None)) or _s(getattr(ln, "name", None)) or "—"
                category = _s(getattr(p, "category", None)) or "—"

                ordered_qty = _safe_float(getattr(ln, "quantity", 0.0), 0.0)
                expected_qty = _expected_qty(ctx, provn, ln)

                issue_status, venue_received_qty, issue_qty, in_invoice, reason = _get_received_info(ctx, provn, lid)

                # Pricing qty (aligned with receive_orders intent)
                price_qty = expected_qty
                if sol_key == "supplementary_delivery":
                    price_qty = float(venue_received_qty or 0.0) if issue_status in ("missing", "partial") else expected_qty
                elif issue_status == "missing" and (in_invoice is False):
                    price_qty = float(venue_received_qty or 0.0)

                gross_unit = _price_for_pid(ctx.products_by_id, pid)
                if gross_unit and gross_unit > 0:
                    pricing = _pricing_for_line(
                        venue_id=int(ctx.order.venue_id),
                        providers_by_name=ctx.providers_by_name,
                        provider_name=provn,
                        pid=pid,
                        qty=float(price_qty),
                        gross_unit=float(gross_unit),
                    )
                    net_unit = float(pricing.get("net_unit", gross_unit) or gross_unit)
                else:
                    net_unit = 0.0

                iva_pct = _iva_pct_for_pid(ctx.products_by_id, pid, 21.0)
                subtotal = float(price_qty) * float(net_unit)
                iva_eur = float(subtotal) * (float(iva_pct) / 100.0)
                total = float(subtotal) + float(iva_eur)

                # CN expected amounts (best-effort: match supplier note item name to product name)
                cn_credit_qty = 0.0
                cn_credit_subtotal = 0.0
                cn_credit_iva = 0.0
                cn_credit_total = 0.0
                if sol_key == "credit_note" and credit_qty_map:
                    key = _norm_name(product_name)
                    cn_credit_qty = float(credit_qty_map.get(key, 0.0) or 0.0)
                    if cn_credit_qty > 0 and net_unit > 0:
                        cn_credit_subtotal = float(cn_credit_qty) * float(net_unit)
                        cn_credit_iva = float(cn_credit_subtotal) * (float(iva_pct) / 100.0)
                        cn_credit_total = float(cn_credit_subtotal) + float(cn_credit_iva)

                rows.append(
                    {
                        "invoice_date": inv_date,
                        "invoice_month": inv_month,
                        "order_id": oid,
                        "provider": provn,
                        "invoice_number": inv_no or "",
                        "solution": sol_key or "",
                        "credit_note_number": cn_no or "",
                        "credit_note_issue_date": cn_issue_dt.date().isoformat() if cn_issue_dt else "",
                        "credit_note_status": cn_status or "",
                        "has_incidence": has_incidence,
                        "incidence_kinds": incidence_kinds,
                        "line_id": lid,
                        "product": product_name,
                        "category": category,
                        "iva_pct": float(iva_pct or 0.0),
                        "subtotal": float(subtotal or 0.0),
                        "iva_eur": float(iva_eur or 0.0),
                        "total": float(total or 0.0),
                        "cn_credit_subtotal": float(cn_credit_subtotal),
                        "cn_credit_iva": float(cn_credit_iva),
                        "cn_credit_total": float(cn_credit_total),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


def _filter_df(df: pd.DataFrame, d_from: date, d_to: date) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (df["invoice_date"].dt.date >= d_from) & (df["invoice_date"].dt.date <= d_to)
    return df.loc[mask].copy()


def _invoice_level(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    g = (
        df.groupby(
            [
                "invoice_date",
                "invoice_month",
                "provider",
                "invoice_number",
                "solution",
                "credit_note_number",
                "credit_note_status",
                "has_incidence",
                "incidence_kinds",
            ],
            dropna=False,
        )
        .agg(
            subtotal=("subtotal", "sum"),
            iva_eur=("iva_eur", "sum"),
            total=("total", "sum"),
            cn_expected_total=("cn_credit_total", "sum"),
            credit_note_issue_date=("credit_note_issue_date", "max"),
        )

        .reset_index()
        .sort_values(["invoice_date", "provider"], ascending=[False, True])
    )
    g["cn_coverage_pct"] = g.apply(
        lambda r: (float(r["cn_expected_total"]) / float(r["total"]) * 100.0) if float(r["total"] or 0.0) > 0 else 0.0,
        axis=1,
    )
    g["cn_remaining_total"] = g.apply(
        lambda r: float(r["total"] or 0.0) - float(r["cn_expected_total"] or 0.0),
        axis=1,
    )
    return g


def _iva_by_rate(df: pd.DataFrame, expected_rates: list[float]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["iva_rate", "base", "iva", "total"])

    tmp = df.copy()
    tmp["iva_rate"] = tmp["iva_pct"].astype(float).round(2)

    g = (
        tmp.groupby("iva_rate", dropna=False)
        .agg(base=("subtotal", "sum"), iva=("iva_eur", "sum"), total=("total", "sum"))
        .reset_index()
        .sort_values("iva_rate", ascending=False)
    )

    seen = set([float(x) for x in g["iva_rate"].tolist() if pd.notna(x)])
    add = []
    for r in expected_rates:
        if float(r) not in seen:
            add.append({"iva_rate": float(r), "base": 0.0, "iva": 0.0, "total": 0.0})
    if add:
        g = pd.concat([g, pd.DataFrame(add)], ignore_index=True).sort_values("iva_rate", ascending=False)

    return g


def reports_page(venue_id: int, venue_role: str) -> None:
    st.header("Reports")
    st.caption("IVA/ΦΠΑ breakdown + credit note coverage + spend analytics.")

    today = datetime.utcnow().date()
    default_start = today.replace(day=1) - timedelta(days=90)

    c0, c1, c2 = st.columns([1.4, 1.2, 1.2], vertical_alignment="center")
    with c0:
        dr = st.date_input("Date range", value=(default_start, today), format="DD/MM/YYYY")
        if isinstance(dr, tuple) and len(dr) == 2:
            d_from, d_to = dr[0], dr[1]
        else:
            d_from, d_to = default_start, today

    with c1:
        vat_profile = st.selectbox("VAT profile", ["Spain (IVA)", "Greece (ΦΠΑ)"], index=0)
    with c2:
        show_outstanding_cn = st.checkbox("Only outstanding credit notes", value=False)

    expected_rates = [21.0, 10.0, 4.0, 0.0] if vat_profile.startswith("Spain") else [24.0, 13.0, 6.0, 4.0]

    df_all = _build_reports_df(int(venue_id))
    if df_all.empty:
        st.info("No history available yet (no CLOSED supplier workflows).")
        return

    df = _filter_df(df_all, d_from, d_to)
    if show_outstanding_cn:
        df = df[(df["solution"] == "credit_note") & (df["credit_note_number"].fillna("").str.strip() == "")]

    if df.empty:
        st.info("No data matches the selected filters.")
        return

    inv = _invoice_level(df)
    df = _round_df(df)
    inv = _round_df(inv)


    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Subtotal", _money(float(df["subtotal"].sum())))
    k2.metric("IVA", _money(float(df["iva_eur"].sum())))
    k3.metric("Total", _money(float(df["total"].sum())))
    k4.metric("Invoices", f"{inv.shape[0]}")

    # --- IVA by rate ---
    st.subheader("IVA by rate")
    iva_tbl = _iva_by_rate(df, expected_rates=expected_rates)
    iva_tbl = _round_df(iva_tbl)

    st.dataframe(
        iva_tbl.rename(columns={"iva_rate": "Rate (%)", "base": "Base", "iva": "IVA", "total": "Total"}),
        use_container_width=True,
        hide_index=True,
    )

    # --- CN matching view ---
    st.subheader("Credit note matching (Invoice total vs CN expected total)")
    cn_inv = inv[inv["solution"] == "credit_note"].copy()
    cn_inv = _round_df(cn_inv)

    if cn_inv.empty:
        st.info("No invoices marked as Credit Note solution in this date range.")
    else:
        cn_inv["Invoice total"] = cn_inv["total"].apply(_money)
        cn_inv["CN expected total"] = cn_inv["cn_expected_total"].apply(_money)
        cn_inv["Coverage % (expected)"] = cn_inv["cn_coverage_pct"].apply(lambda x: f"{float(x):.1f}%")
        cn_inv["Remaining"] = cn_inv["cn_remaining_total"].apply(_money)
        cn_inv["CN number?"] = cn_inv["credit_note_number"].apply(lambda x: "✅" if str(x).strip() else "❌")
        cn_inv["CN issue date"] = cn_inv["credit_note_issue_date"].replace("", "—")


        missing = cn_inv[cn_inv["credit_note_number"].fillna("").str.strip() == ""]
        if not missing.empty:
            st.warning("These are Credit Note solutions but the CN number is missing:")
            st.dataframe(
                missing[["invoice_date", "provider", "invoice_number", "Invoice total", "CN expected total", "incidence_kinds"]],
                use_container_width=True,
                hide_index=True,
            )

        st.dataframe(
            cn_inv[
                [
                    "invoice_date",
                    "provider",
                    "invoice_number",
                    "credit_note_number",
                    "CN issue date",
                    "Invoice total",
                    "CN expected total",
                    "Coverage % (expected)",
                    "Remaining",
                    "incidence_kinds",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


    # quick exports
    st.subheader("Export (Excel-safe)")

    exp0, exp1 = st.columns(2)

    with exp0:
        st.download_button(
            "Download invoice-level CSV (UTF-8)",
            data=_to_csv_excel_utf8(inv),
            file_name=f"reports_invoices_{venue_id}_{d_from.isoformat()}_{d_to.isoformat()}.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download invoice-level XLSX",
            data=_to_xlsx_bytes(inv, sheet_name="Invoices"),
            file_name=f"reports_invoices_{venue_id}_{d_from.isoformat()}_{d_to.isoformat()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with exp1:
        st.download_button(
            "Download line-level CSV (UTF-8)",
            data=_to_csv_excel_utf8(df),
            file_name=f"reports_lines_{venue_id}_{d_from.isoformat()}_{d_to.isoformat()}.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download line-level XLSX",
            data=_to_xlsx_bytes(df, sheet_name="Lines"),
            file_name=f"reports_lines_{venue_id}_{d_from.isoformat()}_{d_to.isoformat()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        
        # -------------------
    # Spend analytics
    # -------------------
    # =========================
    # Spend analytics (pro)
    # =========================
    st.divider()
    st.header("Spend analytics")

    # --- Filter bar ---
    f1, f2, f3, f4 = st.columns([1.1, 1.1, 1.6, 1.0], vertical_alignment="center")

    with f1:
        metric = st.selectbox(
            "Metric",
            ["total", "subtotal", "iva_eur"],
            index=0,
            format_func=lambda x: {"total": "Total (€)", "subtotal": "Base (€)", "iva_eur": "IVA (€)"}[x],
            key="reports_metric",
        )

    with f2:
        top_n = st.slider("Top N", min_value=5, max_value=50, value=20, step=5, key="reports_top_n")

    # Build option lists (sorted by spend so it’s useful)
    prod_spend = (
        df.groupby("product", dropna=False)["total"].sum().sort_values(ascending=False)
        if "product" in df.columns else pd.Series(dtype=float)
    )
    supplier_spend = (
        df.groupby("provider", dropna=False)["total"].sum().sort_values(ascending=False)
        if "provider" in df.columns else pd.Series(dtype=float)
    )
    cat_spend = (
        df.groupby("category", dropna=False)["total"].sum().sort_values(ascending=False)
        if "category" in df.columns else pd.Series(dtype=float)
    )

    with f3:
        selected_products = st.multiselect(
            "Products (filter)",
            options=list(prod_spend.index.astype(str))[:1000],  # avoid huge UI, but still large enough
            default=[],
            help="Search + select products to focus the analytics. Leave empty to include all products.",
            key="reports_products_ms",
        )

    with f4:
        chart_mode = st.selectbox(
            "View",
            ["Charts + tables", "Charts only", "Tables only"],
            index=0,
            key="reports_view_mode",
        )

    # Optional secondary filters (collapsed)
    with st.expander("More filters", expanded=False):
        cA, cB = st.columns(2)
        with cA:
            selected_suppliers = st.multiselect(
                "Suppliers",
                options=list(supplier_spend.index.astype(str)),
                default=[],
                key="reports_suppliers_ms",
            )
        with cB:
            selected_categories = st.multiselect(
                "Categories",
                options=list(cat_spend.index.astype(str)),
                default=[],
                key="reports_categories_ms",
            )

    # --- Apply filters to df (analytics only) ---
    dfA = df.copy()
    dfA = _round_df(dfA)


    if selected_products:
        dfA = dfA[dfA["product"].astype(str).isin(set(selected_products))]

    if "reports_suppliers_ms" in st.session_state and st.session_state["reports_suppliers_ms"]:
        dfA = dfA[dfA["provider"].astype(str).isin(set(st.session_state["reports_suppliers_ms"]))]

    if "reports_categories_ms" in st.session_state and st.session_state["reports_categories_ms"]:
        dfA = dfA[dfA["category"].astype(str).isin(set(st.session_state["reports_categories_ms"]))]

    if dfA.empty:
        st.info("No data matches the analytics filters.")
        return

    # Ensure we never show time-of-day
    # invoice_month is already YYYY-MM; invoice_date becomes date string YYYY-MM-DD in tables.
    dfA = dfA.copy()
    dfA["invoice_date_str"] = dfA["invoice_date"].dt.date.astype(str)

    # --- Helpers for nice tables ---
    def _fmt_currency_cols(t: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        out = t.copy()
        for c in cols:
            if c in out.columns:
                out[c] = out[c].astype(float).map(lambda x: f"€{x:,.2f}")
        return out

    def _top_breakdown(df_in: pd.DataFrame, key: str) -> pd.DataFrame:
        g = (
            df_in.groupby(key, dropna=False)
            .agg(
                total=("total", "sum"),
                subtotal=("subtotal", "sum"),
                iva_eur=("iva_eur", "sum"),
                invoices=("invoice_number", lambda s: s.astype(str).nunique()),
                lines=("line_id", "nunique") if "line_id" in df_in.columns else ("total", "size"),
            )
            .reset_index()
            .sort_values(metric, ascending=False)
            .head(top_n)
        )
        return g

    # --- Monthly timeseries (no time) ---
    by_month = (
        dfA.groupby("invoice_month", dropna=False)
        .agg(total=("total", "sum"), subtotal=("subtotal", "sum"), iva_eur=("iva_eur", "sum"))
        .reset_index()
        .sort_values("invoice_month")
    )

    # --- Breakdowns ---
    by_supplier = _top_breakdown(dfA, "provider")
    by_category = _top_breakdown(dfA, "category")
    by_product = _top_breakdown(dfA, "product")

    # --- Comparison pivot supplier x month ---
    pivot = dfA.pivot_table(
        index="provider",
        columns="invoice_month",
        values=metric,
        aggfunc="sum",
        fill_value=0.0,
    )

    # --- Render: tabs for a pro UX ---
    tabs = st.tabs(["Overview", "By supplier", "By category", "By product", "Compare"])

    # Plotly (if installed) for cleaner labels (no timestamps)
    try:
        import plotly.express as px  # type: ignore
        PLOTLY = True
    except Exception:
        PLOTLY = False

    # ========== Overview ==========
    with tabs[0]:
        k1, k2, k3 = st.columns(3)
        k1.metric("Total", _money(float(dfA["total"].sum())))
        k2.metric("Base", _money(float(dfA["subtotal"].sum())))
        k3.metric("IVA", _money(float(dfA["iva_eur"].sum())))

        st.markdown("#### Monthly trend")

        if chart_mode != "Tables only":
            if PLOTLY:
                fig = px.bar(by_month, x="invoice_month", y=metric)
                fig.update_layout(
                    xaxis_title="Month",
                    yaxis_title=metric.replace("_", " ").upper(),
                )
                fig.update_xaxes(type="category")
                # ensures no weird datetime formatting
                fig.update_traces(hovertemplate="%{x}<br>%{y:.2f}€<extra></extra>")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(by_month.set_index("invoice_month")[metric])

        if chart_mode != "Charts only":
            st.dataframe(
                _fmt_currency_cols(by_month, ["total", "subtotal", "iva_eur"]),
                use_container_width=True,
                hide_index=True,
            )

    # ========== By supplier ==========
    with tabs[1]:
        st.markdown("#### Top suppliers")
        if chart_mode != "Tables only":
            if PLOTLY:
                fig = px.bar(by_supplier, x="provider", y=metric)
                fig.update_layout(xaxis_title="Supplier", yaxis_title=metric.replace("_", " ").upper())
                fig.update_xaxes(type="category")
                fig.update_traces(hovertemplate="%{x}<br>%{y:.2f}€<extra></extra>")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(by_supplier.set_index("provider")[metric])

        if chart_mode != "Charts only":
            st.dataframe(
                _fmt_currency_cols(by_supplier, ["total", "subtotal", "iva_eur"]),
                use_container_width=True,
                hide_index=True,
            )

    # ========== By category ==========
    with tabs[2]:
        st.markdown("#### Top categories")
        if chart_mode != "Tables only":
            if PLOTLY:
                fig = px.bar(by_category, x="category", y=metric)
                fig.update_layout(xaxis_title="Category", yaxis_title=metric.replace("_", " ").upper())
                fig.update_xaxes(type="category")
                fig.update_traces(hovertemplate="%{x}<br>%{y:.2f}€<extra></extra>")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(by_category.set_index("category")[metric])

        if chart_mode != "Charts only":
            st.dataframe(
                _fmt_currency_cols(by_category, ["total", "subtotal", "iva_eur"]),
                use_container_width=True,
                hide_index=True,
            )

    # ========== By product ==========
    with tabs[3]:
        st.markdown("#### Top products")
        if selected_products:
            st.caption(f"Filtered to {len(selected_products)} selected products.")

        if chart_mode != "Tables only":
            if PLOTLY:
                fig = px.bar(by_product, x="product", y=metric)
                fig.update_layout(xaxis_title="Product", yaxis_title=metric.replace("_", " ").upper())
                fig.update_xaxes(type="category")
                fig.update_traces(hovertemplate="%{x}<br>%{y:.2f}€<extra></extra>")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(by_product.set_index("product")[metric])

        if chart_mode != "Charts only":
            st.dataframe(
                _fmt_currency_cols(by_product, ["total", "subtotal", "iva_eur"]),
                use_container_width=True,
                hide_index=True,
            )

    # ========== Compare ==========
    with tabs[4]:
        st.markdown("#### Supplier comparison by month")
        if chart_mode != "Tables only" and PLOTLY:
            # heatmap-style compare (still no timestamps)
            fig = px.imshow(
                pivot.values,
                x=list(pivot.columns.astype(str)),
                y=list(pivot.index.astype(str)),
                aspect="auto",
                labels=dict(x="Month", y="Supplier", color=metric.upper()),
            )
            st.plotly_chart(fig, use_container_width=True)

        if chart_mode != "Charts only":
            pivot_display = pivot.copy()
            # format numbers in a friendly way
            pivot_display = pivot_display.applymap(lambda x: f"€{float(x):,.2f}")
            st.dataframe(pivot_display, use_container_width=True)

    # Optional: quick CSV for the filtered analytics set
    st.download_button(
        "Download analytics dataset (filtered) CSV",
        data=dfA.to_csv(index=False).encode("utf-8"),
        file_name=f"reports_analytics_{venue_id}_{d_from.isoformat()}_{d_to.isoformat()}.csv",
        mime="text/csv",
    )


