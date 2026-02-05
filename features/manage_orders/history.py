from __future__ import annotations



from datetime import datetime
from typing import Any, Optional


import pandas as pd
import streamlit as st
from sqlmodel import select
import html
from core.db import get_session

from core.public_links import  norm_provider

from datetime import datetime, timedelta

from domain.models import (
    Order,
    OrderLine,
    OrderWorkflow,
    Product,
    Provider,
    ProviderLineFollowUp,
    ProviderReceipt,
    SeguimientoTicket, ProviderSendStatus,
    ProviderResolution

)
from domain.models import ProviderSendStatus

from features.manage_orders.receive_orders import (
    _load_order_context,
    _get_history_orders,
    _tickets_for_provider,
    _provider_closed,
    _provider_resolution_summary,
    _parse_ticket_resolution_note,
    _badge,
    _render_expected_lines,
    _safe_float,
    _s,
    _now,
    OrderContext,
    _is_redelivery_item,
)



def _render_history_tab(venue_id: int, *, deep_provider: Optional[str] = None) -> None:
    """
    New History UX:
      - Filter by date range (based on ProviderReceipt.invoice_number_set_at, fallback Order.created_at)
      - List per invoice (per provider) rather than per order
      - Show expected lines per invoice
      - Match credit notes (CN number) to invoice/provider
      - Filter by incidence kind + solution type
    """
    orders = _get_history_orders(int(venue_id))
    if not orders:
        st.info("No closed supplier history yet.")
        return

    # ---------- helpers ----------
    def _invoice_dt_for(ctx: OrderContext, provn: str) -> datetime:
        """
        Best-effort invoice date:
          1) ProviderReceipt.invoice_number_set_at
          2) Order.created_at
        """
        receipt = (ctx.receipts_by_provider or {}).get(provn)
        dt = getattr(receipt, "invoice_number_set_at", None) if receipt else None
        if isinstance(dt, datetime):
            return dt
        when = getattr(ctx.order, "created_at", None)
        return when if isinstance(when, datetime) else _now()

    def _pretty_kind(k: str) -> str:
        k = (k or "").strip().lower()
        return {
            "invoice_discrepancy": "Invoice discrepancy",
            "damaged": "Damaged",
            "wrong_item": "Wrong item",
            "missing": "Missing",
            "operational_missing": "Operational missing",
        }.get(k, k.replace("_", " ").title() if k else "—")

    def _ticket_resolution_str(t: Any) -> str:
        meta = _parse_ticket_resolution_note(_s(getattr(t, "resolution_note", None)))
        r = _s(meta.get("resolution")).strip().lower()
        return r

    def _credit_note_number_for_provider(ctx: OrderContext, provn: str) -> str:
        """
        Best-effort CN number:
          1) ProviderResolution rows (closed credit_note, with credit_note_invoice if present)
          2) Ticket meta (credit_note_invoice=...)
          3) Provider-level parsed supplier solution meta (credit_note_invoice)
        """
        # 1) ProviderResolution rows
        for r in (ctx.resolutions_by_provider.get(provn) or []):
            if (_s(getattr(r, "resolution_type", "")).strip().lower() == "credit_note") and (
                _s(getattr(r, "status", "")).strip().lower() == "closed"
            ):
                cn = _s(getattr(r, "credit_note_invoice", None)).strip()
                if cn:
                    return cn

        # 2) Ticket meta
        for t in _tickets_for_provider(ctx, provn):
            meta = _parse_ticket_resolution_note(_s(getattr(t, "resolution_note", None)))
            cn = _s(meta.get("credit_note_invoice")).strip()
            if cn:
                return cn

        # 3) Provider summary (supplier solution note parsing)
        try:
            summ = _provider_resolution_summary(ctx, provn) or {}
            sol = summ.get("solution") or {}
            cn = _s(sol.get("credit_note_invoice")).strip()
            if cn:
                return cn
        except Exception:
            pass

        return ""

    def _provider_sent_pairs_for_orders(order_ids: list[int]) -> set[tuple[int, str]]:
        """Only show providers actually sent (email/whatsapp/sent flag)."""
        if not order_ids:
            return set()
        with get_session() as s:
            rows = list(
                s.exec(
                    select(ProviderSendStatus).where(ProviderSendStatus.order_id.in_(order_ids))
                ).all()
            )
        out: set[tuple[int, str]] = set()
        for r in rows:
            if (
                bool(getattr(r, "sent", False))
                or bool(getattr(r, "sent_email", False))
                or bool(getattr(r, "sent_whatsapp", False))
            ):
                out.add((int(getattr(r, "order_id", 0) or 0), norm_provider(_s(getattr(r, "provider_name", "")))))
        return out

    # ---------- top filters ----------
    st.markdown("### History")

    # default window: last 30 days (nice UX)
    today = datetime.utcnow().date()
    default_start = today - timedelta(days=30)

    f0, f1, f2, f3 = st.columns([1.4, 1.2, 1.2, 1.2], vertical_alignment="center")
    with f0:
        q = st.text_input(
            "Search provider / invoice / order",
            placeholder="e.g. makro, CN-12, 2025-, #120",
        ).strip().lower()

    with f1:
        dr = st.date_input(
            "Date range",
            value=(default_start, today),
            format="DD/MM/YYYY",
        )
        if isinstance(dr, tuple) and len(dr) == 2:
            d_from, d_to = dr[0], dr[1]
        else:
            d_from, d_to = default_start, today

    # incidence kind filter
    KIND_OPTIONS = [
        ("invoice_discrepancy", "Invoice discrepancy"),
        ("damaged", "Damaged"),
        ("wrong_item", "Wrong item"),
        ("missing", "Missing"),
        ("operational_missing", "Operational missing"),
    ]
    with f2:
        kind_sel = st.multiselect(
            "Incidences",
            options=[k for k, _ in KIND_OPTIONS],
            default=[],
            format_func=lambda k: dict(KIND_OPTIONS).get(k, _pretty_kind(k)),
        )
        kind_sel_set = set([k.strip().lower() for k in kind_sel if k])

    # solution filter
    SOL_OPTIONS = [
        ("credit_note", "Credit note"),
        ("supplementary_delivery", "Re-delivery"),
        ("ok_internal", "Internal"),
        ("closed", "Closed (unknown)"),
        ("reject", "Rejected"),
    ]
    with f3:
        sol_sel = st.multiselect(
            "Solutions",
            options=[k for k, _ in SOL_OPTIONS],
            default=[],
            format_func=lambda k: dict(SOL_OPTIONS).get(k, k),
        )
        sol_sel_set = set([k.strip().lower() for k in sol_sel if k])

    # ---------- build per-invoice rows ----------
    order_ids = [int(o.id) for o in orders if getattr(o, "id", None) is not None]
    sent_pairs = _provider_sent_pairs_for_orders(order_ids)

    rows: list[dict[str, Any]] = []

    # We load contexts to compute per-provider invoice records.
    # This is history (smaller volume) so OK. If it grows, we can cache by order_id.
    for o in orders:
        oid = int(getattr(o, "id", 0) or 0)
        if not oid:
            continue

        ctx = _load_order_context(int(venue_id), oid)

        providers = sorted((ctx.lines_by_provider or {}).keys(), key=lambda x: x.lower())
        providers = [p for p in providers if (oid, norm_provider(p)) in sent_pairs]

        # deep-link: bring a provider to top (optional)
        if deep_provider:
            dp = norm_provider(deep_provider)
            providers.sort(key=lambda p: (0 if norm_provider(p) == dp else 1, p.lower()))

        for prov in providers:
            provn = norm_provider(prov)

            # only closed providers in history
            if not _provider_closed(ctx, provn):
                continue

            receipt = (ctx.receipts_by_provider or {}).get(provn)
            inv_no = _s(getattr(receipt, "invoice_number", None)).strip()
            inv_dt = _invoice_dt_for(ctx, provn)
            inv_date = inv_dt.date()

            # date range filter (inclusive)
            if inv_date < d_from or inv_date > d_to:
                continue

            # incidences for provider
            ts = _tickets_for_provider(ctx, provn)
            # show all tickets (history); but filter selection applies to whether this row appears
            kinds = [(_s(getattr(t, "kind", "")) or "").strip().lower() for t in ts]
            kind_counts: dict[str, int] = {}
            for kk in kinds:
                if not kk:
                    continue
                kind_counts[kk] = kind_counts.get(kk, 0) + 1

            # provider solution summary
            summ = _provider_resolution_summary(ctx, provn) or {}
            mode = (_s(summ.get("mode")) or "").strip().lower()  # credit_note / ok_redelivery / ok_internal / closed
            # normalize for filtering labels
            sol_key = mode
            if sol_key == "ok_redelivery":
                sol_key = "supplementary_delivery"
            elif sol_key == "ok_internal":
                sol_key = "ok_internal"

            cn_no = _credit_note_number_for_provider(ctx, provn) if sol_key == "credit_note" else ""

            # Filter by kind: keep row if it has ANY selected kind (or if none selected)
            if kind_sel_set:
                if not any((k in kind_sel_set) for k in kind_counts.keys()):
                    continue

            # Filter by solution: keep row if matches (or if none selected)
            if sol_sel_set:
                if sol_key not in sol_sel_set:
                    continue

            # Search filter
            if q:
                hay = " ".join(
                    [
                        provn.lower(),
                        inv_no.lower(),
                        f"#{oid}",
                        str(oid),
                        cn_no.lower(),
                    ]
                )
                if q not in hay:
                    continue

            rows.append(
                {
                    "order_id": oid,
                    "provider": provn,
                    "invoice_number": inv_no,
                    "invoice_dt": inv_dt,
                    "solution": sol_key,
                    "credit_note": cn_no,
                    "kind_counts": kind_counts,
                }
            )

    if not rows:
        st.info("No history matches your filters.")
        return

    # newest first
    rows.sort(key=lambda r: r["invoice_dt"] or _now(), reverse=True)

    # ---------- small summary table (quick scan) ----------
    def _sol_badge(sol_key: str, cn: str) -> str:
        if sol_key == "credit_note":
            return _badge("Credit note", "warn") + (f" {_badge(cn or 'CN missing', 'info')}" if cn else f" {_badge('CN missing', 'bad')}")
        if sol_key == "supplementary_delivery":
            return _badge("Re-delivery", "info")
        if sol_key == "ok_internal":
            return _badge("Internal", "ok")
        if sol_key == "reject":
            return _badge("Rejected", "warn")
        return _badge("Closed", "ok")

    preview_rows: list[dict[str, Any]] = []
    for r in rows[:400]:  # keep UI snappy
        kc = r["kind_counts"] or {}
        inc_txt = ", ".join([f"{_pretty_kind(k)}:{int(v)}" for k, v in sorted(kc.items())]) or "—"
        preview_rows.append(
            {
                "Date": (r["invoice_dt"].strftime("%Y-%m-%d") if isinstance(r["invoice_dt"], datetime) else "—"),
                "Provider": r["provider"],
                "Invoice": r["invoice_number"] or "—",
                "Order": f"#{int(r['order_id'])}",
                "Incidences": inc_txt,
                "Solution": r["solution"],
                "Credit note": r["credit_note"] or "—",
            }
        )

    st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

    st.caption("Open any row below to see expected lines, incidences and credit note matching.")

    # ---------- detailed expanders ----------
    for r in rows[:200]:
        oid = int(r["order_id"])
        provn = norm_provider(r["provider"])
        inv_no = _s(r["invoice_number"]) or "—"
        inv_dt = r["invoice_dt"]
        sol_key = _s(r["solution"])
        cn_no = _s(r["credit_note"])

        title = f"{provn} · 🧾 {inv_no} · #{oid} · {inv_dt.strftime('%Y-%m-%d') if isinstance(inv_dt, datetime) else '—'}"
        st.markdown(
            "<div class='voi-card'>"
            f"<div class='voi-title'>{html.escape(title)}</div>"
            f"<div class='voi-muted'>{_sol_badge(sol_key, cn_no)}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        with st.expander("Open invoice", expanded=False):
            ctx = _load_order_context(int(venue_id), oid)

            # Tabs: expected lines always; credit note tab only if credit note; incidences always
            tab_names = ["Invoice / expected lines", "Incidences"]
            if sol_key == "credit_note":
                tab_names.insert(1, "Credit note (matched)")

            tabs = st.tabs(tab_names)

            # --- expected lines ---
            with tabs[0]:
                # Uses your existing component (includes prices + IVA + solution column if present)
                try:
                    _render_expected_lines(
                        ctx,
                        provn,
                        show_prices=True,
                        include_iva=True,
                        supplier_resolution_by_line_id=None,
                    )
                except Exception as e:
                    st.error(f"Could not render expected lines: {e}")

            # --- credit note matching ---
            tab_offset = 1
            if sol_key == "credit_note":
                with tabs[1]:
                    if cn_no:
                        st.markdown(f"**Credit note number:** `{cn_no}`")
                    else:
                        st.warning("Credit note is the recorded solution, but the credit note number is missing.")
                    st.markdown(f"**Related invoice:** `{inv_no}`")

                    # Show supplier solution items if present (best-effort)
                    try:
                        summ = _provider_resolution_summary(ctx, provn) or {}
                        sol = summ.get("solution") or {}
                        items = sol.get("items") or []
                        if items:
                            st.markdown("**Credit note items (from supplier note):**")
                            for it in items:
                                if not isinstance(it, dict):
                                    continue
                                # Only display items that look like credit note items (not re-delivery flagged)
                                if _is_redelivery_item(it):
                                    continue
                                nm = _s(it.get("name")) or "—"
                                qty = _s(it.get("qty")) or ""
                                why = _s(it.get("why")) or ""
                                why_txt = why.replace("_", " ") if why else ""
                                st.markdown(f"- **{nm}** · {qty}" + (f" · {why_txt}" if why_txt else ""))
                        else:
                            st.caption("No structured credit note items found in supplier note.")
                    except Exception:
                        st.caption("No credit note details available.")

                tab_offset = 2

            # --- incidences ---
            with tabs[tab_offset]:
                ts = _tickets_for_provider(ctx, provn)
                if not ts:
                    st.success("No incidences recorded for this invoice/provider.")
                else:
                    # Apply the same kind filter inside (if user selected kinds)
                    if kind_sel_set:
                        ts = [t for t in ts if (_s(getattr(t, "kind", "")).strip().lower() in kind_sel_set)]

                    # Render compact list
                    cards: list[dict[str, Any]] = []
                    for t in ts:
                        kind = (_s(getattr(t, "kind", "")) or "").strip().lower()
                        state = _s(getattr(t, "state", "")) or "—"
                        pname = _s(getattr(t, "product_name", "")) or "—"
                        qty = _safe_float(getattr(t, "qty_invoiced", None), 0.0)
                        res = _ticket_resolution_str(t)
                        cards.append(
                            {
                                "Kind": _pretty_kind(kind),
                                "Product": pname,
                                "Qty": qty,
                                "Resolution": res or "—",
                                "State": state,
                            }
                        )
                    st.dataframe(pd.DataFrame(cards), use_container_width=True, hide_index=True)

