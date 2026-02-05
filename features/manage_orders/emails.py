from __future__ import annotations

"""features.manage_orders.emails

Plain-text email builders for supplier communications.

Design principles
- Centralize *all* supplier-facing email copy here.
- Always include a consistent venue footer (name, address, email, phone, AFM).
- Always include item lines: name — description · qty unit

This module is intentionally framework-agnostic (no Streamlit).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
import html


@dataclass
class VenueEmailContext:
    """Minimal venue details to include in email footers."""
    venue_name: str = ""
    owner_name: str = ""
    address: str = ""
    email: str = ""
    phone: str = ""
    tax_number: str = ""  # AFM / NIF / CIF


# ============================
# Helpers
# ============================

def _s(x: Any) -> str:
    return ("" if x is None else str(x)).strip()


def _fmt_qty(qty: Any) -> str:
    try:
        q = float(qty)
    except Exception:
        return _s(qty)

    if abs(q - int(q)) < 1e-9:
        return str(int(q))

    # avoid trailing zeros noise
    return f"{q:g}"


def normalize_lang(lang: str) -> str:
    """Normalize language codes to: en / es / gr."""
    l = _s(lang).lower()
    if l.startswith("el") or l in {"gr", "gre", "greece", "greek", "ell", "ελληνικά"}:
        return "gr"
    if l.startswith("es") or l in {"sp", "spa", "spanish", "español", "castellano"}:
        return "es"
    return "en"


def _venue_name(venue_ctx: Any) -> str:
    return _s(getattr(venue_ctx, "venue_name", None) or getattr(venue_ctx, "name", None) or "")


def _venue_email_lang(venue_ctx: Any, default: str = "en") -> str:
    return normalize_lang(_s(getattr(venue_ctx, "email_lang", None) or default))


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


# ============================
# Footer + Items
# ============================

def build_venue_footer(ctx: Any) -> str:
    """Always-on footer with venue details."""
    vname = _venue_name(ctx)
    owner = _s(getattr(ctx, "owner_name", None) or "")
    addr = _s(getattr(ctx, "address", None) or "")
    email = _s(getattr(ctx, "email", None) or "")
    phone = _s(getattr(ctx, "phone", None) or "")
    tax = _s(getattr(ctx, "tax_number", None) or getattr(ctx, "tax", None) or "")

    lines: List[str] = []
    lines.append("")
    lines.append("—" * 38)
    if vname:
        lines.append(vname)
    if owner:
        lines.append(owner)
    if addr:
        lines.append(addr)
    if email:
        lines.append(f"Email: {email}")
    if phone:
        lines.append(f"Tel: {phone}")
    if tax:
        # user request: always label as AFM
        lines.append(f"AFM: {tax}")

    return "\n".join(lines).rstrip()


def format_product_lines(items: Iterable[Dict[str, Any]], *, fallback_unit: str = "unit") -> List[str]:
    """Format item lines consistently.

    Output:
      - Name — Description · qty unit

    Description is omitted if empty.
    """
    out: List[str] = []
    for it in (items or []):
        name = _s(it.get("name") or "Product")
        desc = _s(it.get("description") or it.get("desc") or "")
        qty = _fmt_qty(it.get("qty") if it.get("qty") is not None else it.get("quantity"))
        unit = _s(it.get("unit") or fallback_unit)
        left = f"{name} — {desc}" if desc else name
        out.append(f"- {left} · {qty} {unit}".rstrip())
    return out


# ============================
# HTML helpers (email-safe)
# ============================

def _h(x: Any) -> str:
    """HTML-escape helper."""
    return html.escape(_s(x), quote=True)


def _html_items_table(items: Iterable[Dict[str, Any]], *, fallback_unit: str = "unit") -> str:
    """Simple, robust table for items (no CSS classes)."""
    rows: List[str] = []
    for it in (items or []):
        name = _h(it.get("name") or "Product")
        desc_raw = _s(it.get("description") or it.get("desc") or "")
        desc = _h(desc_raw)
        qty = _h(_fmt_qty(it.get("qty") if it.get("qty") is not None else it.get("quantity")))
        unit = _h(it.get("unit") or fallback_unit)
        left = f"{name}<div style=\"font-size:13px;line-height:18px;color:#64748b;margin-top:2px;\">{desc}</div>" if desc_raw else name
        rows.append(
            "<tr>"
            "<td style=\"padding:10px 0;border-bottom:1px solid #e5e7eb;\">"
            f"<div style=\"font-size:15px;line-height:20px;font-weight:700;color:#0f172a;\">{left}</div>"
            "</td>"
            "<td style=\"padding:10px 0;border-bottom:1px solid #e5e7eb;text-align:right;white-space:nowrap;\">"
            f"<div style=\"font-size:15px;line-height:20px;font-weight:800;color:#0f172a;\">{qty} {unit}</div>"
            "</td>"
            "</tr>"
        )

    if not rows:
        rows.append(
            "<tr><td style=\"padding:10px 0;color:#64748b;\">(no items)</td><td></td></tr>"
        )

    return (
        "<table width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" style=\"border-collapse:collapse;\">"
        + "".join(rows)
        + "</table>"
    )


def _html_footer(ctx: Any) -> str:
    """Footer block in HTML (mirrors build_venue_footer())."""
    vname = _h(_venue_name(ctx))
    owner = _h(getattr(ctx, "owner_name", "") or "")
    addr = _h(getattr(ctx, "address", "") or "")
    email_v = _h(getattr(ctx, "email", "") or "")
    phone = _h(getattr(ctx, "phone", "") or "")
    tax = _h(getattr(ctx, "tax_number", "") or getattr(ctx, "tax", "") or "")

    parts: List[str] = []
    if vname:
        parts.append(f"<div style=\"font-weight:800;\">{vname}</div>")
    if owner:
        parts.append(f"<div>{owner}</div>")
    if addr:
        parts.append(f"<div>{addr}</div>")
    if email_v:
        parts.append(f"<div>Email: {email_v}</div>")
    if phone:
        parts.append(f"<div>Tel: {phone}</div>")
    if tax:
        parts.append(f"<div>AFM: {tax}</div>")

    if not parts:
        parts.append("<div style=\"opacity:.7;\">(missing venue data)</div>")

    return (
        "<div style=\"margin-top:18px;padding-top:14px;border-top:1px solid #e5e7eb;"
        "font-size:12px;line-height:18px;color:#64748b;\">"
        + "".join(parts)
        + "</div>"
    )


def _html_button(*, url: str, label: str, bg: str) -> str:
    """Bulletproof, mobile-friendly CTA button.

    - Uses a table wrapper for Outlook compatibility
    - Target size ~48px height
    - Big font for thumbs
    """
    u = _h(url)
    t = _h(label)
    # 'mso-padding-alt' is used by Outlook
    return (
        "<table role=\"presentation\" cellpadding=\"0\" cellspacing=\"0\" border=\"0\" "
        "style=\"margin:18px auto 0 auto;\">"
        "<tr><td align=\"center\" bgcolor=\"" + bg + "\" "
        "style=\"border-radius:10px;\">"
        f"<a href=\"{u}\" target=\"_blank\" "
        "style=\"display:inline-block;min-width:240px;max-width:520px;"
        "padding:16px 22px;font-size:16px;line-height:20px;"
        "font-weight:800;font-family:Arial,Helvetica,sans-serif;"
        "color:#ffffff;text-decoration:none;border-radius:10px;"
        "mso-padding-alt:16px 22px;\">"
        + t
        + "</a>"
        "</td></tr></table>"
        "<div style=\"height:6px;line-height:6px;font-size:6px;\">&nbsp;</div>"
        "<div style=\"text-align:center;font-size:12px;line-height:18px;color:#64748b;\">"
        "If the button doesn’t work, copy/paste this link:<br/>"
        f"<span style=\"word-break:break-all;\">{u}</span></div>"
    )


def _wrap_email_html(*, title: str, subtitle: str, body_html: str, tone: str = "normal") -> str:
    """Shared layout wrapper (600px), safe for Gmail/Outlook."""
    # Tone controls the top-bar color for urgency.
    bar = "#dc2626" if tone == "urgent" else "#2563eb"
    return (
        "<!DOCTYPE html>"
        "<html><head><meta charset=\"utf-8\"></head>"
        "<body style=\"margin:0;padding:0;background:#f6f7f9;\">"
        "<table width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" border=\"0\">"
        "<tr><td align=\"center\" style=\"padding:16px 10px;\">"
        "<table width=\"600\" cellpadding=\"0\" cellspacing=\"0\" border=\"0\" "
        "style=\"width:600px;max-width:600px;background:#ffffff;border-radius:14px;"
        "overflow:hidden;font-family:Arial,Helvetica,sans-serif;\">"
        f"<tr><td style=\"height:6px;background:{bar};line-height:6px;font-size:6px;\">&nbsp;</td></tr>"
        "<tr><td style=\"padding:18px 20px 8px 20px;\">"
        f"<div style=\"font-size:18px;line-height:24px;font-weight:900;color:#0f172a;\">{_h(title)}</div>"
        f"<div style=\"margin-top:4px;font-size:13px;line-height:18px;color:#64748b;\">{_h(subtitle)}</div>"
        "</td></tr>"
        "<tr><td style=\"padding:12px 20px 20px 20px;\">"
        + body_html
        + "</td></tr>"
        "</table>"
        "</td></tr></table></body></html>"
    )


# ============================
# Subjects (consistent)
# ============================

def build_order_subject(*, venue_ctx: Any, order_id: int, provider_name: str, lang: str | None = None) -> str:
    """Subject for initial order email."""
    l = normalize_lang(lang or _venue_email_lang(venue_ctx, "es"))
    venue = _venue_name(venue_ctx) or "Venue"
    provider = _s(provider_name) or "Supplier"
    date_str = _today()

    if l == "gr":
        return f"[{venue}] Παραγγελία #{int(order_id)} — {provider} — {date_str}"
    if l == "es":
        return f"[{venue}] Pedido #{int(order_id)} — {provider} — {date_str}"
    return f"[{venue}] Order #{int(order_id)} — {provider} — {date_str}"


def build_resolution_subject(
    *,
    venue_ctx: Any,
    order_id: int,
    provider_name: str,
    invoice_ref: str = "",
    lang: str | None = None,
) -> str:
    """Subject for supplier action / resolution request."""
    l = normalize_lang(lang or _venue_email_lang(venue_ctx, "en"))
    venue = _venue_name(venue_ctx) or "Venue"
    provider = _s(provider_name) or "Supplier"
    inv = _s(invoice_ref)
    inv_txt = f" — Invoice {inv}" if inv else ""

    if l == "gr":
        return f"[{venue}] Απαιτείται ενέργεια — Παραγγελία #{int(order_id)} — {provider}{inv_txt}"
    if l == "es":
        return f"[{venue}] Acción requerida — Pedido #{int(order_id)} — {provider}{inv_txt}"
    return f"[{venue}] Action required — Order #{int(order_id)} — {provider}{inv_txt}"


def build_urgent_subject(
    *,
    venue_ctx: Any,
    order_id: int,
    provider_name: str,
    source_order_id: Optional[int] = None,
    lang: str | None = None,
) -> str:
    """Subject for urgent re-order request."""
    l = normalize_lang(lang or _venue_email_lang(venue_ctx, "en"))
    venue = _venue_name(venue_ctx) or "Venue"
    provider = _s(provider_name) or "Supplier"
    src = f" (from #{int(source_order_id)})" if source_order_id else ""

    if l == "gr":
        return f"[{venue}] ΕΠΕΙΓΟΝ — Επαναπαραγγελία #{int(order_id)}{src} — {provider}"
    if l == "es":
        return f"[{venue}] URGENTE — #{int(order_id)}— {provider}"
    return f"[{venue}] URGENT — Re-order #{int(order_id)}{src} — {provider}"


# ============================
# Bodies
# ============================

def build_order_email(
    *,
    venue_ctx: Any,
    order_id: int,
    provider_name: str,
    supplier_link: str,
    items: List[Dict[str, Any]],
    lang: str = "es",
) -> str:
    l = normalize_lang(lang)
    venue = _venue_name(venue_ctx) or ("Pedido" if l == "es" else "Order")
    date_str = _today()

    txt: List[str] = []
    if l == "gr":
        txt.append(f"{venue} — Παραγγελία #{int(order_id)} — {date_str}")
        txt.append("")
        txt.append("Γεια σας,")
        txt.append("Σας στέλνουμε την παραγγελία και το link για επιβεβαίωση αποστολής (full / partial / none).")
        txt.append("")
        txt.append("LINK (επιβεβαίωση προμηθευτή):")
    elif l == "en":
        txt.append(f"{venue} — Order #{int(order_id)} — {date_str}")
        txt.append("")
        txt.append("Hello,")
        txt.append("Here is the order and the link to confirm delivery (full / partial / none).")
        txt.append("")
        txt.append("LINK (supplier confirmation):")
    else:
        txt.append(f"{venue} — Pedido #{int(order_id)} — {date_str}")
        txt.append("")
        txt.append("Hola,")
        txt.append("Te comparto el pedido y el link para confirmar el envío (full / partial / none).")
        txt.append("")
        txt.append("LINK (confirmación proveedor):")

    txt.append(supplier_link)
    txt.append("")

    if l == "gr":
        txt.append("ΠΑΡΑΓΓΕΛΙΑ:")
    elif l == "en":
        txt.append("ORDER:")
    else:
        txt.append("PEDIDO:")

    txt.extend(format_product_lines(items, fallback_unit=("unidad" if l == "es" else "unit")))
    txt.append(build_venue_footer(venue_ctx))

    return "\n".join(txt).strip()


def build_order_email_html(
    *,
    venue_ctx: Any,
    order_id: int,
    provider_name: str,
    supplier_link: str,
    items: List[Dict[str, Any]],
    lang: str = "es",
) -> str:
    """HTML version of the normal order email."""
    l = normalize_lang(lang)
    venue = _venue_name(venue_ctx) or "Venue"
    provider = _s(provider_name) or "Supplier"
    date_str = _today()

    if l == "gr":
        title = "Παραγγελία"
        intro = "Σας στέλνουμε την παραγγελία και το link για επιβεβαίωση αποστολής (full / partial / none)."
        btn = "Άνοιγμα παραγγελίας & επιβεβαίωση"
        items_hdr = "Προϊόντα"
        fallback_unit = "τεμ"
    elif l == "en":
        title = "Order"
        intro = "Here is the order and the link to confirm delivery (full / partial / none)."
        btn = "Open order & confirm"
        items_hdr = "Items"
        fallback_unit = "unit"
    else:
        title = "Pedido"
        intro = "Te comparto el pedido y el link para confirmar el envío (full / partial / none)."
        btn = "Abrir pedido y confirmar"
        items_hdr = "Productos"
        fallback_unit = "unidad"

    subtitle = f"#{int(order_id)} · {provider} · {date_str}"

    body_html = (
        f"<div style=\"font-size:14px;line-height:20px;color:#0f172a;\">"
        f"<div style=\"margin:0 0 10px 0;\">{_h(intro)}</div>"
        f"<div style=\"margin-top:14px;font-weight:900;\">{_h(items_hdr)}:</div>"
        + _html_items_table(items, fallback_unit=fallback_unit)
        + _html_button(url=supplier_link, label=btn, bg="#2563eb")
        + _html_footer(venue_ctx)
        + "</div>"
    )

    return _wrap_email_html(title=f"{venue} — {title}", subtitle=subtitle, body_html=body_html, tone="normal")


def build_resolution_request_email(
    *,
    venue_ctx: Any,
    order_id: int,
    provider_name: str,
    supplier_link: str,
    items: List[Dict[str, Any]],
    invoice_ref: str = "",
    lang: str = "en",
) -> str:
    l = normalize_lang(lang)
    inv = _s(invoice_ref)

    txt: List[str] = []
    if l == "gr":
        txt.append("Απαιτείται ενέργεια — Διαφορά τιμολογίου / θέμα παραλαβής")
        txt.append("")
        txt.append("Παρακαλούμε ανοίξτε το link και επιλέξτε λύση (πιστωτικό ή συμπληρωματική παράδοση):")
    elif l == "es":
        txt.append("Acción requerida — Incidencia de factura / recepción")
        txt.append("")
        txt.append("Por favor abre el link y elige una resolución (abono o re-entrega):")
    else:
        txt.append("Action required — Invoice discrepancy / receiving issue")
        txt.append("")
        txt.append("Please open the link and choose a resolution (credit note or re-delivery):")

    # Add invoice reference when available
    if inv:
        txt.append("")
        if l == "gr":
            txt.append(f"Αρ. τιμολογίου: {inv}")
        elif l == "es":
            txt.append(f"Factura: {inv}")
        else:
            txt.append(f"Invoice: {inv}")

    txt.append("")
    txt.append(supplier_link)
    txt.append("")

    if l == "gr":
        txt.append("ΠΡΟΪΟΝΤΑ:")
    elif l == "es":
        txt.append("PRODUCTOS:")
    else:
        txt.append("ITEMS:")

    txt.extend(format_product_lines(items, fallback_unit=("unidad" if l == "es" else "unit")))
    txt.append(build_venue_footer(venue_ctx))

    return "\n".join(txt).strip()


def build_urgent_reorder_email(
    *,
    venue_ctx: Any,
    order_id: int,
    provider_name: str,
    supplier_link: str,
    items: List[Dict[str, Any]],
    source_order_id: Optional[int] = None,
    lang: str = "en",
) -> str:
    l = normalize_lang(lang)
    src_txt = f" (from order #{int(source_order_id)})" if source_order_id else ""

    txt: List[str] = []
    if l == "gr":
        txt.append("ΕΠΕΙΓΟΝ")
        txt.append(f"Παραγγελία: #{int(order_id)}")
        txt.append(f"Προμηθευτής: {_s(provider_name)}")
        txt.append("")
        txt.extend(format_product_lines(items, fallback_unit="τεμ"))
        txt.append("")
        txt.append("Παρακαλούμε ανοίξτε το link για επιβεβαίωση/συντονισμό παράδοσης:")
    elif l == "es":
        txt.append("URGENTE")
        txt.append(f"Pedido: #{int(order_id)}")
        txt.append(f"Proveedor: {_s(provider_name)}")
        txt.append("")
        txt.extend(format_product_lines(items, fallback_unit="unidad"))
        txt.append("")
        txt.append("Por favor abre el link para confirmar/ver y coordinar la entrega:")
    else:
        txt.append("URGENT")
        txt.append(f"Οrder: #{int(order_id)}")
        txt.append(f"Provider: {_s(provider_name)}")
        txt.append("")
        txt.extend(format_product_lines(items, fallback_unit="unit"))
        txt.append("")
        txt.append("Please open the link to view/confirm and coordinate delivery:")

    txt.append(supplier_link)
    txt.append(build_venue_footer(venue_ctx))

    return "\n".join(txt).strip()



def build_urgent_reorder_email_html(
    *,
    venue_ctx: Any,
    order_id: int,
    provider_name: str,
    supplier_link: str,
    items: List[Dict[str, Any]],
    source_order_id: Optional[int] = None,
    lang: str = "en",
) -> str:
    """HTML version of urgent re-order request."""
    l = normalize_lang(lang)
    venue = _venue_name(venue_ctx) or "Venue"
    provider = _s(provider_name) or "Supplier"
    src_txt = f" · from #{int(source_order_id)}" if source_order_id else ""

    if l == "gr":
        title = "ΕΠΕΙΓΟΝ — Επαναπαραγγελία"
        intro = "Παρακαλούμε επιβεβαιώστε διαθεσιμότητα και συντονίστε την παράδοση μέσω του link."
        btn = "Άνοιγμα & επιβεβαίωση επείγοντος"
        items_hdr = "Προϊόντα"
        fallback_unit = "τεμ"
    elif l == "es":
        title = "URGENTE — pedido"
        intro = "Por favor confirma disponibilidad y coordina la entrega usando el link."
        btn = "Abrir y confirmar urgente"
        items_hdr = "Productos"
        fallback_unit = "unidad"
    else:
        title = "URGENT — Re-order"
        intro = "Please confirm availability and coordinate delivery using the link."
        btn = "Open & confirm urgent"
        items_hdr = "Items"
        fallback_unit = "unit"

    subtitle = f"#{int(order_id)} · {provider}{src_txt}"

    body_html = (
        f"<div style=\"font-size:14px;line-height:20px;color:#0f172a;\">"
        f"<div style=\"display:inline-block;padding:6px 10px;border-radius:999px;"
        f"background:#fef2f2;color:#991b1b;font-weight:900;font-size:12px;\">{_h(title)}</div>"
        f"<div style=\"margin-top:10px;\">{_h(intro)}</div>"
        f"<div style=\"margin-top:14px;font-weight:900;\">{_h(items_hdr)}:</div>"
        + _html_items_table(items, fallback_unit=fallback_unit)
        + _html_button(url=supplier_link, label=btn, bg="#dc2626")
        + _html_footer(venue_ctx)
        + "</div>"
    )

    return _wrap_email_html(title=f"{venue} — {title}", subtitle=subtitle, body_html=body_html, tone="urgent")


# ============================
# Convenience: subject + body tuples
# ============================

def build_order_email_full(
    *,
    venue_ctx: Any,
    order_id: int,
    provider_name: str,
    supplier_link: str,
    items: List[Dict[str, Any]],
    lang: str = "es",
) -> Tuple[str, str]:
    subject = build_order_subject(venue_ctx=venue_ctx, order_id=order_id, provider_name=provider_name, lang=lang)
    body = build_order_email(
        venue_ctx=venue_ctx,
        order_id=order_id,
        provider_name=provider_name,
        supplier_link=supplier_link,
        items=items,
        lang=lang,
    )
    return subject, body


def build_resolution_email_full(
    *,
    venue_ctx: Any,
    order_id: int,
    provider_name: str,
    supplier_link: str,
    items: List[Dict[str, Any]],
    invoice_ref: str = "",
    lang: str = "en",
) -> Tuple[str, str]:
    subject = build_resolution_subject(
        venue_ctx=venue_ctx,
        order_id=order_id,
        provider_name=provider_name,
        invoice_ref=invoice_ref,
        lang=lang,
    )
    body = build_resolution_request_email(
        venue_ctx=venue_ctx,
        order_id=order_id,
        provider_name=provider_name,
        supplier_link=supplier_link,
        items=items,
        invoice_ref=invoice_ref,
        lang=lang,
    )
    return subject, body


def build_urgent_email_full(
    *,
    venue_ctx: Any,
    order_id: int,
    provider_name: str,
    supplier_link: str,
    items: List[Dict[str, Any]],
    source_order_id: Optional[int] = None,
    lang: str = "en",
) -> Tuple[str, str]:
    subject = build_urgent_subject(
        venue_ctx=venue_ctx,
        order_id=order_id,
        provider_name=provider_name,
        source_order_id=source_order_id,
        lang=lang,
    )
    body = build_urgent_reorder_email(
        venue_ctx=venue_ctx,
        order_id=order_id,
        provider_name=provider_name,
        supplier_link=supplier_link,
        items=items,
        source_order_id=source_order_id,
        lang=lang,
    )
    return subject, body


# ============================
# Convenience: subject + text + html tuples
# ============================

def build_order_email_full_html(
    *,
    venue_ctx: Any,
    order_id: int,
    provider_name: str,
    supplier_link: str,
    items: List[Dict[str, Any]],
    lang: str = "es",
) -> Tuple[str, str, str]:
    """Return (subject, text_body, html_body) for normal order email."""
    subject, text_body = build_order_email_full(
        venue_ctx=venue_ctx,
        order_id=order_id,
        provider_name=provider_name,
        supplier_link=supplier_link,
        items=items,
        lang=lang,
    )
    html_body = build_order_email_html(
        venue_ctx=venue_ctx,
        order_id=order_id,
        provider_name=provider_name,
        supplier_link=supplier_link,
        items=items,
        lang=lang,
    )
    return subject, text_body, html_body


def build_urgent_email_full_html(
    *,
    venue_ctx: Any,
    order_id: int,
    provider_name: str,
    supplier_link: str,
    items: List[Dict[str, Any]],
    source_order_id: Optional[int] = None,
    lang: str = "en",
) -> Tuple[str, str, str]:
    """Return (subject, text_body, html_body) for urgent re-order request."""
    subject, text_body = build_urgent_email_full(
        venue_ctx=venue_ctx,
        order_id=order_id,
        provider_name=provider_name,
        supplier_link=supplier_link,
        items=items,
        source_order_id=source_order_id,
        lang=lang,
    )
    html_body = build_urgent_reorder_email_html(
        venue_ctx=venue_ctx,
        order_id=order_id,
        provider_name=provider_name,
        supplier_link=supplier_link,
        items=items,
        source_order_id=source_order_id,
        lang=lang,
    )
    return subject, text_body, html_body