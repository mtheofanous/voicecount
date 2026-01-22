"""
modern_receive.py - Μοντέρνο σύστημα παρακολούθησης παραγγελιών & επίλυσης προβλημάτων

Φιλοσοφία:
1. Συγχωνευμένη προβολή όλων των υπό διεκπεραίωση παραγγελιών
2. Αυτόματη ανίχνευση & διαχείριση προβλημάτων
3. Διαδραστική επικοινωνία με προμηθευτές
4. Οπτικά πλούσιο dashboard με real-time ενημέρωση

Δομή:
- 📊 Dashboard επισκόπησης (όλα τα υπό διεκπεραίωση)
- 🔍 Λεπτομερής προβολή ανά προμηθευτή
- 🚨 Έξυπνο σύστημα εντοπισμού προβλημάτων
- 📈 Αναλυτικές μετρήσεις & αναφορές
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List, Tuple
from enum import Enum
import streamlit as st
from sqlmodel import select, func, and_, or_
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from core.db import get_session
from core.public_links import build_seguimiento_url, ROLE_SUPPLIER, ROLE_VENUE, norm_provider
from core.mailer import send_smtp_email
from domain.models import (
    Order, OrderLine, Product, Provider,
    OrderWorkflow, OrderWorkflowEvent, SeguimientoTicket,
    ProviderLineFollowUp, ProviderReceipt, ProviderSendStatus
)


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class IssueType(str, Enum):
    """Τύποι προβλημάτων που μπορούν να προκύψουν"""
    INVOICE_DISCREPANCY = "invoice_discrepancy"      # Διαφορά τιμολογίου-παράδοσης
    PARTIAL_DELIVERY = "partial_delivery"           # Μερική παράδοση
    MISSING_PRODUCT = "missing_product"             # Ελλείπουσα παραγγελία
    QUALITY_ISSUE = "quality_issue"                 # Ποιότητα προϊόντων
    DELAY = "delay"                                 # Καθυστέρηση
    WRONG_PRODUCT = "wrong_product"                 # Λάθος προϊόν
    PRICE_DISCREPANCY = "price_discrepancy"         # Διαφορά τιμής


class ResolutionType(str, Enum):
    """Τρόποι επίλυσης προβλημάτων"""
    CREDIT_NOTE = "credit_note"                     # Πίστωση
    REPLACEMENT = "replacement"                     # Αντικατάσταση
    PARTIAL_REFUND = "partial_refund"              # Μερική επιστροφή
    FUTURE_DISCOUNT = "future_discount"            # Έκπτωση σε μελλοντική
    NO_ACTION = "no_action"                        # Δεν απαιτείται ενέργεια


# =============================================================================
# CORE UI COMPONENTS
# =============================================================================

def _inject_modern_css() -> None:
    """Ενσωμάτωση μοντέρνου CSS για όλη την εφαρμογή"""
    st.markdown("""
    <style>
    /* ΒΑΣΙΚΑ STYLES */
    :root {
        --primary: #3B82F6;
        --primary-dark: #2563EB;
        --secondary: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --info: #8B5CF6;
        --light: #F8FAFC;
        --dark: #1E293B;
        --gray: #64748B;
        --gray-light: #E2E8F0;
        --radius: 12px;
        --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
    }
    
    /* ΚΑΡΤΕΣ */
    .tracking-card {
        background: white;
        border-radius: var(--radius);
        border: 1px solid var(--gray-light);
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
        box-shadow: var(--shadow);
    }
    
    .tracking-card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
    }
    
    .tracking-card--warning {
        border-left: 4px solid var(--warning);
    }
    
    .tracking-card--danger {
        border-left: 4px solid var(--danger);
    }
    
    .tracking-card--success {
        border-left: 4px solid var(--secondary);
    }
    
    .tracking-card--info {
        border-left: 4px solid var(--info);
    }
    
    /* ΜΠΑΤΖΑΚΙΑ */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        line-height: 1;
    }
    
    .badge--primary {
        background-color: #EFF6FF;
        color: var(--primary);
        border: 1px solid #DBEAFE;
    }
    
    .badge--success {
        background-color: #ECFDF5;
        color: var(--secondary);
        border: 1px solid #D1FAE5;
    }
    
    .badge--warning {
        background-color: #FFFBEB;
        color: var(--warning);
        border: 1px solid #FEF3C7;
    }
    
    .badge--danger {
        background-color: #FEF2F2;
        color: var(--danger);
        border: 1px solid #FECACA;
    }
    
    .badge--info {
        background-color: #F5F3FF;
        color: var(--info);
        border: 1px solid #DDD6FE;
    }
    
    /* ΠΡΟΟΔΟΣ */
    .progress-bar {
        height: 8px;
        border-radius: 4px;
        background-color: var(--gray-light);
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .progress-bar__fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    /* TIMELINE */
    .timeline {
        position: relative;
        padding-left: 1.5rem;
        margin: 1rem 0;
    }
    
    .timeline::before {
        content: '';
        position: absolute;
        left: 7px;
        top: 0;
        bottom: 0;
        width: 2px;
        background-color: var(--gray-light);
    }
    
    .timeline-item {
        position: relative;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--gray-light);
    }
    
    .timeline-item:last-child {
        border-bottom: none;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    
    .timeline-item::before {
        content: '';
        position: absolute;
        left: -1.5rem;
        top: 5px;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background-color: var(--primary);
        border: 2px solid white;
        box-shadow: 0 0 0 2px var(--primary);
    }
    
    .timeline-item--warning::before {
        background-color: var(--warning);
        box-shadow: 0 0 0 2px var(--warning);
    }
    
    .timeline-item--success::before {
        background-color: var(--secondary);
        box-shadow: 0 0 0 2px var(--secondary);
    }
    
    /* GRID */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-card {
        background: white;
        border-radius: var(--radius);
        padding: 1.25rem;
        border: 1px solid var(--gray-light);
        text-align: center;
    }
    
    .stat-card__value {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
        margin: 0.5rem 0;
    }
    
    .stat-card__label {
        font-size: 0.875rem;
        color: var(--gray);
    }
    
    /* BUTTONS */
    .button-group {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    
    /* TOOLTIPS */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted var(--gray);
        cursor: help;
    }
    
    .tooltip:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: var(--dark);
        color: white;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        font-size: 0.75rem;
        white-space: nowrap;
        z-index: 1000;
        box-shadow: var(--shadow);
    }
    </style>
    """, unsafe_allow_html=True)


def badge(text: str, variant: str = "primary") -> str:
    """Επιστρέφει HTML για μπατζάκι"""
    return f'<span class="badge badge--{variant}">{text}</span>'


def card(content: str, variant: str = "", css_class: str = "") -> None:
    """Εμφανίζει μια κάρτα"""
    variant_class = f" tracking-card--{variant}" if variant else ""
    st.markdown(f'<div class="tracking-card{variant_class} {css_class}">{content}</div>', unsafe_allow_html=True)


def stat_card(value: Any, label: str, icon: str = "", color: str = "primary") -> None:
    """Εμφανίζει μια στατιστική κάρτα"""
    icon_html = f'<div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>' if icon else ""
    st.markdown(f"""
    <div class="stat-card">
        {icon_html}
        <div class="stat-card__value" style="color: var(--{color})">{value}</div>
        <div class="stat-card__label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def progress_bar(percentage: float, color: str = "primary", label: str = "") -> None:
    """Εμφανίζει μπάρα προόδου"""
    color_map = {
        "primary": "#3B82F6",
        "success": "#10B981",
        "warning": "#F59E0B",
        "danger": "#EF4444"
    }
    fill_color = color_map.get(color, color_map["primary"])
    
    label_html = f'<div style="font-size: 0.875rem; margin-bottom: 0.25rem; color: var(--gray)">{label}</div>' if label else ""
    
    st.markdown(f"""
    {label_html}
    <div class="progress-bar">
        <div class="progress-bar__fill" style="width: {percentage}%; background-color: {fill_color}"></div>
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--gray); margin-top: 0.25rem;">
        <span>0%</span>
        <span>{percentage:.0f}%</span>
        <span>100%</span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# DATA HELPERS
# =============================================================================

def _s(x: Any) -> str:
    """Ασφαλής μετατροπή σε string"""
    return ("" if x is None else str(x)).strip()


def _f(x: Any) -> float:
    """Ασφαλής μετατροπή σε float"""
    try:
        return float(x or 0)
    except:
        return 0.0


def _now() -> datetime:
    return datetime.utcnow()


def get_active_orders(venue_id: int) -> List[Order]:
    """Επιστρέφει ενεργές παραγγελίες"""
    with get_session() as s:
        return list(s.exec(
            select(Order)
            .where(
                Order.venue_id == venue_id,
                Order.status.in_(["ready_to_send", "pending_receive"])
            )
            .order_by(Order.created_at.desc())
        ).all())


def get_order_details(order_id: int) -> Tuple[Order, List[OrderLine], Dict[str, Any]]:
    """Επιστρέφει όλες τις πληροφορίες μιας παραγγελίας"""
    with get_session() as s:
        # Βασική παραγγελία
        order = s.exec(select(Order).where(Order.id == order_id)).first()
        if not order:
            raise ValueError(f"Order {order_id} not found")
        
        # Γραμμές παραγγελίας
        lines = list(s.exec(select(OrderLine).where(OrderLine.order_id == order_id)).all())
        
        # Προϊόντα
        product_ids = [l.product_id for l in lines if l.product_id]
        products = {}
        if product_ids:
            products_list = list(s.exec(select(Product).where(Product.id.in_(product_ids))).all())
            products = {p.id: p for p in products_list}
        
        # Workflow status per provider
        workflows = list(s.exec(
            select(OrderWorkflow)
            .where(OrderWorkflow.order_id == order_id)
        ).all())
        
        # Tickets
        tickets = list(s.exec(
            select(SeguimientoTicket)
            .where(SeguimientoTicket.order_id == order_id)
        ).all())
        
        # Provider receipts
        receipts = list(s.exec(
            select(ProviderReceipt)
            .where(ProviderReceipt.order_id == order_id)
        ).all())
        
        return order, lines, {
            "products": products,
            "workflows": {w.provider_name: w for w in workflows},
            "tickets": tickets,
            "receipts": {r.provider_name: r for r in receipts}
        }


def get_provider_performance(venue_id: int, days: int = 30) -> List[Dict[str, Any]]:
    """Επιστρέφει στατιστικά απόδοσης προμηθευτών"""
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    with get_session() as s:
        # Αριθμός παραγγελιών ανά προμηθευτή
        order_counts = s.exec(
            select(OrderLine.provider, func.count(func.distinct(OrderLine.order_id)))
            .join(Order, Order.id == OrderLine.order_id)
            .where(
                Order.venue_id == venue_id,
                Order.created_at >= cutoff_date,
                OrderLine.provider.is_not(None)
            )
            .group_by(OrderLine.provider)
        ).all()
        
        # Προβλήματα ανά προμηθευτή
        issue_counts = s.exec(
            select(SeguimientoTicket.provider_name, func.count(SeguimientoTicket.id))
            .where(
                SeguimientoTicket.venue_id == venue_id,
                SeguimientoTicket.created_at >= cutoff_date,
                SeguimientoTicket.state == "open"
            )
            .group_by(SeguimientoTicket.provider_name)
        ).all()
        
        # Χρόνος επίλυσης
        resolved_tickets = s.exec(
            select(SeguimientoTicket.provider_name, 
                   func.avg(func.extract('epoch', SeguimientoTicket.resolved_at - SeguimientoTicket.created_at)))
            .where(
                SeguimientoTicket.venue_id == venue_id,
                SeguimientoTicket.resolved_at.is_not(None),
                SeguimientoTicket.created_at >= cutoff_date
            )
            .group_by(SeguimientoTicket.provider_name)
        ).all()
        
    # Συγχώνευση δεδομένων
    performance = {}
    for provider, count in order_counts:
        if provider:
            provider_norm = norm_provider(provider)
            performance[provider_norm] = {
                "total_orders": count,
                "open_issues": 0,
                "avg_resolution_hours": 0,
                "score": 100  # Βασικό σκορ
            }
    
    for provider, issues in issue_counts:
        if provider and provider in performance:
            performance[provider]["open_issues"] = issues
            performance[provider]["score"] -= issues * 10  # Αφαίρεση πόντων για προβλήματα
    
    for provider, avg_seconds in resolved_tickets:
        if provider and provider in performance:
            avg_hours = avg_seconds / 3600 if avg_seconds else 0
            performance[provider]["avg_resolution_hours"] = round(avg_hours, 1)
            if avg_hours < 24:
                performance[provider]["score"] += 10  # Bonus για γρήγορη επίλυση
    
    return [
        {
            "provider": provider,
            **data
        }
        for provider, data in sorted(
            performance.items(), 
            key=lambda x: x[1]["score"], 
            reverse=True
        )
    ]


# =============================================================================
# DASHBOARD COMPONENTS
# =============================================================================

def render_dashboard_overview(venue_id: int):
    """Κεντρικό dashboard με επισκόπηση"""
    st.markdown("## 📊 Dashboard Παρακολούθησης")
    
    # Γρήγορη στατιστική
    with get_session() as s:
        # Στατιστικά
        total_orders = s.exec(
            select(func.count(Order.id))
            .where(Order.venue_id == venue_id, Order.status == "pending_receive")
        ).first() or 0
        
        open_issues = s.exec(
            select(func.count(SeguimientoTicket.id))
            .where(
                SeguimientoTicket.venue_id == venue_id,
                SeguimientoTicket.state == "open"
            )
        ).first() or 0
        
        pending_suppliers = s.exec(
            select(func.count(func.distinct(OrderWorkflow.provider_name)))
            .where(
                OrderWorkflow.venue_id == venue_id,
                OrderWorkflow.state == "ORDER_SENT"
            )
        ).first() or 0
        
        avg_resolution = s.exec(
            select(func.avg(func.extract('epoch', SeguimientoTicket.resolved_at - SeguimientoTicket.created_at)))
            .where(
                SeguimientoTicket.venue_id == venue_id,
                SeguimientoTicket.resolved_at.is_not(None)
            )
        ).first() or 0
        avg_resolution_hours = round((avg_resolution or 0) / 3600, 1)
    
    # Εμφάνιση στατιστικών
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        stat_card(total_orders, "Υπό διεκπεραίωση", "📦", "primary")
    with col2:
        stat_card(open_issues, "Ανοιχτά προβλήματα", "🚨", "danger")
    with col3:
        stat_card(pending_suppliers, "Εκκρεμούν απάντηση", "⏳", "warning")
    with col4:
        stat_card(f"{avg_resolution_hours}ω", "Μ.Ο. επίλυσης", "⚡", "success")
    
    st.markdown("---")
    
    # Γραφήματα απόδοσης
    performance_data = get_provider_performance(venue_id)
    if performance_data:
        df = pd.DataFrame(performance_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Απόδοση Προμηθευτών")
            fig = px.bar(
                df.head(10),
                x="provider",
                y="score",
                color="score",
                color_continuous_scale="RdYlGn",
                labels={"provider": "Προμηθευτής", "score": "Σκορ"}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### ⏱️ Χρόνος Επίλυσης")
            fig = px.scatter(
                df,
                x="total_orders",
                y="avg_resolution_hours",
                size="open_issues",
                color="provider",
                hover_name="provider",
                labels={
                    "total_orders": "Συνολικές παραγγελίες",
                    "avg_resolution_hours": "Μέσος χρόνος επίλυσης (ώρες)",
                    "open_issues": "Ανοιχτά προβλήματα"
                }
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Πρόσφατες δραστηριότητες
    st.markdown("### 🔔 Πρόσφατη Δραστηριότητα")
    with get_session() as s:
        recent_events = s.exec(
            select(OrderWorkflowEvent)
            .join(Order, Order.id == OrderWorkflowEvent.order_id)
            .where(Order.venue_id == venue_id)
            .order_by(OrderWorkflowEvent.at.desc())
            .limit(10)
        ).all()
    
    for event in recent_events:
        timestamp = event.at.strftime("%H:%M")
        st.markdown(f"""
        **{timestamp}** · Παραγγελία #{event.order_id} · {event.provider_name}
        <div style="margin-left: 1rem; color: var(--gray); font-size: 0.9rem;">
        {event.from_state} → {event.to_state}
        </div>
        """, unsafe_allow_html=True)


def render_order_tracking(order: Order, details: Dict[str, Any]):
    """Λεπτομερής παρακολούθηση μιας παραγγελίας"""
    st.markdown(f"### Παραγγελία #{order.id}")
    
    # Πληροφορίες παραγγελίας
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Κατάσταση:** {badge(order.status, 'primary')}")
    with col2:
        st.markdown(f"**Δημιουργήθηκε:** {order.created_at.strftime('%d/%m/%Y %H:%M')}")
    with col3:
        st.markdown(f"**Τελευταία ενημέρωση:** {order.updated_at.strftime('%d/%m/%Y %H:%M') if order.updated_at else '-'}")
    
    st.markdown("---")
    
    # Προμηθευτές και προϊόντα ανά κατηγορία
    tabs = st.tabs(["👥 Προμηθευτές", "📦 Προϊόντα", "📊 Στατιστικά", "🔄 Ιστορικό"])
    
    with tabs[0]:
        render_providers_tab(order, details)
    
    with tabs[1]:
        render_products_tab(order, details)
    
    with tabs[2]:
        render_statistics_tab(order, details)
    
    with tabs[3]:
        render_history_tab(order, details)


def render_providers_tab(order: Order, details: Dict[str, Any]):
    """Καρτέλα προμηθευτών"""
    workflows = details["workflows"]
    receipts = details["receipts"]
    tickets = details["tickets"]
    
    for provider_name, workflow in workflows.items():
        receipt = receipts.get(provider_name)
        provider_tickets = [t for t in tickets if norm_provider(t.provider_name) == norm_provider(provider_name)]
        
        # Υπολογισμός προόδου
        status_map = {
            "ORDER_SENT": 25,
            "SUPPLIER_CONFIRMED_FULL": 50,
            "SUPPLIER_CONFIRMED_PARTIAL": 40,
            "SUPPLIER_CONFIRMED_NONE": 30,
            "RECEIVED": 75,
            "MATCHED_WITH_INVOICE": 90,
            "INVOICE_DISCREPANCY": 60,
            "CLOSED": 100
        }
        progress = status_map.get(workflow.state, 0)
        
        # Χρώμα βάσει κατάστασης
        if "CONFIRMED_FULL" in workflow.state or workflow.state == "MATCHED_WITH_INVOICE":
            color = "success"
        elif "CONFIRMED_PARTIAL" in workflow.state or "DISCREPANCY" in workflow.state:
            color = "warning"
        elif "CONFIRMED_NONE" in workflow.state:
            color = "danger"
        else:
            color = "primary"
        
        # Πληροφορίες παραλαβής
        receipt_info = ""
        if receipt:
            if receipt.received:
                receipt_info = f"{badge('✅ Παραλήφθηκε', 'success')}"
                if receipt.invoice_number:
                    receipt_info += f" · Τιμολόγιο: **{receipt.invoice_number}**"
            else:
                receipt_info = f"{badge('⏳ Αναμένεται', 'warning')}"
        
        # Προβλήματα
        issues_info = ""
        if provider_tickets:
            open_tickets = [t for t in provider_tickets if t.state == "open"]
            if open_tickets:
                issues_info = f"{badge(f'🚨 {len(open_tickets)} προβλήματα', 'danger')}"
        
        # Κάρτα προμηθευτή
        card_content = f"""
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
            <div>
                <h4 style="margin: 0 0 0.5rem 0;">{provider_name}</h4>
                <div style="margin-bottom: 0.5rem;">
                    {badge(workflow.state, color)} {receipt_info} {issues_info}
                </div>
                <div style="font-size: 0.875rem; color: var(--gray);">
                    Τελευταία ενημέρωση: {workflow.updated_at.strftime('%d/%m/%Y %H:%M')}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.25rem; font-weight: 700; color: var(--{color});">{progress}%</div>
                <div style="font-size: 0.75rem; color: var(--gray);">Ολοκλήρωση</div>
            </div>
        </div>
        """
        
        card(card_content, color)
        
        # Μπάρα προόδου
        progress_bar(progress, color, f"Πρόοδος: {workflow.state}")
        
        # Κουμπιά δράσης
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("📧 Επικοινωνία", key=f"contact_{provider_name}_{order.id}", use_container_width=True):
                pass  # Θα υλοποιηθεί με email/WhatsApp
        with col2:
            supplier_link = build_seguimiento_url(
                order_id=order.id,
                provider_name=provider_name,
                role=ROLE_SUPPLIER,
                page_path="seguimiento"
            )
            st.link_button("🔗 Σύνδεσμος", supplier_link, use_container_width=True)
        with col3:
            if st.button("📊 Λεπτομέρειες", key=f"details_{provider_name}_{order.id}", use_container_width=True):
                st.session_state[f"selected_provider_{order.id}"] = provider_name
                st.rerun()
        
        st.markdown("---")


def render_products_tab(order: Order, details: Dict[str, Any]):
    """Καρτέλα προϊόντων"""
    lines = details.get("lines", [])
    products = details["products"]
    
    st.markdown("### 📦 Προϊόντα Παραγγελίας")
    
    for line in lines:
        product = products.get(line.product_id) if line.product_id else None
        product_name = product.name if product else line.spoken_name or "Άγνωστο προϊόν"
        
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            st.markdown(f"**{product_name}**")
            if product and product.description:
                st.caption(product.description[:100] + "..." if len(product.description) > 100 else product.description)
        
        with col2:
            st.markdown(f"**Ποσότητα:** {line.quantity}")
        
        with col3:
            unit = product.unit if product else line.unit or "τεμ."
            st.markdown(f"**Μονάδα:** {unit}")
        
        with col4:
            provider = product.provider_name if product else line.provider
            if provider:
                st.markdown(f"**Προμηθευτής:** {badge(provider, 'info')}")


def render_statistics_tab(order: Order, details: Dict[str, Any]):
    """Καρτέλα στατιστικών"""
    lines = details.get("lines", [])
    products = details["products"]
    
    # Υπολογισμός στατιστικών
    total_products = len(lines)
    total_quantity = sum(line.quantity for line in lines)
    unique_providers = len(set(
        (products.get(line.product_id).provider_name if line.product_id and products.get(line.product_id) else line.provider)
        for line in lines
    ))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        stat_card(total_products, "Συνολικά προϊόντα", "📊", "primary")
    with col2:
        stat_card(total_quantity, "Συνολική ποσότητα", "📦", "success")
    with col3:
        stat_card(unique_providers, "Προμηθευτές", "👥", "info")
    
    # Ανάλυση ανά προμηθευτή
    st.markdown("### Ανάλυση ανά Προμηθευτή")
    provider_stats = {}
    
    for line in lines:
        product = products.get(line.product_id) if line.product_id else None
        provider = (product.provider_name if product else line.provider) or "Άγνωστος"
        provider_norm = norm_provider(provider)
        
        if provider_norm not in provider_stats:
            provider_stats[provider_norm] = {"quantity": 0, "products": 0}
        
        provider_stats[provider_norm]["quantity"] += line.quantity
        provider_stats[provider_norm]["products"] += 1
    
    # Γράφημα
    if provider_stats:
        df = pd.DataFrame([
            {"Προμηθευτής": p, "Ποσότητα": s["quantity"], "Προϊόντα": s["products"]}
            for p, s in provider_stats.items()
        ])
        
        fig = px.pie(df, values='Ποσότητα', names='Προμηθευτής', 
                     title='Κατανομή Ποσότητας ανά Προμηθευτή')
        st.plotly_chart(fig, use_container_width=True)


def render_history_tab(order: Order, details: Dict[str, Any]):
    """Καρτέλα ιστορικού"""
    with get_session() as s:
        events = s.exec(
            select(OrderWorkflowEvent)
            .where(OrderWorkflowEvent.order_id == order.id)
            .order_by(OrderWorkflowEvent.at.desc())
        ).all()
    
    if not events:
        st.info("Δεν υπάρχει ιστορικό για αυτήν την παραγγελία")
        return
    
    for event in events:
        timestamp = event.at.strftime("%d/%m/%Y %H:%M")
        icon = "🔄"
        if "CONFIRMED" in event.to_state:
            icon = "✅"
        elif "DISCREPANCY" in event.to_state:
            icon = "🚨"
        elif "CLOSED" in event.to_state:
            icon = "🏁"
        
        st.markdown(f"""
        <div style="background: white; border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem; border-left: 4px solid var(--primary);">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
                <div style="font-size: 1.25rem;">{icon}</div>
                <div style="font-weight: 600;">{event.provider_name}</div>
                <div style="margin-left: auto; font-size: 0.875rem; color: var(--gray);">{timestamp}</div>
            </div>
            <div style="font-size: 0.875rem; color: var(--dark);">
                <strong>{event.from_state}</strong> → <strong>{event.to_state}</strong>
            </div>
            {f'<div style="font-size: 0.8rem; color: var(--gray); margin-top: 0.25rem;"><em>{event.note}</em></div>' if event.note else ''}
            <div style="font-size: 0.75rem; color: var(--gray); margin-top: 0.25rem;">
                Από: {event.actor or event.actor_role}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_issue_management(venue_id: int):
    """Διαχείριση προβλημάτων"""
    st.markdown("## 🚨 Διαχείριση Προβλημάτων")
    
    # Φίλτρα
    col1, col2, col3 = st.columns(3)
    with col1:
        issue_type = st.selectbox(
            "Τύπος προβλήματος",
            ["Όλα", "Διαφορά τιμολογίου", "Μερική παράδοση", "Ελλείπουσα παραγγελία", 
             "Ποιότητα", "Καθυστέρηση", "Λάθος προϊόν", "Διαφορά τιμής"]
        )
    
    with col2:
        status = st.selectbox(
            "Κατάσταση",
            ["Όλα", "Ανοιχτά", "Σε εξέλιξη", "Επιλυμένα"]
        )
    
    with col3:
        priority = st.selectbox(
            "Προτεραιότητα",
            ["Όλα", "Υψηλή", "Μεσαία", "Χαμηλή"]
        )
    
    # Λήψη δεδομένων
    with get_session() as s:
        query = select(SeguimientoTicket).where(SeguimientoTicket.venue_id == venue_id)
        
        if status != "Όλα":
            if status == "Ανοιχτά":
                query = query.where(SeguimientoTicket.state == "open")
            elif status == "Επιλυμένα":
                query = query.where(SeguimientoTicket.state.like("resolved_%"))
        
        tickets = list(s.exec(query.order_by(SeguimientoTicket.created_at.desc())).all())
    
    if not tickets:
        st.success("🎉 Δεν υπάρχουν προβλήματα για αυτά τα φίλτρα!")
        return
    
    # Εμφάνιση προβλημάτων
    for ticket in tickets:
        # Προσδιορισμός προτεραιότητας
        days_open = (datetime.utcnow() - ticket.created_at).days
        if days_open > 7:
            priority_level = "danger"
            priority_text = "Υψηλή"
        elif days_open > 3:
            priority_level = "warning"
            priority_text = "Μεσαία"
        else:
            priority_level = "info"
            priority_text = "Χαμηλή"
        
        # Χρώμα κατάστασης
        if ticket.state == "open":
            status_color = "danger"
            status_text = "Ανοιχτό"
        else:
            status_color = "success"
            status_text = "Επιλυμένο"
        
        # Κάρτα προβλήματος
        card_content = f"""
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
            <div style="flex: 1;">
                <h4 style="margin: 0 0 0.5rem 0;">{ticket.product_name}</h4>
                <div style="margin-bottom: 0.5rem;">
                    {badge(f'Παραγγελία #{ticket.order_id}', 'info')}
                    {badge(ticket.provider_name, 'primary')}
                    {badge(status_text, status_color)}
                    {badge(priority_text, priority_level)}
                </div>
                <div style="font-size: 0.875rem; color: var(--gray); margin-bottom: 0.5rem;">
                    <strong>Προϊόν:</strong> {ticket.product_name} · {ticket.qty_ordered} {ticket.unit}
                </div>
                {f'<div style="font-size: 0.875rem; color: var(--gray);"><strong>Τιμολόγιο:</strong> {ticket.invoice_number}</div>' if ticket.invoice_number else ''}
                {f'<div style="font-size: 0.875rem; color: var(--dark); margin-top: 0.5rem;"><em>{ticket.note}</em></div>' if ticket.note else ''}
            </div>
            <div style="text-align: right; min-width: 150px;">
                <div style="font-size: 0.875rem; color: var(--gray); margin-bottom: 0.25rem;">
                    Δημιουργήθηκε
                </div>
                <div style="font-size: 1rem; font-weight: 600;">
                    {ticket.created_at.strftime('%d/%m')}
                </div>
                <div style="font-size: 0.75rem; color: var(--gray);">
                    {ticket.created_at.strftime('%H:%M')}
                </div>
                <div style="font-size: 0.75rem; color: var(--danger); margin-top: 0.5rem;">
                    {days_open} ημέρες ανοιχτό
                </div>
            </div>
        </div>
        """
        
        card(card_content, priority_level if ticket.state == "open" else "success")
        
        # Κουμπιά δράσης
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("📞 Καλέστε", key=f"call_{ticket.id}", use_container_width=True):
                pass  # Θα υλοποιηθεί με τηλεφωνική κλήση
        with col2:
            if st.button("📧 Email", key=f"email_{ticket.id}", use_container_width=True):
                pass  # Θα υλοποιηθεί με email
        with col3:
            if st.button("💬 Μήνυμα", key=f"message_{ticket.id}", use_container_width=True):
                pass  # Θα υλοποιηθεί με WhatsApp
        with col4:
            if st.button("📋 Λεπτομέρειες", key=f"details_{ticket.id}", use_container_width=True):
                st.session_state[f"selected_ticket"] = ticket.id
                st.rerun()
        
        st.markdown("---")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def tracking_dashboard(venue_id: int):
    """Κύρια εφαρμογή παρακολούθησης"""
    _inject_modern_css()
    
    # Πλευρικό μενού
    with st.sidebar:
        st.markdown("## 🎯 Παρακολούθηση")
        
        selected_view = st.radio(
            "Επιλογή προβολής",
            ["📊 Dashboard", "📦 Παραγγελίες", "🚨 Προβλήματα", "📈 Αναφορές", "⚙️ Ρυθμίσεις"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Φίλτρα
        st.markdown("### 🔍 Φίλτρα")
        
        col1, col2 = st.columns(2)
        with col1:
            time_filter = st.selectbox(
                "Περίοδος",
                ["Σήμερα", "Τελευταία 7 ημέρες", "Τελευταίο μήνα", "Όλο το ιστορικό"]
            )
        
        with col2:
            status_filter = st.selectbox(
                "Κατάσταση",
                ["Όλες", "Υπό διεκπεραίωση", "Εκκρεμούν", "Επιλυμένες"]
            )
        
        # Γρήγορη στατιστική sidebar
        st.markdown("---")
        st.markdown("### 📈 Γρήγορη Στατιστική")
        
        with get_session() as s:
            active_orders = s.exec(
                select(func.count(Order.id))
                .where(Order.venue_id == venue_id, Order.status == "pending_receive")
            ).first() or 0
            
            urgent_issues = s.exec(
                select(func.count(SeguimientoTicket.id))
                .where(
                    SeguimientoTicket.venue_id == venue_id,
                    SeguimientoTicket.state == "open",
                    SeguimientoTicket.created_at >= datetime.utcnow() - timedelta(days=3)
                )
            ).first() or 0
        
        st.metric("Ενεργές Παραγγελίες", active_orders)
        st.metric("Επείγοντα Προβλήματα", urgent_issues, delta_color="inverse")
    
    # Κύρια περιοχή περιεχομένου
    if selected_view == "📊 Dashboard":
        render_dashboard_overview(venue_id)
    
    elif selected_view == "📦 Παραγγελίες":
        st.markdown("## 📦 Παραγγελίες Υπό Διεκπεραίωση")
        
        # Επιλογή παραγγελίας
        active_orders = get_active_orders(venue_id)
        
        if not active_orders:
            st.success("🎉 Δεν υπάρχουν παραγγελίες υπό διεκπεραίωση!")
            return
        
        order_options = {f"#{o.id} - {o.created_at.strftime('%d/%m/%Y')}": o.id for o in active_orders}
        selected_order_label = st.selectbox("Επιλέξτε παραγγελία", list(order_options.keys()))
        selected_order_id = order_options[selected_order_label]
        
        # Λεπτομέρειες παραγγελίας
        try:
            order, lines, details = get_order_details(selected_order_id)
            details["lines"] = lines
            render_order_tracking(order, details)
        except Exception as e:
            st.error(f"Σφάλμα φόρτωσης παραγγελίας: {e}")
    
    elif selected_view == "🚨 Προβλήματα":
        render_issue_management(venue_id)
    
    elif selected_view == "📈 Αναφορές":
        st.markdown("## 📈 Αναλυτικές Αναφορές")
        
        # Χρονολογικό φίλτρο
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Από ημερομηνία", value=datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("Έως ημερομηνία", value=datetime.now())
        
        # Απόδοση προμηθευτών
        st.markdown("### 📊 Απόδοση Προμηθευτών")
        performance_data = get_provider_performance(venue_id, days=(datetime.now().date() - start_date).days)
        
        if performance_data:
            df = pd.DataFrame(performance_data)
            st.dataframe(
                df.style
                .background_gradient(subset=['score'], cmap='RdYlGn')
                .format({'score': '{:.0f}'}),
                use_container_width=True
            )
            
            # Εξαγωγή δεδομένων
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Εξαγωγή CSV",
                data=csv,
                file_name=f"provider_performance_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    else:  # Ρυθμίσεις
        st.markdown("## ⚙️ Ρυθμίσεις")
        
        with st.expander("🔔 Ειδοποιήσεις", expanded=True):
            email_alerts = st.checkbox("Email ειδοποιήσεις", value=True)
            whatsapp_alerts = st.checkbox("WhatsApp ειδοποιήσεις", value=False)
            sms_alerts = st.checkbox("SMS ειδοποιήσεις", value=False)
            
            alert_types = st.multiselect(
                "Τύποι ειδοποιήσεων",
                ["Νέες παραγγελίες", "Εκκρεμείς παραγγελίες", "Νέα προβλήματα", "Επιλυμένα προβλήματα", "Αλλαγές κατάστασης"],
                default=["Νέα προβλήματα", "Εκκρεμείς παραγγελίες"]
            )
        
        with st.expander("🎨 Προβολή", expanded=False):
            theme = st.selectbox("Θέμα εμφάνισης", ["Φωτεινό", "Σκοτεινό", "Αυτόματο"])
            density = st.select_slider("Πυκνότητα πληροφοριών", ["Ελάχιστη", "Χαμηλή", "Μεσαία", "Υψηλή", "Μέγιστη"], value="Μεσαία")
        
        if st.button("💾 Αποθήκευση Ρυθμίσεων", type="primary"):
            st.success("Οι ρυθμίσεις αποθηκεύτηκαν επιτυχώς!")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Κύρια συνάρτηση εκκίνησης"""
    st.set_page_config(
        page_title="Tracking System | Modern Order Monitoring",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Προσομοίωση venue_id (θα αντικατασταθεί με το πραγματικό σας authentication)
    venue_id = st.session_state.get("venue_id", 1)
    
    # Επικεφαλίδα
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="margin: 0;">📦 Modern Tracking System</h1>
        <p style="color: var(--gray); margin: 0.5rem 0 0 0;">Παρακολούθηση & Διαχείριση Παραγγελιών</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Κύρια εφαρμογή
    tracking_dashboard(venue_id)


if __name__ == "__main__":
    main()