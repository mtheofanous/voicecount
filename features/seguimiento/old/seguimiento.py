"""
modern_seguimiento.py - Μοντέρνο interface αλληλεπίδρασης προμηθευτή-μαγαζιού

Φιλοσοφία:
1. Υπερ-απλό UI για και τους δύο ρόλους
2. Real-time ενημέρωση χωρίς reload
3. Εύκολη λήψη αποφάσεων
4. Αυτόματη καταγραφή όλων των ενεργειών
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Dict, List, Tuple
import streamlit as st
from sqlmodel import select, func
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from core.db import get_session
from core.public_links import ROLE_SUPPLIER, ROLE_VENUE, norm_provider, norm_role, verify_link
from domain.models import (
    Order, OrderLine, Product, OrderWorkflow, OrderWorkflowEvent,
    SeguimientoTicket, ProviderLineFollowUp, ProviderReceipt
)


# =============================================================================
# UI COMPONENTS
# =============================================================================

def _inject_seguimiento_css() -> None:
    """CSS για το seguimiento interface"""
    st.markdown("""
    <style>
    /* FOLLOW-UP SPECIFIC STYLES */
    .followup-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }
    
    .role-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.875rem;
        margin-bottom: 1.5rem;
    }
    
    .role-badge--supplier {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .role-badge--venue {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .action-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    
    .action-card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
        transform: translateY(-1px);
    }
    
    .action-card--pending {
        border-left: 4px solid #F59E0B;
        background: linear-gradient(90deg, #FFFBEB 0%, white 20%);
    }
    
    .action-card--urgent {
        border-left: 4px solid #EF4444;
        background: linear-gradient(90deg, #FEF2F2 0%, white 20%);
        animation: pulse 2s infinite;
    }
    
    .action-card--completed {
        border-left: 4px solid #10B981;
        background: linear-gradient(90deg, #ECFDF5 0%, white 20%);
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.9; }
        100% { opacity: 1; }
    }
    
    .decision-button {
        width: 100%;
        padding: 1rem;
        border-radius: 12px;
        border: 2px solid transparent;
        background: white;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
        margin: 0.5rem 0;
    }
    
    .decision-button:hover {
        transform: translateY(-2px);
        border-color: #3B82F6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }
    
    .decision-button--selected {
        border-color: #3B82F6 !important;
        background: #EFF6FF;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }
    
    .quantity-input {
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        padding: 0.5rem;
        border: 2px solid #E5E7EB;
        border-radius: 8px;
        width: 100px;
        margin: 0 auto;
    }
    
    .quantity-input:focus {
        outline: none;
        border-color: #3B82F6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
    }
    
    .timeline-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 0.5rem;
    }
    
    .timeline-dot--success { background: #10B981; }
    .timeline-dot--warning { background: #F59E0B; }
    .timeline-dot--danger { background: #EF4444; }
    .timeline-dot--info { background: #8B5CF6; }
    
    /* RESPONSIVE */
    @media (max-width: 768px) {
        .followup-container {
            padding: 1rem 0.5rem;
        }
        
        .action-card {
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def role_badge(role: str) -> None:
    """Εμφανίζει το badge του ρόλου"""
    role_class = "role-badge--supplier" if role == ROLE_SUPPLIER else "role-badge--venue"
    role_text = "Προμηθευτής" if role == ROLE_SUPPLIER else "Μαγαζί"
    
    st.markdown(f"""
    <div class="followup-container">
        <div class="role-badge {role_class}">
            👤 {role_text}
        </div>
    """, unsafe_allow_html=True)


def action_card(title: str, content: str, status: str = "pending", icon: str = "📋") -> None:
    """Εμφανίζει μια κάρτα δράσης"""
    status_class = {
        "pending": "action-card--pending",
        "urgent": "action-card--urgent",
        "completed": "action-card--completed"
    }.get(status, "")
    
    st.markdown(f"""
    <div class="action-card {status_class}">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
            <div style="font-size: 1.5rem;">{icon}</div>
            <h3 style="margin: 0;">{title}</h3>
        </div>
        {content}
    </div>
    """, unsafe_allow_html=True)


def decision_option(title: str, description: str, value: str, selected: bool = False) -> bool:
    """Εμφανίζει μια επιλογή απόφασης"""
    selection_class = " decision-button--selected" if selected else ""
    
    st.markdown(f"""
    <div class="decision-button{selection_class}" onclick="this.nextElementSibling.click()">
        <div style="font-weight: 600; margin-bottom: 0.25rem;">{title}</div>
        <div style="font-size: 0.875rem; color: #6B7280;">{description}</div>
    </div>
    """, unsafe_allow_html=True)
    
    return st.button("Επιλογή", key=f"opt_{value}", use_container_width=True)


# =============================================================================
# DATA HELPERS
# =============================================================================

def _s(x: Any) -> str:
    return ("" if x is None else str(x)).strip()


def _f(x: Any) -> Optional[float]:
    try:
        if isinstance(x, list) and x:
            x = x[0]
        s = _s(x)
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _now() -> datetime:
    return datetime.utcnow()


def load_order_context(order_id: int, provider_name: str, role: str) -> Tuple[Dict[str, Any], str]:
    """Φορτώνει όλο το context της παραγγελίας"""
    with get_session() as s:
        # Βασική παραγγελία
        order = s.exec(select(Order).where(Order.id == order_id)).first()
        if not order:
            raise ValueError("Η παραγγελία δεν βρέθηκε")
        
        # Γραμμές παραγγελίας για αυτόν τον προμηθευτή
        lines = list(s.exec(
            select(OrderLine)
            .where(OrderLine.order_id == order_id)
            .join(Product, Product.id == OrderLine.product_id, isouter=True)
            .where(or_(
                Product.provider_name == provider_name,
                OrderLine.provider == provider_name
            ))
        ).all())
        
        # Workflow
        workflow = s.exec(
            select(OrderWorkflow)
            .where(
                OrderWorkflow.order_id == order_id,
                OrderWorkflow.provider_name == provider_name
            )
        ).first()
        
        if not workflow:
            raise ValueError("Το workflow δεν βρέθηκε")
        
        # Προϊόντα
        product_ids = [l.product_id for l in lines if l.product_id]
        products = {}
        if product_ids:
            products_list = list(s.exec(select(Product).where(Product.id.in_(product_ids))).all())
            products = {p.id: p for p in products_list}
        
        # Ιστορικό
        history = list(s.exec(
            select(OrderWorkflowEvent)
            .where(
                OrderWorkflowEvent.order_id == order_id,
                OrderWorkflowEvent.provider_name == provider_name
            )
            .order_by(OrderWorkflowEvent.at.asc())
        ).all())
        
        # Tickets
        tickets = list(s.exec(
            select(SeguimientoTicket)
            .where(
                SeguimientoTicket.order_id == order_id,
                SeguimientoTicket.provider_name == provider_name
            )
        ).all())
        
        # Receipt
        receipt = s.exec(
            select(ProviderReceipt)
            .where(
                ProviderReceipt.order_id == order_id,
                ProviderReceipt.provider_name == provider_name
            )
        ).first()
        
        # Follow-ups
        follow_ups = list(s.exec(
            select(ProviderLineFollowUp)
            .where(
                ProviderLineFollowUp.order_id == order_id,
                ProviderLineFollowUp.provider_name == provider_name
            )
        ).all())
    
    return {
        "order": order,
        "lines": lines,
        "workflow": workflow,
        "products": products,
        "history": history,
        "tickets": tickets,
        "receipt": receipt,
        "follow_ups": {fu.order_line_id: fu for fu in follow_ups}
    }, role


def save_supplier_response(context: Dict[str, Any], responses: Dict[int, Dict[str, Any]], note: str) -> None:
    """Αποθηκεύει την απάντηση του προμηθευτή"""
    with get_session() as s:
        # Αποθήκευση ανά γραμμή
        for line_id, response in responses.items():
            follow_up = s.exec(
                select(ProviderLineFollowUp)
                .where(
                    ProviderLineFollowUp.order_id == context["order"].id,
                    ProviderLineFollowUp.provider_name == context["workflow"].provider_name,
                    ProviderLineFollowUp.order_line_id == line_id
                )
            ).first()
            
            if not follow_up:
                follow_up = ProviderLineFollowUp(
                    venue_id=context["order"].venue_id,
                    order_id=context["order"].id,
                    provider_name=context["workflow"].provider_name,
                    order_line_id=line_id,
                    qty_ordered=response.get("ordered", 0)
                )
            
            follow_up.supplier_status = response["status"]
            follow_up.supplier_qty = response.get("quantity", 0)
            follow_up.supplier_reason = response.get("reason", "")
            follow_up.updated_at = _now()
            follow_up.updated_by = "supplier"
            
            s.add(follow_up)
        
        # Ενημέρωση workflow
        workflow = context["workflow"]
        workflow.state = determine_next_state(responses)
        workflow.updated_at = _now()
        workflow.updated_by_role = "supplier"
        workflow.updated_by = "supplier"
        workflow.note = note
        
        s.add(workflow)
        
        # Προσθήκη ιστορικού
        event = OrderWorkflowEvent(
            venue_id=context["order"].venue_id,
            order_id=context["order"].id,
            provider_name=context["workflow"].provider_name,
            from_state=context["workflow"].state,
            to_state=workflow.state,
            actor_role="supplier",
            actor="supplier",
            at=_now(),
            note=note
        )
        
        s.add(event)
        s.commit()


def determine_next_state(responses: Dict[int, Dict[str, Any]]) -> str:
    """Προσδιορίζει την επόμενη κατάσταση βάσει των απαντήσεων"""
    statuses = [r["status"] for r in responses.values()]
    
    if all(s == "ok" for s in statuses):
        return "SUPPLIER_CONFIRMED_FULL"
    elif all(s == "missing" for s in statuses):
        return "SUPPLIER_CONFIRMED_NONE"
    else:
        return "SUPPLIER_CONFIRMED_PARTIAL"


# =============================================================================
# SUPPLIER INTERFACE
# =============================================================================

def render_supplier_interface(context: Dict[str, Any]) -> None:
    """Interface για προμηθευτές"""
    order = context["order"]
    workflow = context["workflow"]
    lines = context["lines"]
    products = context["products"]
    
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="margin: 0 0 0.5rem 0;">📦 Παραγγελία #{order.id}</h1>
        <p style="color: #6B7280; margin: 0;">Καταστάσεις παράδοσης</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Τρέχουσα κατάσταση
    status_info = {
        "ORDER_SENT": ("⏳ Αναμένεται Απάντηση", "Αναμένεται η απάντησή σας για την παράδοση"),
        "SUPPLIER_CONFIRMED_FULL": ("✅ Πλήρης Παράδοση", "Έχετε επιβεβαιώσει πλήρη παράδοση"),
        "SUPPLIER_CONFIRMED_PARTIAL": ("🟡 Μερική Παράδοση", "Έχετε επιβεβαιώσει μερική παράδοση"),
        "SUPPLIER_CONFIRMED_NONE": ("🔴 Χωρίς Παράδοση", "Έχετε επιβεβαιώσει ότι δεν θα παραδοθεί"),
        "RECEIVED": ("📦 Παραλήφθηκε", "Η παραγγελία παραλήφθηκε από το μαγαζί"),
        "MATCHED_WITH_INVOICE": ("💰 Ταιριάζει με Τιμολόγιο", "Όλα τα στοιχεία ταιριάζουν"),
        "CLOSED": ("🏁 Ολοκληρώθηκε", "Η παραγγελία ολοκληρώθηκε επιτυχώς")
    }
    
    current_status, status_description = status_info.get(
        workflow.state, 
        (workflow.state, "Άγνωστη κατάσταση")
    )
    
    action_card(
        title="Τρέχουσα Κατάσταση",
        content=f"""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{current_status.split()[0]}</div>
            <div style="font-size: 1.25rem; font-weight: 600; margin-bottom: 0.25rem;">
                {current_status}
            </div>
            <div style="color: #6B7280;">
                {status_description}
            </div>
        </div>
        """,
        status="completed" if workflow.state in ["CLOSED", "MATCHED_WITH_INVOICE"] else "pending",
        icon="📊"
    )
    
    if workflow.state == "ORDER_SENT":
        render_supplier_confirmation(context)
    elif workflow.state == "WAITING_SUPPLIER_ACTION":
        render_supplier_issue_resolution(context)
    else:
        render_supplier_readonly(context)


def render_supplier_confirmation(context: Dict[str, Any]) -> None:
    """Φόρμα επιβεβαίωσης προμηθευτή"""
    lines = context["lines"]
    products = context["products"]
    
    action_card(
        title="Επιβεβαίωση Παράδοσης",
        content="""Παρακαλώ επιβεβαιώστε την κατάσταση κάθε προϊόντος.
        Η επιλογή σας θα ενημερώσει αυτόματα το μαγαζί.""",
        status="pending",
        icon="✅"
    )
    
    responses = {}
    
    for line in lines:
        product = products.get(line.product_id) if line.product_id else None
        product_name = product.name if product else line.spoken_name or "Άγνωστο προϊόν"
        
        with st.container():
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown(f"**{product_name}**")
                st.caption(f"Παραγγελία: {line.quantity} {product.unit if product else 'τεμ.'}")
            
            with col2:
                # Κατάσταση
                status = st.radio(
                    "Κατάσταση",
                    ["ok", "partial", "missing"],
                    format_func=lambda x: {
                        "ok": "✅ Πλήρες",
                        "partial": "🟡 Μερικό",
                        "missing": "🔴 Ελλείπει"
                    }[x],
                    horizontal=True,
                    key=f"status_{line.id}"
                )
                
                # Ποσότητα (μόνο για μερική παράδοση)
                quantity = line.quantity
                if status == "partial":
                    quantity = st.number_input(
                        "Ποσότητα",
                        min_value=0.0,
                        max_value=float(line.quantity),
                        value=float(line.quantity),
                        key=f"qty_{line.id}"
                    )
                
                # Αιτία (για μερική/ελλείπουσα)
                reason = ""
                if status in ["partial", "missing"]:
                    reason = st.text_input(
                        "Αιτία (προαιρετικά)",
                        placeholder="Π.χ. Εκτός αποθέματος, Καθυστέρηση, κλπ.",
                        key=f"reason_{line.id}"
                    )
                
                responses[line.id] = {
                    "status": status,
                    "quantity": quantity if status == "partial" else (0 if status == "missing" else line.quantity),
                    "reason": reason,
                    "ordered": line.quantity
                }
        
        st.markdown("---")
    
    # Γενική σημείωση
    note = st.text_area(
        "Γενική Σημείωση (προαιρετικά)",
        placeholder="Προσθέστε οποιαδήποτε επιπλέον πληροφορία...",
        height=100
    )
    
    # Κουμπί υποβολής
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Υποβολή Επιβεβαίωσης", type="primary", use_container_width=True):
            if responses:
                save_supplier_response(context, responses, note)
                st.success("Η επιβεβαίωση σας υποβλήθηκε επιτυχώς!")
                st.rerun()
            else:
                st.error("Παρακαλώ επιβεβαιώστε τουλάχιστον ένα προϊόν")


def render_supplier_issue_resolution(context: Dict[str, Any]) -> None:
    """Επίλυση προβλημάτων από προμηθευτή"""
    tickets = context["tickets"]
    
    action_card(
        title="🚨 Επίλυση Προβλήματος",
        content="""Υπάρχει απόκλιση μεταξύ τιμολογίου και παραλαβής.
        Παρακαλώ επιλέξτε πώς θέλετε να το επιλύσετε.""",
        status="urgent",
        icon="🚨"
    )
    
    # Εμφάνιση προβλημάτων
    st.markdown("### Προβλήματα που απαιτούν επίλυση")
    
    for ticket in tickets:
        if ticket.state == "open":
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{ticket.product_name}**")
                st.caption(f"""
                Τιμολογήθηκε: {ticket.qty_invoiced or ticket.qty_expected} {ticket.unit}
                Παραλήφθηκε: {ticket.qty_received} {ticket.unit}
                Διαφορά: {(ticket.qty_invoiced or ticket.qty_expected) - ticket.qty_received} {ticket.unit}
                """)
            with col2:
                if ticket.invoice_number:
                    st.markdown(f"**Τιμολόγιο:** {ticket.invoice_number}")
    
    # Επιλογή λύσης
    st.markdown("### Επιλογή Λύσης")
    
    solution = st.radio(
        "Πώς θέλετε να επιλυθεί;",
        [
            "credit_note",
            "replacement",
            "partial_refund",
            "future_discount",
            "no_action"
        ],
        format_func=lambda x: {
            "credit_note": "📝 Έκδοση Πίστωσης",
            "replacement": "🔄 Αντικατάσταση Προϊόντος",
            "partial_refund": "💰 Μερική Επιστροφή Χρημάτων",
            "future_discount": "🎯 Έκπτωση σε Μελλοντική Παραγγελία",
            "no_action": "⚪ Δεν Απαιτείται Ενέργεια"
        }[x],
        horizontal=False
    )
    
    # Λεπτομέρειες
    details = st.text_area(
        "Λεπτομέρειες Επίλυσης",
        placeholder="Π.χ. Αριθμός πίστωσης, ημερομηνία αντικατάστασης, κλπ...",
        height=100
    )
    
    if st.button("✅ Οριστικοποίηση Επίλυσης", type="primary"):
        # Ενημέρωση tickets
        with get_session() as s:
            for ticket in tickets:
                if ticket.state == "open":
                    ticket.state = f"resolved_{solution}"
                    ticket.resolution_note = details
                    ticket.resolved_at = _now()
                    ticket.updated_at = _now()
                    s.add(ticket)
            
            # Ενημέρωση workflow
            workflow = context["workflow"]
            workflow.state = "SUPPLIER_CREDIT_NOTE_ISSUED" if solution == "credit_note" else "SUPPLEMENTARY_DELIVERY_SENT"
            workflow.updated_at = _now()
            workflow.updated_by_role = "supplier"
            workflow.note = details
            
            s.add(workflow)
            
            # Προσθήκη ιστορικού
            event = OrderWorkflowEvent(
                venue_id=context["order"].venue_id,
                order_id=context["order"].id,
                provider_name=context["workflow"].provider_name,
                from_state=context["workflow"].state,
                to_state=workflow.state,
                actor_role="supplier",
                actor="supplier",
                at=_now(),
                note=details
            )
            
            s.add(event)
            s.commit()
        
        st.success("Η επίλυση υποβλήθηκε επιτυχώς!")
        st.rerun()


def render_supplier_readonly(context: Dict[str, Any]) -> None:
    """Μόνο για ανάγνωση (όταν δεν υπάρχουν ενέργειες)"""
    workflow = context["workflow"]
    history = context["history"]
    
    st.info("Δεν υπάρχουν ενέργειες που να απαιτούν την προσοχή σας αυτήν τη στιγμή.")
    
    # Ιστορικό
    if history:
        st.markdown("### Ιστορικό Δραστηριοτήτων")
        
        for event in history[-5:]:  # Τελευταίες 5 ενέργειες
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.markdown(event.at.strftime("%d/%m\n%H:%M"))
            with col2:
                st.markdown(f"**{event.from_state} → {event.to_state}**")
                if event.note:
                    st.caption(event.note)
            with col3:
                st.caption(event.actor_role)


# =============================================================================
# VENUE INTERFACE
# =============================================================================

def render_venue_interface(context: Dict[str, Any]) -> None:
    """Interface για μαγαζιά"""
    order = context["order"]
    workflow = context["workflow"]
    
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="margin: 0 0 0.5rem 0;">🏪 Παραγγελία #{order.id}</h1>
        <p style="color: #6B7280; margin: 0;">Παρακολούθηση & Διαχείριση</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Προμηθευτής
    st.markdown(f"""
    <div style="text-align: center; background: #F3F4F6; padding: 1rem; border-radius: 12px; margin-bottom: 1.5rem;">
        <div style="font-size: 1.25rem; font-weight: 600;">{workflow.provider_name}</div>
        <div style="color: #6B7280;">Προμηθευτής</div>
    </div>
    """, unsafe_allow_html=True)
    
    if workflow.state in ["SUPPLIER_CONFIRMED_FULL", "SUPPLIER_CONFIRMED_PARTIAL", "SUPPLIER_CONFIRMED_NONE"]:
        render_venue_receiving(context)
    elif workflow.state == "RECEIVED":
        render_venue_invoice_matching(context)
    elif workflow.state == "WAITING_MANAGER_DECISION":
        render_venue_decision_making(context)
    else:
        render_venue_readonly(context)


def render_venue_receiving(context: Dict[str, Any]) -> None:
    """Λήψη παραγγελίας από μαγαζί"""
    lines = context["lines"]
    products = context["products"]
    follow_ups = context["follow_ups"]
    
    action_card(
        title="📦 Λήψη Παραγγελίας",
        content="""Παρακαλώ επιβεβαιώστε τα προϊόντα που παραλήφθηκαν.
        Συγκρίνετε με την απάντηση του προμηθευτή.""",
        status="pending",
        icon="📦"
    )
    
    # Σύνοψη απάντησης προμηθευτή
    if follow_ups:
        st.markdown("### Απάντηση Προμηθευτή")
        
        for line in lines:
            follow_up = follow_ups.get(line.id)
            if follow_up:
                product = products.get(line.product_id) if line.product_id else None
                product_name = product.name if product else line.spoken_name or "Άγνωστο προϊόν"
                
                status_text = {
                    "ok": "✅ Πλήρες",
                    "partial": "🟡 Μερικό",
                    "missing": "🔴 Ελλείπει"
                }.get(follow_up.supplier_status, follow_up.supplier_status)
                
                col1, col2, col3 = st.columns([3, 1, 2])
                with col1:
                    st.markdown(f"**{product_name}**")
                with col2:
                    st.markdown(status_text)
                with col3:
                    if follow_up.supplier_status == "partial":
                        st.caption(f"{follow_up.supplier_qty} από {line.quantity}")
                    if follow_up.supplier_reason:
                        st.caption(f"*{follow_up.supplier_reason}*")
        
        st.markdown("---")
    
    # Φόρμα λήψης
    st.markdown("### Επιβεβαίωση Παραλαβής")
    
    received_data = {}
    
    for line in lines:
        product = products.get(line.product_id) if line.product_id else None
        product_name = product.name if product else line.spoken_name or "Άγνωστο προϊόν"
        
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.markdown(f"**{product_name}**")
            st.caption(f"Παραγγελία: {line.quantity} {product.unit if product else 'τεμ.'}")
        
        with col2:
            received_qty = st.number_input(
                "Παραλήφθηκε",
                min_value=0.0,
                max_value=float(line.quantity) * 2,  # Επιτρέπουμε παραπάνω για λάθη
                value=float(line.quantity),
                key=f"recv_{line.id}"
            )
        
        with col3:
            condition = st.selectbox(
                "Κατάσταση",
                ["ok", "damaged", "wrong", "expired"],
                format_func=lambda x: {
                    "ok": "✅ Καλό",
                    "damaged": "🚨 Κατεστραμμένο",
                    "wrong": "⚠️ Λάθος",
                    "expired": "📅 Εκλιπόν"
                }[x],
                key=f"cond_{line.id}"
            )
        
        received_data[line.id] = {
            "received": received_qty,
            "condition": condition,
            "ordered": line.quantity
        }
        
        st.markdown("---")
    
    # Γενικές πληροφορίες
    col1, col2 = st.columns(2)
    with col1:
        delivery_person = st.text_input("Όνομα παραλήπτη", placeholder="Π.χ. Γιώργος Παπαδόπουλος")
    with col2:
        delivery_time = st.time_input("Ώρα παραλαβής", value=datetime.now().time())
    
    notes = st.text_area("Σημειώσεις παραλαβής", placeholder="Προσθέστε σημειώσεις...")
    
    # Κουμπί υποβολής
    if st.button("✅ Επιβεβαίωση Παραλαβής", type="primary", use_container_width=True):
        # Αποθήκευση δεδομένων
        with get_session() as s:
            # Ενημέρωση workflow
            workflow = context["workflow"]
            workflow.state = "RECEIVED"
            workflow.updated_at = _now()
            workflow.updated_by_role = "receiving_employee"
            workflow.note = notes
            
            s.add(workflow)
            
            # Προσθήκη ιστορικού
            event = OrderWorkflowEvent(
                venue_id=context["order"].venue_id,
                order_id=context["order"].id,
                provider_name=workflow.provider_name,
                from_state=context["workflow"].state,
                to_state="RECEIVED",
                actor_role="receiving_employee",
                actor=delivery_person or "venue",
                at=_now(),
                note=notes
            )
            
            s.add(event)
            
            # Αποθήκευση follow-ups
            for line_id, data in received_data.items():
                follow_up = s.exec(
                    select(ProviderLineFollowUp)
                    .where(
                        ProviderLineFollowUp.order_id == context["order"].id,
                        ProviderLineFollowUp.provider_name == workflow.provider_name,
                        ProviderLineFollowUp.order_line_id == line_id
                    )
                ).first()
                
                if not follow_up:
                    follow_up = ProviderLineFollowUp(
                        venue_id=context["order"].venue_id,
                        order_id=context["order"].id,
                        provider_name=workflow.provider_name,
                        order_line_id=line_id,
                        qty_ordered=data["ordered"]
                    )
                
                follow_up.venue_qty = data["received"]
                follow_up.venue_comment = notes if data["condition"] != "ok" else None
                follow_up.updated_at = _now()
                follow_up.updated_by = "venue"
                
                s.add(follow_up)
            
            s.commit()
        
        st.success("Η παραλαβή επιβεβαιώθηκε επιτυχώς!")
        st.rerun()


def render_venue_invoice_matching(context: Dict[str, Any]) -> None:
    """Σύγκριση τιμολογίου με παραλαβή"""
    lines = context["lines"]
    products = context["products"]
    
    action_card(
        title="💰 Σύγκριση με Τιμολόγιο",
        content="""Συγκρίνετε τα προϊόντα που παραλάβατε με το τιμολόγιο.
        Αναφέρετε τυχόν αποκλίσεις.""",
        status="pending",
        icon="💰"
    )
    
    invoice_number = st.text_input("Αριθμός Τιμολογίου", placeholder="Π.χ. INV-2024-00123")
    
    st.markdown("### Σύγκριση Ποσοτήτων")
    
    discrepancies = []
    
    for line in lines:
        product = products.get(line.product_id) if line.product_id else None
        product_name = product.name if product else line.spoken_name or "Άγνωστο προϊόν"
        
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            st.markdown(f"**{product_name}**")
            st.caption(f"Παραγγελία: {line.quantity}")
        
        with col2:
            invoiced = st.number_input(
                "Τιμολογήθηκε",
                min_value=0.0,
                value=float(line.quantity),
                key=f"inv_{line.id}"
            )
        
        with col3:
            received = st.number_input(
                "Παραλήφθηκε",
                min_value=0.0,
                value=float(line.quantity),
                key=f"rec_{line.id}"
            )
        
        with col4:
            difference = received - invoiced
            if difference < 0:
                st.error(f"{-difference}")
                discrepancies.append({
                    "product": product_name,
                    "invoiced": invoiced,
                    "received": received,
                    "difference": -difference
                })
            elif difference > 0:
                st.warning(f"+{difference}")
            else:
                st.success("✓")
    
    notes = st.text_area("Σημειώσεις σύγκρισης", placeholder="Προσθέστε σημειώσεις...")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Όλα Ταιριάζουν", type="primary", use_container_width=True):
            if not discrepancies:
                # Αποθήκευση ως ολοκληρωμένο
                with get_session() as s:
                    workflow = context["workflow"]
                    workflow.state = "MATCHED_WITH_INVOICE"
                    workflow.updated_at = _now()
                    workflow.note = notes
                    
                    s.add(workflow)
                    
                    # Προσθήκη ιστορικού
                    event = OrderWorkflowEvent(
                        venue_id=context["order"].venue_id,
                        order_id=context["order"].id,
                        provider_name=workflow.provider_name,
                        from_state="RECEIVED",
                        to_state="MATCHED_WITH_INVOICE",
                        actor_role="receiving_employee",
                        actor="venue",
                        at=_now(),
                        note=notes
                    )
                    
                    s.add(event)
                    s.commit()
                
                st.success("Η σύγκριση ολοκληρώθηκε επιτυχώς!")
                st.rerun()
            else:
                st.error("Υπάρχουν αποκλίσεις. Δεν μπορείτε να ολοκληρώσετε ως 'Όλα Ταιριάζουν'.")
    
    with col2:
        if st.button("🚨 Αναφορά Προβλήματος", type="secondary", use_container_width=True):
            if discrepancies:
                # Δημιουργία tickets
                with get_session() as s:
                    for disc in discrepancies:
                        ticket = SeguimientoTicket(
                            venue_id=context["order"].venue_id,
                            order_id=context["order"].id,
                            provider_name=context["workflow"].provider_name,
                            order_line_id=0,  # Θα χρειαστεί να το βρείτε από την γραμμή
                            kind="invoice_discrepancy",
                            state="open",
                            product_name=disc["product"],
                            unit="τεμ.",  # Θα χρειαστεί να το βρείτε
                            qty_ordered=disc["invoiced"],
                            qty_expected=disc["invoiced"],
                            qty_received=disc["received"],
                            invoice_number=invoice_number,
                            qty_invoiced=disc["invoiced"],
                            note=f"Απόκλιση: {disc['difference']} τεμ.",
                            created_at=_now(),
                            updated_at=_now()
                        )
                        
                        s.add(ticket)
                    
                    # Ενημέρωση workflow
                    workflow = context["workflow"]
                    workflow.state = "INVOICE_DISCREPANCY"
                    workflow.updated_at = _now()
                    workflow.note = notes
                    
                    s.add(workflow)
                    
                    # Προσθήκη ιστορικού
                    event = OrderWorkflowEvent(
                        venue_id=context["order"].venue_id,
                        order_id=context["order"].id,
                        provider_name=workflow.provider_name,
                        from_state="RECEIVED",
                        to_state="INVOICE_DISCREPANCY",
                        actor_role="receiving_employee",
                        actor="venue",
                        at=_now(),
                        note=notes
                    )
                    
                    s.add(event)
                    s.commit()
                
                st.success(f"Δημιουργήθηκαν {len(discrepancies)} tickets για τα προβλήματα!")
                st.rerun()


def render_venue_decision_making(context: Dict[str, Any]) -> None:
    """Λήψη αποφάσεων από διαχειριστή"""
    action_card(
        title="🧠 Απόφαση Διαχειριστή",
        content="""Υπάρχουν προϊόντα που δεν περιλαμβάνονται στο τιμολόγιο.
        Παρακαλώ λάβετε απόφαση για το πώς θα προχωρήσετε.""",
        status="urgent",
        icon="🧠"
    )
    
    st.markdown("### Επιλογές")
    
    # Επιλογές απόφασης
    options = [
        {
            "title": "📦 Επαναπαραγγελία",
            "description": "Παραγγείλετε ξανά από τον ίδιο προμηθευτή",
            "value": "reorder"
        },
        {
            "title": "🔄 Αλλαγή Προμηθευτή",
            "description": "Βρείτε νέο προμηθευτή για αυτό το προϊόν",
            "value": "switch"
        },
        {
            "title": "❌ Δεν Χρειάζεται",
            "description": "Το προϊόν δεν χρειάζεται πλέον",
            "value": "not_needed"
        },
        {
            "title": "💰 Απαίτηση Επιστροφής",
            "description": "Απαιτήστε επιστροφή χρημάτων",
            "value": "refund"
        }
    ]
    
    selected_option = None
    
    for option in options:
        if decision_option(option["title"], option["description"], option["value"]):
            selected_option = option["value"]
    
    if selected_option:
        details = st.text_area(
            "Λεπτομέρειες Απόφασης",
            placeholder="Εξηγήστε την απόφασή σας...",
            height=100
        )
        
        if st.button("✅ Υποβολή Απόφασης", type="primary"):
            # Αποθήκευση απόφασης
            with get_session() as s:
                workflow = context["workflow"]
                workflow.state = {
                    "reorder": "DECISION_REORDER_SAME",
                    "switch": "DECISION_SWITCH_SUPPLIER",
                    "not_needed": "DECISION_NOT_NEEDED",
                    "refund": "WAITING_SUPPLIER_ACTION"
                }[selected_option]
                
                workflow.updated_at = _now()
                workflow.updated_by_role = "manager"
                workflow.note = details
                
                s.add(workflow)
                
                # Προσθήκη ιστορικού
                event = OrderWorkflowEvent(
                    venue_id=context["order"].venue_id,
                    order_id=context["order"].id,
                    provider_name=workflow.provider_name,
                    from_state=context["workflow"].state,
                    to_state=workflow.state,
                    actor_role="manager",
                    actor="venue",
                    at=_now(),
                    note=details
                )
                
                s.add(event)
                s.commit()
            
            st.success("Η απόφασή σας υποβλήθηκε επιτυχώς!")
            st.rerun()


def render_venue_readonly(context: Dict[str, Any]) -> None:
    """Μόνο για ανάγνωση"""
    workflow = context["workflow"]
    
    status_colors = {
        "MATCHED_WITH_INVOICE": "🟢",
        "CLOSED": "✅",
        "SUPPLIER_CREDIT_NOTE_ISSUED": "💰",
        "SUPPLEMENTARY_DELIVERY_SENT": "📦"
    }
    
    status_icon = status_colors.get(workflow.state, "📊")
    
    st.success(f"""
    {status_icon} **{workflow.state}**
    
    Αυτή η παραγγελία δεν απαιτεί περαιτέρω ενέργειες.
    Όλες οι διαδικασίες έχουν ολοκληρωθεί.
    """)
    
    # Προβολή συνοπτικών πληροφοριών
    if context["receipt"]:
        receipt = context["receipt"]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Τιμολόγιο", receipt.invoice_number or "—")
        with col2:
            st.metric("Παραλήφθηκε", "✅" if receipt.received else "❌")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def modern_seguimiento_page():
    """Κύρια σελίδα seguimiento"""
    st.set_page_config(
        page_title="Follow-up System | Order Tracking",
        page_icon="🔄",
        layout="centered"
    )
    
    # Εισαγωγή CSS
    _inject_seguimiento_css()
    
    # Παράμετροι URL
    qp = st.query_params
    order_id = _f(qp.get("order_id"))
    provider = _s(qp.get("provider"))
    role = _s(qp.get("role"))
    sig = _s(qp.get("sig"))
    
    # Επικύρωση παραμέτρων
    if not all([order_id, provider, role, sig]):
        st.error("⚠️ Λείπουν απαραίτητες παράμετροι")
        st.stop()
    
    # Επικύρωση υπογραφής
    if not verify_link(order_id=int(order_id), provider_name=provider, role=role, sig=sig):
        st.error("🔒 Μη έγκυρος ή παραποιημένος σύνδεσμος")
        st.stop()
    
    # Προσδιορισμός ρόλου
    role_norm = norm_role(role)
    provider_norm = norm_provider(provider)
    
    # Φόρτωση δεδομένων
    try:
        context, actual_role = load_order_context(
            order_id=int(order_id),
            provider_name=provider_norm,
            role=role_norm
        )
    except Exception as e:
        st.error(f"❌ Σφάλμα φόρτωσης: {e}")
        st.stop()
    
    # Εμφάνιση badge ρόλου
    role_badge(actual_role)
    
    # Προσθήκη container για consistent styling
    st.markdown('<div class="followup-container">', unsafe_allow_html=True)
    
    # Επιλογή interface βάσει ρόλου
    if actual_role == ROLE_SUPPLIER:
        render_supplier_interface(context)
    else:
        render_venue_interface(context)
    
    # Κλείσιμο container
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.caption(f"Παραγγελία #{order_id} · {provider_norm} · {datetime.now().strftime('%d/%m/%Y %H:%M')}")


if __name__ == "__main__":
    modern_seguimiento_page()