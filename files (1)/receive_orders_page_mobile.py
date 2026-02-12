from __future__ import annotations

"""
Modern Mobile-First Receive Orders Dashboard
Enhanced UX for restaurant/bar order receiving with improved visual design
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
# MODERN UI: Enhanced Styles
# =============================

def _inject_modern_css() -> None:
    """Modern mobile-first CSS with restaurant/bar aesthetics"""
    st.markdown(
        """
<style>
/* ========== Modern Design Tokens ========== */
:root {
    /* Primary Palette - Warm & Inviting */
    --primary: #FF6B35;
    --primary-light: #FF8C61;
    --primary-dark: #E85A2A;
    
    /* Neutrals - Clean & Professional */
    --bg-main: #FAFAFA;
    --card-bg: #FFFFFF;
    --text-primary: #1A1A1A;
    --text-secondary: #6B7280;
    --text-muted: #9CA3AF;
    
    /* Borders & Dividers */
    --border-light: #E5E7EB;
    --border-medium: #D1D5DB;
    
    /* Status Colors */
    --success: #10B981;
    --success-bg: #D1FAE5;
    --success-border: #6EE7B7;
    
    --warning: #F59E0B;
    --warning-bg: #FEF3C7;
    --warning-border: #FCD34D;
    
    --error: #EF4444;
    --error-bg: #FEE2E2;
    --error-border: #FCA5A5;
    
    --info: #3B82F6;
    --info-bg: #DBEAFE;
    --info-border: #93C5FD;
    
    /* Shadows */
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.08);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.10);
    --shadow-lg: 0 10px 25px rgba(0,0,0,0.12);
    
    /* Radius */
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 20px;
    --radius-full: 9999px;
}

/* ========== Base Layout ========== */
.block-container {
    padding-top: 1rem;
    padding-bottom: 6rem; /* Space for bottom nav */
    max-width: 1200px;
    background: var(--bg-main);
}

/* ========== Modern Card System ========== */
.modern-card {
    background: var(--card-bg);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-xl);
    padding: 20px;
    margin: 16px 0;
    box-shadow: var(--shadow-sm);
    transition: all 0.3s ease;
}

.modern-card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}

.modern-card.elevated {
    box-shadow: var(--shadow-md);
}

.modern-card.elevated:hover {
    box-shadow: var(--shadow-lg);
}

/* Card with accent border */
.modern-card.accent-primary {
    border-left: 4px solid var(--primary);
}

.modern-card.accent-success {
    border-left: 4px solid var(--success);
}

.modern-card.accent-warning {
    border-left: 4px solid var(--warning);
}

.modern-card.accent-error {
    border-left: 4px solid var(--error);
}

/* ========== Typography ========== */
.card-title {
    font-weight: 800;
    font-size: 1.25rem;
    color: var(--text-primary);
    margin-bottom: 8px;
    letter-spacing: -0.02em;
}

.card-subtitle {
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--text-secondary);
    margin-bottom: 12px;
}

.text-muted {
    color: var(--text-muted);
    font-size: 0.9rem;
}

.text-small {
    font-size: 0.85rem;
    color: var(--text-secondary);
}

/* ========== Modern Badge System ========== */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: var(--radius-full);
    font-weight: 700;
    font-size: 0.8rem;
    line-height: 1;
    white-space: nowrap;
}

.badge.badge-success {
    background: var(--success-bg);
    color: #065F46;
    border: 1px solid var(--success-border);
}

.badge.badge-warning {
    background: var(--warning-bg);
    color: #92400E;
    border: 1px solid var(--warning-border);
}

.badge.badge-error {
    background: var(--error-bg);
    color: #991B1B;
    border: 1px solid var(--error-border);
}

.badge.badge-info {
    background: var(--info-bg);
    color: #1E40AF;
    border: 1px solid var(--info-border);
}

.badge.badge-neutral {
    background: #F3F4F6;
    color: var(--text-primary);
    border: 1px solid var(--border-medium);
}

/* Badge with icon */
.badge .icon {
    font-size: 0.9rem;
}

/* ========== KPI Dashboard Cards ========== */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin: 20px 0;
}

.kpi-card {
    background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-lg);
    padding: 16px;
    text-align: center;
    transition: all 0.3s ease;
}

.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-md);
    border-color: var(--primary-light);
}

.kpi-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 2rem;
    font-weight: 900;
    color: var(--text-primary);
    line-height: 1;
}

.kpi-value.success {
    color: var(--success);
}

.kpi-value.warning {
    color: var(--warning);
}

.kpi-value.error {
    color: var(--error);
}

/* ========== Product Line Cards ========== */
.product-line {
    background: var(--card-bg);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-md);
    padding: 16px;
    margin: 10px 0;
    transition: all 0.2s ease;
}

.product-line:hover {
    border-color: var(--primary-light);
    box-shadow: var(--shadow-sm);
}

.product-line.has-issue {
    border-color: var(--error);
    background: linear-gradient(135deg, #FFFFFF 0%, var(--error-bg) 100%);
}

.product-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 8px;
}

.product-name {
    font-weight: 800;
    font-size: 1rem;
    color: var(--text-primary);
    flex: 1;
}

.product-quantity {
    font-weight: 800;
    font-size: 1.1rem;
    color: var(--primary);
    white-space: nowrap;
}

.product-description {
    color: var(--text-muted);
    font-size: 0.85rem;
    margin-top: 4px;
    line-height: 1.4;
}

/* ========== Pills & Tags ========== */
.pill-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 12px;
}

.pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: var(--radius-full);
    font-size: 0.8rem;
    font-weight: 700;
    border: 1px solid var(--border-light);
    background: #F9FAFB;
    color: var(--text-primary);
}

.pill.pill-success {
    background: var(--success-bg);
    color: #065F46;
    border-color: var(--success-border);
}

.pill.pill-warning {
    background: var(--warning-bg);
    color: #92400E;
    border-color: var(--warning-border);
}

.pill.pill-error {
    background: var(--error-bg);
    color: #991B1B;
    border-color: var(--error-border);
}

/* ========== Invoice Tables ========== */
.invoice-table-wrapper {
    margin: 16px 0;
    border-radius: var(--radius-md);
    overflow: hidden;
    border: 1px solid var(--border-light);
}

.invoice-table {
    width: 100%;
    border-collapse: collapse;
    font-variant-numeric: tabular-nums;
}

.invoice-table thead th {
    background: linear-gradient(180deg, #F9FAFB 0%, #F3F4F6 100%);
    color: var(--text-secondary);
    font-weight: 800;
    text-transform: uppercase;
    font-size: 0.7rem;
    letter-spacing: 0.05em;
    padding: 12px;
    text-align: left;
    border-bottom: 2px solid var(--border-medium);
}

.invoice-table tbody td {
    padding: 12px;
    border-bottom: 1px solid var(--border-light);
    font-size: 0.9rem;
}

.invoice-table tbody tr:last-child td {
    border-bottom: none;
}

.invoice-table tbody tr:hover {
    background: var(--bg-main);
}

.invoice-table td.product-name {
    font-weight: 700;
    color: var(--text-primary);
}

.invoice-table td.numeric {
    text-align: right;
    font-weight: 600;
}

.invoice-table tfoot td {
    background: #F9FAFB;
    font-weight: 800;
    padding: 14px 12px;
    border-top: 2px solid var(--border-medium);
}

/* ========== Dividers ========== */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-light), transparent);
    margin: 20px 0;
}

.divider.thick {
    height: 2px;
    background: var(--border-medium);
}

/* ========== Action Buttons Enhanced ========== */
.action-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 16px;
}

/* ========== Comment Boxes ========== */
.comment-box {
    background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
    border: 1px solid var(--warning-border);
    border-radius: var(--radius-md);
    padding: 12px 16px;
    margin: 12px 0;
}

.comment-box.info {
    background: linear-gradient(135deg, var(--info-bg) 0%, #BFDBFE 100%);
    border-color: var(--info-border);
}

.comment-box.success {
    background: linear-gradient(135deg, var(--success-bg) 0%, #A7F3D0 100%);
    border-color: var(--success-border);
}

.comment-label {
    font-weight: 800;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    margin-bottom: 4px;
}

.comment-text {
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--text-primary);
    line-height: 1.5;
}

/* ========== Status Timeline ========== */
.timeline {
    position: relative;
    padding-left: 24px;
    margin: 20px 0;
}

.timeline::before {
    content: '';
    position: absolute;
    left: 7px;
    top: 8px;
    bottom: 8px;
    width: 2px;
    background: var(--border-light);
}

.timeline-item {
    position: relative;
    margin-bottom: 20px;
}

.timeline-item::before {
    content: '';
    position: absolute;
    left: -20px;
    top: 6px;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--card-bg);
    border: 3px solid var(--primary);
    box-shadow: 0 0 0 4px var(--card-bg);
}

.timeline-item.completed::before {
    background: var(--success);
    border-color: var(--success);
}

.timeline-content {
    background: var(--card-bg);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-md);
    padding: 12px;
}

/* ========== Empty States ========== */
.empty-state {
    text-align: center;
    padding: 60px 20px;
}

.empty-state-icon {
    font-size: 4rem;
    opacity: 0.3;
    margin-bottom: 16px;
}

.empty-state-title {
    font-size: 1.2rem;
    font-weight: 800;
    color: var(--text-primary);
    margin-bottom: 8px;
}

.empty-state-text {
    font-size: 0.95rem;
    color: var(--text-secondary);
}

/* ========== Loading States ========== */
@keyframes shimmer {
    0% { background-position: -1000px 0; }
    100% { background-position: 1000px 0; }
}

.loading-skeleton {
    background: linear-gradient(90deg, #F3F4F6 25%, #E5E7EB 50%, #F3F4F6 75%);
    background-size: 2000px 100%;
    animation: shimmer 2s infinite;
    border-radius: var(--radius-md);
}

/* ========== Mobile Optimizations ========== */
@media (max-width: 768px) {
    .modern-card {
        padding: 16px;
        margin: 12px 0;
    }
    
    .kpi-grid {
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
    }
    
    .kpi-value {
        font-size: 1.75rem;
    }
    
    .card-title {
        font-size: 1.1rem;
    }
    
    .product-header {
        flex-direction: column;
        gap: 8px;
    }
    
    .product-quantity {
        align-self: flex-start;
    }
}

/* ========== Touch-Friendly Interactions ========== */
@media (hover: none) {
    .modern-card:active {
        transform: scale(0.98);
    }
    
    .kpi-card:active {
        transform: scale(0.95);
    }
}

/* ========== Accessibility ========== */
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border-width: 0;
}

/* Focus styles for keyboard navigation */
button:focus-visible,
input:focus-visible,
select:focus-visible {
    outline: 3px solid var(--primary);
    outline-offset: 2px;
}
</style>
""",
        unsafe_allow_html=True,
    )


# =============================
# Re-export all functions from original module
# This ensures compatibility while using modern styles
# =============================

# Import all necessary functions from the original module
import sys
import importlib.util

# Load the original module
spec = importlib.util.spec_from_file_location(
    "receive_orders_page_original", 
    "/mnt/project/receive_orders_page.py"
)
original_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original_module)

# Re-export all functions except _inject_css
for name in dir(original_module):
    if not name.startswith('_') or name == '_load_order_context':
        globals()[name] = getattr(original_module, name)

# Override the CSS injection
globals()['_inject_css'] = _inject_modern_css


# =============================
# Enhanced wrapper for tracking dashboard
# =============================

def tracking_dashboard(
    venue_id: int,
    *,
    deep_order_id: Optional[int] = None,
    deep_provider: Optional[str] = None,
) -> None:
    """
    Modern mobile-first tracking dashboard
    Uses enhanced CSS and maintains all original functionality
    """
    # Inject modern CSS instead of original
    _inject_modern_css()
    
    # Call the original tracking dashboard with all the same logic
    return original_module.tracking_dashboard(
        venue_id,
        deep_order_id=deep_order_id,
        deep_provider=deep_provider,
    )


# Export all other functions
def find_alternative_providers(*args: Any, **kwargs: Any):
    return original_module.find_alternative_providers(*args, **kwargs)

def upsert_provider_invoice_number(*args: Any, **kwargs: Any):
    return original_module.upsert_provider_invoice_number(*args, **kwargs)

def save_all_received_for_provider(*args: Any, **kwargs: Any):
    return original_module.save_all_received_for_provider(*args, **kwargs)

def request_supplier_resolution(*args: Any, **kwargs: Any):
    return original_module.request_supplier_resolution(*args, **kwargs)

def supplier_decision_from_venue(*args: Any, **kwargs: Any):
    return original_module.supplier_decision_from_venue(*args, **kwargs)

def venue_verify_and_close(*args: Any, **kwargs: Any):
    return original_module.venue_verify_and_close(*args, **kwargs)

def resolve_operational_missing(*args: Any, **kwargs: Any):
    return original_module.resolve_operational_missing(*args, **kwargs)
