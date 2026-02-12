"""
MODERN CONTAINER STYLING GUIDE
===============================
This file demonstrates how to use modern styled containers in your Streamlit app
for restaurant/bar order management
"""

import streamlit as st

# =============================
# EXAMPLE 1: Basic Modern Container with Colored Background
# =============================

def example_basic_colored_container():
    """Example of using st.container with custom background colors"""
    
    # Define custom CSS for this container
    css = """
        .st-key-order-summary {
            background: linear-gradient(135deg, #FEF9E7 0%, #FFF5DC 100%);
            border-radius: 20px;
            padding: 24px;
            border: 1px solid #F4E5C2;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
    """
    st.html(f"<style>{css}</style>")
    
    with st.container(key="order-summary"):
        st.markdown("### 📦 Order Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Items", "24", "+3")
        with col2:
            st.metric("Total Cost", "€342.50", "+€25")
        with col3:
            st.metric("Suppliers", "5", "+1")


# =============================
# EXAMPLE 2: Provider Card with Accent Border
# =============================

def example_provider_card():
    """Example of a provider card with left accent border"""
    
    css = """
        .st-key-provider-makro {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 20px;
            border-left: 4px solid #FF6B35;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            margin: 12px 0;
        }
    """
    st.html(f"<style>{css}</style>")
    
    with st.container(key="provider-makro"):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### 🏪 MAKRO")
            st.markdown("📋 Invoice: **2026-001234**")
            st.markdown("📅 Order Date: **Feb 12, 2026**")
        with col2:
            st.button("✅ Verify", use_container_width=True)
            st.button("⚠️ Issue", use_container_width=True, type="secondary")


# =============================
# EXAMPLE 3: Status Cards with Different Colors
# =============================

def example_status_cards():
    """Example of status cards with different background colors"""
    
    # Success Card (Green theme)
    css_success = """
        .st-key-status-complete {
            background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
            border-radius: 16px;
            padding: 18px;
            border: 1px solid #6EE7B7;
            margin: 10px 0;
        }
    """
    
    # Warning Card (Amber theme)
    css_warning = """
        .st-key-status-pending {
            background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
            border-radius: 16px;
            padding: 18px;
            border: 1px solid #FCD34D;
            margin: 10px 0;
        }
    """
    
    # Error Card (Red theme)
    css_error = """
        .st-key-status-issue {
            background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
            border-radius: 16px;
            padding: 18px;
            border: 1px solid #FCA5A5;
            margin: 10px 0;
        }
    """
    
    st.html(f"<style>{css_success}{css_warning}{css_error}</style>")
    
    # Complete orders
    with st.container(key="status-complete"):
        st.markdown("✅ **Completed Orders** • 12 orders verified")
        
    # Pending orders
    with st.container(key="status-pending"):
        st.markdown("⏳ **Pending Review** • 5 orders awaiting verification")
        
    # Orders with issues
    with st.container(key="status-issue"):
        st.markdown("⚠️ **Orders with Issues** • 3 orders need attention")


# =============================
# EXAMPLE 4: KPI Dashboard Cards
# =============================

def example_kpi_dashboard():
    """Example of KPI cards with hover effects"""
    
    css = """
        /* KPI Card 1 */
        .st-key-kpi-pending {
            background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
            border-radius: 14px;
            padding: 20px;
            border: 1px solid #E5E7EB;
            text-align: center;
            transition: all 0.3s ease;
        }
        .st-key-kpi-pending:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
            border-color: #FF6B35;
        }
        
        /* KPI Card 2 */
        .st-key-kpi-issues {
            background: linear-gradient(135deg, #FFFFFF 0%, #FEF2F2 100%);
            border-radius: 14px;
            padding: 20px;
            border: 1px solid #FCA5A5;
            text-align: center;
            transition: all 0.3s ease;
        }
        .st-key-kpi-issues:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(239, 68, 68, 0.15);
        }
        
        /* KPI Card 3 */
        .st-key-kpi-urgent {
            background: linear-gradient(135deg, #FFFFFF 0%, #FEF3C7 100%);
            border-radius: 14px;
            padding: 20px;
            border: 1px solid #FCD34D;
            text-align: center;
            transition: all 0.3s ease;
        }
        .st-key-kpi-urgent:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(245, 158, 11, 0.15);
        }
    """
    st.html(f"<style>{css}</style>")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(key="kpi-pending"):
            st.markdown("**PENDING PRODUCTS**")
            st.markdown("<h1 style='margin:8px 0; color:#FF6B35;'>47</h1>", unsafe_allow_html=True)
            
    with col2:
        with st.container(key="kpi-issues"):
            st.markdown("**OPEN ISSUES**")
            st.markdown("<h1 style='margin:8px 0; color:#EF4444;'>12</h1>", unsafe_allow_html=True)
            
    with col3:
        with st.container(key="kpi-urgent"):
            st.markdown("**URGENT REQUESTS**")
            st.markdown("<h1 style='margin:8px 0; color:#F59E0B;'>3</h1>", unsafe_allow_html=True)


# =============================
# EXAMPLE 5: Product Line Card
# =============================

def example_product_line():
    """Example of a product line item with issue highlighting"""
    
    css_normal = """
        .st-key-product-normal {
            background: #FFFFFF;
            border-radius: 12px;
            padding: 16px;
            border: 1px solid #E5E7EB;
            margin: 8px 0;
            transition: all 0.2s ease;
        }
        .st-key-product-normal:hover {
            border-color: #FF6B35;
            box-shadow: 0 2px 8px rgba(255, 107, 53, 0.1);
        }
    """
    
    css_issue = """
        .st-key-product-issue {
            background: linear-gradient(135deg, #FFFFFF 0%, #FEE2E2 100%);
            border-radius: 12px;
            padding: 16px;
            border: 2px solid #FCA5A5;
            margin: 8px 0;
        }
    """
    
    st.html(f"<style>{css_normal}{css_issue}</style>")
    
    # Normal product
    with st.container(key="product-normal"):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**Jamón Ibérico de Bellota**")
            st.markdown("<small style='color:#6B7280;'>SKU: JAM-001 • 2.5kg</small>", unsafe_allow_html=True)
        with col2:
            st.markdown("**✓ 5 units**")
    
    # Product with issue
    with st.container(key="product-issue"):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**Aceite de Oliva Virgen Extra**")
            st.markdown("<small style='color:#6B7280;'>SKU: ACE-012 • 5L</small>", unsafe_allow_html=True)
            st.markdown("⚠️ <small style='color:#DC2626;'>**Received: 8 units** • Expected: 10 units</small>", unsafe_allow_html=True)
        with col2:
            st.markdown("**✗ -2 units**")


# =============================
# EXAMPLE 6: Search and Filter Bar
# =============================

def example_search_filter_bar():
    """Example of a sticky search/filter bar"""
    
    css = """
        .st-key-search-bar {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 16px;
            border: 1px solid #E5E7EB;
            margin: 16px 0;
            position: sticky;
            top: 60px;
            z-index: 100;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
    """
    st.html(f"<style>{css}</style>")
    
    with st.container(key="search-bar"):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text_input("🔍 Search orders, suppliers, or invoices...", 
                         placeholder="e.g., MAKRO, 2026-, #12",
                         label_visibility="collapsed")
        with col2:
            st.selectbox("Filter", ["All", "Pending", "Issues", "Completed"],
                        label_visibility="collapsed")


# =============================
# EXAMPLE 7: Timeline/History Container
# =============================

def example_timeline():
    """Example of a timeline/history view"""
    
    css = """
        .st-key-timeline {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 20px;
            border: 1px solid #E5E7EB;
            position: relative;
        }
        
        /* Timeline line */
        .st-key-timeline::before {
            content: '';
            position: absolute;
            left: 32px;
            top: 50px;
            bottom: 50px;
            width: 2px;
            background: linear-gradient(180deg, #FF6B35, #E5E7EB);
        }
    """
    st.html(f"<style>{css}</style>")
    
    with st.container(key="timeline"):
        st.markdown("### 📜 Order History")
        
        # Timeline items would go here
        st.markdown("🟢 **Verified** • Feb 12, 14:30")
        st.markdown("🟡 **Received** • Feb 12, 10:15")
        st.markdown("🔵 **Shipped** • Feb 11, 16:00")
        st.markdown("⚪ **Ordered** • Feb 10, 09:30")


# =============================
# EXAMPLE 8: Full Page Layout
# =============================

def example_full_page_layout():
    """Complete page layout example combining multiple container styles"""
    
    st.markdown("# 📦 Receive Orders")
    
    # KPI Dashboard
    st.markdown("## Dashboard")
    example_kpi_dashboard()
    
    # Search/Filter
    example_search_filter_bar()
    
    # Status Cards
    st.markdown("## Status Overview")
    example_status_cards()
    
    # Provider Card
    st.markdown("## Active Orders")
    example_provider_card()
    
    # Product Lines
    st.markdown("### Products")
    example_product_line()
    
    # Timeline
    st.markdown("## History")
    example_timeline()


# =============================
# MAIN DEMO
# =============================

if __name__ == "__main__":
    st.set_page_config(page_title="Modern Container Styling Guide", layout="wide")
    
    st.title("🎨 Modern Container Styling Guide")
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Basic Examples",
        "📊 KPI Dashboard",
        "🏪 Provider Cards",
        "📄 Full Layout"
    ])
    
    with tab1:
        st.markdown("## Basic Colored Containers")
        example_basic_colored_container()
        
        st.markdown("---")
        st.markdown("## Status Cards")
        example_status_cards()
        
        st.markdown("---")
        st.markdown("## Product Lines")
        example_product_line()
    
    with tab2:
        st.markdown("## KPI Dashboard")
        example_kpi_dashboard()
        
        st.markdown("### Code Example")
        st.code('''
css = """
    .st-key-kpi-pending {
        background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #E5E7EB;
        transition: all 0.3s ease;
    }
    .st-key-kpi-pending:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
    }
"""
st.html(f"<style>{css}</style>")

with st.container(key="kpi-pending"):
    st.markdown("**PENDING PRODUCTS**")
    st.metric("", "47")
        ''', language='python')
    
    with tab3:
        st.markdown("## Provider Cards")
        example_provider_card()
        
        st.markdown("### Code Example")
        st.code('''
css = """
    .st-key-provider-makro {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 20px;
        border-left: 4px solid #FF6B35;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
"""
st.html(f"<style>{css}</style>")

with st.container(key="provider-makro"):
    st.markdown("### 🏪 MAKRO")
    st.markdown("📋 Invoice: **2026-001234**")
        ''', language='python')
    
    with tab4:
        st.markdown("## Complete Page Layout")
        example_full_page_layout()
