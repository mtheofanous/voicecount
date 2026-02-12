"""
PRACTICAL INTEGRATION GUIDE
============================
How to add modern container styling to your receive_orders_page.py

This file shows specific examples of where and how to add modern containers
in your existing codebase.
"""

# =============================
# STEP 1: Add Modern CSS to Your Page
# =============================

def inject_modern_css():
    """
    Add this function to your receive_orders_page.py
    Call it at the beginning of tracking_dashboard()
    """
    css = """
    <style>
    /* Modern Design System */
    :root {
        --primary: #FF6B35;
        --success: #10B981;
        --warning: #F59E0B;
        --error: #EF4444;
        --bg-main: #FAFAFA;
        --card-bg: #FFFFFF;
        --border: #E5E7EB;
    }
    
    /* Provider Card Container */
    .st-key-provider-card {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 20px;
        border-left: 4px solid var(--primary);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin: 12px 0;
        transition: all 0.3s ease;
    }
    
    .st-key-provider-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
    }
    
    /* KPI Cards */
    .st-key-kpi-pending {
        background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
        border-radius: 14px;
        padding: 18px;
        border: 1px solid var(--border);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .st-key-kpi-pending:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        border-color: var(--primary);
    }
    
    .st-key-kpi-issues {
        background: linear-gradient(135deg, #FFFFFF 0%, #FEE2E2 100%);
        border-radius: 14px;
        padding: 18px;
        border: 1px solid #FCA5A5;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .st-key-kpi-issues:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(239, 68, 68, 0.15);
    }
    
    /* Product Line - Normal */
    .st-key-product-line {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid var(--border);
        margin: 10px 0;
        transition: all 0.2s ease;
    }
    
    .st-key-product-line:hover {
        border-color: var(--primary);
        box-shadow: 0 2px 8px rgba(255, 107, 53, 0.1);
    }
    
    /* Product Line - With Issue */
    .st-key-product-issue {
        background: linear-gradient(135deg, #FFFFFF 0%, #FEE2E2 100%);
        border-radius: 12px;
        padding: 16px;
        border: 2px solid #FCA5A5;
        margin: 10px 0;
    }
    
    /* Search Bar */
    .st-key-search-bar {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 16px;
        border: 1px solid var(--border);
        margin: 16px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Status Badge Container */
    .st-key-status-complete {
        background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
        border-radius: 12px;
        padding: 12px 16px;
        border: 1px solid #6EE7B7;
        margin: 8px 0;
    }
    
    .st-key-status-pending {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border-radius: 12px;
        padding: 12px 16px;
        border: 1px solid #FCD34D;
        margin: 8px 0;
    }
    
    .st-key-status-issue {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border-radius: 12px;
        padding: 12px 16px;
        border: 1px solid #FCA5A5;
        margin: 8px 0;
    }
    
    /* Invoice Summary Container */
    .st-key-invoice-summary {
        background: linear-gradient(135deg, #FEF9E7 0%, #FFF5DC 100%);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #F4E5C2;
        margin: 16px 0;
    }
    
    /* Timeline Container */
    .st-key-timeline {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid var(--border);
        position: relative;
    }
    </style>
    """
    st.html(css)


# =============================
# STEP 2: Modify KPI Section
# =============================

def render_kpi_section_modern(pending, issues, redeliveries, credits, urgent):
    """
    Replace the existing KPI section in tracking_dashboard() with this
    
    OLD CODE (around line 5469):
    st.markdown(
        "<div class='voi-kpi'>"
        f"<div class='k'>...</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    
    NEW CODE:
    """
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        with st.container(key="kpi-pending"):
            st.markdown("**Pending**")
            st.markdown(f"<h2 style='margin:0; color:#FF6B35;'>{pending}</h2>", 
                       unsafe_allow_html=True)
    
    with col2:
        with st.container(key="kpi-issues"):
            st.markdown("**Issues**")
            st.markdown(f"<h2 style='margin:0; color:#EF4444;'>{issues}</h2>", 
                       unsafe_allow_html=True)
    
    with col3:
        with st.container(key="kpi-pending"):
            st.markdown("**Re-delivery**")
            st.markdown(f"<h2 style='margin:0; color:#F59E0B;'>{redeliveries}</h2>", 
                       unsafe_allow_html=True)
    
    with col4:
        with st.container(key="kpi-pending"):
            st.markdown("**Credits**")
            st.markdown(f"<h2 style='margin:0; color:#10B981;'>{credits}</h2>", 
                       unsafe_allow_html=True)
    
    with col5:
        with st.container(key="kpi-issues"):
            st.markdown("**Urgent**")
            st.markdown(f"<h2 style='margin:0; color:#EF4444;'>{urgent}</h2>", 
                       unsafe_allow_html=True)


# =============================
# STEP 3: Modify Search Bar
# =============================

def render_search_bar_modern():
    """
    Replace the search bar section (around line 5504-5508) with this
    
    OLD CODE:
    f1, f2 = st.columns([2.2, 1.0], vertical_alignment="center")
    with f1:
        q = st.text_input(...)
    with f2:
        expand_all = st.toggle(...)
    
    NEW CODE:
    """
    with st.container(key="search-bar"):
        f1, f2 = st.columns([2.2, 1.0], vertical_alignment="center")
        with f1:
            q = st.text_input(
                "Search provider / invoice / order", 
                placeholder="e.g. makro, 2026-, #12"
            ).strip().lower()
        with f2:
            expand_all = st.toggle("Expand all", value=False)
    
    return q, expand_all


# =============================
# STEP 4: Modify Provider Cards
# =============================

def render_provider_card_modern(provider_name, invoice_number, order_id, order_date):
    """
    Wrap provider rendering in a modern container
    
    In _render_receive_provider_panel() function, add container wrapper:
    
    OLD CODE:
    def _render_receive_provider_panel(ctx, prov):
        # existing code...
    
    NEW CODE:
    """
    def _render_receive_provider_panel(ctx, prov):
        # Generate unique key for this provider card
        provider_key = f"provider-card-{order_id}-{norm_provider(prov)}"
        
        with st.container(key=provider_key):
            # All existing rendering code goes here
            # ... existing code ...
            pass


# =============================
# STEP 5: Modify Product Lines
# =============================

def render_product_line_modern(product_name, expected_qty, received_qty, has_issue):
    """
    Wrap each product line in a styled container
    
    In the product rendering loop, wrap with container:
    
    OLD CODE:
    for line in lines:
        st.markdown(f"<div class='line'>...</div>")
    
    NEW CODE:
    """
    for idx, line in enumerate(lines):
        # Determine if product has an issue
        has_issue = (received_qty != expected_qty) or line.has_discrepancy
        
        # Use different container key based on status
        container_key = f"product-issue-{idx}" if has_issue else f"product-line-{idx}"
        
        with st.container(key=container_key):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{product_name}**")
                if has_issue:
                    st.markdown(
                        f"⚠️ <small style='color:#DC2626;'>"
                        f"Received: {received_qty} • Expected: {expected_qty}</small>",
                        unsafe_allow_html=True
                    )
            
            with col2:
                if has_issue:
                    st.markdown(f"**✗ {received_qty - expected_qty}**")
                else:
                    st.markdown(f"**✓ {received_qty}**")


# =============================
# STEP 6: Modify Invoice Summary
# =============================

def render_invoice_summary_modern(subtotal, discount, tax, total):
    """
    Wrap invoice summary in a colored container
    
    Where you show the invoice totals, wrap with:
    """
    with st.container(key="invoice-summary"):
        st.markdown("### 📋 Invoice Summary")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"**Subtotal:** €{subtotal:.2f}")
            if discount > 0:
                st.markdown(f"**Discount:** -€{discount:.2f}")
            st.markdown(f"**Tax (IVA):** €{tax:.2f}")
        
        with col2:
            st.markdown(
                f"<h3 style='text-align:right; color:#FF6B35;'>"
                f"Total: €{total:.2f}</h3>",
                unsafe_allow_html=True
            )


# =============================
# STEP 7: Status Messages with Containers
# =============================

def render_status_messages_modern():
    """
    Replace status messages with styled containers
    
    OLD CODE:
    st.success("✅ Nothing pending to receive right now.")
    
    NEW CODE:
    """
    # Success status
    with st.container(key="status-complete"):
        st.markdown("✅ **All orders verified** • Nothing pending to receive")
    
    # Pending status
    with st.container(key="status-pending"):
        st.markdown("⏳ **Orders awaiting verification** • 5 orders need review")
    
    # Issue status
    with st.container(key="status-issue"):
        st.markdown("⚠️ **Attention required** • 3 orders have discrepancies")


# =============================
# STEP 8: Complete Integration Example
# =============================

def tracking_dashboard_modern_example(venue_id, deep_order_id=None, deep_provider=None):
    """
    Complete example showing how to integrate all modern containers
    This is a simplified version showing the structure
    """
    
    # 1. Inject modern CSS
    inject_modern_css()
    
    # 2. Page title
    st.markdown("# 📦 Receive & Track Orders")
    
    # 3. Modern KPI section
    pending = 47
    issues = 12
    redeliveries = 8
    credits = 5
    urgent = 3
    
    render_kpi_section_modern(pending, issues, redeliveries, credits, urgent)
    
    # 4. Tabs
    tab1, tab2, tab3 = st.tabs(["📦 Receive", "🚨 Incidences", "⚡ Urgent"])
    
    with tab1:
        # 5. Search bar
        q, expand_all = render_search_bar_modern()
        
        # 6. Provider cards
        for order in orders:
            provider_key = f"provider-card-{order.id}"
            
            with st.container(key=provider_key):
                st.markdown(f"### 🏪 {order.provider_name}")
                st.markdown(f"📋 **Invoice:** {order.invoice_number}")
                st.markdown(f"📅 **Date:** {order.date}")
                
                st.markdown("---")
                
                # 7. Product lines
                for idx, product in enumerate(order.products):
                    has_issue = product.received != product.expected
                    container_key = f"product-{'issue' if has_issue else 'line'}-{idx}"
                    
                    with st.container(key=container_key):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{product.name}**")
                            if has_issue:
                                st.markdown(
                                    f"⚠️ Received: {product.received} • "
                                    f"Expected: {product.expected}"
                                )
                        with col2:
                            st.markdown(f"**{product.received} units**")
                
                # 8. Invoice summary
                with st.container(key=f"invoice-summary-{order.id}"):
                    st.markdown("### 📋 Summary")
                    st.markdown(f"**Total:** €{order.total:.2f}")
                
                # 9. Actions
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.button("✅ Verify", key=f"verify-{order.id}")
                with col2:
                    st.button("⚠️ Report Issue", key=f"issue-{order.id}")
                with col3:
                    st.button("📄 Details", key=f"details-{order.id}")
    
    with tab2:
        render_status_messages_modern()
        # ... incidences code ...
    
    with tab3:
        # ... urgent code ...
        pass


# =============================
# QUICK START CHECKLIST
# =============================

"""
QUICK START CHECKLIST:
=====================

1. ✅ Add inject_modern_css() function to your receive_orders_page.py

2. ✅ Call inject_modern_css() at the start of tracking_dashboard()
   
   def tracking_dashboard(venue_id, deep_order_id=None, deep_provider=None):
       inject_modern_css()  # ADD THIS LINE
       # ... rest of code

3. ✅ Wrap KPI section in modern containers (Step 2)

4. ✅ Wrap search bar in container (Step 3)

5. ✅ Wrap each provider card in container (Step 4)
   
   In _render_receive_provider_panel():
       with st.container(key=f"provider-card-{order_id}"):
           # existing code...

6. ✅ Wrap each product line in container (Step 5)

7. ✅ Wrap invoice summaries in containers (Step 6)

8. ✅ Test on mobile device or browser DevTools mobile view

BENEFITS:
---------
✨ Modern, professional appearance
📱 Better mobile experience
🎨 Consistent design language
🚀 Smooth hover animations
♿ Better visual hierarchy
🔧 Easy to customize colors
"""

# =============================
# COLOR CUSTOMIZATION
# =============================

"""
CUSTOMIZING COLORS:
==================

To change the color scheme, modify the CSS variables in inject_modern_css():

CURRENT (Restaurant/Bar Orange):
--primary: #FF6B35;

ALTERNATIVES:

Blue (Professional):
--primary: #3B82F6;

Green (Fresh/Organic):
--primary: #10B981;

Purple (Modern/Tech):
--primary: #8B5CF6;

Red (Bold/Energetic):
--primary: #EF4444;

Teal (Clean/Medical):
--primary: #14B8A6;

Just change the --primary variable and all components will update!
"""
