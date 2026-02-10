# fast_app.py
# Mobile-first, fast navigation shell for Streamlit Cloud
# - avoids st.tabs (heavy DOM)
# - lazy-loads page modules only when needed
# - caches bootstrapping + DB engines
# - URL-sync via ?page=... for deep links
#
# Drop-in: run this file instead of app.py
#
# Depends on:
# - core.config.ensure_google_credentials_file
# - core.init.init_db
# - features.auth_and_manage.auth_multi_tenant (auth + venue context)
# - core.url_nav (query params helpers)

from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv
import warnings
import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------------
# 1) Load ENV FIRST
# -------------------------------
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

# -------------------------------
# 2) Page config (mobile-friendly)
# -------------------------------
st.set_page_config(
    page_title="Voi",
    page_icon="🧾",
    layout="centered",
)

# -------------------------------
# 3) LIGHT imports only (fast)
# -------------------------------
from core.config import ensure_google_credentials_file
from core.init import init_db
from features.auth_and_manage.auth_multi_tenant import (
    init_auth_db,
    auth_gate,
    require_login,
    current_user,
    current_account,
    clear_auth,
    current_active_venue,
    current_venues_for_user,
    manage_organization_ui,
)
from core.url_nav import qp_int, qp_str, set_query_params

warnings.filterwarnings("ignore", message=".*use_container_width.*")
warnings.filterwarnings("ignore", message=".*label.*got an empty value.*")

# -------------------------------
# 4) One-time bootstrap (HUGE speedup)
# -------------------------------
@st.cache_resource(show_spinner=False)
def bootstrap_once():
    ensure_google_credentials_file()
    init_db()
    init_auth_db()
    return True

def _css():
    st.markdown(
        """
<style>
/* Mobile-first spacing: leave room for bottom bar */
.block-container{
  padding:3.75rem 0.75rem 5.5rem;
  max-width:100%;
}

/* Big taps everywhere */
button{
  min-height:44px;
  border-radius:14px;
  font-weight:900;
}

/* --- Bottom Tab Bar --- */
.voi-tabbar{
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  z-index: 9999;

  padding: 10px 10px calc(10px + env(safe-area-inset-bottom));
  background: rgba(255,255,255,.96);
  border-top: 1px solid rgba(148,163,184,.35);

  backdrop-filter: saturate(180%) blur(12px);
}


.voi-tabs{
  display:flex;
  gap:8px;
  justify-content:space-between;
  max-width:1100px;
  margin:0 auto;
}

.voi-tab{
  flex:1 1 0;
  text-decoration:none !important;
  color:#0f172a !important;
  border:1px solid rgba(148,163,184,.35);
  border-radius:16px;
  padding:10px 8px;
  background:#fff;
  text-align:center;
  font-weight:900;
  line-height:1.05;
  min-height:48px;
  display:flex;
  flex-direction:column;
  justify-content:center;
  gap:4px;
}

.voi-tab .ic{font-size:1.05rem;}
.voi-tab .tx{font-size:.78rem; opacity:.9; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}

.voi-tab.active{
  border-color: rgba(37,99,235,.45);
  box-shadow:0 6px 18px rgba(2,6,23,.06);
}


@media (min-width: 900px){
  /* On desktop, keep it but make it slightly tighter */
  .voi-tab .tx{font-size:.82rem;}
}
</style>
""",
        unsafe_allow_html=True,
    )


#  -------------------------------
# 5) Navigation + venue selector (URL-sync, mobile-friendly)
def _go(page_key: str, **extra_qp: str) -> None:
    """Navigate to a page and sync URL (?page=...)."""
    st.session_state["page"] = page_key
    
    # Preserve session token in URL
    params = {"page": page_key}
    token = st.session_state.get("_session_token")
    if token:
        params["st"] = token
    
    # Add any extra query params
    params.update({k: str(v) for k, v in extra_qp.items() if v is not None and str(v).strip()})
    
    set_query_params(**params)
    st.rerun()
    
def _bottom_tabbar(current_page: str) -> None:
    tabs = [
        ("new", "➕", "New"),
        ("orders", "📦", "Orders"),
        ("tracking", "✅", "Receive"),
        ("history", "🗂️", "History"),
        ("reports", "📈", "Reports"),
        ("catalog", "🧾", "Catalog"),
    ]

    # Get session token to include in links
    token = st.session_state.get("_session_token", "")
    token_param = f"&st={token}" if token else ""

    items = []
    for key, icon, label in tabs:
        active = "active" if key == current_page else ""
        href = f"?page={key}{token_param}"
        items.append(
f"""<a class="voi-tab {active}" href="{href}" target="_self">
  <div class="ic">{icon}</div>
  <div class="tx">{label}</div>
</a>"""
        )

    html = f"""<div class="voi-tabbar">
  <div class="voi-tabs">
    {''.join(items)}
  </div>
</div>"""

    st.markdown(html, unsafe_allow_html=True)




def _venue_selector_compact() -> int:
    """Compact venue selector (fast on mobile)."""
    venues = current_venues_for_user()
    if not venues:
        st.warning("You don't have access to any venue yet.")
        st.stop()

    labels, ids = [], []
    for v, role in venues:
        labels.append(f"{v['name']} — {role}")
        ids.append(int(v["id"]))

    st.session_state.setdefault("active_venue_id", ids[0])
    if int(st.session_state["active_venue_id"]) not in ids:
        st.session_state["active_venue_id"] = ids[0]

    idx = ids.index(int(st.session_state["active_venue_id"]))
    chosen = st.selectbox("Venue", options=labels, index=idx, label_visibility="collapsed")
    chosen_id = ids[labels.index(chosen)]
    if chosen_id != int(st.session_state["active_venue_id"]):
        st.session_state["active_venue_id"] = chosen_id
        st.rerun()
    return chosen_id


# -------------------------------
# 6) Lazy page loaders (import on demand)
# -------------------------------
def _page_catalog():
    from features.catalog import catalog_tab
    return catalog_tab

def _page_new_order():
    from features.create_order import new_order_tab
    return new_order_tab

def _page_orders():
    from features.manage_orders.orders import orders_tab
    return orders_tab

def _page_tracking():
    # wrapper lazily imports heavy implementation
    from features.manage_orders.receive_orders import tracking_dashboard
    return tracking_dashboard

def _page_history():
    from features.manage_orders.history import _render_history_tab
    return _render_history_tab

def _page_reports():
    from features.manage_orders.reports import reports_page
    return reports_page


PAGES = {
    "new": ("➕ New order", _page_new_order),
    "orders": ("📦 Orders", _page_orders),
    "tracking": ("✅ Receive / Tracking", _page_tracking),
    "history": ("🗂️ History", _page_history),
    "reports": ("📈 Reports", _page_reports),
    "catalog": ("🧾 Catalog", _page_catalog),
}

# Preferred order for the segmented control
PAGE_KEYS = ["new", "orders", "tracking", "history", "reports", "catalog"]


def _home_card():
    st.markdown("<div class='voi-card'>", unsafe_allow_html=True)
    st.title("Voi")
    st.markdown("<div class='voi-muted'>Fast, mobile-first navigation.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)



def _call_page(fn, venue_id: int, **kwargs):
    """Call a page function but only pass kwargs it actually accepts.

    This prevents TypeError when some pages don't support deep-link params.
    """
    import inspect
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        # Builtins / callables without signature: best effort
        return fn(venue_id)

    params = sig.parameters
    accepted = {k: v for k, v in kwargs.items() if k in params}
    return fn(venue_id, **accepted)


def main():
    bootstrap_once()
    _css()

    # Read page from URL (auth_gate now preserves these params)
    try:
        url_page = st.query_params.get("page", "").strip().lower()
    except:
        url_page = ""
    
    if url_page and url_page in PAGES:
        st.session_state["page"] = url_page
    elif "page" not in st.session_state:
        st.session_state["page"] = "orders"
    
    # Read other deep-link params
    try:
        deep_order_id = int(st.query_params.get("order_id", 0)) or None
    except:
        deep_order_id = None
    
    try:
        deep_provider = st.query_params.get("provider", "").strip() or None
    except:
        deep_provider = None
        
    try:
        deep_status = st.query_params.get("status", "").strip().lower() or None
    except:
        deep_status = None

    # Auth gate (now preserves URL params across reruns)
    auth_gate(show_manage_org=True, show_venue_selector=False)
    require_login()

    # Debug log for session state and page key
    logging.debug(f"Session state: {st.session_state}")
    logging.debug(f"Current page_key: {st.session_state.get('page')}")

    u = current_user() or {}
    account_role = (u.get("account_role") or "member").lower()

    active = current_active_venue()

    # If user has NO venues yet
    if not active:
        st.warning("You don't have access to any venue yet.")
        if account_role in {"owner", "admin", "manager"}:
            st.info("Create your first venue below.")
            manage_organization_ui(venue_role="owner")
        else:
            st.info("Ask an admin to grant you access.")
        return

    # Debug log for resolved page
    logging.debug(f"Resolved page_key after deep-link sync: {st.session_state['page']}")
    
    # Top bar (moved out of auth_gate)
    u = current_user()
    acc = current_account()
    
    with st.container(horizontal=True):


        st.markdown(
            f"**Account:** {acc['name'] if acc else '—'}")
        
        
        venue_id = _venue_selector_compact()

        # Resolve role for the selected venue (needed by some pages like orders_tab)
        _venues = current_venues_for_user() or []
        _role_by_id = {int(v["id"]): role for (v, role) in _venues}
        venue_role = _role_by_id.get(int(venue_id))



        if st.button("Logout", key="logout_btn_app", use_container_width=True):
            clear_auth()
            st.rerun()



    # Render page (no fixed-height container)
    page_key = st.session_state["page"]
    title, loader = PAGES.get(page_key, PAGES["orders"])
    page_fn = loader()

    if page_key == "tracking":
        _call_page(page_fn, venue_id, venue_role=venue_role, deep_order_id=deep_order_id, deep_provider=(deep_provider or None))
    elif page_key == "orders":
        _call_page(page_fn, venue_id, venue_role=venue_role, deep_order_id=deep_order_id, deep_provider=(deep_provider or None), deep_status=(deep_status or None))
    else:
        _call_page(page_fn, venue_id, venue_role=venue_role)

    _bottom_tabbar(page_key)



if __name__ == "__main__":
    main()
