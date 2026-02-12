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
from fixed_container import st_fixed_container
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


/* Make the very first block (your top bar) sticky */
section.main .block-container > div:first-child {
    position: sticky;
    top: 0;
    z-index: 9999;

    background: rgba(255,255,255,.96);
    backdrop-filter: saturate(180%) blur(10px);
    border-bottom: 1px solid rgba(148,163,184,.35);

    padding: 10px 14px;
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
        ("history", "📈", "History"),
        ("catalog", "🧾", "Catalog"),
        ("manage_org", "⚙️", "Manage Org"),
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

import streamlit as st
from streamlit.components.v1 import html as components_html


def _handle_actions_from_query_params():
    """Logout triggered from topbar link (?action=logout)."""
    action = (st.query_params.get("action", "") or "").strip().lower()
    if action == "logout":
        clear_auth()
        st.query_params.clear()
        st.rerun()


def _inject_topbar(account_name: str):
    """Injects a true fixed top bar into parent DOM (won't scroll away)."""
    token = st.session_state.get("_session_token", "")
    token_param = f"&st={token}" if token else ""
    logout_href = f"?action=logout{token_param}"

    # We inject CSS + HTML into parent.document, and also pad the main content
    # so your page doesn't hide under the fixed top bar + sticky selector.
    components_html(
        f"""
<script>
(function() {{
  const doc = parent.document;

  const BAR_H = 52;      // top bar height
  const DOCK_H = 58;     // venue selector "dock" height (approx)
  const PAD_TOP = BAR_H + DOCK_H;

  // ---------- CSS (once) ----------
  if (!doc.getElementById("voi-topbar-style")) {{
    const style = doc.createElement("style");
    style.id = "voi-topbar-style";
    style.textContent = `
      #voi-topbar-root {{
        position: fixed;
        top: 0; left: 0; right: 0;
        z-index: 999999;
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(0,0,0,0.10);
        padding: 10px 14px;
      }}
      #voi-topbar-inner {{
        max-width: 1100px;
        margin: 0 auto;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
      }}
      #voi-topbar-acc {{
        font-weight: 700;
        font-size: 14px;
        color: rgba(0,0,0,0.85);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }}
      #voi-topbar-logout {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 8px 10px;
        border-radius: 10px;
        border: 1px solid rgba(0,0,0,0.12);
        text-decoration: none;
        color: rgba(0,0,0,0.85);
        font-size: 13px;
        font-weight: 600;
      }}
      #voi-topbar-logout:active {{ transform: translateY(1px); }}
    `;
    doc.head.appendChild(style);
  }}

  // ---------- Root element ----------
  let root = doc.getElementById("voi-topbar-root");
  if (!root) {{
    root = doc.createElement("div");
    root.id = "voi-topbar-root";
    doc.body.appendChild(root);
  }}

  // ---------- Render ----------
  root.innerHTML = `
    <div id="voi-topbar-inner">
      <div id="voi-topbar-acc">Account: {account_name or "—"}</div>
      <a id="voi-topbar-logout" href="{logout_href}" target="_self">Logout</a>
    </div>
  `;

  // ---------- Pad main content so it doesn't go under bar+selector ----------
  const app = doc.querySelector('[data-testid="stAppViewContainer"]');
  if (app) {{
    app.style.paddingTop = PAD_TOP + "px";
  }}
}})();
</script>
        """,
        height=0,
        scrolling=False,
    )


def _make_venue_selector_sticky():
    """
    After the marker is rendered, this finds its Streamlit wrapper in parent DOM
    and turns that wrapper into a sticky "dock" under the fixed top bar.
    """
    components_html(
        """
<script>
(function() {
  const doc = parent.document;
  const marker = doc.getElementById("voi-venue-marker");
  if (!marker) return;

  // Find closest wrapper Streamlit uses for blocks
  let wrapper = marker;
  while (wrapper && wrapper !== doc.body) {
    if (wrapper.getAttribute && wrapper.getAttribute("data-testid") === "stVerticalBlockBorderWrapper") break;
    wrapper = wrapper.parentElement;
  }
  if (!wrapper || wrapper === doc.body) return;

  // Apply sticky styles
  wrapper.style.position = "sticky";
  wrapper.style.top = "52px";            // match BAR_H above
  wrapper.style.zIndex = "999998";
  wrapper.style.background = "rgba(255,255,255,0.95)";
  wrapper.style.backdropFilter = "blur(10px)";
  wrapper.style.borderBottom = "1px solid rgba(0,0,0,0.10)";
  wrapper.style.padding = "10px 14px 10px 14px";

  // Marker itself can be removed/hidden
  marker.style.display = "none";
})();
</script>
        """,
        height=0,
        scrolling=False,
    )



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

def _page_manage_org():
    from features.auth_and_manage import manage_organization_ui
    return manage_organization_ui

def _page_tracking():
    # wrapper lazily imports heavy implementation
    from features.manage_orders.receive_orders import tracking_dashboard
    return tracking_dashboard

def _page_history():
    from features.manage_orders.history_reports import history_reports_page
    return history_reports_page


PAGES = {
    "new": ("➕ New order", _page_new_order),
    "orders": ("📦 Orders", _page_orders),
    "tracking": ("✅ Receive / Tracking", _page_tracking),
    "history": ("📈  History", _page_history),
    "catalog": ("🧾 Catalog", _page_catalog),
    "manage_org": ("⚙️ Manage Org", _page_manage_org),
}

# Preferred order for the segmented control
PAGE_KEYS = ["new", "orders", "tracking", "history", "catalog", "manage_org"]


def _home_card():
    st.markdown("<div class='voi-card'>", unsafe_allow_html=True)
    st.title("Voi")
    st.markdown("<div class='voi-muted'>Fast, mobile-first navigation.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)



def _call_page(fn, venue_id: int, **kwargs):
    """
    Call a page function, passing only the arguments it actually accepts.
    Supports pages WITH or WITHOUT venue_id.
    """
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        # Fallback: try venue_id first, then no args
        try:
            return fn(venue_id)
        except TypeError:
            return fn()

    params = sig.parameters

    call_kwargs = {}

    # Pass venue_id only if accepted
    if "venue_id" in params:
        call_kwargs["venue_id"] = venue_id

    # Pass other kwargs only if accepted
    for k, v in kwargs.items():
        if k in params:
            call_kwargs[k] = v

    return fn(**call_kwargs)



def main():
    bootstrap_once()
    _css()

    # Handle logout / actions early
    _handle_actions_from_query_params()

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

    logging.debug(f"Resolved page_key after deep-link sync: {st.session_state['page']}")

    u = current_user()
    acc = current_account()

    # ---------------- TOP BAR (true fixed) ----------------
    account_name = acc["name"] if acc else "—"
    _inject_topbar(account_name)

    # ---------------- Sticky venue selector dock ----------------
    # Marker must be rendered in Streamlit layout first:
    st.markdown('<div id="voi-venue-marker"></div>', unsafe_allow_html=True)
    venue_id = _venue_selector_compact()
    # Then we "upgrade" the wrapper of that block to sticky via JS:
    _make_venue_selector_sticky()

    _venues = current_venues_for_user() or []
    _role_by_id = {int(v["id"]): role for (v, role) in _venues}
    venue_role = _role_by_id.get(int(venue_id))

    # ---------------- Render page ----------------
    page_key = st.session_state["page"]
    title, loader = PAGES.get(page_key, PAGES["orders"])
    page_fn = loader()

    if page_key == "tracking":
        _call_page(page_fn, venue_id, venue_role=venue_role, deep_order_id=deep_order_id, deep_provider=(deep_provider or None))
    elif page_key == "orders":
        _call_page(page_fn, venue_id, venue_role=venue_role, deep_order_id=deep_order_id, deep_provider=(deep_provider or None), deep_status=(deep_status or None))
    elif page_key == "history":
        _call_page(page_fn, venue_id, venue_role=venue_role, deep_provider=deep_provider)
    else:
        _call_page(page_fn, venue_id, venue_role=venue_role)

    _bottom_tabbar(page_key)


if __name__ == "__main__":
    main()
