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


# -------------------------------
# 5) Micro-UX CSS (fast taps + less vertical waste)
# -------------------------------
def _css():
    st.markdown(
        """
<style>
/* tighter top padding */
.block-container{padding-top:0.9rem; padding-bottom:5rem; max-width:1100px;}
/* bigger tap targets */
div.stButton > button, div.stDownloadButton > button{
  height:44px; border-radius:14px; font-weight:900;
}
/* make radios look like segmented control */
div[role="radiogroup"] > label{
  padding:0.35rem 0.6rem;
  border-radius:999px;
  border:1px solid rgba(148,163,184,.55);
  margin-right:8px;
}
div[role="radiogroup"]{gap:8px;}
/* subtle cards */
.voi-card{border:1px solid rgba(148,163,184,.35); border-radius:18px; padding:14px 14px; background:#fff;}
.voi-muted{color:#64748b;}
/* mobile: reduce spacing */
@media (max-width: 640px){
  .block-container{padding-left:0.75rem; padding-right:0.75rem;}
}
</style>
""",
        unsafe_allow_html=True,
    )


def _go(page_key: str, **extra_qp: str) -> None:
    """Navigate to a page and sync URL (?page=...)."""
    st.session_state["page"] = page_key
    set_query_params(page=page_key, **{k: str(v) for k, v in extra_qp.items() if v is not None and str(v).strip()})
    st.rerun()


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


def main():
    bootstrap_once()
    _css()

    # Auth gate (no global venue selector; we render our own compact one)
    auth_gate(show_manage_org=True, show_venue_selector=False)
    require_login()

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

    # Sync deep-link params
    deep_page = (qp_str("page", "").strip().lower() or "")
    deep_order_id = qp_int("order_id")
    deep_provider = (qp_str("provider", "").strip() or "")
    deep_status = (qp_str("status", "").strip().lower() or "")

    # Resolve initial page
    if "page" not in st.session_state:
        st.session_state["page"] = deep_page if deep_page in PAGES else "orders"
    elif deep_page in PAGES and deep_page != st.session_state["page"]:
        # URL changed externally
        st.session_state["page"] = deep_page

    # Top: venue selector + quick nav
    top_l, top_r = st.columns([1.2, 2.0], vertical_alignment="center")
    with top_l:
        venue_id = _venue_selector_compact()

    with top_r:
        # Segmented control feel via horizontal radio
        labels = [PAGES[k][0] for k in PAGE_KEYS]
        key_to_label = {k: PAGES[k][0] for k in PAGE_KEYS}
        label_to_key = {v: k for k, v in key_to_label.items()}

        current_label = key_to_label.get(st.session_state["page"], PAGES["orders"][0])
        sel = st.radio(
            "Navigation",
            options=labels,
            index=labels.index(current_label),
            horizontal=True,
            label_visibility="collapsed",
        )
        chosen_key = label_to_key.get(sel, "orders")
        if chosen_key != st.session_state["page"]:
            _go(chosen_key)

    st.divider()

    # Render page
    page_key = st.session_state["page"]
    title, loader = PAGES.get(page_key, PAGES["orders"])
    page_fn = loader()

    # Page call signatures differ; handle deep links only where useful
    if page_key == "tracking":
        page_fn(venue_id, deep_order_id=deep_order_id, deep_provider=(deep_provider or None))
    elif page_key == "orders":
        page_fn(venue_id, deep_order_id=deep_order_id, deep_provider=(deep_provider or None), deep_status=(deep_status or None))
    else:
        page_fn(venue_id)


if __name__ == "__main__":
    main()
