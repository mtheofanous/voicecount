"""
app.py — Streamlit entrypoint (clean router)
"""

import streamlit as st
st.set_page_config(page_title="Voi", page_icon="🧾", layout="centered")

from core.init import init_db
from pathlib import Path
from core.db import engine, get_session
from domain.models import Product
from features.manage_orders.orders import orders_tab
from features.manage_orders.receive_orders import tracking_dashboard
from features.create_order import new_order_tab
from features.catalog import catalog_tab
import os
from dotenv import load_dotenv
import warnings
from features.auth_and_manage.auth_multi_tenant import current_venues_for_user  
from core.url_nav import read_page_from_url, write_page_to_url  # ✅ NEW

warnings.filterwarnings("ignore", message=".*use_container_width.*")
warnings.filterwarnings("ignore", message=".*label.*got an empty value.*")

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

from features.auth_and_manage.auth_multi_tenant import (
    init_auth_db,
    auth_gate,
    require_login,
    current_user,
    current_active_venue,
    manage_organization_ui,
)


def _go(page_key: str) -> None:
    """Navigate to a page and sync URL."""
    st.session_state["page"] = page_key
    write_page_to_url(page_key)
    st.rerun()

def venue_selector_home_only() -> None:
    venues = current_venues_for_user()
    if not venues:
        st.warning("You don't have access to any venue yet.")
        st.stop()

    labels, ids = [], []
    for v, role in venues:
        labels.append(f"{v['name']} — {role}")
        ids.append(v["id"])

    st.session_state.setdefault("active_venue_id", ids[0])
    if st.session_state["active_venue_id"] not in ids:
        st.session_state["active_venue_id"] = ids[0]

    idx = ids.index(st.session_state["active_venue_id"])
    chosen_label = st.selectbox("Active bar/restaurant", options=labels, index=idx)

    chosen_id = ids[labels.index(chosen_label)]
    if chosen_id != st.session_state["active_venue_id"]:
        st.session_state["active_venue_id"] = chosen_id
        st.rerun()


def home_page(pages: dict[str, str]) -> None:


    venue_selector_home_only() 

    
    # Simple grid of big buttons
    cols = st.columns(2)
    i = 0
    for key, label in pages.items():
        if key == "home":
            continue
        with cols[i % 2]:
            if st.button(label, use_container_width=True):
                _go(key)
        i += 1


def main():
    init_db()
    init_auth_db()

    # Login UI
    auth_gate(show_manage_org=True, show_venue_selector=False)

    require_login()

    u = current_user() or {}
    account_role = (u.get("account_role") or "member").lower()

    active = current_active_venue()

    # ✅ CASE 1: user has NO venues yet
    if not active:
        st.warning("You don't have access to any venue yet.")

        if account_role in {"owner", "admin", "manager"}:
            st.info("Create your first venue below.")
            manage_organization_ui(venue_role="owner")  # treat account admins as owner here
        else:
            st.info("Ask an admin to grant you access.")
        st.stop()

    # ✅ CASE 2: user has an active venue -> normal app
    venue, venue_role = active
    venue_id = venue["id"]

    # ---- Pages by role ----
    if venue_role in {"owner", "manager"}:
        PAGES = {
            "home": "🏠 Home",
            "manage_org": "🏢 Manage organization",
            "catalog": "📦 Catálogo",
            "new_order": "🆕 Nuevo pedido",
            "orders": "📜 Pedidos",
            "tracking": "📍 Track order",
        }
    else:
        PAGES = {
            "home": "🏠 Home",
            "new_order": "🆕 Nuevo pedido",
            "orders": "📜 Pedidos",
        }

    allowed = set(PAGES.keys())

    # ---- URL -> Session sync (first run / refresh / shared links) ----
    st.session_state.setdefault("page", "home")
    page_from_url = read_page_from_url(allowed_pages=allowed, default_page="home")
    if st.session_state["page"] != page_from_url:
        st.session_state["page"] = page_from_url

    # If URL had an invalid page, normalize it (optional but nice)
    if page_from_url not in allowed:
        write_page_to_url("home")

    page = st.session_state["page"]

    # ---- Back to home button (except on home) ----
    if page != "home":
        top_left, top_right = st.columns([3, 3], vertical_alignment="center")
        with top_left:
            if st.button("⬅️ Home"):
                _go("home")


    # ---- ROUTER ----
    if page == "home":
        home_page(PAGES)

    elif page == "manage_org":
        manage_organization_ui(venue_role=venue_role)

    elif page == "catalog":
        catalog_tab(venue_id=venue_id, venue_role=venue_role)

    elif page == "new_order":
        new_order_tab(venue_id, venue_role)

    elif page == "orders":
        orders_tab(venue_id, venue_role)

    elif page == "tracking":
        tracking_dashboard(venue_id)

    else:
        # Safety fallback
        _go("home")


main()


