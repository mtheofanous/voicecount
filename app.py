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
from features.create_order import new_order_tab
from features.catalog import catalog_tab
import os
from dotenv import load_dotenv
import warnings
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

def main():
    
        
    init_db()
    init_auth_db()

    # Login UI
    auth_gate(show_manage_org=True)
    require_login()

    u = current_user() or {}
    account_role = (u.get("account_role") or "member").lower()

    active = current_active_venue()

    # ✅ CASE 1: user has NO venues yet
    if not active:
        st.warning("You don't have access to any venue yet.")

        if account_role in {"owner", "admin", "manager"}:
            st.info("Create your first venue below.")
            # ✅ Let them access org management even without an active venue
            manage_organization_ui(venue_role="owner")  # treat account admins as owner here
        else:
            st.info("Ask an admin to grant you access.")
        st.stop()

    # ✅ CASE 2: user has an active venue -> normal app
    venue, venue_role = active
    venue_id = venue["id"]

    if venue_role in {"owner", "manager"}:
        main_tabs = ["🏢 Manage organization", "🆕 Nuevo pedido", "📜 Pedidos"]
    else:
        main_tabs = ["🆕 Nuevo pedido", "📜 Pedidos"]

    # Keep selection across reruns (st.tabs always returns to first tab)
    st.session_state.setdefault("main_tab", main_tabs[0])
    if st.session_state["main_tab"] not in main_tabs:
        st.session_state["main_tab"] = main_tabs[0]


    selected = st.radio(
        "Navigation",  # Add a non-empty label
        main_tabs,
        horizontal=True,
        key="main_tab",
        label_visibility="collapsed",  # This hides it visually but keeps it accessible
    )

    if selected == "🏢 Manage organization":
        manage_organization_ui(venue_role=venue_role)
    elif selected == "🆕 Nuevo pedido":
        new_order_tab(venue_id, venue_role)
    elif selected == "📜 Pedidos":
        orders_tab(venue_id, venue_role)

main()

