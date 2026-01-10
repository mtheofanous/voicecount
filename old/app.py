"""
app.py — Streamlit entrypoint (clean router)

This file is intentionally small so your Streamlit Cloud deployment stays simple.
For now it *reuses* your existing modules (vint.py, orders_modern_ui_v12.py, auth_multi_tenant.py).

As you refactor, you’ll progressively move logic out of vint.py into /app/* packages,
but app.py can stay basically the same.
"""

import streamlit as st

# Auth / org mgmt
from features.auth_and_manage.auth_multi_tenant import (
    init_auth_db,
    auth_gate,
    require_login,
    current_active_venue,
    manage_organization_ui,
)

# Core app features (currently still living in vint.py)
from vint import (
    init_db,
    ensure_sqlite_columns,   # local sqlite migrations (safe to keep)
    catalog_tab,
    new_order_tab,
    engine,
    get_session,
    Product,
    distinct_units,
    safe_str,
    num_or_default,
    coalesce_unit,
)

# Orders UI (modern)
from features.manage_orders.orders import orders_tab_modern


st.set_page_config(page_title="Voi", page_icon="🧾", layout="wide")


def orders_tab(venue_id: int, role: str | None) -> None:
    """
    Small adapter so the rest of the app doesn’t need to pass a lot of dependencies around.
    Later, when you move parsing/utils/models into dedicated modules, you’ll update the imports here.
    """
    orders_tab_modern(
        venue_id=venue_id,
        role=role,
        engine=engine,
        get_session=get_session,
        Product=Product,
        distinct_units=distinct_units,
        safe_str=safe_str,
        num_or_default=num_or_default,
        coalesce_unit=coalesce_unit,
    )


def main() -> None:
    # --- DB init (today: sqlite; later: Supabase/Postgres) ---
    init_db()
    # Keep this for local dev with SQLite. When you switch to Supabase, you’ll likely remove it
    # (or replace with Alembic migrations).
    ensure_sqlite_columns()

    # --- Auth DB init ---
    init_auth_db()

    # --- Login / org gating ---
    auth_gate(show_manage_org=False)
    require_login()

    active = current_active_venue()
    if not active:
        st.warning("No venue access yet. Ask an admin to grant you access.")
        st.stop()

    venue, venue_role = active
    venue_id = venue["id"]

    # --- Top-level navigation ---
    if venue_role in {"owner", "manager"}:
        tabs = st.tabs(["🏢 Manage organization", "📦 Catálogo", "🆕 Nuevo pedido", "📜 Pedidos"])
        with tabs[0]:
            manage_organization_ui(venue_role=venue_role)
        with tabs[1]:
            catalog_tab(venue_id, venue_role)
        with tabs[2]:
            new_order_tab(venue_id, venue_role)
        with tabs[3]:
            orders_tab(venue_id, venue_role)
    else:
        tabs = st.tabs(["🆕 Nuevo pedido", "📜 Pedidos"])
        with tabs[0]:
            new_order_tab(venue_id, venue_role)
        with tabs[1]:
            orders_tab(venue_id, venue_role)


if __name__ == "__main__":
    main()
