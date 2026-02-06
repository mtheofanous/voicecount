# -------------------------------
# 1) Load ENV FIRST (before any app / DB imports)
# -------------------------------
from pathlib import Path
from dotenv import load_dotenv
import warnings

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)
# import os
# import streamlit as st
# from core.config import get_database_url

# print("ENV DATABASE_URL:", os.getenv("DATABASE_URL"))
# print("SECRETS has DATABASE_URL:", ("DATABASE_URL" in st.secrets))
# if "DATABASE_URL" in st.secrets:
#     print("SECRETS DATABASE_URL:", st.secrets["DATABASE_URL"])

# print("get_database_url():", get_database_url())




# from core.config import get_database_url
# print("FULL DATABASE_URL:", get_database_url())
# # Optional but VERY useful while debugging
# import os
# print("DATABASE_URL host:",
#       os.getenv("DATABASE_URL", "NOT SET").split("@")[-1].split("/")[0])

# -------------------------------
# 2) Streamlit config
# -------------------------------
import streamlit as st

st.set_page_config(
    page_title="Voi",
    page_icon="🧾",
    layout="centered",
)



# -------------------------------
# 3) Now it is SAFE to import DB + app modules
# -------------------------------
from core.config import ensure_google_credentials_file
from core.init import init_db

from features.manage_orders.orders import orders_tab
from features.manage_orders.receive_orders import *
from features.create_order import new_order_tab
from features.catalog import catalog_tab
from features.manage_orders.reports import reports_page
from features.manage_orders.history import _render_history_tab

from features.auth_and_manage.auth_multi_tenant import (
    init_auth_db,
    auth_gate,
    require_login,
    current_user,
    current_active_venue,
    manage_organization_ui,
    current_venues_for_user,
)

# URL helpers
from core.url_nav import qp_int, qp_str, set_query_params

# -------------------------------
# 4) Warnings (cosmetic)
# -------------------------------
warnings.filterwarnings("ignore", message=".*use_container_width.*")
warnings.filterwarnings("ignore", message=".*label.*got an empty value.*")



def _go(page_key: str) -> None:
    """Navigate to a page and sync URL (?page=...)."""
    st.session_state["page"] = page_key
    set_query_params(page=page_key)
    st.rerun()


def venue_selector_home_only() -> None:
    """Show the active venue selector ONLY on the home (navigation) page."""
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
    st.title("Welcome 👋")
    st.caption("Choose what you want to do.")

    # ✅ Venue selector ONLY here
    venue_selector_home_only()

    st.divider()

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
    ensure_google_credentials_file()
    init_db()
    init_auth_db()

    # Login UI (selector hidden globally; shown only on Home)
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

    # -----------------------------
    # MVP deep-link params
    # -----------------------------
    deep_page = (qp_str("page", "").strip().lower() or "")
    deep_order_id = qp_int("order_id")
    deep_status = (qp_str("status", "").strip().lower() or None)
    deep_provider = (qp_str("provider", "").strip() or None)

    # ---- Pages by role ----
    if venue_role in {"owner", "manager"}:
        PAGES = {
            
                "home": "🏠 Inicio",
                "manage_org": "🏢 Organización",
                "catalog": "📦 Catálogo",
                "new_order": "📝 Notas",
                "orders": "📜 Pedidos",
                "tracking": "📊 Dashboard",
                "history":"📚 History",
                "reports": "Reports"
            }
    else:
        PAGES = {
            "home": "🏠 Inicio",
            "new_order": "📝 Notas",
            "orders": "📜 Pedidos",
            "tracking": "📊 Dashboard",
        }

    allowed = set(PAGES.keys())

    # ---- URL -> Session sync ----
    st.session_state.setdefault("page", "home")

    if deep_page in allowed and st.session_state["page"] != deep_page:
        st.session_state["page"] = deep_page

    # Normalize invalid/missing ?page to home (keeps URL clean)
    if deep_page not in allowed:
        set_query_params(page="home")

    page = st.session_state["page"]

    # ---- Back to home button (except on home) ----
    if page != "home":
        top_left, top_right = st.columns([1, 3], vertical_alignment="center")
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
        orders_tab(
            venue_id,
            venue_role,
            deep_order_id=deep_order_id,
            deep_status=deep_status,
        )

    elif page == "tracking":
        tracking_dashboard(
            venue_id,
            deep_order_id=deep_order_id,
            deep_provider=deep_provider,
        )
    

    elif page == "history":
        _render_history_tab(int(venue_id), deep_provider=deep_provider)
        

        
    elif page =="reports":
        reports_page(venue_id=int(venue_id), venue_role=venue_role)

    else:
        _go("home")


main()
