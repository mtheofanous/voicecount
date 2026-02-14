from __future__ import annotations

# Mobile-first, fast navigation shell for Streamlit
# Production-safe changes:
# - Never put auth tokens in the URL
# - Restore auth from session token (optional one-time legacy URL token support)
# - On invalid token, clear session + remove URL token to avoid blank-page loops
# - Remove secret-leaking debug logs

from pathlib import Path
from dotenv import load_dotenv
import warnings
import logging
import inspect

import streamlit as st
from streamlit_float import float_init, float_css_helper
from streamlit.components.v1 import html as components_html

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

st.set_page_config(
    page_title="Voi",
    page_icon="🧾",
    layout="centered",
)

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
)
from core.url_nav import set_query_params

# If you still have legacy links with ?st=..., we consume them once then delete.
LEGACY_URL_TOKEN_SUPPORT = True

warnings.filterwarnings("ignore", message=".*use_container_width.*")
from sqlalchemy.exc import SAWarning
warnings.filterwarnings(
    "ignore",
    category=SAWarning,
    message=r".*already contains a class with the same class name and module name.*",
)


@st.cache_resource(show_spinner=False)
def bootstrap_once() -> bool:
    ensure_google_credentials_file()
    init_db()
    init_auth_db()
    return True

# Custom CSS for the app (mobile-first, clean, modern). Adjust as needed.
def _css() -> None:
    st.markdown(
        """
<style>

/* Hide Streamlit chrome (we render our own fixed topbar) */
[data-testid="stHeader"]{display:none;}
header{display:none;}
[data-testid="stToolbar"]{display:none;}
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
.block-container{
  padding:0.75rem 0.75rem 5.5rem;
  max-width:100%;
}

button{
  min-height:44px;
  border-radius:14px;
  font-weight:900;
}

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
  .voi-tab .tx{font-size:.82rem;}
}
</style>
""",
        unsafe_allow_html=True,
    )



def _ui_float_init() -> None:
    """Init floating UI + glass styles (safe to call on every rerun)."""
    float_init()
    st.markdown(
        """
<style>
/* Hide Streamlit chrome (we render our own bars) */
[data-testid="stHeader"]{display:none;}
header{display:none;}
[data-testid="stToolbar"]{display:none;}
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}

/* Make room for fixed bars */
.block-container{padding-top:.75rem;padding-bottom:6.5rem;}

/* Modern button baseline */
.stButton>button{
  border-radius: 14px;
  height: 44px;
  font-weight: 650;
  border: 1px solid rgba(148,163,184,.28);
}

/* Primary buttons a bit punchier */
.stButton>button[kind="primary"]{
  border: 1px solid rgba(59,130,246,.35);
}

/* Reduce column gap on mobile */
@media (max-width: 520px){
  .block-container{padding-left:.6rem;padding-right:.6rem;}
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _go(page_key: str, **extra_qp: str) -> None:
    """Navigate to a page and sync URL (?page=...). Never add auth tokens to URL."""
    st.session_state["page"] = page_key
    params = {"page": page_key}
    params.update({k: str(v) for k, v in extra_qp.items() if v is not None and str(v).strip()})
    set_query_params(**params)
    st.rerun()


def _bottom_tabbar(current_page: str) -> None:
    """Fixed glass bottom bar (mobile-first, session-safe)."""
    bar = st.container(horizontal=True,gap="small")
    with bar:
        # c1, c2, c3 = st.columns(3, gap="small")

        if st.button(
            "➕",
            use_container_width=True,
            type=("primary" if current_page == "new" else "secondary"),
            key="bb_new",
        ):
            _go("new")

        if st.button(
            "📦",
            use_container_width=True,
            type=("primary" if current_page == "orders" else "secondary"),
            key="bb_orders",
        ):
            _go("orders")

        if st.button(
            "✅",
            use_container_width=True,
            type=("primary" if current_page == "tracking" else "secondary"),
            key="bb_tracking",
        ):
            _go("tracking")

    css = float_css_helper(
        left="0",
        right="0",
        bottom="0",
        width="100%",
        background="rgba(255,255,255,.72)",
        z_index="9998",
    )
    css += (
        "padding: 10px 12px;"
        "border-top: 1px solid rgba(148,163,184,.22);"
        "backdrop-filter: blur(14px) saturate(180%);"
        "-webkit-backdrop-filter: blur(14px) saturate(180%);"
        "box-shadow: 0 -10px 30px rgba(15,23,42,.06);"
    )
    bar.float(css)

    st.markdown("<div style='height:98px'></div>", unsafe_allow_html=True)

def _handle_actions_from_query_params() -> None:
    action = (st.query_params.get("action", "") or "").strip().lower()
    if action == "logout":
        clear_auth()
        try:
            st.query_params.clear()
        except Exception:
            set_query_params()
        st.rerun()


def _inject_topbar(
    account_name: str,
    venue_name: str,
    show_manage_org: bool,
    is_manage_page: bool,
    is_catalog_page: bool,
    is_history_page: bool,
) -> None:
    """Fixed glass topbar (mobile-first, session-safe)."""
    bar = st.container()
    with bar:
        left, mid, right = st.columns([2.4, 3.6, 1.0], gap="small")

        with left:
            st.markdown(
                f"""
                <div style="line-height:1.05">
                  <div style="font-weight:800;font-size:.95rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                    {account_name}
                  </div>
                  <div style="opacity:.72;font-size:.80rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                    {venue_name}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with mid:
            n = 3 if show_manage_org else 2
            cols = st.columns(n, gap="small")
            i = 0

            if show_manage_org:
                if cols[i].button(
                    "⚙️ Manage",
                    use_container_width=True,
                    type=("primary" if is_manage_page else "secondary"),
                    key="top_manage_btn",
                ):
                    _go("manage_org")
                i += 1

            if cols[i].button(
                "📒 Catalog",
                use_container_width=True,
                type=("primary" if is_catalog_page else "secondary"),
                key="top_catalog_btn",
            ):
                _go("catalog")
            i += 1

            if cols[i].button(
                "🕘 History",
                use_container_width=True,
                type=("primary" if is_history_page else "secondary"),
                key="top_history_btn",
            ):
                _go("history")

        with right:
            if st.button("🚪", use_container_width=True, type="secondary", key="top_logout_btn"):
                clear_auth()
                try:
                    st.query_params.clear()
                except Exception:
                    set_query_params()
                st.rerun()

    css = float_css_helper(
        left="0",
        right="0",
        top="0",
        width="100%",
        background="rgba(255,255,255,.72)",
        z_index="9999",
    )
    css += (
        "padding: 10px 12px;"
        "border-bottom: 1px solid rgba(148,163,184,.22);"
        "backdrop-filter: blur(14px) saturate(180%);"
        "-webkit-backdrop-filter: blur(14px) saturate(180%);"
        "box-shadow: 0 8px 28px rgba(15,23,42,.06);"
    )
    bar.float(css)

    st.markdown("<div style='height:78px'></div>", unsafe_allow_html=True)

def _page_catalog():
    from features.catalog import catalog_tab
    return catalog_tab


def _page_new_order():
    from features.create_order.new_order_tab import new_order_tab
    return new_order_tab


def _page_orders():
    from features.manage_orders.orders import orders_tab
    return orders_tab


def _page_manage_org():
    from features.auth_and_manage.auth_multi_tenant import manage_organization_ui
    return manage_organization_ui


def _page_tracking():
    from features.manage_orders.receive_orders import tracking_dashboard
    return tracking_dashboard


def _page_history():
    from features.manage_orders.history_reports import history_reports_page
    return history_reports_page


PAGES = {
    "new": ("➕", _page_new_order),
    "orders": ("📦", _page_orders),
    "tracking": ("✅", _page_tracking),
    "history": ("📈", _page_history),
    "catalog": ("🧾", _page_catalog),
    "manage_org": ("⚙️", _page_manage_org),
}


def _call_page(fn, venue_id: int, **kwargs):
    """Call a page function passing only accepted args (fast + robust)."""
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
    except (TypeError, ValueError):
        try:
            return fn(venue_id)
        except TypeError:
            return fn()

    call_kwargs = {}
    if "venue_id" in params:
        call_kwargs["venue_id"] = venue_id

    for k, v in kwargs.items():
        if k in params:
            call_kwargs[k] = v

    return fn(**call_kwargs)


def _restore_auth_from_token() -> None:
    """Restore auth_ctx from a persisted session token. Never keep tokens in URL."""
    token = (st.session_state.get("_session_token") or "").strip()

    if not token and LEGACY_URL_TOKEN_SUPPORT:
        try:
            url_token = (st.query_params.get("st") or "").strip()
        except Exception:
            url_token = ""
        if url_token:
            token = url_token
            st.session_state["_session_token"] = token
            try:
                del st.query_params["st"]
            except Exception:
                pass

    if not token:
        return

    try:
        from features.auth_and_manage.auth_multi_tenant import _get_session_from_token
        session_data = _get_session_from_token(token)
    except Exception as e:
        logging.error(f"❌ Error validating token: {e}")
        session_data = None

    if session_data:
        st.session_state["auth_ctx"] = {
            "user_id": session_data["user_id"],
            "account_id": session_data["account_id"],
        }
        return

    logging.warning("⚠️ Invalid token - clearing session")
    st.session_state.pop("_session_token", None)
    st.session_state.pop("auth_ctx", None)
    try:
        if "st" in st.query_params:
            del st.query_params["st"]
    except Exception:
        pass
    st.rerun()


def main():
    bootstrap_once()
    _css()

    # Floating glass bars (top/bottom)
    _ui_float_init()

    # Enable floating containers (top/bottom bars)
    float_init()

    _handle_actions_from_query_params()
    _restore_auth_from_token()

    try:
        url_page = (st.query_params.get("page") or "").strip().lower()
    except Exception:
        url_page = ""

    if url_page and url_page in PAGES:
        st.session_state["page"] = url_page
    elif "page" not in st.session_state:
        st.session_state["page"] = "orders"

    try:
        deep_order_id = int(st.query_params.get("order_id", 0)) or None
    except Exception:
        deep_order_id = None

    try:
        deep_provider = (st.query_params.get("provider") or "").strip() or None
    except Exception:
        deep_provider = None

    try:
        deep_status = (st.query_params.get("status") or "").strip().lower() or None
    except Exception:
        deep_status = None

    auth_gate(show_manage_org=True, show_venue_selector=False)
    require_login()

    u = current_user() or {}
    account_role = (u.get("account_role") or "member").lower()
    acc = current_account()

    active = current_active_venue()
    if not active:
        st.warning("You don't have access to any venue yet.")
        if account_role in {"owner", "admin", "manager"}:
            st.info("Create your first venue in Manage Org.")
        else:
            st.info("Ask an admin to grant you access.")
        return

    venue, venue_role = active
    venue_id = int(venue["id"])
    venue_name = venue.get("name", "—")

    account_name = acc["name"] if acc else "—"
    page_key = st.session_state.get("page", "orders")
    show_manage_org = account_role in {"owner", "admin", "manager"}

    _inject_topbar(
        account_name=account_name,
        venue_name=venue_name,
        show_manage_org=show_manage_org,
        is_manage_page=(page_key == "manage_org"),
        is_catalog_page=(page_key == "catalog"),
        is_history_page=(page_key == "history"),
    )

    _, loader = PAGES.get(page_key, PAGES["orders"])
    page_fn = loader()

    if page_key == "tracking":
        _call_page(
            page_fn,
            venue_id,
            venue_role=venue_role,
            deep_order_id=deep_order_id,
            deep_provider=deep_provider,
        )
    elif page_key == "orders":
        _call_page(
            page_fn,
            venue_id,
            venue_role=venue_role,
            deep_order_id=deep_order_id,
            deep_provider=deep_provider,
            deep_status=deep_status,
        )
    elif page_key == "history":
        _call_page(page_fn, venue_id, venue_role=venue_role, deep_provider=deep_provider)
    else:
        _call_page(page_fn, venue_id, venue_role=venue_role)

    _bottom_tabbar(page_key)


if __name__ == "__main__":
    main()
