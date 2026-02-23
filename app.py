# Mobile-first, fast navigation shell for Streamlit Cloud
# - avoids st.tabs (heavy DOM)
# - lazy-loads page modules only when needed
# - caches bootstrapping + DB engines
# - URL-sync via ?page=... for deep links
#
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
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
    page_icon="ðŸ§¾",
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
from core.i18n import t, get_lang, set_lang

warnings.filterwarnings("ignore", message=".*use_container_width.*")
warnings.filterwarnings("ignore", message=".*label.*got an empty value.*")
from sqlalchemy.exc import SAWarning
warnings.filterwarnings(
    "ignore",
    category=SAWarning,
    message=r".*already contains a class with the same class name and module name.*",
)

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
  cursor:pointer;
  user-select:none;
  -webkit-tap-highlight-color:transparent;
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
    

def _nav_triggers_early() -> None:
    """Handle bottom-tab navigation *early* to avoid rendering the previous page.

    The bottom bar uses JS to click hidden Streamlit buttons (keys: _tabbar_<page>).
    Those buttons must exist in the DOM; we render them near the top of the script,
    and if any is clicked we update URL+state and stop immediately.
    """
    tabs = ["new", "borrador", "orders", "tracking"]

    for key in tabs:
        if st.button("_", key=f"_tabbar_{key}"):
            _go(key, stop=True)

    # Keep these triggers offscreen and non-interactive (JS will still click them)
    st.markdown(
        """
        <style>
        [class*="st-key-_tabbar_"] {
            position: fixed !important;
            left: -9999px !important;
            height: 0 !important;
            overflow: hidden !important;
            pointer-events: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def _bottom_tabbar(current_page: str) -> None:
    """Mobile bottom navigation.

    Option B (redirect): Use real links (target=_self) so a tap navigates directly
    to the new URL (?page=...), instead of clicking hidden Streamlit buttons that
    require an extra rerun. This removes the 'double-run' feeling on page switch.
    """
    tabs = [
        ("new",      "➕", t("nav.new")),
        ("borrador", "📝", t("nav.draft")),
        ("orders",   "📦", t("nav.orders")),
        ("tracking", "✅", t("nav.receive")),
    ]

    token = st.session_state.get("_session_token", "")
    token_param = f"&st={token}" if token else ""

    items = []
    for key, icon, label in tabs:
        active = "active" if key == current_page else ""
        href = f"?page={key}{token_param}"
        items.append(
            f'<a class="voi-tab {active}" data-page="{key}" href="{href}" target="_self">'
            f'<div class="ic">{icon}</div>'
            f'<div class="tx">{label}</div>'
            f"</a>"
        )

    html = f"""<div class="voi-tabbar"><div class="voi-tabs">{''.join(items)}</div></div>"""
    st.markdown(html, unsafe_allow_html=True)


from streamlit.components.v1 import html as components_html


def _handle_actions_from_query_params():
    """Logout triggered from topbar link (?action=logout)."""
    action = (st.query_params.get("action", "") or "").strip().lower()
    if action == "logout":
        clear_auth()
        st.query_params.clear()
        st.rerun()


def _inject_topbar(
    account_name: str,
    venue_name: str,
    show_manage_org: bool,
    is_manage_page: bool,
    is_catalog_page: bool,
    is_history_page: bool,
):
    token = st.session_state.get("_session_token", "")
    token_param = f"&st={token}" if token else ""

    logout_href = f"?action=logout{token_param}"
    manage_href = f"?page=manage_org{token_param}"
    catalog_href = f"?page=catalog{token_param}"
    history_href = f"?page=history{token_param}"

    manage_active = "active" if is_manage_page else ""
    catalog_active = "active" if is_catalog_page else ""
    history_active = "active" if is_history_page else ""

    # Language switcher hrefs (preserve current page + token)
    current_lang = get_lang()
    current_page = st.session_state.get("page", "orders")
    page_param = f"page={current_page}"
    lang_el_href = f"?{page_param}&lang=el{token_param}"
    lang_es_href = f"?{page_param}&lang=es{token_param}"
    lang_en_href = f"?{page_param}&lang=en{token_param}"
    lang_el_active = "voi-lang-active" if current_lang == "el" else ""
    lang_es_active = "voi-lang-active" if current_lang == "es" else ""
    lang_en_active = "voi-lang-active" if current_lang == "en" else ""

    manage_icon_html = (
        f'<a class="voi-topbar-icon {manage_active}" href="{manage_href}" target="_self" title="{t("nav.manage_org")}">⚙️</a>'
        if show_manage_org else ""
    )
    catalog_icon_html = f'<a class="voi-topbar-icon {catalog_active}" href="{catalog_href}" target="_self" title="{t("nav.catalog")}">🧾</a>'
    history_icon_html = f'<a class="voi-topbar-icon {history_active}" href="{history_href}" target="_self" title="{t("nav.history")}">📈</a>'

    lang_switcher_html = (
        f'<div id="voi-lang-switcher">'
        f'<a class="voi-lang-btn {lang_el_active}" href="{lang_el_href}" target="_self">ΕΛ</a>'
        f'<a class="voi-lang-btn {lang_es_active}" href="{lang_es_href}" target="_self">ES</a>'
        f'<a class="voi-lang-btn {lang_en_active}" href="{lang_en_href}" target="_self">EN</a>'
        f'</div>'
    )

    # show venue next to account
    acc_venue = f"{account_name or '-'} — {venue_name or '-'}"

    components_html(
        f"""
<script>
(function() {{
  const doc = parent.document;

  const BAR_H = 52;
  const PAD_TOP = BAR_H;   // no sticky venue dock anymore

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
      #voi-topbar-right {{
        display: flex;
        align-items: center;
        gap: 8px;
      }}
      .voi-topbar-icon {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 10px;
        border: 1px solid rgba(0,0,0,0.12);
        text-decoration: none;
        font-size: 18px;
        color: rgba(0,0,0,0.85);
      }}
      .voi-topbar-icon.active {{
        background: rgba(0,0,0,0.08);
      }}
      #voi-topbar-logout {{
        padding: 8px 10px;
        border-radius: 10px;
        border: 1px solid rgba(0,0,0,0.12);
        text-decoration: none;
        color: rgba(0,0,0,0.85);
        font-size: 13px;
        font-weight: 600;
      }}
      #voi-lang-switcher {{
        display: flex;
        align-items: center;
        gap: 2px;
        border: 1px solid rgba(0,0,0,0.12);
        border-radius: 10px;
        overflow: hidden;
        padding: 2px;
        background: #f8fafc;
      }}
      .voi-lang-btn {{
        padding: 4px 8px;
        border-radius: 7px;
        text-decoration: none !important;
        color: rgba(0,0,0,0.50);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.03em;
      }}
      .voi-lang-btn:hover {{
        background: rgba(0,0,0,0.06);
        color: rgba(0,0,0,0.85);
      }}
      .voi-lang-active {{
        background: #fff;
        color: rgba(0,0,0,0.9) !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.10);
      }}
    `;
    doc.head.appendChild(style);
  }}

  let root = doc.getElementById("voi-topbar-root");
  if (!root) {{
    root = doc.createElement("div");
    root.id = "voi-topbar-root";
    doc.body.appendChild(root);
  }}

  root.innerHTML = `
    <div id="voi-topbar-inner">
      <div id="voi-topbar-acc">{acc_venue}</div>
      <div id="voi-topbar-right">
        {lang_switcher_html}
        {catalog_icon_html}
        {history_icon_html}
        {manage_icon_html}
        <a id="voi-topbar-logout" href="{logout_href}" target="_self">{t("nav.logout")}</a>
      </div>
    </div>
  `;

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

# -------------------------------
# 6) Lazy page loaders (import on demand)
# -------------------------------
def _page_catalog():
    from features.catalog import catalog_tab
    return catalog_tab

def _page_new_order():
    from features.create_order.new_order_tab import new_order_tab
    return new_order_tab

def _page_borrador():
    from features.manage_orders.orders import borrador_tab
    return borrador_tab

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

#pages
PAGES = {
    "new": ("➕", _page_new_order),
    "borrador": ("📝", _page_borrador),
    "orders": ("📦", _page_orders),
    "tracking": ("✅", _page_tracking),
    "history": ("📈", _page_history),
    "catalog": ("🧾", _page_catalog),
    "manage_org": ("⚙️", _page_manage_org),
}

# Preferred order for the segmented control
PAGE_KEYS = ["new", "borrador", "orders", "tracking", "history", "catalog", "manage_org"]


def _home_card():
    st.markdown("<div class='voi-card'>", unsafe_allow_html=True)
    st.title("Voi")
    st.markdown("<div class='voi-muted'>Fast, mobile-first navigation.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)



import inspect as _inspect
from functools import lru_cache as _lru_cache

@_lru_cache(maxsize=32)
def _page_params(fn) -> frozenset | None:
    """Cache inspect.signature() results so we don't re-run reflection on every render."""
    try:
        return frozenset(_inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return None


def _call_page(fn, venue_id: int, **kwargs):
    """
    Call a page function, passing only the arguments it actually accepts.
    Supports pages WITH or WITHOUT venue_id.
    """
    params = _page_params(fn)
    if params is None:
        # Fallback: try venue_id first, then no args
        try:
            return fn(venue_id)
        except TypeError:
            return fn()

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

    # ✅ Perf: consume bottom-tab clicks early so we don't render the previous page
    _nav_triggers_early()

    # ✅ CRITICAL FIX: ALWAYS restore auth from token if token exists
    # Don't skip restoration just because auth_ctx exists - it might be stale!
    
    # Get token from URL (highest priority)
    token_from_url = None
    try:
        token_from_url = st.query_params.get("st", "").strip()
    except:
        pass
    
    # Get token from session (fallback)
    token_from_session = st.session_state.get("_session_token", "").strip()
    
    # Use URL token if available, otherwise session token
    active_token = token_from_url or token_from_session
    
    if active_token:
        # Always update session with the active token
        st.session_state["_session_token"] = active_token

        # ✅ PERF: Only re-validate against DB if the token changed or TTL expired (5 min).
        # This eliminates the DB round-trip on every Streamlit rerun.
        import time as _time
        _last_at = st.session_state.get("_token_validated_at", 0)
        _last_for = st.session_state.get("_token_validated_for", "")
        _token_fresh = (active_token == _last_for) and (_time.time() - _last_at) < 300

        if not _token_fresh:
            logging.debug("Validating token against DB...")
            try:
                from features.auth_and_manage.auth_multi_tenant import _get_session_from_token
                session_data = _get_session_from_token(active_token)

                if session_data:
                    st.session_state["auth_ctx"] = {
                        "user_id": session_data["user_id"],
                        "account_id": session_data["account_id"]
                    }
                    st.session_state["_token_validated_at"] = _time.time()
                    st.session_state["_token_validated_for"] = active_token
                    logging.debug(f"Auth restored: user={session_data['user_id']}")
                else:
                    logging.warning("Invalid token - clearing session")
                    st.session_state.pop("_session_token", None)
                    st.session_state.pop("auth_ctx", None)
                    st.session_state.pop("_token_validated_at", None)
                    st.session_state.pop("_token_validated_for", None)

            except Exception as e:
                logging.error(f"Error validating token: {e}")
                st.session_state.pop("_session_token", None)
                st.session_state.pop("auth_ctx", None)
                st.session_state.pop("_token_validated_at", None)
                st.session_state.pop("_token_validated_for", None)
    else:
        logging.debug("No token found")
    
    # Handle logout / actions early
    _handle_actions_from_query_params()

    # Read language from URL and store in session state
    try:
        url_lang = st.query_params.get("lang", "").strip().lower()
        if url_lang in ("el", "es", "en"):
            set_lang(url_lang)
        elif "lang" not in st.session_state:
            set_lang("el")  # default: Greek
    except Exception:
        pass

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

    # Auth gate (preserves URL params)
    auth_gate(show_manage_org=True, show_venue_selector=False)
    require_login()

    logging.debug(f"page={st.session_state.get('page')} venue={st.session_state.get('active_venue_id')}")

    u = current_user() or {}
    account_role = (u.get("account_role") or "member").lower()
    acc = current_account()

    page_key = st.session_state.get("page", "orders")

    active = current_active_venue()

    # ✅ Let Manage Org work even if user has no venue yet
    if not active:
        st.warning(t("msg.no_venue_access"))

        u0 = current_user() or {}
        account_role = (u0.get("account_role") or "member").lower()

        if page_key == "manage_org":
            # Render org management so user can create the first venue
            # manage_organization_ui expects a venue_role; without venues, treat owner/admin/manager as owner
            can_create = account_role in {"owner", "admin", "manager"}
            manage_organization_ui(venue_role=("owner" if can_create else "viewer"))
            st.stop()

        # Default message for other pages
        if account_role in {"owner", "admin", "manager"}:
            st.info(t("msg.create_first_venue"))
        else:
            st.info(t("msg.ask_admin_access"))
        return


    venue, venue_role = active
    venue_id = int(venue["id"])
    venue_name = venue.get("name", "")


    # ---------------- TOP BAR (true fixed) ----------------
    account_name = acc["name"] if acc else ""
    # page_key = st.session_state.get("page", "orders")

    show_manage_org = account_role in {"owner", "admin", "manager"}

    _inject_topbar(
        account_name=account_name,
        venue_name=venue_name,  # âœ… show venue next to account
        show_manage_org=show_manage_org,
        is_manage_page=(page_key == "manage_org"),
        is_catalog_page=(page_key == "catalog"),
        is_history_page=(page_key == "history"),
    )

    # ---------------- Render page ----------------
    title, loader = PAGES.get(page_key, PAGES["orders"])
    page_fn = loader()

    if page_key == "tracking":
        _call_page(
            page_fn,
            venue_id,
            venue_role=venue_role,
            deep_order_id=deep_order_id,
            deep_provider=(deep_provider or None),
        )
    elif page_key == "borrador":
        _call_page(
            page_fn,
            venue_id,
            venue_role=venue_role,
            deep_order_id=deep_order_id,
        )
    elif page_key == "orders":
        _call_page(
            page_fn,
            venue_id,
            venue_role=venue_role,
            deep_order_id=deep_order_id,
            deep_provider=(deep_provider or None),
            deep_status=(deep_status or None),
        )
    elif page_key == "history":
        _call_page(page_fn, venue_id, venue_role=venue_role, deep_provider=deep_provider)
    else:
        _call_page(page_fn, venue_id, venue_role=venue_role)

    # Render bottom bar in a fixed container so it appears immediately
    # and doesn't wait for page content to finish
    with st_fixed_container(mode="fixed", position="bottom", border=False, margin="0px"):
        _bottom_tabbar(page_key)


if __name__ == "__main__":
    main()

