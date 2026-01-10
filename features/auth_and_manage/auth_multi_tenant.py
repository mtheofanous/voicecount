"""
auth_multi_tenant.py
--------------------
Drop-in authentication + multi-tenant data model for your Streamlit + SQLModel app.
Optimized for performance with minimal reruns.
"""

from __future__ import annotations

import os
import base64
import hmac
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Any
import time

import pandas as pd
import streamlit as st
from sqlmodel import SQLModel, Field, Session, create_engine, select
from sqlalchemy import text, UniqueConstraint
from sqlalchemy.exc import IntegrityError

from core.config import get_database_url
from domain.models import Product, Provider, ProviderDiscountRule
from core.db import get_session as get_domain_session


# =========================================================
# Performance helpers
# =========================================================

def debounce(wait_time: float = 0.3):
    """Decorator to prevent rapid function calls"""
    def decorator(func):
        last_called = {}
        
        def wrapper(*args, **kwargs):
            key = f"debounce_{func.__name__}_{str(args)}_{str(kwargs)}"
            current_time = time.time()
            
            if key in last_called:
                if current_time - last_called[key] < wait_time:
                    return None
            
            last_called[key] = current_time
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def with_loading_spinner(func):
    """Add loading spinner for expensive operations"""
    def wrapper(*args, **kwargs):
        with st.spinner("Processing..."):
            return func(*args, **kwargs)
    return wrapper


# =========================================================
# Database
# =========================================================

AUTH_DB_URL = get_database_url()  # reuse your existing DB by default


@st.cache_resource
def get_auth_engine():
    """Create and cache the SQLAlchemy/SQLModel engine (connection pool) per process."""
    return create_engine(AUTH_DB_URL, echo=False, pool_pre_ping=True)


def get_auth_session() -> Session:
    return Session(get_auth_engine())


# =========================================================
# Password hashing (stdlib)
# =========================================================

_PBKDF2_ITERS = 200_000
_SALT_BYTES = 16
_KEY_BYTES = 32


def _pbkdf2_hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERS, dklen=_KEY_BYTES)


def hash_password(password: str) -> str:
    """
    Returns a compact string: pbkdf2_sha256$<iters>$<salt_b64>$<hash_b64>
    """
    salt = os.urandom(_SALT_BYTES)
    key = _pbkdf2_hash_password(password, salt)
    return "pbkdf2_sha256${}${}${}".format(
        _PBKDF2_ITERS,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(key).decode("ascii"),
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iters_s, salt_b64, hash_b64 = stored.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        iters = int(iters_s)
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(hash_b64.encode("ascii"))
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters, dklen=len(expected))
        return hmac.compare_digest(candidate, expected)
    except Exception:
        return False


# =========================================================
# Models (multi-tenant)
# =========================================================

class Account(SQLModel, table=True):
    __tablename__ = "account"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow)


class User(SQLModel, table=True):
    __tablename__ = "user"
    __table_args__ = (
        UniqueConstraint("account_id", "email", name="uq_user_account_email"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)

    account_id: int = Field(foreign_key="account.id", index=True)

    full_name: str
    email: str = Field(index=True)
    password_hash: str

    # Global account role (optional): owner/manager/
    account_role: str = Field(default="owner", index=True)

    is_active: bool = Field(default=True, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Venue(SQLModel, table=True):
    __tablename__ = "venue"
    __table_args__ = (
        UniqueConstraint("account_id", "tax_number", name="uq_venue_account_tax"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    account_id: int = Field(foreign_key="account.id", index=True)

    # Required by you:
    name: str
    tax_number: str = Field(index=True)
    address: str
    phone: str
    email: str

    # Optional (used for invoice-style header in messages)
    owner_name: Optional[str] = Field(default=None)

    # Per-venue email/WhatsApp templates (configured in Auth → "Configure emails")
    # Placeholders supported: {order_id}, {date}, {venue_name}
    email_subject_tpl: Optional[str] = Field(default=None)
    email_opening_tpl: Optional[str] = Field(default=None)
    email_closing_tpl: Optional[str] = Field(default=None)

    created_at: datetime = Field(default_factory=datetime.utcnow)


class VenueUser(SQLModel, table=True):
    """
    Association table: many-to-many between Venue and User with per-venue permission.
    """
    __tablename__ = "venue_user"
    __table_args__ = (
        UniqueConstraint("venue_id", "user_id", name="uq_venue_user"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)

    venue_id: int = Field(foreign_key="venue.id", index=True)
    user_id: int = Field(foreign_key="user.id", index=True)

    # per-venue role examples: owner, manager, staff, viewer
    role: str = Field(default="staff", index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow)


# =========================================================
# Tiny SQLite migration with performance
# =========================================================

def _has_column(conn, table: str, column: str) -> bool:
    cols = conn.execute(text(f"PRAGMA table_info('{table}')")).fetchall()
    existing = {row[1] for row in cols}
    return column in existing


@st.cache_resource
def init_auth_db_once() -> bool:
    """Initialize auth DB once per session"""
    SQLModel.metadata.create_all(get_auth_engine())
    return True


@with_loading_spinner
def backfill_account_roles():
    """
    Fix legacy users where account_role is NULL or incorrect.
    Ensures at least one owner per account.
    """
    with get_auth_session() as s:
        accounts = s.exec(select(Account)).all()

        for acc in accounts:
            users = s.exec(
                select(User)
                .where(User.account_id == acc.id, User.is_active == True)
                .order_by(User.created_at.asc())
            ).all()

            if not users:
                continue

            # First user becomes owner
            first = users[0]
            if not first.account_role or first.account_role.strip() == "":
                first.account_role = "owner"
                s.add(first)

            # Normalize bad roles
            for u in users:
                if u.account_role not in {"owner", "manager", "staff", "viewer"}:
                    u.account_role = "staff"
                    s.add(u)

        s.commit()


def init_auth_db() -> None:
    """
    Creates tables and adds missing columns on older DBs.
    Safe to call on every run.
    """
    # Initialize once
    db_initialized = init_auth_db_once()
    
    if not db_initialized:
        st.error("Failed to initialize auth database")
        return
    
    # Run migrations in background if needed
    with get_auth_engine().begin() as conn:
        # account
        if not _has_column(conn, "account", "created_at"):
            conn.execute(text("ALTER TABLE account ADD COLUMN created_at TEXT"))

        # user
        for col, ddl in [
            ("full_name", "ALTER TABLE user ADD COLUMN full_name TEXT"),
            ("account_id", "ALTER TABLE user ADD COLUMN account_id INTEGER"),
            ("email", "ALTER TABLE user ADD COLUMN email TEXT"),
            ("password_hash", "ALTER TABLE user ADD COLUMN password_hash TEXT"),
            ("account_role", "ALTER TABLE user ADD COLUMN account_role TEXT"),
            ("is_active", "ALTER TABLE user ADD COLUMN is_active INTEGER"),
            ("created_at", "ALTER TABLE user ADD COLUMN created_at TEXT"),
        ]:
            if not _has_column(conn, "user", col):
                conn.execute(text(ddl))

        # venue
        for col, ddl in [
            ("account_id", "ALTER TABLE venue ADD COLUMN account_id INTEGER"),
            ("name", "ALTER TABLE venue ADD COLUMN name TEXT"),
            ("tax_number", "ALTER TABLE venue ADD COLUMN tax_number TEXT"),
            ("address", "ALTER TABLE venue ADD COLUMN address TEXT"),
            ("phone", "ALTER TABLE venue ADD COLUMN phone TEXT"),
            ("email", "ALTER TABLE venue ADD COLUMN email TEXT"),
            ("owner_name", "ALTER TABLE venue ADD COLUMN owner_name TEXT"),
            ("email_subject_tpl", "ALTER TABLE venue ADD COLUMN email_subject_tpl TEXT"),
            ("email_opening_tpl", "ALTER TABLE venue ADD COLUMN email_opening_tpl TEXT"),
            ("email_closing_tpl", "ALTER TABLE venue ADD COLUMN email_closing_tpl TEXT"),
            ("created_at", "ALTER TABLE venue ADD COLUMN created_at TEXT"),
        ]:
            if not _has_column(conn, "venue", col):
                conn.execute(text(ddl))

        # venue_user
        for col, ddl in [
            ("venue_id", "ALTER TABLE venue_user ADD COLUMN venue_id INTEGER"),
            ("user_id", "ALTER TABLE venue_user ADD COLUMN user_id INTEGER"),
            ("role", "ALTER TABLE venue_user ADD COLUMN role TEXT"),
            ("created_at", "ALTER TABLE venue_user ADD COLUMN created_at TEXT"),
        ]:
            if not _has_column(conn, "venue_user", col):
                conn.execute(text(ddl))
    
    # Run backfill in background
    try:
        backfill_account_roles()
    except Exception:
        pass  # Silent fail, will retry next time


# =========================================================
# Auth session helpers
# =========================================================

@dataclass(frozen=True)
class AuthContext:
    user_id: int
    account_id: int


def _set_auth(user_id: int, account_id: int) -> None:
    st.session_state["auth_ctx"] = {"user_id": int(user_id), "account_id": int(account_id)}


def clear_auth() -> None:
    st.session_state.pop("auth_ctx", None)
    st.session_state.pop("active_venue_id", None)
    # Clear auth caches
    _cached_user_dict.clear()
    _cached_account_dict.clear()
    _cached_venues_for_user.clear()


def is_logged_in() -> bool:
    ctx = st.session_state.get("auth_ctx")
    return isinstance(ctx, dict) and "user_id" in ctx and "account_id" in ctx


def current_auth() -> Optional[AuthContext]:
    if not is_logged_in():
        return None
    ctx = st.session_state["auth_ctx"]
    return AuthContext(user_id=int(ctx["user_id"]), account_id=int(ctx["account_id"]))


# =========================================================
# Cache (production) with performance improvements
# =========================================================

def invalidate_auth_caches() -> None:
    """Clear cached auth reads after any write (create/update/delete)."""
    _cached_user_dict.clear()
    _cached_account_dict.clear()
    _cached_venues_for_user.clear()


@st.cache_data(ttl=300)  # Cache for 5 minutes
def _cached_user_dict(user_id: int) -> Optional[dict]:
    with get_auth_session() as s:
        u = s.exec(select(User).where(User.id == int(user_id), User.is_active == True)).first()
        if not u:
            return None
        return {
            "id": int(u.id),
            "account_id": int(u.account_id),
            "full_name": u.full_name,
            "email": u.email,
            "account_role": u.account_role,
        }


@st.cache_data(ttl=300)  # Cache for 5 minutes
def _cached_account_dict(account_id: int) -> Optional[dict]:
    with get_auth_session() as s:
        acc = s.exec(select(Account).where(Account.id == int(account_id))).first()
        if not acc:
            return None
        return {"id": int(acc.id), "name": acc.name}


@st.cache_data(ttl=60)  # Cache for 1 minute
def _cached_venues_for_user(user_id: int) -> List[Tuple[dict, str]]:
    """User-scoped venues + roles."""
    with get_auth_session() as s:
        links = s.exec(select(VenueUser).where(VenueUser.user_id == int(user_id))).all()
        if not links:
            return []
        venue_ids = [l.venue_id for l in links]
        venues = s.exec(select(Venue).where(Venue.id.in_(venue_ids))).all()
        role_by_vid = {l.venue_id: l.role for l in links}

        out: List[Tuple[dict, str]] = []
        for v in venues:
            out.append((
                {
                    "id": int(v.id),
                    "account_id": int(v.account_id),
                    "name": v.name,
                    "tax_number": v.tax_number,
                    "address": v.address,
                    "phone": v.phone,
                    "email": v.email,
                },
                role_by_vid.get(v.id, "staff"),
            ))
        return out


def current_user() -> Optional[dict]:
    ctx = current_auth()
    if not ctx:
        return None
    return _cached_user_dict(ctx.user_id)


def current_account() -> Optional[dict]:
    ctx = current_auth()
    if not ctx:
        return None
    return _cached_account_dict(ctx.account_id)


def current_venues_for_user() -> List[Tuple[dict, str]]:
    """Returns list of (venue_dict, role) for the logged-in user."""
    ctx = current_auth()
    if not ctx:
        return []
    return _cached_venues_for_user(ctx.user_id)


def current_active_venue() -> Optional[Tuple[dict, str]]:
    """
    Returns (venue_dict, role) for the selected venue in session_state['active_venue_id'].
    """
    venues = current_venues_for_user()
    if not venues:
        return None

    active_id = st.session_state.get("active_venue_id")
    if active_id is None:
        v0, r0 = venues[0]
        st.session_state["active_venue_id"] = v0["id"]
        return v0, r0

    for v, r in venues:
        if int(v["id"]) == int(active_id):
            return v, r

    v0, r0 = venues[0]
    st.session_state["active_venue_id"] = v0["id"]
    return v0, r0


# =========================================================
# CRUD helpers with performance improvements
# =========================================================

@st.cache_data(ttl=60)  # Cache for 1 minute
def list_venues_for_account_cached(account_id: int) -> List[Venue]:
    """Cached venue listing for account"""
    with get_auth_session() as s:
        return s.exec(select(Venue).where(Venue.account_id == account_id).order_by(Venue.name.asc())).all()


@st.cache_data(ttl=60)  # Cache for 1 minute
def list_users_for_account_cached(account_id: int) -> List[User]:
    """Cached user listing for account"""
    with get_auth_session() as s:
        return s.exec(select(User).where(User.account_id == account_id).order_by(User.email.asc())).all()


def list_venues_for_account(account_id: int) -> List[Venue]:
    return list_venues_for_account_cached(account_id)


def list_users_for_account(account_id: int) -> List[User]:
    return list_users_for_account_cached(account_id)


def list_all_venues() -> List[Venue]:
    with get_auth_session() as s:
        return s.exec(select(Venue).order_by(Venue.name.asc())).all()


def list_all_users() -> List[User]:
    with get_auth_session() as s:
        return s.exec(select(User).order_by(User.email.asc())).all()


@with_loading_spinner
@debounce(0.5)
def update_user_basic(user_id: int, *, full_name: str, email: str, account_role: str, is_active: bool) -> None:
    with get_auth_session() as s:
        u = s.exec(select(User).where(User.id == user_id)).first()
        if not u:
            raise ValueError("User not found")
        
        before = {"full_name": u.full_name, "email": u.email, "account_role": u.account_role, "is_active": u.is_active}
        u.full_name = full_name
        u.email = email
        u.account_role = account_role
        u.is_active = bool(is_active)
        s.add(u)
        s.commit()
        
    invalidate_auth_caches()


@with_loading_spinner
@debounce(0.5)
def reset_user_password(user_id: int, new_password: str) -> None:
    if len(new_password or "") < 8:
        raise ValueError("Password must be at least 8 characters.")
    
    with get_auth_session() as s:
        u = s.exec(select(User).where(User.id == user_id)).first()
        if not u:
            raise ValueError("User not found")
        u.password_hash = hash_password(new_password)
        s.add(u)
        s.commit()
    
    invalidate_auth_caches()


@with_loading_spinner
@debounce(0.5)
def update_venue_basic(venue_id: int, *, name: str, tax_number: str, address: str, phone: str, email: str) -> None:
    with get_auth_session() as s:
        v = s.exec(select(Venue).where(Venue.id == venue_id)).first()
        if not v:
            raise ValueError("Venue not found")
        
        before = {"name": v.name, "tax_number": v.tax_number, "address": v.address, "phone": v.phone, "email": v.email}
        v.name = name
        v.tax_number = tax_number
        v.address = address
        v.phone = phone
        v.email = email
        s.add(v)
        s.commit()
    
    invalidate_auth_caches()


@with_loading_spinner
def create_account_with_owner(
    account_name: str,
    owner_full_name: str,
    owner_email: str,
    owner_password: str,
) -> Tuple[int, int]:
    account_name = (account_name or "").strip()
    owner_full_name = (owner_full_name or "").strip()
    owner_email = (owner_email or "").strip().lower()

    if not account_name:
        raise ValueError("Account name is required.")
    if not owner_full_name:
        raise ValueError("Full name is required.")
    if not owner_email or "@" not in owner_email:
        raise ValueError("Valid email is required.")
    if not owner_password or len(owner_password) < 8:
        raise ValueError("Password must be at least 8 characters.")

    with get_auth_session() as s:
        # ensure email is not used in any account
        exists = s.exec(select(User).where(User.email == owner_email)).first()
        if exists:
            raise ValueError("This email is already registered.")

        acc = Account(name=account_name)
        s.add(acc)
        s.commit()
        s.refresh(acc)

        u = User(
            account_id=int(acc.id),
            full_name=owner_full_name,
            email=owner_email,
            password_hash=hash_password(owner_password),
            account_role="owner",
            is_active=True,
        )
        s.add(u)
        s.commit()
        s.refresh(u)
        
        invalidate_auth_caches()
        return int(acc.id), int(u.id)


@with_loading_spinner
def authenticate(email: str, password: str) -> Optional[Tuple[int, int]]:
    email = (email or "").strip().lower()
    if not email or not password:
        return None

    with get_auth_session() as s:
        u = s.exec(select(User).where(User.email == email, User.is_active == True)).first()
        if not u:
            return None
        if not verify_password(password, u.password_hash or ""):
            return None

        # Return IDs ONLY
        return int(u.id), int(u.account_id)


@with_loading_spinner
@debounce(0.5)
def create_venue(
    account_id: int,
    name: str,
    tax_number: str,
    address: str,
    phone: str,
    email: str,
) -> Venue:
    name = (name or "").strip()
    tax_number = (tax_number or "").strip()
    address = (address or "").strip()
    phone = (phone or "").strip()
    email = (email or "").strip().lower()

    if not all([name, tax_number, address, phone, email]):
        raise ValueError("All venue fields are required.")

    with get_auth_session() as s:
        # Friendly pre-check (avoids crashing UI)
        existing = s.exec(
            select(Venue).where(
                Venue.account_id == int(account_id),
                Venue.tax_number == tax_number,
            )
        ).first()
        if existing:
            raise ValueError("A venue with this Tax Number already exists in this account.")

        try:
            v = Venue(
                account_id=int(account_id),
                name=name,
                tax_number=tax_number,
                address=address,
                phone=phone,
                email=email,
            )
            s.add(v)
            s.commit()
            s.refresh(v)
            invalidate_auth_caches()
            return v

        except IntegrityError as e:
            s.rollback()
            msg = "A venue with this Tax Number already exists in this account."
            raise ValueError(msg) from e


@with_loading_spinner
@debounce(0.5)
def add_user_to_venue(venue_id: int, user_id: int, role: str) -> VenueUser:
    role = (role or "staff").strip().lower()
    if role not in {"owner", "manager", "staff", "viewer"}:
        raise ValueError("Role must be one of: owner, manager, staff, viewer.")

    with get_auth_session() as s:
        existing = s.exec(
            select(VenueUser).where(
                VenueUser.venue_id == venue_id,
                VenueUser.user_id == user_id,
            )
        ).first()

        if existing:
            existing.role = role
            s.add(existing)
            s.commit()
            s.refresh(existing)
            invalidate_auth_caches()
            return existing
        else:
            link = VenueUser(
                venue_id=int(venue_id),
                user_id=int(user_id),
                role=role,
            )
            s.add(link)
            s.commit()
            s.refresh(link)
            invalidate_auth_caches()
            return link


# =========================================================
# Streamlit UI with performance improvements
# =========================================================

def require_login() -> None:
    if not is_logged_in():
        st.stop()


def auth_gate(
    *,
    title: str = "Welcome",
    show_manage_org: bool = True,
) -> None:
    """
    If logged out -> show Login/Sign up.
    If logged in -> show account + venue selector + optional org management.
    """
    # Initialize once
    init_auth_db()

    # Anti-flicker guard
    rerun_key = "auth_rerun_guard"
    if st.session_state.get(rerun_key, False):
        st.session_state[rerun_key] = False
        time.sleep(0.1)
        return

    if not is_logged_in():
        st.title(title)
        
        # Use tabs for login/signup
        t_login, t_signup = st.tabs(["🔐 Login", "✨ Sign up"])

        with t_login:
            # Wrap in form to prevent reruns
            with st.form(key="login_form", clear_on_submit=False):
                email = st.text_input("Email", key="login_email").strip().lower()
                pwd = st.text_input("Password", type="password", key="login_pwd")
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    submitted = st.form_submit_button("Login", type="primary")
                with col2:
                    st.caption("Use the same email you registered with.")

            if submitted:
                with st.spinner("Authenticating..."):
                    res = authenticate(email, pwd)
                    if not res:
                        st.error("Invalid email or password.")
                    else:
                        user_id, account_id = res
                        _set_auth(user_id, account_id)
                        st.success("Logged in ✅")
                        st.session_state[rerun_key] = True
                        time.sleep(0.5)
                        st.rerun()

        with t_signup:
            st.caption("Create a new company account. The first user becomes the owner.")

            # Wrap in form
            with st.form(key="signup_form", clear_on_submit=False):
                acc_name = st.text_input("Company / Account name", key="su_acc_name")
                full_name = st.text_input("Your full name", key="su_full_name")
                email = st.text_input("Email", key="su_email").strip().lower()
                pwd = st.text_input("Password (min 8 chars)", type="password", key="su_pwd")
                pwd2 = st.text_input("Repeat password", type="password", key="su_pwd2")

                submitted = st.form_submit_button("Create account", type="primary")

            if submitted:
                with st.spinner("Creating account..."):
                    try:
                        if pwd != pwd2:
                            raise ValueError("Passwords do not match.")

                        acc_id, user_id = create_account_with_owner(acc_name, full_name, email, pwd)
                        _set_auth(user_id, acc_id)
                        invalidate_auth_caches()
                        st.success("Account created ✅")
                        st.session_state[rerun_key] = True
                        time.sleep(0.5)
                        st.rerun()

                    except Exception as e:
                        st.error(str(e))

        st.stop()

    # Logged-in area
    u = current_user()
    acc = current_account()

    # Top bar
    top_left, top_right = st.columns([3, 1])
    with top_left:
        st.markdown(f"**Logged in:** {u['full_name'] if u else '—'}  •  **Account:** {acc['name'] if acc else '—'}")

    with top_right:
        if st.button("Logout", key="logout_btn_gat", width='stretch'):
            clear_auth()
            st.session_state[rerun_key] = True
            st.rerun()

    # Venue selector with caching
    venues = current_venues_for_user()
    if venues:
        labels = []
        ids = []
        roles = {}
        for v, role in venues:
            labels.append(f"{v['name']} — {role}")
            ids.append(v["id"])
            roles[v["id"]] = role

        active = st.session_state.get("active_venue_id")
        if active not in ids:
            st.session_state["active_venue_id"] = ids[0]
            active = ids[0]

        idx = ids.index(active)
        
        # Use selectbox with on_change to prevent reruns
        chosen_label = st.selectbox(
            "Active bar/restaurant", 
            options=labels, 
            index=idx, 
            key="active_venue_select",
            on_change=lambda: None
        )
        
        chosen_id = ids[labels.index(chosen_label)]
        if st.session_state.get("active_venue_id") != chosen_id:
            st.session_state["active_venue_id"] = chosen_id
            # Small delay before rerun to prevent flickering
            time.sleep(0.1)
            st.session_state[rerun_key] = True
            st.rerun()

    else:
        st.warning("You don't have access to any venue yet.")

        u0 = current_user() or {}
        role0 = (u0.get("account_role") or "member").lower()

        if role0 in {"owner", "manager"}:
            st.info("Create your first venue below to start using the app.")

            if show_manage_org:
                # Render org management in place
                manage_organization_ui(venue_role="owner")

            st.stop()
        else:
            st.info("Ask an admin to grant you access.")
            st.stop()

    if not show_manage_org:
        return


@st.cache_data(ttl=60)
def get_venue_email_templates_cached(venue_id: int) -> Optional[Tuple[str, str, str]]:
    """Cache venue email templates"""
    with get_auth_session() as s:
        venue = s.exec(select(Venue).where(Venue.id == venue_id)).first()
        if not venue:
            return None
        return (
            venue.email_subject_tpl or "",
            venue.email_opening_tpl or "",
            venue.email_closing_tpl or ""
        )


def manage_organization_ui(*, venue_role: str) -> None:
    """
    Manage Organization with performance improvements.
    """
    u = current_user()
    acc = current_account()

    if not u or not acc:
        st.stop()

    st.subheader("🏢 Manage organization")

    # Gate by ACTIVE venue role
    if (venue_role or "").lower() not in {"owner", "manager"}:
        st.info("You don't have permission to access organization management.")
        st.stop()

    # Use tabs for better organization
    org_tabs = st.tabs(["👤 Users", "🏪 Venues", "✉️ Email Templates", "📇 Providers"])

    # ------------------------------------------------------------------
    # TAB 1: Users
    # ------------------------------------------------------------------
    with org_tabs[0]:
        st.markdown("### User Management")
        
        # Create user section
        with st.expander("✉️ Create new user", expanded=False):
            with st.form("create_user_form", clear_on_submit=True):
                c1, c2 = st.columns(2)
                with c1:
                    new_full = st.text_input("Full name", key="new_user_full")
                    new_email = st.text_input("Email", key="new_user_email").strip().lower()
                with c2:
                    new_pwd = st.text_input("Temp password (min 8 chars)", type="password", key="new_user_pwd")
                    new_role = st.selectbox(
                        "Account role",
                        options=["owner", "manager", "staff", "viewer"],
                        index=2,
                        key="new_user_role",
                    )

                submitted = st.form_submit_button("Create user", type="primary")

            if submitted:
                try:
                    if not new_email or "@" not in new_email:
                        raise ValueError("Valid email required.")
                    if not new_pwd or len(new_pwd) < 8:
                        raise ValueError("Password must be at least 8 characters.")

                    with get_auth_session() as s:
                        exists = s.exec(
                            select(User).where(User.account_id == acc["id"], User.email == new_email)
                        ).first()
                        if exists:
                            raise ValueError("This email already exists in this account.")

                        nu = User(
                            account_id=acc["id"],
                            full_name=(new_full or "").strip() or new_email,
                            email=new_email,
                            password_hash=hash_password(new_pwd),
                            account_role=new_role,
                            is_active=True,
                        )
                        s.add(nu)
                        s.commit()
                        s.refresh(nu)

                        # Auto-grant access to the currently active venue
                        active_vid = st.session_state.get("active_venue_id")
                        if active_vid:
                            add_user_to_venue(int(active_vid), int(nu.id), "staff")

                    invalidate_auth_caches()
                    st.success("User created ✅")
                    time.sleep(0.5)
                    st.rerun()

                except Exception as e:
                    st.error(str(e))

        # Permissions matrix
        with st.expander("📋 Permissions matrix", expanded=False):
            # Load data with caching
            all_venues = list_venues_for_account(acc["id"])
            all_users = list_users_for_account(acc["id"])

            if not all_venues or not all_users:
                st.info("No venues or users found.")
            else:
                # Load links
                venue_ids = [v.id for v in all_venues]
                user_ids = [u2.id for u2 in all_users]

                with get_auth_session() as s:
                    links = s.exec(
                        select(VenueUser).where(
                            VenueUser.venue_id.in_(venue_ids),
                            VenueUser.user_id.in_(user_ids),
                        )
                    ).all()

                role_map = {(l.user_id, l.venue_id): l.role for l in links}

                # Matrix display
                venue_cols = [f"{v.name} (#{v.id})" for v in all_venues]
                user_rows = [f"{u2.email} — {u2.full_name} (#{u2.id})" for u2 in all_users]

                matrix = []
                for usr in all_users:
                    row = []
                    for v in all_venues:
                        row.append(role_map.get((usr.id, v.id), "—"))
                    matrix.append(row)

                df = pd.DataFrame(matrix, index=user_rows, columns=venue_cols)
                st.caption("Role per user per venue (— means no access)")
                st.dataframe(df, width='stretch')

                # Edit permissions
                st.divider()
                st.subheader("Edit permissions")
                
                user_opt = {f"{u2.email} — {u2.full_name} (#{u2.id})": u2 for u2 in all_users}
                venue_opt = {f"{v.name} (#{v.id})": v for v in all_venues}

                c1, c2, c3 = st.columns(3)
                with c1:
                    u_label = st.selectbox("User", list(user_opt.keys()), key="mx_user")
                with c2:
                    v_label = st.selectbox("Venue", list(venue_opt.keys()), key="mx_venue")
                with c3:
                    current_role = role_map.get((user_opt[u_label].id, venue_opt[v_label].id))
                    role_options = ["owner", "manager", "staff", "viewer"]
                    default_idx = role_options.index(current_role) if current_role in role_options else 2
                    new_role = st.selectbox("Role", role_options, index=default_idx, key="mx_role")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Save permission", type="primary", width='stretch'):
                        try:
                            with get_auth_session() as s:
                                link = s.exec(
                                    select(VenueUser).where(
                                        VenueUser.user_id == user_opt[u_label].id,
                                        VenueUser.venue_id == venue_opt[v_label].id,
                                    )
                                ).first()

                                if link:
                                    link.role = new_role
                                else:
                                    link = VenueUser(
                                        user_id=user_opt[u_label].id,
                                        venue_id=venue_opt[v_label].id,
                                        role=new_role,
                                    )
                                    s.add(link)

                                s.commit()

                            invalidate_auth_caches()
                            st.success("Permission saved ✅")
                            time.sleep(0.5)
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
                
                with col2:
                    if st.button("🗑️ Remove access", type="secondary", width='stretch'):
                        try:
                            with get_auth_session() as s:
                                link = s.exec(
                                    select(VenueUser).where(
                                        VenueUser.user_id == user_opt[u_label].id,
                                        VenueUser.venue_id == venue_opt[v_label].id,
                                    )
                                ).first()

                                if link:
                                    s.delete(link)
                                    s.commit()
                                    invalidate_auth_caches()
                                    st.success("Access removed ✅")
                                    time.sleep(0.5)
                                    st.rerun()
                                else:
                                    st.info("No access to remove.")
                        except Exception as e:
                            st.error(str(e))

    # ------------------------------------------------------------------
    # TAB 2: Venues
    # ------------------------------------------------------------------
    with org_tabs[1]:
        st.markdown("### Venue Management")
        
        if (venue_role or "").lower() == "owner":
            # Create venue
            with st.expander("➕ Add a bar/restaurant", expanded=False):
                with st.form("create_venue_form", clear_on_submit=True):
                    c1, c2 = st.columns(2)
                    with c1:
                        v_name = st.text_input("Name", key="v_name")
                        v_tax = st.text_input("Tax number", key="v_tax")
                        v_phone = st.text_input("Phone", key="v_phone")
                    with c2:
                        v_email = st.text_input("Email", key="v_email").strip().lower()
                        v_addr = st.text_input("Address", key="v_addr")

                    submitted = st.form_submit_button("Create venue", type="primary")

                if submitted:
                    try:
                        v = create_venue(acc["id"], v_name, v_tax, v_addr, v_phone, v_email)
                        # owner gets access automatically
                        add_user_to_venue(v.id, u["id"], "owner")
                        invalidate_auth_caches()
                        st.success(f"Venue created ✅ ({v.name})")
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

            # Edit venues
            with st.expander("🏪 Edit venues", expanded=False):
                venues = list_venues_for_account(acc["id"])
                
                if not venues:
                    st.info("No venues found.")
                else:
                    venue_opts = {f"{v.name} (#{v.id})": v for v in venues}
                    
                    with st.form("edit_venue_form", clear_on_submit=False):
                        v_label = st.selectbox("Select venue", options=list(venue_opts.keys()), key="mv_select_venue")
                        chosen_venue = venue_opts[v_label]

                        c1, c2 = st.columns(2)
                        with c1:
                            new_name = st.text_input("Name", value=chosen_venue.name or "", key=f"mv_name_{chosen_venue.id}")
                            new_tax = st.text_input("Tax number", value=chosen_venue.tax_number or "", key=f"mv_tax_{chosen_venue.id}")
                            new_phone = st.text_input("Phone", value=chosen_venue.phone or "", key=f"mv_phone_{chosen_venue.id}")
                        with c2:
                            new_email = st.text_input("Email", value=(chosen_venue.email or "").lower(), key=f"mv_email_{chosen_venue.id}")
                            new_addr = st.text_input("Address", value=chosen_venue.address or "", key=f"mv_addr_{chosen_venue.id}")

                        do_save = st.form_submit_button("💾 Save changes", type="primary")

                    if do_save:
                        try:
                            new_name = (new_name or "").strip()
                            new_tax = (new_tax or "").strip()
                            new_phone = (new_phone or "").strip()
                            new_email = (new_email or "").strip().lower()
                            new_addr = (new_addr or "").strip()

                            if not all([new_name, new_tax, new_phone, new_email, new_addr]):
                                raise ValueError("All venue fields are required.")

                            with get_auth_session() as s:
                                v = s.exec(
                                    select(Venue).where(Venue.id == chosen_venue.id, Venue.account_id == acc["id"])
                                ).first()
                                if not v:
                                    raise ValueError("Venue not found in this account.")

                                dup = s.exec(
                                    select(Venue).where(
                                        Venue.account_id == acc["id"],
                                        Venue.tax_number == new_tax,
                                        Venue.id != v.id,
                                    )
                                ).first()
                                if dup:
                                    raise ValueError("This tax number is already used by another venue in this account.")

                                v.name = new_name
                                v.tax_number = new_tax
                                v.phone = new_phone
                                v.email = new_email
                                v.address = new_addr

                                s.add(v)
                                s.commit()

                            invalidate_auth_caches()
                            st.success("Venue updated ✅")
                            time.sleep(0.5)
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
        else:
            st.info("Only account owners can manage venues.")

    # ------------------------------------------------------------------
    # TAB 3: Email Templates
    # ------------------------------------------------------------------
    with org_tabs[2]:
        st.markdown("### Email Templates")

        if (u.get("account_role") or "").lower() not in {"owner", "manager"}:
            st.info("Only account owners/managers can configure email templates.")
        else:
            venues = list_venues_for_account(acc["id"])

            if not venues:
                st.info("No venues found in this account.")
            else:
                venue_opts = {f"{v.name} (#{v.id})": v for v in venues}
                chosen_label = st.selectbox(
                    "Select venue",
                    options=list(venue_opts.keys()),
                    key="cfg_email_venue_sel",
                )
                chosen_venue = venue_opts[chosen_label]

                # Get owner name
                with get_auth_session() as s:
                    owner_name = s.exec(
                        select(User.full_name)
                        .where(
                            User.account_id == acc["id"],
                            User.account_role == "owner",
                            User.is_active == True,
                        )
                        .order_by(User.created_at.asc())
                    ).first()

                owner_name = owner_name or u.get("full_name", "") or ""

                # ----------------------------
                # Standardized subject + close
                # ----------------------------
                def build_subject(order_id: int, venue_name: str, date_str: str) -> str:
                    return f"Pedido #{order_id} — {venue_name} — {date_str}"

                def build_closing(owner_name: str, lang: str) -> str:
                    # You asked to keep this always (Spanish). If you want localized closings later,
                    # we can switch by lang here.
                    return (
                        "Gracias.\n\n"
                        f"Atentamente,\n{owner_name or ''}".strip()
                    )

                # ----------------------------
                # Language packs (opening + footer)
                # ----------------------------
                LANGS = {
                    "Español (ES)": "es",
                    "English (EN)": "en",
                    "Ελληνικά (GR)": "gr",
                }

                DEFAULT_OPEN = {
                    "es": (
                        "Hola,\n\n"
                        "Adjunto el pedido actualizado. Por favor, confirma disponibilidad y plazos de entrega.\n"
                    ),
                    "en": (
                        "Hello,\n\n"
                        "Please find the updated order attached. Kindly confirm availability and delivery lead times.\n"
                    ),
                    "gr": (
                        "Γεια σας,\n\n"
                        "Σας επισυνάπτω την ενημερωμένη παραγγελία. Παρακαλώ επιβεβαιώστε διαθεσιμότητα και χρόνο παράδοσης.\n"
                    ),
                }

                def build_legal_footer(v, lang: str) -> str:
                    """
                    Simple compliance footer. Uses venue data as 'company' identity.
                    If you store a separate legal entity on Account, swap values here.
                    """
                    company = (v.name or "").strip()
                    address = (v.address or "").strip()
                    vat = (v.tax_number or "").strip()

                    if lang == "en":
                        a = "Address"
                        vlabel = "VAT / Tax ID"
                        notice = "This email (and any attachments) may contain confidential information."
                    elif lang == "gr":
                        a = "Διεύθυνση"
                        vlabel = "ΑΦΜ"
                        notice = "Αυτό το email (και τυχόν συνημμένα) μπορεί να περιέχει εμπιστευτικές πληροφορίες."
                    else:
                        a = "Dirección"
                        vlabel = "CIF/NIF"
                        notice = "Este email (y cualquier adjunto) puede contener información confidencial."

                    lines = []
                    if company:
                        lines.append(company)
                    if address:
                        lines.append(f"{a}: {address}")
                    if vat:
                        lines.append(f"{vlabel}: {vat}")

                    # If no data, still return a minimal footer line
                    base = " | ".join(lines) if lines else "(company details incomplete)"
                    return f"{base}\n{notice}"

                # ----------------------------
                # Check missing fields (sending requirements)
                # ----------------------------
                missing = []
                if not (chosen_venue.address or "").strip():
                    missing.append("address")
                if not (chosen_venue.tax_number or "").strip():
                    missing.append("tax number")
                if not (chosen_venue.email or "").strip():
                    missing.append("email")
                if not (chosen_venue.phone or "").strip():
                    missing.append("phone")

                if missing:
                    st.warning(
                        "This venue is missing required fields for sending: "
                        + ", ".join(missing)
                        + ". Please complete them in **Manage venues**."
                    )

                # ----------------------------
                # Load current settings
                # ----------------------------
                current_lang = (getattr(chosen_venue, "email_lang", None) or "es").strip().lower()
                if current_lang not in {"es", "en", "gr"}:
                    current_lang = "es"

                current_cc = (getattr(chosen_venue, "email_cc", "") or "").strip()
                current_bcc = (getattr(chosen_venue, "email_bcc", "") or "").strip()

                # per-language opening fields
                open_es = (getattr(chosen_venue, "email_opening_tpl_es", None) or "").strip()
                open_en = (getattr(chosen_venue, "email_opening_tpl_en", None) or "").strip()
                open_gr = (getattr(chosen_venue, "email_opening_tpl_gr", None) or "").strip()

                # If empty, use defaults
                if not open_es:
                    open_es = DEFAULT_OPEN["es"]
                if not open_en:
                    open_en = DEFAULT_OPEN["en"]
                if not open_gr:
                    open_gr = DEFAULT_OPEN["gr"]

                st.info(
                    "✅ **Subject and signature are standardized** for consistency.\n\n"
                    "You can configure:\n"
                    "- Supplier language (ES/EN/GR)\n"
                    "- Opening message (per language)\n"
                    "- CC / BCC rules per supplier"
                )

                # ----------------------------
                # Form (language + per-language openings + cc/bcc)
                # ----------------------------
                with st.form("email_templates_form", clear_on_submit=False):
                    st.caption("Placeholders supported: {order_id}  {date}  {venue_name}")

                    lang_label_default = next((k for k, v in LANGS.items() if v == current_lang), "Español (ES)")
                    lang_label = st.selectbox(
                        "Supplier language",
                        options=list(LANGS.keys()),
                        index=list(LANGS.keys()).index(lang_label_default),
                        key=f"cfg_email_lang_{chosen_venue.id}",
                        help="This language is used for the opening + legal footer in outgoing supplier emails.",
                    )
                    lang_code = LANGS[lang_label]

                    cc = st.text_input(
                        "CC (comma-separated emails)",
                        value=current_cc,
                        key=f"cfg_email_cc_{chosen_venue.id}",
                        help="Example: ops@company.com, finance@company.com",
                    )
                    bcc = st.text_input(
                        "BCC (comma-separated emails)",
                        value=current_bcc,
                        key=f"cfg_email_bcc_{chosen_venue.id}",
                        help="Example: audit@company.com",
                    )

                    st.markdown("**Opening message (per language)**")
                    cab_es = st.text_area(
                        "Opening (ES)",
                        value=open_es,
                        height=110,
                        key=f"cfg_email_open_es_{chosen_venue.id}",
                    )
                    cab_en = st.text_area(
                        "Opening (EN)",
                        value=open_en,
                        height=110,
                        key=f"cfg_email_open_en_{chosen_venue.id}",
                    )
                    cab_gr = st.text_area(
                        "Opening (GR)",
                        value=open_gr,
                        height=110,
                        key=f"cfg_email_open_gr_{chosen_venue.id}",
                    )

                    saved = st.form_submit_button("💾 Save settings", type="primary")

                if saved:
                    try:
                        with get_auth_session() as s:
                            v = s.exec(
                                select(Venue).where(
                                    Venue.id == chosen_venue.id,
                                    Venue.account_id == acc["id"],
                                )
                            ).first()
                            if not v:
                                raise ValueError("Venue not found.")

                            # Save per-supplier language + CC/BCC
                            v.email_lang = (lang_code or "es").strip().lower()
                            v.email_cc = (cc or "").strip()
                            v.email_bcc = (bcc or "").strip()

                            # Save per-language openings
                            v.email_opening_tpl_es = (cab_es or "").strip()
                            v.email_opening_tpl_en = (cab_en or "").strip()
                            v.email_opening_tpl_gr = (cab_gr or "").strip()

                            # Enforce standardization: clear old customizable subject/closing if they exist
                            if hasattr(v, "email_subject_tpl"):
                                v.email_subject_tpl = None
                            if hasattr(v, "email_closing_tpl"):
                                v.email_closing_tpl = None

                            # Optional: keep legacy single-field opening in sync (if your send code uses it)
                            # This sets email_opening_tpl to the currently selected language opening.
                            if hasattr(v, "email_opening_tpl"):
                                chosen_open = {"es": v.email_opening_tpl_es, "en": v.email_opening_tpl_en, "gr": v.email_opening_tpl_gr}.get(v.email_lang, v.email_opening_tpl_es)
                                v.email_opening_tpl = (chosen_open or "").strip()

                            s.add(v)
                            s.commit()

                        invalidate_auth_caches()
                        st.success("Email settings saved ✅")
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

                # ----------------------------
                # Preview (realistic email)
                # ----------------------------
                st.divider()
                st.subheader("👁️ Preview (real email)")

                def _render_tpl_safe(tpl: str, *, order_id: int, date_str: str, venue_name: str) -> str:
                    try:
                        return (tpl or "").format(order_id=order_id, date=date_str, venue_name=venue_name)
                    except Exception:
                        return tpl or ""

                def _invoice_header_plain(v: Venue) -> str:
                    lines = []
                    if (v.name or "").strip():
                        lines.append(v.name.strip())
                    if (owner_name or "").strip():
                        lines.append(f"Attn: {owner_name.strip()}")
                    if (v.address or "").strip():
                        lines.append(f"Address: {v.address.strip()}")
                    if (v.tax_number or "").strip():
                        lines.append(f"Tax ID: {v.tax_number.strip()}")
                    if (v.email or "").strip():
                        lines.append(f"Email: {v.email.strip()}")
                    if (v.phone or "").strip():
                        lines.append(f"Phone: {v.phone.strip()}")
                    return "\n".join(lines).strip()

                def _invoice_header_html(v: Venue) -> str:
                    parts = []
                    if (v.name or "").strip():
                        parts.append(f"<div style='font-weight:700;font-size:16px;'>{v.name.strip()}</div>")
                    if (owner_name or "").strip():
                        parts.append(f"<div>Attn: {owner_name.strip()}</div>")
                    if (v.address or "").strip():
                        parts.append(f"<div>Address: {v.address.strip()}</div>")
                    if (v.tax_number or "").strip():
                        parts.append(f"<div>Tax ID: {v.tax_number.strip()}</div>")
                    if (v.email or "").strip():
                        parts.append(f"<div>Email: {v.email.strip()}</div>")
                    if (v.phone or "").strip():
                        parts.append(f"<div>Phone: {v.phone.strip()}</div>")

                    if not parts:
                        return "<div style='opacity:.7'>(missing venue data)</div>"

                    return (
                        "<div style='border:1px solid rgba(49,51,63,.15);"
                        "border-radius:12px;padding:12px;background:rgba(255,255,255,.03);'>"
                        + "".join(parts)
                        + "</div>"
                    )

                # Sample preview values
                from datetime import datetime
                sample_order_id = 1234
                sample_date = datetime.now().strftime("%Y-%m-%d")
                venue_name = (chosen_venue.name or "").strip()

                # Which language is active for preview?
                preview_lang = (getattr(chosen_venue, "email_lang", None) or current_lang or "es").strip().lower()
                if preview_lang not in {"es", "en", "gr"}:
                    preview_lang = "es"

                # Pick opening by language
                opening_by_lang = {
                    "es": getattr(chosen_venue, "email_opening_tpl_es", "") or DEFAULT_OPEN["es"],
                    "en": getattr(chosen_venue, "email_opening_tpl_en", "") or DEFAULT_OPEN["en"],
                    "gr": getattr(chosen_venue, "email_opening_tpl_gr", "") or DEFAULT_OPEN["gr"],
                }
                raw_opening = opening_by_lang.get(preview_lang, DEFAULT_OPEN["es"])

                preview_subject = build_subject(sample_order_id, venue_name, sample_date)
                preview_open = _render_tpl_safe(raw_opening, order_id=sample_order_id, date_str=sample_date, venue_name=venue_name)
                preview_close = build_closing(owner_name, preview_lang)
                preview_footer = build_legal_footer(chosen_venue, preview_lang)

                sample_lines_plain = "\n".join([
                    "• 3 caja — Cerveza Estrella",
                    "• 2 kg — Tomate pera",
                    "• 1 unidad — Aceite de oliva 5L",
                ]).strip()

                sample_lines_html = (
                    "<ul style='margin:8px 0 0 18px;padding:0;'>"
                    "<li>3 caja — Cerveza Estrella</li>"
                    "<li>2 kg — Tomate pera</li>"
                    "<li>1 unidad — Aceite de oliva 5L</li>"
                    "</ul>"
                )

                # Preview controls
                mode = st.radio("Preview format", ["Plain text", "HTML"], horizontal=True, key="email_tpl_preview_mode")

                to_email = (chosen_venue.email or "").strip() or "supplier@example.com"
                from_email = (u.get("email") or "").strip() or "your@email.com"
                cc_preview = (getattr(chosen_venue, "email_cc", "") or "").strip()
                bcc_preview = (getattr(chosen_venue, "email_bcc", "") or "").strip()

                # Card styles
                st.markdown(
                    """
                    <style>
           
                    .email-meta{display:grid;grid-template-columns:90px 1fr;gap:6px 12px;font-size:.92rem;opacity:.9;}
                    .email-meta b{opacity:.85;}
                    .email-body{margin-top:14px;border-top:1px solid rgba(49,51,63,.12);padding-top:14px;line-height:1.5;}
                    .muted{opacity:.7;}
                    .mono{white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace;}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                if mode == "Plain text":
                    header_plain = _invoice_header_plain(chosen_venue)
                    body_plain = "\n\n".join([
                        header_plain or "(missing venue data)",
                        preview_open.strip(),
                        "Pedido:",
                        sample_lines_plain,
                        preview_close.strip(),
                        "---",
                        preview_footer.strip(),
                    ]).strip()

                    st.markdown("<div class='email-card'>", unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <div class='email-meta'>
                        <b>To:</b><div>{to_email}</div>
                        <b>CC:</b><div>{cc_preview or "<span class='muted'>(none)</span>"}</div>
                        <b>BCC:</b><div>{bcc_preview or "<span class='muted'>(none)</span>"}</div>
                        <b>From:</b><div>{from_email}</div>
                        <b>Subject:</b><div>{preview_subject}</div>
                        <b>Lang:</b><div>{preview_lang.upper()}</div>
                        </div>
                        <div class='email-body mono'>{body_plain}</div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                else:
                    header_html = _invoice_header_html(chosen_venue)

                    footer_html = (
                        "<div style='margin-top:14px;border-top:1px solid rgba(49,51,63,.12);padding-top:10px;opacity:.85;font-size:.9rem;white-space:pre-wrap;'>"
                        + (preview_footer or "").replace("\n", "<br>")
                        + "</div>"
                    )

                    body_html = (
                        "<div style='white-space:pre-wrap;'>"
                        + (preview_open or "").replace("\n", "<br>")
                        + "</div>"
                        + "<div style='margin-top:10px;font-weight:600;'>Pedido:</div>"
                        + sample_lines_html
                        + "<div style='margin-top:12px;white-space:pre-wrap;'>"
                        + (preview_close or "").replace("\n", "<br>")
                        + "</div>"
                        + footer_html
                    )

                    st.markdown("<div class='email-card'>", unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <div class='email-meta'>
                        <b>To:</b><div>{to_email}</div>
                        <b>CC:</b><div>{cc_preview or "<span class='muted'>(none)</span>"}</div>
                        <b>BCC:</b><div>{bcc_preview or "<span class='muted'>(none)</span>"}</div>
                        <b>From:</b><div>{from_email}</div>
                        <b>Subject:</b><div>{preview_subject}</div>
                        <b>Lang:</b><div>{preview_lang.upper()}</div>
                        </div>
                        <div class='email-body'>
                        {header_html}
                        <div style='margin-top:12px;'>{body_html}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                st.caption("Rendered subject (copy/paste)")
                st.code(preview_subject, language="text")


    # ------------------------------------------------------------------
    # TAB 4: Providers (supplier directory)
    # ------------------------------------------------------------------
    
    
        # ------------------------------------------------------------------
    # TAB 4: Providers (supplier directory)
    # ------------------------------------------------------------------
    with org_tabs[3]:
        active_vid = st.session_state.get("active_venue_id")
        if active_vid is None:
            st.info("No active venue selected.")
            return

        venue_id = int(active_vid)

        # Check if catalog exists
        with get_domain_session() as s:
            any_product = s.exec(select(Product.id).where(Product.venue_id == venue_id).limit(1)).first()

        if not any_product:
            st.info("Catalog not uploaded yet. Upload products first to manage providers.")
            return

        st.markdown("### 📇 Supplier Directory")
        st.caption("Manage supplier details for the active venue.")

        # -----------------------------
        # Cached loaders
        # -----------------------------
        @st.cache_data(ttl=60)
        def load_providers_cached(venue_id: int) -> List[Provider]:
            with get_domain_session() as s:
                return s.exec(
                    select(Provider).where(Provider.venue_id == venue_id).order_by(Provider.name.asc())
                ).all()

        @st.cache_data(ttl=60)
        def load_provider_names_from_catalog(venue_id: int) -> List[str]:
            with get_domain_session() as s:
                prov_from_catalog = s.exec(
                    select(Product.provider_name).where(Product.venue_id == venue_id)
                ).all()
                return sorted({(p or "").strip() for p in prov_from_catalog if (p or "").strip()})

        @st.cache_data(ttl=60)
        def load_products_cached(venue_id: int) -> List[Product]:
            with get_domain_session() as s:
                return s.exec(
                    select(Product).where(Product.venue_id == venue_id).order_by(Product.name.asc())
                ).all()

        @st.cache_data(ttl=60)
        def load_rules_cached(venue_id: int, provider_id: int) -> List["ProviderDiscountRule"]:
            with get_domain_session() as s:
                return s.exec(
                    select(ProviderDiscountRule)
                    .where(
                        ProviderDiscountRule.venue_id == venue_id,
                        ProviderDiscountRule.provider_id == provider_id,
                    )
                    .order_by(ProviderDiscountRule.is_active.desc(), ProviderDiscountRule.min_qty.asc())
                ).all()

        # -----------------------------
        # Helper functions
        # -----------------------------
        def _norm_pipe_list(v: str) -> str:
            raw = (v or "").strip()
            if not raw:
                return ""
            parts = [p.strip() for p in raw.replace(";", "|").replace(",", "|").split("|")]
            parts = [p for p in parts if p]
            seen = set()
            out = []
            for p in parts:
                k = p.lower()
                if k in seen:
                    continue
                seen.add(k)
                out.append(p)
            return " | ".join(out)

        def _pipe_to_list(v: str) -> list[str]:
            v = (v or "").strip()
            if not v:
                return []
            return [p.strip() for p in v.split("|") if p.strip()]

        # -----------------------------
        # Load data
        # -----------------------------
        providers = load_providers_cached(venue_id)
        prov_names_from_catalog = load_provider_names_from_catalog(venue_id)

        existing_names = {(p.name or "").strip() for p in providers}
        missing_names = [n for n in prov_names_from_catalog if n not in existing_names]

        # Auto-create missing providers
        if missing_names:
            col1, col2 = st.columns([1.4, 2.6])
            with col1:
                if st.button(
                    f"➕ Create {len(missing_names)} providers from catalog",
                    key="prov_bootstrap_btn",
                    width="stretch",
                ):
                    with get_domain_session() as s:
                        for n in missing_names:
                            s.add(Provider(venue_id=venue_id, name=n))
                        s.commit()

                    # clear cached providers so selectbox updates
                    load_providers_cached.clear()
                    st.success("Providers created ✅")
                    time.sleep(0.5)
                    st.rerun()

            with col2:
                st.info("Automatically create providers that appear in your catalog.")

        # Reload after potential creation
        providers = load_providers_cached(venue_id)

        if not providers:
            st.warning("No providers found.")
            return

        # -----------------------------
        # Select provider
        # -----------------------------
        provider_by_name = {p.name: p for p in providers if (p.name or "").strip()}
        selected_name = st.selectbox(
            "Select provider",
            options=list(provider_by_name.keys()),
            key=f"prov_select_{venue_id}",
        )
        selected_provider = provider_by_name[selected_name]

        # Build lists for dropdowns
        emails_norm = _norm_pipe_list(selected_provider.emails or "")
        phones_norm = _norm_pipe_list(selected_provider.phones or "")
        email_list = _pipe_to_list(emails_norm)
        phone_list = _pipe_to_list(phones_norm)

        # Current marked contacts
        current_order_email = (selected_provider.order_email or "").strip()
        current_order_phone = (selected_provider.order_phone or "").strip()

        if current_order_email and current_order_email not in email_list:
            email_list = [current_order_email] + email_list
        if current_order_phone and current_order_phone not in phone_list:
            phone_list = [current_order_phone] + phone_list

        # -----------------------------
        # Edit provider form
        # -----------------------------
        with st.form(key=f"prov_form_{venue_id}_{selected_provider.id}", clear_on_submit=False):
            col1, col2 = st.columns(2)
            with col1:
                tax_number = st.text_input("Tax ID", value=selected_provider.tax_number or "")
            with col2:
                address = st.text_input("Address", value=selected_provider.address or "")

            emails_input = st.text_area(
                "Emails (separate with | , ;)",
                value=(selected_provider.emails or ""),
                height=80,
                help="Multiple emails separated by | , or ;",
            )
            phones_input = st.text_area(
                "Phones (separate with | , ;)",
                value=(selected_provider.phones or ""),
                height=80,
                help="Multiple phone numbers separated by | , or ;",
            )

            # Normalize inputs
            emails_norm_new = _norm_pipe_list(emails_input)
            phones_norm_new = _norm_pipe_list(phones_input)
            email_list_new = _pipe_to_list(emails_norm_new)
            phone_list_new = _pipe_to_list(phones_norm_new)

            st.divider()
            st.subheader("Order Contact")

            # Email for orders
            if email_list_new:
                selected_email_for_orders = st.selectbox(
                    "Primary email for orders",
                    options=email_list_new,
                    index=email_list_new.index(current_order_email) if current_order_email in email_list_new else 0,
                    key=f"prov_order_email_{venue_id}_{selected_provider.id}",
                )
                use_this_email = st.checkbox(
                    "Use this email for orders",
                    value=bool(current_order_email) and (current_order_email == selected_email_for_orders),
                    key=f"prov_order_email_tick_{venue_id}_{selected_provider.id}",
                )
            else:
                st.info("Add at least one email above to mark one for orders.")
                selected_email_for_orders = ""
                use_this_email = False

            # Phone for orders
            if phone_list_new:
                selected_phone_for_orders = st.selectbox(
                    "Primary phone for orders",
                    options=phone_list_new,
                    index=phone_list_new.index(current_order_phone) if current_order_phone in phone_list_new else 0,
                    key=f"prov_order_phone_{venue_id}_{selected_provider.id}",
                )
                use_this_phone = st.checkbox(
                    "Use this phone for orders",
                    value=bool(current_order_phone) and (current_order_phone == selected_phone_for_orders),
                    key=f"prov_order_phone_tick_{venue_id}_{selected_provider.id}",
                )
            else:
                st.info("Add at least one phone above to mark one for orders.")
                selected_phone_for_orders = ""
                use_this_phone = False

            submitted = st.form_submit_button("💾 Save provider", type="primary")

        if submitted:
            emails_final = emails_norm_new or None
            phones_final = phones_norm_new or None

            order_email_final = (selected_email_for_orders.strip() if use_this_email else None) or None
            order_phone_final = (selected_phone_for_orders.strip() if use_this_phone else None) or None

            # Validation
            if use_this_email and not order_email_final:
                st.warning("You selected to use email for orders but no email is selected.")
                st.stop()
            if use_this_phone and not order_phone_final:
                st.warning("You selected to use phone for orders but no phone is selected.")
                st.stop()

            with get_domain_session() as s:
                obj = s.exec(
                    select(Provider).where(
                        Provider.id == int(selected_provider.id),
                        Provider.venue_id == venue_id,
                    )
                ).first()

                if obj is None:
                    st.error("Provider not found (maybe it was deleted).")
                    st.stop()

                obj.tax_number = (tax_number or "").strip() or None
                obj.address = (address or "").strip() or None
                obj.emails = emails_final
                obj.phones = phones_final
                obj.order_email = order_email_final
                obj.order_phone = order_phone_final
                obj.updated_at = datetime.utcnow()

                s.add(obj)
                s.commit()

            # clear cached providers to show updated data in UI
            load_providers_cached.clear()

            st.success("Provider saved ✅")
            time.sleep(0.5)
            st.rerun()

        # ==============================================================
        # Discount rules (ALWAYS visible for the selected provider)
        # ==============================================================
        
        # ==============================================================
        # Discount rules (ALWAYS visible for the selected provider)
        # ==============================================================
        st.divider()
        st.markdown("### 🏷️ Discount rules")
        st.caption("Define discounts that apply only when certain conditions are met (e.g., qty ≥ 10).")
        st.info(f"Editing rules for: **{selected_provider.name}**")

        rule_kind_options = [
            "line_pct",
            "line_net_price",
            "prev_month_pct",
            "prev_month_net_price",
        ]

        # -----------------------------
        # Products dropdown: ONLY products from this provider
        # -----------------------------
        @st.cache_data(ttl=60)
        def load_provider_products_cached(venue_id: int, provider_name: str) -> List[Product]:
            provider_name = (provider_name or "").strip()
            with get_domain_session() as s:
                return s.exec(
                    select(Product)
                    .where(
                        Product.venue_id == venue_id,
                        Product.provider_name == provider_name,
                    )
                    .order_by(Product.name.asc())
                ).all()

        products_for_rules = load_provider_products_cached(venue_id, selected_provider.name)

        if not products_for_rules:
            st.warning("This provider has no products in the catalog (Product.provider_name).")
            st.caption("Tip: check your catalog upload — products must have provider_name exactly matching this provider.")
            prod_id_to_label: Dict[int, str] = {}
        else:
            # Stable labels (avoid duplicates): "Tomate triturado (#123)"
            prod_id_to_label = {
                int(p.id): f"{(p.name or '').strip() or '(no name)'} (#{int(p.id)})"
                for p in products_for_rules
                if p.id is not None
            }

        label_to_prod_id = {v: k for k, v in prod_id_to_label.items()}
        product_options = ["(all products)"] + sorted(label_to_prod_id.keys())

        # -----------------------------
        # Load existing rules for this provider
        # -----------------------------
        @st.cache_data(ttl=60)
        def load_rules_cached(venue_id: int, provider_id: int) -> List["ProviderDiscountRule"]:
            with get_domain_session() as s:
                return s.exec(
                    select(ProviderDiscountRule)
                    .where(
                        ProviderDiscountRule.venue_id == venue_id,
                        ProviderDiscountRule.provider_id == provider_id,
                    )
                    .order_by(
                        ProviderDiscountRule.is_active.desc(),
                        ProviderDiscountRule.rule_kind.asc(),
                        ProviderDiscountRule.min_qty.asc(),
                    )
                ).all()

        rules = load_rules_cached(venue_id, int(selected_provider.id))

        def _rule_label(r: "ProviderDiscountRule") -> str:
            rk = (getattr(r, "rule_kind", "") or "line_pct").strip()
            return {
                "line_pct": "Line % (min qty)",
                "line_net_price": "Line NET price (min qty)",
                "prev_month_pct": "Prev month % (threshold)",
                "prev_month_net_price": "Prev month NET price (threshold)",
            }.get(rk, rk)

        # -----------------------------
        # Visualize saved rules (readable summary)
        # -----------------------------
        if rules:
            summary_rows = []
            for r in rules:
                if r.product_id is None:
                    prod_txt = "(all products)"
                else:
                    prod_txt = prod_id_to_label.get(
                        int(r.product_id),
                        f"(product #{int(r.product_id)} not in this provider)"
                    )

                summary_rows.append(
                    {
                        "active": bool(getattr(r, "is_active", True)),
                        "rule_kind": _rule_label(r),
                        "product": prod_txt,
                        "min_qty": float(getattr(r, "min_qty", 0.0) or 0.0),
                        "prev_month_min_qty": float(getattr(r, "prev_month_min_qty", 0.0) or 0.0),
                        "discount_%": float(getattr(r, "discount_percent", 0.0) or 0.0),
                        "price_override": float(getattr(r, "price_override", 0.0) or 0.0),
                        "note": getattr(r, "note", "") or "",
                        "id": int(r.id) if r.id is not None else None,
                    }
                )

            with st.expander("👀 View existing saved rules", expanded=True):
                st.dataframe(
                    pd.DataFrame(summary_rows),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("No discount rules saved yet for this provider.")

        # -----------------------------
        # Build editor dataframe (shows saved rules too)
        # -----------------------------
        rows: List[Dict[str, Any]] = []
        for r in rules:
            # product label for editor must match selectbox options
            if r.product_id is None:
                prod_label = "(all products)"
            else:
                prod_label = prod_id_to_label.get(int(r.product_id), f"(unknown product #{int(r.product_id)})")

            rk = (getattr(r, "rule_kind", "") or "line_pct").strip()
            if rk not in rule_kind_options:
                rk = "line_pct"

            rows.append(
                {
                    "id": int(r.id) if r.id is not None else None,
                    "provider": selected_provider.name,
                    "active": bool(getattr(r, "is_active", True)),
                    "product": prod_label,
                    "rule_kind": rk,
                    "min_qty": float(getattr(r, "min_qty", 0.0) or 0.0),
                    "prev_month_min_qty": float(getattr(r, "prev_month_min_qty", 0.0) or 0.0),
                    "discount_%": float(getattr(r, "discount_percent", 0.0) or 0.0),
                    "price_override": float(getattr(r, "price_override", 0.0) or 0.0),
                    "note": getattr(r, "note", "") or "",
                }
            )

        df_rules = pd.DataFrame(rows)
        if df_rules.empty:
            df_rules = pd.DataFrame(
                [
                    {
                        "id": None,
                        "provider": selected_provider.name,
                        "active": True,
                        "product": "(all products)",
                        "rule_kind": "line_pct",
                        "min_qty": 0.0,
                        "prev_month_min_qty": 0.0,
                        "discount_%": 0.0,
                        "price_override": 0.0,
                        "note": "",
                    }
                ]
            )

        edited_rules = st.data_editor(
            df_rules,
            key=f"prov_rules_editor_{venue_id}_{selected_provider.id}",
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            column_config={
                "id": st.column_config.NumberColumn("id", disabled=True),
                "provider": st.column_config.TextColumn("provider", disabled=True),
                "active": st.column_config.CheckboxColumn("active"),
                "product": st.column_config.SelectboxColumn("product", options=product_options),
                "rule_kind": st.column_config.SelectboxColumn("rule_kind", options=rule_kind_options),
                "min_qty": st.column_config.NumberColumn("min_qty", min_value=0.0, step=1.0),
                "prev_month_min_qty": st.column_config.NumberColumn("prev_month_min_qty", min_value=0.0, step=1.0),
                "discount_%": st.column_config.NumberColumn("discount_%", min_value=0.0, max_value=100.0, step=0.5),
                "price_override": st.column_config.NumberColumn("price_override", min_value=0.0, step=0.1),
                "note": st.column_config.TextColumn("note"),
            },
        )

        csave, cinfo = st.columns([1, 3])
        with csave:
            save_rules = st.button(
                "💾 Save rules",
                type="primary",
                key=f"prov_rules_save_{venue_id}_{selected_provider.id}",
                width="stretch",
            )
        with cinfo:
            st.caption(
                "• line_* rules use min_qty (this order line)\n"
                "• prev_month_* rules use prev_month_min_qty (previous calendar month total)\n"
                "• net price rules use price_override"
            )

        if save_rules:
            cleaned: List[Dict[str, Any]] = []
            for _, row in edited_rules.iterrows():
                rid = row.get("id")
                rid_int = int(rid) if pd.notna(rid) and str(rid).strip() != "" else None

                prod_label = str(row.get("product") or "").strip() or "(all products)"
                if prod_label == "(all products)":
                    product_id = None
                else:
                    product_id = label_to_prod_id.get(prod_label)
                    if product_id is None:
                        st.error(f"Unknown product selection: {prod_label}")
                        st.stop()

                rk = str(row.get("rule_kind") or "line_pct").strip()
                if rk not in rule_kind_options:
                    st.error(f"Unknown rule_kind: {rk}")
                    st.stop()

                # read numbers safely
                min_qty = float(row.get("min_qty") or 0.0)
                prev_m = float(row.get("prev_month_min_qty") or 0.0)
                disc = float(row.get("discount_%") or 0.0)
                price_override = float(row.get("price_override") or 0.0)

                min_qty = max(0.0, min_qty)
                prev_m = max(0.0, prev_m)
                disc = max(0.0, min(100.0, disc))
                price_override = max(0.0, price_override)

                # -----------------------------
                # Validation by rule type
                # -----------------------------
                if rk.endswith("_pct"):
                    if disc <= 0:
                        st.error("For % rules, discount_% must be > 0.")
                        st.stop()
                    price_override_db = None
                else:
                    # net price
                    if price_override <= 0:
                        st.error("For net price rules, price_override must be > 0.")
                        st.stop()
                    disc = 0.0
                    price_override_db = price_override

                if rk.startswith("prev_month"):
                    if prev_m <= 0:
                        st.error("For prev_month rules, prev_month_min_qty must be > 0.")
                        st.stop()
                    prev_m_db = prev_m
                    # min_qty not meaningful here; keep 0 in DB
                    min_qty_db = 0.0
                else:
                    prev_m_db = None
                    min_qty_db = min_qty

                # Skip truly-empty placeholder rows (only for line_pct with 0s, etc.)
                if (
                    rid_int is None
                    and product_id is None
                    and rk == "line_pct"
                    and min_qty_db <= 0
                    and disc <= 0
                ):
                    continue

                cleaned.append(
                    {
                        "id": rid_int,
                        "is_active": bool(row.get("active")),
                        "product_id": product_id,
                        "rule_kind": rk,
                        "min_qty": float(min_qty_db),
                        "prev_month_min_qty": (float(prev_m_db) if prev_m_db is not None else None),
                        "discount_percent": float(disc),
                        "price_override": price_override_db,
                        "note": (str(row.get("note") or "").strip() or None),
                    }
                )

            existing_by_id = {int(r.id): r for r in rules if r.id is not None}
            keep_ids = {c["id"] for c in cleaned if c["id"] is not None}

            with get_domain_session() as s:
                # Delete removed
                for rid, obj in existing_by_id.items():
                    if rid not in keep_ids:
                        s.delete(obj)

                # Upsert rows
                for c in cleaned:
                    if c["id"] is None:
                        obj = ProviderDiscountRule(
                            venue_id=venue_id,
                            provider_id=int(selected_provider.id),
                            product_id=c["product_id"],
                            rule_kind=c["rule_kind"],
                            min_qty=float(c["min_qty"]),
                            prev_month_min_qty=c["prev_month_min_qty"],
                            discount_percent=float(c["discount_percent"]),
                            price_override=c["price_override"],
                            note=c["note"],
                            is_active=bool(c["is_active"]),
                            updated_at=datetime.utcnow(),
                        )
                        s.add(obj)
                    else:
                        obj = s.exec(
                            select(ProviderDiscountRule).where(
                                ProviderDiscountRule.id == int(c["id"]),
                                ProviderDiscountRule.venue_id == venue_id,
                                ProviderDiscountRule.provider_id == int(selected_provider.id),
                            )
                        ).first()

                        if not obj:
                            # very unlikely, but keep robust
                            obj = ProviderDiscountRule(
                                venue_id=venue_id,
                                provider_id=int(selected_provider.id),
                            )

                        obj.product_id = c["product_id"]
                        obj.rule_kind = c["rule_kind"]
                        obj.min_qty = float(c["min_qty"])
                        obj.prev_month_min_qty = c["prev_month_min_qty"]
                        obj.discount_percent = float(c["discount_percent"])
                        obj.price_override = c["price_override"]
                        obj.note = c["note"]
                        obj.is_active = bool(c["is_active"])
                        obj.updated_at = datetime.utcnow()
                        s.add(obj)

                s.commit()

            # Clear caches so you immediately SEE what you saved
            load_rules_cached.clear()
            load_provider_products_cached.clear()

            st.success("Rules saved ✅")
            time.sleep(0.4)
            st.rerun()
