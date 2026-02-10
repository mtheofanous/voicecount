"""
auth_multi_tenant.py
--------------------
Drop-in authentication + multi-tenant data model for your Streamlit + SQLModel app.
Optimized for performance with minimal reruns.
"""

from __future__ import annotations
import json
import os
import base64
import hmac
import hashlib
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, List, Tuple, Dict, Any
import time
import re
import pandas as pd
import streamlit as st
from sqlmodel import SQLModel, Field, Session, create_engine, select
from sqlalchemy import text, UniqueConstraint, inspect
from sqlalchemy.exc import IntegrityError
from core.mailer import send_smtp_email
from core.config import get_database_url
from domain.models import Product, Provider, ProviderDiscountRule, SeguimientoTicket, ProviderReceipt
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

    # Accountant contact (for exporting / emailing credit note reports)
    accountant_name: Optional[str] = Field(default=None)
    accountant_email: Optional[str] = Field(default=None)

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
    """
    Cross-database column existence check.
    - Postgres: uses SQLAlchemy inspector
    - SQLite: inspector also works, but keep PRAGMA fallback for safety
    """
    try:
        insp = inspect(conn)
        cols = insp.get_columns(table)
        return any(c.get("name") == column for c in cols)
    except Exception:
        # Fallback (SQLite only)
        rows = conn.execute(text(f"PRAGMA table_info('{table}')")).fetchall()
        existing = {row[1] for row in rows}
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


from sqlalchemy import text

def init_auth_db() -> None:
    """
    Creates tables and (ONLY for SQLite) adds missing columns on older DBs.
    Safe to call on every run.
    """
    # 1) Ensure tables exist (works for Postgres + SQLite)
    db_initialized = init_auth_db_once()
    if not db_initialized:
        st.error("Failed to initialize auth database")
        return

    # 2) Only run legacy ALTER TABLE migrations on SQLite
    with get_auth_engine().begin() as conn:
        dialect = conn.dialect.name  # "sqlite" | "postgresql" | ...

        if dialect == "sqlite":
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
                ("accountant_name", "ALTER TABLE venue ADD COLUMN accountant_name TEXT"),
                ("accountant_email", "ALTER TABLE venue ADD COLUMN accountant_email TEXT"),
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

        else:
            # Postgres/Supabase: no SQLite PRAGMA/ALTER hacks.
            # If you later need migrations, add Alembic.
            pass

    # 3) Backfill roles (works in both DBs)
    try:
        backfill_account_roles()
    except Exception:
        pass



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
    # Also clear persisted auth cookie (keeps VOI-tab <a href> navigation from logging users out)
    _cookie_del("voi_auth")
    # Clear auth caches
    _cached_user_dict.clear()
    _cached_account_dict.clear()
    _cached_venues_for_user.clear()




# -------------------------------
# Persistent auth (Streamlit Cloud-safe)
# - Streamlit session_state is tied to a websocket; plain <a href> navigation can create a new session.
# - We persist a signed token in a cookie so auth can be restored after a reconnect.
# -------------------------------

def _auth_secret() -> bytes:
    # Configure in Streamlit Cloud → App → Settings → Secrets:
    # AUTH_SECRET = "a-long-random-string"
    secret = (os.getenv("AUTH_SECRET") or "").strip()
    if not secret:
        # Dev fallback (NOT secure); set AUTH_SECRET in production.
        secret = "dev-insecure-secret-change-me"
    return secret.encode("utf-8")


def _sign(payload: str) -> str:
    return hmac.new(_auth_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _make_token(user_id: int, account_id: int, *, ttl_hours: int = 72) -> str:
    exp = int(time.time() + ttl_hours * 3600)
    data = {"user_id": int(user_id), "account_id": int(account_id), "exp": exp}
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    sig = _sign(payload)
    b64 = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")
    return f"{b64}.{sig}"


def _parse_token(token: str) -> Optional[dict]:
    try:
        b64, sig = token.split(".", 1)
        payload = base64.urlsafe_b64decode(b64.encode("ascii")).decode("utf-8")
        if not hmac.compare_digest(_sign(payload), sig):
            return None
        data = json.loads(payload)
        if int(data.get("exp", 0)) < int(time.time()):
            return None
        return data
    except Exception:
        return None


def _cookie_manager():
    """Return a CookieManager instance (client-side cookies via a component).

    Notes:
    - Streamlit's built-in st.context.cookies is READ-ONLY and, on Community Cloud,
      cookies are often filtered at the proxy layer, so it may be empty.
    - extra_streamlit_components.CookieManager works on Cloud because it interacts
      with cookies client-side via a Streamlit component.
    """
    try:
        import extra_streamlit_components as stx
    except Exception:
        return None

    cm_key = "__voi_cookie_manager__"
    if cm_key not in st.session_state:
        # Creating the component can trigger one rerun; we do it once.
        st.session_state[cm_key] = stx.CookieManager()
    return st.session_state[cm_key]


def _cookie_get(name: str) -> Optional[str]:
    cm = _cookie_manager()
    if cm is None:
        return None
    try:
        return cm.get(name)
    except Exception:
        return None


def _cookie_set(name: str, value: str, *, max_age_seconds: int = 72 * 3600) -> None:
    cm = _cookie_manager()
    if cm is None:
        return
    try:
        cm.set(name, value, max_age=max_age_seconds)
    except Exception:
        return


def _cookie_del(name: str) -> None:
    cm = _cookie_manager()
    if cm is None:
        return
    try:
        # CookieManager supports delete in recent versions; if not, overwrite with short expiry.
        if hasattr(cm, "delete"):
            cm.delete(name)
        else:
            cm.set(name, "", max_age=1)
    except Exception:
        return


def restore_auth_from_cookie() -> None:
    """Restore st.session_state['auth_ctx'] from a signed cookie, if present.

    This prevents 'random' logouts on Streamlit Cloud when the user navigates using
    plain HTML links (e.g., bottom VOI tabs using <a href="?page=...">).
    """
    if is_logged_in():
        return

    tok = _cookie_get("voi_auth")
    if not tok:
        return

    data = _parse_token(tok)
    if not data:
        return

    user_id = int(data["user_id"])
    account_id = int(data["account_id"])

    # Safety: user must exist and belong to the same account
    u = _cached_user_dict(user_id)
    if not u or int(u.get("account_id", 0)) != account_id:
        return

    _set_auth(user_id, account_id)

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
    show_venue_selector: bool = True,  # ✅ NEW
) -> None:
    """
    If logged out -> show Login/Sign up.
    If logged in -> show account + (optional) venue selector + optional org management.
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
                        _cookie_set("voi_auth", _make_token(user_id, account_id))
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
                        _cookie_set("voi_auth", _make_token(user_id, acc_id))
                        invalidate_auth_caches()
                        st.success("Account created ✅")
                        st.session_state[rerun_key] = True
                        time.sleep(0.5)
                        st.rerun()

                    except Exception as e:
                        st.error(str(e))

        st.stop()

    # -------------------------
    # Logged-in area
    # -------------------------
    u = current_user()
    acc = current_account()

    # Top bar
    top_left, top_right = st.columns([3, 1])
    with top_left:
        st.markdown(
            f"**Logged in:** {u['full_name'] if u else '—'}  •  **Account:** {acc['name'] if acc else '—'}"
        )

    with top_right:
        if st.button("Logout", key="logout_btn_gat", use_container_width=True):
            clear_auth()
            st.session_state[rerun_key] = True
            st.rerun()

    # -------------------------
    # Venue selector (OPTIONAL)
    # -------------------------
    if show_venue_selector:
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

            chosen_label = st.selectbox(
                "Active bar/restaurant",
                options=labels,
                index=idx,
                key="active_venue_select",
                on_change=lambda: None,
            )

            chosen_id = ids[labels.index(chosen_label)]
            if st.session_state.get("active_venue_id") != chosen_id:
                st.session_state["active_venue_id"] = chosen_id
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
                    manage_organization_ui(venue_role="owner")

                st.stop()
            else:
                st.info("Ask an admin to grant you access.")
                st.stop()

    # -------------------------
    # Org management (OPTIONAL)
    # -------------------------
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

# =========================================================
# Accountant exports (credit note report)
# =========================================================

_META_KV_RE = re.compile(r"\b([a-zA-Z_]+)=([^|]+)")


def _parse_supplier_meta(note: str) -> dict:
    """Parse the structured supplier note written by seguimiento.py.

    Expected examples:
      "[SUPPLIER] credit_note | credit_note_invoice=CN-12 | invoice=3423 | items=ABOKANTO:1TEM(missing_in_invoice)"
      "[SUPPLIER] supplementary_delivery | eta=2026-01-18 morning | items=..."
    """
    txt = (note or "").strip()
    out = {"solution": "", "credit_note": "", "invoice": "", "eta": "", "items": []}
    if not txt:
        return out

    low = txt.lower()
    if "credit_note" in low and "supplementary" not in low:
        out["solution"] = "credit_note"
    elif "supplementary_delivery" in low or "re_delivery" in low:
        out["solution"] = "re_delivery"

    # normalize separators to make parsing robust
    parts = [p.strip() for p in txt.replace("|", "|").split("|") if p.strip()]
    for p in parts:
        m = _META_KV_RE.search(p)
        if not m:
            continue
        k = (m.group(1) or "").strip().lower()
        v = (m.group(2) or "").strip()
        if k in {"credit_note_invoice", "credit_note"}:
            out["credit_note"] = v
        elif k in {"invoice", "invoice_number"}:
            out["invoice"] = v
        elif k == "eta":
            out["eta"] = v
        elif k == "items":
            # items format: "P1:1TEM(missing_in_invoice) | P2:2kg(damaged_wrong)" or with slashes
            raw = v
            raw = raw.replace("/", " | ")
            item_parts = [x.strip() for x in raw.split("|") if x.strip()]
            parsed = []
            for it in item_parts:
                # name:qtyunit(reason)
                # e.g. "ABOKANTO:1TEM(missing_in_invoice)"
                name = it
                qty = ""
                unit = ""
                reason = ""
                # reason
                if "(" in it and ")" in it:
                    try:
                        reason = it[it.rfind("(") + 1 : it.rfind(")")].strip()
                        it2 = it[: it.rfind("(")].strip()
                    except Exception:
                        it2 = it
                else:
                    it2 = it
                if ":" in it2:
                    name, rest = it2.split(":", 1)
                    name = name.strip()
                    rest = rest.strip()
                    # split numeric prefix from unit
                    m2 = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([a-zA-ZÀ-ÿ]+.*)?$", rest)
                    if m2:
                        qty = m2.group(1) or ""
                        unit = (m2.group(2) or "").strip()
                    else:
                        qty = rest
                parsed.append({"product": name.strip(), "qty": qty, "unit": unit, "reason": reason})
            out["items"] = parsed

    return out


def _build_credit_note_report_df(*, venue_id: int, date_from: datetime, date_to: datetime) -> pd.DataFrame:
    """
    One row per credit-note case (order_id + supplier) in the date range.
    Greek letters are preserved (unicode).
    """

    with get_domain_session() as s:
        tickets = list(
            s.exec(
                select(SeguimientoTicket)
                .where(SeguimientoTicket.venue_id == int(venue_id))
                .where(SeguimientoTicket.resolution_note.is_not(None))
            ).all()
        )

        receipts = list(
            s.exec(
                select(ProviderReceipt)
                .where(ProviderReceipt.venue_id == int(venue_id))
            ).all()
        )

    # Map provider receipts so we can get related invoice and invoice issue date
    receipt_map = {(int(r.order_id), (r.provider_name or "").strip()): r for r in receipts}

    # Group into one row per (order_id, supplier, credit_note_number, related_invoice)
    grouped: dict[tuple, dict] = {}

    for t in tickets:
        ts = getattr(t, "resolved_at", None) or getattr(t, "updated_at", None) or getattr(t, "created_at", None)
        if not ts:
            continue
        if ts < date_from or ts > date_to:
            continue

        meta = _parse_supplier_meta(getattr(t, "resolution_note", "") or "")
        if meta.get("solution") != "credit_note":
            continue

        order_id = int(getattr(t, "order_id", 0) or 0)
        prov = (getattr(t, "provider_name", "") or "").strip()

        receipt = receipt_map.get((order_id, prov))

        related_invoice = (
            (meta.get("invoice") or "").strip()
            or ((getattr(receipt, "invoice_number", None) or "").strip() if receipt else "")
            or "—"
        )

        # Invoice issue date: best effort from receipt.invoice_number_set_at (if present)
        inv_issue_dt = None
        if receipt is not None:
            inv_issue_dt = getattr(receipt, "invoice_number_set_at", None) or getattr(receipt, "created_at", None)

        inv_issue_str = inv_issue_dt.strftime("%Y-%m-%d") if hasattr(inv_issue_dt, "strftime") else "—"

        credit_note_no = (meta.get("credit_note") or "").strip() or "—"

        # Credit note issue date:
        # If supplier doesn't provide a separate date, we use the verification timestamp.
        cn_issue_str = ts.strftime("%Y-%m-%d")

        key = (order_id, prov, related_invoice, credit_note_no)

        # Build reason string with per-product quantities
        items = meta.get("items") or []
        if not items:
            # fallback to the ticket itself
            items = [{
                "product": (getattr(t, "product_name", "") or "Product").strip(),
                "qty": str(getattr(t, "qty_invoiced", "") or "").strip(),
                "unit": (getattr(t, "unit", "") or "").strip(),
                "reason": (getattr(t, "kind", "") or "").strip(),
            }]

        def _nice_reason(r: str) -> str:
            r = (r or "").strip()
            mapping = {
                "missing_in_invoice": "Missing (in invoice)",
                "missing_not_in_invoice": "Missing (not in invoice)",
                "damaged_wrong": "Damaged/Wrong",
                "invoice_discrepancy": "Missing (in invoice)",
                "operational_missing": "Missing (not in invoice)",
            }
            return mapping.get(r, r.replace("_", " ").title() if r else "Issue")

        reason_parts = []
        for it in items:
            pname = (it.get("product") or "").strip() or "Product"
            qty = (it.get("qty") or "").strip()
            unit = (it.get("unit") or "").strip()
            why = _nice_reason(it.get("reason") or "")
            # Greek product names stay intact here
            if qty:
                qtxt = f"{qty}{(' ' + unit) if unit else ''}".strip()
                reason_parts.append(f"{pname}: {qtxt} — {why}")
            else:
                reason_parts.append(f"{pname} — {why}")

        reason_str = " | ".join(reason_parts) if reason_parts else "—"

        if key not in grouped:
            grouped[key] = {
                "Verified at": ts.strftime("%Y-%m-%d %H:%M"),
                "Order ID": order_id,
                "Supplier": prov,
                "Related invoice #": related_invoice,
                "Invoice Issue Date": inv_issue_str,
                "Credit note Number": credit_note_no,
                "Credit Note Issue Date": cn_issue_str,
                "Reason": reason_str,
            }
        else:
            # If multiple tickets contribute, merge reasons (avoid duplicates)
            prev = grouped[key].get("Reason") or ""
            merged = prev.split(" | ") if prev else []
            for p in reason_parts:
                if p not in merged:
                    merged.append(p)
            grouped[key]["Reason"] = " | ".join(merged)

    df = pd.DataFrame(list(grouped.values()))
    if df.empty:
        return df

    # Order: newest first
    df = df.sort_values(["Verified at", "Order ID", "Supplier"], ascending=[False, False, True])
    return df


def _df_to_xlsx_bytes(df: pd.DataFrame, *, sheet_name: str = "credit_notes") -> bytes:
    import io
    from openpyxl.styles import Font, Alignment, PatternFill
    from openpyxl.utils import get_column_letter

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])

        ws = writer.book[writer.sheets[sheet_name[:31]].title]

        # Pretty header
        header_fill = PatternFill("solid", fgColor="1F4E79")  # dark blue
        header_font = Font(bold=True, color="FFFFFF")
        header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = header_align

        # Freeze header row and add filter
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        # Alignment for body
        body_align = Alignment(vertical="top", wrap_text=False)
        reason_wrap = Alignment(vertical="top", wrap_text=True)

        # Set column widths based on content length (unicode-safe)
        for col_idx, col_name in enumerate(df.columns, start=1):
            col_letter = get_column_letter(col_idx)
            # Calculate max length among header + values
            values = [str(col_name)]
            values += [str(v) for v in df[col_name].astype(str).fillna("").tolist()[:2000]]  # cap for speed
            max_len = max((len(v) for v in values), default=10)

            # Reason column wider
            if col_name == "Reason":
                width = min(max(40, max_len), 90)
            else:
                width = min(max(14, max_len + 2), 45)

            ws.column_dimensions[col_letter].width = width

            # Apply alignment
            for row_idx in range(2, ws.max_row + 1):
                cell = ws[f"{col_letter}{row_idx}"]
                cell.alignment = reason_wrap if col_name == "Reason" else body_align

        # Make rows a bit taller for wrapped Reason
        for r in range(2, ws.max_row + 1):
            ws.row_dimensions[r].height = 20

    return bio.getvalue()

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

    # st.tabs resets to the first tab on any rerun.
    # Use a radio selector with session_state so saving stays on the same tab.
    org_tab_labels = ["👤 Users", "🏪 Venues", "✉️ Email Templates", "📇 Providers"]
    st.session_state.setdefault("org_tab", org_tab_labels[0])
    if st.session_state["org_tab"] not in org_tab_labels:
        st.session_state["org_tab"] = org_tab_labels[0]

    org_selected = st.radio(
        "Organization",
        org_tab_labels,
        horizontal=True,
        key="org_tab",
        label_visibility="collapsed",
    )

    # ------------------------------------------------------------------
    # USERS
    # ------------------------------------------------------------------
    if org_selected == "👤 Users":
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

                # Wrap editing in a form so changing selectors doesn't rerun the whole page.
                with st.form("mx_permissions_form", clear_on_submit=False):
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

                    b1, b2 = st.columns(2)
                    with b1:
                        save_clicked = st.form_submit_button("💾 Save permission", type="primary", use_container_width=True)
                    with b2:
                        remove_clicked = st.form_submit_button("🗑️ Remove access", use_container_width=True)

                if save_clicked:
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
                        time.sleep(0.3)
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

                if remove_clicked:
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
                                time.sleep(0.3)
                                st.rerun()
                            else:
                                st.info("No access to remove.")
                    except Exception as e:
                        st.error(str(e))

    # ------------------------------------------------------------------
    # VENUES
    # ------------------------------------------------------------------
    if org_selected == "🏪 Venues":
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

            
            # -------------------------------------------------
            # Accountant expander (store contact + export/send)
            # -------------------------------------------------
            with st.expander("🧾 Accountant", expanded=False):
                st.caption("Save your accountant contact, generate an Excel for credit notes, download it, or email it directly.")

                # Reload latest venue record (avoid stale cached object)
                with get_auth_session() as s:
                    vdb = s.exec(
                        select(Venue).where(Venue.id == chosen_venue.id, Venue.account_id == acc["id"])
                    ).first()

                if not vdb:
                    st.error("Venue not found.")
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        acc_name = st.text_input(
                            "Accountant name",
                            value=(getattr(vdb, "accountant_name", None) or ""),
                            key=f"acct_name_{vdb.id}",
                        )
                    with c2:
                        acc_email = st.text_input(
                            "Accountant email",
                            value=(getattr(vdb, "accountant_email", None) or ""),
                            key=f"acct_email_{vdb.id}",
                        )

                    if st.button("💾 Save accountant", type="primary", use_container_width=True, key=f"acct_save_{vdb.id}"):
                        try:
                            with get_auth_session() as s:
                                vv = s.exec(
                                    select(Venue).where(Venue.id == vdb.id, Venue.account_id == acc["id"])
                                ).first()
                                if not vv:
                                    raise ValueError("Venue not found.")
                                vv.accountant_name = (acc_name or "").strip() or None
                                vv.accountant_email = (acc_email or "").strip().lower() or None
                                s.add(vv)
                                s.commit()
                            invalidate_auth_caches()
                            st.success("Accountant saved ✅")
                        except Exception as e:
                            st.error(str(e))

                    st.divider()
                    st.markdown("### Credit note Excel")

                    # Date range
                    # Date range
                    today = datetime.utcnow().date()
                    d_from = st.date_input(
                        "From",
                        value=date(today.year, today.month, 1),
                        key=f"acct_from_{vdb.id}",
                    )
                    d_to = st.date_input(
                        "To",
                        value=today,
                        key=f"acct_to_{vdb.id}",
                    )
                    # Normalize range to datetimes (inclusive)
                    dt_from = datetime.combine(d_from, datetime.min.time())
                    dt_to = datetime.combine(d_to, datetime.max.time())

                    gen_key = f"acct_xlsx_{vdb.id}_{d_from.isoformat()}_{d_to.isoformat()}"

                    colg1, colg2 = st.columns([1, 1])
                    with colg1:
                        do_gen = st.button("📄 Generate Excel", key=f"acct_gen_{vdb.id}", use_container_width=True)
                    with colg2:
                        do_send = st.button("✉️ Email accountant", key=f"acct_send_{vdb.id}", use_container_width=True)

                    if do_gen:
                        df = _build_credit_note_report_df(venue_id=int(vdb.id), date_from=dt_from, date_to=dt_to)
                        if df.empty:
                            st.info("No credit notes found in this date range.")
                            st.session_state.pop(gen_key, None)
                        else:
                            xbytes = _df_to_xlsx_bytes(df)
                            st.session_state[gen_key] = {
                                "bytes": xbytes,
                                "rows": int(len(df)),
                                "preview": df.head(50),
                            }
                            st.success(f"Generated ✅ ({len(df)} rows)")

                    payload = st.session_state.get(gen_key)
                    if payload:
                        st.caption(f"Rows: {payload.get('rows', 0)}")
                        st.dataframe(payload.get("preview"), use_container_width=True, hide_index=True)

                        filename = f"credit_notes_{vdb.name}_{d_from.isoformat()}_{d_to.isoformat()}.xlsx".replace(" ", "_")
                        st.download_button(
                            "⬇️ Download Excel",
                            data=payload.get("bytes") or b"",
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key=f"acct_dl_{vdb.id}",
                        )

                    if do_send:
                        # Ensure we have bytes to attach
                        payload = st.session_state.get(gen_key)
                        if not payload:
                            df = _build_credit_note_report_df(venue_id=int(vdb.id), date_from=dt_from, date_to=dt_to)
                            if df.empty:
                                st.info("No credit notes found in this date range.")
                            else:
                                st.session_state[gen_key] = {
                                    "bytes": _df_to_xlsx_bytes(df),
                                    "rows": int(len(df)),
                                    "preview": df.head(50),
                                }
                                payload = st.session_state.get(gen_key)

                        acct_to = (getattr(vdb, "accountant_email", None) or acc_email or "").strip()
                        if not acct_to or "@" not in acct_to:
                            st.error("Please set a valid accountant email above.")
                        elif not payload:
                            st.error("Nothing to send.")
                        else:
                            filename = f"credit_notes_{vdb.name}_{d_from.isoformat()}_{d_to.isoformat()}.xlsx".replace(" ", "_")
                            subject = f"Credit notes to book — {vdb.name} — {d_from.isoformat()} to {d_to.isoformat()}"
                            body = (
                                f"Hi{(' ' + (acc_name or '').strip()) if (acc_name or '').strip() else ''},\n\n"
                                f"Attached are the confirmed credit notes for {vdb.name} for the period {d_from.isoformat()} to {d_to.isoformat()}.\n"
                                "Each row includes supplier, related invoice number, credit note number, and credited items/quantities.\n\n"
                                "Thanks."
                            ).strip()

                            try:
                                send_smtp_email(
                                    to=[acct_to],
                                    subject=subject,
                                    text_body=body,
                                    reply_to=(u.get("email") or None),
                                    attachments=[(
                                        filename,
                                        payload.get("bytes") or b"",
                                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    )],
                                )
                                st.success("Email sent ✅")
                            except Exception as e:
                                st.error(f"Email failed: {e}")
        else:
            st.info("Only account owners can manage venues.")

    # ------------------------------------------------------------------
    # TAB 3: Email Templates
    # ------------------------------------------------------------------
    if org_selected == "✉️ Email Templates":
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

    # -----------------------------
    # Provider helpers
    # -----------------------------
    import json
    import re


    DAY_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    DAY_LABELS = {
        "mon": "Mon", "tue": "Tue", "wed": "Wed", "thu": "Thu",
        "fri": "Fri", "sat": "Sat", "sun": "Sun"
    }
    TIME_RANGE_RE = re.compile(r"^\d{2}:\d{2}-\d{2}:\d{2}$")

    PRESET_SLOTS = ["08:00-14:00", "16:00-20:00"]

    def parse_delivery_schedule_json(raw: str) -> Dict[str, List[str]]:
        raw = (raw or "").strip()
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except Exception:
            return {}
        out: Dict[str, List[str]] = {}
        for day, slots in (data or {}).items():
            d = (str(day or "").strip().lower())
            if d not in DAY_KEYS:
                continue
            clean = []
            for s in (slots or []):
                s = str(s or "").strip()
                if TIME_RANGE_RE.match(s) and s not in clean:
                    clean.append(s)
            if clean:
                out[d] = clean
        return out

    def dump_delivery_schedule_json(schedule: Dict[str, List[str]]) -> str:
        clean: Dict[str, List[str]] = {}
        for d in DAY_KEYS:
            slots = schedule.get(d) or []
            valid = []
            for s in slots:
                s = str(s or "").strip()
                if TIME_RANGE_RE.match(s) and s not in valid:
                    valid.append(s)
            if valid:
                clean[d] = valid
        return json.dumps(clean, ensure_ascii=False)


    
        # ------------------------------------------------------------------
    # TAB 4: Providers (supplier directory)
    # ------------------------------------------------------------------
    if org_selected == "📇 Providers":
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
        # -----------------------------
        # Edit provider form
        # -----------------------------
        form_key = f"prov_form_{venue_id}_{selected_provider.id}"

        # load current schedule from provider
        current_schedule = parse_delivery_schedule_json(getattr(selected_provider, "delivery_schedule_json", None))

        with st.form(key=form_key, clear_on_submit=False):
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

            # Preferred contacts (only if list has items)
            c3, c4 = st.columns(2)
            with c3:
                order_email = st.selectbox(
                    "Order email (preferred)",
                    options=([""] + email_list_new) if email_list_new else [""],
                    index=([""] + email_list_new).index(current_order_email)
                    if (email_list_new and current_order_email in ([""] + email_list_new))
                    else 0,
                    help="This email will be used for sending order / incidence emails.",
                )
            with c4:
                order_phone = st.selectbox(
                    "Order phone (preferred)",
                    options=([""] + phone_list_new) if phone_list_new else [""],
                    index=([""] + phone_list_new).index(current_order_phone)
                    if (phone_list_new and current_order_phone in ([""] + phone_list_new))
                    else 0,
                    help="This phone can be used for WhatsApp / calls.",
                )

            st.markdown("---")
            st.markdown("### 🚚 Delivery schedule")
            st.caption("Choose delivery days and time windows (e.g., 08:00-14:00, 16:00-20:00).")

            # pick days
            default_days = [d for d in DAY_KEYS if d in current_schedule]
            days_selected = st.multiselect(
                "Delivery days",
                options=DAY_KEYS,
                default=default_days,
                format_func=lambda d: DAY_LABELS.get(d, d),
            )

            # build schedule from UI
            schedule_new: Dict[str, List[str]] = {}

            # quick add preset to all selected days
            preset_to_all = st.multiselect(
                "Quick add slots to ALL selected days (optional)",
                options=PRESET_SLOTS,
                default=[],
            )

            for d in days_selected:
                st.markdown(f"**{DAY_LABELS.get(d, d)}**")

                # merge presets + existing unique
                existing_slots = list(current_schedule.get(d, []))
                merged_default = []
                for s in (existing_slots + preset_to_all):
                    if s not in merged_default:
                        merged_default.append(s)

                slots = st.multiselect(
                    f"Time slots for {DAY_LABELS.get(d, d)}",
                    options=sorted(set(PRESET_SLOTS + merged_default)),
                    default=merged_default,
                    key=f"slots_{venue_id}_{selected_provider.id}_{d}",
                )

                # custom slot input
                custom = st.text_input(
                    f"Add custom slot for {DAY_LABELS.get(d, d)} (format HH:MM-HH:MM)",
                    key=f"customslot_{venue_id}_{selected_provider.id}_{d}",
                    placeholder="e.g. 06:30-10:30",
                ).strip()

                if custom:
                    if TIME_RANGE_RE.match(custom):
                        if custom not in slots:
                            slots = slots + [custom]
                            st.caption(f"✅ Added: {custom}")
                    else:
                        st.error(f"Invalid time slot: {custom} (use HH:MM-HH:MM)")

                # final slots for day
                clean_slots = []
                for s in slots:
                    s = str(s or "").strip()
                    if TIME_RANGE_RE.match(s) and s not in clean_slots:
                        clean_slots.append(s)

                if clean_slots:
                    schedule_new[d] = clean_slots

                st.markdown("")

            # serialize to JSON for DB
            delivery_schedule_json_new = dump_delivery_schedule_json(schedule_new) if schedule_new else None

            btn1, btn2, btn3 = st.columns([1.2, 1.2, 2.6])
            with btn1:
                submitted = st.form_submit_button("💾 Save", use_container_width=True)
            with btn2:
                deleted = st.form_submit_button("🗑️ Delete", use_container_width=True)
            with btn3:
                st.caption("Save updates supplier details for this venue.")


        # -----------------------------
        # Handle form actions (outside form)
        # -----------------------------
        if submitted:
            with get_domain_session() as s:
                p = s.exec(select(Provider).where(Provider.id == int(selected_provider.id))).first()
                if p:
                    p.tax_number = (tax_number or "").strip() or None
                    p.address = (address or "").strip() or None
                    p.emails = emails_norm_new or None
                    p.phones = phones_norm_new or None
                    p.order_email = (order_email or "").strip() or None
                    p.order_phone = (order_phone or "").strip() or None
                    p.delivery_schedule_json = delivery_schedule_json_new
                    p.updated_at = datetime.utcnow()
                    s.add(p)
                    s.commit()

            load_providers_cached.clear()
            st.success("Saved ✅")
            st.rerun()

        if deleted:
            with get_domain_session() as s:
                p = s.exec(select(Provider).where(Provider.id == int(selected_provider.id))).first()
                if p:
                    s.delete(p)
                    s.commit()

            load_providers_cached.clear()
            st.warning("Deleted")
            st.rerun()

        st.divider()
        st.markdown("### 🏷️ Discount rules")
        st.caption("Define discounts that apply only when certain conditions are met.")

        # Friendly explanations for non-technical users
        with st.expander("ℹ️ How discount rules work", expanded=False):
            st.markdown(
                """
        ### 🏷️ What is a discount rule?

        A discount rule tells the system **when a supplier gives you a better price**  
        and **what kind of discount you get**.

        Think of it as:  
        👉 *“If I buy **this much**, the supplier charges me **this price**.”*

        ---

        ### 📦 Delivery-based rules (this delivery)

        **1) Percentage discount (per product / per line)**  
        > “If I order at least **X units** of this product **today**, I get **Y% off**.”

        **2) Fixed NET unit price (per product / per line)**  
        > “If I order at least **X units** of this product **today**, the unit price becomes **€Z NET**.”

        ---

        ### 📊 Monthly-based rules (last month’s volume)

        **3) Percentage discount (based on last month)**  
        > “If last month I bought at least **X units total**, I get **Y% off** now.”

        **4) Fixed NET unit price (based on last month)**  
        > “If last month I bought at least **X units total**, the unit price becomes **€Z NET** now.”

        ---

        ### 🎯 Which products does the rule apply to?

        - **All products** → applies to everything from this supplier  
        - **Single product** → applies only to the selected product
                """
            )

            # ✅ Example boxes (real-world)
            st.info("**Example (delivery-based %):** Order **10 tomatoes** → rule says **10% off** → price becomes **€0.90** instead of **€1.00**.")
            st.info("**Example (delivery-based NET):** Order **10 tomatoes** → rule says **NET €1.20** → unit price becomes **€1.20** (ignores % discounts).")
            st.info("**Example (monthly-based %):** Last month you bought **120 units total** → rule says **5% off** → your price today is **5% cheaper**.")
            st.info("**Example (monthly-based NET):** Last month you bought **120 units total** → rule says **NET €0.80** → unit price becomes **€0.80**.")



        st.info(f"Editing rules for: **{selected_provider.name}**")

        rule_kind_options = [
            "line_pct",
            "line_net_price",
            "prev_month_pct",
            "prev_month_net_price",
        ]

        def _kind_human(rk: str) -> str:
            rk = (rk or "").strip()
            return {
                "line_pct": "Line % (min qty)",
                "line_net_price": "Line NET price (min qty)",
                "prev_month_pct": "Prev month % (threshold)",
                "prev_month_net_price": "Prev month NET price (threshold)",
            }.get(rk, rk)

        def _kind_desc(rk: str) -> str:
            rk = (rk or "").strip()
            return {
                "line_pct": "If qty on this line ≥ min qty → apply a % discount.",
                "line_net_price": "If qty on this line ≥ min qty → override unit price (NET).",
                "prev_month_pct": "If last month total qty ≥ threshold → apply a % discount.",
                "prev_month_net_price": "If last month total qty ≥ threshold → override unit price (NET).",
            }.get(rk, "")
            
        def _kind_icon(rk: str) -> str:
            rk = (rk or "").strip()
            return "📊" if rk.startswith("prev_month") else "📦"


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
            st.caption("Tip: products must have provider_name exactly matching this provider.")
            prod_id_to_label: Dict[int, str] = {}
        else:
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

        # -----------------------------
        # Add new rule (modern form)
        # -----------------------------
        with st.expander("➕ Add a new discount rule", expanded=(len(rules) == 0)):
            with st.form(key=f"add_rule_{venue_id}_{selected_provider.id}"):
                c1, c2 = st.columns([1.25, 1])
                with c1:
                    rk = st.selectbox(
                        "Rule type",
                        rule_kind_options,
                        format_func=_kind_human,
                    )
                    st.caption(_kind_desc(rk))
                with c2:
                    prod_label = st.selectbox(
                        "Applies to",
                        product_options,
                        help="Choose (all products) or a specific product.",
                    )

                c3, c4 = st.columns([1, 1])
                with c3:
                    if rk.startswith("prev_month"):
                        threshold = st.number_input("Prev month threshold (qty)", min_value=1.0, step=1.0, value=1.0)
                        min_qty = 0.0
                    else:
                        min_qty = st.number_input("Min qty (this order line)", min_value=1.0, step=1.0, value=1.0)
                        threshold = 0.0

                with c4:
                    if rk.endswith("_pct"):
                        disc = st.number_input("Discount %", min_value=0.5, max_value=100.0, step=0.5, value=5.0)
                        price_override = 0.0
                    else:
                        price_override = st.number_input("NET unit price override", min_value=0.01, step=0.1, value=1.0)
                        disc = 0.0

                note = st.text_input("Note (optional)")
                is_active = st.checkbox("Active", value=True)
                submitted = st.form_submit_button("Create rule", type="primary")

            if submitted:
                # map product label
                if prod_label == "(all products)":
                    product_id = None
                else:
                    product_id = label_to_prod_id.get(prod_label)
                    if product_id is None:
                        st.error(f"Unknown product selection: {prod_label}")
                        st.stop()

                rk_db = (rk or "line_pct").strip()
                if rk_db not in rule_kind_options:
                    st.error(f"Unknown rule_kind: {rk_db}")
                    st.stop()

                # validation
                if rk_db.endswith("_pct") and disc <= 0:
                    st.error("For % rules, Discount % must be > 0.")
                    st.stop()
                if rk_db.endswith("_net_price") and price_override <= 0:
                    st.error("For NET price rules, NET unit price override must be > 0.")
                    st.stop()
                if rk_db.startswith("prev_month") and threshold <= 0:
                    st.error("For prev month rules, threshold must be > 0.")
                    st.stop()
                if (not rk_db.startswith("prev_month")) and min_qty <= 0:
                    st.error("For line rules, min qty must be > 0.")
                    st.stop()

                with get_domain_session() as s:
                    obj = ProviderDiscountRule(
                        venue_id=venue_id,
                        provider_id=int(selected_provider.id),
                        product_id=product_id,
                        rule_kind=rk_db,
                        min_qty=float(min_qty if not rk_db.startswith("prev_month") else 0.0),
                        prev_month_min_qty=float(threshold) if rk_db.startswith("prev_month") else None,
                        discount_percent=float(disc) if rk_db.endswith("_pct") else 0.0,
                        price_override=float(price_override) if rk_db.endswith("_net_price") else None,
                        note=(note or "").strip() or None,
                        is_active=bool(is_active),
                        updated_at=datetime.utcnow(),
                    )
                    s.add(obj)
                    s.commit()

                load_rules_cached.clear()
                st.success("Rule created ✅")
                time.sleep(0.2)
                st.rerun()

        # -----------------------------
        # Existing rules (modern cards)
        # -----------------------------
        if not rules:
            st.info("No discount rules saved yet for this provider.")
        else:
            st.markdown("#### Existing rules")

            def _scope_text(r: ProviderDiscountRule) -> str:
                if getattr(r, "product_id", None) is None:
                    return "(all products)"
                pid = int(getattr(r, "product_id"))
                return prod_id_to_label.get(pid, f"Product #{pid}")

            def _trigger_text(r: ProviderDiscountRule) -> str:
                rk = (getattr(r, "rule_kind", "") or "line_pct").strip()
                if rk.startswith("prev_month"):
                    th = float(getattr(r, "prev_month_min_qty", 0.0) or 0.0)
                    return f"Last month ≥ {th:g} units"
                mq = float(getattr(r, "min_qty", 0.0) or 0.0)
                return f"This line ≥ {mq:g} units"

            def _effect_text(r: ProviderDiscountRule) -> str:
                rk = (getattr(r, "rule_kind", "") or "line_pct").strip()
                if rk.endswith("_pct"):
                    d = float(getattr(r, "discount_percent", 0.0) or 0.0)
                    return f"Discount {d:g}%"
                p = float(getattr(r, "price_override", 0.0) or 0.0)
                return f"NET €{p:,.2f} / unit"

            for r in rules:
                rid = int(r.id) if getattr(r, "id", None) is not None else 0
                rk = (getattr(r, "rule_kind", "") or "line_pct").strip()

                active = bool(getattr(r, "is_active", True))
                scope = _scope_text(r)
                trig = _trigger_text(r)
                eff = _effect_text(r)
                note = (getattr(r, "note", None) or "").strip()

                # Card header
                with st.container(border=True):
                    topc1, topc2, topc3 = st.columns([2.2, 1.2, 1.1], vertical_alignment="center")
                    with topc1:
                        st.markdown(f"**{_kind_icon(rk)} {_kind_human(rk)}**")

                        st.caption(_kind_desc(rk))
                    with topc2:
                        st.write("**Scope**")
                        st.write(scope)
                    with topc3:
                        st.write("**Status**")
                        st.write("✅ Active" if active else "⏸️ Inactive")

                    st.markdown(
                        f"- **Trigger:** {trig}\n- **Effect:** {eff}"
                        + (f"\n- **Note:** {note}" if note else "")
                    )

                    with st.expander("✏️ Edit this rule", expanded=False):
                        with st.form(key=f"edit_rule_{venue_id}_{selected_provider.id}_{rid}"):
                            c1, c2 = st.columns([1.25, 1])
                            with c1:
                                rk_new = st.selectbox(
                                    "Rule type",
                                    rule_kind_options,
                                    index=rule_kind_options.index(rk) if rk in rule_kind_options else 0,
                                    format_func=_kind_human,
                                )
                                st.caption(f"{_kind_icon(rk_new)} {_kind_desc(rk_new)}")
                            with c2:
                                scope_default = scope if scope in product_options else "(all products)"
                                prod_new = st.selectbox("Applies to", product_options, index=product_options.index(scope_default))

                            c3, c4 = st.columns([1, 1])
                            with c3:
                                if rk_new.startswith("prev_month"):
                                    th0 = float(getattr(r, "prev_month_min_qty", 0.0) or 0.0) or 1.0
                                    threshold_new = st.number_input("Prev month threshold (qty)", min_value=1.0, step=1.0, value=float(th0))
                                    min_qty_new = 0.0
                                else:
                                    mq0 = float(getattr(r, "min_qty", 0.0) or 0.0) or 1.0
                                    min_qty_new = st.number_input("Min qty (this order line)", min_value=1.0, step=1.0, value=float(mq0))
                                    threshold_new = 0.0

                            with c4:
                                if rk_new.endswith("_pct"):
                                    d0 = float(getattr(r, "discount_percent", 0.0) or 0.0) or 1.0
                                    disc_new = st.number_input("Discount %", min_value=0.5, max_value=100.0, step=0.5, value=float(d0))
                                    price_new = 0.0
                                else:
                                    p0 = float(getattr(r, "price_override", 0.0) or 0.0) or 1.0
                                    price_new = st.number_input("NET unit price override", min_value=0.01, step=0.1, value=float(p0))
                                    disc_new = 0.0

                            note_new = st.text_input("Note (optional)", value=note)
                            active_new = st.checkbox("Active", value=active)

                            bsave, bdel = st.columns([1, 1])
                            save_clicked = bsave.form_submit_button("Save changes", type="primary")
                            delete_clicked = bdel.form_submit_button("Delete rule")

                        if save_clicked:
                            # map product label
                            if prod_new == "(all products)":
                                product_id_new = None
                            else:
                                product_id_new = label_to_prod_id.get(prod_new)
                                if product_id_new is None:
                                    st.error(f"Unknown product selection: {prod_new}")
                                    st.stop()

                            # validation
                            if rk_new.endswith("_pct") and disc_new <= 0:
                                st.error("For % rules, Discount % must be > 0.")
                                st.stop()
                            if rk_new.endswith("_net_price") and price_new <= 0:
                                st.error("For NET price rules, NET unit price override must be > 0.")
                                st.stop()
                            if rk_new.startswith("prev_month") and threshold_new <= 0:
                                st.error("For prev month rules, threshold must be > 0.")
                                st.stop()
                            if (not rk_new.startswith("prev_month")) and min_qty_new <= 0:
                                st.error("For line rules, min qty must be > 0.")
                                st.stop()

                            with get_domain_session() as s:
                                obj = s.exec(
                                    select(ProviderDiscountRule).where(
                                        ProviderDiscountRule.id == int(rid),
                                        ProviderDiscountRule.venue_id == venue_id,
                                        ProviderDiscountRule.provider_id == int(selected_provider.id),
                                    )
                                ).first()
                                if not obj:
                                    st.error("Rule not found.")
                                    st.stop()

                                obj.product_id = product_id_new
                                obj.rule_kind = rk_new
                                obj.min_qty = float(min_qty_new if not rk_new.startswith("prev_month") else 0.0)
                                obj.prev_month_min_qty = float(threshold_new) if rk_new.startswith("prev_month") else None
                                obj.discount_percent = float(disc_new) if rk_new.endswith("_pct") else 0.0
                                obj.price_override = float(price_new) if rk_new.endswith("_net_price") else None
                                obj.note = (note_new or "").strip() or None
                                obj.is_active = bool(active_new)
                                obj.updated_at = datetime.utcnow()
                                s.add(obj)
                                s.commit()

                            load_rules_cached.clear()
                            st.success("Rule updated ✅")
                            time.sleep(0.2)
                            st.rerun()

                        if delete_clicked:
                            with get_domain_session() as s:
                                obj = s.exec(
                                    select(ProviderDiscountRule).where(
                                        ProviderDiscountRule.id == int(rid),
                                        ProviderDiscountRule.venue_id == venue_id,
                                        ProviderDiscountRule.provider_id == int(selected_provider.id),
                                    )
                                ).first()
                                if obj:
                                    s.delete(obj)
                                    s.commit()

                            load_rules_cached.clear()
                            st.success("Rule deleted ✅")
                            time.sleep(0.2)
                            st.rerun()
