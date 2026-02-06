# core/db.py
from __future__ import annotations

import streamlit as st
from contextlib import contextmanager
from sqlmodel import Session, create_engine
from core.config import get_database_url


@st.cache_resource(show_spinner=False)
def get_engine(db_url: str):
    """
    Create ONE SQLAlchemy engine per DATABASE_URL.
    Cached per Streamlit server process.
    """
    return create_engine(
        db_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=5,        # good default for Supabase
        max_overflow=5,     # allow short spikes
        pool_timeout=30,    # seconds
    )


@contextmanager
def get_session() -> Session:
    """
    Context-managed DB session.

    Guarantees:
    - commit on success (only if something changed)
    - rollback on error
    - close always

    NOTE:
    - expire_on_commit=False prevents DetachedInstanceError when returning ORM objects
      from a session that commits/closes.
    """
    session = Session(get_engine(get_database_url()), expire_on_commit=False)
    try:
        yield session

        # Only commit if something changed (faster + avoids expiring objects unnecessarily)
        if session.new or session.dirty or session.deleted:
            session.commit()

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
