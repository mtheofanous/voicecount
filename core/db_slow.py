# core/db.py
from __future__ import annotations

import streamlit as st
from sqlmodel import Session, create_engine
from core.config import get_database_url


@st.cache_resource
def get_engine(db_url: str):
    # db_url is part of the cache key, so if you change DATABASE_URL,
    # Streamlit will build a new engine instead of reusing the old SQLite one.
    return create_engine(
        db_url,
        echo=False,
        pool_pre_ping=True,
    )


def get_session() -> Session:
    return Session(get_engine(get_database_url()))
