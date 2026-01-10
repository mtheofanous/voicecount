# core/db.py
from __future__ import annotations

from sqlmodel import Session, create_engine

from core.config import get_database_url


DATABASE_URL = get_database_url()

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,  # good for cloud / dropped connections
)


def get_session() -> Session:
    return Session(engine)
