# core/init.py
from sqlmodel import SQLModel
from core.db import get_engine
from core.config import get_database_url


def init_db() -> None:
    # Ensure models are imported so tables register into metadata
    from domain import models  # keep this as you already have

    SQLModel.metadata.create_all(get_engine(get_database_url()))

