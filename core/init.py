# core/init.py
import os
from sqlmodel import SQLModel
from core.db import get_engine
from core.config import get_database_url

# Guard para evitar trabajo repetido si alguien llama init_db() múltiples veces
_INITIALIZED = False


def init_db() -> None:
    """
    Production-friendly DB init.

    - Imports models once to register tables into SQLModel.metadata.
    - Optionally runs create_all (dev only), controlled by env.
    - Does not re-run work if called again in the same process.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    # Ensure models are imported so tables register into metadata
    # (keep as you already had, but do it once)
    from domain import models  # noqa: F401

    # In production, prefer migrations (Alembic) and disable create_all.
    # Default = enabled for local/dev, disabled when env says so.
    auto_create = os.getenv("DB_AUTO_CREATE", "1").strip().lower() not in {"0", "false", "no"}

    if auto_create:
        engine = get_engine(get_database_url())
        SQLModel.metadata.create_all(engine)

    _INITIALIZED = True

