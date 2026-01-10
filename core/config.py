# core/config.py
"""
Central configuration for the application.

- Handles database URL resolution
- Works locally (SQLite)
- Works on Streamlit Cloud (secrets)
- Works later with Supabase (Postgres)
"""

from __future__ import annotations

import os
import streamlit as st


def get_database_url() -> str:
    """
    Resolve the database connection string.

    Priority:
    1. Streamlit Cloud secrets
    2. Environment variable
    3. Local SQLite fallback
    """

    # 1️⃣ Streamlit Cloud (Settings → Secrets)
    if "DATABASE_URL" in st.secrets:
        return st.secrets["DATABASE_URL"]

    # 2️⃣ Local environment variable
    if os.getenv("DATABASE_URL"):
        return os.getenv("DATABASE_URL")

    # 3️⃣ Local SQLite fallback (dev only)
    return "sqlite:///voicecount2.db"
