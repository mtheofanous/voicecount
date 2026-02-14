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
from pathlib import Path


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


def ensure_google_credentials_file():
    raw_env = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()

    # ✅ If env var contains JSON, treat it as creds and write to /tmp
    if raw_env.startswith("{") and "private_key" in raw_env:
        p = Path("/tmp/google-creds.json")
        p.write_text(raw_env, encoding="utf-8")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(p)
        return

    # If already set to a file path, keep it
    if raw_env:
        return

    creds = None
    if "GOOGLE_CREDENTIALS_JSON" in st.secrets:
        creds = st.secrets["GOOGLE_CREDENTIALS_JSON"]
    if not creds:
        creds = os.getenv("GOOGLE_CREDENTIALS_JSON")

    if not creds:
        return

    p = Path("/tmp/google-creds.json")
    p.write_text(creds, encoding="utf-8")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(p)