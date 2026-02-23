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
    """
    Ensures GOOGLE_APPLICATION_CREDENTIALS points to a valid file path.
    Works for:
    - Local .env setups
    - Streamlit Cloud secrets
    - Accidental JSON stored directly in GOOGLE_APPLICATION_CREDENTIALS
    """

    raw_env = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()

    # -------------------------------------------------
    # 1️⃣ If env var contains JSON directly -> fix it
    # -------------------------------------------------
    if raw_env.startswith("{") and "private_key" in raw_env:
        p = Path("/tmp/google-creds.json")
        p.write_text(raw_env, encoding="utf-8")
        try:
            p.chmod(0o600)  # owner-read-only — prevent other processes from reading the key
        except Exception:
            pass
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(p)
        return

    # -------------------------------------------------
    # 2️⃣ If env var is already a valid file path -> keep it
    # -------------------------------------------------
    if raw_env and Path(raw_env).exists():
        return

    # -------------------------------------------------
    # 3️⃣ Otherwise try loading from Streamlit secrets
    # -------------------------------------------------
    creds = None

    if "GOOGLE_CREDENTIALS_JSON" in st.secrets:
        creds = st.secrets["GOOGLE_CREDENTIALS_JSON"]

    if not creds:
        creds = os.getenv("GOOGLE_CREDENTIALS_JSON")

    if not creds:
        raise RuntimeError(
            "Google credentials not found. "
            "Set GOOGLE_CREDENTIALS_JSON in Streamlit secrets or env."
        )

    # Write JSON to temp file (owner-readable only)
    p = Path("/tmp/google-creds.json")
    p.write_text(creds, encoding="utf-8")
    try:
        p.chmod(0o600)
    except Exception:
        pass

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(p)