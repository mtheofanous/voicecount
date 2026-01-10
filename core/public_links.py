# core/public_links.py
from __future__ import annotations

import hashlib
import hmac
import os
from urllib.parse import urlencode

ROLE_SUPPLIER = "supplier"
ROLE_VENUE = "venue"
VALID_ROLES = {ROLE_SUPPLIER, ROLE_VENUE}

DEFAULT_LOCAL_BASE_URL = "http://localhost:8501"
DEFAULT_DEV_SECRET = "dev-secret-change-me"


def _secret() -> bytes:
    """
    Shared secret for signing public links.
    Set VOICECOUNT_PUBLIC_LINK_SECRET in env/secrets for production.
    """
    s = (os.getenv("VOICECOUNT_PUBLIC_LINK_SECRET") or "").strip()
    if not s:
        s = DEFAULT_DEV_SECRET
    return s.encode("utf-8")


def norm_provider(x: str) -> str:
    x = (x or "").strip()
    return x or "(Sin proveedor)"


def norm_role(x: str) -> str:
    x = (x or "").strip().lower()
    return x if x in VALID_ROLES else ROLE_SUPPLIER


def sign_link(*, order_id: int, provider_name: str, role: str) -> str:
    """
    Sign a link payload.
    IMPORTANT: role is part of the signature so it can't be changed by editing the URL.
    """
    msg = f"{int(order_id)}|{norm_provider(provider_name)}|{norm_role(role)}".encode("utf-8")
    return hmac.new(_secret(), msg, hashlib.sha256).hexdigest()


def verify_link(*, order_id: int, provider_name: str, role: str, sig: str) -> bool:
    """
    Verify the signature for the given payload.
    """
    expected = sign_link(order_id=order_id, provider_name=provider_name, role=role)
    return hmac.compare_digest((sig or "").strip().lower(), expected.lower())


def get_public_base_url() -> str:
    """
    Base URL where the public tracking page is served.

    Examples:
      - Local:   http://localhost:8501
      - Cloud:   https://your-app.streamlit.app
      - Domain:  https://orders.yourdomain.com
    """
    base = (os.getenv("VOICECOUNT_PUBLIC_BASE_URL") or "").strip()
    if not base:
        base = DEFAULT_LOCAL_BASE_URL
    return base.rstrip("/")


def build_seguimiento_url(
    *,
    order_id: int,
    provider_name: str,
    role: str,
    page_path: str = "seguimiento",
) -> str:
    """
    Build a signed public URL for the seguimiento page.

    Streamlit Pages convention:
      BASE_URL/<page_path>?order_id=...&provider=...&role=...&sig=...

    page_path examples:
      - "seguimiento" (if you have pages/seguimiento.py)
      - ""            (if you route inside app.py via query params)
    """
    role_n = norm_role(role)
    prov_n = norm_provider(provider_name)

    sig = sign_link(order_id=order_id, provider_name=prov_n, role=role_n)

    base = get_public_base_url()
    path = f"/{page_path.lstrip('/')}" if page_path else ""

    qs = urlencode(
        {
            "order_id": int(order_id),
            "provider": prov_n,
            "role": role_n,
            "sig": sig,
        }
    )

    return f"{base}{path}?{qs}"
