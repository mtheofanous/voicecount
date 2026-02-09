from __future__ import annotations

"""
receive_orders (thin wrapper)

This project originally had a very large `receive_orders.py` page (~6k lines).
On Streamlit Cloud, big modules slow down cold starts and make reruns heavier.

Strategy:
- Keep this file tiny.
- Lazily import the heavy implementation (`receive_orders_page.py`) only when the
  Receive/Tracking page (or its helpers) are actually used.

Public API is preserved: anything importing `receive_orders` will still find the
same callables.
"""

from typing import Optional, Any


def _mod():
    # Works both when running as a package (features.manage_orders.*)
    # and when running as flat files (Streamlit Cloud quick tests).
    try:
        from . import receive_orders_page_mobile_fast as m  # type: ignore
        return m
    except Exception:
        import receive_orders_page_mobile_fast as m  # type: ignore
        return m


# -----------------------------
# Public API re-exports
# -----------------------------

def tracking_dashboard(
    venue_id: int,
    *,
    deep_order_id: Optional[int] = None,
    deep_provider: Optional[str] = None,
) -> None:
    return _mod().tracking_dashboard(venue_id, deep_order_id=deep_order_id, deep_provider=deep_provider)


def find_alternative_providers(*args: Any, **kwargs: Any):
    return _mod().find_alternative_providers(*args, **kwargs)


def upsert_provider_invoice_number(*args: Any, **kwargs: Any):
    return _mod().upsert_provider_invoice_number(*args, **kwargs)


def save_all_received_for_provider(*args: Any, **kwargs: Any):
    return _mod().save_all_received_for_provider(*args, **kwargs)


def request_supplier_resolution(*args: Any, **kwargs: Any):
    return _mod().request_supplier_resolution(*args, **kwargs)


def supplier_decision_from_venue(*args: Any, **kwargs: Any):
    return _mod().supplier_decision_from_venue(*args, **kwargs)


def venue_verify_and_close(*args: Any, **kwargs: Any):
    return _mod().venue_verify_and_close(*args, **kwargs)


def resolve_operational_missing(*args: Any, **kwargs: Any):
    return _mod().resolve_operational_missing(*args, **kwargs)
