from __future__ import annotations

"""features.manage_orders.receive_orders

Fast, safe lazy-loader for Receive/Tracking.

Your previous wrapper swallowed *all* exceptions during import, so if
`receive_orders_page.py` exists but fails *inside* (e.g. missing dependency),
you would incorrectly see a 'ModuleNotFoundError' saying it couldn't import.

This version:
- Uses importlib.import_module (correct package resolution)
- Only skips when the *module truly doesn't exist*
- If the module exists but raises an error while importing, we re-raise the
  original exception (so you see the real cause).
"""

from typing import Any, Optional
import importlib


def _try_import(mod_path: str):
    try:
        return importlib.import_module(mod_path)
    except ModuleNotFoundError as e:
        # Only treat as "module missing" if the missing name is exactly the module we tried
        if e.name == mod_path:
            return None
        # Otherwise, a dependency inside the module is missing -> that's the real error
        raise


def _mod():
    pkg = __package__  # e.g. "features.manage_orders"
    candidates = (
        "receive_orders_page_fast",   # optional optimized file
        "receive_orders_page",        # canonical implementation (should exist)
        "receive_orders_page_mobile", # legacy/optional name
    )

    # 1) Package-relative imports (preferred)
    if pkg:
        for name in candidates:
            m = _try_import(f"{pkg}.{name}")
            if m is not None:
                return m

    # 2) Flat imports (fallback)
    for name in candidates:
        m = _try_import(name)
        if m is not None:
            return m

    raise ModuleNotFoundError(
        "Could not import Receive/Tracking implementation. Tried: "
        + ", ".join(candidates)
        + (f" (package={pkg})" if pkg else "")
    )


# -----------------------------
# Public API re-exports
# -----------------------------

def tracking_dashboard(
    venue_id: int,
    *,
    deep_order_id: Optional[int] = None,
    deep_provider: Optional[str] = None,
) -> None:
    return _mod().tracking_dashboard(
        venue_id,
        deep_order_id=deep_order_id,
        deep_provider=deep_provider,
    )


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


# -----------------------------
# Lazy re-exports for internal helpers (used by history/reports)
# -----------------------------
# -----------------------------
# Lazy re-exports for internal helpers (used by history/reports)
# -----------------------------
# -----------------------------
# Lazy re-exports for internal helpers (used by history/reports)
# -----------------------------
from typing import TYPE_CHECKING
import importlib

_IMPL_MODULE = None

def _mod():
    global _IMPL_MODULE
    if _IMPL_MODULE is None:
        _IMPL_MODULE = importlib.import_module(
            "features.manage_orders.receive_orders_page"
        )
    return _IMPL_MODULE


if TYPE_CHECKING:
    from .receive_orders_page import OrderContext as OrderContext  # noqa: F401


def __getattr__(name: str):
    """
    Lazily forward any attribute to receive_orders_page, if it exists there.
    """
    m = _mod()
    if hasattr(m, name):
        return getattr(m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    m = _mod()
    return sorted(set(globals().keys()) | set(dir(m)))


