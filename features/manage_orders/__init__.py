# features/manage_orders/__init__.py
"""
Manage Orders feature module.

Public API:
- orders_tab: main entry point used by app.py
"""

from .orders import *

__all__ = [
    "orders_tab",
]