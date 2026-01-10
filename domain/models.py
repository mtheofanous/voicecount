"""
models.py — SQLModel table definitions (single source of truth)

Goal: stop redefining models across multiple files (vint.py + orders_modern_ui_v12.py).
These definitions combine the "rich" Orders workflow fields with the Product schema you already use.

Next refactor step:
- Import these models everywhere instead of defining them inline.
"""

from __future__ import annotations
from sqlalchemy import UniqueConstraint
from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class Product(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    venue_id: int = Field(index=True)

    # core catalog fields
    name: str = Field(index=True)
    description: Optional[str] = Field(default=None)
    category: Optional[str] = Field(default=None)

    unit: str = Field(default="unidad")
    quantity: float = Field(default=1.0)
    price: float = Field(default=0.0)
    iva: float = Field(default=21.0)

    provider_name: Optional[str] = Field(default=None, index=True)
    provider_email: Optional[str] = Field(default=None)
    provider_phone: Optional[str] = Field(default=None)
    provider_address: Optional[str] = Field(default=None)

    aliases: Optional[str] = Field(default=None)

    created_at: datetime = Field(default_factory=datetime.utcnow)

    # optional backwards compatibility with older code/DB that used default_qty
    default_qty: Optional[float] = Field(default=None)


class Order(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    venue_id: int = Field(index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    # Audit
    created_by: Optional[str] = Field(default=None, index=True)
    updated_at: Optional[datetime] = Field(default=None, index=True)
    updated_by: Optional[str] = Field(default=None, index=True)

    # Verification (for history)
    verified_at: Optional[datetime] = Field(default=None, index=True)
    verified_by: Optional[str] = Field(default=None, index=True)

    # Workflow
    # draft | ready_to_send | pending_receive | final
    status: str = Field(default="draft", index=True)

    title: Optional[str] = Field(default=None)
    note: Optional[str] = Field(default=None)


class OrderLine(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    venue_id: int = Field(index=True)

    order_id: int = Field(foreign_key="order.id", index=True)
    product_id: Optional[int] = Field(default=None, foreign_key="product.id", index=True)

    spoken_name: str = Field(default="")
    quantity: float = Field(default=1.0)
    unit: Optional[str] = Field(default="unidad")
    confidence: Optional[float] = Field(default=0.0)

    # matching helpers (used in your existing pipeline)
    matched_name: Optional[str] = Field(default=None)
    provider: Optional[str] = Field(default=None)

    # Line audit
    updated_at: Optional[datetime] = Field(default=None, index=True)
    updated_by: Optional[str] = Field(default=None, index=True)

    # Receiving workflow (per line)
    received_ok: Optional[bool] = Field(default=None, index=True)
    missing_qty: Optional[float] = Field(default=None)
    received_at: Optional[datetime] = Field(default=None, index=True)
    received_by: Optional[str] = Field(default=None, index=True)

    # Missing clarification (for supplier follow-up)
    # unknown | in_invoice | not_in_invoice
    missing_invoice_status: Optional[str] = Field(default="unknown", index=True)
    missing_note: Optional[str] = Field(default=None)


class OrderPresence(SQLModel, table=True):
    """
    Lightweight presence tracking for collaborative editing / multi-user awareness.
    """

    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)

    order_id: int = Field(index=True)
    venue_id: int = Field(index=True)

    actor: str = Field(index=True)       # user email or identifier
    session_id: str = Field(index=True)  # browser/session id

    last_seen_at: datetime = Field(default_factory=datetime.utcnow, index=True)

class ProviderReceipt(SQLModel, table=True):
    __tablename__ = "provider_receipt"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)

    venue_id: int = Field(index=True)
    order_id: int = Field(index=True)

    provider_name: str = Field(index=True)

    # status
    received: bool = Field(default=False, index=True)
    received_at: Optional[datetime] = Field(default=None, index=True)
    received_by: Optional[str] = Field(default=None, index=True)
    note: Optional[str] = Field(default=None)


    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    
    
class Provider(SQLModel, table=True):
    """
    Provider directory per venue.
    emails/phones: allow multiple values using a pipe-separated string:
      "a@x.com | b@y.com"
    """
    __tablename__ = "provider"
    __table_args__ = (
        UniqueConstraint("venue_id", "name", name="uq_provider_venue_name"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    venue_id: int = Field(index=True)

    name: str = Field(index=True)
    tax_number: Optional[str] = Field(default=None, index=True)
    address: Optional[str] = Field(default=None)

    emails: Optional[str] = Field(default=None)  # "a@x.com | b@y.com"
    phones: Optional[str] = Field(default=None)  # "+34... | +30..."
    order_email: Optional[str] = Field(default=None, index=True)
    order_phone: Optional[str] = Field(default=None, index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    
    
class ProviderDiscountRule(SQLModel, table=True):
    """Discount rules configured by a supplier (provider).

    Supports:
      - Global rules per provider (product_id = NULL)
      - Product-specific rules (product_id set)

    Example:
      - "10% discount if you order more than 10" => min_qty=10, discount_percent=10
    """

    __tablename__ = "provider_discount_rule"
    __table_args__ = (
        UniqueConstraint("provider_id", "product_id", "min_qty", name="uq_provider_rule"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)

    venue_id: int = Field(index=True)
    provider_id: int = Field(foreign_key="provider.id", index=True)

    # NULL => applies to any product from the provider
    product_id: Optional[int] = Field(default=None, foreign_key="product.id", index=True)

    # Condition
    min_qty: float = Field(default=0.0)

    # Effect
    discount_percent: float = Field(default=0.0)
    rule_kind: str = Field(default="line_pct", index=True)

    note: Optional[str] = Field(default=None)
    is_active: bool = Field(default=True, index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    
    
class ProviderLineFollowUp(SQLModel, table=True):
    __tablename__ = "provider_line_followup"
    __table_args__ = (
        UniqueConstraint("order_id", "provider_name", "order_line_id", name="uq_followup_line"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)

    venue_id: int = Field(index=True)
    order_id: int = Field(index=True)

    provider_name: str = Field(index=True)
    order_line_id: int = Field(index=True)

    qty_ordered: Optional[float] = Field(default=None)

    # --- Supplier side ---
    supplier_status: str = Field(default="ok", index=True)   # ok/partial/missing
    supplier_qty: Optional[float] = Field(default=None)
    supplier_reason: Optional[str] = Field(default=None)
    supplier_comment: Optional[str] = Field(default=None)

    # --- Venue side ---
    venue_qty: Optional[float] = Field(default=None)
    venue_comment: Optional[str] = Field(default=None)

    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)

class ProviderSendStatus(SQLModel, table=True):
    __tablename__ = "provider_send_status"
    __table_args__ = (
        UniqueConstraint("order_id", "provider_name", name="uq_provider_send"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)

    venue_id: int = Field(index=True)
    order_id: int = Field(index=True)
    provider_name: str = Field(index=True)

    # High-level: was "sent" at least once?
    sent: bool = Field(default=False, index=True)

    # Per-channel tracking (BONUS PRO)
    sent_email: bool = Field(default=False, index=True)
    sent_whatsapp: bool = Field(default=False, index=True)
    sent_txt: bool = Field(default=False, index=True)

    # Audit info
    sent_at: Optional[datetime] = Field(default=None, index=True)
    sent_by: Optional[str] = Field(default=None, index=True)

    # Reliability / support
    send_attempts: int = Field(default=0)
    last_error: Optional[str] = Field(default=None)

    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    
    
class SeguimientoTicket(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    venue_id: int = Field(index=True)
    order_id: int = Field(index=True)
    provider_name: str = Field(index=True)
    order_line_id: int = Field(index=True)

    kind: str = Field(index=True)      # supplier_short | delivery_mismatch | invoice_mismatch
    state: str = Field(index=True)     # open | resolved

    product_name: str = ""
    unit: str = ""

    qty_ordered: float = 0.0
    qty_expected: float = 0.0
    qty_received: float = 0.0

    invoice_number: Optional[str] = None
    qty_invoiced: Optional[float] = None
    unit_price_expected: Optional[float] = None
    unit_price_invoiced: Optional[float] = None

    note: Optional[str] = None
    resolution_note: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None