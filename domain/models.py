"""
models.py — SQLModel table definitions (single source of truth)

Goal: stop redefining models across multiple files (vint.py + orders_modern_ui_v12.py).
These definitions combine the "rich" Orders workflow fields with the Product schema you already use.

Next refactor step:
- Import these models everywhere instead of defining them inline.
"""

from __future__ import annotations
from sqlalchemy import UniqueConstraint
from datetime import datetime, date
from typing import Optional

from sqlmodel import Field, SQLModel
#   
print(">>> domain.models imported")
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

class VenueTranscriptionSettings(SQLModel, table=True):
    """Per-venue transcription configuration.

    Stored in main DB (same as Product/Order tables) so the UI can be controlled centrally from ADMIN3.
    """
    __tablename__ = "venue_transcription_settings"
    __table_args__ = (
        UniqueConstraint("venue_id", name="uq_venue_transcription_settings_venue_id"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    venue_id: int = Field(index=True)

    # OpenAI Whisper API | Google Speech-to-Text | Faster-Whisper (local)
    asr_backend: str = Field(default="OpenAI Whisper API")
    # auto | es | el | en
    lang_code: str = Field(default="auto")
    samplerate: int = Field(default=16000)

    # Optional: if you want to globally hide user controls in UI (kept for future use)
    hide_user_controls: bool = Field(default=True)

    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_by: Optional[str] = Field(default=None, index=True)

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
    __table_args__ = (
        UniqueConstraint("order_id", "provider_name", name="uq_provider_receipt"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)

    venue_id: int = Field(index=True)
    order_id: int = Field(index=True)
    provider_name: str = Field(index=True)

    # supplier declaration (from tracking link)
    supplier_declaration: Optional[str] = Field(default=None, index=True)  # full | partial | none
    supplier_declared_at: Optional[datetime] = Field(default=None, index=True)
    supplier_declared_by: Optional[str] = Field(default=None, index=True)
    supplier_declared_comment: Optional[str] = Field(default=None)

    # real invoice number (can be set by supplier or venue later)
    invoice_number: Optional[str] = Field(default=None, index=True)
    invoice_number_set_at: Optional[datetime] = Field(default=None, index=True)
    invoice_number_set_by: Optional[str] = Field(default=None, index=True)

    # status
    received: bool = Field(default=False, index=True)
    received_at: Optional[datetime] = Field(default=None, index=True)
    received_by: Optional[str] = Field(default=None, index=True)

    # ✅ recommended: overall closure marker (optional, but useful)
    all_resolutions_closed_at: Optional[datetime] = Field(default=None, index=True)
    all_resolutions_closed_by: Optional[str] = Field(default=None, index=True)

    note: Optional[str] = Field(default=None)

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_by: Optional[str] = None
    
    
class Provider(SQLModel, table=True):
    """
    Provider directory per venue.

    delivery_schedule_json example:
    {
      "mon": ["08:00-14:00", "16:00-20:00"],
      "thu": ["08:00-14:00"]
    }
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

    emails: Optional[str] = Field(default=None)
    phones: Optional[str] = Field(default=None)
    order_email: Optional[str] = Field(default=None, index=True)
    order_phone: Optional[str] = Field(default=None, index=True)

    # ✅ Delivery days + hour slots
    delivery_schedule_json: Optional[str] = Field(default=None)

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    
    
class ProviderDiscountRule(SQLModel, table=True):
    """Discount rules configured by a supplier (provider).

    Supports:
      - Global rules per provider (product_id = NULL)
      - Product-specific rules (product_id set)

    Rule kinds:
      - line_pct
      - line_net_price
      - prev_month_pct
      - prev_month_net_price
    """

    __tablename__ = "provider_discount_rule"
    __table_args__ = (
        UniqueConstraint("provider_id", "product_id", "rule_kind", "min_qty", name="uq_provider_rule"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)

    venue_id: int = Field(index=True)
    provider_id: int = Field(foreign_key="provider.id", index=True)

    # NULL => applies to any product from the provider
    product_id: Optional[int] = Field(default=None, foreign_key="product.id", index=True)

    # -----------------------
    # Condition
    # -----------------------
    # For line_* rules: minimum qty on this line
    # For prev_month_* rules: you can still use min_qty OR use prev_month_min_qty (recommended)
    min_qty: float = Field(default=0.0)

    # For prev_month_* rules: last month total qty threshold
    prev_month_min_qty: Optional[float] = Field(default=None)

    # -----------------------
    # Effect
    # -----------------------
    # For *_pct rules
    discount_percent: float = Field(default=0.0)

    # For *_net_price rules (NET unit price override)
    price_override: Optional[float] = Field(default=None)

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
    invoice_listed: Optional[bool] = Field(default=None, index=True)
    qty_invoiced: Optional[float] = Field(default=None)
    venue_comment: Optional[str] = Field(default=None)

    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_by: Optional[str] = None

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
    __tablename__ = "seguimientoticket"
    __table_args__ = ({"extend_existing": True},)

    id: Optional[int] = Field(default=None, primary_key=True)
    venue_id: int = Field(index=True)
    order_id: int = Field(index=True)
    provider_name: str = Field(index=True)
    order_line_id: int = Field(index=True)

    kind: str = Field(index=True)
    state: str = Field(index=True)

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


# ====== STATE MACHINE CONSTANTS (exactly your list) ======
ORDER_STATES = {
    "ORDER_SENT",
    "SUPPLIER_CONFIRMED_FULL",
    "SUPPLIER_CONFIRMED_PARTIAL",
    "SUPPLIER_CONFIRMED_NONE",
    "RECEIVED",
    "MATCHED_WITH_INVOICE",
    "INVOICE_DISCREPANCY",
    "WAITING_SUPPLIER_ACTION",
    "OPERATIONAL_MISSING_PRODUCT",
    "WAITING_MANAGER_DECISION",
    "SUPPLIER_CREDIT_NOTE_ISSUED",
    "SUPPLEMENTARY_DELIVERY_SENT",
    "DECISION_REORDER_SAME",
    "DECISION_SWITCH_SUPPLIER",
    "DECISION_NOT_NEEDED",
    "CLOSED",
}

class OrderWorkflow(SQLModel, table=True):
    """
    1 row per (order_id, provider_name): current state + who/when.
    """
    __tablename__ = "order_workflow"
    __table_args__ = (
        UniqueConstraint("order_id", "provider_name", name="uq_workflow_order_provider"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)

    venue_id: int = Field(index=True)
    order_id: int = Field(index=True)
    provider_name: str = Field(index=True)

    state: str = Field(default="ORDER_SENT", index=True)

    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_by_role: Optional[str] = Field(default=None, index=True)  # supplier/venue/manager/system
    updated_by: Optional[str] = Field(default=None, index=True)       # identifier/email if internal

    # Optional lightweight notes (no payments, no prices)
    note: Optional[str] = None


class OrderWorkflowEvent(SQLModel, table=True):
    """
    Append-only history of transitions (auditable).
    """
    __tablename__ = "order_workflow_event"
    __table_args__ = ({"extend_existing": True},)

    id: Optional[int] = Field(default=None, primary_key=True)

    venue_id: int = Field(index=True)
    order_id: int = Field(index=True)
    provider_name: str = Field(index=True)

    from_state: str = Field(index=True)
    to_state: str = Field(index=True)

    actor_role: str = Field(index=True)          # supplier / receiving_employee / manager / system / venue
    actor: Optional[str] = Field(default=None)   # user identifier if internal
    at: datetime = Field(default_factory=datetime.utcnow, index=True)

    note: Optional[str] = None
    
    
class UrgentReorderRequest(SQLModel, table=True):
    """
    Created when an incidence is marked as 'Order urgent'
    and user clicks 'Save & request decision'.
    """
    __tablename__ = "urgentreorderrequest"
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)

    # product info
    product_name: str
    product_normalized: str
    quantity: float
    unit: str

    # original provider (from invoice)
    original_provider_id: Optional[int] = None
    original_provider_name: Optional[str] = None

    # linkage
    incidence_id: Optional[int] = None
    order_id: Optional[int] = None  # created when first supplier is contacted

    # decision
    selected_provider_id: Optional[int] = None
    selected_provider_name: Optional[str] = None

    status: str = Field(default="pending")
    # pending | sent | done | cancelled

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    
class ProviderResolution(SQLModel, table=True):
    """
    One row per resolution-track per provider per order.
    This allows closing re-delivery independently from credit note.

    resolution_type:
    - "supplementary_delivery"
    - "credit_note"
    - "reject"

    status:
      - "expected"  (venue expects it / supplier agreed, not delivered/issued yet)
      - "issued"    (supplier issued the doc / shipped the goods)
      - "verified"  (venue/accounting verified)
      - "closed"    (explicitly closed)
      - "cancelled" (no longer needed / replaced by other solution)
    """
    __tablename__ = "provider_resolution"
    __table_args__ = (
        UniqueConstraint("order_id", "provider_name", "resolution_type", name="uq_provider_resolution"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)

    venue_id: int = Field(index=True)
    order_id: int = Field(index=True)
    provider_name: str = Field(index=True)

    resolution_type: str = Field(index=True) 
    status: str = Field(default="expected", index=True)

    # --- Common refs / notes ---
    reference_number: Optional[str] = Field(default=None, index=True)  # credit note number OR delivery note ref
    due_date: Optional[date] = Field(default=None, index=True)         # optional ETA or accounting due date
    note: Optional[str] = Field(default=None)

    # meta_json can hold items breakdown (optional), without creating extra tables
    meta_json: Optional[str] = Field(default=None)

    # --- audit timeline ---
    expected_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    expected_by: Optional[str] = Field(default=None, index=True)       # venue/supplier actor id
    expected_by_role: Optional[str] = Field(default=None, index=True)  # venue/supplier/system

    issued_at: Optional[datetime] = Field(default=None, index=True)
    issued_by: Optional[str] = Field(default=None, index=True)
    issued_by_role: Optional[str] = Field(default=None, index=True)

    verified_at: Optional[datetime] = Field(default=None, index=True)
    verified_by: Optional[str] = Field(default=None, index=True)
    verified_by_role: Optional[str] = Field(default=None, index=True)

    closed_at: Optional[datetime] = Field(default=None, index=True)
    closed_by: Optional[str] = Field(default=None, index=True)
    closed_by_role: Optional[str] = Field(default=None, index=True)

    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_by: Optional[str] = Field(default=None, index=True)