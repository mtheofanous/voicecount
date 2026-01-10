from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class Product(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)

    category: Optional[str] = None
    unit: Optional[str] = Field(default="unidad")
    
    # ✅ NEW: default quantity (your "cantidad")
    default_qty: float = Field(default=1.0)

    provider_name: Optional[str] = Field(default=None, index=True)
    provider_email: Optional[str] = None
    provider_phone: Optional[str] = None
    provider_address: Optional[str] = None
    
    aliases: Optional[str] = Field(default=None)   # ✅ ADD THIS

    created_at: datetime = Field(default_factory=datetime.utcnow)


class Order(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field(default="draft")  # draft|final
    title: Optional[str] = None
    note: Optional[str] = None


class OrderLine(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    order_id: int = Field(foreign_key="order.id")
    product_id: Optional[int] = Field(default=None, foreign_key="product.id")

    spoken_name: str
    quantity: float = 1.0
    unit: Optional[str] = Field(default="unidad")
    confidence: Optional[float] = Field(default=0.0)

    matched_name: Optional[str] = None
    provider: Optional[str] = None
