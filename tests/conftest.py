"""
conftest.py — shared fixtures and Streamlit mocking.

Mocks streamlit and all heavy external dependencies BEFORE any app
module is imported, so tests run without a running Streamlit server,
a real database, or an audio device.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch
from typing import Optional
import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# 1. Mock streamlit BEFORE any app module is imported
# ─────────────────────────────────────────────────────────────────────────────

def _make_streamlit_mock() -> MagicMock:
    """Return a MagicMock that behaves enough like streamlit for imports."""
    mock = MagicMock()

    # cache decorators must return the original function unchanged
    mock.cache_data = lambda *args, **kwargs: (
        (lambda f: f) if not args else args[0]
        if callable(args[0]) else (lambda f: f)
    )
    mock.cache_resource = lambda *args, **kwargs: (
        (lambda f: f) if not args else args[0]
        if callable(args[0]) else (lambda f: f)
    )
    mock.fragment = lambda *args, **kwargs: (
        (lambda f: f) if not args else args[0]
        if callable(args[0]) else (lambda f: f)
    )

    # session_state behaves like a dict
    mock.session_state = {}

    # stop / error / warning / info don't raise by default
    mock.stop = MagicMock(side_effect=RuntimeError("st.stop() called"))
    mock.error = MagicMock()
    mock.warning = MagicMock()
    mock.info = MagicMock()
    mock.success = MagicMock()

    return mock


_st_mock = _make_streamlit_mock()
sys.modules["streamlit"] = _st_mock
sys.modules["streamlit.components"] = MagicMock()
sys.modules["streamlit.components.v1"] = MagicMock()

# Mock heavy audio / ASR dependencies
sys.modules["streamlit_float"] = MagicMock()
sys.modules["streamlit_audiorecorder"] = MagicMock()
sys.modules["streamlit_webrtc"] = MagicMock()
sys.modules["av"] = MagicMock()
sys.modules["pydub"] = MagicMock()
sys.modules["pydub.audio_segment"] = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.cloud"] = MagicMock()
sys.modules["google.cloud.speech"] = MagicMock()
sys.modules["google.cloud.speech_v1"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["faster_whisper"] = MagicMock()

# Mock asr_google so voice_and_orders_utils imports cleanly
asr_mod = types.ModuleType("features.utils.asr_google")
asr_mod.asr_google = MagicMock(return_value="mocked transcript")  # type: ignore
sys.modules["features.utils.asr_google"] = asr_mod

# Mock prefix_stripper
ps_mod = types.ModuleType("features.utils.prefix_stripper")
ps_mod.clean_fragment = lambda x: x  # type: ignore
sys.modules["features.utils.prefix_stripper"] = ps_mod


# ─────────────────────────────────────────────────────────────────────────────
# 2. Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def make_product():
    """Factory that creates a lightweight Product-like object for tests."""
    from domain.models import Product

    def _factory(
        id: int = 1,
        venue_id: int = 1,
        name: str = "Leche Entera",
        category: str = "Lácteos",
        unit: str = "unit",
        quantity: float = 1.0,
        price: float = 1.20,
        iva: float = 4.0,
        provider_name: Optional[str] = "Proveedor A",
        provider_email: Optional[str] = "a@prov.com",
        provider_phone: Optional[str] = None,
        provider_address: Optional[str] = None,
        aliases: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Product:
        p = Product(
            id=id,
            venue_id=venue_id,
            name=name,
            category=category,
            unit=unit,
            quantity=quantity,
            price=price,
            iva=iva,
            provider_name=provider_name,
            provider_email=provider_email,
            provider_phone=provider_phone,
            provider_address=provider_address,
            aliases=aliases,
            description=description,
        )
        return p

    return _factory


@pytest.fixture
def sample_products(make_product):
    """A small realistic product catalog."""
    return [
        make_product(id=1, name="Leche Entera",   aliases="leche|leche entera|milk",   provider_name="Makro",    unit="unit", category="Lácteos"),
        make_product(id=2, name="Tomate Triturado",aliases="tomate|tomates",             provider_name="Makro",    unit="kg",   category="Conservas"),
        make_product(id=3, name="Aceite de Oliva", aliases="aceite|aove|olive oil",      provider_name="Sysco",    unit="l",    category="Aceites"),
        make_product(id=4, name="Pan de Molde",    aliases="pan|bread|pan molde",        provider_name="Bimbo",    unit="pack", category="Panadería"),
        make_product(id=5, name="Azúcar Blanco",   aliases="azucar|sugar|azúcar",        provider_name="Makro",    unit="kg",   category="Básicos"),
        make_product(id=6, name="Sal Fina",        aliases="sal|salt",                   provider_name="Sysco",    unit="kg",   category="Básicos"),
        make_product(id=7, name="Producto Sin ID", aliases="test",                       provider_name=None,       unit="unit", category=None),
    ]
    # Note: product id=7 is intentionally kept with an id for alias testing


@pytest.fixture
def empty_df():
    """Empty DataFrame with standard order columns."""
    return pd.DataFrame(columns=["matched_product_id", "matched_name", "unit", "unit_custom", "provider"])


@pytest.fixture
def sample_df():
    """Small DataFrame representing parsed order candidates."""
    return pd.DataFrame({
        "matched_product_id": [1, 2, None, 4],
        "matched_name":       ["Leche Entera", "Tomate Triturado", "Aceite de Oliva", None],
        "unit":               ["unit", "kg", "Other…", "pack"],
        "unit_custom":        ["",     "",   "botella", ""],
        "quantity":           [2.0,    1.5,  1.0,       3.0],
    })
