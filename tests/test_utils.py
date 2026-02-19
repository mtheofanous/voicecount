"""
test_utils.py — tests for features/utils/voice_and_orders_utils.py

Covers: safe_str, str_or_default, normalize_key, num_or_default,
        distinct_units, coalesce_unit
"""
from __future__ import annotations
import math
import pytest
import pandas as pd
from features.utils.voice_and_orders_utils import (
    safe_str,
    str_or_default,
    normalize_key,
    num_or_default,
    distinct_units,
    coalesce_unit,
)


# ─────────────────────────────────────────────────────────────────────────────
# safe_str
# ─────────────────────────────────────────────────────────────────────────────

class TestSafeStr:

    def test_none_returns_default(self):
        assert safe_str(None) == ""

    def test_none_with_custom_default(self):
        assert safe_str(None, default="n/a") == "n/a"

    def test_nan_returns_default(self):
        assert safe_str(float("nan")) == ""

    def test_pandas_na_returns_default(self):
        assert safe_str(pd.NA) == ""

    def test_pandas_nan_returns_default(self):
        assert safe_str(pd.NaT) == ""

    def test_integer_converted(self):
        assert safe_str(42) == "42"

    def test_float_converted(self):
        assert safe_str(3.14) == "3.14"

    def test_string_returned_as_is(self):
        assert safe_str("leche") == "leche"

    def test_empty_string_returned(self):
        assert safe_str("") == ""

    def test_zero_converted(self):
        assert safe_str(0) == "0"

    def test_false_converted(self):
        assert safe_str(False) == "False"


# ─────────────────────────────────────────────────────────────────────────────
# str_or_default
# ─────────────────────────────────────────────────────────────────────────────

class TestStrOrDefault:

    def test_none_returns_default(self):
        assert str_or_default(None) == ""

    def test_none_with_custom_default(self):
        assert str_or_default(None, default="—") == "—"

    def test_whitespace_only_returns_default(self):
        assert str_or_default("   ") == ""

    def test_whitespace_with_custom_default(self):
        assert str_or_default("   ", default="—") == "—"

    def test_nan_returns_default(self):
        assert str_or_default(float("nan")) == ""

    def test_pandas_na_returns_default(self):
        assert str_or_default(pd.NA) == ""

    def test_valid_string_stripped(self):
        assert str_or_default("  leche  ") == "leche"

    def test_valid_string_no_strip_needed(self):
        assert str_or_default("leche") == "leche"

    def test_number_converted(self):
        assert str_or_default(5) == "5"

    def test_empty_string_returns_default(self):
        assert str_or_default("") == ""


# ─────────────────────────────────────────────────────────────────────────────
# normalize_key
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeKey:

    def test_none_returns_empty(self):
        assert normalize_key(None) == ""

    def test_empty_returns_empty(self):
        assert normalize_key("") == ""

    def test_lowercases(self):
        assert normalize_key("LECHE") == "leche"

    def test_strips_spaces(self):
        assert normalize_key("  leche  ") == "leche"

    def test_collapses_multiple_spaces(self):
        assert normalize_key("leche   entera") == "leche entera"

    def test_mixed_case_with_spaces(self):
        assert normalize_key("  Leche  Entera  ") == "leche entera"

    def test_single_word(self):
        assert normalize_key("kg") == "kg"


# ─────────────────────────────────────────────────────────────────────────────
# num_or_default
# ─────────────────────────────────────────────────────────────────────────────

class TestNumOrDefault:

    def test_none_returns_default(self):
        assert num_or_default(None) == 0.0

    def test_none_with_custom_default(self):
        assert num_or_default(None, default=1.0) == 1.0

    def test_nan_returns_default(self):
        assert num_or_default(float("nan")) == 0.0

    def test_pandas_na_returns_default(self):
        assert num_or_default(pd.NA) == 0.0

    def test_integer_converted(self):
        assert num_or_default(5) == 5.0

    def test_float_returned(self):
        assert num_or_default(2.5) == 2.5

    def test_string_number_converted(self):
        assert num_or_default("3.5") == 3.5

    def test_invalid_string_returns_default(self):
        assert num_or_default("abc") == 0.0

    def test_invalid_string_with_custom_default(self):
        assert num_or_default("abc", default=99.0) == 99.0

    def test_zero(self):
        assert num_or_default(0) == 0.0

    def test_negative(self):
        assert num_or_default(-1.5) == -1.5


# ─────────────────────────────────────────────────────────────────────────────
# distinct_units
# ─────────────────────────────────────────────────────────────────────────────

class TestDistinctUnits:

    def test_empty_list_returns_defaults(self):
        result = distinct_units([])
        assert isinstance(result, list)
        assert len(result) > 0
        assert "kg" in result

    def test_returns_sorted_list(self, make_product):
        # Sorting only applies to product-derived units, not the hardcoded fallback
        products = [
            make_product(id=1, unit="unidad"),
            make_product(id=2, unit="kg"),
            make_product(id=3, unit="botella"),
        ]
        result = distinct_units(products)
        assert result == sorted(result)

    def test_products_with_units(self, make_product):
        products = [
            make_product(id=1, unit="kg"),
            make_product(id=2, unit="unit"),
            make_product(id=3, unit="l"),
        ]
        result = distinct_units(products)
        assert "kg" in result
        assert "unit" in result
        assert "l" in result

    def test_deduplicates_units(self, make_product):
        products = [
            make_product(id=1, unit="kg"),
            make_product(id=2, unit="kg"),
            make_product(id=3, unit="kg"),
        ]
        result = distinct_units(products)
        assert result.count("kg") == 1

    def test_none_units_skipped(self, make_product):
        products = [
            make_product(id=1, unit=None),
            make_product(id=2, unit=""),
            make_product(id=3, unit="kg"),
        ]
        result = distinct_units(products)
        assert "kg" in result
        # None / empty should not appear
        assert "" not in result

    def test_fallback_defaults_when_all_units_empty(self, make_product):
        products = [make_product(id=1, unit=None)]
        result = distinct_units(products)
        assert "kg" in result  # from defaults


# ─────────────────────────────────────────────────────────────────────────────
# coalesce_unit
# ─────────────────────────────────────────────────────────────────────────────

class TestCoalesceUnit:

    def test_x_provided_returns_x_lowercased(self):
        assert coalesce_unit("KG") == "kg"

    def test_x_empty_falls_back_to_product(self, make_product):
        p = make_product(unit="kg")
        assert coalesce_unit("", prod_obj=p) == "kg"

    def test_x_none_falls_back_to_product(self, make_product):
        p = make_product(unit="l")
        assert coalesce_unit(None, prod_obj=p) == "l"

    def test_x_none_no_product_returns_default(self):
        assert coalesce_unit(None) == "unidad"

    def test_x_none_no_product_custom_default(self):
        assert coalesce_unit(None, default="unit") == "unit"

    def test_product_no_unit_falls_back_to_default(self, make_product):
        p = make_product(unit=None)
        assert coalesce_unit("", prod_obj=p) == "unidad"

    def test_x_whitespace_falls_back_to_product(self, make_product):
        p = make_product(unit="pack")
        assert coalesce_unit("   ", prod_obj=p) == "pack"

    def test_x_valid_ignores_product(self, make_product):
        p = make_product(unit="kg")
        assert coalesce_unit("unit", prod_obj=p) == "unit"
