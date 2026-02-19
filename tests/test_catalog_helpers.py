"""
test_catalog_helpers.py — tests for catalog helper functions in
features/create_order/new_order_tab.py

Covers: build_google_phrases, build_alias_indexes, alias_suggestions,
        add_provider_column_fast, apply_unit_choice
"""
from __future__ import annotations
import pytest
import pandas as pd
from features.create_order.new_order_tab import (
    build_google_phrases,
    build_alias_indexes,
    alias_suggestions,
    add_provider_column_fast,
    apply_unit_choice,
)


# ─────────────────────────────────────────────────────────────────────────────
# build_google_phrases
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildGooglePhrases:

    def test_empty_list_returns_empty(self):
        assert build_google_phrases([]) == []

    def test_product_name_included(self, make_product):
        p = make_product(id=1, name="Leche Entera", aliases=None)
        result = build_google_phrases([p])
        assert "Leche Entera" in result

    def test_aliases_included(self, make_product):
        p = make_product(id=1, name="Leche Entera", aliases="leche|milk|lait")
        result = build_google_phrases([p])
        assert "leche" in result
        assert "milk" in result
        assert "lait" in result

    def test_aliases_stripped(self, make_product):
        p = make_product(id=1, name="Leche Entera", aliases=" leche | milk ")
        result = build_google_phrases([p])
        assert "leche" in result
        assert "milk" in result

    def test_empty_aliases_skipped(self, make_product):
        p = make_product(id=1, name="Leche Entera", aliases="||")
        result = build_google_phrases([p])
        # only name should appear, empty alias segments skipped
        assert "Leche Entera" in result

    def test_deduplicates_by_lowercase(self, make_product):
        # "Leche" and "leche" are the same after dedup
        p = make_product(id=1, name="Leche", aliases="Leche|leche")
        result = build_google_phrases([p])
        lowered = [r.lower() for r in result]
        assert lowered.count("leche") == 1

    def test_multiple_products(self, sample_products):
        result = build_google_phrases(sample_products)
        assert len(result) > 0
        # Product names should be present
        names = [p.name for p in sample_products if p.name]
        for name in names:
            assert name in result

    def test_none_aliases_handled(self, make_product):
        p = make_product(id=1, name="Aceite", aliases=None)
        result = build_google_phrases([p])
        assert "Aceite" in result

    def test_no_empty_strings_in_output(self, sample_products):
        result = build_google_phrases(sample_products)
        assert all(r.strip() for r in result)


# ─────────────────────────────────────────────────────────────────────────────
# build_alias_indexes
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildAliasIndexes:

    def test_returns_three_keys(self, sample_products):
        result = build_alias_indexes(sample_products)
        assert "alias_to_pids" in result
        assert "token_to_pids" in result
        assert "alias_to_products" in result

    def test_empty_list_returns_empty_dicts(self):
        result = build_alias_indexes([])
        assert result["alias_to_pids"] == {}
        assert result["token_to_pids"] == {}
        assert result["alias_to_products"] == {}

    def test_product_name_indexed(self, make_product):
        p = make_product(id=1, name="Leche Entera", aliases=None)
        result = build_alias_indexes([p])
        # normalized name should be a key
        assert "leche entera" in result["alias_to_pids"]
        assert 1 in result["alias_to_pids"]["leche entera"]

    def test_aliases_indexed(self, make_product):
        p = make_product(id=1, name="Leche Entera", aliases="leche|milk")
        result = build_alias_indexes([p])
        assert "leche" in result["alias_to_pids"]
        assert "milk" in result["alias_to_pids"]

    def test_provider_name_indexed(self, make_product):
        p = make_product(id=1, name="Leche Entera", provider_name="Makro", aliases=None)
        result = build_alias_indexes([p])
        assert "makro" in result["alias_to_pids"]

    def test_tokens_indexed(self, make_product):
        p = make_product(id=1, name="Leche Entera", aliases=None)
        result = build_alias_indexes([p])
        assert "leche" in result["token_to_pids"]
        assert "entera" in result["token_to_pids"]
        assert 1 in result["token_to_pids"]["leche"]

    def test_single_char_tokens_not_indexed(self, make_product):
        p = make_product(id=1, name="A B C", aliases=None)
        result = build_alias_indexes([p])
        # single-char tokens should not appear
        assert "a" not in result["token_to_pids"]
        assert "b" not in result["token_to_pids"]

    def test_product_without_id_skipped(self, make_product):
        p = make_product(id=None, name="Sin ID", aliases="test")
        result = build_alias_indexes([p])
        assert "sin id" not in result["alias_to_pids"]

    def test_no_duplicate_pids_per_alias(self, make_product):
        # Same product indexed twice shouldn't add duplicate pids
        p = make_product(id=1, name="Leche", aliases="leche|leche")
        result = build_alias_indexes([p])
        assert result["alias_to_pids"]["leche"].count(1) == 1

    def test_multiple_products_same_alias(self, make_product):
        p1 = make_product(id=1, name="Leche A", aliases="leche")
        p2 = make_product(id=2, name="Leche B", aliases="leche")
        result = build_alias_indexes([p1, p2])
        pids = result["alias_to_pids"]["leche"]
        assert 1 in pids
        assert 2 in pids

    def test_alias_to_products_contains_name(self, make_product):
        p = make_product(id=1, name="Leche Entera", aliases="leche")
        result = build_alias_indexes([p])
        assert "Leche Entera" in result["alias_to_products"]["leche"]


# ─────────────────────────────────────────────────────────────────────────────
# alias_suggestions
# ─────────────────────────────────────────────────────────────────────────────

class TestAliasSuggestions:

    @pytest.fixture
    def indexes(self, sample_products):
        # alias_suggestions only accepts alias_to_pids + token_to_pids
        full = build_alias_indexes(sample_products)
        return {k: full[k] for k in ("alias_to_pids", "token_to_pids")}

    def test_empty_query_returns_empty(self, indexes):
        result = alias_suggestions("", **indexes)
        assert result == []

    def test_none_query_returns_empty(self, indexes):
        result = alias_suggestions(None, **indexes)
        assert result == []

    def test_exact_alias_match(self, indexes):
        # "leche" is an alias of product id=1
        result = alias_suggestions("leche", **indexes)
        assert 1 in result

    def test_exact_name_match(self, indexes):
        result = alias_suggestions("tomate triturado", **indexes)
        assert 2 in result

    def test_token_based_match(self, indexes):
        # "aceite" is a token in "Aceite de Oliva"
        result = alias_suggestions("aceite", **indexes)
        assert 3 in result

    def test_no_match_returns_empty(self, indexes):
        result = alias_suggestions("xyznonexistentproduct", **indexes)
        assert result == []

    def test_limit_respected(self, indexes):
        result = alias_suggestions("leche", **indexes, limit=1)
        assert len(result) <= 1

    def test_single_char_query_returns_empty(self, indexes):
        # Single-char tokens are excluded, so "a" finds nothing
        result = alias_suggestions("a", **indexes)
        assert result == []

    def test_multi_token_scores_higher(self, make_product):
        """Product matching more tokens should rank first."""
        p1 = make_product(id=1, name="Pan Molde Bimbo", aliases=None)
        p2 = make_product(id=2, name="Pan",              aliases=None)
        full = build_alias_indexes([p1, p2])
        idx = {k: full[k] for k in ("alias_to_pids", "token_to_pids")}
        result = alias_suggestions("pan molde", **idx)
        # p1 matches both "pan" AND "molde" → should rank before p2
        assert result[0] == 1

    def test_returns_list_of_ints(self, indexes):
        result = alias_suggestions("leche", **indexes)
        assert all(isinstance(pid, int) for pid in result)


# ─────────────────────────────────────────────────────────────────────────────
# add_provider_column_fast
# ─────────────────────────────────────────────────────────────────────────────

class TestAddProviderColumnFast:

    @pytest.fixture
    def pid_map(self):
        return {1: "Makro", 2: "Makro", 3: "Sysco", 4: "Bimbo"}

    @pytest.fixture
    def name_map(self):
        from core.normalization import normalize_text
        return {
            normalize_text("Leche Entera"):    "Makro",
            normalize_text("Tomate Triturado"): "Makro",
            normalize_text("Aceite de Oliva"):  "Sysco",
        }

    def test_fills_provider_from_pid(self, sample_df, pid_map, name_map):
        result = add_provider_column_fast(sample_df, pid_to_provider=pid_map, norm_name_to_provider=name_map)
        assert result.loc[0, "provider"] == "Makro"
        assert result.loc[1, "provider"] == "Makro"
        assert result.loc[3, "provider"] == "Bimbo"

    def test_fallback_to_name_when_pid_missing(self, sample_df, pid_map, name_map):
        # row index 2 has no matched_product_id (None), falls back to name
        result = add_provider_column_fast(sample_df, pid_to_provider=pid_map, norm_name_to_provider=name_map)
        assert result.loc[2, "provider"] == "Sysco"

    def test_provider_column_is_string_dtype(self, sample_df, pid_map, name_map):
        result = add_provider_column_fast(sample_df, pid_to_provider=pid_map, norm_name_to_provider=name_map)
        assert str(result["provider"].dtype) == "string"

    def test_missing_pid_column_handled(self, name_map):
        df = pd.DataFrame({"matched_name": ["Leche Entera", "Tomate Triturado"]})
        result = add_provider_column_fast(df, pid_to_provider={}, norm_name_to_provider=name_map)
        assert "provider" in result.columns

    def test_missing_name_column_handled(self, pid_map):
        df = pd.DataFrame({"matched_product_id": [1, 2]})
        result = add_provider_column_fast(df, pid_to_provider=pid_map, norm_name_to_provider={})
        assert result.loc[0, "provider"] == "Makro"

    def test_does_not_mutate_input(self, sample_df, pid_map, name_map):
        original_cols = list(sample_df.columns)
        add_provider_column_fast(sample_df, pid_to_provider=pid_map, norm_name_to_provider=name_map)
        assert list(sample_df.columns) == original_cols

    def test_empty_df_returns_empty_with_provider_col(self, pid_map, name_map):
        df = pd.DataFrame()
        result = add_provider_column_fast(df, pid_to_provider=pid_map, norm_name_to_provider=name_map)
        assert isinstance(result, pd.DataFrame)


# ─────────────────────────────────────────────────────────────────────────────
# apply_unit_choice
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyUnitChoice:

    def test_regular_units_normalized(self):
        df = pd.DataFrame({"unit": ["kg", "unidad", "pack"], "unit_custom": ["", "", ""]})
        result = apply_unit_choice(df)
        assert list(result["unit"]) == ["kg", "unit", "pack"]

    def test_other_uses_unit_custom(self):
        df = pd.DataFrame({"unit": ["Other…"], "unit_custom": ["botella"]})
        result = apply_unit_choice(df)
        # "botella" is not in UNIT_SYNONYMS, returned as-is (normalized)
        assert result.loc[0, "unit"] == "botella"

    def test_other_with_empty_custom_defaults_to_unit(self):
        df = pd.DataFrame({"unit": ["Other…"], "unit_custom": [""]})
        result = apply_unit_choice(df)
        assert result.loc[0, "unit"] == "unit"

    def test_other_with_none_custom_defaults_to_unit(self):
        df = pd.DataFrame({"unit": ["Other…"], "unit_custom": [None]})
        result = apply_unit_choice(df)
        assert result.loc[0, "unit"] == "unit"

    def test_missing_unit_column_created(self):
        df = pd.DataFrame({"quantity": [1.0]})
        result = apply_unit_choice(df)
        assert "unit" in result.columns

    def test_missing_unit_custom_column_handled(self):
        df = pd.DataFrame({"unit": ["kg"]})
        result = apply_unit_choice(df)
        assert result.loc[0, "unit"] == "kg"

    def test_does_not_mutate_input(self, sample_df):
        original = sample_df.copy()
        apply_unit_choice(sample_df)
        pd.testing.assert_frame_equal(sample_df, original)

    def test_empty_unit_becomes_default(self):
        df = pd.DataFrame({"unit": [""], "unit_custom": [""]})
        result = apply_unit_choice(df)
        assert result.loc[0, "unit"] == "unit"

    def test_kg_alias_normalized(self):
        df = pd.DataFrame({"unit": ["kilos"], "unit_custom": [""]})
        result = apply_unit_choice(df)
        assert result.loc[0, "unit"] == "kg"

    def test_other_with_kg_custom(self):
        df = pd.DataFrame({"unit": ["Other…"], "unit_custom": ["kilos"]})
        result = apply_unit_choice(df)
        assert result.loc[0, "unit"] == "kg"

    def test_mixed_rows(self, sample_df):
        result = apply_unit_choice(sample_df)
        assert "unit" in result.columns
        # row 2 had "Other…" + "botella" custom
        assert result.loc[2, "unit"] == "botella"
        # row 0 had "unit" → stays "unit"
        assert result.loc[0, "unit"] == "unit"
