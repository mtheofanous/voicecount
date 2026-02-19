"""
test_normalization.py — tests for core/normalization.py

Covers: normalize_text, normalize_unit, canonicalize_headers,
        cleanup_asr_transcript, match_keys
"""
from __future__ import annotations
import pytest
from core.normalization import (
    normalize_text,
    normalize_unit,
    canonicalize_headers,
    cleanup_asr_transcript,
    match_keys,
    UNIT_SYNONYMS,
    HEADER_SYNONYMS,
)


# ─────────────────────────────────────────────────────────────────────────────
# normalize_text
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeText:

    def test_none_returns_empty(self):
        assert normalize_text(None) == ""

    def test_empty_string_returns_empty(self):
        assert normalize_text("") == ""

    def test_whitespace_only_returns_empty(self):
        assert normalize_text("   ") == ""

    def test_lowercases(self):
        assert normalize_text("LECHE") == "leche"

    def test_strips_leading_trailing_spaces(self):
        assert normalize_text("  leche  ") == "leche"

    def test_collapses_multiple_spaces(self):
        assert normalize_text("leche   entera") == "leche entera"

    def test_preserves_spanish_enie(self):
        result = normalize_text("jalapeño")
        assert "ñ" in result
        assert result == "jalapeño"

    def test_preserves_enie_uppercase(self):
        result = normalize_text("JALAPEÑO")
        assert "ñ" in result

    def test_removes_greek_tonos(self):
        # Greek with accent: άρτος → αρτος
        assert normalize_text("άρτος") == "αρτος"

    def test_removes_greek_accents(self):
        assert normalize_text("Γάλα") == "γαλα"

    def test_numeric_value_converted(self):
        assert normalize_text(123) == "123"

    def test_float_converted(self):
        assert normalize_text(1.5) == "1.5"

    def test_mixed_case_with_spaces(self):
        assert normalize_text("  Leche Entera  ") == "leche entera"

    def test_spanish_accented_vowels_removed(self):
        result = normalize_text("café")
        assert result == "cafe"

    def test_preserves_numbers_in_string(self):
        assert normalize_text("Tomate 400g") == "tomate 400g"


# ─────────────────────────────────────────────────────────────────────────────
# normalize_unit
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeUnit:

    def test_none_returns_unit(self):
        assert normalize_unit(None) == "unit"

    def test_empty_returns_unit(self):
        assert normalize_unit("") == "unit"

    def test_whitespace_returns_unit(self):
        assert normalize_unit("   ") == "unit"

    def test_kg_variations(self):
        for v in ["kg", "KG", "kilo", "kilos", "kgs"]:
            assert normalize_unit(v) == "kg", f"Failed for: {v!r}"

    def test_greek_kg_variations(self):
        for v in ["κιλο", "κιλά", "κιλα"]:
            assert normalize_unit(v) == "kg", f"Failed for: {v!r}"

    def test_gram_variations(self):
        for v in ["g", "gr", "gram", "grams", "gramo", "gramos"]:
            assert normalize_unit(v) == "g", f"Failed for: {v!r}"

    def test_unit_variations(self):
        for v in ["unit", "unidad", "ud", "uds", "pieza", "piezas"]:
            assert normalize_unit(v) == "unit", f"Failed for: {v!r}"

    def test_pack_variations(self):
        for v in ["pack", "packs", "paquete", "paquetes"]:
            assert normalize_unit(v) == "pack", f"Failed for: {v!r}"

    def test_box_variations(self):
        for v in ["box", "boxes", "caja", "cajas"]:
            assert normalize_unit(v) == "box", f"Failed for: {v!r}"

    def test_unknown_unit_returned_normalized(self):
        # Unknown units come back normalized (lowercased etc.) but not mapped
        result = normalize_unit("botella")
        assert result == "botella"  # not in UNIT_SYNONYMS, returned as-is

    def test_case_insensitive(self):
        assert normalize_unit("KG") == "kg"
        assert normalize_unit("Pack") == "pack"

    def test_all_synonyms_map_to_canonical(self):
        """Every key in UNIT_SYNONYMS should round-trip cleanly."""
        for key, expected in UNIT_SYNONYMS.items():
            result = normalize_unit(key)
            assert result == expected, f"UNIT_SYNONYMS[{key!r}] → expected {expected!r}, got {result!r}"


# ─────────────────────────────────────────────────────────────────────────────
# canonicalize_headers
# ─────────────────────────────────────────────────────────────────────────────

class TestCanonicalizeHeaders:

    def test_empty_list(self):
        assert canonicalize_headers([]) == []

    def test_known_spanish_headers(self):
        result = canonicalize_headers(["nombre", "unidad", "cantidad"])
        assert "name" in result
        assert "unit" in result
        assert "quantity" in result

    def test_known_english_headers(self):
        result = canonicalize_headers(["name", "unit", "quantity", "price"])
        assert result == ["name", "unit", "quantity", "price"]

    def test_unknown_headers_returned_normalized(self):
        result = canonicalize_headers(["custom_field"])
        assert result == ["custom_field"]

    def test_mixed_case_headers(self):
        result = canonicalize_headers(["Nombre", "UNIDAD"])
        assert "name" in result
        assert "unit" in result

    def test_provider_headers(self):
        result = canonicalize_headers(["proveedor", "email"])
        assert "provider_name" in result
        assert "provider_email" in result

    def test_preserves_order(self):
        cols = ["nombre", "cantidad", "precio"]
        result = canonicalize_headers(cols)
        assert len(result) == 3
        assert result[0] == "name"
        assert result[1] == "quantity"
        assert result[2] == "price"


# ─────────────────────────────────────────────────────────────────────────────
# cleanup_asr_transcript
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanupAsrTranscript:

    def test_none_returns_empty(self):
        assert cleanup_asr_transcript(None) == ""

    def test_empty_returns_empty(self):
        assert cleanup_asr_transcript("") == ""

    def test_removes_leading_zero_prefix(self):
        # "0, 3 αλάιμ" → "3 αλάιμ"
        assert cleanup_asr_transcript("0, 3 alime") == "3 alime"

    def test_removes_zero_with_dot(self):
        assert cleanup_asr_transcript("0. 5 tomates") == "5 tomates"

    def test_removes_bullet_middot(self):
        result = cleanup_asr_transcript("· leche 2")
        assert "·" not in result

    def test_removes_bullet_dot(self):
        result = cleanup_asr_transcript("• aceite 1")
        assert "•" not in result

    def test_collapses_multiple_spaces(self):
        result = cleanup_asr_transcript("leche   2   kg")
        assert "  " not in result

    def test_normal_transcript_unchanged(self):
        text = "leche 2 litros, tomate 1 kg"
        assert cleanup_asr_transcript(text) == text

    def test_strips_leading_trailing_whitespace(self):
        assert cleanup_asr_transcript("  leche 2  ") == "leche 2"


# ─────────────────────────────────────────────────────────────────────────────
# match_keys
# ─────────────────────────────────────────────────────────────────────────────

class TestMatchKeys:

    def test_none_returns_empty_list(self):
        assert match_keys(None) == []

    def test_empty_returns_empty_list(self):
        assert match_keys("") == []

    def test_whitespace_returns_empty_list(self):
        assert match_keys("   ") == []

    def test_simple_ascii_returns_list(self):
        keys = match_keys("leche")
        assert isinstance(keys, list)
        assert len(keys) >= 1
        assert "leche" in keys

    def test_greek_text_adds_transliteration(self):
        # Greek "γαλα" should produce at least the original key
        keys = match_keys("γάλα")
        assert isinstance(keys, list)
        assert len(keys) >= 1

    def test_accented_spanish_produces_keys(self):
        keys = match_keys("café")
        assert isinstance(keys, list)
        assert len(keys) >= 1

    def test_returns_only_nonempty_strings(self):
        keys = match_keys("Leche Entera")
        assert all(isinstance(k, str) and k for k in keys)

    def test_no_duplicates(self):
        keys = match_keys("leche")
        assert len(keys) == len(set(keys))
