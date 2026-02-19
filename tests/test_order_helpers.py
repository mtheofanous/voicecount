"""
test_order_helpers.py — tests for pure helper functions in
features/manage_orders/orders.py

Covers: _status_chip, _strip_accents, _norm_words,
        _remove_name_words_from_description
"""
from __future__ import annotations
import pytest
from features.manage_orders.orders import (
    _status_chip,
    _strip_accents,
    _norm_words,
    _remove_name_words_from_description,
)


# ─────────────────────────────────────────────────────────────────────────────
# _status_chip
# ─────────────────────────────────────────────────────────────────────────────

class TestStatusChip:

    @pytest.fixture(autouse=True)
    def _spanish(self, monkeypatch):
        """Force Spanish so tests are independent of runtime session state."""
        monkeypatch.setattr("core.i18n.get_lang", lambda: "es")

    def test_draft(self):
        assert _status_chip("draft") == "📝 Borrador"

    def test_ready_to_send(self):
        assert _status_chip("ready_to_send") == "📤 Listo"

    def test_pending_receive(self):
        assert _status_chip("pending_receive") == "📦 Pendiente"

    def test_final(self):
        assert _status_chip("final") == "✅ Historial"

    def test_none_returns_dash(self):
        assert _status_chip(None) == "—"

    def test_empty_returns_dash(self):
        assert _status_chip("") == "—"

    def test_unknown_status_returned_as_is(self):
        result = _status_chip("some_custom_status")
        assert result == "some_custom_status"

    def test_case_insensitive(self):
        # _status_chip lowercases the input before matching
        assert _status_chip("DRAFT") == "📝 Borrador"
        assert _status_chip("Draft") == "📝 Borrador"

    def test_all_statuses_have_emoji(self):
        statuses = ["draft", "ready_to_send", "pending_receive", "final"]
        for s in statuses:
            result = _status_chip(s)
            assert any(c in result for c in "📝📤📦✅"), f"No emoji in result for {s!r}: {result!r}"


# ─────────────────────────────────────────────────────────────────────────────
# _strip_accents
# ─────────────────────────────────────────────────────────────────────────────

class TestStripAccents:

    def test_empty_string(self):
        assert _strip_accents("") == ""

    def test_none_treated_as_empty(self):
        # None → "None" via `s or ""` fallback
        assert _strip_accents(None) == ""

    def test_plain_ascii_unchanged(self):
        assert _strip_accents("leche") == "leche"

    def test_spanish_acute_removed(self):
        assert _strip_accents("café") == "cafe"
        assert _strip_accents("médico") == "medico"

    def test_spanish_enie_stripped(self):
        # ñ → n after combining mark removal
        result = _strip_accents("jalapeño")
        assert "ñ" not in result
        assert "n" in result

    def test_greek_tonos_removed(self):
        assert _strip_accents("άρτος") == "αρτος"

    def test_mixed_accented_string(self):
        result = _strip_accents("Café Γάλα")
        assert "é" not in result
        assert "ά" not in result

    def test_numbers_unchanged(self):
        assert _strip_accents("abc123") == "abc123"


# ─────────────────────────────────────────────────────────────────────────────
# _norm_words
# ─────────────────────────────────────────────────────────────────────────────

class TestNormWords:

    def test_none_returns_empty(self):
        assert _norm_words(None) == []

    def test_empty_string_returns_empty(self):
        assert _norm_words("") == []

    def test_simple_words(self):
        result = _norm_words("leche entera")
        assert result == ["leche", "entera"]

    def test_accented_words_stripped(self):
        result = _norm_words("Café Médico")
        assert "cafe" in result
        assert "medico" in result

    def test_numbers_included(self):
        result = _norm_words("leche 2 litros")
        assert "2" in result
        assert "leche" in result
        assert "litros" in result

    def test_underscores_excluded(self):
        # [^\W_]+ means underscores are word separators, not included
        result = _norm_words("ready_to_send")
        assert "ready" in result
        assert "to" in result
        assert "send" in result
        assert "_" not in result

    def test_special_chars_excluded(self):
        result = _norm_words("leche! tomate?")
        assert result == ["leche", "tomate"]

    def test_lowercases_all(self):
        result = _norm_words("LECHE ENTERA")
        assert all(w == w.lower() for w in result)

    def test_greek_words(self):
        result = _norm_words("Γάλα φρέσκο")
        assert len(result) == 2
        # Should be accent-free + lowercase
        assert result[0] == "γαλα"
        assert result[1] == "φρεσκο"


# ─────────────────────────────────────────────────────────────────────────────
# _remove_name_words_from_description
# ─────────────────────────────────────────────────────────────────────────────

class TestRemoveNameWordsFromDescription:

    def test_empty_name_returns_desc_unchanged(self):
        assert _remove_name_words_from_description("", "fresco entero") == "fresco entero"

    def test_none_name_returns_desc(self):
        assert _remove_name_words_from_description(None, "fresco entero") == "fresco entero"

    def test_empty_desc_returns_empty(self):
        assert _remove_name_words_from_description("leche", "") == ""

    def test_none_desc_returns_empty(self):
        assert _remove_name_words_from_description("leche", None) == ""

    def test_removes_exact_name_words(self):
        result = _remove_name_words_from_description(
            "Leche Entera",
            "Leche Entera fresca de vaca"
        )
        assert "leche" not in result.lower()
        assert "entera" not in result.lower()
        assert "fresca" in result.lower()

    def test_case_insensitive_removal(self):
        result = _remove_name_words_from_description(
            "LECHE",
            "leche fresca"
        )
        assert "leche" not in result.lower()
        assert "fresca" in result.lower()

    def test_accent_insensitive_removal(self):
        result = _remove_name_words_from_description(
            "café",
            "café molido premium"
        )
        # "cafe" and "café" should both be matched
        assert "cafe" not in result.lower() and "café" not in result.lower()
        assert "molido" in result.lower()

    def test_all_words_removed_returns_empty(self):
        result = _remove_name_words_from_description("leche entera", "leche entera")
        assert result == ""

    def test_unrelated_description_unchanged(self):
        result = _remove_name_words_from_description(
            "leche",
            "fresco natural premium"
        )
        assert "fresco" in result
        assert "natural" in result
        assert "premium" in result

    def test_numbers_in_name_removed_from_desc(self):
        result = _remove_name_words_from_description(
            "botella 750",
            "botella 750 ml vidrio"
        )
        assert "ml" in result
        assert "vidrio" in result
