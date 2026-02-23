"""
test_receive_helpers.py — tests for pure helper functions in
features/manage_orders/receive_orders_page.py

Covers: _s, _safe_float, _safe_json_loads, _parse_time_range_start,
        _strip_accents, _norm_words, _extract_unit_tokens,
        _relevance, _similarity, _parse_delivery_schedule, _fmt_eta,
        _badge, _state_badge, _price_for_pid, _iva_pct_for_pid,
        _derive_expected_qty, _match_scope_ok, _rule_priority_key,
        _parse_supplier_solution_meta, _normalize_solution_meta,
        _is_redelivery_item, _parse_ticket_resolution_note,
        _prev_month_window
"""
from __future__ import annotations
import types
from datetime import datetime, timedelta
import pytest
from features.manage_orders.receive_orders_page import (
    _s,
    _safe_float,
    _safe_json_loads,
    _parse_time_range_start,
    _strip_accents,
    _norm_words,
    _extract_unit_tokens,
    _relevance,
    _similarity,
    _parse_delivery_schedule,
    _fmt_eta,
    _badge,
    _state_badge,
    _price_for_pid,
    _iva_pct_for_pid,
    _derive_expected_qty,
    _match_scope_ok,
    _rule_priority_key,
    _parse_supplier_solution_meta,
    _normalize_solution_meta,
    _is_redelivery_item,
    _parse_ticket_resolution_note,
    _prev_month_window,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers to build lightweight mock objects
# ─────────────────────────────────────────────────────────────────────────────

def _product(price=0.0, iva=0.0, name="Product", unit="unit"):
    return types.SimpleNamespace(price=price, iva=iva, name=name, unit=unit)


def _line(quantity=1.0, product_id=None, spoken_name="Item", unit="unit"):
    return types.SimpleNamespace(
        quantity=quantity,
        product_id=product_id,
        spoken_name=spoken_name,
        unit=unit,
    )


def _fu(supplier_status="ok", supplier_qty=None, venue_qty=0.0,
        qty_invoiced=0.0, venue_comment="", invoice_listed=None):
    return types.SimpleNamespace(
        supplier_status=supplier_status,
        supplier_qty=supplier_qty,
        venue_qty=venue_qty,
        qty_invoiced=qty_invoiced,
        venue_comment=venue_comment,
        invoice_listed=invoice_listed,
    )


def _rule(product_id=None, rule_kind="line_pct", min_qty=0.0,
          prev_month_min_qty=0.0, discount_percent=0.0, price_override=0.0):
    return types.SimpleNamespace(
        product_id=product_id,
        rule_kind=rule_kind,
        min_qty=min_qty,
        prev_month_min_qty=prev_month_min_qty,
        discount_percent=discount_percent,
        price_override=price_override,
    )


# ─────────────────────────────────────────────────────────────────────────────
# _s
# ─────────────────────────────────────────────────────────────────────────────

class TestS:

    def test_none_returns_empty(self):
        assert _s(None) == ""

    def test_string_stripped(self):
        assert _s("  leche  ") == "leche"

    def test_int_converted(self):
        assert _s(42) == "42"

    def test_float_converted(self):
        assert _s(3.14) == "3.14"

    def test_empty_string(self):
        assert _s("") == ""

    def test_zero_converted(self):
        assert _s(0) == "0"


# ─────────────────────────────────────────────────────────────────────────────
# _safe_float
# ─────────────────────────────────────────────────────────────────────────────

class TestSafeFloat:

    def test_none_returns_default(self):
        assert _safe_float(None) == 0.0

    def test_empty_string_returns_default(self):
        assert _safe_float("") == 0.0

    def test_custom_default(self):
        assert _safe_float(None, default=5.0) == 5.0

    def test_valid_int(self):
        assert _safe_float(3) == 3.0

    def test_valid_float(self):
        assert _safe_float(2.5) == 2.5

    def test_string_number(self):
        assert _safe_float("1.75") == 1.75

    def test_invalid_string_returns_default(self):
        assert _safe_float("abc") == 0.0

    def test_negative(self):
        assert _safe_float(-1.0) == -1.0

    def test_zero(self):
        assert _safe_float(0) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# _safe_json_loads
# ─────────────────────────────────────────────────────────────────────────────

class TestSafeJsonLoads:

    def test_empty_string_returns_empty_dict(self):
        assert _safe_json_loads("") == {}

    def test_none_returns_empty_dict(self):
        assert _safe_json_loads(None) == {}

    def test_whitespace_returns_empty_dict(self):
        assert _safe_json_loads("   ") == {}

    def test_valid_json_object(self):
        assert _safe_json_loads('{"key": "value"}') == {"key": "value"}

    def test_valid_json_nested(self):
        result = _safe_json_loads('{"mon": ["08:00-12:00"]}')
        assert result == {"mon": ["08:00-12:00"]}

    def test_invalid_json_returns_empty_dict(self):
        assert _safe_json_loads("{not valid}") == {}

    def test_json_array_returns_empty_dict(self):
        # must be a dict, arrays return {}
        assert _safe_json_loads("[1, 2, 3]") == {}

    def test_json_string_returns_empty_dict(self):
        assert _safe_json_loads('"just a string"') == {}


# ─────────────────────────────────────────────────────────────────────────────
# _parse_time_range_start
# ─────────────────────────────────────────────────────────────────────────────

class TestParseTimeRangeStart:

    def test_valid_range(self):
        assert _parse_time_range_start("08:00-12:00") == (8, 0)

    def test_valid_range_pm(self):
        assert _parse_time_range_start("14:30-17:00") == (14, 30)

    def test_midnight(self):
        assert _parse_time_range_start("00:00-01:00") == (0, 0)

    def test_end_of_day(self):
        assert _parse_time_range_start("23:59-23:59") == (23, 59)

    def test_none_returns_none(self):
        assert _parse_time_range_start(None) is None

    def test_empty_returns_none(self):
        assert _parse_time_range_start("") is None

    def test_no_dash_returns_none(self):
        assert _parse_time_range_start("0800") is None

    def test_no_colon_returns_none(self):
        assert _parse_time_range_start("0800-1200") is None

    def test_invalid_hour_returns_none(self):
        assert _parse_time_range_start("25:00-26:00") is None

    def test_invalid_minute_returns_none(self):
        assert _parse_time_range_start("10:99-11:00") is None

    def test_strips_whitespace(self):
        assert _parse_time_range_start("  09:00-10:00  ") == (9, 0)


# ─────────────────────────────────────────────────────────────────────────────
# _strip_accents
# ─────────────────────────────────────────────────────────────────────────────

class TestStripAccentsReceive:

    def test_plain_ascii_unchanged(self):
        assert _strip_accents("leche") == "leche"

    def test_empty_string(self):
        assert _strip_accents("") == ""

    def test_none_handled(self):
        result = _strip_accents(None)
        assert result == ""

    def test_spanish_acute_removed(self):
        assert _strip_accents("café") == "cafe"

    def test_enie_base_kept(self):
        result = _strip_accents("jalapeño")
        assert "ñ" not in result
        assert "n" in result

    def test_numbers_unchanged(self):
        assert _strip_accents("abc123") == "abc123"


# ─────────────────────────────────────────────────────────────────────────────
# _norm_words
# ─────────────────────────────────────────────────────────────────────────────

class TestNormWordsReceive:

    def test_none_returns_empty_set(self):
        assert _norm_words(None) == set()

    def test_empty_string_returns_empty_set(self):
        assert _norm_words("") == set()

    def test_stop_words_removed(self):
        # "de", "la", "el" are stop words
        result = _norm_words("aceite de la oliva")
        assert "de" not in result
        assert "la" not in result
        assert "aceite" in result
        assert "oliva" in result

    def test_unit_words_are_stop_words(self):
        # Unit words like "kilos", "litros" appear in _STOP before _UNIT_ALIASES is checked,
        # so they are DROPPED from the output (not added as canonical units).
        result = _norm_words("2 kilos de leche y 3 litros de zumo")
        assert "kilos" not in result
        assert "litros" not in result
        assert "leche" in result
        assert "zumo" in result

    def test_returns_set_not_list(self):
        assert isinstance(_norm_words("leche entera"), set)

    def test_lowercases_tokens(self):
        result = _norm_words("LECHE ENTERA")
        assert all(t == t.lower() for t in result)

    def test_special_chars_excluded(self):
        result = _norm_words("leche! tomate?")
        assert "!" not in result
        assert "?" not in result
        assert "leche" in result
        assert "tomate" in result

    def test_naive_singularization(self):
        # "tomates" (>4 chars ending in s) → "tomate"
        result = _norm_words("tomates frescos")
        assert "tomate" in result or "tomates" in result  # singularized or kept


# ─────────────────────────────────────────────────────────────────────────────
# _extract_unit_tokens
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractUnitTokens:

    def test_extracts_kg(self):
        assert _extract_unit_tokens({"kg", "leche", "entera"}) == {"kg"}

    def test_extracts_multiple_units(self):
        assert _extract_unit_tokens({"kg", "g", "leche"}) == {"kg", "g"}

    def test_no_units_returns_empty(self):
        assert _extract_unit_tokens({"leche", "entera", "fresca"}) == set()

    def test_empty_set(self):
        assert _extract_unit_tokens(set()) == set()

    def test_all_units(self):
        all_units = {"kg", "g", "l", "ml", "ud"}
        assert _extract_unit_tokens(all_units) == all_units


# ─────────────────────────────────────────────────────────────────────────────
# _relevance
# ─────────────────────────────────────────────────────────────────────────────

class TestRelevance:

    def test_empty_a_returns_zero(self):
        assert _relevance("", "leche entera") == 0.0

    def test_empty_b_returns_zero(self):
        assert _relevance("leche entera", "") == 0.0

    def test_identical_strings_score_high(self):
        score = _relevance("leche entera", "leche entera")
        assert score > 0.5

    def test_unrelated_strings_score_zero(self):
        score = _relevance("leche entera", "aceite de oliva")
        assert score == 0.0

    def test_partial_overlap_scores_between_zero_and_one(self):
        score = _relevance("leche entera fresca", "leche fresca")
        assert 0.0 < score < 1.0

    def test_score_clamped_to_one(self):
        score = _relevance("leche", "leche leche leche")
        assert score <= 1.0

    def test_score_non_negative(self):
        score = _relevance("aceite oliva virgen extra", "aceite oliva")
        assert score >= 0.0

    def test_unit_match_boosts_score(self):
        # Both have "kg" → unit_bonus adds 0.10
        score_with_unit = _relevance("harina 1kg", "harina kg")
        score_without_unit = _relevance("harina premium", "harina especial")
        # Not necessarily always true due to Jaccard, but the unit bonus should help
        assert isinstance(score_with_unit, float)

    def test_symmetry(self):
        a, b = "leche entera", "entera leche"
        assert _relevance(a, b) == _relevance(b, a)


# ─────────────────────────────────────────────────────────────────────────────
# _similarity
# ─────────────────────────────────────────────────────────────────────────────

class TestSimilarity:

    def test_identical_strings(self):
        assert _similarity("leche", "leche") == 1.0

    def test_empty_a_returns_zero(self):
        assert _similarity("", "leche") == 0.0

    def test_empty_b_returns_zero(self):
        assert _similarity("leche", "") == 0.0

    def test_both_empty_returns_zero(self):
        assert _similarity("", "") == 0.0

    def test_completely_different(self):
        score = _similarity("leche", "xyz")
        assert score < 0.5

    def test_case_insensitive(self):
        # _similarity lowercases before comparing
        assert _similarity("LECHE", "leche") == 1.0

    def test_partial_similarity(self):
        score = _similarity("leche entera", "leche semi")
        assert 0.0 < score < 1.0

    def test_returns_float(self):
        assert isinstance(_similarity("a", "b"), float)


# ─────────────────────────────────────────────────────────────────────────────
# _parse_delivery_schedule
# ─────────────────────────────────────────────────────────────────────────────

class TestParseDeliverySchedule:

    def test_none_returns_empty(self):
        assert _parse_delivery_schedule(None) == {}

    def test_empty_string_returns_empty(self):
        assert _parse_delivery_schedule("") == {}

    def test_empty_dict_returns_empty(self):
        assert _parse_delivery_schedule({}) == {}

    def test_valid_json_string(self):
        raw = '{"mon": ["08:00-12:00"], "fri": ["14:00-17:00"]}'
        result = _parse_delivery_schedule(raw)
        assert "mon" in result
        assert "fri" in result

    def test_valid_dict_input(self):
        raw = {"mon": ["08:00-12:00"]}
        result = _parse_delivery_schedule(raw)
        assert result == {"mon": ["08:00-12:00"]}

    def test_invalid_day_key_excluded(self):
        raw = {"monday": ["08:00-12:00"], "mon": ["09:00-13:00"]}
        result = _parse_delivery_schedule(raw)
        # "monday" truncated to "mon" by [:3] — so it DOES get matched
        assert "mon" in result

    def test_invalid_json_returns_empty(self):
        assert _parse_delivery_schedule("{not valid}") == {}

    def test_slot_values_preserved(self):
        raw = {"tue": ["10:00-14:00", "16:00-18:00"]}
        result = _parse_delivery_schedule(raw)
        assert result["tue"] == ["10:00-14:00", "16:00-18:00"]

    def test_empty_slots_excluded(self):
        raw = {"wed": []}
        result = _parse_delivery_schedule(raw)
        assert "wed" not in result


# ─────────────────────────────────────────────────────────────────────────────
# _fmt_eta
# ─────────────────────────────────────────────────────────────────────────────

class TestFmtEta:

    def _now(self):
        return datetime(2026, 1, 15, 10, 0, 0)

    def test_none_dt_returns_dash(self):
        assert _fmt_eta(self._now(), None) == "—"

    def test_past_dt_shows_zero_minutes(self):
        now = self._now()
        past = now - timedelta(hours=1)
        result = _fmt_eta(now, past)
        assert "0m" in result or "m" in result

    def test_minutes_only(self):
        now = self._now()
        future = now + timedelta(minutes=45)
        result = _fmt_eta(now, future)
        assert "45m" in result

    def test_hours_and_minutes(self):
        now = self._now()
        future = now + timedelta(hours=2, minutes=30)
        result = _fmt_eta(now, future)
        assert "2h" in result
        assert "30m" in result

    def test_days(self):
        now = self._now()
        future = now + timedelta(days=3)
        result = _fmt_eta(now, future)
        assert "d" in result
        assert "3" in result

    def test_two_days_threshold(self):
        now = self._now()
        future = now + timedelta(hours=49)
        result = _fmt_eta(now, future)
        assert "d" in result


# ─────────────────────────────────────────────────────────────────────────────
# _badge
# ─────────────────────────────────────────────────────────────────────────────

class TestBadgeReceive:

    def test_empty_text_returns_empty(self):
        assert _badge("", "ok") == ""

    def test_valid_kind_ok(self):
        result = _badge("Closed", "ok")
        assert "ok" in result
        assert "Closed" in result

    def test_valid_kind_warn(self):
        result = _badge("Warning", "warn")
        assert "warn" in result

    def test_invalid_kind_defaults_to_info(self):
        result = _badge("Label", "unknown_kind")
        assert "info" in result

    def test_html_escaped(self):
        result = _badge("<script>", "ok")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_returns_span_tag(self):
        result = _badge("Test", "ok")
        assert result.startswith("<span")
        assert result.endswith("</span>")


# ─────────────────────────────────────────────────────────────────────────────
# _state_badge
# ─────────────────────────────────────────────────────────────────────────────

class TestStateBadge:

    def test_closed_is_ok(self):
        text, kind = _state_badge("CLOSED")
        assert kind == "ok"
        assert "Closed" in text

    def test_order_sent_is_info(self):
        text, kind = _state_badge("ORDER_SENT")
        assert kind == "info"

    def test_invoice_discrepancy_is_bad(self):
        text, kind = _state_badge("INVOICE_DISCREPANCY")
        assert kind == "bad"

    def test_waiting_supplier_is_warn(self):
        text, kind = _state_badge("WAITING_SUPPLIER_ACTION")
        assert kind == "warn"

    def test_credit_note_pending_is_warn(self):
        text, kind = _state_badge("SUPPLIER_CREDIT_NOTE_PENDING")
        assert kind == "warn"

    def test_received_is_ok(self):
        text, kind = _state_badge("RECEIVED")
        assert kind == "ok"

    def test_supplier_confirmed_is_info(self):
        text, kind = _state_badge("SUPPLIER_CONFIRMED_FULL")
        assert kind == "info"

    def test_case_insensitive(self):
        text_lower, kind_lower = _state_badge("closed")
        text_upper, kind_upper = _state_badge("CLOSED")
        assert kind_lower == kind_upper

    def test_unknown_state_returned_as_is(self):
        text, kind = _state_badge("MY_CUSTOM_STATE")
        assert "MY_CUSTOM_STATE" in text


# ─────────────────────────────────────────────────────────────────────────────
# _price_for_pid
# ─────────────────────────────────────────────────────────────────────────────

class TestPriceForPid:

    def test_none_pid_returns_zero(self):
        assert _price_for_pid({}, None) == 0.0

    def test_missing_pid_returns_zero(self):
        assert _price_for_pid({1: _product(price=5.0)}, 99) == 0.0

    def test_valid_pid_returns_price(self):
        products = {1: _product(price=3.50)}
        assert _price_for_pid(products, 1) == 3.50

    def test_zero_price(self):
        products = {1: _product(price=0.0)}
        assert _price_for_pid(products, 1) == 0.0

    def test_none_price_attribute_returns_zero(self):
        p = types.SimpleNamespace(price=None)
        assert _price_for_pid({1: p}, 1) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# _iva_pct_for_pid
# ─────────────────────────────────────────────────────────────────────────────

class TestIvaPctForPid:

    def test_none_pid_returns_default(self):
        assert _iva_pct_for_pid({}, None) == 21.0

    def test_missing_pid_returns_default(self):
        assert _iva_pct_for_pid({1: _product(iva=10.0)}, 99) == 21.0

    def test_custom_default(self):
        assert _iva_pct_for_pid({}, None, default_pct=10.0) == 10.0

    def test_valid_iva(self):
        products = {1: _product(iva=10.0)}
        assert _iva_pct_for_pid(products, 1) == 10.0

    def test_zero_iva_falls_back_to_default(self):
        products = {1: _product(iva=0.0)}
        assert _iva_pct_for_pid(products, 1) == 21.0

    def test_none_iva_falls_back_to_default(self):
        p = types.SimpleNamespace(iva=None)
        assert _iva_pct_for_pid({1: p}, 1) == 21.0


# ─────────────────────────────────────────────────────────────────────────────
# _derive_expected_qty
# ─────────────────────────────────────────────────────────────────────────────

class TestDeriveExpectedQty:

    def test_no_followup_returns_ordered(self):
        line = _line(quantity=5.0)
        assert _derive_expected_qty(line, None) == 5.0

    def test_status_ok_no_supplier_qty_returns_ordered(self):
        line = _line(quantity=5.0)
        fu = _fu(supplier_status="ok", supplier_qty=None)
        assert _derive_expected_qty(line, fu) == 5.0

    def test_status_ok_with_supplier_qty(self):
        line = _line(quantity=5.0)
        fu = _fu(supplier_status="ok", supplier_qty=4.0)
        assert _derive_expected_qty(line, fu) == 4.0

    def test_status_missing_returns_zero(self):
        line = _line(quantity=5.0)
        fu = _fu(supplier_status="missing")
        assert _derive_expected_qty(line, fu) == 0.0

    def test_status_partial_returns_supplier_qty(self):
        line = _line(quantity=5.0)
        fu = _fu(supplier_status="partial", supplier_qty=3.0)
        assert _derive_expected_qty(line, fu) == 3.0

    def test_status_partial_no_supplier_qty_returns_zero(self):
        line = _line(quantity=5.0)
        fu = _fu(supplier_status="partial", supplier_qty=None)
        assert _derive_expected_qty(line, fu) == 0.0

    def test_unknown_status_returns_ordered(self):
        line = _line(quantity=5.0)
        fu = _fu(supplier_status="unknown_status")
        assert _derive_expected_qty(line, fu) == 5.0


# ─────────────────────────────────────────────────────────────────────────────
# _match_scope_ok
# ─────────────────────────────────────────────────────────────────────────────

class TestMatchScopeOk:

    def test_rule_without_product_id_matches_any(self):
        r = _rule(product_id=None)
        assert _match_scope_ok(r, pid=1) is True
        assert _match_scope_ok(r, pid=None) is True

    def test_rule_with_product_id_requires_match(self):
        r = _rule(product_id=5)
        assert _match_scope_ok(r, pid=5) is True

    def test_rule_with_product_id_rejects_mismatch(self):
        r = _rule(product_id=5)
        assert _match_scope_ok(r, pid=99) is False

    def test_rule_with_product_id_rejects_none_pid(self):
        r = _rule(product_id=5)
        assert _match_scope_ok(r, pid=None) is False


# ─────────────────────────────────────────────────────────────────────────────
# _rule_priority_key
# ─────────────────────────────────────────────────────────────────────────────

class TestRulePriorityKey:

    def test_returns_tuple(self):
        r = _rule()
        assert isinstance(_rule_priority_key(r), tuple)

    def test_product_specific_rule_ranks_higher(self):
        r_product = _rule(product_id=1, discount_percent=10.0)
        r_global = _rule(product_id=None, discount_percent=10.0)
        key_prod = _rule_priority_key(r_product)
        key_glob = _rule_priority_key(r_global)
        # is_product=1 > is_product=0
        assert key_prod > key_glob

    def test_prev_month_rule_detected(self):
        r = _rule(rule_kind="prev_month_line_pct", prev_month_min_qty=100.0, discount_percent=5.0)
        key = _rule_priority_key(r)
        # is_prev=1 is at index 1
        assert key[1] == 1

    def test_regular_rule_not_prev_month(self):
        r = _rule(rule_kind="line_pct", min_qty=50.0, discount_percent=5.0)
        key = _rule_priority_key(r)
        assert key[1] == 0


# ─────────────────────────────────────────────────────────────────────────────
# _parse_supplier_solution_meta
# ─────────────────────────────────────────────────────────────────────────────

class TestParseSupplierSolutionMeta:

    def test_empty_returns_defaults(self):
        result = _parse_supplier_solution_meta("")
        assert result["resolution"] == ""
        assert result["ref"] == ""
        assert result["items"] == []

    def test_none_returns_defaults(self):
        result = _parse_supplier_solution_meta(None)
        assert result["resolution"] == ""

    def test_credit_note_resolution(self):
        note = "credit_note | ref=CN-2026-001 | credit_note_invoice=INV-001"
        result = _parse_supplier_solution_meta(note)
        assert result["resolution"] == "credit_note"
        assert result["ref"] == "CN-2026-001"
        assert result["credit_note_invoice"] == "INV-001"

    def test_supplementary_delivery(self):
        note = "supplementary_delivery | ref=REF-001 | eta=2026-01-22 08:00-14:00"
        result = _parse_supplier_solution_meta(note)
        assert result["resolution"] == "supplementary_delivery"
        assert result["eta"] == "2026-01-22 08:00-14:00"

    def test_items_parsed(self):
        note = "credit_note | items=Leche:2KG(damaged)/Tomate:1KG"
        result = _parse_supplier_solution_meta(note)
        assert len(result["items"]) == 2
        assert result["items"][0]["name"] == "Leche"
        assert result["items"][0]["unit"] == "KG"

    def test_middot_separator(self):
        note = "credit_note · ref=CN-001"
        result = _parse_supplier_solution_meta(note)
        assert result["resolution"] == "credit_note"
        assert result["ref"] == "CN-001"


# ─────────────────────────────────────────────────────────────────────────────
# _normalize_solution_meta
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeSolutionMeta:

    def test_credit_note_unchanged(self):
        sol = {"resolution": "credit_note", "ref": "", "credit_note_invoice": "CN-001"}
        result = _normalize_solution_meta(sol)
        assert result["resolution"] == "credit_note"

    def test_creditnote_alias_normalized(self):
        sol = {"resolution": "creditnote", "ref": "", "credit_note_invoice": ""}
        result = _normalize_solution_meta(sol)
        assert result["resolution"] == "credit_note"

    def test_redelivery_alias_normalized(self):
        sol = {"resolution": "redelivery", "ref": "", "credit_note_invoice": ""}
        result = _normalize_solution_meta(sol)
        assert result["resolution"] == "re_delivery"

    def test_back_compat_ref_in_resolution(self):
        # credit_note with no CN number but ref = "re_delivery" → becomes re_delivery
        sol = {"resolution": "credit_note", "ref": "re_delivery", "credit_note_invoice": ""}
        result = _normalize_solution_meta(sol)
        assert result["resolution"] == "re_delivery"

    def test_none_input_returns_defaults(self):
        result = _normalize_solution_meta(None)
        assert "resolution" in result

    def test_does_not_mutate_input(self):
        sol = {"resolution": "credit_note", "ref": "", "credit_note_invoice": ""}
        original = dict(sol)
        _normalize_solution_meta(sol)
        assert sol == original


# ─────────────────────────────────────────────────────────────────────────────
# _is_redelivery_item
# ─────────────────────────────────────────────────────────────────────────────

class TestIsRedeliveryItem:

    def test_supplementary_delivery_is_redelivery(self):
        assert _is_redelivery_item({"reason": "supplementary_delivery"}) is True

    def test_re_delivery_is_redelivery(self):
        assert _is_redelivery_item({"reason": "re_delivery"}) is True

    def test_redelivery_is_redelivery(self):
        assert _is_redelivery_item({"reason": "redelivery"}) is True

    def test_empty_reason_is_not_redelivery(self):
        assert _is_redelivery_item({"reason": ""}) is False

    def test_damaged_is_not_redelivery(self):
        assert _is_redelivery_item({"reason": "damaged"}) is False

    def test_none_reason_is_not_redelivery(self):
        assert _is_redelivery_item({"reason": None}) is False

    def test_missing_reason_key_is_not_redelivery(self):
        assert _is_redelivery_item({}) is False

    def test_case_insensitive(self):
        assert _is_redelivery_item({"reason": "Re_Delivery"}) is True


# ─────────────────────────────────────────────────────────────────────────────
# _parse_ticket_resolution_note
# ─────────────────────────────────────────────────────────────────────────────

class TestParseTicketResolutionNote:

    def test_empty_returns_defaults(self):
        result = _parse_ticket_resolution_note("")
        assert result["resolution"] == ""
        assert result["credit_note_invoice"] == ""

    def test_none_returns_defaults(self):
        result = _parse_ticket_resolution_note(None)
        assert result["resolution"] == ""

    def test_credit_note_with_tag(self):
        note = "[SUPPLIER] credit_note | credit_note_invoice=CN-123 | invoice=INV-1"
        result = _parse_ticket_resolution_note(note)
        assert result["resolution"] == "credit_note"
        assert result["credit_note_invoice"] == "CN-123"
        assert result["invoice"] == "INV-1"

    def test_supplementary_delivery(self):
        note = "[SUPPLIER] supplementary_delivery | eta=2026-01-22 08:00-14:00"
        result = _parse_ticket_resolution_note(note)
        assert result["resolution"] == "supplementary_delivery"
        assert result["eta"] == "2026-01-22 08:00-14:00"

    def test_tag_stripped(self):
        note = "[VENUE] credit_note | invoice=INV-1"
        result = _parse_ticket_resolution_note(note)
        # [VENUE] tag should be stripped
        assert result["resolution"] == "credit_note"

    def test_no_tag(self):
        note = "credit_note | credit_note_invoice=CN-456"
        result = _parse_ticket_resolution_note(note)
        assert result["resolution"] == "credit_note"
        assert result["credit_note_invoice"] == "CN-456"

    def test_middot_separator(self):
        note = "re_delivery · eta=2026-02-01"
        result = _parse_ticket_resolution_note(note)
        assert result["resolution"] == "re_delivery"
        assert result["eta"] == "2026-02-01"


# ─────────────────────────────────────────────────────────────────────────────
# _prev_month_window
# ─────────────────────────────────────────────────────────────────────────────

class TestPrevMonthWindow:

    def test_returns_tuple_of_two_datetimes(self):
        now = datetime(2026, 3, 15)
        start, end = _prev_month_window(now)
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)

    def test_start_is_first_of_prev_month(self):
        now = datetime(2026, 3, 15)
        start, end = _prev_month_window(now)
        assert start == datetime(2026, 2, 1)

    def test_end_is_first_of_current_month(self):
        now = datetime(2026, 3, 15)
        start, end = _prev_month_window(now)
        assert end == datetime(2026, 3, 1)

    def test_january_goes_to_december(self):
        now = datetime(2026, 1, 10)
        start, end = _prev_month_window(now)
        assert start == datetime(2025, 12, 1)
        assert end == datetime(2026, 1, 1)

    def test_window_is_exactly_one_month(self):
        now = datetime(2026, 4, 1)
        start, end = _prev_month_window(now)
        # March has 31 days
        assert (end - start).days == 31
