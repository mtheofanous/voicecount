"""
test_seguimiento_helpers.py — tests for pure helper functions in
features/seguimiento/seguimiento.py

Covers: _parse_delivery_schedule_json, _day_key_for_date, _badge,
        _product_html, _step_index, _derive_declaration_from_lines
"""
from __future__ import annotations
from datetime import datetime
import pytest
from features.seguimiento.seguimiento import (
    _parse_delivery_schedule_json,
    _day_key_for_date,
    _badge,
    _product_html,
    _step_index,
    _derive_declaration_from_lines,
)


# ─────────────────────────────────────────────────────────────────────────────
# _parse_delivery_schedule_json
# ─────────────────────────────────────────────────────────────────────────────

class TestParseDeliveryScheduleJson:

    def test_empty_string_returns_empty(self):
        assert _parse_delivery_schedule_json("") == {}

    def test_none_returns_empty(self):
        assert _parse_delivery_schedule_json(None) == {}

    def test_whitespace_returns_empty(self):
        assert _parse_delivery_schedule_json("   ") == {}

    def test_invalid_json_returns_empty(self):
        assert _parse_delivery_schedule_json("{not valid}") == {}

    def test_valid_single_day(self):
        raw = '{"mon": ["08:00-12:00"]}'
        result = _parse_delivery_schedule_json(raw)
        assert result == {"mon": ["08:00-12:00"]}

    def test_valid_multiple_days(self):
        raw = '{"mon": ["08:00-12:00"], "fri": ["14:00-17:00"]}'
        result = _parse_delivery_schedule_json(raw)
        assert "mon" in result
        assert "fri" in result

    def test_invalid_day_key_excluded(self):
        raw = '{"monday": ["08:00-12:00"]}'
        result = _parse_delivery_schedule_json(raw)
        # "monday" is not a valid key (must be mon/tue/wed/thu/fri/sat/sun)
        assert "monday" not in result

    def test_valid_days_all_accepted(self):
        raw = '{"mon": ["08:00-12:00"], "tue": ["09:00-13:00"], "wed": ["10:00-14:00"], "thu": ["11:00-15:00"], "fri": ["12:00-16:00"], "sat": ["08:00-12:00"], "sun": ["10:00-14:00"]}'
        result = _parse_delivery_schedule_json(raw)
        assert len(result) == 7

    def test_invalid_slot_format_excluded(self):
        # _TIME_RANGE_RE only accepts valid HH:MM-HH:MM format
        raw = '{"mon": ["not-a-time", "08:00-12:00"]}'
        result = _parse_delivery_schedule_json(raw)
        # invalid format excluded; valid one kept
        assert result.get("mon") == ["08:00-12:00"]

    def test_empty_slots_excluded(self):
        raw = '{"mon": []}'
        result = _parse_delivery_schedule_json(raw)
        assert "mon" not in result

    def test_duplicate_slots_deduped(self):
        raw = '{"mon": ["08:00-12:00", "08:00-12:00"]}'
        result = _parse_delivery_schedule_json(raw)
        assert result["mon"].count("08:00-12:00") == 1

    def test_day_key_lowercased(self):
        # Day keys are normalized to lowercase
        raw = '{"MON": ["08:00-12:00"]}'
        result = _parse_delivery_schedule_json(raw)
        assert "mon" in result


# ─────────────────────────────────────────────────────────────────────────────
# _day_key_for_date
# ─────────────────────────────────────────────────────────────────────────────

class TestDayKeyForDate:

    def test_monday(self):
        # 2026-02-16 is a Monday
        dt = datetime(2026, 2, 16)
        assert _day_key_for_date(dt) == "mon"

    def test_tuesday(self):
        dt = datetime(2026, 2, 17)
        assert _day_key_for_date(dt) == "tue"

    def test_wednesday(self):
        dt = datetime(2026, 2, 18)
        assert _day_key_for_date(dt) == "wed"

    def test_thursday(self):
        dt = datetime(2026, 2, 19)
        assert _day_key_for_date(dt) == "thu"

    def test_friday(self):
        dt = datetime(2026, 2, 20)
        assert _day_key_for_date(dt) == "fri"

    def test_saturday(self):
        dt = datetime(2026, 2, 21)
        assert _day_key_for_date(dt) == "sat"

    def test_sunday(self):
        dt = datetime(2026, 2, 22)
        assert _day_key_for_date(dt) == "sun"

    def test_returns_string(self):
        dt = datetime(2026, 1, 1)
        assert isinstance(_day_key_for_date(dt), str)

    def test_result_is_lowercase(self):
        for i in range(7):
            # Monday Jan 5 2026 is a Monday → weekday 0
            dt = datetime(2026, 1, 5 + i)
            key = _day_key_for_date(dt)
            assert key == key.lower()


# ─────────────────────────────────────────────────────────────────────────────
# _badge (seguimiento version — different signature from receive_orders_page)
# ─────────────────────────────────────────────────────────────────────────────

class TestBadgeSeguimiento:

    def test_returns_span_tag(self):
        result = _badge("OK")
        assert result.startswith("<span")
        assert result.endswith("</span>")

    def test_text_in_output(self):
        result = _badge("Closed")
        assert "Closed" in result

    def test_default_kind_is_info(self):
        result = _badge("Label")
        assert "info" in result

    def test_custom_kind_ok(self):
        result = _badge("Good", kind="ok")
        assert "ok" in result

    def test_custom_kind_warn(self):
        result = _badge("Warning", kind="warn")
        assert "warn" in result

    def test_custom_kind_bad(self):
        result = _badge("Error", kind="bad")
        assert "bad" in result

    def test_empty_text(self):
        # Should still return a span (seguimiento badge doesn't guard empty text)
        result = _badge("")
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# _product_html
# ─────────────────────────────────────────────────────────────────────────────

class TestProductHtml:

    def test_name_only_no_desc(self):
        result = _product_html("Leche Entera")
        assert "Leche Entera" in result
        assert "product-name" in result
        # no desc div when desc is empty
        assert "product-desc" not in result

    def test_name_and_desc(self):
        result = _product_html("Leche Entera", "Fresca de vaca")
        assert "Leche Entera" in result
        assert "Fresca de vaca" in result
        assert "product-desc" in result

    def test_empty_name_defaults_to_product(self):
        result = _product_html("")
        assert "Product" in result

    def test_none_name_defaults_to_product(self):
        result = _product_html(None)
        assert "Product" in result

    def test_desc_whitespace_treated_as_empty(self):
        result = _product_html("Leche", "   ")
        # desc is stripped, becomes empty → no desc div
        assert "product-desc" not in result

    def test_returns_html_string(self):
        result = _product_html("Leche")
        assert "<div" in result
        assert "</div>" in result

    def test_none_desc_treated_as_empty(self):
        result = _product_html("Leche", None)
        assert "product-desc" not in result


# ─────────────────────────────────────────────────────────────────────────────
# _step_index
# ─────────────────────────────────────────────────────────────────────────────

class TestStepIndex:

    def test_order_sent_is_step_0(self):
        assert _step_index("ORDER_SENT") == 0

    def test_supplier_confirmed_full_is_step_1(self):
        assert _step_index("SUPPLIER_CONFIRMED_FULL") == 1

    def test_supplier_confirmed_partial_is_step_1(self):
        assert _step_index("SUPPLIER_CONFIRMED_PARTIAL") == 1

    def test_received_is_step_2(self):
        assert _step_index("RECEIVED") == 2

    def test_partially_received_is_step_2(self):
        assert _step_index("PARTIALLY_RECEIVED") == 2

    def test_not_received_is_step_2(self):
        assert _step_index("NOT_RECEIVED") == 2

    def test_matched_with_invoice_is_step_2(self):
        assert _step_index("MATCHED_WITH_INVOICE") == 2

    def test_waiting_supplier_action_is_step_3(self):
        assert _step_index("WAITING_SUPPLIER_ACTION") == 3

    def test_invoice_discrepancy_is_step_3(self):
        assert _step_index("INVOICE_DISCREPANCY") == 3

    def test_supplier_credit_note_issued_is_step_3(self):
        assert _step_index("SUPPLIER_CREDIT_NOTE_ISSUED") == 3

    def test_closed_is_step_4(self):
        assert _step_index("CLOSED") == 4

    def test_unknown_state_is_step_0(self):
        assert _step_index("SOME_UNKNOWN_STATE") == 0

    def test_empty_string_is_step_0(self):
        assert _step_index("") == 0

    def test_none_is_step_0(self):
        assert _step_index(None) == 0

    def test_case_insensitive(self):
        assert _step_index("closed") == 4
        assert _step_index("Closed") == 4


# ─────────────────────────────────────────────────────────────────────────────
# _derive_declaration_from_lines
# ─────────────────────────────────────────────────────────────────────────────

class TestDeriveDeclarationFromLines:

    def test_empty_payload_returns_partial(self):
        # No choices → choices list is empty → falls to "partial"
        result = _derive_declaration_from_lines({})
        assert result == "partial"

    def test_none_payload_returns_partial(self):
        result = _derive_declaration_from_lines(None)
        assert result == "partial"

    def test_all_full_returns_full(self):
        payload = {
            1: {"choice": "full"},
            2: {"choice": "full"},
            3: {"choice": "full"},
        }
        assert _derive_declaration_from_lines(payload) == "full"

    def test_all_none_returns_none(self):
        payload = {
            1: {"choice": "none"},
            2: {"choice": "none"},
        }
        assert _derive_declaration_from_lines(payload) == "none"

    def test_mixed_returns_partial(self):
        payload = {
            1: {"choice": "full"},
            2: {"choice": "none"},
        }
        assert _derive_declaration_from_lines(payload) == "partial"

    def test_one_full_returns_full(self):
        payload = {1: {"choice": "full"}}
        assert _derive_declaration_from_lines(payload) == "full"

    def test_missing_choice_key_defaults_to_full(self):
        # If "choice" key is missing, defaults to "full"
        payload = {1: {}, 2: {"choice": "full"}}
        assert _derive_declaration_from_lines(payload) == "full"

    def test_none_value_in_payload_defaults_to_full(self):
        payload = {1: None, 2: {"choice": "full"}}
        assert _derive_declaration_from_lines(payload) == "full"

    def test_partial_choice_with_full_returns_partial(self):
        payload = {
            1: {"choice": "full"},
            2: {"choice": "partial"},
        }
        assert _derive_declaration_from_lines(payload) == "partial"
