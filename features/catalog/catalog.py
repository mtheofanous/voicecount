from __future__ import annotations

"""features.catalog

Owner/Manager-only Catalog UI.

Features:
- Role-gated tab (only owner/manager)
- List products WITHOUT showing aliases
- Dedicated "Add product" form
- Editable table for existing products
- Delete checkbox column + delete action (hard-delete or archive)
- Filters (provider/category), search, and "show archived" toggle

Implementation note (archive mode):
This project currently has no migrations, and Product has no is_deleted column.
To support a safe "soft-delete" without schema changes, archive mode marks a
product as archived by prefixing its name with "[ARCHIVED] " and setting its
category to "__archived__" (reserved value), while keeping the row in the DB.
"""

from datetime import datetime
from typing import Any, Optional

import pandas as pd
import streamlit as st
from sqlmodel import select
from sqlalchemy.orm import load_only

from core.db import get_session
from domain.models import Product


ARCHIVE_CATEGORY = "__archived__"
ARCHIVE_PREFIX = "[ARCHIVED] "


def _s(x: Any) -> str:
    return ("" if x is None else str(x)).strip()


def _now() -> datetime:
    return datetime.utcnow()


def _is_archived(p: Product) -> bool:
    cat = _s(getattr(p, "category", ""))
    name = _s(getattr(p, "name", ""))
    return cat == ARCHIVE_CATEGORY or name.startswith(ARCHIVE_PREFIX)


@st.cache_data(ttl=30, show_spinner=False, hash_funcs={type(lambda: None): lambda _: "session_fn"})
def _list_products_cached(_get_session_fn, venue_id: int, refresh_token: int) -> list[Product]:
    _ = refresh_token
    with _get_session_fn() as s:
        products = list(
            s.exec(
                select(Product)
                .options(load_only(
                    Product.id, Product.venue_id, Product.name, Product.description,
                    Product.category, Product.unit, Product.quantity, Product.price, Product.iva,
                    Product.provider_name, Product.provider_email, Product.provider_phone, Product.provider_address
                ))
                .where(Product.venue_id == int(venue_id))
                .order_by(Product.name.asc(), Product.provider_name.asc())
            ).all()
        )
        # Force-load all attributes while session is active
        for p in products:
            _ = p.id, p.name, p.description, p.category, p.unit, p.quantity, p.price, p.iva
            _ = p.provider_name, p.provider_email, p.provider_phone, p.provider_address
        return products


def _bump_refresh(venue_id: int) -> None:
    k = f"catalog_refresh_token_{int(venue_id)}"
    st.session_state[k] = int(st.session_state.get(k, 0) or 0) + 1


def _refresh_token(venue_id: int) -> int:
    return int(st.session_state.get(f"catalog_refresh_token_{int(venue_id)}", 0) or 0)


def _product_rows(products: list[Product]) -> pd.DataFrame:
    rows = []
    for p in products:
        rows.append(
            {
                "delete": False,
                "id": getattr(p, "id", None),
                "name": _s(getattr(p, "name", "")),
                "description": _s(getattr(p, "description", "")),
                "category": _s(getattr(p, "category", "")),
                "unit": _s(getattr(p, "unit", "unidad")) or "unidad",
                "quantity": float(getattr(p, "quantity", 1.0) or 1.0),
                "price": float(getattr(p, "price", 0.0) or 0.0),
                "iva": float(getattr(p, "iva", 21.0) or 21.0),
                "provider_name": _s(getattr(p, "provider_name", "")),
                "provider_email": _s(getattr(p, "provider_email", "")),
                "provider_phone": _s(getattr(p, "provider_phone", "")),
                "provider_address": _s(getattr(p, "provider_address", "")),
                "archived": _is_archived(p),
                # NOTE: aliases intentionally not included in UI
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            [
                {
                    "delete": False,
                    "id": pd.NA,
                    "name": "",
                    "description": "",
                    "category": "",
                    "unit": "unidad",
                    "quantity": 1.0,
                    "price": 0.0,
                    "iva": 21.0,
                    "provider_name": "",
                    "provider_email": "",
                    "provider_phone": "",
                    "provider_address": "",
                    "archived": False,
                }
            ]
        )
    return df


def _upsert_products_from_df(*, venue_id: int, df: pd.DataFrame) -> tuple[int, int]:
    """Persist edits from the data editor.

    Returns: (created_count, updated_count)
    """
    created = 0
    updated = 0
    now = _now()

    df2 = df.copy()
    # ignore helper columns
    for col in ["delete", "archived"]:
        if col in df2.columns:
            df2 = df2.drop(columns=[col])

    if "id" not in df2.columns:
        df2["id"] = pd.NA

    # normalize numeric columns
    for col in ["quantity", "price", "iva"]:
        if col not in df2.columns:
            df2[col] = 0.0
        df2[col] = pd.to_numeric(df2[col], errors="coerce").fillna(0.0)

    with get_session() as s:
        for _, r in df2.iterrows():
            pid = r.get("id")
            name = _s(r.get("name"))
            if not name:
                # skip empty rows
                continue

            obj: Optional[Product] = None
            if pd.notna(pid):
                try:
                    obj = s.exec(
                        select(Product).where(
                            Product.id == int(pid),
                            Product.venue_id == int(venue_id),
                        )
                    ).first()
                except Exception:
                    obj = None

            if obj is None:
                obj = Product(venue_id=int(venue_id), name=name)
                created += 1
            else:
                updated += 1

            obj.venue_id = int(venue_id)
            obj.name = name
            obj.description = _s(r.get("description")) or None
            obj.category = _s(r.get("category")) or None
            obj.unit = (_s(r.get("unit")) or "unidad").lower()
            obj.quantity = float(r.get("quantity") or 0.0) or 0.0
            obj.price = float(r.get("price") or 0.0) or 0.0
            obj.iva = float(r.get("iva") or 0.0) or 0.0
            obj.provider_name = _s(r.get("provider_name")) or None
            obj.provider_email = _s(r.get("provider_email")) or None
            obj.provider_phone = _s(r.get("provider_phone")) or None
            obj.provider_address = _s(r.get("provider_address")) or None

            # IMPORTANT: we do not touch aliases here (this screen is "no aliases")

            # Back-compat fields
            try:
                if getattr(obj, "created_at", None) is None:
                    obj.created_at = now
            except Exception:
                pass

            s.add(obj)

        s.commit()

    return created, updated


def _create_product_from_form(*, venue_id: int, data: dict[str, Any]) -> int:
    name = _s(data.get("name"))
    if not name:
        raise ValueError("Name is required")

    p = Product(
        venue_id=int(venue_id),
        name=name,
        description=_s(data.get("description")) or None,
        category=_s(data.get("category")) or None,
        unit=(_s(data.get("unit")) or "unidad").lower(),
        quantity=float(data.get("quantity") or 1.0) or 1.0,
        price=float(data.get("price") or 0.0) or 0.0,
        iva=float(data.get("iva") or 21.0) or 21.0,
        provider_name=_s(data.get("provider_name")) or None,
        provider_email=_s(data.get("provider_email")) or None,
        provider_phone=_s(data.get("provider_phone")) or None,
        provider_address=_s(data.get("provider_address")) or None,
    )

    with get_session() as s:
        s.add(p)
        s.commit()
        s.refresh(p)
        return int(p.id) if p.id is not None else 0


def _archive_products(*, venue_id: int, product_ids: list[int]) -> int:
    now = _now()
    archived = 0
    with get_session() as s:
        for pid in product_ids:
            obj = s.exec(
                select(Product).where(Product.id == int(pid), Product.venue_id == int(venue_id))
            ).first()
            if not obj:
                continue

            # make idempotent
            if not _s(obj.name).startswith(ARCHIVE_PREFIX):
                obj.name = f"{ARCHIVE_PREFIX}{_s(obj.name)}"
            obj.category = ARCHIVE_CATEGORY
            if obj.description:
                if "[ARCHIVED" not in obj.description:
                    obj.description = f"{obj.description}\n[ARCHIVED {now.date().isoformat()}]"
            else:
                obj.description = f"[ARCHIVED {now.date().isoformat()}]"
            s.add(obj)
            archived += 1
        s.commit()
    return archived


def _hard_delete_products(*, venue_id: int, product_ids: list[int]) -> tuple[int, list[int]]:
    """Attempt hard delete; returns (deleted_count, failed_ids)."""
    deleted = 0
    failed: list[int] = []
    with get_session() as s:
        for pid in product_ids:
            obj = s.exec(
                select(Product).where(Product.id == int(pid), Product.venue_id == int(venue_id))
            ).first()
            if not obj:
                continue
            try:
                s.delete(obj)
                deleted += 1
            except Exception:
                failed.append(int(pid))
        try:
            s.commit()
        except Exception:
            s.rollback()
            return 0, product_ids
    return deleted, failed


def catalog_tab(*, venue_id: int, venue_role: str) -> None:
    venue_role = (_s(venue_role) or "member").lower()
    if venue_role not in {"owner", "manager"}:
        st.info("Only Owner/Manager can access the catalog.")
        return

    st.title("📦 Catálogo")
    st.caption("Admin view: see all products (aliases are hidden here).")

    with st.spinner("Loading catalog..."):
        products = _list_products_cached(get_session, int(venue_id), _refresh_token(int(venue_id)))
        df_all = _product_rows(products)

    # Build dropdown options from DB (excluding archived category)
    existing_categories = sorted(
        {
            _s(c)
            for c in df_all.get("category", pd.Series(dtype=str)).dropna().tolist()
            if _s(c) and _s(c) != ARCHIVE_CATEGORY
        }
    )
    existing_units = sorted({_s(u).lower() for u in df_all.get("unit", pd.Series(dtype=str)).dropna().tolist() if _s(u)})
    if "unidad" not in existing_units:
        existing_units = ["unidad"] + existing_units

    existing_providers = sorted(
        {_s(p) for p in df_all.get("provider_name", pd.Series(dtype=str)).dropna().tolist() if _s(p)}
    )

    # Provider details map (best-effort autofill)
    provider_details: dict[str, dict[str, str]] = {}
    for _, r in df_all.iterrows():
        pn = _s(r.get("provider_name"))
        if not pn:
            continue
        if pn not in provider_details:
            provider_details[pn] = {
                "provider_email": _s(r.get("provider_email")),
                "provider_phone": _s(r.get("provider_phone")),
                "provider_address": _s(r.get("provider_address")),
            }
        else:
            # fill blanks if later row has data
            for k in ["provider_email", "provider_phone", "provider_address"]:
                if not provider_details[pn].get(k) and _s(r.get(k)):
                    provider_details[pn][k] = _s(r.get(k))

    # -------------------------
    # Filters / Search
    # -------------------------
    st.subheader("Browse & filter")
    providers = sorted([p for p in df_all["provider_name"].dropna().unique().tolist() if _s(p)])
    categories = sorted([c for c in df_all["category"].dropna().unique().tolist() if _s(c) and _s(c) != ARCHIVE_CATEGORY])

    f1, f2, f3 = st.columns([1.2, 1.2, 1.6], vertical_alignment="center")
    with f1:
        provider_filter = st.multiselect("Provider", options=providers, default=[])
    with f2:
        category_filter = st.multiselect("Category", options=categories, default=[])
    with f3:
        search = st.text_input("Search (name/description)", value="", placeholder="e.g. tomate, leche, cava…")

    show_archived = st.toggle("Show archived", value=False)

    df = df_all.copy()
    if not show_archived:
        df = df[df["archived"] == False]  # noqa: E712

    if provider_filter:
        df = df[df["provider_name"].isin(provider_filter)]
    if category_filter:
        df = df[df["category"].isin(category_filter)]
    if _s(search):
        q = _s(search).lower()
        df = df[
            df["name"].astype(str).str.lower().str.contains(q, na=False)
            | df["description"].astype(str).str.lower().str.contains(q, na=False)
        ]

    st.divider()

    # -------------------------
    # Dedicated Add Product form (dropdown + "add new")
    # -------------------------
    st.subheader("Add product")

    add_new_label = "➕ Add new…"

    with st.form(key=f"catalog_add_form_{int(venue_id)}", clear_on_submit=True):
        c1, c2 = st.columns([1.35, 1.0])

        with c1:
            name = st.text_input("Name*", value="")
            description = st.text_area("Description", value="", height=80)

        with c2:
            # Category: dropdown of existing + add new
            cat_options = [""] + existing_categories + [add_new_label]
            category_pick = st.selectbox("Category", options=cat_options, index=0)
            if category_pick == add_new_label:
                category = st.text_input("New category", value="", placeholder="e.g. Verduras")
            else:
                category = category_pick

            # Unit (Unidad): dropdown of existing + add new
            unit_options = existing_units + [add_new_label]
            unit_pick = st.selectbox("Unidad (Unit)", options=unit_options, index=unit_options.index("unidad") if "unidad" in unit_options else 0)
            if unit_pick == add_new_label:
                unit = st.text_input("New unit", value="", placeholder="e.g. caja, kg, botella")
            else:
                unit = unit_pick

        c3, c4, c5 = st.columns(3)
        with c3:
            quantity = st.number_input("Default qty", min_value=0.0, value=1.0, step=1.0)
        with c4:
            price = st.number_input("Price", min_value=0.0, value=0.0, step=0.01)
        with c5:
            iva = st.number_input("IVA %", min_value=0.0, value=21.0, step=0.5)

        st.markdown("**Provider (optional)**")

        # Provider: dropdown of existing + add new
        provider_options = [""] + existing_providers + [add_new_label]
        provider_pick = st.selectbox("Provider", options=provider_options, index=0)

        if provider_pick == add_new_label:
            provider_name = st.text_input("New provider name", value="", placeholder="e.g. Distribuciones XX")
            default_email = ""
            default_phone = ""
            default_address = ""
        else:
            provider_name = provider_pick
            defaults = provider_details.get(provider_name or "", {})
            default_email = _s(defaults.get("provider_email"))
            default_phone = _s(defaults.get("provider_phone"))
            default_address = _s(defaults.get("provider_address"))

        p1, p2 = st.columns(2)
        with p1:
            provider_email = st.text_input("Provider email", value=default_email)
        with p2:
            provider_phone = st.text_input("Provider phone", value=default_phone)

        provider_address = st.text_input("Provider address", value=default_address)

        submitted = st.form_submit_button("➕ Add product", type="primary", use_container_width=True)
        if submitted:
            try:
                # If user selected add-new but left it blank, keep it empty
                category_final = _s(category)
                unit_final = _s(unit) or "unidad"
                provider_final = _s(provider_name)

                new_id = _create_product_from_form(
                    venue_id=int(venue_id),
                    data={
                        "name": name,
                        "description": description,
                        "category": category_final,
                        "unit": unit_final,
                        "quantity": quantity,
                        "price": price,
                        "iva": iva,
                        "provider_name": provider_final,
                        "provider_email": provider_email,
                        "provider_phone": provider_phone,
                        "provider_address": provider_address,
                    },
                )
                st.success(f"Created ✓ (ID: {new_id})")
                _bump_refresh(int(venue_id))
                st.rerun()
            except Exception as e:
                st.error(f"Could not create product: {e}")

    st.divider()

    # -------------------------
    # Editable table + actions
    # -------------------------
    st.subheader("Edit products")
    st.caption("Aliases are hidden. Use the Delete checkbox + the delete action below.")

    editor_key = f"catalog_editor_df_{int(venue_id)}"
    filter_sig = (
        tuple(sorted(provider_filter)),
        tuple(sorted(category_filter)),
        _s(search).lower(),
        bool(show_archived),
        len(df),
    )
    sig_key = f"catalog_editor_sig_{int(venue_id)}"
    if st.session_state.get(sig_key) != filter_sig:
        st.session_state[editor_key] = df.reset_index(drop=True)
        st.session_state[sig_key] = filter_sig

    with st.form(key=f"catalog_editor_form_{int(venue_id)}"):
        edited = st.data_editor(
            st.session_state.get(editor_key, df.reset_index(drop=True)),
            hide_index=True,
            num_rows="fixed",  # editing only; create via Add Product form
            use_container_width=True,
            column_config={
                "delete": st.column_config.CheckboxColumn("Delete", width="small"),
                "archived": st.column_config.CheckboxColumn("Archived", disabled=True, width="small"),
                "id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
                "name": st.column_config.TextColumn("Name", required=True, width="medium"),
                "description": st.column_config.TextColumn("Description", width="large"),
                "category": st.column_config.TextColumn("Category", width="medium"),
                "unit": st.column_config.TextColumn("Unit", width="small"),
                "quantity": st.column_config.NumberColumn("Default qty", min_value=0.0, step=1.0, width="small"),
                "price": st.column_config.NumberColumn("Price", min_value=0.0, step=0.01, width="small"),
                "iva": st.column_config.NumberColumn("IVA %", min_value=0.0, step=0.5, width="small"),
                "provider_name": st.column_config.TextColumn("Provider", width="medium"),
                "provider_email": st.column_config.TextColumn("Provider email", width="medium"),
                "provider_phone": st.column_config.TextColumn("Provider phone", width="medium"),
                "provider_address": st.column_config.TextColumn("Provider address", width="large"),
            },
        )

        a1, a2, a3 = st.columns([1.2, 1.2, 2.0], vertical_alignment="center")
        with a1:
            save = st.form_submit_button("💾 Save edits", type="primary", use_container_width=True)
        with a2:
            discard = st.form_submit_button("↩️ Discard", use_container_width=True)
        with a3:
            delete_mode = st.radio(
                "Delete mode",
                options=["Archive (recommended)", "Hard delete"],
                horizontal=True,
                label_visibility="collapsed",
                help="Archive keeps the product in the DB but hides it by default. Hard delete removes it (may fail if referenced by past orders).",
            )
            do_delete = st.form_submit_button("🗑️ Delete checked", use_container_width=True)

        if discard:
            st.session_state.pop(editor_key, None)
            st.session_state.pop(sig_key, None)
            st.rerun()

        if save:
            try:
                st.session_state[editor_key] = edited
                created, updated = _upsert_products_from_df(venue_id=int(venue_id), df=edited)
                st.success(f"Saved ✓ (created: {created}, updated: {updated})")
                st.session_state.pop(editor_key, None)
                st.session_state.pop(sig_key, None)
                _bump_refresh(int(venue_id))
                st.rerun()
            except Exception as e:
                st.error(f"Could not save edits: {e}")

        if do_delete:
            try:
                ids = []
                if "delete" in edited.columns and "id" in edited.columns:
                    for _, r in edited.iterrows():
                        if bool(r.get("delete")) and pd.notna(r.get("id")):
                            try:
                                ids.append(int(r.get("id")))
                            except Exception:
                                pass
                ids = sorted(set(ids))

                if not ids:
                    st.warning("No products checked for deletion.")
                else:
                    if delete_mode == "Hard delete":
                        deleted, failed = _hard_delete_products(venue_id=int(venue_id), product_ids=ids)
                        msg = f"Hard-deleted: {deleted}"
                        if failed:
                            archived = _archive_products(venue_id=int(venue_id), product_ids=failed)
                            msg += f" | Could not hard-delete {len(failed)} (likely referenced); archived instead: {archived}"
                        st.success(msg)
                    else:
                        archived = _archive_products(venue_id=int(venue_id), product_ids=ids)
                        st.success(f"Archived: {archived}")

                    st.session_state.pop(editor_key, None)
                    st.session_state.pop(sig_key, None)
                    _bump_refresh(int(venue_id))
                    st.rerun()
            except Exception as e:
                st.error(f"Delete action failed: {e}")
