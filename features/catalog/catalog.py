from __future__ import annotations

import hashlib
import io
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
from sqlmodel import select

from core.db import get_session
from domain.models import Product
from core.normalization import (
    normalize_text,
    normalize_unit,
    canonicalize_headers,
)


def to_float(val: Any, default: float) -> float:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        s = str(val).strip()
        if not s:
            return default
        s = s.replace(",", ".")
        return float(s)
    except Exception:
        return default


def price_with_iva(price: float, iva_pct: float) -> float:
    try:
        return round(float(price) * (1.0 + float(iva_pct) / 100.0), 2)
    except Exception:
        return round(float(price or 0.0), 2)


def normalize_product_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Canonicalize + normalize an input row into Product fields.
    Notes:
      - name/provider/category/etc are normalized for consistency
      - unit is normalized via normalization helpers
      - quantity uses quantity or default_qty
      - iva is a percentage (e.g., 21, 10, 4)
    """
    name = normalize_text(row.get("name"))
    description = normalize_text(row.get("description")) or None
    category = normalize_text(row.get("category")) or None

    provider_name = normalize_text(row.get("provider_name")) or None
    provider_email = normalize_text(row.get("provider_email")) or None
    provider_phone = normalize_text(row.get("provider_phone")) or None
    provider_address = normalize_text(row.get("provider_address")) or None
    aliases = normalize_text(row.get("aliases")) or None

    unit = normalize_unit(row.get("unit"))

    q = row.get("quantity")
    if q is None or (isinstance(q, float) and pd.isna(q)):
        q = row.get("default_qty")
    quantity = to_float(q, 1.0)

    price = to_float(row.get("price"), 0.0)

    # ✅ IVA (%)
    iva = to_float(row.get("iva"), 21.0)
    if iva < 0:
        iva = 0.0
    if iva > 100:
        iva = 100.0

    return {
        "name": name,
        "description": description,
        "category": category,
        "unit": unit,
        "quantity": quantity,
        "price": price,
        "iva": iva,
        "provider_name": provider_name,
        "provider_email": provider_email,
        "provider_phone": provider_phone,
        "provider_address": provider_address,
        "aliases": aliases,
    }


# ======================================================================================
# Utilities
# ======================================================================================

def file_signature(uploaded_file) -> str:
    """Stable signature for streamlit uploaded file to avoid double imports in same session."""
    data = uploaded_file.getvalue()
    h = hashlib.sha256(data).hexdigest()
    return f"{uploaded_file.name}:{len(data)}:{h}"


def dedupe_key(payload: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Dedupe key used to ignore duplicates:
      (normalized_provider_name, normalized_unit, normalized_name)
    Accent-normalization is already applied via normalize_text/normalize_unit.
    """
    prov = normalize_text(payload.get("provider_name") or "")
    unit = normalize_unit(payload.get("unit") or "unit")
    name = normalize_text(payload.get("name") or "")
    return (prov, unit, name)


# ======================================================================================
# MAIN TAB
# ======================================================================================

def catalog_tab(venue_id: int, role: str | None = None):
    st.subheader("📦 Catálogo de productos")

    NS = f"catalog_{venue_id}_"

    def K(name: str) -> str:
        return NS + name

    # Load products
    with get_session() as s:
        products: List[Product] = list(
            s.exec(select(Product).where(Product.venue_id == venue_id)).all()
        )

    # Tabs
    t_import, t_manual, t_edit, t_export = st.tabs(["📥 Import", "➕ Manual", "✏️ Edit", "⬇️ Export"])

    # =========================================================
    # 1) IMPORT (CSV/EXCEL) — duplicates are ignored automatically
    # =========================================================
    with t_import:
        st.markdown("### Importar archivo")
        st.caption(
            "Columnas (cualquier idioma): name, unit, quantity, price, iva, provider_name, provider_email, "
            "provider_phone, provider_address, category, aliases. "
            "Compat: default_qty → quantity. "
            "✅ Duplicados se ignoran automáticamente (incluye acentos/tonos)."
        )

        st.session_state.setdefault(K("last_import_product_ids"), [])
        st.session_state.setdefault(K("imported_file_sigs"), set())

        uploaded = st.file_uploader(
            "Sube CSV o Excel",
            type=["csv", "xlsx", "xls"],
            key=K("products_file_uploader"),
        )

        # Undo last import
        undo_col, _ = st.columns([1, 3])
        with undo_col:
            if st.session_state[K("last_import_product_ids")]:
                if st.button("↩️ Undo last import", type="secondary", key=K("undo_last_import_btn")):
                    ids_to_delete = list(st.session_state[K("last_import_product_ids")])
                    deleted = 0
                    with get_session() as s:
                        for pid in ids_to_delete:
                            obj = s.exec(select(Product).where(Product.id == pid)).first()
                            if obj:
                                s.delete(obj)
                                deleted += 1
                        s.commit()
                    st.session_state[K("last_import_product_ids")] = []
                    st.success(f"Undo completado ✅ Eliminados {deleted} productos.")
                    st.rerun()
            else:
                st.caption("No hay un import reciente para deshacer.")

        def _read_products_file(up) -> pd.DataFrame:
            filename = (up.name or "").lower()
            data = up.getvalue()
            bio = io.BytesIO(data)

            if filename.endswith(".csv"):
                try:
                    df_ = pd.read_csv(bio, encoding="utf-8")
                except UnicodeDecodeError:
                    bio.seek(0)
                    df_ = pd.read_csv(bio, encoding="cp1253")  # Greek Windows encoding
            else:
                df_ = pd.read_excel(bio)

            df_.columns = canonicalize_headers([str(c) for c in df_.columns])
            return df_

        if uploaded is None:
            st.info("Sube un archivo para empezar.")
        else:
            sig = file_signature(uploaded)

            try:
                df = _read_products_file(uploaded)
            except Exception as e:
                st.error(f"No pude leer el archivo: {e}")
                return

            st.dataframe(df.head(50).reset_index(drop=True), width="stretch")

            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                do_import = st.button("✅ Importar", type="primary", key=K("do_import_btn"))
            with c2:
                reset = st.button("↩️ Permitir re-importar", key=K("reset_reimport_btn"))
            with c3:
                st.caption("El import solo se ejecuta al hacer click. Evita doble-import del mismo archivo en esta sesión.")

            if reset:
                st.session_state[K("imported_file_sigs")].discard(sig)
                st.success("OK — este archivo se puede re-importar en esta sesión.")

            if do_import:
                if sig in st.session_state[K("imported_file_sigs")]:
                    st.warning("Este archivo ya se importó en esta sesión. Si quieres re-importarlo, pulsa reset.")
                    return

                if "name" not in df.columns:
                    st.error("El archivo debe tener una columna 'name' (o equivalente: Nombre / Όνομα).")
                    return

                # Ensure optional columns exist
                for c in [
                    "unit", "quantity", "price", "iva",
                    "provider_name", "provider_email", "provider_phone", "provider_address",
                    "category", "aliases", "default_qty",
                ]:
                    if c not in df.columns:
                        df[c] = pd.NA

                # Build existing dedupe keys from DB (for this venue)
                with get_session() as s:
                    existing_products: List[Product] = list(
                        s.exec(select(Product).where(Product.venue_id == venue_id)).all()
                    )

                existing_keys = set()
                for p in existing_products:
                    payload0 = {
                        "name": getattr(p, "name", "") or "",
                        "unit": getattr(p, "unit", "unit") or "unit",
                        "provider_name": getattr(p, "provider_name", "") or "",
                    }
                    existing_keys.add(dedupe_key(payload0))

                # Import rows: ignore duplicates (DB + within file)
                created = 0
                skipped_dup = 0
                skipped_empty = 0
                inserted_ids: List[int] = []

                # Track duplicates within the same file too
                seen_in_file = set(existing_keys)

                with get_session() as s:
                    for _, r in df.iterrows():
                        payload = normalize_product_row(r.to_dict())

                        if not payload["name"]:
                            skipped_empty += 1
                            continue

                        k = dedupe_key(payload)
                        if k in seen_in_file:
                            skipped_dup += 1
                            continue

                        # Backward compat: keep default_qty mirrored if your model still has it
                        if hasattr(Product, "default_qty"):
                            payload["default_qty"] = payload["quantity"]

                        # If DB/model hasn't been migrated yet, avoid crashing
                        if not hasattr(Product, "iva"):
                            payload.pop("iva", None)

                        p = Product(venue_id=venue_id, **payload)
                        s.add(p)
                        s.flush()  # to get p.id
                        inserted_ids.append(p.id)

                        seen_in_file.add(k)
                        created += 1

                    s.commit()

                st.session_state[K("imported_file_sigs")].add(sig)
                st.session_state[K("last_import_product_ids")] = inserted_ids

                st.success(f"Importado: {created} ✅")
                if skipped_dup:
                    st.info(f"Ignorados por duplicados: {skipped_dup} (incluye acentos/tonos normalizados)")
                if skipped_empty:
                    st.info(f"Filas ignoradas por nombre vacío: {skipped_empty}")

                with get_session() as s:
                    n = len(list(s.exec(select(Product).where(Product.venue_id == venue_id)).all()))
                    st.success(f"Productos en catálogo (venue {venue_id}): {n}")

                st.rerun()

    # =========================================================
    # 2) MANUAL ADD
    # =========================================================
    with t_manual:
        st.markdown("### Añadir productos manualmente")

        rows = st.session_state.get(K("manual_rows"))
        if rows is None:
            rows = [{
                "name": "",
                "description": "",
                "unit": "unidad",
                "quantity": 1.0,
                "price": 0.0,
                "iva": 21.0,
                "provider_name": "",
                "category": "",
            }]
            st.session_state[K("manual_rows")] = rows

        edited = st.data_editor(
            rows,
            num_rows="dynamic",
            key=K("manual_editor"),
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("name", required=True),
                "description": st.column_config.TextColumn("description"),
                "unit": st.column_config.TextColumn("unit"),
                "quantity": st.column_config.NumberColumn("quantity", min_value=0.0),
                "price": st.column_config.NumberColumn("price", min_value=0.0),
                "iva": st.column_config.NumberColumn("IVA (%)", min_value=0.0, max_value=100.0, step=1.0),
                "provider_name": st.column_config.TextColumn("provider_name"),
                "category": st.column_config.TextColumn("category"),
            },
        )
        st.session_state[K("manual_rows")] = edited

        if st.button("💾 Guardar productos", type="primary", key=K("manual_save")):
            created = 0
            skipped_dup = 0

            with get_session() as s:
                # Build existing keys
                existing_products: List[Product] = list(
                    s.exec(select(Product).where(Product.venue_id == venue_id)).all()
                )
                existing_keys = set()
                for p in existing_products:
                    existing_keys.add(
                        dedupe_key({
                            "name": getattr(p, "name", "") or "",
                            "unit": getattr(p, "unit", "unit") or "unit",
                            "provider_name": getattr(p, "provider_name", "") or "",
                        })
                    )

                for r in edited:
                    payload = normalize_product_row(r)
                    if not payload["name"]:
                        continue

                    k = dedupe_key(payload)
                    if k in existing_keys:
                        skipped_dup += 1
                        continue

                    if hasattr(Product, "default_qty"):
                        payload["default_qty"] = payload["quantity"]

                    if not hasattr(Product, "iva"):
                        payload.pop("iva", None)

                    p = Product(venue_id=venue_id, **payload)
                    s.add(p)
                    existing_keys.add(k)
                    created += 1

                s.commit()

            st.success(f"Guardados {created} productos ✅")
            if skipped_dup:
                st.info(f"Ignorados por duplicados: {skipped_dup}")
            st.rerun()

    # =========================================================
    # 3) EDIT
    # =========================================================
    with t_edit:
        st.markdown("### Editar catálogo")

        # Reload products fresh for edit tab
        with get_session() as s:
            products = list(s.exec(select(Product).where(Product.venue_id == venue_id)).all())

        if not products:
            st.info("No hay productos aún.")
        else:
            data = []
            for p in products:
                p_price = float(getattr(p, "price", 0.0) or 0.0)
                p_iva = float(getattr(p, "iva", 21.0) or 21.0) if hasattr(p, "iva") else 21.0

                data.append({
                    "id": p.id,
                    "name": getattr(p, "name", ""),
                    "description": getattr(p, "description", "") or "",
                    "category": getattr(p, "category", "") or "",
                    "unit": getattr(p, "unit", "unit"),
                    "quantity": float(getattr(p, "quantity", 1.0) or 1.0),
                    "price": p_price,
                    "iva": p_iva,
                    "price_with_iva": price_with_iva(p_price, p_iva),
                    "provider_name": getattr(p, "provider_name", "") or "",
                    "provider_email": getattr(p, "provider_email", "") or "",
                    "provider_phone": getattr(p, "provider_phone", "") or "",
                    "provider_address": getattr(p, "provider_address", "") or "",
                    "aliases": getattr(p, "aliases", "") or "",
                })

            df_edit = pd.DataFrame(data)

            disabled_cols = ["id", "price_with_iva"]
            if not hasattr(Product, "iva"):
                # if model not migrated yet, make it read-only too
                disabled_cols.append("iva")

            edited_df = st.data_editor(
                df_edit,
                key=K("edit_editor"),
                use_container_width=True,
                disabled=disabled_cols,
                column_config={
                    "quantity": st.column_config.NumberColumn("quantity", min_value=0.0),
                    "price": st.column_config.NumberColumn("price", min_value=0.0),
                    "iva": st.column_config.NumberColumn("IVA (%)", min_value=0.0, max_value=100.0, step=1.0),
                    "price_with_iva": st.column_config.NumberColumn("price_with_iva", disabled=True),
                },
            )

            if st.button("💾 Guardar cambios", type="primary", key=K("save_edits")):
                with get_session() as s:
                    for _, row in edited_df.iterrows():
                        pid = int(row["id"])
                        obj = s.exec(select(Product).where(Product.id == pid)).first()
                        if not obj:
                            continue

                        payload = normalize_product_row(row.to_dict())
                        if not payload["name"]:
                            continue

                        # Do not attempt to set iva if model doesn't have it yet
                        if not hasattr(obj, "iva"):
                            payload.pop("iva", None)

                        for k, v in payload.items():
                            setattr(obj, k, v)

                        if hasattr(obj, "default_qty"):
                            obj.default_qty = obj.quantity

                    s.commit()
                st.success("Cambios guardados ✅")
                st.rerun()

    # =========================================================
    # 4) EXPORT
    # =========================================================
    with t_export:
        st.markdown("### Exportar catálogo")

        # Reload products fresh for export tab
        with get_session() as s:
            products = list(s.exec(select(Product).where(Product.venue_id == venue_id)).all())

        if not products:
            st.info("No hay productos para exportar.")
        else:
            export_rows = []
            for p in products:
                export_rows.append({
                    "id": p.id,
                    "category": getattr(p, "category", None),
                    "quantity": getattr(p, "quantity", None),
                    "price": getattr(p, "price", None),
                    "iva": getattr(p, "iva", 21.0) if hasattr(p, "iva") else None,
                    "provider_name": getattr(p, "provider_name", None),
                    "provider_email": getattr(p, "provider_email", None),
                    "provider_phone": getattr(p, "provider_phone", None),
                    "provider_address": getattr(p, "provider_address", None),
                    "unit": getattr(p, "unit", None),
                    "name": getattr(p, "name", None),
                    "description": getattr(p, "description", None),
                    "created_at": getattr(p, "created_at", None),
                    "aliases": getattr(p, "aliases", None),
                })

            exp = pd.DataFrame(export_rows)
            csv_bytes = exp.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                "⬇️ Descargar CSV",
                data=csv_bytes,
                file_name="catalog_export.csv",
                mime="text/csv",
                key=K("download_csv"),
            )
