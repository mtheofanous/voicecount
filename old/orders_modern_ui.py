
"""
orders_modern_ui.py
-------------------
Drop-in module to modernize the Orders page:
- Order status workflow: draft -> ready_to_send -> final
- Audit fields: created_by / updated_by + timestamps
- Live presence: see who is currently editing an order (heartbeat)

Assumptions:
- You already have: engine, get_session(), Product model, distinct_units(), safe_str(), etc.
- You are using SQLModel + SQLite (but works on other DBs too; migrations below are SQLite-safe).
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

import pandas as pd
import streamlit as st
from sqlmodel import SQLModel, Field, select
from sqlalchemy import text

# -----------------------------
# Models (extend_existing=True)
# -----------------------------

class Order(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    venue_id: int = Field(index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    # Audit
    created_by: Optional[str] = Field(default=None, index=True)
    updated_at: Optional[datetime] = Field(default=None, index=True)
    updated_by: Optional[str] = Field(default=None, index=True)

    # Workflow
    status: str = Field(default="draft", index=True)  # draft | ready_to_send | final

    title: Optional[str] = None
    note: Optional[str] = None


class OrderLine(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    venue_id: int = Field(index=True)

    order_id: int = Field(foreign_key="order.id", index=True)
    product_id: Optional[int] = Field(default=None, foreign_key="product.id", index=True)

    spoken_name: str = ""
    quantity: float = 1.0
    unit: Optional[str] = Field(default="unidad")
    confidence: Optional[float] = Field(default=0.0)

    matched_name: Optional[str] = None
    provider: Optional[str] = None

    # Line audit (optional but useful)
    updated_at: Optional[datetime] = Field(default=None, index=True)
    updated_by: Optional[str] = Field(default=None, index=True)


class OrderPresence(SQLModel, table=True):
    """
    Live presence for collaborative editing.
    Each browser session keeps a heartbeat row.
    """
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    order_id: int = Field(index=True)
    venue_id: int = Field(index=True)

    actor: str = Field(index=True)
    session_id: str = Field(index=True)

    last_seen_at: datetime = Field(default_factory=datetime.utcnow, index=True)


# -----------------------------
# Helpers (user + session)
# -----------------------------

def current_actor() -> str:
    """Best-effort: pull current user identity from Streamlit session_state."""
    u = st.session_state.get("user")
    if isinstance(u, dict):
        for k in ("email", "username", "name"):
            if u.get(k):
                return str(u[k])

    for k in ("user_email", "email", "username", "auth_email"):
        if st.session_state.get(k):
            return str(st.session_state[k])

    return "unknown"


def current_session_id() -> str:
    """Stable browser-session id for presence. Stored in session_state."""
    if "presence_session_id" not in st.session_state:
        st.session_state["presence_session_id"] = str(uuid.uuid4())
    return str(st.session_state["presence_session_id"])


# -----------------------------
# SQLite-safe migrations
# -----------------------------

def ensure_orders_migrations(engine) -> None:
    """
    Create tables + add missing columns for existing SQLite installs.
    - Safe to call on every app start.
    """
    SQLModel.metadata.create_all(engine)

    def _table_cols(conn, table: str) -> set:
        rows = conn.execute(text(f"PRAGMA table_info('{table}')")).fetchall()
        return {r[1] for r in rows}

    with engine.begin() as conn:
        # --- order ---
        if "order" in _sqlite_tables(conn):
            existing = _table_cols(conn, "order")
            for col, ddl in [
                ("venue_id",    'ALTER TABLE "order" ADD COLUMN venue_id INTEGER'),
                ("created_by",  'ALTER TABLE "order" ADD COLUMN created_by TEXT'),
                ("updated_at",  'ALTER TABLE "order" ADD COLUMN updated_at TEXT'),
                ("updated_by",  'ALTER TABLE "order" ADD COLUMN updated_by TEXT'),
                ("status",      'ALTER TABLE "order" ADD COLUMN status TEXT'),
                ("title",       'ALTER TABLE "order" ADD COLUMN title TEXT'),
                ("note",        'ALTER TABLE "order" ADD COLUMN note TEXT'),
                ("created_at",  'ALTER TABLE "order" ADD COLUMN created_at TEXT'),
            ]:
                if col not in existing:
                    conn.execute(text(ddl))

        # --- orderline ---
        if "orderline" in _sqlite_tables(conn):
            existing = _table_cols(conn, "orderline")
            for col, ddl in [
                ("venue_id",    "ALTER TABLE orderline ADD COLUMN venue_id INTEGER"),
                ("order_id",    "ALTER TABLE orderline ADD COLUMN order_id INTEGER"),
                ("product_id",  "ALTER TABLE orderline ADD COLUMN product_id INTEGER"),
                ("spoken_name", "ALTER TABLE orderline ADD COLUMN spoken_name TEXT"),
                ("matched_name","ALTER TABLE orderline ADD COLUMN matched_name TEXT"),
                ("provider",    "ALTER TABLE orderline ADD COLUMN provider TEXT"),
                ("quantity",    "ALTER TABLE orderline ADD COLUMN quantity REAL"),
                ("unit",        "ALTER TABLE orderline ADD COLUMN unit TEXT"),
                ("confidence",  "ALTER TABLE orderline ADD COLUMN confidence REAL"),
                ("updated_at",  "ALTER TABLE orderline ADD COLUMN updated_at TEXT"),
                ("updated_by",  "ALTER TABLE orderline ADD COLUMN updated_by TEXT"),
            ]:
                if col not in existing:
                    conn.execute(text(ddl))

        # --- orderpresence ---
        if "orderpresence" not in _sqlite_tables(conn):
            # SQLModel.metadata.create_all already handled creation,
            # but on older deployments we keep this check for safety.
            SQLModel.metadata.create_all(engine)


def _sqlite_tables(conn) -> set:
    rows = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
    return {r[0] for r in rows}


# -----------------------------
# Presence: heartbeat + listing
# -----------------------------

def touch_presence(session, venue_id: int, order_id: int, actor: str, session_id: str, ttl_seconds: int = 45) -> None:
    """
    Upsert presence row and cleanup stale rows.
    Call this on every render of the order page (or every edit action).
    """
    now = datetime.utcnow()

    # Cleanup stale
    cutoff = now - timedelta(seconds=ttl_seconds)
    stale = session.exec(
        select(OrderPresence).where(
            OrderPresence.venue_id == venue_id,
            OrderPresence.order_id == order_id,
            OrderPresence.last_seen_at < cutoff,
        )
    ).all()
    for r in stale:
        session.delete(r)

    # Upsert (by session_id)
    row = session.exec(
        select(OrderPresence).where(
            OrderPresence.venue_id == venue_id,
            OrderPresence.order_id == order_id,
            OrderPresence.session_id == session_id,
        )
    ).first()

    if row:
        row.actor = actor
        row.last_seen_at = now
        session.add(row)
    else:
        session.add(OrderPresence(
            venue_id=venue_id,
            order_id=order_id,
            actor=actor,
            session_id=session_id,
            last_seen_at=now,
        ))

    session.commit()


def get_active_editors(session, venue_id: int, order_id: int, ttl_seconds: int = 45) -> List[Tuple[str, datetime]]:
    now = datetime.utcnow()
    cutoff = now - timedelta(seconds=ttl_seconds)
    rows = session.exec(
        select(OrderPresence).where(
            OrderPresence.venue_id == venue_id,
            OrderPresence.order_id == order_id,
            OrderPresence.last_seen_at >= cutoff,
        ).order_by(OrderPresence.last_seen_at.desc())
    ).all()
    return [(r.actor, r.last_seen_at) for r in rows]


# -----------------------------
# UI helpers
# -----------------------------

def inject_modern_orders_css() -> None:
    st.markdown("""
    <style>
    .block-container { max-width: 1050px; padding-top: 1.1rem; padding-bottom: 4rem; }
    .order-shell { border: 1px solid rgba(0,0,0,0.07); border-radius: 18px; padding: 14px 14px; background: rgba(0,0,0,0.02); }
    .order-top { display:flex; align-items:center; justify-content:space-between; gap: 12px; }
    .order-meta { color: rgba(0,0,0,0.65); font-size: 0.88rem; margin-top: 2px; }
    .badge { display:inline-flex; align-items:center; gap:8px; padding: 6px 10px; border-radius: 999px; font-size: 0.85rem; border: 1px solid rgba(0,0,0,0.10); background: white; }
    .badge-dot { width:8px; height:8px; border-radius:999px; background: rgba(0,0,0,0.25); }
    .badge-draft .badge-dot { background: #f59e0b; }
    .badge-ready .badge-dot { background: #10b981; }
    .badge-final .badge-dot { background: #64748b; }
    .pill { display:inline-flex; align-items:center; padding: 4px 8px; border-radius: 999px; font-size: 0.82rem; border: 1px solid rgba(0,0,0,0.08); background: white; margin-right: 6px; }
    .muted { color: rgba(0,0,0,0.55); }
    </style>
    """, unsafe_allow_html=True)


def _badge_class(status: str) -> str:
    s = (status or "draft").strip().lower()
    if s == "ready_to_send":
        return "badge badge-ready"
    if s == "final":
        return "badge badge-final"
    return "badge badge-draft"


def _status_label(status: str) -> str:
    s = (status or "draft").strip().lower()
    if s == "ready_to_send":
        return "Listo para enviar"
    if s == "final":
        return "Final"
    return "Borrador"


# -----------------------------
# Main: modern Orders tab
# -----------------------------

def orders_tab_modern(
    *,
    venue_id: int,
    role: Optional[str] = None,
    engine=None,
    get_session=None,
    Product=None,
    distinct_units=None,
    safe_str=None,
    num_or_default=None,
    coalesce_unit=None,
) -> None:
    """
    Modern replacement for your orders_tab().

    You MUST pass your project objects:
      - engine
      - get_session() contextmanager
      - Product model
      - distinct_units(products) -> list[str]
      - safe_str(x) -> str
      - num_or_default(x, default)
      - coalesce_unit(unit_from_line, prod, default)
    """
    assert engine is not None, "Pass engine="
    assert get_session is not None, "Pass get_session="
    assert Product is not None, "Pass Product="
    assert distinct_units is not None, "Pass distinct_units="
    assert safe_str is not None, "Pass safe_str="
    assert num_or_default is not None, "Pass num_or_default="
    assert coalesce_unit is not None, "Pass coalesce_unit="

    inject_modern_orders_css()
    st.subheader("📜 Pedidos")

    actor = current_actor()
    sid = current_session_id()

    # --- Load orders ---
    with get_session() as s:
        orders_all = s.exec(
            select(Order).where(Order.venue_id == venue_id).order_by(Order.created_at.desc())
        ).all()

    if not orders_all:
        st.info("No hay pedidos aún. Crea uno en 'Nuevo pedido'.")
        return

    # --- Filter + search ---
    f1, f2 = st.columns([2, 3], vertical_alignment="center")
    with f1:
        status_filter = st.radio(
            "Filtrar",
            options=["all", "draft", "ready_to_send", "final"],
            format_func=lambda x: {"all": "Todos", "draft": "Borrador", "ready_to_send": "Listo", "final": "Final"}[x],
            horizontal=True,
            key="orders_filter_status_modern",
        )
    with f2:
        q = st.text_input("Buscar (título o #id)", value="", key="orders_search_modern")

    def match(o: Order) -> bool:
        if status_filter != "all" and (o.status or "draft") != status_filter:
            return False
        if q.strip():
            qs = q.strip().lower()
            return (qs in str(o.id).lower()) or (qs in (o.title or "").lower())
        return True

    orders = [o for o in orders_all if match(o)]
    if not orders:
        st.warning("No hay pedidos con ese filtro/búsqueda.")
        return

    labels = [f"#{o.id} — {(o.title or o.created_at.strftime('%Y-%m-%d %H:%M'))}" for o in orders]
    idx = st.selectbox("Selecciona un pedido", range(len(orders)), format_func=lambda i: labels[i], key="orders_picker_modern")
    order = orders[idx]
    st.session_state["active_order_id"] = order.id

    # force editor reload on change
    if st.session_state.get("last_orders_tab_order_id_modern") != order.id:
        st.session_state["last_orders_tab_order_id_modern"] = order.id
        st.session_state.pop(f"order_editor_df_{order.id}", None)
        st.session_state.pop(f"order_editor_widget_{order.id}", None)

    # --- Presence heartbeat ---
    with get_session() as s:
        touch_presence(s, venue_id=venue_id, order_id=order.id, actor=actor, session_id=sid, ttl_seconds=45)
        active = get_active_editors(s, venue_id=venue_id, order_id=order.id, ttl_seconds=45)

    # --- Header card (status + audit + presence) ---
    st.markdown('<div class="order-shell">', unsafe_allow_html=True)

    left, right = st.columns([3, 2], vertical_alignment="center")
    with left:
        updated_at_txt = order.updated_at.strftime("%Y-%m-%d %H:%M") if order.updated_at else "—"
        updated_by_txt = order.updated_by or "—"
        created_by_txt = order.created_by or "—"

        st.markdown(
            f"""
            <div class="order-top">
              <div>
                <div style="font-size:1.05rem; font-weight:650;">Pedido #{order.id}</div>
                <div class="order-meta">
                  Creado: {order.created_at.strftime('%Y-%m-%d %H:%M')} · Por: {created_by_txt}<br/>
                  Última edición: {updated_at_txt} · Por: {updated_by_txt}
                </div>
              </div>
              <div class="{_badge_class(order.status)}">
                <span class="badge-dot"></span>
                {_status_label(order.status)}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Presence pills
        if active:
            pills = []
            seen_now = datetime.utcnow()
            for a, ts in active[:6]:
                age = int((seen_now - ts).total_seconds())
                pills.append(f'<span class="pill">{a} <span class="muted">· {age}s</span></span>')
            st.markdown("👀 Editando ahora: " + " ".join(pills), unsafe_allow_html=True)
        else:
            st.caption("👀 Nadie está editando ahora mismo.")

    with right:
        pretty = {"draft": "Borrador", "ready_to_send": "Listo para enviar", "final": "Final"}
        status_options = ["draft", "ready_to_send", "final"]
        current_idx = status_options.index(order.status) if order.status in status_options else 0

        new_status = st.selectbox(
            "Estado",
            options=status_options,
            index=current_idx,
            format_func=lambda x: pretty.get(x, x),
            key=f"order_status_{order.id}_modern",
        )

        quick = st.button(
            "✅ Marcar como listo para enviar",
            type="primary",
            disabled=(order.status == "ready_to_send"),
            key=f"btn_ready_{order.id}_modern",
            use_container_width=True,
        )
        if quick:
            new_status = "ready_to_send"
            st.session_state[f"order_status_{order.id}_modern"] = "ready_to_send"

        if new_status != order.status:
            with get_session() as s:
                o = s.exec(select(Order).where(Order.id == order.id)).first()
                if o:
                    o.status = new_status
                    o.updated_at = datetime.utcnow()
                    o.updated_by = actor
                    if not o.created_by:
                        o.created_by = actor
                    s.add(o)
                    s.commit()
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # --- Title ---
    cA, cB = st.columns([3, 1], vertical_alignment="center")
    with cA:
        order_title = st.text_input("Título", value=order.title or "", key=f"order_title_{order.id}_modern")
    with cB:
        note = st.text_input("Nota", value=order.note or "", key=f"order_note_{order.id}_modern")

    if (order.title or "") != order_title or (order.note or "") != note:
        with get_session() as s:
            o = s.exec(select(Order).where(Order.id == order.id)).first()
            if o:
                o.title = order_title or None
                o.note = note or None
                o.updated_at = datetime.utcnow()
                o.updated_by = actor
                if not o.created_by:
                    o.created_by = actor
                s.add(o)
                s.commit()
        st.rerun()

    # ---------- Load products ----------
    with get_session() as s:
        products = s.exec(
            select(Product).where(Product.venue_id == venue_id).order_by(Product.name.asc(), Product.provider_name.asc())
        ).all()

    def _product_label(p) -> str:
        prov = getattr(p, "provider_name", None) or "(Sin proveedor)"
        return f"{p.name} — {prov} (#{p.id})"

    placeholder_product = "— Selecciona producto —"
    product_labels = [placeholder_product] + [_product_label(p) for p in products]
    label_to_product = {lbl: p for lbl, p in zip(product_labels[1:], products)}
    id_to_product = {getattr(p, "id", None): p for p in products}

    unit_opts = distinct_units(products)
    unit_placeholder = "—"
    unit_options = [unit_placeholder] + sorted({u.strip().lower() for u in (unit_opts or []) if str(u).strip()})
    for u in ["unidad", "kg", "g", "l", "ml", "caja", "paquete"]:
        if u not in unit_options:
            unit_options.append(u)

    def _apply_product_selection(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        for i, row in df.iterrows():
            lbl = safe_str(row.get("product_label")).strip()
            pid = row.get("product_id")
            prod = None
            if lbl and lbl != placeholder_product and lbl in label_to_product:
                prod = label_to_product[lbl]
            elif pd.notna(pid) and pid in id_to_product:
                prod = id_to_product.get(pid)

            if prod:
                df.at[i, "product_id"] = getattr(prod, "id", None)
                df.at[i, "matched_name"] = getattr(prod, "name", "") or ""
                df.at[i, "provider"] = getattr(prod, "provider_name", "") or ""
                df.at[i, "unit"] = safe_str(getattr(prod, "unit", "")).strip().lower() or "unidad"
                if lbl == "" or lbl == placeholder_product:
                    df.at[i, "product_label"] = _product_label(prod)
            else:
                if lbl == placeholder_product or not lbl:
                    df.at[i, "product_id"] = pd.NA
                    df.at[i, "matched_name"] = ""
                    df.at[i, "provider"] = ""
        return df

    # ---------- Load lines from DB ----------
    with get_session() as s:
        lines = s.exec(select(OrderLine).where(OrderLine.order_id == order.id)).all()

    rows = []
    for ln in lines:
        pid = getattr(ln, "product_id", None)
        prod = id_to_product.get(pid) if pid is not None else None
        if prod:
            lbl = _product_label(prod)
            matched_name = getattr(prod, "name", "") or (ln.matched_name or "")
            provider = getattr(prod, "provider_name", "") or ""
            unit_val = safe_str(getattr(prod, "unit", "")).strip().lower() or "unidad"
            qty_val = getattr(ln, "quantity", None)
        else:
            lbl = placeholder_product
            matched_name = (ln.matched_name or "")
            provider = ""
            unit_val = safe_str(getattr(ln, "unit", "")).strip().lower() or "unidad"
            qty_val = getattr(ln, "quantity", None)

        rows.append({
            "line_id": getattr(ln, "id", pd.NA),
            "product_label": lbl,
            "product_id": pid if pid is not None else pd.NA,
            "matched_name": matched_name,
            "provider": provider,
            "quantity": qty_val,
            "unit": unit_val,
            "eliminar": False,
        })

    visible_cols = ["product_label", "quantity", "unit", "eliminar"]
    full_cols = ["line_id", "product_label", "product_id", "matched_name", "provider", "quantity", "unit", "eliminar"]

    def coerce_df(obj, columns):
        if isinstance(obj, pd.DataFrame):
            df = obj.copy()
        elif isinstance(obj, (list, tuple)):
            df = pd.DataFrame(obj) if obj else pd.DataFrame(columns=columns)
        elif isinstance(obj, dict):
            df = pd.DataFrame([obj])
        else:
            df = pd.DataFrame(columns=columns)

        for c in columns:
            if c not in df.columns:
                df[c] = pd.NA
        if "eliminar" in df.columns:
            df["eliminar"] = df["eliminar"].fillna(False)
        return df[columns]

    def norm_types(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        for col in ["product_label", "matched_name", "provider", "unit"]:
            if col in df.columns:
                df[col] = df[col].astype("string")
        for col in ["quantity", "line_id", "product_id"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "eliminar" in df.columns:
            df["eliminar"] = df["eliminar"].fillna(False).astype(bool)
        return df

    state_key_df = f"order_editor_df_{order.id}"
    state_key_widget = f"order_editor_widget_{order.id}"
    baseline_key = f"{state_key_df}__baseline"

    if state_key_df not in st.session_state:
        df0 = coerce_df(rows, full_cols)
        df0 = norm_types(df0)
        df0 = _apply_product_selection(df0)
        # drop placeholders at init
        df0 = df0[df0["product_label"] != placeholder_product].reset_index(drop=True)
        st.session_state[state_key_df] = df0
        st.session_state[baseline_key] = df0.copy(deep=True)

    # ---------- Add line (popover) ----------
    df_full_for_add = norm_types(coerce_df(st.session_state.get(state_key_df), full_cols))
    used_ids = set(pd.to_numeric(df_full_for_add["product_id"], errors="coerce").dropna().astype(int).tolist())
    remaining_products = [p for p in products if getattr(p, "id", None) not in used_ids]
    remaining_labels = [_product_label(p) for p in remaining_products]

    with st.popover("➕ Añadir línea"):
        if not remaining_products:
            st.info("Ya has añadido todos los productos del catálogo a este pedido.")
        else:
            new_lbl = st.selectbox("Producto", options=remaining_labels, key=f"add_prod_{order.id}_modern")
            new_qty = st.number_input("Cantidad", min_value=0.0, value=1.0, step=1.0, key=f"add_qty_{order.id}_modern")
            add_confirm = st.button("✅ Añadir", type="primary", key=f"add_confirm_{order.id}_modern")
            if add_confirm:
                p = label_to_product.get(new_lbl)
                if p is not None:
                    df_tmp = norm_types(coerce_df(st.session_state.get(state_key_df), full_cols))
                    df_tmp.loc[len(df_tmp)] = {
                        "line_id": pd.NA,
                        "product_label": _product_label(p),
                        "product_id": int(getattr(p, "id")),
                        "matched_name": getattr(p, "name", "") or "",
                        "provider": getattr(p, "provider_name", "") or "",
                        "quantity": float(new_qty),
                        "unit": safe_str(getattr(p, "unit", "")).strip().lower() or "unidad",
                        "eliminar": False,
                    }
                    df_tmp = norm_types(df_tmp)
                    df_tmp = _apply_product_selection(df_tmp).reset_index(drop=True)
                    st.session_state[state_key_df] = df_tmp
                    st.session_state[baseline_key] = df_tmp.copy(deep=True)
                    st.session_state.pop(state_key_widget, None)
                    # stamp order audit immediately (so "last modified" updates even before saving lines)
                    with get_session() as s:
                        o = s.exec(select(Order).where(Order.id == order.id)).first()
                        if o:
                            o.updated_at = datetime.utcnow()
                            o.updated_by = actor
                            if not o.created_by:
                                o.created_by = actor
                            s.add(o)
                            s.commit()
                    st.rerun()

    st.subheader("🧾 Líneas del pedido")

    colcfg = {
        "product_label": st.column_config.Column("Producto", disabled=True, width="large",
                                                 help="Producto fijado. Para cambiar: elimina y añade de nuevo."),
        "quantity": st.column_config.NumberColumn("Cantidad", min_value=0.0, step=1.0, format="%.2f"),
        "unit": st.column_config.Column("Unidad", disabled=True, width="small",
                                        help="Unidad fija del catálogo (por product_id)."),
        "eliminar": st.column_config.CheckboxColumn("Eliminar"),
    }

    df_full = norm_types(coerce_df(st.session_state.get(state_key_df), full_cols)).reset_index(drop=True)
    df_view = df_full[visible_cols].copy()

    edited_view = st.data_editor(
        df_view.reset_index(drop=True),
        width="stretch",
        num_rows="fixed",
        column_config=colcfg,
        hide_index=True,
        key=state_key_widget,
    )

    # Sync visible -> full
    df_vis = norm_types(coerce_df(edited_view, visible_cols)).reset_index(drop=True)
    n = min(len(df_full), len(df_vis))
    df_full = df_full.iloc[:n].reset_index(drop=True)
    df_vis = df_vis.iloc[:n].reset_index(drop=True)
    for c in visible_cols:
        df_full[c] = df_vis[c].values

    df_full = norm_types(df_full)
    df_full = _apply_product_selection(df_full).reset_index(drop=True)
    st.session_state[state_key_df] = df_full

    # Actions
    c1, c2 = st.columns([1, 1])
    apply_btn = c1.button("💾 Guardar cambios", type="primary", key=f"apply_changes_{order.id}_modern")
    discard_btn = c2.button("↩️ Deshacer cambios", key=f"discard_changes_{order.id}_modern")

    if discard_btn:
        if baseline_key in st.session_state:
            st.session_state[state_key_df] = st.session_state[baseline_key].copy(deep=True)
        else:
            st.session_state[state_key_df] = norm_types(coerce_df(rows, full_cols))
        st.session_state.pop(state_key_widget, None)
        st.rerun()

    if apply_btn:
        df_edit = norm_types(coerce_df(st.session_state.get(state_key_df), full_cols))

        to_delete_ids = (
            pd.to_numeric(df_edit.loc[df_edit["eliminar"] == True, "line_id"], errors="coerce")
            .dropna().astype(int).tolist()
        )
        df_keep = df_edit[df_edit["eliminar"] != True].copy()

        df_keep["product_id"] = pd.to_numeric(df_keep["product_id"], errors="coerce")
        df_keep["line_id"] = pd.to_numeric(df_keep["line_id"], errors="coerce")
        df_keep["quantity"] = pd.to_numeric(df_keep["quantity"], errors="coerce")
        df_keep = df_keep.dropna(subset=["product_id"]).copy()
        df_keep["product_id"] = df_keep["product_id"].astype(int)

        if not df_keep.empty:
            df_keep = (
                df_keep.groupby("product_id", as_index=False)
                .agg({
                    "line_id": "first",
                    "product_label": "first",
                    "matched_name": "first",
                    "provider": "first",
                    "unit": "first",
                    "quantity": "sum",
                })
            )

        with get_session() as s:
            if to_delete_ids:
                lines_db = s.exec(select(OrderLine).where(OrderLine.id.in_(to_delete_ids))).all()
                for ln in lines_db:
                    s.delete(ln)

            # upsert
            for _, row in df_keep.iterrows():
                pid = int(row["product_id"])
                prod = id_to_product.get(pid)
                matched_name = getattr(prod, "name", None) if prod else safe_str(row.get("matched_name")).strip() or None
                unit_val = safe_str(getattr(prod, "unit", "") if prod else row.get("unit")).strip().lower() or "unidad"
                qty = float(row.get("quantity") or 0.0)

                line_id = row.get("line_id")
                line_id_int = int(line_id) if line_id is not None and str(line_id) != "nan" else None

                if line_id_int is not None:
                    ln = s.exec(select(OrderLine).where(OrderLine.id == line_id_int)).first()
                    if ln:
                        ln.product_id = pid
                        ln.matched_name = matched_name
                        ln.quantity = qty
                        ln.unit = unit_val
                        ln.updated_at = datetime.utcnow()
                        ln.updated_by = actor
                        s.add(ln)
                    else:
                        s.add(OrderLine(
                            order_id=order.id,
                            venue_id=venue_id,
                            product_id=pid,
                            matched_name=matched_name,
                            quantity=qty,
                            unit=unit_val,
                            spoken_name=(matched_name or ""),
                            updated_at=datetime.utcnow(),
                            updated_by=actor,
                        ))
                else:
                    s.add(OrderLine(
                        order_id=order.id,
                        venue_id=venue_id,
                        product_id=pid,
                        matched_name=matched_name,
                        quantity=qty,
                        unit=unit_val,
                        spoken_name=(matched_name or ""),
                        updated_at=datetime.utcnow(),
                        updated_by=actor,
                    ))

            # ✅ stamp parent order audit
            ord_db = s.exec(select(Order).where(Order.id == order.id)).first()
            if ord_db:
                ord_db.updated_at = datetime.utcnow()
                ord_db.updated_by = actor
                if not ord_db.created_by:
                    ord_db.created_by = actor
                s.add(ord_db)

            s.commit()

        st.success("Cambios guardados ✅")
        st.session_state.pop(state_key_widget, None)
        st.rerun()

    # -------------------------------------------------
    # Supplier drafts section (uses your existing helpers)
    # -------------------------------------------------
    st.subheader("✉️ Borradores para proveedores")

    df_current = norm_types(coerce_df(st.session_state.get(state_key_df), full_cols))
    name_to_product = {getattr(p, "name", ""): p for p in products}

    grouped: Dict[str, Dict[str, object]] = {}
    for _, row in df_current.iterrows():
        matched_name = safe_str(row.get("matched_name")).strip()
        pid = row.get("product_id")

        prod = None
        if pid is not None and pd.notna(pid):
            try:
                prod = id_to_product.get(int(pid))
            except Exception:
                prod = None
        if not prod and matched_name:
            prod = name_to_product.get(matched_name)

        if not prod:
            continue

        prov = getattr(prod, "provider_name", None) or "(Sin proveedor)"
        grouped.setdefault(prov, {
            "provider_email": getattr(prod, "provider_email", "") or "",
            "provider_phone": getattr(prod, "provider_phone", "") or "",
            "provider_address": getattr(prod, "provider_address", "") or "",
            "lines": []
        })
        grouped[prov]["lines"].append({
            "product": matched_name,
            "quantity": num_or_default(row.get("quantity"), None),
            "unit": coalesce_unit(row.get("unit"), prod, "unidad"),
        })

    if not grouped:
        st.info("No hay líneas con producto del catálogo para agrupar por proveedor.")
        return

    cc = st.text_input("Código país para WhatsApp (ej. +34 España, +30 Grecia)", value="+34", key=f"wa_cc_{order.id}_modern")

    def normalize_phone(raw: str) -> str:
        raw = (raw or "").strip()
        digits = ''.join(ch for ch in raw if ch.isdigit() or ch == '+')
        if not digits:
            return ''
        if digits.startswith('+'):
            return digits
        return f"{cc}{digits if not digits.startswith(('0',)) else digits.lstrip('0')}"

    default_subject = f"Pedido — {(order_title or ('#'+str(order.id)))} — {datetime.now().strftime('%Y-%m-%d')}"
    default_open = "Hola,\n\nAdjunto el pedido actualizado. Por favor, confirma disponibilidad y plazos:"
    default_close = "\n\nGracias y un saludo."

    subject_tpl = st.text_input("Asunto del email", value=default_subject, key=f"email_subject_{order.id}_modern")
    opening_tpl = st.text_area("Cabecera del mensaje", value=default_open, height=80, key=f"email_open_{order.id}_modern")
    closing_tpl = st.text_area("Cierre del mensaje", value=default_close, height=60, key=f"email_close_{order.id}_modern")

    st.caption("Ajusta estos textos; abajo verás los borradores por proveedor con enlaces directos y TXT descargable.")

    import urllib.parse as up

    for prov, meta in grouped.items():
        st.markdown(f"### 🧑‍💼 {prov}")

        body_lines = [opening_tpl.strip(), ""]
        for ln in meta["lines"]:  # type: ignore
            qty = ln["quantity"] if ln["quantity"] is not None else ""
            unit = ln["unit"]
            body_lines.append(f"- {ln['product']}: {qty} {unit}".strip())
        body_lines.append(closing_tpl.strip())
        body_text = "\n".join(body_lines)

        provider_email = str(meta.get("provider_email", "") or "")
        provider_phone = str(meta.get("provider_phone", "") or "")
        provider_addr = str(meta.get("provider_address", "") or "")

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.caption("📧 Email")
            st.write(provider_email or "—")
        with c2:
            st.caption("📱 Teléfono")
            st.write(provider_phone or "—")
        with c3:
            st.caption("📍 Dirección")
            st.write(provider_addr or "—")

        ca, cb, ccx = st.columns([1, 1, 2])
        with ca:
            if provider_email:
                mailto = f"mailto:{up.quote(provider_email)}?subject={up.quote(subject_tpl)}&body={up.quote(body_text)}"
                st.markdown(f"[📧 Abrir email]({mailto})")
            else:
                st.caption("(Sin email del proveedor)")
        with cb:
            phone_norm = normalize_phone(provider_phone)
            if phone_norm:
                wa = f"https://wa.me/{phone_norm.replace('+','')}?text={up.quote(body_text)}"
                st.markdown(f"[📲 Abrir WhatsApp]({wa})")
            else:
                st.caption("(Sin teléfono del proveedor)")
        with ccx:
            txt_name = f"pedido_{prov}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
            st.download_button(
                label="⬇️ Descargar TXT del mensaje",
                data=body_text.encode("utf-8"),
                file_name=txt_name,
                mime="text/plain",
                key=f"dl_txt_{order.id}_{prov}_modern"
            )

        with st.expander("👁️ Vista previa del mensaje"):
            st.code(f"Asunto: {subject_tpl}\n\n{body_text}", language="text")

        st.divider()
