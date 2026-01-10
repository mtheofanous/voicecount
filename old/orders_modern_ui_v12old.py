
"""
orders_modern_ui_v10.py
-----------------------
Modern Orders workflow (Streamlit + SQLModel + SQLite-safe migrations)

Workflow:
- draft (Borrador)
- ready_to_send (Listo para enviar)
- pending_receive (Pendiente de recibir)
- final (Historial)  -> read-only

Extras:
- Audit fields on Order: created_by/updated_by + timestamps, verified_by/verified_at
- Presence: see who is editing now (heartbeat)
- Send flow: choose WhatsApp, Email, or both, safe word required; after generating send links, order moves to pending_receive
- Receiving flow: verify received lines, record missing quantities (clamped to ordered qty),
  clarify whether missing appears on invoice, optional note, create a new draft with missing items, and mark all good -> move to history
- History: read-only, shows creator + verifier, and receipt status (Complete/Partial)

Assumptions (you pass these in orders_tab_modern):
- engine
- get_session() contextmanager returning SQLModel Session
- Product model (with provider fields: provider_name/email/phone/address and unit)
- distinct_units(products) -> list[str]
- safe_str(x) -> str
- num_or_default(x, default)
- coalesce_unit(unit_from_line, prod, default)
"""

from __future__ import annotations

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

    # Verification (for history)
    verified_at: Optional[datetime] = Field(default=None, index=True)
    verified_by: Optional[str] = Field(default=None, index=True)

    # Workflow
    status: str = Field(default="draft", index=True)  # draft | ready_to_send | pending_receive | final

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

    # Line audit
    updated_at: Optional[datetime] = Field(default=None, index=True)
    updated_by: Optional[str] = Field(default=None, index=True)

    # Receiving workflow (per line)
    received_ok: Optional[bool] = Field(default=None, index=True)
    missing_qty: Optional[float] = Field(default=None)
    received_at: Optional[datetime] = Field(default=None, index=True)
    received_by: Optional[str] = Field(default=None, index=True)

    # Missing clarification (for supplier follow-up)
    missing_invoice_status: Optional[str] = Field(default="unknown", index=True)  # unknown | in_invoice | not_in_invoice
    missing_note: Optional[str] = Field(default=None)


class OrderPresence(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    order_id: int = Field(index=True)
    venue_id: int = Field(index=True)

    actor: str = Field(index=True)
    session_id: str = Field(index=True)

    last_seen_at: datetime = Field(default_factory=datetime.utcnow, index=True)


# -----------------------------
# Helpers
# -----------------------------

def current_actor() -> str:
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
    if "presence_session_id" not in st.session_state:
        st.session_state["presence_session_id"] = str(uuid.uuid4())
    return str(st.session_state["presence_session_id"])


# -----------------------------
# SQLite-safe migrations
# -----------------------------

def _sqlite_tables(conn) -> set:
    rows = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
    return {r[0] for r in rows}


def ensure_orders_migrations(engine) -> None:
    """Create only the Orders tables + run SQLite-safe ALTERs.

    Important: we intentionally DO NOT call SQLModel.metadata.create_all(engine)
    because that can try to (re)create indexes for unrelated tables (e.g. Product)
    and SQLite index names are global, which may raise 'index ... already exists'.
    """
    # Create only the tables defined in this module (checkfirst=True is safe)
    with engine.begin() as conn:
        for tbl in (Order.__table__, OrderLine.__table__, OrderPresence.__table__):
            try:
                tbl.create(conn, checkfirst=True)
            except Exception:
                # If a table already exists or SQLite quirks occur, continue to ALTER section
                pass

    def _table_cols(conn, table: str) -> set:
        rows = conn.execute(text(f"PRAGMA table_info('{table}')")).fetchall()
        return {r[1] for r in rows}

    with engine.begin() as conn:
        tables = _sqlite_tables(conn)

        # --- order ---
        if "order" in tables:
            existing = _table_cols(conn, "order")
            for col, ddl in [
                ("venue_id",    'ALTER TABLE "order" ADD COLUMN venue_id INTEGER'),
                ("created_at",  'ALTER TABLE "order" ADD COLUMN created_at TEXT'),
                ("created_by",  'ALTER TABLE "order" ADD COLUMN created_by TEXT'),
                ("updated_at",  'ALTER TABLE "order" ADD COLUMN updated_at TEXT'),
                ("updated_by",  'ALTER TABLE "order" ADD COLUMN updated_by TEXT'),
                ("verified_at", 'ALTER TABLE "order" ADD COLUMN verified_at TEXT'),
                ("verified_by", 'ALTER TABLE "order" ADD COLUMN verified_by TEXT'),
                ("status",      'ALTER TABLE "order" ADD COLUMN status TEXT'),
                ("title",       'ALTER TABLE "order" ADD COLUMN title TEXT'),
                ("note",        'ALTER TABLE "order" ADD COLUMN note TEXT'),
            ]:
                if col not in existing:
                    conn.execute(text(ddl))

        # --- orderline ---
        if "orderline" in tables:
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
                # receiving
                ("received_ok", "ALTER TABLE orderline ADD COLUMN received_ok INTEGER"),
                ("missing_qty", "ALTER TABLE orderline ADD COLUMN missing_qty REAL"),
                ("received_at", "ALTER TABLE orderline ADD COLUMN received_at TEXT"),
                ("received_by", "ALTER TABLE orderline ADD COLUMN received_by TEXT"),
                # clarification
                ("missing_invoice_status", "ALTER TABLE orderline ADD COLUMN missing_invoice_status TEXT"),
                ("missing_note", "ALTER TABLE orderline ADD COLUMN missing_note TEXT"),
            ]:
                if col not in existing:
                    conn.execute(text(ddl))

        # --- orderpresence --- created via metadata.create_all


# -----------------------------
# Presence
# -----------------------------

def touch_presence(session, venue_id: int, order_id: int, actor: str, session_id: str, ttl_seconds: int = 45) -> None:
    now = datetime.utcnow()
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
    .block-container { max-width: 1100px; padding-top: 1.1rem; padding-bottom: 4rem; }
    .order-shell { border: 1px solid rgba(0,0,0,0.07); border-radius: 18px; padding: 14px 14px; background: rgba(0,0,0,0.02); }
    .order-top { display:flex; align-items:center; justify-content:space-between; gap: 12px; }
    .order-meta { color: rgba(0,0,0,0.65); font-size: 0.88rem; margin-top: 2px; }
    .badge { display:inline-flex; align-items:center; gap:8px; padding: 6px 10px; border-radius: 999px; font-size: 0.85rem; border: 1px solid rgba(0,0,0,0.10); background: white; }
    .badge-dot { width:8px; height:8px; border-radius:999px; background: rgba(0,0,0,0.25); }
    .badge-draft .badge-dot { background: #f59e0b; }
    .badge-ready .badge-dot { background: #10b981; }
    .badge-pending .badge-dot { background: #3b82f6; }
    .badge-final .badge-dot { background: #64748b; }
    .pill { display:inline-flex; align-items:center; padding: 4px 8px; border-radius: 999px; font-size: 0.82rem; border: 1px solid rgba(0,0,0,0.08); background: white; margin-right: 6px; }
    .muted { color: rgba(0,0,0,0.55); }
    </style>
    """, unsafe_allow_html=True)


def _badge_class(status: str) -> str:
    s = (status or "draft").strip().lower()
    if s == "ready_to_send":
        return "badge badge-ready"
    if s == "pending_receive":
        return "badge badge-pending"
    if s == "final":
        return "badge badge-final"
    return "badge badge-draft"


def _status_label(status: str) -> str:
    s = (status or "draft").strip().lower()
    if s == "ready_to_send":
        return "Listo para enviar"
    if s == "pending_receive":
        return "Pendiente de recibir"
    if s == "final":
        return "Historial"
    return "Borrador"


def _receipt_label(is_complete: bool) -> str:
    return "✅ Completo" if is_complete else "⚠️ Parcial"


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

    # --- Filter + search (status + text + date range) ---
    f1, f2, f3 = st.columns([2, 3, 3], vertical_alignment="center")
    with f1:
        status_filter = st.radio(
            "Filtrar",
            options=["all", "draft", "ready_to_send", "pending_receive", "final"],
            format_func=lambda x: {
                "all": "Todos",
                "draft": "Borrador",
                "ready_to_send": "Listo",
                "pending_receive": "Pendiente recibir",
                "final": "Historial",
            }[x],
            horizontal=True,
            key="orders_filter_status_modern",
        )
    with f2:
        q = st.text_input("Buscar (título o #id)", value="", key="orders_search_modern")
    with f3:
        date_range = st.date_input(
            "Fecha (rango)",
            value=None,
            key="orders_date_range_modern",
            help="Filtra por fecha de creación del pedido. Puedes elegir un rango.",
        )

    def match(o: Order) -> bool:
        if status_filter != "all" and (o.status or "draft") != status_filter:
            return False

        if date_range is not None:
            if isinstance(date_range, tuple) and len(date_range) == 2:
                d0, d1 = date_range
                if d0 is not None and o.created_at.date() < d0:
                    return False
                if d1 is not None and o.created_at.date() > d1:
                    return False
            else:
                d0 = date_range
                if d0 is not None and o.created_at.date() != d0:
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
        st.session_state.pop(f"order_editor_df_{order.id}__baseline", None)

    is_final = (order.status or "").strip().lower() == "final"
    is_ready = (order.status or "").strip().lower() == "ready_to_send"
    is_pending = (order.status or "").strip().lower() == "pending_receive"

    # --- Presence heartbeat ---
    with get_session() as s:
        touch_presence(s, venue_id=venue_id, order_id=order.id, actor=actor, session_id=sid, ttl_seconds=45)
        active = get_active_editors(s, venue_id=venue_id, order_id=order.id, ttl_seconds=45)

    # --- Load products ---
    with get_session() as s:
        products = s.exec(
            select(Product).where(Product.venue_id == venue_id).order_by(Product.name.asc(), Product.provider_name.asc())
        ).all()

    def _product_label(p) -> str:
        prov = getattr(p, "provider_name", None) or "(Sin proveedor)"
        return f"{p.name} — {prov} (#{p.id})"

    placeholder_product = "— Selecciona producto —"
    label_to_product = {_product_label(p): p for p in products}
    id_to_product = {getattr(p, "id", None): p for p in products}

    # Receipt summary (for pending/final)
    with get_session() as s:
        lines_for_summary = s.exec(select(OrderLine).where(OrderLine.order_id == order.id)).all()
    total_missing = sum(float(getattr(ln, "missing_qty", 0.0) or 0.0) for ln in lines_for_summary)
    receipt_complete = (total_missing <= 0.0)

    # --- Header card ---
    st.markdown('<div class="order-shell">', unsafe_allow_html=True)
    left, right = st.columns([3, 2], vertical_alignment="center")

    with left:
        updated_at_txt = order.updated_at.strftime("%Y-%m-%d %H:%M") if order.updated_at else "—"
        created_by_txt = order.created_by or "—"
        updated_by_txt = order.updated_by or "—"
        verified_at_txt = order.verified_at.strftime("%Y-%m-%d %H:%M") if order.verified_at else "—"
        verified_by_txt = order.verified_by or "—"

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

        if active:
            pills = []
            seen_now = datetime.utcnow()
            for a, ts in active[:6]:
                age = int((seen_now - ts).total_seconds())
                pills.append(f'<span class="pill">{a} <span class="muted">· {age}s</span></span>')
            st.markdown("👀 Editando ahora: " + " ".join(pills), unsafe_allow_html=True)

        if is_final:
            st.markdown(
                f'<span class="pill">{_receipt_label(receipt_complete)}</span>'
                f'<span class="pill">🧾 Verificado: {verified_at_txt} · {verified_by_txt}</span>',
                unsafe_allow_html=True,
            )

    with right:
        pretty = {
            "draft": "Borrador",
            "ready_to_send": "Listo para enviar",
            "pending_receive": "Pendiente de recibir",
            "final": "Historial",
        }
        status_options = ["draft", "ready_to_send", "pending_receive", "final"]
        current_idx = status_options.index(order.status) if order.status in status_options else 0

        new_status = st.selectbox(
            "Estado",
            options=status_options,
            index=current_idx,
            format_func=lambda x: pretty.get(x, x),
            key=f"order_status_{order.id}_modern",
            disabled=is_final,
        )

        quick = st.button(
            "✅ Marcar como listo para enviar",
            type="primary",
            disabled=is_final or (order.status == "ready_to_send"),
            key=f"btn_ready_{order.id}_modern",
            use_container_width=True,
        )
        if quick:
            with get_session() as s:
                o = s.exec(select(Order).where(Order.id == order.id)).first()
                if o:
                    o.status = "ready_to_send"
                    o.updated_at = datetime.utcnow()
                    o.updated_by = actor
                    if not o.created_by:
                        o.created_by = actor
                    s.add(o)
                    s.commit()
            st.rerun()

        if (not is_final) and new_status != order.status:
            if new_status in {"pending_receive", "final"}:
                st.info("Ese estado se asigna automáticamente por el flujo (Enviar / Verificar).")
            else:
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

        can_delete = (order.status in {"draft", "ready_to_send"}) and (role in {None, "owner", "manager"}) and (not is_final)
        with st.popover("🗑️ Eliminar pedido", disabled=not can_delete):
            if not can_delete:
                st.info("Solo se pueden eliminar pedidos en **Borrador** o **Listo** (y con permisos).")
            else:
                st.warning("Esto eliminará el pedido y TODAS sus líneas. No se puede deshacer.")
                safeword = f"DELETE {order.id}"
                typed = st.text_input("Palabra segura", placeholder=safeword, key=f"delete_safeword_{order.id}")
                confirm = st.button("Eliminar definitivamente", type="primary", key=f"btn_delete_{order.id}")
                if confirm:
                    if (typed or "").strip() != safeword:
                        st.error("Palabra segura incorrecta.")
                    else:
                        with get_session() as s:
                            pres = s.exec(select(OrderPresence).where(
                                OrderPresence.venue_id == venue_id,
                                OrderPresence.order_id == order.id
                            )).all()
                            for r in pres:
                                s.delete(r)
                            lines_db = s.exec(select(OrderLine).where(OrderLine.order_id == order.id)).all()
                            for ln in lines_db:
                                s.delete(ln)
                            o = s.exec(select(Order).where(Order.id == order.id)).first()
                            if o:
                                s.delete(o)
                            s.commit()
                        st.success(f"Pedido #{order.id} eliminado ✅")
                        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # --- Title/Note (locked in history) ---
    cA, cB = st.columns([3, 1], vertical_alignment="center")
    with cA:
        order_title = st.text_input("Título", value=order.title or "", key=f"order_title_{order.id}_modern", disabled=is_final)
    with cB:
        note = st.text_input("Nota", value=order.note or "", key=f"order_note_{order.id}_modern", disabled=is_final)

    if (not is_final) and ((order.title or "") != order_title or (order.note or "") != note):
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

    
    # -------------------------------------------------
    # 📊 Incidencias por proveedor (vista rápida)
    # -------------------------------------------------
    with st.expander("📊 Incidencias por proveedor (últimos pedidos)", expanded=False):
        with get_session() as s:
            # Join orders + lines to compute provider incident stats for this venue
            rows_join = s.exec(
                select(OrderLine, Order).where(
                    Order.venue_id == venue_id,
                    OrderLine.order_id == Order.id,
                    OrderLine.venue_id == venue_id,
                )
            ).all()

        if not rows_join:
            st.caption("Sin datos aún.")
        else:
            stats = {}
            for ln, o in rows_join:
                prov = (getattr(ln, "provider", None) or "(Sin proveedor)")
                oid = getattr(o, "id", None)
                created_at = getattr(o, "created_at", None)
                miss = float(getattr(ln, "missing_qty", 0.0) or 0.0)

                s0 = stats.setdefault(prov, {
                    "Proveedor": prov,
                    "Pedidos (con líneas)": set(),
                    "Pedidos con incidencias": set(),
                    "Líneas con incidencias": 0,
                    "Unidades faltantes": 0.0,
                    "Última incidencia": None,
                })

                if oid is not None:
                    s0["Pedidos (con líneas)"].add(int(oid))
                if miss > 0.0:
                    if oid is not None:
                        s0["Pedidos con incidencias"].add(int(oid))
                    s0["Líneas con incidencias"] += 1
                    s0["Unidades faltantes"] += miss
                    if created_at:
                        if (s0["Última incidencia"] is None) or (created_at > s0["Última incidencia"]):
                            s0["Última incidencia"] = created_at

            out = []
            for prov, s0 in stats.items():
                total_orders = len(s0["Pedidos (con líneas)"])
                inc_orders = len(s0["Pedidos con incidencias"])
                rate = (inc_orders / total_orders) if total_orders else 0.0
                last = s0["Última incidencia"].strftime("%Y-%m-%d") if s0["Última incidencia"] else "—"
                out.append({
                    "Proveedor": prov,
                    "Pedidos": total_orders,
                    "Con incidencias": inc_orders,
                    "% pedidos con incidencia": round(rate * 100, 1),
                    "Líneas con incidencia": int(s0["Líneas con incidencias"]),
                    "Unidades faltantes": round(float(s0["Unidades faltantes"]), 2),
                    "Última incidencia": last,
                })

            df_stats = pd.DataFrame(out).sort_values(by=["% pedidos con incidencia", "Unidades faltantes"], ascending=False)
            st.dataframe(df_stats, width="stretch", hide_index=True)
            st.caption("Tip: Esto usa los datos de recepción (faltantes) para ayudarte a detectar proveedores con incidencias recurrentes.")

# ---------- Load lines from DB ----------
    with get_session() as s:
        lines = s.exec(select(OrderLine).where(OrderLine.order_id == order.id)).all()

    # Build editor dataframe
    rows = []
    for ln in lines:
        pid = getattr(ln, "product_id", None)
        prod = id_to_product.get(pid) if pid is not None else None
        name = (getattr(prod, "name", None) if prod else None) or (ln.matched_name or ln.spoken_name or "")
        provider = (getattr(prod, "provider_name", None) if prod else None) or (ln.provider or "")
        unit_val = (getattr(prod, "unit", None) if prod else None) or (ln.unit or "unidad")
        rows.append({
            "line_id": getattr(ln, "id", pd.NA),
            "product_label": _product_label(prod) if prod else name,
            "product_id": pid if pid is not None else pd.NA,
            "matched_name": name,
            "provider": provider,
            "quantity": getattr(ln, "quantity", None),
            "unit": (str(unit_val).strip().lower() if unit_val else "unidad"),
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
        df0 = norm_types(coerce_df(rows, full_cols))
        st.session_state[state_key_df] = df0
        st.session_state[baseline_key] = df0.copy(deep=True)

    st.subheader("🧾 Líneas del pedido")

    colcfg = {
        "product_label": st.column_config.TextColumn("Producto", disabled=True),
        "quantity": st.column_config.NumberColumn("Cantidad", min_value=0.0, step=1.0, format="%.2f", disabled=is_final),
        "unit": st.column_config.TextColumn("Unidad", disabled=True),
        "eliminar": st.column_config.CheckboxColumn("Eliminar", disabled=is_final),
    }

    df_full = norm_types(coerce_df(st.session_state.get(state_key_df), full_cols)).reset_index(drop=True)
    df_view = df_full[visible_cols].copy()

    edited_view = st.data_editor(
        df_view,
        width="stretch",
        hide_index=True,
        num_rows="fixed",
        column_config=colcfg,
        key=state_key_widget,
        disabled=is_final,
    )

    df_vis = norm_types(coerce_df(edited_view, visible_cols)).reset_index(drop=True)
    n = min(len(df_full), len(df_vis))
    df_full = df_full.iloc[:n].reset_index(drop=True)
    df_vis = df_vis.iloc[:n].reset_index(drop=True)
    for c in visible_cols:
        df_full[c] = df_vis[c].values
    st.session_state[state_key_df] = df_full

    c1, c2 = st.columns([1, 1])
    apply_btn = c1.button("💾 Guardar cambios", type="primary", key=f"apply_changes_{order.id}_modern", disabled=is_final)
    discard_btn = c2.button("↩️ Deshacer cambios", key=f"discard_changes_{order.id}_modern", disabled=is_final)

    if discard_btn and (not is_final):
        st.session_state[state_key_df] = st.session_state.get(baseline_key, df_full).copy(deep=True)
        st.session_state.pop(state_key_widget, None)
        st.rerun()

    if apply_btn and (not is_final):
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
                    "matched_name": "first",
                    "provider": "first",
                    "unit": "first",
                    "quantity": "sum",
                })
            )

        now = datetime.utcnow()
        with get_session() as s:
            if to_delete_ids:
                lines_db = s.exec(select(OrderLine).where(OrderLine.id.in_(to_delete_ids))).all()
                for ln in lines_db:
                    s.delete(ln)

            for _, row in df_keep.iterrows():
                pid = int(row["product_id"])
                line_id = row.get("line_id")
                line_id_int = int(line_id) if line_id is not None and str(line_id) != "nan" else None
                qty = float(row.get("quantity") or 0.0)
                unit_val = str(row.get("unit") or "unidad").strip().lower()

                if line_id_int is not None:
                    ln = s.exec(select(OrderLine).where(OrderLine.id == line_id_int)).first()
                    if ln:
                        ln.product_id = pid
                        ln.matched_name = str(row.get("matched_name") or "").strip() or None
                        ln.provider = str(row.get("provider") or "").strip() or None
                        ln.quantity = qty
                        ln.unit = unit_val
                        ln.updated_at = now
                        ln.updated_by = actor
                        s.add(ln)
                else:
                    s.add(OrderLine(
                        order_id=order.id,
                        venue_id=venue_id,
                        product_id=pid,
                        matched_name=str(row.get("matched_name") or "").strip() or None,
                        provider=str(row.get("provider") or "").strip() or None,
                        quantity=qty,
                        unit=unit_val,
                        spoken_name=str(row.get("matched_name") or ""),
                        updated_at=now,
                        updated_by=actor,
                    ))

            o = s.exec(select(Order).where(Order.id == order.id)).first()
            if o:
                o.updated_at = now
                o.updated_by = actor
                if not o.created_by:
                    o.created_by = actor
                s.add(o)
            s.commit()

        st.success("Cambios guardados ✅")
        st.session_state[baseline_key] = st.session_state[state_key_df].copy(deep=True)
        st.rerun()

    # -------------------------------------------------
    # Receiving workflow (Pendiente de recibir)
    # -------------------------------------------------
    if is_pending:
        st.subheader("📦 Recepción del pedido")

        with get_session() as s:
            lines_rx = s.exec(select(OrderLine).where(OrderLine.order_id == order.id)).all()

        if not lines_rx:
            st.info("Este pedido no tiene líneas.")
        else:
            rx_rows = []
            for ln in lines_rx:
                pid = getattr(ln, "product_id", None)
                prod = id_to_product.get(pid) if pid is not None else None
                name = (getattr(prod, "name", None) if prod else None) or (ln.matched_name or ln.spoken_name or "Producto")
                unit_val = (getattr(prod, "unit", None) if prod else None) or (ln.unit or "unidad")
                qty = float(getattr(ln, "quantity", 0.0) or 0.0)
                ok = getattr(ln, "received_ok", None)
                miss = getattr(ln, "missing_qty", None)
                rx_rows.append({
                    "line_id": getattr(ln, "id", None),
                    "producto": name,
                    "pedido": qty,
                    "unidad": str(unit_val).strip().lower(),
                    "recibido_ok": True if ok is True else False,
                    "faltante": float(miss or 0.0),
                    "factura": (getattr(ln, "missing_invoice_status", None) or "unknown"),
                    "nota": (getattr(ln, "missing_note", None) or ""),
                })

            rx_df = pd.DataFrame(rx_rows)
            total_lines = len(rx_df)
            ok_lines = int((rx_df["recibido_ok"] == True).sum())
            st.progress(ok_lines / total_lines if total_lines else 0.0, text=f"{ok_lines}/{total_lines} líneas confirmadas")

            st.caption("Marca lo que llegó. Si falta algo, pon cantidad faltante (máx = pedido). Y aclara si está en la factura.")

            rx_edit = st.data_editor(
                rx_df,
                width="stretch",
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "producto": st.column_config.TextColumn("Producto", disabled=True),
                    "pedido": st.column_config.NumberColumn("Pedido", disabled=True, format="%.2f"),
                    "unidad": st.column_config.TextColumn("Unidad", disabled=True),
                    "recibido_ok": st.column_config.CheckboxColumn("Recibido", help="Marca si esa línea llegó completa"),
                    "faltante": st.column_config.NumberColumn("Faltante", min_value=0.0, step=1.0, format="%.2f"),
                    "factura": st.column_config.SelectboxColumn(
                        "Factura",
                        options=["unknown", "in_invoice", "not_in_invoice"],
                        format_func=lambda x: {"unknown":"—", "in_invoice":"Está en factura", "not_in_invoice":"No está en factura"}.get(x, x),
                        help="Aclara si el faltante aparece en la factura. Útil para reclamar/seguir con el proveedor."
                    ),
                    "nota": st.column_config.TextColumn("Nota", help="Ej: entregan mañana / dicen sin stock", max_chars=80),
                },
                key=f"rx_editor_{order.id}",
            )

            # Normalize & clamp
            rx_edit = rx_edit.copy()
            rx_edit.loc[rx_edit["recibido_ok"] == True, "faltante"] = 0.0
            rx_edit["faltante"] = pd.to_numeric(rx_edit["faltante"], errors="coerce").fillna(0.0)
            rx_edit["pedido"] = pd.to_numeric(rx_edit["pedido"], errors="coerce").fillna(0.0)
            rx_edit["faltante"] = rx_edit.apply(lambda r: max(0.0, min(float(r["faltante"]), float(r["pedido"]))), axis=1)
            rx_edit.loc[rx_edit["faltante"] > 0.0, "recibido_ok"] = False

            c_rx1, c_rx2, c_rx3 = st.columns([1, 1, 2], vertical_alignment="center")
            with c_rx1:
                save_rx = st.button("💾 Guardar recepción", type="primary", key=f"save_rx_{order.id}")
            with c_rx2:
                all_good = st.button("✅ Todo OK → Historial", key=f"all_good_{order.id}")
            with c_rx3:
                has_missing = float(rx_edit["faltante"].fillna(0).sum()) > 0.0
                create_missing = st.button("➕ Crear borrador con faltantes", disabled=not has_missing, key=f"create_missing_{order.id}")

            if save_rx or all_good or create_missing:
                with get_session() as s:
                    now = datetime.utcnow()

                    for _, r in rx_edit.iterrows():
                        lid = int(r["line_id"])
                        ln = s.exec(select(OrderLine).where(OrderLine.id == lid)).first()
                        if not ln:
                            continue

                        ordered = float(r.get("pedido") or 0.0)
                        falt = float(r.get("faltante") or 0.0)
                        falt = max(0.0, min(falt, ordered))

                        okv = bool(r.get("recibido_ok") or False)
                        ln.received_ok = okv and (falt == 0.0)
                        ln.missing_qty = falt

                        inv = (r.get("factura") or "unknown")
                        note_txt = (r.get("nota") or "").strip()
                        if falt == 0.0:
                            inv = "unknown"
                            note_txt = ""
                        ln.missing_invoice_status = str(inv)
                        ln.missing_note = note_txt

                        ln.received_at = now
                        ln.received_by = actor
                        ln.updated_at = now
                        ln.updated_by = actor
                        s.add(ln)

                    o = s.exec(select(Order).where(Order.id == order.id)).first()
                    if o:
                        o.updated_at = now
                        o.updated_by = actor
                        s.add(o)

                    if all_good:
                        any_missing = float(rx_edit["faltante"].fillna(0).sum()) > 0.0
                        if any_missing:
                            st.error("Hay faltantes. Pon faltante = 0 en todas las líneas para marcar Todo OK.")
                            s.commit()
                        else:
                            if o:
                                o.status = "final"
                                o.verified_at = now
                                o.verified_by = actor
                                s.add(o)
                            s.commit()
                            st.success("Pedido movido a Historial ✅")
                            st.rerun()

                    if create_missing:
                        missing_rows = rx_edit[rx_edit["faltante"].fillna(0) > 0].copy()
                        if missing_rows.empty:
                            st.info("No hay faltantes.")
                            s.commit()
                        else:
                            new_order = Order(
                                venue_id=venue_id,
                                status="draft",
                                title=f"Faltantes de #{order.id}",
                                created_by=actor,
                                created_at=now,
                                updated_at=now,
                                updated_by=actor,
                            )
                            s.add(new_order)
                            s.commit()
                            s.refresh(new_order)

                            for _, r in missing_rows.iterrows():
                                lid = int(r["line_id"])
                                ln_old = s.exec(select(OrderLine).where(OrderLine.id == lid)).first()
                                if not ln_old:
                                    continue
                                qty_missing = float(r["faltante"] or 0.0)
                                s.add(OrderLine(
                                    venue_id=venue_id,
                                    order_id=new_order.id,
                                    product_id=getattr(ln_old, "product_id", None),
                                    matched_name=getattr(ln_old, "matched_name", None),
                                    provider=getattr(ln_old, "provider", None),
                                    unit=getattr(ln_old, "unit", None),
                                    quantity=qty_missing,
                                    spoken_name=(getattr(ln_old, "spoken_name", "") or getattr(ln_old, "matched_name", "") or ""),
                                    confidence=getattr(ln_old, "confidence", 0.0) or 0.0,
                                    missing_invoice_status=getattr(ln_old, "missing_invoice_status", "unknown") or "unknown",
                                    missing_note=getattr(ln_old, "missing_note", None),
                                    updated_at=now,
                                    updated_by=actor,
                                ))
                            s.commit()
                            st.success(f"Borrador creado ✅ (Pedido #{new_order.id})")
                            st.rerun()

                    s.commit()
                    st.success("Recepción guardada ✅")
                    st.rerun()

        st.divider()

    # -------------------------------------------------
    # Supplier drafts + Send (only in ready_to_send)
    # -------------------------------------------------
    if not is_ready:
        if is_final:
            st.caption("📚 Historial: solo lectura.")
        elif is_pending:
            st.caption("📦 Recepción: disponible arriba.")
        else:
            st.caption("✉️ Los borradores para proveedores se muestran cuando el pedido está en **Listo para enviar**.")
        return

    st.subheader("✉️ Borradores para proveedores")

    # Group from DB lines (source of truth, includes clarification)
    with get_session() as s:
        lines_db = s.exec(select(OrderLine).where(OrderLine.order_id == order.id)).all()

    grouped: Dict[str, Dict[str, object]] = {}
    for ln in lines_db:
        pid = getattr(ln, "product_id", None)
        prod = id_to_product.get(pid) if pid is not None else None
        name = (getattr(prod, "name", None) if prod else None) or (ln.matched_name or ln.spoken_name or "")
        if not name:
            continue
        prov = (getattr(prod, "provider_name", None) if prod else None) or (ln.provider or "(Sin proveedor)")

        grouped.setdefault(prov, {
            "provider_email": (getattr(prod, "provider_email", "") if prod else "") or "",
            "provider_phone": (getattr(prod, "provider_phone", "") if prod else "") or "",
            "provider_address": (getattr(prod, "provider_address", "") if prod else "") or "",
            "lines": []
        })
        grouped[prov]["lines"].append({
            "product": name,
            "quantity": num_or_default(getattr(ln, "quantity", None), None),
            "unit": coalesce_unit(getattr(ln, "unit", None), prod, "unidad"),
            "missing_invoice_status": (getattr(ln, "missing_invoice_status", None) or "unknown"),
            "missing_note": (getattr(ln, "missing_note", None) or ""),
        })

    if not grouped:
        st.info("No hay líneas con producto del catálogo para agrupar por proveedor.")
        return

    # Send UI
    st.subheader("📤 Enviar a proveedores")

    send_c1, send_c2, send_c3 = st.columns([1, 1, 2], vertical_alignment="center")
    with send_c1:
        send_whatsapp = st.checkbox("WhatsApp", value=True, key=f"send_wa_{order.id}")
    with send_c2:
        send_email = st.checkbox("Email", value=True, key=f"send_email_{order.id}")
    with send_c3:
        move_to_pending = st.checkbox("Pasar a Pendiente de recibir después de enviar", value=True, key=f"send_pending_{order.id}")

    if not (send_whatsapp or send_email):
        st.info("Selecciona al menos un canal (WhatsApp y/o Email).")
        send_generated = False
    else:
        with st.popover("✅ Enviar ahora"):
            st.warning("Esto generará enlaces listos para enviar (WhatsApp Web / mailto).")
            safeword = f"SEND {order.id}"
            typed = st.text_input("Palabra segura", placeholder=safeword, key=f"send_safeword_{order.id}")
            do_send = st.button("Generar envíos", type="primary", key=f"btn_do_send_{order.id}")

            if do_send:
                if (typed or "").strip() != safeword:
                    st.error("Palabra segura incorrecta.")
                else:
                    st.session_state[f"send_generated_{order.id}"] = True

                    with get_session() as s:
                        o = s.exec(select(Order).where(Order.id == order.id)).first()
                        if o:
                            o.updated_at = datetime.utcnow()
                            o.updated_by = actor
                            if not o.created_by:
                                o.created_by = actor
                            if move_to_pending:
                                o.status = "pending_receive"
                            s.add(o)
                            s.commit()

                    st.success("Generado ✅ Abajo tienes los enlaces por proveedor.")
                    st.rerun()

        send_generated = bool(st.session_state.get(f"send_generated_{order.id}", False))

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

    import urllib.parse as up

    for prov, meta in grouped.items():
        st.markdown(f"### 🧑‍💼 {prov}")

        # Reclamación (si hay contexto de faltantes / factura / nota en alguna línea)
        has_claim_context = False
        for _ln in meta["lines"]:  # type: ignore
            inv = (_ln.get("missing_invoice_status") or "unknown")
            note_txt = (_ln.get("missing_note") or "").strip()
            if inv != "unknown" or note_txt:
                has_claim_context = True
                break

        claim_key = f"claim_{order.id}_{prov}_modern"
        c_claim1, c_claim2 = st.columns([1, 3], vertical_alignment="center")
        with c_claim1:
            if has_claim_context:
                if st.button("🧾 Incluir reclamación", key=claim_key):
                    st.session_state[claim_key] = True
            else:
                st.caption("")

        with c_claim2:
            if has_claim_context:
                st.caption("Añade automáticamente una sección de reclamación (faltantes/factura/notas) en el mensaje al proveedor.")
            else:
                st.caption("")

        body_lines = [opening_tpl.strip(), ""]

        if st.session_state.get(claim_key, False):
            body_lines.append("—")
            body_lines.append("🧾 *Reclamación / seguimiento*")
            body_lines.append("En el último pedido hubo incidencias. Detalle:")
            for _ln in meta["lines"]:  # type: ignore
                inv = (_ln.get("missing_invoice_status") or "unknown")
                note_txt = (_ln.get("missing_note") or "").strip()
                # Only list items with context (avoid noise)
                if inv == "unknown" and not note_txt:
                    continue
                inv_txt = {"in_invoice":"Está en factura", "not_in_invoice":"No está en factura", "unknown":"—"}.get(inv, inv)
                parts = []
                if inv != "unknown":
                    parts.append(f"Factura: {inv_txt}")
                if note_txt:
                    parts.append(f"Nota: {note_txt}")
                suffix = " · ".join(parts)
                body_lines.append(f"- {_ln.get('product')}: {suffix}".strip())
            body_lines.append("—")
            body_lines.append("Por favor, confirmad cómo lo resolvemos y si debéis incluir abono/reenvío.")
            body_lines.append("")
        for ln in meta["lines"]:  # type: ignore
            qty = ln.get("quantity") if ln.get("quantity") is not None else ""
            unit = ln.get("unit", "")
            extra = ""
            inv = (ln.get("missing_invoice_status") or "unknown")
            note_txt = (ln.get("missing_note") or "").strip()
            if inv != "unknown" or note_txt:
                inv_txt = {"in_invoice":"en factura", "not_in_invoice":"no en factura", "unknown":"—"}.get(inv, inv)
                parts = []
                if inv != "unknown":
                    parts.append(f"factura: {inv_txt}")
                if note_txt:
                    parts.append(f"nota: {note_txt}")
                extra = "  (" + " · ".join(parts) + ")"

            body_lines.append(f"- {ln.get('product')}: {qty} {unit}{extra}".strip())

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

        if not send_generated:
            st.caption("🔒 Enlaces desactivados hasta generar envíos.")
        else:
            ca, cb, ccx = st.columns([1, 1, 2])
            with ca:
                if send_email and provider_email:
                    mailto = f"mailto:{up.quote(provider_email)}?subject={up.quote(subject_tpl)}&body={up.quote(body_text)}"
                    st.markdown(f"[📧 Abrir email]({mailto})")
                else:
                    st.caption("(Email desactivado o sin email del proveedor)")
            with cb:
                phone_norm = normalize_phone(provider_phone)
                if send_whatsapp and phone_norm:
                    wa = f"https://wa.me/{phone_norm.replace('+','')}?text={up.quote(body_text)}"
                    st.markdown(f"[📲 Abrir WhatsApp]({wa})")
                else:
                    st.caption("(WhatsApp desactivado o sin teléfono del proveedor)")
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
