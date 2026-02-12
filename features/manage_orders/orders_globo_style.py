# Replace the section from line 1209 to line 1430 in your orders.py file
# This creates a Globo-style mobile-optimized product selection interface

    # -----------------------------
    # GLOBO-STYLE PRODUCT SELECTION
    # -----------------------------
    with st.expander("➕ Añadir productos", expanded=True):
        # Backwards compatible cleanup (older sessions may still carry legacy keys)
        legacy_reset_flag = f"{editor_key}__qa_reset_qty"
        if st.session_state.get(legacy_reset_flag):
            qty_prefix = f"{editor_key}__qa_qty_"
            for k in list(st.session_state.keys()):
                if isinstance(k, str) and k.startswith(qty_prefix):
                    st.session_state.pop(k, None)
            st.session_state.pop(legacy_reset_flag, None)

        # Search bar
        q = st.text_input(
            "Buscar",
            key=f"{editor_key}__qa_search",
            placeholder="Buscar productos...",
        ).strip().lower()

        # Prepare order state
        df_current = _sanitize_editor_df(st.session_state[df_state_key])
        qty_by_pid = _qty_by_product(df_current)

        # Filter by search
        base_pids = sorted(label_by_id.keys())
        if q:
            base_pids = [pid for pid in base_pids if q in label_by_id.get(pid, "").lower()]

        # Session keys
        prov_key = f"{editor_key}__qa_prov"
        cat_key = f"{editor_key}__qa_cat"
        hide_key = f"{editor_key}__qa_hide"

        st.session_state.setdefault(prov_key, "Todos")
        st.session_state.setdefault(cat_key, "Todas")
        
        current_prov = st.session_state[prov_key]
        current_cat = st.session_state[cat_key]
        hide_in_order = bool(st.session_state.get(hide_key, False))

        # --- Compute cascading options ---
        # Get available providers based on current category
        if current_cat != "Todas":
            provs_for_cat = sorted({prov_by_pid.get(pid, "") for pid in base_pids if cat_by_pid.get(pid, "") == current_cat})
        else:
            provs_for_cat = sorted({prov_by_pid.get(pid, "") for pid in base_pids})
        provs_for_cat = [p for p in provs_for_cat if p]

        # Get available categories based on current provider
        if current_prov != "Todos":
            cats_for_prov = sorted({cat_by_pid.get(pid, "") for pid in base_pids if prov_by_pid.get(pid, "") == current_prov})
        else:
            cats_for_prov = sorted({cat_by_pid.get(pid, "") for pid in base_pids})
        cats_for_prov = [c for c in cats_for_prov if c]

        # Reset invalid selections
        if current_prov != "Todos" and current_prov not in provs_for_cat:
            st.session_state[prov_key] = "Todos"
            current_prov = "Todos"

        if current_cat != "Todas" and current_cat not in cats_for_prov:
            st.session_state[cat_key] = "Todas"
            current_cat = "Todas"

        # -----------------------------
        # HORIZONTAL SCROLLING PROVIDERS (Globo style)
        # -----------------------------
        st.markdown("**Proveedores**")
        
        # Custom CSS for horizontal scrolling pills
        st.markdown("""
        <style>
        .pill-container {
            display: flex;
            gap: 8px;
            overflow-x: auto;
            padding: 8px 0;
            margin-bottom: 16px;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: thin;
        }
        .pill-container::-webkit-scrollbar {
            height: 6px;
        }
        .pill-container::-webkit-scrollbar-track {
            background: transparent;
        }
        .pill-container::-webkit-scrollbar-thumb {
            background: rgba(49,51,63,.25);
            border-radius: 999px;
        }
        .pill {
            display: inline-block;
            padding: 10px 18px;
            border-radius: 999px;
            background: rgba(49,51,63,.08);
            border: 1px solid rgba(49,51,63,.18);
            font-size: 0.9rem;
            font-weight: 600;
            white-space: nowrap;
            cursor: pointer;
            transition: all 0.2s ease;
            user-select: none;
        }
        .pill-active {
            background: #FDB913;
            border-color: #FDB913;
            color: #000;
            font-weight: 700;
        }
        </style>
        """, unsafe_allow_html=True)

        # Create provider pills
        prov_cols = st.columns(len(["Todos"] + provs_for_cat))
        for i, prov in enumerate(["Todos"] + provs_for_cat):
            with prov_cols[i]:
                is_active = prov == current_prov
                btn_type = "primary" if is_active else "secondary"
                if st.button(
                    prov,
                    key=f"{prov_key}_{i}",
                    type=btn_type,
                    use_container_width=True,
                ):
                    st.session_state[prov_key] = prov
                    st.rerun()

        st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)

        # -----------------------------
        # HORIZONTAL SCROLLING CATEGORIES (Globo style)
        # -----------------------------
        st.markdown("**Categorías**")
        
        # Re-evaluate categories after provider selection
        if st.session_state[prov_key] != "Todos":
            cat_opts = sorted({cat_by_pid.get(pid, "") for pid in base_pids if prov_by_pid.get(pid, "") == st.session_state[prov_key]})
        else:
            cat_opts = sorted({cat_by_pid.get(pid, "") for pid in base_pids})
        cat_opts = [c for c in cat_opts if c]

        # Create category pills
        cat_cols = st.columns(len(["Todas"] + cat_opts))
        for i, cat in enumerate(["Todas"] + cat_opts):
            with cat_cols[i]:
                is_active = cat == current_cat
                btn_type = "primary" if is_active else "secondary"
                if st.button(
                    cat,
                    key=f"{cat_key}_{i}",
                    type=btn_type,
                    use_container_width=True,
                ):
                    st.session_state[cat_key] = cat
                    st.rerun()

        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

        # ---------- Reset paging when filters change ----------
        selected_prov = st.session_state[prov_key]
        selected_cat = st.session_state[cat_key]
        
        filters_sig = (q, selected_cat, selected_prov, bool(hide_in_order))
        sig_key = f"{editor_key}__qa_filters_sig"
        page_key = f"{editor_key}__qa_page"

        if st.session_state.get(sig_key) != filters_sig:
            st.session_state[sig_key] = filters_sig
            st.session_state[page_key] = 1

        # ---------- Apply filters ----------
        pids = base_pids

        if selected_cat != "Todas":
            pids = [pid for pid in pids if cat_by_pid.get(pid, "") == selected_cat]

        if selected_prov != "Todos":
            pids = [pid for pid in pids if prov_by_pid.get(pid, "") == selected_prov]

        if hide_in_order:
            pids = [pid for pid in pids if float(qty_by_pid.get(pid, 0.0) or 0.0) <= 0.0]

        total = len(pids)

        # ---------- Paging ----------
        page_size = 20  # Reduced for mobile (2 columns x 10 rows)

        st.session_state.setdefault(page_key, 1)
        total_pages = max(1, (total + page_size - 1) // page_size)
        st.session_state[page_key] = min(st.session_state[page_key], total_pages)

        # Paging controls + toggle
        with st.container():
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                if st.button("◀", disabled=st.session_state[page_key] <= 1, use_container_width=True):
                    st.session_state[page_key] -= 1
                    st.rerun()

            with col2:
                st.markdown(
                    f"<div style='text-align: center; padding-top: 8px; font-weight: 600; color: #64748b'>"
                    f"{total} productos · Pág {st.session_state[page_key]}/{total_pages}"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
            with col3:
                if st.button("▶", disabled=st.session_state[page_key] >= total_pages, use_container_width=True):
                    st.session_state[page_key] += 1
                    st.rerun()

        hide_in_order = st.toggle(
            "Ocultar productos ya en pedido",
            key=hide_key,
        )

        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

        # ---------- 2-COLUMN GRID (Mobile optimized, like Globo) ----------
        start_i = (st.session_state[page_key] - 1) * page_size
        end_i = start_i + page_size
        pids_page = pids[start_i:end_i]

        # Custom CSS for product cards
        st.markdown("""
        <style>
        .product-card {
            border: 1px solid rgba(49,51,63,.12);
            border-radius: 12px;
            padding: 12px;
            background: #fff;
            margin-bottom: 12px;
            position: relative;
        }
        .product-card-in-order {
            border: 2px solid #2196F3;
            background: rgba(33,150,243,0.04);
        }
        .product-name {
            font-weight: 700;
            font-size: 0.95rem;
            line-height: 1.3;
            margin-bottom: 4px;
            color: #1a1a1a;
        }
        .product-desc {
            font-size: 0.85rem;
            color: #64748b;
            margin-bottom: 8px;
            line-height: 1.2;
        }
        .product-unit {
            font-size: 0.85rem;
            font-weight: 600;
            color: #475569;
            margin-bottom: 8px;
        }
        .in-order-badge {
            display: inline-block;
            background: #2196F3;
            color: white;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
            margin-top: 6px;
        }
        .add-button {
            margin-top: 8px;
        }
        </style>
        """, unsafe_allow_html=True)

        # Render products in 2-column grid
        with st.container():
            # Process products in pairs for 2-column layout
            for row_idx in range(0, len(pids_page), 2):
                cols = st.columns(2, gap="small")
                
                # Left column product
                if row_idx < len(pids_page):
                    pid = pids_page[row_idx]
                    with cols[0]:
                        p = products_by_id.get(pid)
                        unit_txt = (_s(getattr(p, "unit", "")) or "unidad").lower()

                        label = label_by_id.get(pid, str(pid))
                        parts = label.split(" — ", 1)
                        name = parts[0]
                        rest = parts[1] if len(parts) > 1 else ""

                        existing_qty = float(qty_by_pid.get(pid, 0.0) or 0.0)
                        in_order = existing_qty > 0

                        # Product card
                        card_class = "product-card-in-order" if in_order else "product-card"
                        
                        card_html = f"""
                        <div class="{card_class}">
                            <div class="product-name">{name}</div>
                            {f'<div class="product-desc">{rest}</div>' if rest else ''}
                            <div class="product-unit">{unit_txt}</div>
                            {f'<span class="in-order-badge">✓ En pedido: {existing_qty:g}</span>' if in_order else ''}
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)

                        # Add form
                        form_key = f"{editor_key}__qa_form_{pid}"
                        with st.form(key=form_key, clear_on_submit=False):
                            qty_val = st.number_input(
                                "Cantidad",
                                min_value=0,
                                step=1,
                                value=0,
                                key=f"{editor_key}__qa_qty_{pid}__{st.session_state[qa_nonce_key]}",
                                label_visibility="collapsed",
                            )
                            submitted = st.form_submit_button(
                                "➕ Sumar" if in_order else "➕ Añadir",
                                use_container_width=True,
                                type="primary" if not in_order else "secondary",
                            )

                        if submitted:
                            _add_product_to_df(pid, qty_val)
                
                # Right column product
                if row_idx + 1 < len(pids_page):
                    pid = pids_page[row_idx + 1]
                    with cols[1]:
                        p = products_by_id.get(pid)
                        unit_txt = (_s(getattr(p, "unit", "")) or "unidad").lower()

                        label = label_by_id.get(pid, str(pid))
                        parts = label.split(" — ", 1)
                        name = parts[0]
                        rest = parts[1] if len(parts) > 1 else ""

                        existing_qty = float(qty_by_pid.get(pid, 0.0) or 0.0)
                        in_order = existing_qty > 0

                        # Product card
                        card_class = "product-card-in-order" if in_order else "product-card"
                        
                        card_html = f"""
                        <div class="{card_class}">
                            <div class="product-name">{name}</div>
                            {f'<div class="product-desc">{rest}</div>' if rest else ''}
                            <div class="product-unit">{unit_txt}</div>
                            {f'<span class="in-order-badge">✓ En pedido: {existing_qty:g}</span>' if in_order else ''}
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)

                        # Add form
                        form_key = f"{editor_key}__qa_form_{pid}"
                        with st.form(key=form_key, clear_on_submit=False):
                            qty_val = st.number_input(
                                "Cantidad",
                                min_value=0,
                                step=1,
                                value=0,
                                key=f"{editor_key}__qa_qty_{pid}__{st.session_state[qa_nonce_key]}",
                                label_visibility="collapsed",
                            )
                            submitted = st.form_submit_button(
                                "➕ Sumar" if in_order else "➕ Añadir",
                                use_container_width=True,
                                type="primary" if not in_order else "secondary",
                            )

                        if submitted:
                            _add_product_to_df(pid, qty_val)

