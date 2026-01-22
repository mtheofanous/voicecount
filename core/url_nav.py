# core/url_nav.py
from __future__ import annotations

from typing import Dict, Optional
import streamlit as st


def get_query_params() -> Dict[str, str]:
    """Return query params as flat dict[str,str] (first value only)."""
    if hasattr(st, "query_params"):
        qp = dict(st.query_params)
        out: Dict[str, str] = {}
        for k, v in qp.items():
            if isinstance(v, (list, tuple)):
                out[k] = str(v[0]) if v else ""
            else:
                out[k] = str(v)
        return out

    qp_old = st.experimental_get_query_params()
    return {k: (v[0] if isinstance(v, list) and v else "") for k, v in qp_old.items()}


def set_query_params(**kwargs: str) -> None:
    """Set query params (Streamlit old/new compatible)."""
    if hasattr(st, "query_params"):
        for k, v in kwargs.items():
            vv = "" if v is None else str(v)
            if vv.strip() == "":
                try:
                    del st.query_params[k]
                except Exception:
                    pass
            else:
                st.query_params[k] = vv
        return

    clean = {k: str(v) for k, v in kwargs.items() if v is not None and str(v).strip() != ""}
    st.experimental_set_query_params(**clean)


def qp_str(name: str, default: str = "") -> str:
    v = (get_query_params().get(name) or "").strip()
    return v if v else default


def qp_int(name: str) -> Optional[int]:
    raw = (get_query_params().get(name) or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None
