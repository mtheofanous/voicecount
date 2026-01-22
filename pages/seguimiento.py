import streamlit as st
from core.public_links import verify_link, norm_provider, norm_role
from features.seguimiento.seguimiento import seguimiento_app


def qp(name: str):
    v = st.query_params.get(name)
    if not v:
        return None
    return v if isinstance(v, str) else v[0]


def main():
    # --- read URL params ---
    order_id = qp("order_id")
    provider = qp("provider")
    role = qp("role")

    # token can be called token OR sig in the URL
    token = qp("token") or qp("sig")

    if not all([order_id, provider, role, token]):
        st.error("❌ Invalid or incomplete link.")
        st.stop()

    try:
        order_id = int(order_id)
    except ValueError:
        st.error("❌ Invalid order_id.")
        st.stop()

    provider_n = norm_provider(provider)
    role_n = norm_role(role)

    # --- IMPORTANT PART ---
    # verify_link ONLY accepts sig=
    is_valid = verify_link(
        order_id=order_id,
        provider_name=provider_n,
        role=role_n,
        sig=token,        # ✅ sig, NOT token
    )

    if not is_valid:
        st.error("🔒 Invalid or expired link.")
        st.stop()

    # --- render public page ---
    seguimiento_app(
        order_id=order_id,
        provider_name=provider_n,
        role=role_n,
        token=token,      # seguimiento_app DOES accept token
    )


if __name__ == "__main__":
    main()
