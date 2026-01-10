import streamlit as st
import streamlit.components.v1 as components


def style_button(
    widget_label,
    font_color=None,
    background_color=None,
    width=None,
    height=None,
    padding_x=None,
    padding_y=None,
    font_size=None,
    border_radius=None,
    hover_font_color=None,
    hover_background_color=None,
    active_font_color=None,
    active_background_color=None,
    border=None,
    box_shadow=None,
):
    # Prebuild JS fragments so we don't send 'None' to JS
    base_font_color = font_color or ""
    base_bg_color = background_color or ""
    base_font_size = font_size or ""
    base_width = width or ""
    base_height = height or ""
    base_radius = border_radius or ""
    base_pad_x = padding_x or ""
    base_pad_y = padding_y or ""
    base_border = border or ""
    base_shadow = box_shadow or ""

    hover_font = hover_font_color or base_font_color
    hover_bg = hover_background_color or base_bg_color
    active_font = active_font_color or hover_font
    active_bg = active_background_color or hover_bg

    html_code = f"""
    <script>
    (function() {{
        const parentDoc = window.parent.document;
        const buttons = parentDoc.querySelectorAll('button');

        for (let i = 0; i < buttons.length; i++) {{
            const btn = buttons[i];
            if (btn.innerText.trim() === "{widget_label}") {{

                // --- Base styles ---
                if ("{base_font_color}") btn.style.setProperty('color', "{base_font_color}", 'important');
                if ("{base_bg_color}") btn.style.setProperty('background', "{base_bg_color}", 'important');
                if ("{base_width}") btn.style.setProperty('width', "{base_width}", 'important');
                if ("{base_height}") btn.style.setProperty('height', "{base_height}", 'important');
                if ("{base_radius}") btn.style.setProperty('border-radius', "{base_radius}", 'important');
                if ("{base_border}") btn.style.setProperty('border', "{base_border}", 'important');
                if ("{base_shadow}") btn.style.setProperty('box-shadow', "{base_shadow}", 'important');

                if ("{base_pad_x}") {{
                    btn.style.setProperty('padding-left', "{base_pad_x}");
                    btn.style.setProperty('padding-right', "{base_pad_x}");
                }}
                if ("{base_pad_y}") {{
                    btn.style.setProperty('padding-top', "{base_pad_y}");
                    btn.style.setProperty('padding-bottom', "{base_pad_y}");
                }}

                // FONT SIZE: apply to button AND inner span(s)
                if ("{base_font_size}") {{
                    btn.style.setProperty('font-size', "{base_font_size}", 'important');
                    const spans = btn.querySelectorAll('span, p, div');
                    spans.forEach(el => {{
                        el.style.setProperty('font-size', "{base_font_size}", 'important');
                    }});
                }}

                const origColor = btn.style.color;
                const origBg = btn.style.background;

                // --- Hover ---
                btn.addEventListener('mouseenter', function() {{
                    if ("{hover_font}") btn.style.setProperty('color', "{hover_font}", 'important');
                    if ("{hover_bg}") btn.style.setProperty('background', "{hover_bg}", 'important');
                }});

                btn.addEventListener('mouseleave', function() {{
                    btn.style.setProperty('color', origColor, 'important');
                    btn.style.setProperty('background', origBg, 'important');
                }});

                // --- Active (pressed) ---
                btn.addEventListener('mousedown', function() {{
                    if ("{active_font}") btn.style.setProperty('color', "{active_font}", 'important');
                    if ("{active_bg}") btn.style.setProperty('background', "{active_bg}", 'important');
                }});

                btn.addEventListener('mouseup', function() {{
                    const rect = btn.getBoundingClientRect();
                    const isInside =
                        window.event.clientX >= rect.left &&
                        window.event.clientX <= rect.right &&
                        window.event.clientY >= rect.top &&
                        window.event.clientY <= rect.bottom;

                    if (isInside) {{
                        if ("{hover_font}") btn.style.setProperty('color', "{hover_font}", 'important');
                        if ("{hover_bg}") btn.style.setProperty('background', "{hover_bg}", 'important');
                    }} else {{
                        btn.style.setProperty('color', origColor, 'important');
                        btn.style.setProperty('background', origBg, 'important');
                    }}
                }});
            }}
        }}
    }})();
    </script>
    """

    components.html(html_code, height=0, width=0)