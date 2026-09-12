"""
Professional CSS styling, animations, and custom components for the Streamlit dashboard.
"""
import streamlit as st

# Professional color palette - Modern & Bright
COLORS = {
    "primary": "#0099FF",      # Bright electric blue
    "secondary": "#00D4FF",    # Cyan/turquoise
    "accent": "#FF6B35",       # Vibrant coral-orange
    "light_bg": "#F8FBFF",     # Very light blue-white
    "surface": "#E8F4FF",      # Light sky blue
    "success": "#00B894",      # Fresh green
    "warning": "#FF7675",      # Coral red
    "text": "#0A1628",         # Dark blue-black
    "text_light": "#4A5568",   # Medium gray
    "border": "#B3D9FF",       # Light blue border
}

def apply_professional_styling():
    """Apply custom CSS styling to the entire dashboard with animations."""
    css = f"""
    <style>
        /* ==================== GLOBAL STYLING ==================== */
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        html, body, [data-testid="stAppViewContainer"] {{
            background: linear-gradient(135deg, {COLORS["light_bg"]} 0%, #F8FAFF 100%);
            color: {COLORS["text"]};
        }}

        /* Font imports */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

        body, [data-testid="stAppViewContainer"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-weight: 400;
        }}

        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            letter-spacing: -0.5px;
        }}

        h1 {{
            font-size: 2.5rem;
            background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
            animation: fadeInDown 0.6s ease-out;
        }}

        h2 {{
            color: {COLORS["primary"]};
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            animation: slideInLeft 0.5s ease-out;
        }}

        h3 {{
            color: {COLORS["primary"]};
            margin-top: 1.2rem;
            margin-bottom: 0.8rem;
            animation: slideInLeft 0.4s ease-out;
        }}

        /* ==================== ANIMATIONS ==================== */
        @keyframes fadeInDown {{
            from {{
                opacity: 0;
                transform: translateY(-20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @keyframes slideInLeft {{
            from {{
                opacity: 0;
                transform: translateX(-20px);
            }}
            to {{
                opacity: 1;
                transform: translateX(0);
            }}
        }}

        @keyframes slideInRight {{
            from {{
                opacity: 0;
                transform: translateX(20px);
            }}
            to {{
                opacity: 1;
                transform: translateX(0);
            }}
        }}

        @keyframes fadeIn {{
            from {{
                opacity: 0;
            }}
            to {{
                opacity: 1;
            }}
        }}

        @keyframes scaleIn {{
            from {{
                opacity: 0;
                transform: scale(0.95);
            }}
            to {{
                opacity: 1;
                transform: scale(1);
            }}
        }}

        @keyframes pulse {{
            0%, 100% {{
                opacity: 1;
            }}
            50% {{
                opacity: 0.8;
            }}
        }}

        /* ==================== SIDEBAR STYLING ==================== */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #FFFFFF 0%, {COLORS["surface"]} 100%);
            color: {COLORS["text"]};
            border-right: 2px solid {COLORS["border"]};
        }}

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
            color: {COLORS["text"]};
        }}

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: {COLORS["primary"]};
            background: none;
            -webkit-text-fill-color: unset;
        }}

        [data-testid="stSidebar"] label {{
            color: {COLORS["text"]};
            font-weight: 500;
        }}

        [data-testid="stSidebar"] [role="option"] {{
            background-color: {COLORS["primary"]};
            color: white;
            border-radius: 6px;
        }}

        /* ==================== BUTTON STYLING ==================== */
        button[kind="primary"] {{
            background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%) !important;
            color: white !important;
            border: none !important;
            padding: 0.75rem 2rem !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 12px rgba({int("0F", 16)}, {int("2A", 16)}, {int("3D", 16)}, 0.2) !important;
            animation: fadeIn 0.5s ease-out !important;
        }}

        button[kind="primary"]:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba({int("0F", 16)}, {int("2A", 16)}, {int("3D", 16)}, 0.3) !important;
        }}

        button[kind="primary"]:active {{
            transform: translateY(0) !important;
        }}

        button[kind="secondary"], button[kind=""] {{
            background-color: {COLORS["surface"]} !important;
            color: {COLORS["primary"]} !important;
            border: 2px solid {COLORS["border"]} !important;
            padding: 0.6rem 1.5rem !important;
            font-weight: 500 !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }}

        button[kind="secondary"]:hover, button[kind=""]:hover {{
            background-color: {COLORS["primary"]} !important;
            color: white !important;
            border-color: {COLORS["primary"]} !important;
            transform: translateY(-2px) !important;
        }}

        /* ==================== INPUT STYLING ==================== */
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] select,
        [data-testid="stTextArea"] textarea {{
            border-radius: 8px !important;
            border: 2px solid {COLORS["border"]} !important;
            padding: 0.75rem 1rem !important;
            font-family: 'Inter', sans-serif !important;
            transition: all 0.3s ease !important;
        }}

        [data-testid="stTextInput"] input:focus,
        [data-testid="stNumberInput"] input:focus,
        [data-testid="stSelectbox"] select:focus,
        [data-testid="stTextArea"] textarea:focus {{
            border-color: {COLORS["primary"]} !important;
            box-shadow: 0 0 0 3px rgba({int("0F", 16)}, {int("2A", 16)}, {int("3D", 16)}, 0.1) !important;
        }}

        /* ==================== METRIC STYLING ==================== */
        [data-testid="metric-container"] {{
            background: linear-gradient(135deg, white 0%, {COLORS["surface"]} 100%);
            padding: 1.5rem !important;
            border-radius: 12px !important;
            border: 1px solid {COLORS["border"]} !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
            transition: all 0.3s ease !important;
            animation: scaleIn 0.5s ease-out;
        }}

        [data-testid="metric-container"]:hover {{
            transform: translateY(-4px) !important;
            box-shadow: 0 8px 24px rgba({int("0F", 16)}, {int("2A", 16)}, {int("3D", 16)}, 0.15) !important;
        }}

        [data-testid="metric-container"] [data-testid="stMetricLabel"] {{
            color: {COLORS["text_light"]} !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        [data-testid="metric-container"] [data-testid="stMetricValue"] {{
            color: {COLORS["primary"]} !important;
            font-weight: 700 !important;
            font-size: 1.8rem !important;
        }}

        [data-testid="metric-container"] [data-testid="stMetricDelta"] {{
            color: {COLORS["accent"]} !important;
            font-weight: 600 !important;
        }}

        /* ==================== TAB STYLING ==================== */
        [data-testid="stTabs"] [role="tab"] {{
            background-color: transparent !important;
            border-bottom: 3px solid transparent !important;
            color: {COLORS["text_light"]} !important;
            font-weight: 600 !important;
            padding: 1rem !important;
            transition: all 0.3s ease !important;
        }}

        [data-testid="stTabs"] [role="tab"]:hover {{
            color: {COLORS["primary"]} !important;
            border-bottom-color: {COLORS["accent"]} !important;
        }}

        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
            color: {COLORS["primary"]} !important;
            border-bottom-color: {COLORS["accent"]} !important;
            background-color: {COLORS["light_bg"]} !important;
            border-radius: 8px 8px 0 0 !important;
        }}

        /* ==================== DIVIDER STYLING ==================== */
        [data-testid="stHorizontalBlock"] hr {{
            border: none !important;
            height: 2px !important;
            background: linear-gradient(90deg, transparent, {COLORS["border"]}, transparent) !important;
            margin: 1.5rem 0 !important;
        }}

        /* ==================== EXPANDER STYLING ==================== */
        [data-testid="stExpander"] button {{
            background-color: {COLORS["surface"]} !important;
            border-radius: 8px !important;
            color: {COLORS["primary"]} !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }}

        [data-testid="stExpander"] button:hover {{
            background-color: {COLORS["primary"]} !important;
            color: white !important;
        }}

        /* ==================== SLIDER STYLING ==================== */
        [data-testid="stSlider"] {{
            animation: fadeIn 0.5s ease-out;
        }}

        /* ==================== DATA TABLE STYLING ==================== */
        [data-testid="stDataFrame"] {{
            border-radius: 12px !important;
            overflow: hidden !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
        }}

        [data-testid="stDataFrame"] thead {{
            background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%) !important;
        }}

        [data-testid="stDataFrame"] thead th {{
            color: white !important;
            font-weight: 600 !important;
            border: none !important;
        }}

        [data-testid="stDataFrame"] tbody td {{
            border-color: {COLORS["border"]} !important;
            padding: 0.8rem !important;
        }}

        [data-testid="stDataFrame"] tbody tr:hover {{
            background-color: {COLORS["light_bg"]} !important;
        }}

        /* ==================== ALERT STYLING ==================== */
        [data-testid="stAlert"] {{
            border-radius: 12px !important;
            border-left: 4px solid !important;
            padding: 1rem !important;
            animation: slideInRight 0.4s ease-out;
        }}

        [data-testid="stAlert"][kind="info"] {{
            background-color: rgba({int("0F", 16)}, {int("2A", 16)}, {int("3D", 16)}, 0.1) !important;
            border-color: {COLORS["primary"]} !important;
            color: {COLORS["primary"]} !important;
        }}

        [data-testid="stAlert"][kind="success"] {{
            background-color: rgba({int("27", 16)}, {int("AE", 16)}, {int("60", 16)}, 0.1) !important;
            border-color: {COLORS["success"]} !important;
            color: {COLORS["success"]} !important;
        }}

        [data-testid="stAlert"][kind="warning"] {{
            background-color: rgba({int("E7", 16)}, {int("4C", 16)}, {int("3C", 16)}, 0.1) !important;
            border-color: {COLORS["warning"]} !important;
            color: {COLORS["warning"]} !important;
        }}

        /* ==================== SPINNER STYLING ==================== */
        [data-testid="stSpinner"] {{
            animation: spin 1s linear infinite;
        }}

        @keyframes spin {{
            from {{ transform: rotate(0deg); }}
            to {{ transform: rotate(360deg); }}
        }}

        /* ==================== CAPTION & TEXT ==================== */
        [data-testid="stCaptionContainer"] {{
            color: {COLORS["text_light"]} !important;
            font-size: 0.9rem !important;
            margin-top: 0.5rem !important;
        }}

        [data-testid="stMarkdownContainer"] p {{
            line-height: 1.6;
            color: {COLORS["text"]};
        }}

        /* ==================== CHECKBOX STYLING ==================== */
        [data-testid="stCheckbox"] {{
            animation: fadeIn 0.5s ease-out;
        }}

        [data-testid="stCheckbox"] label {{
            font-weight: 500;
            color: {COLORS["text"]};
        }}

        /* ==================== COLUMN SPACING ==================== */
        [data-testid="stHorizontalBlock"] > div {{
            gap: 1.5rem !important;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def custom_metric(label, value, delta=None, icon="📊", help_text=None):
    """Render a custom metric with better styling."""
    col = st.container()
    with col:
        html = f"""
        <div style="
            background: linear-gradient(135deg, white 0%, {COLORS['surface']} 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid {COLORS['border']};
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
            animation: scaleIn 0.5s ease-out;
        " class="metric-box">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <p style="
                        color: {COLORS['text_light']};
                        font-size: 0.85rem;
                        font-weight: 600;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                        margin: 0;
                    ">{label}</p>
                    <p style="
                        color: {COLORS['primary']};
                        font-size: 1.8rem;
                        font-weight: 700;
                        margin: 0.5rem 0 0;
                        font-family: 'Poppins', sans-serif;
                    ">{value}</p>
                    {f'<p style="color: {COLORS["accent"]}; font-weight: 600; margin: 0.3rem 0 0;">{delta}</p>' if delta else ''}
                </div>
                <div style="font-size: 2rem;">{icon}</div>
            </div>
            {f'<p style="color: {COLORS["text_light"]}; font-size: 0.8rem; margin: 0.8rem 0 0;">{help_text}</p>' if help_text else ''}
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)


def render_header(title, subtitle="", icon="🚀"):
    """Render a professional header with animation."""
    html = f"""
    <div style="animation: fadeInDown 0.6s ease-out;">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;">
            <span style="font-size: 2.5rem;">{icon}</span>
            <h1 style="margin: 0; background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                {title}
            </h1>
        </div>
        {f'<p style="color: {COLORS["text_light"]}; font-size: 1rem; margin: 0; margin-left: 3.5rem;">{subtitle}</p>' if subtitle else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_info_box(text, icon="ℹ️", box_type="info"):
    """Render a professional info box with animation."""
    colors_map = {
        "info": (COLORS["primary"], "rgba(15, 42, 61, 0.1)"),
        "success": (COLORS["success"], "rgba(39, 174, 96, 0.1)"),
        "warning": (COLORS["warning"], "rgba(231, 76, 60, 0.1)"),
    }
    color, bg_color = colors_map.get(box_type, colors_map["info"])

    html = f"""
    <div style="
        background-color: {bg_color};
        border-left: 4px solid {color};
        padding: 1rem;
        border-radius: 8px;
        animation: slideInRight 0.4s ease-out;
        display: flex;
        gap: 1rem;
        align-items: flex-start;
    ">
        <span style="font-size: 1.5rem; flex-shrink: 0;">{icon}</span>
        <p style="color: {color}; margin: 0; font-weight: 500;">{text}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def create_gradient_button(label: str, key: str, color_start: str = None, color_end: str = None):
    """Create a gradient button with styling."""
    if color_start is None:
        color_start = COLORS["primary"]
    if color_end is None:
        color_end = COLORS["secondary"]

    return st.button(
        label=label,
        key=key,
        use_container_width=True,
        type="primary"
    )
