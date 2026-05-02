"""Shared UI helpers: theme injection, metric cards, headers."""
from __future__ import annotations

import streamlit as st

# Color palette (single source of truth)
COLORS = {
    "bg":         "#0e1117",
    "bg_alt":     "#161a25",
    "surface":    "#1c2333",
    "surface_2":  "#222a3d",
    "border":     "#2a3142",
    "border_hi":  "#3a4358",
    "text":       "#e6edf3",
    "text_dim":   "#a6adbb",
    "text_muted": "#8b949e",
    "primary":    "#4f9eff",
    "violet":     "#9b6dff",
    "green":      "#3fb950",
    "red":        "#f85149",
    "amber":      "#ffa726",
    "pink":       "#ff6b9d",
}

THEME_CSS = f"""
<style>
    /* ============ Global background ============ */
    html, body, .stApp,
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background: {COLORS['bg']} !important;
        color: {COLORS['text']} !important;
    }}
    .stApp {{
        background: linear-gradient(135deg, {COLORS['bg']} 0%, #131826 100%) !important;
    }}
    [data-testid="stHeader"] {{
        background: rgba(14,17,23,0.85) !important;
        backdrop-filter: blur(8px);
        border-bottom: 1px solid {COLORS['border']};
    }}

    .stApp, .stApp p, .stApp span, .stApp label, .stApp div {{
        color: {COLORS['text']};
    }}
    .stApp small, [data-testid="stCaptionContainer"] {{
        color: {COLORS['text_muted']} !important;
    }}

    /* ============ Sidebar ============ */
    section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {{
        background: linear-gradient(180deg, {COLORS['bg_alt']} 0%, {COLORS['bg']} 100%) !important;
        border-right: 1px solid {COLORS['border']};
    }}
    section[data-testid="stSidebar"] * {{ color: {COLORS['text']} !important; }}
    section[data-testid="stSidebar"] a {{
        color: {COLORS['text_dim']} !important;
        font-weight: 500;
        padding: 0.45rem 0.8rem !important;
        border-radius: 8px;
        transition: all 0.15s ease;
    }}
    section[data-testid="stSidebar"] a:hover {{
        background: {COLORS['surface']} !important;
        color: {COLORS['primary']} !important;
    }}
    section[data-testid="stSidebar"] a[aria-current="page"] {{
        background: rgba(79,158,255,0.15) !important;
        color: {COLORS['primary']} !important;
        border-left: 3px solid {COLORS['primary']};
    }}

    /* ============ Headers ============ */
    h1 {{
        background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['violet']} 50%, {COLORS['pink']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }}
    h2, h3, h4, h5, h6 {{
        color: {COLORS['text']} !important;
        font-weight: 700 !important;
    }}
    h2 {{
        border-bottom: 1px solid {COLORS['border']};
        padding-bottom: 0.5rem;
        margin-top: 1.5rem !important;
    }}

    /* ============ Metric cards ============ */
    [data-testid="stMetric"] {{
        background: linear-gradient(135deg, {COLORS['surface']} 0%, {COLORS['bg_alt']} 100%);
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }}
    [data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        border-color: {COLORS['primary']};
    }}
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {{
        color: {COLORS['text_muted']} !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600 !important;
    }}
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {{
        color: {COLORS['text']} !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
    }}
    [data-testid="stMetricDelta"] * {{ font-weight: 600 !important; }}

    /* ============ Buttons ============ */
    .stButton > button, .stDownloadButton > button {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, #6c5ce7 100%);
        color: #ffffff !important;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.5rem;
        font-weight: 600;
        transition: all 0.15s ease;
        box-shadow: 0 2px 8px rgba(79,158,255,0.3);
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(79,158,255,0.5);
        filter: brightness(1.1);
        color: #ffffff !important;
    }}
    .stButton > button:active {{ transform: translateY(0); }}
    .stButton > button[kind="secondary"] {{
        background: {COLORS['surface']};
        color: {COLORS['text']} !important;
        border: 1px solid {COLORS['border']};
        box-shadow: none;
    }}

    /* ============ Tabs ============ */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background: transparent;
        border-bottom: 1px solid {COLORS['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-bottom: none;
        border-radius: 8px 8px 0 0;
        padding: 0.55rem 1.3rem;
        color: {COLORS['text_dim']} !important;
        font-weight: 600;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {COLORS['text']} !important;
        background: {COLORS['surface_2']};
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {COLORS['surface_2']} 0%, {COLORS['surface']} 100%) !important;
        color: {COLORS['primary']} !important;
        border-color: {COLORS['primary']} !important;
    }}
    .stTabs [data-baseweb="tab-panel"] {{ padding-top: 1rem; }}

    /* ============ Form labels ============ */
    .stSelectbox label, .stTextInput label, .stNumberInput label,
    .stDateInput label, .stTextArea label, .stMultiSelect label,
    .stRadio label, .stCheckbox label, .stSlider label {{
        color: {COLORS['text']} !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }}

    /* Form inputs */
    .stSelectbox > div > div, .stTextInput > div > div,
    .stNumberInput > div > div, .stDateInput > div > div,
    .stTextArea textarea, .stMultiSelect > div > div {{
        background: {COLORS['surface']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 8px !important;
        color: {COLORS['text']} !important;
    }}
    .stSelectbox > div > div:hover, .stTextInput > div > div:hover,
    .stNumberInput > div > div:hover, .stDateInput > div > div:hover,
    .stMultiSelect > div > div:hover {{
        border-color: {COLORS['border_hi']} !important;
    }}
    .stTextInput input, .stNumberInput input, .stDateInput input,
    .stTextArea textarea, .stSelectbox div[data-baseweb="select"] * {{
        color: {COLORS['text']} !important;
    }}
    .stSelectbox div[data-baseweb="select"] svg,
    .stMultiSelect div[data-baseweb="select"] svg {{
        fill: {COLORS['text_dim']} !important;
    }}

    /* Select dropdown menu (BaseWeb popover) */
    div[data-baseweb="popover"], div[data-baseweb="popover"] ul {{
        background: {COLORS['surface']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="popover"] li {{
        color: {COLORS['text']} !important;
        background: transparent !important;
    }}
    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="popover"] li[aria-selected="true"] {{
        background: rgba(79,158,255,0.15) !important;
        color: {COLORS['primary']} !important;
    }}

    /* Number input +/- buttons */
    .stNumberInput button {{
        background: {COLORS['surface_2']} !important;
        color: {COLORS['text']} !important;
        border-color: {COLORS['border']} !important;
    }}

    /* Slider */
    .stSlider [data-baseweb="slider"] [role="slider"] {{
        background: {COLORS['primary']} !important;
        border-color: {COLORS['primary']} !important;
    }}

    /* ============ Alerts / status / expander ============ */
    .stAlert {{
        border-radius: 10px;
        border-left-width: 4px;
        background: {COLORS['surface']} !important;
        color: {COLORS['text']} !important;
    }}
    .stAlert * {{ color: {COLORS['text']} !important; }}

    [data-testid="stStatusWidget"], [data-testid="stExpander"] {{
        background: {COLORS['surface']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 10px !important;
    }}
    [data-testid="stExpander"] summary {{
        color: {COLORS['text']} !important;
        font-weight: 600;
    }}

    /* ============ Tables / DataFrames ============ */
    [data-testid="stDataFrame"], [data-testid="stTable"] {{
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        overflow: hidden;
    }}
    [data-testid="stDataFrame"] * {{ color: {COLORS['text']}; }}

    /* ============ Code blocks ============ */
    .stCodeBlock, pre, code {{
        background: {COLORS['bg_alt']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 8px !important;
        color: {COLORS['text']} !important;
    }}

    /* ============ Markdown links ============ */
    .stMarkdown a {{
        color: {COLORS['primary']} !important;
        text-decoration: none;
        font-weight: 500;
    }}
    .stMarkdown a:hover {{
        text-decoration: underline;
        color: {COLORS['violet']} !important;
    }}

    /* ============ Custom utility classes ============ */
    .hero-card {{
        background: linear-gradient(135deg, rgba(79,158,255,0.08) 0%, rgba(155,109,255,0.08) 100%);
        border: 1px solid {COLORS['border']};
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    }}
    .hero-title {{
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['violet']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .hero-sub {{
        color: {COLORS['text_dim']};
        font-size: 1.05rem;
        line-height: 1.6;
    }}
    .badge {{
        display: inline-block;
        padding: 0.28rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.45rem;
        margin-bottom: 0.3rem;
    }}
    .badge-green {{ background: rgba(63,185,80,0.15);  color: #5dd97a !important; border: 1px solid rgba(63,185,80,0.45); }}
    .badge-blue  {{ background: rgba(79,158,255,0.15); color: #79b8ff !important; border: 1px solid rgba(79,158,255,0.45); }}
    .badge-red   {{ background: rgba(248,81,73,0.15);  color: #ff8983 !important; border: 1px solid rgba(248,81,73,0.45); }}
    .badge-amber {{ background: rgba(255,167,38,0.15); color: #ffc26b !important; border: 1px solid rgba(255,167,38,0.45); }}

    .feature-card {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
        transition: all 0.2s ease;
    }}
    .feature-card:hover {{
        border-color: {COLORS['primary']};
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(79,158,255,0.15);
    }}
    .feature-icon  {{ font-size: 2rem; margin-bottom: 0.6rem; }}
    .feature-title {{ color: {COLORS['text']}; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.35rem; }}
    .feature-desc  {{ color: {COLORS['text_dim']}; font-size: 0.9rem; line-height: 1.5; }}

    .ai-response {{
        background: linear-gradient(135deg, rgba(155,109,255,0.06) 0%, rgba(79,158,255,0.06) 100%);
        border-left: 3px solid {COLORS['violet']};
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        color: {COLORS['text']};
        line-height: 1.65;
        margin-top: 0.8rem;
        white-space: pre-wrap;
    }}

    /* Smooth fade-in to mask micro-flash on rerun */
    .block-container {{
        padding-top: 2rem !important;
        animation: fadeIn 200ms ease-in;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to   {{ opacity: 1; }}
    }}
</style>
"""


def apply_theme() -> None:
    """Inject the global dark theme CSS. Call once at top of every page."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", icon: str = "") -> None:
    """Render a styled page header card."""
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">{icon} {title}</div>
            <div class="hero-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(label: str, kind: str = "blue") -> str:
    """Return HTML for a colored badge."""
    cls = {"green": "badge-green", "blue": "badge-blue",
           "red": "badge-red", "amber": "badge-amber"}.get(kind, "badge-blue")
    return f'<span class="badge {cls}">{label}</span>'


def feature_card(icon: str, title: str, desc: str) -> str:
    return (
        f'<div class="feature-card">'
        f'<div class="feature-icon">{icon}</div>'
        f'<div class="feature-title">{title}</div>'
        f'<div class="feature-desc">{desc}</div>'
        f'</div>'
    )


def ai_response(text: str) -> None:
    st.markdown(f'<div class="ai-response">{text}</div>', unsafe_allow_html=True)
