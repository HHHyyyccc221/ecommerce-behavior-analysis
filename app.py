import os
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from preprocess import LAMBDA_DECAY, SESSION_GAP_SEC, clean_events, load_data

import os
import requests

def ensure_data_files(data_dir: str = ".") -> None:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    HF_REPO  = os.environ.get("HF_REPO", "")
    
    files_to_download = [
        "events.csv",
        "item_properties_part1.csv",
        "item_properties_part2.csv",
    ]
    
    for filename in files_to_download:
        local_path = os.path.join(data_dir, filename)
        if os.path.exists(local_path):
            continue
        if not HF_TOKEN or not HF_REPO:
            raise FileNotFoundError(f"{filename} not found and HF_TOKEN/HF_REPO not set.")
        url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{filename}"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        import streamlit as st
        with st.spinner(f"Downloading {filename} (first run only)..."):
            r = requests.get(url, headers=headers, stream=True)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)


# ──────────────────────────────────────────────────────────────────────────────
# 0. Global CSS Custom Theme Injection (Retro Ledger Style)
# ──────────────────────────────────────────────────────────────────────────────
def apply_custom_theme():
    st.markdown(
        """
        <style>
        /* Import Google Fonts: Lora (serif for headers), DM Sans (sans-serif for body) */
        @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@400;500;700&display=swap');

        /* Squeeze main container to fit everything on the first screen */
        .block-container {
            padding-top: 3.2rem !important;
            padding-bottom: 1rem !important;
        }

        /* Base App Colors */
        .stApp {
            background-color: #F4F1EA !important; /* Cream Background */
        }

        /* Basic Typography */
        html, body, [class*="css"], .stMarkdown p, .stMarkdown li {
            font-family: 'DM Sans', sans-serif !important;
            color: #1A1A1A !important;
        }

        /* Unified Header Styles - Compacted to delete redundant empty lines */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Lora', serif !important;
            color: #1A1A1A !important;
            font-weight: 700 !important;
        }
        h1 { font-size: 2.2rem !important; margin-top: 0rem !important; margin-bottom: 0.5rem !important; border-bottom: 2px solid #1A1A1A; padding-bottom: 0.3rem !important; line-height: 1.2 !important; }
        h2 { font-size: 1.8rem !important; margin-top: 0.5rem !important; margin-bottom: 0.5rem !important; line-height: 1.2 !important; }
        h3 { font-size: 1.4rem !important; margin-top: 0.5rem !important; margin-bottom: 0.5rem !important; line-height: 1.2 !important; }

        /* Compress Streamlit vertical blocks to remove unwanted blank space */
        [data-testid="stVerticalBlock"] {
            gap: 0.5rem !important;
        }
        
        /* Fine-tune column flex layout for alignment */
        [data-testid="stHorizontalBlock"] {
            align-items: flex-start !important;
        }

        /* Sidebar Design */
        [data-testid="stSidebar"] {
            background-color: #2C5545 !important; /* Dark Forest Green */
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #E2B659 !important; /* Retro Gold for sidebar headers */
            border-bottom: none;
        }
        [data-testid="stSidebar"] *, [data-testid="stSidebar"] p {
            color: #F4F1EA !important; /* Cream for regular sidebar text */
        }
        
        /* Fix visibility of inputs inside the sidebar */
        [data-testid="stSidebar"] .stTextInput input,
        [data-testid="stSidebar"] .stNumberInput input,
        [data-testid="stSidebar"] select,
        [data-testid="stSidebar"] [data-baseweb="base-input"] input {
            background-color: #243D30 !important;
            color: #E8DFC8 !important;
            -webkit-text-fill-color: #E8DFC8 !important;
            border: 1px solid #3A5C48 !important;
            border-radius: 0px !important;
            font-weight: 500 !important;
        }
        [data-testid="stSidebar"] label {
            color: #F4F1EA !important; /* Ensure labels are not overridden */
        }

        /* Fix selectbox text color inside sidebar */
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background-color: #243D30 !important;
            border: 1px solid #3A5C48 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] [class*="placeholder"],
        [data-testid="stSidebar"] [data-baseweb="select"] [class*="singleValue"],
        [data-testid="stSidebar"] [data-baseweb="select"] [class*="option"],
        [data-testid="stSidebar"] [data-baseweb="select"] span,
        [data-testid="stSidebar"] [data-baseweb="select"] div,
        [data-testid="stSidebar"] [data-baseweb="select"] p {
            color: #E8DFC8 !important;
            -webkit-text-fill-color: #E8DFC8 !important;
        }
        /* dropdown menu list */
        [data-testid="stSidebar"] [data-baseweb="popover"] li,
        [data-testid="stSidebar"] [data-baseweb="menu"] li,
        [data-testid="stSidebar"] [role="listbox"] li,
        [data-testid="stSidebar"] [role="option"] {
            background-color: #243D30 !important;
            color: #E8DFC8 !important;
        }
        [data-testid="stSidebar"] [role="option"]:hover {
            background-color: #2E5040 !important;
        }

        /* Button Design (High contrast to fix visibility issues) */
        div.stButton > button {
            background-color: #1A1A1A !important;
            color: #F4F1EA !important;
            border: 2px solid #1A1A1A !important;
            border-radius: 0px !important;
            padding: 0.5rem 1rem !important;
            font-weight: bold !important;
            font-family: 'DM Sans', sans-serif !important;
            box-shadow: 4px 4px 0px #E2B659 !important; /* Gold hard shadow */
            transition: all 0.2s ease-in-out !important;
        }
        div.stButton > button:hover {
            background-color: #E2B659 !important;
            color: #1A1A1A !important;
            box-shadow: 4px 4px 0px #1A1A1A !important; /* Black hard shadow */
        }
        div.stButton > button * {
            color: inherit !important;
        }

        /* Metric Cards */
        [data-testid="stMetric"] {
            background-color: #FFFFFF !important;
            border: 2px solid #1A1A1A !important;
            border-radius: 0px !important;
            padding: 1rem !important;
            box-shadow: 4px 4px 0px #1A1A1A !important;
        }
        [data-testid="stMetricValue"] {
            font-family: 'Lora', serif !important;
            color: #2C5545 !important;
        }

        /* Alerts/Info Boxes */
        .stAlert {
            background-color: #FFFFFF !important;
            border: 2px solid #1A1A1A !important;
            border-radius: 0px !important;
            box-shadow: 4px 4px 0px #1A1A1A !important;
            color: #1A1A1A !important;
        }
        .stAlert [data-testid="stMarkdownContainer"], .stAlert p {
            color: #1A1A1A !important;
        }

        /* DataFrame Tables */
        .stDataFrame {
            border: 2px solid #1A1A1A !important;
            border-radius: 0px !important;
            box-shadow: 4px 4px 0px #1A1A1A !important;
        }

        /* General Block / Recommendation Cards Container */
        [data-testid="stVerticalBlock"] > div.element-container > div.stContainer {
            background-color: #FFFFFF !important;
            border: 2px solid #1A1A1A !important;
            border-radius: 0px !important;
            box-shadow: 4px 4px 0px #1A1A1A !important;
            padding: 1rem !important;
        }

        /* ── Sidebar: Remove default blank lines from header/subheader ─── */
        /* header → h1 */
        [data-testid="stSidebar"] [data-testid="stHeadingWithActionElements"] h1,
        [data-testid="stSidebar"] h1 {
            margin-top: 0.6rem !important;
            margin-bottom: 0 !important;
            padding-bottom: 0.2rem !important;
            line-height: 1.2 !important;
        }
        /* subheader → h2 */
        [data-testid="stSidebar"] [data-testid="stHeadingWithActionElements"] h2,
        [data-testid="stSidebar"] h2 {
            margin-top: 0.5rem !important;
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
            line-height: 1.2 !important;
        }
        /* caption right below title */
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            margin-top: 0.05rem !important;
            margin-bottom: 0.2rem !important;
            padding-top: 0 !important;
        }
        /* number_input spacing */
        [data-testid="stSidebar"] [data-testid="stNumberInput"] {
            margin-top: 0 !important;
            margin-bottom: 0.25rem !important;
        }
        /* radio spacing & font size adjustments */
        [data-testid="stSidebar"] .stRadio label p {
            font-size: 1.15rem !important; /* 放大侧边栏菜单字号 */
            padding-top: 0.2rem !important;
            padding-bottom: 0.2rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] {
            margin-top: 0.05rem !important;
            margin-bottom: 0.1rem !important;
        }
        /* slider spacing */
        [data-testid="stSidebar"] [data-testid="stSlider"] {
            margin-top: 0 !important;
            margin-bottom: 0.2rem !important;
        }
        /* Remove default vertical gap added by Streamlit to each element-container */
        [data-testid="stSidebar"] .element-container {
            margin-bottom: 0 !important;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.1rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. Model & Preprocessing Load (Synchronized with Training Phase)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model_and_preprocess() -> Tuple[object, List[str], object]:
    """
    Load the trained champion model (LightGBM GBT) as well as the 
    feature_names and StandardScaler used during training.
    """
    model_paths = [
        "gbt_lgbm.pkl",
        os.path.join("saved_models", "gbt_lgbm.pkl"),
    ]
    model = None
    for path in model_paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                model = pickle.load(f)
            break
    if model is None:
        raise FileNotFoundError("Model file gbt_lgbm.pkl not found. Please ensure it is in the root directory or saved_models/.")

    cache_dir = "cache"
    feat_path = os.path.join(cache_dir, "feature_names.npy")
    scaler_path = os.path.join(cache_dir, "scaler.pkl")
    if not (os.path.exists(feat_path) and os.path.exists(scaler_path)):
        raise FileNotFoundError(
            "Feature cache files cache/feature_names.npy or cache/scaler.pkl not found.\n"
            "Please run the training script first (e.g., main.py / run_preprocessing_cached) to generate cache before starting the app."
        )

    feature_names = np.load(feat_path, allow_pickle=True).tolist()
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, feature_names, scaler


def build_feature_vector_from_inputs(
    inputs: Dict[str, float],
    feature_names: List[str],
    scaler,
) -> np.ndarray:
    """
    Convert raw inputs from the sidebar into the exact feature vector used during training.
    """
    sess_view_cnt = inputs["sess_view_cnt"]
    sess_cart_cnt = inputs["sess_cart_cnt"]
    sess_duration_sec = inputs["sess_duration_sec"]
    sess_unique_items = inputs["sess_unique_items"]

    sess_cart_view_ratio = sess_cart_cnt / (sess_view_cnt + 1.0)

    user_recency_sec = inputs["user_recency_hours"] * 3600.0
    user_freq_total = inputs["user_freq_total"]
    user_cart_freq = inputs["user_cart_freq"]

    recent_view_delta_sec = inputs["recent_view_hours"] * 3600.0
    recent_cart_delta_sec = inputs["recent_cart_hours"] * 3600.0
    user_decayed_view = sess_view_cnt * np.exp(-LAMBDA_DECAY * recent_view_delta_sec)
    user_decayed_cart = sess_cart_cnt * np.exp(-LAMBDA_DECAY * recent_cart_delta_sec)

    user_cat_breadth = inputs.get("user_cat_breadth", 0.0)
    user_cat_concentration = inputs.get("user_cat_concentration", 0.0)

    raw_feat = {
        "sess_view_cnt": sess_view_cnt,
        "sess_cart_cnt": sess_cart_cnt,
        "sess_duration_sec": sess_duration_sec,
        "sess_unique_items": sess_unique_items,
        "sess_cart_view_ratio": sess_cart_view_ratio,
        "user_recency_sec": user_recency_sec,
        "user_freq_total": user_freq_total,
        "user_cart_freq": user_cart_freq,
        "user_decayed_view": user_decayed_view,
        "user_decayed_cart": user_decayed_cart,
        "user_cat_breadth": user_cat_breadth,
        "user_cat_concentration": user_cat_concentration,
    }

    vector = np.array([[float(raw_feat.get(name, 0.0)) for name in feature_names]], dtype=float)
    vector_scaled = scaler.transform(vector)
    return vector_scaled


def predict_proba_single(model, X_scaled: np.ndarray) -> float:
    """Predict a single sample and return the probability of purchase intent (label 1)."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0, 1]
    else:
        if hasattr(model, "decision_function"):
            score = model.decision_function(X_scaled)[0]
            proba = 1.0 / (1.0 + np.exp(-score))
        else:
            proba = float(model.predict(X_scaled)[0])
    return float(proba)


def render_gauge(prob: float) -> None:
    """Draw a Plotly gauge chart matching the retro color palette."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100.0,
            number={"suffix": "%", "font": {"size": 32, "color": "#1A1A1A"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#1A1A1A"},
                "bar": {"color": "#1A1A1A"},
                "steps": [
                    {"range": [0, 40], "color": "#F4F1EA"},  # Cream
                    {"range": [40, 70], "color": "#E2B659"},  # Accent Gold
                    {"range": [70, 85], "color": "#59738A"},  # Slate Blue
                    {"range": [85, 100], "color": "#2C5545"}, # Forest Green
                ],
                "threshold": {
                    "line": {"color": "#D35A5A", "width": 4}, # Danger Red
                    "thickness": 0.75,
                    "value": prob * 100.0,
                },
            },
            title={"text": "Predicted Purchase Intent", "font": {"family": "Lora", "size": 20, "color": "#1A1A1A"}},
        )
    )
    # 调整边距和整体高度，使其在首屏完美显示
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        height=320, 
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'DM Sans, sans-serif', 'color': '#1A1A1A'}
    )
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Module 1 Helpers: Behavior Cache Load & Precision Marketing Matrix
# ──────────────────────────────────────────────────────────────────────────────

BEHAVIOR_CACHE_DIR = "behavior_cache"

@st.cache_resource(show_spinner=False)
def load_behavior_cache() -> dict:
    result: dict = {"session_behavior": None, "user_behavior": None}
    sp = os.path.join(BEHAVIOR_CACHE_DIR, "session_behavior.parquet")
    up = os.path.join(BEHAVIOR_CACHE_DIR, "user_behavior.parquet")
    if os.path.exists(sp):
        result["session_behavior"] = pd.read_parquet(sp)
    if os.path.exists(up):
        result["user_behavior"] = pd.read_parquet(up)
    return result

_BEHAVIOR_LABEL = {
    "focused":  "Intra-category Repeated Comparison",
    "explorer": "Cross-category Exploration",
    "normal":   "Normal Browsing",
}

_MARKETING_MATRIX: dict = {
    ("focused",  "high"):   "High Intent + Intra-category comparison: User is comparing prices. Suggest pushing **time-limited coupons immediately** to eliminate price concerns and close the deal.",
    ("focused",  "medium"): "Medium Intent + Intra-category comparison: User is hesitating among similar items. Suggest showing **positive reviews and sales rankings** to build trust, along with small discounts.",
    ("focused",  "low"):    "Low Intent + Intra-category comparison: Exploration phase. Suggest pushing **category best-sellers** to guide discovery without aggressive promotions.",
    ("explorer", "high"):   "High Intent + Cross-category exploration: Clear but broad needs. Suggest **search guidance + personalized recommendations** to help them lock onto target items quickly.",
    ("explorer", "medium"): "Medium Intent + Cross-category exploration: Broad exploration phase. Suggest showing **best-sellers + related category combos** to narrow down choices.",
    ("explorer", "low"):    "Low Intent + Cross-category exploration: Scattered interest. Suggest focusing on **brand exposure and content seeding**, reducing promotional intensity.",
    ("normal",   "high"):   "High Intent + Normal browsing: Strong purchase signal. Suggest **pushing coupons or limited-time discounts immediately** to drive orders.",
    ("normal",   "medium"): "Medium Intent + Normal browsing: Some interest. Suggest recommending **related items or user reviews** to further activate purchase intent.",
    ("normal",   "low"):    "Low Intent + Normal browsing: Suggest focusing on **brand exposure**, reducing aggressive promotional strategies.",
}

def _prob_level(prob: float) -> str:
    if prob >= 0.6: return "high"
    if prob >= 0.3: return "medium"
    return "low"

def render_behavior_marketing(prob: float, behavior_type: str) -> None:
    level  = _prob_level(prob)
    label  = _BEHAVIOR_LABEL.get(behavior_type, "Normal Browsing")
    advice = _MARKETING_MATRIX.get(
        (behavior_type, level),
        _MARKETING_MATRIX[("normal", level)],
    )
    # 删除多余空行，改用 flex 居中对齐，与左侧浑然一体
    st.markdown(
        """
        <div style='display: flex; align-items: center; justify-content: center; margin-top: 0rem; margin-bottom: 0.5rem;'>
            <h3 style='text-align: center; margin: 0; font-size: 1.35rem;'>Precision Marketing Suggestions</h3>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.info(advice)

def marketing_suggestion(prob: float) -> str:
    if prob >= 0.8: return "High purchase intent (>80%): Suggest issuing attractive coupons or limited-time discounts immediately to drive orders."
    if prob >= 0.5: return "Medium purchase intent (50%~80%): Recommend related items or moderate discounts to further activate purchase interest."
    if prob >= 0.3: return "Lower purchase intent (30%~50%): Suggest building trust through content seeding and showcasing reviews."
    return "Extremely low purchase intent (<30%): Focus on brand exposure and reduce aggressive promotional strategies."


# ──────────────────────────────────────────────────────────────────────────────
# Module 2: BG/NBD Customer Lifetime Value Analysis
# ──────────────────────────────────────────────────────────────────────────────

BGNBD_CACHE_DIR = "bgnbd_cache"

@st.cache_resource(show_spinner=False)
def load_bgnbd_cache() -> dict:
    import json as _json
    def _p(name): return os.path.join(BGNBD_CACHE_DIR, name)
    result: dict = {
        "events":         None,
        "rfm":            None,
        "bgf":            None,
        "heatmap_matrix": None,
        "heatmap_meta":   None,
        "p_alive":        None,
    }

    if os.path.exists(_p("events_cleaned.parquet")):
        result["events"] = pd.read_parquet(_p("events_cleaned.parquet"))
    if os.path.exists(_p("rfm.parquet")):
        result["rfm"] = pd.read_parquet(_p("rfm.parquet"))
    if os.path.exists(_p("bgf_params.json")):
        import json as _json2
        from lifetimes import BetaGeoFitter as _BGF
        with open(_p("bgf_params.json")) as f:
            _params = _json2.load(f)
        _bgf = _BGF(penalizer_coef=0.0)
        _bgf.params_ = _params
        result["bgf"] = _bgf
    if os.path.exists(_p("heatmap_matrix.npy")):
        result["heatmap_matrix"] = np.load(_p("heatmap_matrix.npy"))
    if os.path.exists(_p("heatmap_meta.json")):
        with open(_p("heatmap_meta.json")) as f:
            result["heatmap_meta"] = _json.load(f)
    if os.path.exists(_p("p_alive.parquet")):
        result["p_alive"] = pd.read_parquet(_p("p_alive.parquet"))

    return result


FUNNEL_CACHE_DIR = "funnel_cache"

@st.cache_resource(show_spinner=False)
def load_funnel_cache() -> dict:
    result: dict = {"cat_funnel": None, "item_funnel": None}
    cp = os.path.join(FUNNEL_CACHE_DIR, "cat_funnel.parquet")
    ip = os.path.join(FUNNEL_CACHE_DIR, "item_funnel.parquet")
    if os.path.exists(cp):
        result["cat_funnel"] = pd.read_parquet(cp)
    if os.path.exists(ip):
        result["item_funnel"] = pd.read_parquet(ip)
    return result

def _render_funnel_section() -> None:
    cache = load_funnel_cache()
    cat_df   = cache["cat_funnel"]
    item_df  = cache["item_funnel"]

    if cat_df is None:
        st.warning("Funnel cache not found, please run: `python prepare_funnel_cache.py` first.")
        return

    st.caption("Display the view → add-to-cart → purchase conversion rates for the top N categories by page views, revealing which categories are most effective at driving purchases.")

    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        top_n = st.slider("Display the top N categories", min_value=5, max_value=30, value=15, step=5)
    with col_ctrl2:
        metric = st.radio(
            "Conversion Rate Metric",
            options=["view_to_cart_rate", "cart_to_tx_rate", "view_to_tx_rate"],
            format_func=lambda x: {
                "view_to_cart_rate": "View → Cart Rate",
                "cart_to_tx_rate":   "Cart → Purchase Rate",
                "view_to_tx_rate":   "View → Purchase Total Rate",
            }.get(x, x),
            horizontal=True,
        )

    top_cats = cat_df.head(top_n).copy()
    top_cats = top_cats.sort_values(metric, ascending=True)
    top_cats["rate_pct"] = (top_cats[metric] * 100).round(2)
    top_cats["cat_label"] = "Category " + top_cats["categoryid"].astype(str)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_cats["rate_pct"],
        y=top_cats["cat_label"],
        orientation="h",
        text=top_cats["rate_pct"].apply(lambda v: f"{v:.2f}%"),
        textposition="outside",
        marker_color="#2C5545", # Forest Green Theme Color
    ))
    fig.update_layout(
        xaxis_title="Conversion Rate (%)",
        yaxis_title="Category",
        height=max(320, top_n * 24),
        margin=dict(l=20, r=60, t=20, b=40),
        xaxis=dict(ticksuffix="%"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'DM Sans, sans-serif', 'color': '#1A1A1A'}
    )
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data(show_spinner=False)
def build_rfm_from_transactions(events: pd.DataFrame) -> pd.DataFrame:
    try:
        from lifetimes.utils import summary_data_from_transaction_data
    except Exception as e:
        raise ImportError("Missing dependency 'lifetimes'. Please install: pip install lifetimes") from e

    tx = events[events["event"] == "transaction"].copy()
    if tx.empty:
        return pd.DataFrame(columns=["frequency", "recency", "T"])

    tx["transaction_dt"] = pd.to_datetime(tx["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    dedup_cols = [c for c in ["visitorid", "transactionid", "transaction_dt"] if c in tx.columns]
    if dedup_cols:
        tx = tx.drop_duplicates(dedup_cols)

    rfm = summary_data_from_transaction_data(
        transactions=tx, customer_id_col="visitorid",
        datetime_col="transaction_dt", freq="D",
    )
    rfm["frequency"] = rfm["frequency"].clip(lower=0)
    return rfm

@st.cache_resource(show_spinner=False)
def fit_bgnbd_model(rfm: pd.DataFrame):
    try:
        from lifetimes import BetaGeoFitter
    except Exception as e:
        raise ImportError("Missing dependency 'lifetimes'. Please install: pip install lifetimes") from e

    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(rfm["frequency"], rfm["recency"], rfm["T"])
    return bgf

def _standardize_events_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    warnings_list: List[str] = []
    if df is None or df.empty:
        return df, warnings_list

    cols_lower = {c: c.strip().lower() for c in df.columns}
    inv = {}
    for orig, low in cols_lower.items():
        inv.setdefault(low, []).append(orig)

    def pick(*candidates: str) -> str | None:
        for cand in candidates:
            if cand in inv:
                return inv[cand][0]
        return None

    visitor_col = pick("visitorid", "visitor_id", "user_id", "userid", "user", "customer_id", "customerid")
    event_col = pick("event", "event_type", "type", "action", "behavior")
    ts_col = pick("timestamp", "ts", "time", "datetime", "event_time")

    rename_map = {}
    if visitor_col and visitor_col != "visitorid":
        rename_map[visitor_col] = "visitorid"
        warnings_list.append(f"Column `{visitor_col}` identified as `visitorid`.")
    if event_col and event_col != "event":
        rename_map[event_col] = "event"
        warnings_list.append(f"Column `{event_col}` identified as `event`.")
    if ts_col and ts_col != "timestamp":
        rename_map[ts_col] = "timestamp"
        warnings_list.append(f"Column `{ts_col}` identified as `timestamp`.")

    if rename_map:
        df = df.rename(columns=rename_map)

    return df, warnings_list


# ──────────────────────────────────────────────────────────────────────────────
# Module 3: TD‑Multifaceted‑FPMC Smart Recommendation Engine
# ──────────────────────────────────────────────────────────────────────────────

def _mock_category_name_prefix(categoryid: Optional[str]) -> str:
    buckets = [
        "Premium Electronics", "Stylish Home Goods", "Beauty & Personal Care", "Sports & Outdoors",
        "Baby & Kids", "Food & Beverage", "Books & Entertainment", "Office & Stationery",
    ]
    if categoryid is None or (isinstance(categoryid, float) and np.isnan(categoryid)):
        return "Featured Products"
    idx = abs(hash(str(categoryid))) % len(buckets)
    return buckets[idx]

_PREFIX_TO_KEYWORD_POOL = {
    "Premium Electronics": ["laptop", "smartphone", "tech"],
    "Stylish Home Goods": ["furniture", "home", "lamp"],
    "Beauty & Personal Care": ["cosmetics", "perfume", "makeup"],
    "Sports & Outdoors": ["fitness", "running", "outdoor"],
    "Baby & Kids": ["toy", "baby", "diaper"],
    "Food & Beverage": ["food", "drink", "snack"],
    "Books & Entertainment": ["book", "vinyl", "movie"],
    "Office & Stationery": ["office", "stationery", "pen"],
}

def _mock_price_from_item_id(item_id: str) -> float:
    x = abs(hash(str(item_id))) % 900
    base = 99 + x
    tail = [0.00, 0.90, 0.99][abs(hash("tail:" + str(item_id))) % 3]
    return float(base) + tail

def get_item_details(item_id: str, item_to_cat: pd.Series, position_index: int = 0) -> Dict[str, str]:
    item_id = str(item_id)
    cat = item_to_cat.get(item_id)
    prefix = _mock_category_name_prefix(None if cat is None else str(cat))
    name = f"{prefix} - {item_id}"
    price = _mock_price_from_item_id(item_id)

    prefix_to_file = {
        "Premium Electronics": "electronics", "Stylish Home Goods": "home",
        "Beauty & Personal Care": "beauty",   "Sports & Outdoors": "sport",
        "Baby & Kids": "baby",                "Food & Beverage": "food",
        "Books & Entertainment": "book",      "Office & Stationery": "office",
    }
    file_prefix = prefix_to_file.get(prefix, "office")
    idx = int(position_index) % 5

    image_path = os.path.join("assets", f"{file_prefix}_{idx}.jpg")
    fallback_path = os.path.join("assets", "default_icon.svg")
    if not os.path.exists(image_path):
        image_path = fallback_path

    return {"image_path": image_path, "name": name, "price": f"¥{price:,.2f}"}

@st.cache_data(show_spinner=False)
def load_events_for_reco_default(data_dir: str = ".") -> pd.DataFrame:
    ensure_data_files(data_dir=data_dir)
    events, _, _ = load_data(data_dir=data_dir)
    events = clean_events(events)
    if "itemid" in events.columns:
        events["itemid"] = events["itemid"].astype(str)
    events["visitorid"] = events["visitorid"].astype(str)
    return events

@st.cache_data(show_spinner=False)
def load_item_category_latest(data_dir: str = ".") -> pd.Series:
    p1 = os.path.join(data_dir, "item_properties_part1.csv")
    p2 = os.path.join(data_dir, "item_properties_part2.csv")
    frames = []
    if os.path.exists(p1): frames.append(pd.read_csv(p1, dtype={"itemid": str}))
    if os.path.exists(p2): frames.append(pd.read_csv(p2, dtype={"itemid": str}))
    if not frames: return pd.Series(dtype="object")

    props = pd.concat(frames, ignore_index=True)
    needed = {"itemid", "property", "value"}
    if not needed.issubset(props.columns): return pd.Series(dtype="object")

    cat = props[props["property"] == "categoryid"][["itemid", "value"] + (["timestamp"] if "timestamp" in props.columns else [])].copy()
    if cat.empty: return pd.Series(dtype="object")

    cat["itemid"] = cat["itemid"].astype(str)
    cat["value"] = cat["value"].astype(str)

    if "timestamp" in cat.columns:
        cat["timestamp"] = pd.to_numeric(cat["timestamp"], errors="coerce")
        cat = cat.dropna(subset=["timestamp"])
        cat = cat.sort_values("timestamp")
        cat = cat.drop_duplicates("itemid", keep="last")
    else:
        cat = cat.drop_duplicates("itemid", keep="last")

    cat_map = cat.set_index("itemid")["value"].rename("categoryid")
    return cat_map

@st.cache_data(show_spinner=False)
def build_transition_tables(
    events: pd.DataFrame,
    item_to_cat: pd.Series,
    topk: int = 80,
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, List[Tuple[str, float]]], Dict[str, List[str]], List[str]]:
    df = events[["visitorid", "ts_sec", "itemid"]].dropna().copy()
    df["visitorid"] = df["visitorid"].astype(str)
    df["itemid"] = df["itemid"].astype(str)
    df = df.sort_values(["visitorid", "ts_sec"])

    df["next_itemid"] = df.groupby("visitorid")["itemid"].shift(-1)
    df["next_ts"] = df.groupby("visitorid")["ts_sec"].shift(-1)
    df["dt"] = df["next_ts"] - df["ts_sec"]
    df = df.dropna(subset=["next_itemid", "dt"])
    df = df[(df["dt"] > 0) & (df["dt"] <= float(SESSION_GAP_SEC))]

    pair = df.groupby(["itemid", "next_itemid"]).size().rename("cnt").reset_index()
    pair["prob"] = pair["cnt"] / pair.groupby("itemid")["cnt"].transform("sum")
    pair = pair.sort_values(["itemid", "prob"], ascending=[True, False])
    item_next: Dict[str, List[Tuple[str, float]]] = (
        pair.groupby("itemid").head(topk).groupby("itemid")[["next_itemid", "prob"]]
        .apply(lambda x: list(map(tuple, x.values.tolist()))).to_dict()
    )

    cat_series = item_to_cat
    df["cat"] = df["itemid"].map(cat_series)
    df["next_cat"] = df["next_itemid"].map(cat_series)
    cat_df = df.dropna(subset=["cat", "next_cat"])
    cat_pair = cat_df.groupby(["cat", "next_cat"]).size().rename("cnt").reset_index()
    cat_pair["prob"] = cat_pair["cnt"] / cat_pair.groupby("cat")["cnt"].transform("sum")
    cat_pair = cat_pair.sort_values(["cat", "prob"], ascending=[True, False])
    cat_next: Dict[str, List[Tuple[str, float]]] = (
        cat_pair.groupby("cat").head(topk).groupby("cat")[["next_cat", "prob"]]
        .apply(lambda x: list(map(tuple, x.values.tolist()))).to_dict()
    )

    tmp = df[["itemid"]].copy()
    tmp["cat"] = df["itemid"].map(cat_series)
    tmp = tmp.dropna(subset=["cat"])
    item_pop = tmp.groupby(["cat", "itemid"]).size().rename("cnt").reset_index()
    item_pop = item_pop.sort_values(["cat", "cnt"], ascending=[True, False])
    cat_to_items: Dict[str, List[str]] = (
        item_pop.groupby("cat").head(200).groupby("cat")["itemid"].apply(list).to_dict()
    )

    global_hot = events.dropna(subset=["itemid"])["itemid"].astype(str).value_counts().head(500).index.tolist()
    return item_next, cat_next, cat_to_items, global_hot

def _softmax_top(scores: Dict[str, float], topn: int = 5) -> List[Tuple[str, float]]:
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: max(topn, 1)]
    if not items: return []
    vals = np.array([max(v, 0.0) for _, v in items], dtype=float)
    if vals.sum() <= 0: probs = np.ones_like(vals) / len(vals)
    else:
        ex = np.exp(vals - vals.max())
        probs = ex / ex.sum()
    return [(items[i][0], float(probs[i])) for i in range(len(items))]

RECO_CACHE_DIR = "reco_cache"

@st.cache_resource(show_spinner=False)
def load_reco_cache() -> dict:
    result: dict = {
        "precomputed_recs": {}, "transitions": {}, "active_users": np.array([], dtype=object), "top_users": np.array([], dtype=object),
    }
    recs_path   = os.path.join(RECO_CACHE_DIR, "precomputed_recs.pkl")
    trans_path  = os.path.join(RECO_CACHE_DIR, "transitions.pkl")
    active_path = os.path.join(RECO_CACHE_DIR, "active_users.npy")
    top_path    = os.path.join(RECO_CACHE_DIR, "top_users.npy")

    if os.path.exists(recs_path):
        with open(recs_path, "rb") as f: result["precomputed_recs"] = pickle.load(f)
    if os.path.exists(trans_path):
        with open(trans_path, "rb") as f: result["transitions"] = pickle.load(f)
    if os.path.exists(active_path): result["active_users"] = np.load(active_path, allow_pickle=True)
    if os.path.exists(top_path): result["top_users"] = np.load(top_path, allow_pickle=True)
    return result

def recommend_td_multifaceted_fpmc(
    visitorid: str, events: pd.DataFrame, item_to_cat: pd.Series,
    item_next: Dict[str, List[Tuple[str, float]]], cat_next: Dict[str, List[Tuple[str, float]]],
    cat_to_items: Dict[str, List[str]], global_hot: List[str],
    k: int = 5, alpha: float = 0.25, beta: float = 0.55, gamma: float = 0.20, use_time_decay: bool = True,
) -> List[Tuple[str, str, float]]:
    u = str(visitorid).strip()
    u_ev = events[events["visitorid"].astype(str) == u].dropna(subset=["itemid"]).copy()
    if u_ev.empty: return [(it, "hot", 0.2) for it in global_hot[:k]]

    u_ev = u_ev.sort_values("ts_sec")
    last3  = u_ev.tail(3)[["itemid", "ts_sec"]].values.tolist()
    t_last = float(last3[-1][1])

    hist = u_ev.tail(200)[["itemid", "ts_sec"]].copy()
    hist["cat"] = hist["itemid"].astype(str).map(item_to_cat)
    hist = hist.dropna(subset=["cat"])
    if not hist.empty:
        hist["w"] = np.exp(-LAMBDA_DECAY * (t_last - hist["ts_sec"].astype(float))) if use_time_decay else 1.0
        pref = hist.groupby("cat")["w"].sum()
        pref = pref / (pref.sum() + 1e-12)
    else: pref = pd.Series(dtype=float)

    seen: set = set(u_ev["itemid"].astype(str).unique().tolist())
    scores:  Dict[str, float] = {}
    reasons: Dict[str, str]   = {}

    for item_i, ts_i in last3:
        item_i = str(item_i)
        w = float(np.exp(-LAMBDA_DECAY * float(t_last - float(ts_i)))) if use_time_decay else 1.0
        for nxt, p in item_next.get(item_i, [])[:80]:
            if nxt in seen: continue
            scores[nxt] = scores.get(nxt, 0.0) + beta * w * float(p)
            reasons.setdefault(nxt, "seq")

    for item_i, ts_i in last3:
        item_i = str(item_i)
        cat_i  = item_to_cat.get(item_i)
        if cat_i is None or (isinstance(cat_i, float) and np.isnan(cat_i)): continue
        w = float(np.exp(-LAMBDA_DECAY * float(t_last - float(ts_i)))) if use_time_decay else 1.0
        for next_cat, p in cat_next.get(str(cat_i), [])[:20]:
            cand = cat_to_items.get(str(next_cat), [])[:80]
            if not cand: continue
            share = float(p) / max(len(cand), 1)
            for it in cand:
                if it in seen: continue
                scores[it] = scores.get(it, 0.0) + gamma * w * share
                if reasons.get(it) != "seq": reasons.setdefault(it, "cat")

    if not pref.empty and scores:
        for it in list(scores.keys()):
            cat = item_to_cat.get(str(it))
            if cat is None or (isinstance(cat, float) and np.isnan(cat)): continue
            scores[it] += alpha * float(pref.get(str(cat), 0.0))

    if len(scores) < k:
        for it in global_hot:
            if it in seen or it in scores: continue
            scores[it] = 1e-6
            reasons.setdefault(it, "hot")
            if len(scores) >= k * 3: break

    top = _softmax_top(scores, topn=k)
    return [(it, {"seq":"seq","cat":"cat"}.get(reasons.get(it,"hot"),"hot"), float(p)) for it, p in top][:k]

@st.cache_data(show_spinner=False)
def _run_realtime_reco(
    visitorid: str, events_hash: int, events: pd.DataFrame, item_to_cat: pd.Series,
    item_next_pkl: bytes, cat_next_pkl: bytes, cat_to_items_pkl: bytes, global_hot_tuple: tuple,
    alpha: float, beta: float, gamma: float, use_time_decay: bool,
) -> List[Tuple[str, str, float]]:
    item_next    = pickle.loads(item_next_pkl)
    cat_next     = pickle.loads(cat_next_pkl)
    cat_to_items = pickle.loads(cat_to_items_pkl)
    global_hot   = list(global_hot_tuple)
    return recommend_td_multifaceted_fpmc(
        visitorid=visitorid, events=events, item_to_cat=item_to_cat,
        item_next=item_next, cat_next=cat_next, cat_to_items=cat_to_items,
        global_hot=global_hot, k=5, alpha=alpha, beta=beta, gamma=gamma, use_time_decay=use_time_decay,
    )

_REASON_EXPLAIN = {
    "seq": "Based on your recently viewed items",
    "cat": "Popular items from your interested categories",
    "hot": "Global popular items",
}

def _render_rec_cards(recs: List[Tuple[str, str, float]], item_to_cat: pd.Series) -> None:
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            if i >= len(recs):
                st.caption("—")
                continue
            itemid, reason, prob = recs[i]
            label   = {"seq": "Seq", "cat": "Cat"}.get(reason, "Hot")
            explain = _REASON_EXPLAIN.get(reason, "Global popular items")
            details = get_item_details(itemid, item_to_cat=item_to_cat, position_index=i)
            with st.container(border=True):
                img = details["image_path"]
                if os.path.exists(img): st.image(img, use_container_width=True)
                else: st.caption("No image")
                st.markdown(f"**{details['name']}**")
                st.caption(f"{label} · {prob:.2%}")
                st.caption(explain)

def _load_user_events(visitorid: str, data_dir: str) -> Optional[pd.DataFrame]:
    ensure_data_files(data_dir=data_dir)  
    from preprocess import clean_events as _clean_ev
    events_path = os.path.join(data_dir, "events.csv")
    if not os.path.exists(events_path): return None
    chunks: List[pd.DataFrame] = []
    for chunk in pd.read_csv(events_path, chunksize=200_000, dtype={"visitorid": str, "itemid": str}):
        mask = chunk["visitorid"] == visitorid
        if mask.any(): chunks.append(chunk[mask])
    if not chunks: return None
    df = pd.concat(chunks, ignore_index=True)
    df = _clean_ev(df)
    df["itemid"] = df["itemid"].astype(str)
    return df

def render_reco_module(data_dir: str = ".") -> None:
    ensure_data_files(data_dir=data_dir)
    st.subheader("Module 3: Smart Recommendation Engine")

    cache         = load_reco_cache()
    transitions   = cache["transitions"]
    top_users_arr = cache["top_users"]
    top_users: List[str] = [str(u) for u in top_users_arr.tolist()]
    has_transitions = bool(transitions)

    st.sidebar.markdown("### Recommendation Mode")
    if st.session_state.pop("jump_force_manual", False): st.session_state["_reco_user_mode"] = "Manual Input"
    user_mode = st.sidebar.radio(
        label="user_source", options=["Example User", "Manual Input"],
        index=0, key="_reco_user_mode", label_visibility="collapsed",
    )

    if "jump_to_reco_user" in st.session_state:
        st.session_state["_manual_input_id"] = st.session_state.pop("jump_to_reco_user")

    if user_mode == "Example User":
        if not top_users:
            st.sidebar.warning("No top_users cache found. Run `prepare_reco_data.py` first.")
            return
        visitorid = st.sidebar.selectbox("Select Example User", options=top_users)
    else:
        visitorid = st.sidebar.text_input("Manual Input", key="_manual_input_id").strip()
        if not visitorid:
            st.info("Enter a Visitor ID in the sidebar to get real-time recommendations.")
            return

    st.sidebar.markdown("---")
    algo_mode = st.sidebar.radio(
        "Algorithm Version", options=["FPMC", "TD-Multifaceted-FPMC"],
        format_func=lambda x: {"FPMC": "**FPMC** (equal weights, no time decay)", "TD-Multifaceted-FPMC": "**TD-Multifaceted-FPMC** (time decay)"}.get(x, x),
        index=1,
    )
    is_basic       = algo_mode == "FPMC"
    alpha          = 0.0  if is_basic else 0.25
    gamma          = 0.0  if is_basic else 0.20
    use_time_decay = not is_basic

    item_to_cat = load_item_category_latest(data_dir=data_dir)

    if has_transitions:
        item_next    = transitions["item_next"]
        cat_next     = transitions["cat_next"]
        cat_to_items = transitions["cat_to_items"]
        global_hot   = transitions["global_hot"]
    else:
        with st.spinner("Building transition tables (first run only)..."):
            events_full = load_events_for_reco_default(data_dir=data_dir)
            item_next, cat_next, cat_to_items, global_hot = build_transition_tables(events_full, item_to_cat)

    with st.spinner(f"Fetching events for `{visitorid}`..."):
        user_events = _load_user_events(visitorid, data_dir)

    if user_events is None:
        st.error(f"`events.csv` not found in `{data_dir}`.")
        return
    if user_events.empty:
        st.sidebar.error(f"Visitor ID `{visitorid}` not found.")
        return

    item_next_pkl    = pickle.dumps(item_next)
    cat_next_pkl     = pickle.dumps(cat_next)
    cat_to_items_pkl = pickle.dumps(cat_to_items)

    spinner_label = "Running FPMC..." if is_basic else "Running TD-Multifaceted-FPMC..."
    with st.spinner(spinner_label):
        recs = _run_realtime_reco(
            visitorid=visitorid, events_hash=hash(visitorid), events=user_events,
            item_to_cat=item_to_cat, item_next_pkl=item_next_pkl, cat_next_pkl=cat_next_pkl,
            cat_to_items_pkl=cat_to_items_pkl, global_hot_tuple=tuple(global_hot),
            alpha=alpha, beta=0.55, gamma=gamma, use_time_decay=use_time_decay,
        )

    top1_prob = float(recs[0][2]) if recs else 0.0
    st.markdown(f"**User ID:** `{visitorid}`  ·  Algorithm: **{algo_mode}**")
    col_m, col_p = st.columns([1, 3])
    col_m.metric("Top-1 Confidence", f"{top1_prob:.2%}")
    col_p.progress(min(max(top1_prob, 0.0), 1.0))
    _render_rec_cards(recs, item_to_cat)


def render_bgnbd_module(data_dir: str = ".") -> None:
    st.subheader("Module 2: User Behavior Analysis")
    st.sidebar.markdown("### Dataset Loading")
    dataset_mode = st.sidebar.selectbox(
        "Select dataset source", options=["Default dataset (events.csv)", "Upload custom dataset"], index=0,
    )

    if dataset_mode == "Default dataset (events.csv)":
        cache = load_bgnbd_cache()
        missing_keys = [k for k in ("rfm", "bgf", "heatmap_matrix", "heatmap_meta", "p_alive") if cache[k] is None]
        if missing_keys:
            st.warning(f"Incomplete cache files (missing: {', '.join(missing_keys)}).\n\nPlease run: `python prepare_bgnbd_cache.py`, then refresh the page.")
            return

        rfm            = cache["rfm"]
        bgf            = cache["bgf"]
        heatmap_matrix = cache["heatmap_matrix"]
        heatmap_meta   = cache["heatmap_meta"]
        p_alive_df     = cache["p_alive"]
        events_for_tx  = cache["events"]

        if events_for_tx is not None:
            tx_cnt = int((events_for_tx["event"] == "transaction").sum())
            st.caption(f"Cleaned events count: {len(events_for_tx):,}. The number of transaction events: {tx_cnt:,}")

        st.markdown("### Frequency-Recency Matrix Heatmap")
        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            freq_range    = heatmap_meta["freq_range"]
            recency_range = heatmap_meta["recency_range"]
            im = ax.imshow(
                heatmap_matrix, aspect="auto", origin="lower",
                extent=[recency_range[-1], recency_range[0], freq_range[0], freq_range[-1]],
                cmap="coolwarm",
            )
            cbar = fig.colorbar(im, ax=ax)
            ax.set_xlabel("Recency: days since last purchase (t_x)", color="#1A1A1A")
            ax.set_ylabel("Frequency (repeat purchases)", color="#1A1A1A")
            ax.set_title("BG/NBD: Expected Transactions", color="#1A1A1A", fontweight="bold")
            ax.tick_params(colors="#1A1A1A")
            cbar.ax.tick_params(colors="#1A1A1A")
            cbar.set_label("Expected Purchases (next 1 period)", color="#1A1A1A")

            col_fig, col_leg = st.columns([3, 2])
            with col_fig:
                st.pyplot(fig)
            with col_leg:
                st.markdown(
                    "<div style='padding:0.5rem 0.2rem'>"
                    "<p style='font-size:0.79rem;color:#6B6050;margin:0 0 0.9rem;line-height:1.5'>"
                    "x-axis: right → left, t_x increases<br>right = recently purchased, left = purchased long ago</p>"
                    "<div style='margin-bottom:0.6rem;padding:0.55rem 0.8rem;"
                    "border-left:3px solid #C0392B;background:#FEF0EE;border-radius:5px'>"
                    "<strong style='color:#1A1610;font-size:0.85rem'>Top-right</strong>"
                    "<span style='color:#3D3428;font-size:0.82rem'> High frequency + Recent purchase<br>"
                    "<span style='color:#6B6050'>Core active, strongest purchase intent</span></span></div>"
                    "<div style='margin-bottom:0.6rem;padding:0.55rem 0.8rem;"
                    "border-left:3px solid #E67E22;background:#FEF9F0;border-radius:5px'>"
                    "<strong style='color:#1A1610;font-size:0.85rem'>Top-left</strong>"
                    "<span style='color:#3D3428;font-size:0.82rem'> High frequency + Old purchase<br>"
                    "<span style='color:#6B6050'>High churn risk, key warning</span></span></div>"
                    "<div style='margin-bottom:0.6rem;padding:0.55rem 0.8rem;"
                    "border-left:3px solid #2E86C1;background:#EBF5FB;border-radius:5px'>"
                    "<strong style='color:#1A1610;font-size:0.85rem'>Bottom-right</strong>"
                    "<span style='color:#3D3428;font-size:0.82rem'> Low frequency + Recent purchase<br>"
                    "<span style='color:#6B6050'>New customer, shallow purchase history</span></span></div>"
                    "<div style='padding:0.55rem 0.8rem;"
                    "border-left:3px solid #7F8C8D;background:#F4F6F6;border-radius:5px'>"
                    "<strong style='color:#1A1610;font-size:0.85rem'>Bottom-left</strong>"
                    "<span style='color:#3D3428;font-size:0.82rem'> Low frequency + Old purchase<br>"
                    "<span style='color:#6B6050'>Low value or churned</span></span></div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            plt.close(fig)
        except Exception as e:
            st.error(f"Frequency-Recency Matrix Heatmap drawing failed: {e}")

        st.markdown("### Category Conversion Analysis")
        _render_funnel_section()

        st.markdown("### Churn Risk Alert (High-Value Customers)")
        freq_max     = int(max(1, np.nanmax(rfm["frequency"])))
        default_freq = int(max(1, np.nanpercentile(rfm["frequency"], 90)))
        freq_threshold = st.slider("High-Value Threshold: Min Frequency (Repeat Purchases)", min_value=0, max_value=freq_max, value=min(default_freq, freq_max), step=1)
        alive_threshold = st.slider("Churn Alert Threshold: Max P(Alive) (Lower is more dangerous)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

        alert = p_alive_df.copy()
        if alert.index.name == "visitorid": alert = alert.reset_index()
        elif "visitorid" not in alert.columns: alert = alert.reset_index().rename(columns={"index": "visitorid"})

        alert = alert[(alert["frequency"] >= freq_threshold) & (alert["p_alive"] < alive_threshold)].copy()
        alert["churn_risk"] = 1.0 - alert["p_alive"]
        alert = alert.sort_values(["churn_risk", "frequency"], ascending=[False, False])

        show_cols = [c for c in ["visitorid", "frequency", "last_transaction_dt", "p_alive", "churn_risk"] if c in alert.columns]
        st.caption("Filtering logic: High historical contribution (high Frequency) but low probability of being alive (P(Alive) < threshold).")
        st.dataframe(alert[show_cols].head(200), use_container_width=True)

        st.markdown("---")
        st.markdown("### Generate recovery recommendations for lost users")
        st.caption("Select a lost user, then jump to the recommendation engine to generate personalized recovery recommendations.")
        if not alert.empty and "visitorid" in alert.columns:
            churn_users = alert["visitorid"].head(20).tolist()
            selected_churn_user = st.selectbox("Select a lost user", options=churn_users, key="churn_to_reco_user")
            if st.button("Generate Recommendations for Selected User", type="primary", key="churn_to_reco_btn"):
                st.session_state["jump_to_reco_user"] = selected_churn_user
                st.session_state["jump_to_module"] = "Module 3: Smart Recommendation Engine"
                st.session_state["jump_force_manual"] = True
                st.rerun()
        return

    uploaded = st.sidebar.file_uploader("Upload CSV (must include visitorid/event/timestamp or adaptable columns)", type=["csv"])
    if uploaded is None:
        st.info("Please upload a custom dataset CSV in the sidebar, then the BG/NBD analysis will run automatically.")
        return

    try: raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    raw, warn_msgs = _standardize_events_columns(raw)
    for msg in warn_msgs: st.caption(msg)

    missing = [c for c in ["visitorid", "event", "timestamp"] if c not in raw.columns]
    if missing:
        st.error(f"Uploaded dataset is missing necessary columns for analysis.\n\nMissing: {', '.join(missing)}\nExpected: visitorid, event, timestamp (timestamp in milliseconds or seconds Unix time).")
        return

    raw = raw.copy()
    raw["visitorid"] = raw["visitorid"].astype(str)
    raw["event"]     = raw["event"].astype(str)
    raw["timestamp"] = pd.to_numeric(raw["timestamp"], errors="coerce")
    raw = raw.dropna(subset=["visitorid", "event", "timestamp"])

    ts_med = float(np.nanmedian(raw["timestamp"].values))
    if ts_med < 1e12:
        raw["timestamp"] = raw["timestamp"] * 1000.0
        st.caption("Timestamp detected in seconds, automatically converted to milliseconds before cleaning.")

    with st.spinner("Cleaning uploaded dataset..."): events = clean_events(raw)

    tx_cnt = int((events["event"] == "transaction").sum())
    st.caption(f"Cleaned events count: {len(events):,}. The number of transaction events: {tx_cnt:,}")

    with st.spinner("Building RFM data..."): rfm = build_rfm_from_transactions(events)

    if rfm.empty:
        st.warning("No transaction events detected in current data, BG/NBD analysis cannot be performed.")
        return

    with st.spinner("Fitting BG/NBD model..."): bgf = fit_bgnbd_model(rfm)

    st.markdown("### Frequency-Recency Matrix Heatmap")
    try:
        from lifetimes.plotting import plot_frequency_recency_matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        plot_frequency_recency_matrix(bgf, ax=ax)
        ax.set_title("BG/NBD: Expected Transactions", color="#1A1A1A", fontweight="bold")
        ax.xaxis.label.set_color('#1A1A1A')
        ax.yaxis.label.set_color('#1A1A1A')
        ax.tick_params(colors="#1A1A1A")
        
        _, mid, _ = st.columns([2, 3, 2])
        with mid: st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Frequency-Recency Matrix Heatmap drawing failed: {e}")

    st.markdown(
        "**Physical Meaning**:\n"
        "- **Top-right (High Frequency + High Recency)**: Core active customers.\n"
        "- **Bottom-right (High Frequency + Low Recency)**: High historical contribution but not active recently, may be at risk of loss (key warning).\n"
        "- **Top-left (Low Frequency + High Recency)**: New customers or occasional buyers.\n"
        "- **Bottom-left (Low Frequency + Low Recency)**: Low-value or churned customers."
    )

    st.markdown("### Churn Risk Alert (High-Value Customers)")
    freq_max     = int(max(1, np.nanmax(rfm["frequency"])))
    default_freq = int(max(1, np.nanpercentile(rfm["frequency"], 90)))
    freq_threshold = st.slider("High-Value Threshold: Min Frequency (Repeat Purchases)", min_value=0, max_value=freq_max, value=min(default_freq, freq_max), step=1)
    alive_threshold = st.slider("Churn Alert Threshold: Max P(Alive) (Lower is more dangerous)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    palive_np = bgf.conditional_probability_alive(rfm["frequency"], rfm["recency"], rfm["T"])
    rfm["p_alive"] = pd.Series(palive_np, index=rfm.index, name="p_alive")

    tx = events[events["event"] == "transaction"].copy()
    tx["transaction_dt"] = pd.to_datetime(tx["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    last_buy = tx.groupby("visitorid")["transaction_dt"].max().rename("last_transaction_dt")

    alert = rfm.join(last_buy).reset_index().rename(columns={"index": "visitorid"})
    alert = alert[(alert["frequency"] >= freq_threshold) & (alert["p_alive"] < alive_threshold)].copy()
    alert["churn_risk"] = 1.0 - alert["p_alive"]
    alert = alert.sort_values(["churn_risk", "frequency"], ascending=[False, False])

    st.caption("Filtering logic: High historical contribution (high Frequency) but low probability of being alive (P(Alive) < threshold).")
    st.dataframe(alert[["visitorid", "frequency", "last_transaction_dt", "p_alive", "churn_risk"]].head(200), use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Streamlit Page Layout
# ──────────────────────────────────────────────────────────────────────────────

def render_intent_prediction_module() -> None:
    st.subheader("Module 1: Purchase Probability Prediction")
    st.markdown("Based on the **GBT** model, the real-time behavioral characteristics of users are mapped to the exact feature space that is identical to the training stage, and the probability of purchase intention and marketing suggestions are provided.")

    try:
        model, feature_names, scaler = load_model_and_preprocess()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    st.sidebar.header("User Behavior Input")
    st.sidebar.subheader("Behavior Characteristics")
    st.sidebar.caption("Based on the current session's browsing frequency, purchase attempts, and duration of stay, it reflects the immediate intention of this visit.")
    sess_view_cnt = st.sidebar.number_input("Views Count", min_value=0, value=5, step=1)
    sess_cart_cnt = st.sidebar.number_input("Add-to-cart Count", min_value=0, value=1, step=1)
    sess_duration_sec = st.sidebar.number_input("Session Duration (seconds)", min_value=0.0, value=300.0, step=10.0)
    sess_unique_items = st.sidebar.number_input("Unique Items Viewed in Session", min_value=0, value=3, step=1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("RFM / Historical User Features")
    st.sidebar.caption("By combining the user's historical activity level and purchase preferences, the long-term value can be depicted.")
    user_recency_hours = st.sidebar.number_input("Hours Since Last Visit (Recency)", min_value=0.0, value=12.0, step=1.0)
    user_freq_total = st.sidebar.number_input("Total Historical Behaviors (Frequency)", min_value=0.0, value=50.0, step=1.0)
    user_cart_freq = st.sidebar.number_input("Historical Add-to-cart Count", min_value=0.0, value=10.0, step=1.0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Time-decay Features")
    recent_view_hours = st.sidebar.number_input("Hours Since Last View", min_value=0.0, value=1.0, step=0.5)
    recent_cart_hours = st.sidebar.number_input("Hours Since Last Add-to-cart", min_value=0.0, value=2.0, step=0.5)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Behavior Type (Current Session)")
    behavior_type_input = st.sidebar.radio(
        "Session Behavior Pattern",
        options=["focused", "explorer", "normal"],
        format_func=lambda x: {"focused": "Intra-category Repeated Comparison", "explorer": "Cross-category Exploration", "normal": "Normal Browsing"}.get(x, x),
        index=2,
    )
    st.sidebar.caption("Manually select based on the current conversation situation.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Category Preferences (Optional)")
    st.sidebar.caption("Describe the breadth and concentration of users' preferences for different product categories.")
    user_cat_breadth = st.sidebar.number_input("Historical Category Breadth", min_value=0.0, value=5.0, step=1.0)
    user_cat_concentration = st.sidebar.slider("Category Concentration (0~1)", min_value=0.0, max_value=1.0, value=0.4, step=0.05)

    inputs = {
        "sess_view_cnt": float(sess_view_cnt), "sess_cart_cnt": float(sess_cart_cnt),
        "sess_duration_sec": float(sess_duration_sec), "sess_unique_items": float(sess_unique_items),
        "user_recency_hours": float(user_recency_hours), "user_freq_total": float(user_freq_total),
        "user_cart_freq": float(user_cart_freq), "recent_view_hours": float(recent_view_hours),
        "recent_cart_hours": float(recent_cart_hours), "user_cat_breadth": float(user_cat_breadth),
        "user_cat_concentration": float(user_cat_concentration),
    }

    if "m1_prob" not in st.session_state:
        st.session_state["m1_prob"] = None
        st.session_state["m1_behavior"] = "normal"

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Real-time Purchase Intent Prediction")
        if st.button("Predict Now", type="primary"):
            X_scaled = build_feature_vector_from_inputs(inputs, feature_names, scaler)
            prob = predict_proba_single(model, X_scaled)

            _view = float(inputs.get("sess_view_cnt", 0))
            _cart = float(inputs.get("sess_cart_cnt", 0))
            if _cart == 0:
                if _view == 0: prob = min(prob, 0.12)
                elif _view == 1: prob = min(prob, 0.28)

            st.session_state["m1_prob"] = prob
            st.session_state["m1_behavior"] = behavior_type_input

        if st.session_state["m1_prob"] is not None:
            prob = st.session_state["m1_prob"]
            render_gauge(prob)
        else:
            st.info("Fill in the user's real-time behavioral features on the left, then click 'Predict Now' above to get the results.")

    with col_right:
        if st.session_state["m1_prob"] is not None:
            render_behavior_marketing(st.session_state["m1_prob"], st.session_state["m1_behavior"])
        else:
            # 删除多余空行，改用 flex 居中对齐，与左侧浑然一体
            st.markdown(
                """
                <div style='display: flex; align-items: center; justify-content: center; margin-top: 0rem; margin-bottom: 0.5rem;'>
                    <h3 style='text-align: center; margin: 0; font-size: 1.35rem;'>Precision Marketing Suggestions</h3>
                </div>
                """, 
                unsafe_allow_html=True
            )
            st.info("After clicking 'Predict Now', precision marketing suggestions will be displayed here.")

def main():
    st.set_page_config(
        page_title="System for User Purchase Behavior Analysis and Prediction",
        page_icon=None,
        layout="wide",
    ) 
    
    # Inject Custom CSS to modify the colors of the plus and minus buttons to retro gold #E2B659
    st.markdown(
        """
        <style>
        /* Targets plus and minus buttons for all number_inputs */
        [data-testid="stNumberInputStepUp"] svg,
        [data-testid="stNumberInputStepDown"] svg {
            fill: #E2B659 !important;
            color: #E2B659 !important;
        }
        
        /* Fallback for text/icon color of the buttons themselves */
        [data-testid="stNumberInputStepUp"],
        [data-testid="stNumberInputStepDown"] {
            color: #E2B659 !important;
        }
        
        /* Hover effect (optional, slightly brightens to increase interactivity) */
        [data-testid="stNumberInputStepUp"]:hover svg,
        [data-testid="stNumberInputStepDown"]:hover svg {
            fill: #F3C76A !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Inject the main CSS Theme here
    apply_custom_theme()

    st.title("Analysis and Prediction of User Purchase Behaviors")
    st.sidebar.markdown(
        """<div style="padding:0.2rem 0 0.4rem 0; display: flex; align-items: center;">
<span style="font-family:'Playfair Display',Georgia,serif;font-size:2.0rem;font-weight:700;color:#E2B659;letter-spacing:0.06em">Modules</span>
</div>""",
        unsafe_allow_html=True,
    )
    
    # 更新模块官方名称
    _module_options = [
        "Module 1: Purchase Probability Prediction",
        "Module 2: User Behavior Analysis",
        "Module 3: Smart Recommendation Engine",
    ]
    
    if "jump_to_module" in st.session_state:
        st.session_state["_module_radio"] = st.session_state.pop("jump_to_module")
        
    module = st.sidebar.radio(
        "Select", 
        options=_module_options, 
        index=0, 
        key="_module_radio", 
        label_visibility="collapsed",
        # 利用 format_func 切割字符串，使侧边栏只显示冒号后面的名称
        format_func=lambda x: x.split(": ")[1]
    )
    
    st.sidebar.markdown(
        "<div style='border-top:2px solid #B5860D;margin:0.3rem 0 0.5rem 0'></div>",
        unsafe_allow_html=True,
    )

    if module == "Module 1: Purchase Probability Prediction":
        render_intent_prediction_module()
    elif module == "Module 2: User Behavior Analysis":
        render_bgnbd_module(data_dir=".")
    else:
        render_reco_module(data_dir=".")

if __name__ == "__main__":
    main()