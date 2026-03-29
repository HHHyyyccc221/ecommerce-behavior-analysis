import os
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from preprocess import LAMBDA_DECAY, SESSION_GAP_SEC, clean_events, load_data

# ──────────────────────────────────────────────────────────────────────────────
# 0. 注入全局 CSS 自定义主题 (复古账本风格)
# ──────────────────────────────────────────────────────────────────────────────
def apply_custom_theme():
    st.markdown(
        """
        <style>
        /* 导入 Google 字体：Lora(衬线体用于标题), DM Sans(无衬线用于正文) */
        @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@400;500;700&display=swap');

        /* 基础应用色彩 */
        .stApp {
            background-color: #F4F1EA !important; /* 米白背景 */
        }

        /* 基础排版 */
        html, body, [class*="css"], .stMarkdown p, .stMarkdown li {
            font-family: 'DM Sans', sans-serif !important;
            color: #1A1A1A !important;
        }

        /* 统一标题样式 */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Lora', serif !important;
            color: #1A1A1A !important;
            font-weight: 700 !important;
        }
        h1 { font-size: 2.2rem !important; margin-bottom: 1.5rem !important; border-bottom: 2px solid #1A1A1A; padding-bottom: 0.5rem; }
        h2 { font-size: 1.8rem !important; margin-top: 2rem !important; margin-bottom: 1rem !important; }
        h3 { font-size: 1.4rem !important; margin-top: 1.5rem !important; margin-bottom: 0.8rem !important; }

        /* 侧边栏设计 */
        [data-testid="stSidebar"] {
            background-color: #2C5545 !important; /* 深森林绿 */
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #E2B659 !important; /* 侧栏标题使用复古金 */
            border-bottom: none;
        }
        [data-testid="stSidebar"] *, [data-testid="stSidebar"] p {
            color: #F4F1EA !important; /* 侧栏普通文字米白色 */
        }
        
        /* 修复侧边栏内 Input 框的可见性 */
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
            color: #F4F1EA !important; /* 保证 label 不会被覆盖 */
        }

        /* 修复侧边栏 selectbox 内部显示文字颜色 */
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

        /* 按钮设计（高对比度，解决看不清的问题） */
        div.stButton > button {
            background-color: #1A1A1A !important;
            color: #F4F1EA !important;
            border: 2px solid #1A1A1A !important;
            border-radius: 0px !important;
            padding: 0.5rem 1rem !important;
            font-weight: bold !important;
            font-family: 'DM Sans', sans-serif !important;
            box-shadow: 4px 4px 0px #E2B659 !important; /* 金色硬阴影 */
            transition: all 0.2s ease-in-out !important;
        }
        div.stButton > button:hover {
            background-color: #E2B659 !important;
            color: #1A1A1A !important;
            box-shadow: 4px 4px 0px #1A1A1A !important; /* 黑色硬阴影 */
        }
        div.stButton > button * {
            color: inherit !important;
        }

        /* 数据指标卡片 Metric */
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

        /* 提示框 Alert/Info */
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

        /* 数据表格 DataFrame */
        .stDataFrame {
            border: 2px solid #1A1A1A !important;
            border-radius: 0px !important;
            box-shadow: 4px 4px 0px #1A1A1A !important;
        }

        /* 通用区块/推荐卡片容器 */
        [data-testid="stVerticalBlock"] > div.element-container > div.stContainer {
            background-color: #FFFFFF !important;
            border: 2px solid #1A1A1A !important;
            border-radius: 0px !important;
            box-shadow: 4px 4px 0px #1A1A1A !important;
            padding: 1rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. 模型与特征变换加载（与训练阶段完全同步）
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model_and_preprocess() -> Tuple[object, List[str], object]:
    """
    加载已训练好的冠军模型（LightGBM GBT）以及
    训练阶段使用的 feature_names 与 StandardScaler。
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
        raise FileNotFoundError("未找到 gbt_lgbm.pkl，请确认模型文件是否位于项目根目录或 saved_models/ 下。")

    cache_dir = "cache"
    feat_path = os.path.join(cache_dir, "feature_names.npy")
    scaler_path = os.path.join(cache_dir, "scaler.pkl")
    if not (os.path.exists(feat_path) and os.path.exists(scaler_path)):
        raise FileNotFoundError(
            "未找到特征缓存文件 cache/feature_names.npy 或 cache/scaler.pkl。\n"
            "请先运行训练脚本（如 main.py / run_preprocessing_cached），生成缓存后再启动本应用。"
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
    将侧边栏中的原始输入转换为与训练阶段完全一致的特征向量。
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
    """对单个样本进行预测，返回购买意向为 1 的概率。"""
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
    """使用 Plotly 绘制符合新配色的仪表盘。"""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100.0,
            number={"suffix": "%", "font": {"size": 32, "color": "#1A1A1A"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#1A1A1A"},
                "bar": {"color": "#1A1A1A"},
                "steps": [
                    {"range": [0, 40], "color": "#F4F1EA"},  # 米白
                    {"range": [40, 70], "color": "#E2B659"},  # 强调金
                    {"range": [70, 85], "color": "#59738A"},  # 灰蓝
                    {"range": [85, 100], "color": "#2C5545"}, # 森林绿
                ],
                "threshold": {
                    "line": {"color": "#D35A5A", "width": 4}, # 危险红
                    "thickness": 0.75,
                    "value": prob * 100.0,
                },
            },
            title={"text": "Predicted Purchase Intent", "font": {"family": "Lora", "size": 20, "color": "#1A1A1A"}},
        )
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=80, b=40),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'DM Sans, sans-serif', 'color': '#1A1A1A'}
    )
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# 模块一辅助：行为类型缓存加载 & 精细化营销建议矩阵
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
    "focused":  "类目内反复比较型",
    "explorer": "类目间跳跃型",
    "normal":   "普通浏览型",
}

_MARKETING_MATRIX: dict = {
    ("focused",  "high"):   "高意向 + 类目内反复比较：用户正在比价，建议**立即推送限时优惠券**，消除价格顾虑，促成下单。",
    ("focused",  "medium"): "中意向 + 类目内反复比较：用户在同类商品中犹豫，建议展示**好评与销量排名**，强化信任，并附小额优惠。",
    ("focused",  "low"):    "低意向 + 类目内反复比较：用户探索阶段，建议推送**类目热销榜**，引导发现更多选择，暂不激进促销。",
    ("explorer", "high"):   "高意向 + 跨类目跳跃：用户需求明确但范围宽泛，建议**搜索引导 + 个性化推荐**，帮助快速锁定目标商品。",
    ("explorer", "medium"): "中意向 + 跨类目跳跃：用户处于广泛探索阶段，建议展示**热销榜 + 相关类目组合推荐**，缩小决策范围。",
    ("explorer", "low"):    "低意向 + 跨类目跳跃：用户兴趣分散，建议以**品牌曝光和内容种草**为主，降低促销力度。",
    ("normal",   "high"):   "高意向 + 普通浏览：购买信号强，建议**即时推送优惠券或限时折扣**，直接引导下单。",
    ("normal",   "medium"): "中意向 + 普通浏览：有一定兴趣，建议推荐**相关商品或用户评价**，进一步激活购买意愿。",
    ("normal",   "low"):    "低意向 + 普通浏览：建议以**品牌曝光**为主，可减少激进促销策略。",
}

def _prob_level(prob: float) -> str:
    if prob >= 0.6: return "high"
    if prob >= 0.3: return "medium"
    return "low"

def render_behavior_marketing(prob: float, behavior_type: str) -> None:
    level  = _prob_level(prob)
    label  = _BEHAVIOR_LABEL.get(behavior_type, "普通浏览型")
    advice = _MARKETING_MATRIX.get(
        (behavior_type, level),
        _MARKETING_MATRIX[("normal", level)],
    )
    st.markdown("<div style='margin-top:60px'></div>", unsafe_allow_html=True)
    st.markdown("### 精细营销建议")
    st.info(advice)

def marketing_suggestion(prob: float) -> str:
    if prob >= 0.8: return "高购买意向（>80%）：建议立即发放有吸引力的优惠券或限时折扣，引导下单。"
    if prob >= 0.5: return "中等购买意向（50%~80%）：可推荐相关商品或适度优惠，进一步激活购买兴趣。"
    if prob >= 0.3: return "偏低购买意向（30%~50%）：建议通过内容种草、展示评价等方式提升信任度。"
    return "极低购买意向（<30%）：建议以品牌曝光为主，可减少激进促销策略。"


# ──────────────────────────────────────────────────────────────────────────────
# 模块二：BG/NBD 客户长期价值分析
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
        st.warning("未找到漏斗缓存，请先运行：`python prepare_funnel_cache.py`")
        return

    st.caption("展示浏览量 Top N 类目的 浏览→加购→购买 各阶段转化率，揭示哪些类目最能促成下单。")

    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        top_n = st.slider("展示 Top N 类目", min_value=5, max_value=30, value=15, step=5)
    with col_ctrl2:
        metric = st.radio(
            "转化率指标",
            options=["view_to_cart_rate", "cart_to_tx_rate", "view_to_tx_rate"],
            format_func=lambda x: {
                "view_to_cart_rate": "浏览→加购转化率",
                "cart_to_tx_rate":   "加购→购买转化率",
                "view_to_tx_rate":   "浏览→购买总转化率",
            }.get(x, x),
            horizontal=True,
        )

    top_cats = cat_df.head(top_n).copy()
    top_cats = top_cats.sort_values(metric, ascending=True)
    top_cats["rate_pct"] = (top_cats[metric] * 100).round(2)
    top_cats["cat_label"] = "类目 " + top_cats["categoryid"].astype(str)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_cats["rate_pct"],
        y=top_cats["cat_label"],
        orientation="h",
        text=top_cats["rate_pct"].apply(lambda v: f"{v:.2f}%"),
        textposition="outside",
        marker_color="#2C5545", # 更换为森林绿主题色
    ))
    fig.update_layout(
        xaxis_title="转化率 (%)",
        yaxis_title="类目",
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
        raise ImportError("缺少依赖 lifetimes。请先安装：pip install lifetimes") from e

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
        raise ImportError("缺少依赖 lifetimes。请先安装：pip install lifetimes") from e

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
        warnings_list.append(f"已将列 `{visitor_col}` 识别为 `visitorid`。")
    if event_col and event_col != "event":
        rename_map[event_col] = "event"
        warnings_list.append(f"已将列 `{event_col}` 识别为 `event`。")
    if ts_col and ts_col != "timestamp":
        rename_map[ts_col] = "timestamp"
        warnings_list.append(f"已将列 `{ts_col}` 识别为 `timestamp`。")

    if rename_map:
        df = df.rename(columns=rename_map)

    return df, warnings_list


# ──────────────────────────────────────────────────────────────────────────────
# 模块三：TD‑Multifaceted‑FPMC 智能推荐引擎
# ──────────────────────────────────────────────────────────────────────────────

def _mock_category_name_prefix(categoryid: Optional[str]) -> str:
    buckets = [
        "高端电子产品", "时尚家居用品", "美妆个护精选", "运动户外装备",
        "母婴儿童用品", "食品饮料甄选", "图书文娱周边", "办公学习用品",
    ]
    if categoryid is None or (isinstance(categoryid, float) and np.isnan(categoryid)):
        return "精选商品"
    idx = abs(hash(str(categoryid))) % len(buckets)
    return buckets[idx]

_PREFIX_TO_KEYWORD_POOL = {
    "高端电子产品": ["laptop", "smartphone", "tech"],
    "时尚家居用品": ["furniture", "home", "lamp"],
    "美妆个护精选": ["cosmetics", "perfume", "makeup"],
    "运动户外装备": ["fitness", "running", "outdoor"],
    "母婴儿童用品": ["toy", "baby", "diaper"],
    "食品饮料甄选": ["food", "drink", "snack"],
    "图书文娱周边": ["book", "vinyl", "movie"],
    "办公学习用品": ["office", "stationery", "pen"],
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
        "高端电子产品": "electronics", "时尚家居用品": "home",
        "美妆个护精选": "beauty",       "运动户外装备": "sport",
        "母婴儿童用品": "baby",         "食品饮料甄选": "food",
        "图书文娱周边": "book",         "办公学习用品": "office",
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
    "seq": "基于您最近浏览的商品序列推荐",
    "cat": "来自您感兴趣类目的热门商品",
    "hot": "全站热门商品",
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
            explain = _REASON_EXPLAIN.get(reason, "全站热门商品")
            details = get_item_details(itemid, item_to_cat=item_to_cat, position_index=i)
            with st.container(border=True):
                img = details["image_path"]
                if os.path.exists(img): st.image(img, use_container_width=True)
                else: st.caption("No image")
                st.markdown(f"**{details['name']}**")
                st.caption(f"{label} · {prob:.2%}")
                st.caption(explain)

def _load_user_events(visitorid: str, data_dir: str) -> Optional[pd.DataFrame]:
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
    st.subheader("模块三：智能推荐引擎")

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
        format_func=lambda x: {"FPMC": "FPMC (均等权重，无时间衰减)", "TD-Multifaceted-FPMC": "TD-Multifaceted-FPMC (时间衰减 e⁻λΔt)"}.get(x, x),
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
    st.subheader("客户长期价值分析（BG/NBD）")
    st.sidebar.markdown("### 数据集加载")
    dataset_mode = st.sidebar.selectbox(
        "选择数据集来源", options=["Default dataset (events.csv)", "Upload custom dataset"], index=0,
    )

    if dataset_mode == "Default dataset (events.csv)":
        cache = load_bgnbd_cache()
        missing_keys = [k for k in ("rfm", "bgf", "heatmap_matrix", "heatmap_meta", "p_alive") if cache[k] is None]
        if missing_keys:
            st.warning(f"缓存文件不完整（缺少：{', '.join(missing_keys)}）。\n\n请先运行：`python prepare_bgnbd_cache.py`，再刷新页面。")
            return

        rfm            = cache["rfm"]
        bgf            = cache["bgf"]
        heatmap_matrix = cache["heatmap_matrix"]
        heatmap_meta   = cache["heatmap_meta"]
        p_alive_df     = cache["p_alive"]
        events_for_tx  = cache["events"]

        if events_for_tx is not None:
            tx_cnt = int((events_for_tx["event"] == "transaction").sum())
            st.caption(f"清洗后事件数：{len(events_for_tx):,}；其中 transaction 事件数：{tx_cnt:,}")

        st.markdown("### Recency vs Frequency 矩阵热力图")
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
                    "x 轴右→左 t_x 增大<br>右侧=最近刚买，左侧=很久没买</p>"
                    "<div style='margin-bottom:0.6rem;padding:0.55rem 0.8rem;"
                    "border-left:3px solid #C0392B;background:#FEF0EE;border-radius:5px'>"
                    "<strong style='color:#1A1610;font-size:0.85rem'>右上</strong>"
                    "<span style='color:#3D3428;font-size:0.82rem'> 高频 + 最近刚买<br>"
                    "<span style='color:#6B6050'>核心活跃，购买意愿最强</span></span></div>"
                    "<div style='margin-bottom:0.6rem;padding:0.55rem 0.8rem;"
                    "border-left:3px solid #E67E22;background:#FEF9F0;border-radius:5px'>"
                    "<strong style='color:#1A1610;font-size:0.85rem'>左上</strong>"
                    "<span style='color:#3D3428;font-size:0.82rem'> 高频 + 很久没买<br>"
                    "<span style='color:#6B6050'>流失风险高，重点预警</span></span></div>"
                    "<div style='margin-bottom:0.6rem;padding:0.55rem 0.8rem;"
                    "border-left:3px solid #2E86C1;background:#EBF5FB;border-radius:5px'>"
                    "<strong style='color:#1A1610;font-size:0.85rem'>右下</strong>"
                    "<span style='color:#3D3428;font-size:0.82rem'> 低频 + 最近刚买<br>"
                    "<span style='color:#6B6050'>新客，历史较浅</span></span></div>"
                    "<div style='padding:0.55rem 0.8rem;"
                    "border-left:3px solid #7F8C8D;background:#F4F6F6;border-radius:5px'>"
                    "<strong style='color:#1A1610;font-size:0.85rem'>左下</strong>"
                    "<span style='color:#3D3428;font-size:0.82rem'> 低频 + 很久没买<br>"
                    "<span style='color:#6B6050'>低价值或已流失</span></span></div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            plt.close(fig)
        except Exception as e:
            st.error(f"热力图绘制失败：{e}")

        st.markdown("### 类目转化分析")
        _render_funnel_section()

        st.markdown("### 流失风险预警（高价值客户）")
        freq_max     = int(max(1, np.nanmax(rfm["frequency"])))
        default_freq = int(max(1, np.nanpercentile(rfm["frequency"], 90)))
        freq_threshold = st.slider("高价值阈值：Frequency（重复购买次数）下限", min_value=0, max_value=freq_max, value=min(default_freq, freq_max), step=1)
        alive_threshold = st.slider("流失预警阈值：P(Alive) 上限（越低越危险）", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

        alert = p_alive_df.copy()
        if alert.index.name == "visitorid": alert = alert.reset_index()
        elif "visitorid" not in alert.columns: alert = alert.reset_index().rename(columns={"index": "visitorid"})

        alert = alert[(alert["frequency"] >= freq_threshold) & (alert["p_alive"] < alive_threshold)].copy()
        alert["churn_risk"] = 1.0 - alert["p_alive"]
        alert = alert.sort_values(["churn_risk", "frequency"], ascending=[False, False])

        show_cols = [c for c in ["visitorid", "frequency", "last_transaction_dt", "p_alive", "churn_risk"] if c in alert.columns]
        st.caption("筛选逻辑：历史贡献高（Frequency 高）但存活概率低（P(Alive) < 阈值）。")
        st.dataframe(alert[show_cols].head(200), use_container_width=True)

        st.markdown("---")
        st.markdown("### 为流失用户生成挽回推荐")
        st.caption("选择一位流失风险用户，一键跳转至推荐引擎，生成个性化挽回推荐。")
        if not alert.empty and "visitorid" in alert.columns:
            churn_users = alert["visitorid"].head(20).tolist()
            selected_churn_user = st.selectbox("选择流失风险用户", options=churn_users, key="churn_to_reco_user")
            if st.button("为该用户生成推荐", type="primary", key="churn_to_reco_btn"):
                st.session_state["jump_to_reco_user"] = selected_churn_user
                st.session_state["jump_to_module"] = "模块三：智能推荐引擎"
                st.session_state["jump_force_manual"] = True
                st.rerun()
        return

    uploaded = st.sidebar.file_uploader("上传 CSV（需包含 visitorid/event/timestamp 或可自动适配的列）", type=["csv"])
    if uploaded is None:
        st.info("请在侧边栏上传自定义数据集 CSV，然后将自动运行 BG/NBD 分析。")
        return

    try: raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"CSV 读取失败：{e}")
        return

    raw, warn_msgs = _standardize_events_columns(raw)
    for msg in warn_msgs: st.caption(msg)

    missing = [c for c in ["visitorid", "event", "timestamp"] if c not in raw.columns]
    if missing:
        st.error(f"上传数据集缺少必要列，无法分析。\n\n缺少列：{', '.join(missing)}\n期望列：visitorid, event, timestamp（timestamp 为毫秒或秒级 Unix 时间戳）。")
        return

    raw = raw.copy()
    raw["visitorid"] = raw["visitorid"].astype(str)
    raw["event"]     = raw["event"].astype(str)
    raw["timestamp"] = pd.to_numeric(raw["timestamp"], errors="coerce")
    raw = raw.dropna(subset=["visitorid", "event", "timestamp"])

    ts_med = float(np.nanmedian(raw["timestamp"].values))
    if ts_med < 1e12:
        raw["timestamp"] = raw["timestamp"] * 1000.0
        st.caption("检测到 timestamp 可能为秒级，已自动转换为毫秒级后再清洗。")

    with st.spinner("正在清洗上传数据集…"): events = clean_events(raw)

    tx_cnt = int((events["event"] == "transaction").sum())
    st.caption(f"清洗后事件数：{len(events):,}；其中 transaction 事件数：{tx_cnt:,}")

    with st.spinner("正在构建 RFM 数据…"): rfm = build_rfm_from_transactions(events)

    if rfm.empty:
        st.warning("当前数据中未检测到 transaction 事件，无法进行 BG/NBD 分析。")
        return

    with st.spinner("正在拟合 BG/NBD 模型…"): bgf = fit_bgnbd_model(rfm)

    st.markdown("### Recency vs Frequency 矩阵热力图")
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
        st.error(f"热力图绘制失败：{e}")

    st.markdown(
        "**物理意义**：\n"
        "- **右上（高 Frequency + 高 Recency）**：核心活跃客户。\n"
        "- **右下（高 Frequency + 低 Recency）**：历史贡献高但近期不活跃，可能流失（重点预警）。\n"
        "- **左上（低 Frequency + 高 Recency）**：新客或偶发客。\n"
        "- **左下（低 Frequency + 低 Recency）**：低价值或已流失客户。"
    )

    st.markdown("### 流失风险预警（高价值客户）")
    freq_max     = int(max(1, np.nanmax(rfm["frequency"])))
    default_freq = int(max(1, np.nanpercentile(rfm["frequency"], 90)))
    freq_threshold = st.slider("高价值阈值：Frequency（重复购买次数）下限", min_value=0, max_value=freq_max, value=min(default_freq, freq_max), step=1)
    alive_threshold = st.slider("流失预警阈值：P(Alive) 上限（越低越危险）", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    palive_np = bgf.conditional_probability_alive(rfm["frequency"], rfm["recency"], rfm["T"])
    rfm["p_alive"] = pd.Series(palive_np, index=rfm.index, name="p_alive")

    tx = events[events["event"] == "transaction"].copy()
    tx["transaction_dt"] = pd.to_datetime(tx["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    last_buy = tx.groupby("visitorid")["transaction_dt"].max().rename("last_transaction_dt")

    alert = rfm.join(last_buy).reset_index().rename(columns={"index": "visitorid"})
    alert = alert[(alert["frequency"] >= freq_threshold) & (alert["p_alive"] < alive_threshold)].copy()
    alert["churn_risk"] = 1.0 - alert["p_alive"]
    alert = alert.sort_values(["churn_risk", "frequency"], ascending=[False, False])

    st.caption("筛选逻辑：历史贡献高（Frequency 高）但存活概率低（P(Alive) < 阈值）。")
    st.dataframe(alert[["visitorid", "frequency", "last_transaction_dt", "p_alive", "churn_risk"]].head(200), use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Streamlit 页面布局
# ──────────────────────────────────────────────────────────────────────────────

def render_intent_prediction_module() -> None:
    st.subheader("模块一：实时意图预测")
    st.markdown("本页面基于GBT模型，将用户的实时行为特征映射到与训练阶段完全一致的特征空间，并给出购买意向概率及营销建议。")

    try:
        model, feature_names, scaler = load_model_and_preprocess()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    st.sidebar.header("User Behavior Input")
    st.sidebar.subheader("Behavior Characteristics")
    st.sidebar.caption("Based on the current session's browsing frequency, purchase attempts, and duration of stay, it reflects the immediate intention of this visit.")
    sess_view_cnt = st.sidebar.number_input("浏览次数（view 数）", min_value=0, value=5, step=1)
    sess_cart_cnt = st.sidebar.number_input("加购次数（addtocart 数）", min_value=0, value=1, step=1)
    sess_duration_sec = st.sidebar.number_input("会话持续时间（秒）", min_value=0.0, value=300.0, step=10.0)
    sess_unique_items = st.sidebar.number_input("本次会话触达的不同商品数", min_value=0, value=3, step=1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("RFM / 用户历史特征")
    st.sidebar.caption("结合用户的历史活跃度与加购偏好，刻画长期价值。")
    user_recency_hours = st.sidebar.number_input("距上次访问时间（小时，Recency）", min_value=0.0, value=12.0, step=1.0)
    user_freq_total = st.sidebar.number_input("历史总行为次数（Frequency）", min_value=0.0, value=50.0, step=1.0)
    user_cart_freq = st.sidebar.number_input("历史加购次数（Cart Frequency）", min_value=0.0, value=10.0, step=1.0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("时间衰减相关（Time-decay）")
    st.sidebar.caption("使用公式 w = exp(−λΔt) 对近期行为赋予更高权重，与训练阶段完全同步。")
    recent_view_hours = st.sidebar.number_input("最近一次浏览距今时间（小时）", min_value=0.0, value=1.0, step=0.5)
    recent_cart_hours = st.sidebar.number_input("最近一次加购距今时间（小时）", min_value=0.0, value=2.0, step=0.5)

    st.sidebar.markdown("---")
    st.sidebar.subheader("行为类型（当前会话）")
    behavior_type_input = st.sidebar.radio(
        "本次会话行为模式",
        options=["focused", "explorer", "normal"],
        format_func=lambda x: {"focused": "类目内反复比较", "explorer": "跨类目跳跃", "normal": "普通浏览"}.get(x, x),
        index=2,
    )
    st.sidebar.caption("可根据当前会话实际情况手动选择，或运行 prepare_behavior_cache.py 自动识别。")

    st.sidebar.markdown("---")
    st.sidebar.subheader("类别偏好（可选）")
    st.sidebar.caption("刻画用户对不同商品类别的广度与集中程度。")
    user_cat_breadth = st.sidebar.number_input("历史涉猎的商品类别数（Breadth）", min_value=0.0, value=5.0, step=1.0)
    user_cat_concentration = st.sidebar.slider("类别集中度（0~1，越高越集中）", min_value=0.0, max_value=1.0, value=0.4, step=0.05)

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
        st.subheader("实时购买意向预测")
        if st.button("立即预测", type="primary"):
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
            st.info("在左侧填写用户的实时行为特征，然后点击上方“立即预测”按钮以获取结果。")

    with col_right:
        if st.session_state["m1_prob"] is not None:
            render_behavior_marketing(st.session_state["m1_prob"], st.session_state["m1_behavior"])
        else:
            st.markdown("<div style='margin-top:60px'></div>", unsafe_allow_html=True)
            st.markdown("### 精细营销建议")
            st.info("点击「立即预测」后，此处将显示精细化营销建议。")

def main():
    st.set_page_config(
        page_title="System for User Purchase Behavior Analysis and Prediction",
        page_icon=None,
        layout="wide",
    ) # 注入自定义 CSS，修改加减号按钮的颜色为复古金 #E2B659
    st.markdown(
        """
        <style>
        /* 针对所有 number_input 的加号和减号按钮 */
        [data-testid="stNumberInputStepUp"] svg,
        [data-testid="stNumberInputStepDown"] svg {
            fill: #E2B659 !important;
            color: #E2B659 !important;
        }
        
        /* 针对按钮本身的文字/图标颜色 fallback */
        [data-testid="stNumberInputStepUp"],
        [data-testid="stNumberInputStepDown"] {
            color: #E2B659 !important;
        }
        
        /* 鼠标悬浮时的效果（可选，稍微提亮一点点以增加互动感） */
        [data-testid="stNumberInputStepUp"]:hover svg,
        [data-testid="stNumberInputStepDown"]:hover svg {
            fill: #F3C76A !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 在这里注入 CSS 主题
    apply_custom_theme()

    st.title("Analysis and Prediction of User Purchase Behaviors")
    st.sidebar.markdown("### Modules")
    _module_options = [
        "模块一：实时意图预测",
        "模块二：客户长期价值分析（BG/NBD）",
        "模块三：智能推荐引擎",
    ]
    if "jump_to_module" in st.session_state:
        st.session_state["_module_radio"] = st.session_state.pop("jump_to_module")
    module = st.sidebar.radio(
        "Select", options=_module_options, index=0, key="_module_radio", label_visibility="collapsed",
    )

    if module == "模块一：实时意图预测":
        render_intent_prediction_module()
    elif module == "模块二：客户长期价值分析（BG/NBD）":
        render_bgnbd_module(data_dir=".")
    else:
        render_reco_module(data_dir=".")

if __name__ == "__main__":
    main()
