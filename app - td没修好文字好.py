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
# 1. 模型与特征变换加载（与训练阶段完全同步）
# ──────────────────────────────────────────────────────────────────────────────


@st.cache_resource
def load_model_and_preprocess() -> Tuple[object, List[str], object]:
    """
    加载已训练好的冠军模型（LightGBM GBT）以及
    训练阶段使用的 feature_names 与 StandardScaler。
    """
    # 尝试两种常见路径，兼容你当前目录结构
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
    将侧边栏中的原始输入转换为与训练阶段完全一致的特征向量，
    包括：
      - Session 特征
      - RFM / 用户历史特征
      - 时间衰减特征（w = exp(-λ * Δt)）
    最终输出经过 StandardScaler 归一化后的 1×N 向量。
    """
    # Session 级特征
    sess_view_cnt = inputs["sess_view_cnt"]
    sess_cart_cnt = inputs["sess_cart_cnt"]
    sess_duration_sec = inputs["sess_duration_sec"]
    sess_unique_items = inputs["sess_unique_items"]

    # 会话内加购/浏览比率（与 preprocess.build_features 中定义一致）
    sess_cart_view_ratio = sess_cart_cnt / (sess_view_cnt + 1.0)

    # 用户历史 + RFM 特征
    # recency 以“秒”为单位，侧栏中以“小时”输入
    user_recency_sec = inputs["user_recency_hours"] * 3600.0
    user_freq_total = inputs["user_freq_total"]
    user_cart_freq = inputs["user_cart_freq"]

    # 时间衰减特征：使用与训练阶段相同的公式
    #   w = exp(-LAMBDA_DECAY * Δt)
    # 这里 Δt 由用户输入的“最近行为距今的时间（小时）”转换为秒
    recent_view_delta_sec = inputs["recent_view_hours"] * 3600.0
    recent_cart_delta_sec = inputs["recent_cart_hours"] * 3600.0
    user_decayed_view = sess_view_cnt * np.exp(-LAMBDA_DECAY * recent_view_delta_sec)
    user_decayed_cart = sess_cart_cnt * np.exp(-LAMBDA_DECAY * recent_cart_delta_sec)

    # 类别偏好特征（如果没有真实统计，可由业务人员根据经验输入，默认 0）
    user_cat_breadth = inputs.get("user_cat_breadth", 0.0)
    user_cat_concentration = inputs.get("user_cat_concentration", 0.0)

    # 组装与训练阶段一模一样的特征字典
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

    # 按 feature_names 的顺序组装特征向量，保持与训练时 100% 一致
    vector = np.array([[float(raw_feat.get(name, 0.0)) for name in feature_names]], dtype=float)

    # 使用训练时的 StandardScaler 做同样的归一化
    vector_scaled = scaler.transform(vector)
    return vector_scaled


def predict_proba_single(model, X_scaled: np.ndarray) -> float:
    """对单个样本进行预测，返回购买意向为 1 的概率。"""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0, 1]
    else:
        # 保险起见，兼容无 predict_proba 的模型
        if hasattr(model, "decision_function"):
            score = model.decision_function(X_scaled)[0]
            proba = 1.0 / (1.0 + np.exp(-score))
        else:
            proba = float(model.predict(X_scaled)[0])
    return float(proba)


def render_gauge(prob: float) -> None:
    """使用 Plotly 绘制彩色仪表盘，展示购买意向概率。"""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100.0,
            number={"suffix": "%", "font": {"size": 32}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#666"},
                "bar": {"color": "#4C6EF5"},
                "steps": [
                    {"range": [0, 40], "color": "#E3F2FD"},
                    {"range": [40, 70], "color": "#BBDEFB"},
                    {"range": [70, 85], "color": "#90CAF9"},
                    {"range": [85, 100], "color": "#64B5F6"},
                ],
                "threshold": {
                    "line": {"color": "#E03131", "width": 4},
                    "thickness": 0.75,
                    "value": prob * 100.0,
                },
            },
            title={"text": "Predicted Purchase Intent", "font": {"size": 20}},
        )
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=80, b=40),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def marketing_suggestion(prob: float) -> str:
    """根据预测概率给出简单的营销建议。"""
    if prob >= 0.8:
        return "高购买意向（>80%）：建议立即发放有吸引力的优惠券或限时折扣，引导下单。"
    if prob >= 0.5:
        return "中等购买意向（50%~80%）：可推荐相关商品或适度优惠，进一步激活购买兴趣。"
    if prob >= 0.3:
        return "偏低购买意向（30%~50%）：建议通过内容种草、展示评价等方式提升信任度。"
    return "极低购买意向（<30%）：建议以品牌曝光为主，可减少激进促销策略。"


# ──────────────────────────────────────────────────────────────────────────────
# 模块二：BG/NBD 客户长期价值分析
# ──────────────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def load_clean_events_for_bgnbd(data_dir: str = ".") -> pd.DataFrame:
    """
    读取并清洗 events.csv，确保与 preprocess.py 的加载/清洗逻辑一致。
    仅返回清洗后的 events（含 ts_sec）。
    """
    events, _, _ = load_data(data_dir=data_dir)
    events = clean_events(events)
    return events

@st.cache_data(show_spinner=False)
def build_rfm_from_transactions(events: pd.DataFrame) -> pd.DataFrame:
    """
    使用 lifetimes 将交易序列转为 RFM（frequency, recency, T）。
    frequency 为“重复购买次数”（总购买次数 - 1）。
    """
    try:
        from lifetimes.utils import summary_data_from_transaction_data
    except Exception as e:  # pragma: no cover
        raise ImportError("缺少依赖 lifetimes。请先安装：pip install lifetimes") from e

    tx = events[events["event"] == "transaction"].copy()
    if tx.empty:
        return pd.DataFrame(columns=["frequency", "recency", "T"])

    tx["transaction_dt"] = pd.to_datetime(tx["timestamp"], unit="ms", utc=True).dt.tz_convert(None)

    # 去重，避免同一交易被多次计数（兼容 transactionid 为空的情况）
    dedup_cols = ["visitorid", "transactionid", "transaction_dt"]
    dedup_cols = [c for c in dedup_cols if c in tx.columns]
    if dedup_cols:
        tx = tx.drop_duplicates(dedup_cols)

    rfm = summary_data_from_transaction_data(
        transactions=tx,
        customer_id_col="visitorid",
        datetime_col="transaction_dt",
        freq="D",
    )
    rfm["frequency"] = rfm["frequency"].clip(lower=0)
    return rfm


@st.cache_resource(show_spinner=False)
def fit_bgnbd_model(rfm: pd.DataFrame):
    try:
        from lifetimes import BetaGeoFitter
    except Exception as e:  # pragma: no cover
        raise ImportError("缺少依赖 lifetimes。请先安装：pip install lifetimes") from e

    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(rfm["frequency"], rfm["recency"], rfm["T"])
    return bgf


@st.cache_resource(show_spinner=False)
def fit_bgnbd_model_default(data_dir: str = "."):
    """
    默认数据集（本地 events.csv）专用：缓存已拟合的 BG/NBD 模型，
    避免页面切换或刷新时重复训练。
    """
    events = load_clean_events_for_bgnbd(data_dir=data_dir)
    rfm = build_rfm_from_transactions(events)
    if rfm.empty:
        raise ValueError("默认数据集中未检测到 transaction 事件，无法拟合 BG/NBD 模型。")
    return fit_bgnbd_model(rfm)


def _standardize_events_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    尝试将上传数据集列名适配为 visitorid / event / timestamp。
    返回 (df, warnings)。
    """
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
        "高端电子产品",
        "时尚家居用品",
        "美妆个护精选",
        "运动户外装备",
        "母婴儿童用品",
        "食品饮料甄选",
        "图书文娱周边",
        "办公学习用品",
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
        "高端电子产品": "electronics",
        "时尚家居用品": "home",
        "美妆个护精选": "beauty",
        "运动户外装备": "sport",
        "母婴儿童用品": "baby",
        "食品饮料甄选": "food",
        "图书文娱周边": "book",
        "办公学习用品": "office",
    }
    file_prefix = prefix_to_file.get(prefix, "office")

    # 为了避免“同类 5 个推荐卡片出现重复图”，图片索引仅使用推荐位置（0~4）。
    # 这样即使 5 个商品属于同一类别，也会稳定展示该类别下 5 张不同素材图。
    idx = int(position_index) % 5

    image_path = os.path.join("assets", f"{file_prefix}_{idx}.jpg")
    fallback_path = os.path.join("assets", "default_icon.svg")
    if not os.path.exists(image_path):
        image_path = fallback_path

    return {"image_path": image_path, "name": name, "price": f"¥{price:,.2f}"}

@st.cache_data(show_spinner=False)
def load_events_for_reco_default(data_dir: str = ".") -> pd.DataFrame:
    """默认数据集：加载并清洗 events.csv（复用 preprocess.py 逻辑），用于推荐模块。"""
    events, _, _ = load_data(data_dir=data_dir)
    events = clean_events(events)
    # 推荐模块只关心有 itemid 的行为
    if "itemid" in events.columns:
        events["itemid"] = events["itemid"].astype(str)
    events["visitorid"] = events["visitorid"].astype(str)
    return events


@st.cache_data(show_spinner=False)
def load_item_category_latest(data_dir: str = ".") -> pd.Series:
    """
    合并 item_properties_part1/2，并提取每个 item 最新的 categoryid。
    仅保留每个物品的最新 categoryid（性能优化）。
    返回：index=itemid, value=categoryid
    """
    p1 = os.path.join(data_dir, "item_properties_part1.csv")
    p2 = os.path.join(data_dir, "item_properties_part2.csv")
    frames = []
    if os.path.exists(p1):
        frames.append(pd.read_csv(p1, dtype={"itemid": str}))
    if os.path.exists(p2):
        frames.append(pd.read_csv(p2, dtype={"itemid": str}))
    if not frames:
        return pd.Series(dtype="object")

    props = pd.concat(frames, ignore_index=True)
    needed = {"itemid", "property", "value"}
    if not needed.issubset(props.columns):
        return pd.Series(dtype="object")

    cat = props[props["property"] == "categoryid"][["itemid", "value"] + (["timestamp"] if "timestamp" in props.columns else [])].copy()
    if cat.empty:
        return pd.Series(dtype="object")

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
    """
    预计算：
      - item->next_item 概率表
      - category->next_category 概率表
      - category->top_items（用于将类别关联落到具体物品）
      - 全局热门 items（兜底）
    """
    df = events[["visitorid", "ts_sec", "itemid"]].dropna().copy()
    df["visitorid"] = df["visitorid"].astype(str)
    df["itemid"] = df["itemid"].astype(str)
    df = df.sort_values(["visitorid", "ts_sec"])

    df["next_itemid"] = df.groupby("visitorid")["itemid"].shift(-1)
    df["next_ts"] = df.groupby("visitorid")["ts_sec"].shift(-1)
    df["dt"] = df["next_ts"] - df["ts_sec"]
    df = df.dropna(subset=["next_itemid", "dt"])
    df = df[(df["dt"] > 0) & (df["dt"] <= float(SESSION_GAP_SEC))]

    # item -> next_item 概率
    pair = df.groupby(["itemid", "next_itemid"]).size().rename("cnt").reset_index()
    pair["prob"] = pair["cnt"] / pair.groupby("itemid")["cnt"].transform("sum")
    pair = pair.sort_values(["itemid", "prob"], ascending=[True, False])
    item_next: Dict[str, List[Tuple[str, float]]] = (
        pair.groupby("itemid")
        .head(topk)
        .groupby("itemid")[["next_itemid", "prob"]]
        .apply(lambda x: list(map(tuple, x.values.tolist())))
        .to_dict()
    )

    # category -> next_category 概率
    cat_series = item_to_cat
    df["cat"] = df["itemid"].map(cat_series)
    df["next_cat"] = df["next_itemid"].map(cat_series)
    cat_df = df.dropna(subset=["cat", "next_cat"])
    cat_pair = cat_df.groupby(["cat", "next_cat"]).size().rename("cnt").reset_index()
    cat_pair["prob"] = cat_pair["cnt"] / cat_pair.groupby("cat")["cnt"].transform("sum")
    cat_pair = cat_pair.sort_values(["cat", "prob"], ascending=[True, False])
    cat_next: Dict[str, List[Tuple[str, float]]] = (
        cat_pair.groupby("cat")
        .head(topk)
        .groupby("cat")[["next_cat", "prob"]]
        .apply(lambda x: list(map(tuple, x.values.tolist())))
        .to_dict()
    )

    # category -> top items（按事件频次）
    tmp = df[["itemid"]].copy()
    tmp["cat"] = df["itemid"].map(cat_series)
    tmp = tmp.dropna(subset=["cat"])
    item_pop = tmp.groupby(["cat", "itemid"]).size().rename("cnt").reset_index()
    item_pop = item_pop.sort_values(["cat", "cnt"], ascending=[True, False])
    cat_to_items: Dict[str, List[str]] = (
        item_pop.groupby("cat").head(200).groupby("cat")["itemid"].apply(list).to_dict()
    )

    # 全局热门兜底
    global_hot = (
        events.dropna(subset=["itemid"])["itemid"].astype(str).value_counts().head(500).index.tolist()
    )

    return item_next, cat_next, cat_to_items, global_hot


def _softmax_top(scores: Dict[str, float], topn: int = 5) -> List[Tuple[str, float]]:
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: max(topn, 1)]
    if not items:
        return []
    vals = np.array([max(v, 0.0) for _, v in items], dtype=float)
    if vals.sum() <= 0:
        probs = np.ones_like(vals) / len(vals)
    else:
        ex = np.exp(vals - vals.max())
        probs = ex / ex.sum()
    return [(items[i][0], float(probs[i])) for i in range(len(items))]


# ── Offline cache loaders ─────────────────────────────────────────────────────

RECO_CACHE_DIR = "reco_cache"


@st.cache_resource(show_spinner=False)
def load_reco_cache() -> dict:
    """
    Load all offline-precomputed artefacts from reco_cache/ into memory once.
    Returns dict with keys: precomputed_recs, transitions, active_users, top_users.
    Missing files return safe empty defaults so the app never hard-crashes.
    """
    result: dict = {
        "precomputed_recs": {},
        "transitions":      {},
        "active_users":     np.array([], dtype=object),
        "top_users":        np.array([], dtype=object),
    }
    recs_path   = os.path.join(RECO_CACHE_DIR, "precomputed_recs.pkl")
    trans_path  = os.path.join(RECO_CACHE_DIR, "transitions.pkl")
    active_path = os.path.join(RECO_CACHE_DIR, "active_users.npy")
    top_path    = os.path.join(RECO_CACHE_DIR, "top_users.npy")

    if os.path.exists(recs_path):
        with open(recs_path, "rb") as f:
            result["precomputed_recs"] = pickle.load(f)
    if os.path.exists(trans_path):
        with open(trans_path, "rb") as f:
            result["transitions"] = pickle.load(f)
    if os.path.exists(active_path):
        result["active_users"] = np.load(active_path, allow_pickle=True)
    if os.path.exists(top_path):
        result["top_users"] = np.load(top_path, allow_pickle=True)
    return result


def recommend_td_multifaceted_fpmc(
    visitorid: str,
    events: pd.DataFrame,
    item_to_cat: pd.Series,
    item_next: Dict[str, List[Tuple[str, float]]],
    cat_next: Dict[str, List[Tuple[str, float]]],
    cat_to_items: Dict[str, List[str]],
    global_hot: List[str],
    k: int = 5,
    alpha: float = 0.25,
    beta: float = 0.55,
    gamma: float = 0.20,
    use_time_decay: bool = True,
) -> List[Tuple[str, str, float]]:
    """
    TD-Multifaceted-FPMC approximation:
      Score = alpha * user_preference + beta * time_weighted_transition + gamma * category_affinity
    Returns top-k as [(itemid, reason, prob)].
    """
    u = str(visitorid).strip()
    u_ev = events[events["visitorid"].astype(str) == u].dropna(subset=["itemid"]).copy()
    if u_ev.empty:
        return [(it, "hot", 0.2) for it in global_hot[:k]]

    u_ev = u_ev.sort_values("ts_sec")
    last3  = u_ev.tail(3)[["itemid", "ts_sec"]].values.tolist()
    t_last = float(last3[-1][1])

    # User preference: decayed category frequency over last 200 actions
    hist = u_ev.tail(200)[["itemid", "ts_sec"]].copy()
    hist["cat"] = hist["itemid"].astype(str).map(item_to_cat)
    hist = hist.dropna(subset=["cat"])
    if not hist.empty:
        hist["w"] = np.exp(-LAMBDA_DECAY * (t_last - hist["ts_sec"].astype(float))) if use_time_decay else 1.0
        pref = hist.groupby("cat")["w"].sum()
        pref = pref / (pref.sum() + 1e-12)
    else:
        pref = pd.Series(dtype=float)

    seen: set = set(u_ev["itemid"].astype(str).unique().tolist())
    scores:  Dict[str, float] = {}
    reasons: Dict[str, str]   = {}

    # 1) Time-weighted item->next_item transitions
    for item_i, ts_i in last3:
        item_i = str(item_i)
        w = float(np.exp(-LAMBDA_DECAY * float(t_last - float(ts_i)))) if use_time_decay else 1.0
        for nxt, p in item_next.get(item_i, [])[:80]:
            if nxt in seen:
                continue
            scores[nxt] = scores.get(nxt, 0.0) + beta * w * float(p)
            reasons.setdefault(nxt, "seq")

    # 2) Category->next_category affinity, mapped to concrete items
    for item_i, ts_i in last3:
        item_i = str(item_i)
        cat_i  = item_to_cat.get(item_i)
        if cat_i is None or (isinstance(cat_i, float) and np.isnan(cat_i)):
            continue
        w = float(np.exp(-LAMBDA_DECAY * float(t_last - float(ts_i)))) if use_time_decay else 1.0
        for next_cat, p in cat_next.get(str(cat_i), [])[:20]:
            cand = cat_to_items.get(str(next_cat), [])[:80]
            if not cand:
                continue
            share = float(p) / max(len(cand), 1)
            for it in cand:
                if it in seen:
                    continue
                scores[it] = scores.get(it, 0.0) + gamma * w * share
                if reasons.get(it) != "seq":
                    reasons.setdefault(it, "cat")

    # 3) Boost candidates whose category matches user preference
    if not pref.empty and scores:
        for it in list(scores.keys()):
            cat = item_to_cat.get(str(it))
            if cat is None or (isinstance(cat, float) and np.isnan(cat)):
                continue
            scores[it] += alpha * float(pref.get(str(cat), 0.0))

    # Fallback: pad with global popular items if too few candidates
    if len(scores) < k:
        for it in global_hot:
            if it in seen or it in scores:
                continue
            scores[it] = 1e-6
            reasons.setdefault(it, "hot")
            if len(scores) >= k * 3:
                break

    top = _softmax_top(scores, topn=k)
    return [(it, {"seq":"seq","cat":"cat"}.get(reasons.get(it,"hot"),"hot"), float(p)) for it, p in top][:k]


@st.cache_data(show_spinner=False)
def _run_realtime_reco(
    visitorid: str,
    events_hash: int,
    events: pd.DataFrame,
    item_to_cat: pd.Series,
    item_next_pkl: bytes,
    cat_next_pkl: bytes,
    cat_to_items_pkl: bytes,
    global_hot_tuple: tuple,
    alpha: float,
    beta: float,
    gamma: float,
    use_time_decay: bool,
) -> List[Tuple[str, str, float]]:
    """
    @st.cache_data wrapper around recommend_td_multifaceted_fpmc.
    Serialised dict bytes are used as hashable cache keys so repeated
    sidebar interactions for the same user ID return instantly.
    """
    item_next    = pickle.loads(item_next_pkl)
    cat_next     = pickle.loads(cat_next_pkl)
    cat_to_items = pickle.loads(cat_to_items_pkl)
    global_hot   = list(global_hot_tuple)
    return recommend_td_multifaceted_fpmc(
        visitorid=visitorid,
        events=events,
        item_to_cat=item_to_cat,
        item_next=item_next,
        cat_next=cat_next,
        cat_to_items=cat_to_items,
        global_hot=global_hot,
        k=5,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        use_time_decay=use_time_decay,
    )


def _render_rec_cards(
    recs: List[Tuple[str, str, float]],
    item_to_cat: pd.Series,
) -> None:
    """Render 5 recommendation product cards side-by-side."""
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            if i >= len(recs):
                st.caption("—")
                continue
            itemid, reason, prob = recs[i]
            label   = {"seq": "Seq", "cat": "Cat"}.get(reason, "Hot")
            details = get_item_details(itemid, item_to_cat=item_to_cat, position_index=i)
            with st.container(border=True):
                img = details["image_path"]
                if os.path.exists(img):
                    st.image(img, use_container_width=True)
                else:
                    st.caption("No image")
                st.markdown(f"**{details['name']}**")
                st.caption(details["price"])
                st.caption(f"{label} · {prob:.2%}")


def render_reco_module(data_dir: str = ".") -> None:
    st.subheader("Module 3: Intelligent Recommendation Engine")

    # ── Load offline cache once (@st.cache_resource) ──────────────────────────
    cache            = load_reco_cache()
    precomputed_recs = cache["precomputed_recs"]
    transitions      = cache["transitions"]
    top_users_arr    = cache["top_users"]
    top_users: List[str] = [str(u) for u in top_users_arr.tolist()]
    has_cache = bool(precomputed_recs) and bool(transitions)

    # ── Sidebar: mode switch ───────────────────────────────────────────────────
    st.sidebar.markdown("### Recommendation Mode")
    user_mode = st.sidebar.radio(
        label="user_source",
        options=["Example User", "Manual Input"],
        index=0,
        label_visibility="collapsed",
    )

    # User input immediately below the mode radio
    if user_mode == "Example User":
        if not top_users:
            top_users = list(precomputed_recs.keys())
        visitorid = st.sidebar.selectbox("Select Example User", options=top_users)
    else:
        visitorid = st.sidebar.text_input("Manual Input").strip()

    st.sidebar.markdown("---")
    algo_mode = st.sidebar.radio(
        "Algorithm Version",
        options=["FPMC", "TD-Multifaceted-FPMC"],
        index=1,
    )
    is_basic       = algo_mode == "FPMC"
    alpha          = 0.0  if is_basic else 0.25
    gamma          = 0.0  if is_basic else 0.20
    use_time_decay = not is_basic

    # ═══════════════════════════════════════════════════════════════════════════
    # MODE A — Pre-warmed Examples: instant result from precomputed_recs.pkl
    # ═══════════════════════════════════════════════════════════════════════════
    if user_mode == "Example User":
        if not has_cache:
            st.warning(
                "No offline cache found in `reco_cache/`. "
                "Run `python prepare_reco_data.py` first, then restart the app."
            )
            return

        # Instant — no algorithm call ──────────────────────────────────────────
        recs: List[Tuple[str, str, float]] = precomputed_recs.get(str(visitorid), [])
        if not recs:
            st.info(f"No precomputed result for user `{visitorid}`. Try another.")
            return

        item_to_cat = load_item_category_latest(data_dir=data_dir)
        top1_prob   = float(recs[0][2])

        st.markdown(f"**User ID:** `{visitorid}`  ·  Source: offline cache")
        col_m, col_p = st.columns([1, 3])
        col_m.metric("Top-1 Confidence", f"{top1_prob:.2%}")
        col_p.progress(min(max(top1_prob, 0.0), 1.0))
        _render_rec_cards(recs, item_to_cat)

    # ═══════════════════════════════════════════════════════════════════════════
    # MODE B — Manual Input: chunked CSV read + real-time FPMC inference
    # ═══════════════════════════════════════════════════════════════════════════
    else:
        if not visitorid:
            st.info("Enter a Visitor ID in the sidebar to get real-time recommendations.")
            return

        item_to_cat = load_item_category_latest(data_dir=data_dir)

        # Use cached transitions if available; otherwise build on-the-fly once
        if has_cache:
            item_next    = transitions["item_next"]
            cat_next     = transitions["cat_next"]
            cat_to_items = transitions["cat_to_items"]
            global_hot   = transitions["global_hot"]
        else:
            with st.spinner("Building transition tables (first run only)..."):
                events_full = load_events_for_reco_default(data_dir=data_dir)
                item_next, cat_next, cat_to_items, global_hot = build_transition_tables(
                    events_full, item_to_cat
                )

        # Chunked read: load only this visitor's rows to minimise memory usage
        events_path = os.path.join(data_dir, "events.csv")
        if not os.path.exists(events_path):
            st.error(f"`events.csv` not found in `{data_dir}`.")
            return

        user_chunks: List[pd.DataFrame] = []
        with st.spinner(f"Fetching events for `{visitorid}`..."):
            for chunk in pd.read_csv(
                events_path,
                chunksize=200_000,
                dtype={"visitorid": str, "itemid": str},
            ):
                mask = chunk["visitorid"] == visitorid
                if mask.any():
                    user_chunks.append(chunk[mask])

        if not user_chunks:
            st.sidebar.error(f"Visitor ID `{visitorid}` not found.")
            return

        from preprocess import clean_events as _clean_ev
        user_events = pd.concat(user_chunks, ignore_index=True)
        user_events = _clean_ev(user_events)
        user_events["itemid"] = user_events["itemid"].astype(str)

        # Pickle tables for @st.cache_data hashability
        item_next_pkl    = pickle.dumps(item_next)
        cat_next_pkl     = pickle.dumps(cat_next)
        cat_to_items_pkl = pickle.dumps(cat_to_items)

        with st.spinner("Running TD-Multifaceted-FPMC..."):
            recs = _run_realtime_reco(
                visitorid        = visitorid,
                events_hash      = hash(visitorid),
                events           = user_events,
                item_to_cat      = item_to_cat,
                item_next_pkl    = item_next_pkl,
                cat_next_pkl     = cat_next_pkl,
                cat_to_items_pkl = cat_to_items_pkl,
                global_hot_tuple = tuple(global_hot),
                alpha            = alpha,
                beta             = 0.55,
                gamma            = gamma,
                use_time_decay   = use_time_decay,
            )

        top1_prob = float(recs[0][2]) if recs else 0.0
        st.markdown(f"**User ID:** `{visitorid}`  ·  Source: real-time inference")
        col_m, col_p = st.columns([1, 3])
        col_m.metric("Top-1 Confidence", f"{top1_prob:.2%}")
        col_p.progress(min(max(top1_prob, 0.0), 1.0))
        _render_rec_cards(recs, item_to_cat)


def render_bgnbd_module(data_dir: str = ".") -> None:
    st.subheader("客户长期价值分析（BG/NBD）")
    st.markdown(
        "将客户历史购买序列转换为 BG/NBD 的 RFM：\n"
        "- **Frequency**：重复购买次数（总购买次数 - 1）\n"
        "- **Recency**：首次购买到最后一次购买的时间跨度\n"
        "- **T**：首次购买到观察窗口结束的时间跨度\n"
        "并计算每位客户的 **P(Alive)**（存活概率）用于流失预警。"
    )

    st.sidebar.markdown("### 数据集加载")
    dataset_mode = st.sidebar.selectbox(
        "选择数据集来源",
        options=["Default dataset (events.csv)", "Upload custom dataset"],
        index=0,
    )

    events: pd.DataFrame | None = None
    if dataset_mode == "Default dataset (events.csv)":
        with st.spinner("正在加载并清洗 events.csv（已启用缓存）..."):
            events = load_clean_events_for_bgnbd(data_dir=data_dir)
    else:
        uploaded = st.sidebar.file_uploader("上传 CSV（需包含 visitorid/event/timestamp 或可自动适配的列）", type=["csv"])
        if uploaded is None:
            st.info("请在侧边栏上传自定义数据集 CSV，然后将自动运行 BG/NBD 分析。")
            return

        try:
            raw = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"CSV 读取失败：{e}")
            return

        raw, warn_msgs = _standardize_events_columns(raw)
        for msg in warn_msgs:
            st.caption(msg)

        missing = [c for c in ["visitorid", "event", "timestamp"] if c not in raw.columns]
        if missing:
            st.error(
                "上传数据集缺少必要列，无法分析。\n\n"
                f"缺少列：{', '.join(missing)}\n"
                "期望列：visitorid, event, timestamp（timestamp 为毫秒或秒级 Unix 时间戳）。"
            )
            return

        # 尝试将 timestamp 统一到毫秒（clean_events 使用毫秒窗口判断）
        raw = raw.copy()
        raw["visitorid"] = raw["visitorid"].astype(str)
        raw["event"] = raw["event"].astype(str)
        raw["timestamp"] = pd.to_numeric(raw["timestamp"], errors="coerce")
        raw = raw.dropna(subset=["visitorid", "event", "timestamp"])

        ts_med = float(np.nanmedian(raw["timestamp"].values))
        if ts_med < 1e12:  # 大概率是秒级
            raw["timestamp"] = raw["timestamp"] * 1000.0
            st.caption("检测到 timestamp 可能为秒级，已自动转换为毫秒级后再清洗。")

        with st.spinner("正在清洗上传数据集（与 preprocess.py 一致）..."):
            events = clean_events(raw)

    tx_cnt = int((events["event"] == "transaction").sum())
    st.caption(f"清洗后事件数：{len(events):,}；其中 transaction 事件数：{tx_cnt:,}")

    with st.spinner("正在构建 RFM 数据（已启用缓存）..."):
        rfm = build_rfm_from_transactions(events)

    if rfm.empty:
        st.warning("当前数据中未检测到 transaction 事件，无法进行 BG/NBD 分析。")
        return

    with st.spinner("正在拟合 BG/NBD 模型（已启用缓存）..."):
        if dataset_mode == "Default dataset (events.csv)":
            bgf = fit_bgnbd_model_default(data_dir=data_dir)
        else:
            bgf = fit_bgnbd_model(rfm)

    st.markdown("#### Recency vs Frequency 矩阵热力图")
    try:
        from lifetimes.plotting import plot_frequency_recency_matrix

        plt.figure(figsize=(8, 6))
        plot_frequency_recency_matrix(bgf)
        plt.title("BG/NBD: Expected Transactions (Recency vs Frequency)")
        _, mid, _ = st.columns([2, 3, 2])
        with mid:
            st.pyplot(plt.gcf())
    except Exception as e:
        st.error(f"热力图绘制失败：{e}")

    st.markdown(
        "**物理意义**：\n"
        "- **右上（高 Frequency + 高 Recency）**：核心活跃客户。\n"
        "- **右下（高 Frequency + 低 Recency）**：历史贡献高但近期不活跃，可能流失（重点预警）。\n"
        "- **左上（低 Frequency + 高 Recency）**：新客或偶发客。\n"
        "- **左下（低 Frequency + 低 Recency）**：低价值或已流失客户。"
    )

    st.markdown("#### 流失风险预警（高价值客户）")
    freq_max = int(max(1, np.nanmax(rfm["frequency"])))
    default_freq = int(max(1, np.nanpercentile(rfm["frequency"], 90)))
    freq_threshold = st.slider(
        "高价值阈值：Frequency（重复购买次数）下限",
        min_value=0,
        max_value=freq_max,
        value=min(default_freq, freq_max),
        step=1,
    )
    alive_threshold = st.slider(
        "流失预警阈值：P(Alive) 上限（越低越危险）",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )

    palive_np = bgf.conditional_probability_alive(rfm["frequency"], rfm["recency"], rfm["T"])
    rfm["p_alive"] = pd.Series(palive_np, index=rfm.index, name="p_alive")

    tx = events[events["event"] == "transaction"].copy()
    tx["transaction_dt"] = pd.to_datetime(tx["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    last_buy = tx.groupby("visitorid")["transaction_dt"].max().rename("last_transaction_dt")

    alert = rfm.join(last_buy)
    alert = alert.reset_index().rename(columns={"index": "visitorid"})
    alert = alert[(alert["frequency"] >= freq_threshold) & (alert["p_alive"] < alive_threshold)].copy()
    alert["churn_risk"] = 1.0 - alert["p_alive"]
    alert = alert.sort_values(["churn_risk", "frequency"], ascending=[False, False])

    st.caption("筛选逻辑：历史贡献高（Frequency 高）但存活概率低（P(Alive) < 阈值）。")
    st.dataframe(
        alert[["visitorid", "frequency", "last_transaction_dt", "p_alive", "churn_risk"]].head(200),
        use_container_width=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Streamlit 页面布局
# ──────────────────────────────────────────────────────────────────────────────


def render_intent_prediction_module() -> None:
    st.subheader("模块一：实时意图预测")
    st.markdown("本页面基于GBT模型，将用户的实时行为特征映射到与训练阶段完全一致的特征空间，并给出购买意向概率及营销建议。")

    # 加载模型与特征变换
    try:
        model, feature_names, scaler = load_model_and_preprocess()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    # ── 侧边栏：实时行为特征输入 ──────────────────────────────────────────────
    st.sidebar.header("会话与历史行为输入")

    st.sidebar.subheader("Session 行为特征（当前会话）")
    sess_view_cnt = st.sidebar.number_input("浏览次数（view 数）", min_value=0, value=5, step=1)
    sess_cart_cnt = st.sidebar.number_input("加购次数（addtocart 数）", min_value=0, value=1, step=1)
    sess_duration_sec = st.sidebar.number_input("会话持续时间（秒）", min_value=0.0, value=300.0, step=10.0)
    sess_unique_items = st.sidebar.number_input("本次会话触达的不同商品数", min_value=0, value=3, step=1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("RFM / 用户历史特征")
    user_recency_hours = st.sidebar.number_input(
        "距上次访问时间（小时，Recency）", min_value=0.0, value=12.0, step=1.0
    )
    user_freq_total = st.sidebar.number_input(
        "历史总行为次数（Frequency）", min_value=0.0, value=50.0, step=1.0
    )
    user_cart_freq = st.sidebar.number_input(
        "历史加购次数（Cart Frequency）", min_value=0.0, value=10.0, step=1.0
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("时间衰减相关（Time-decay）")
    recent_view_hours = st.sidebar.number_input(
        "最近一次浏览距今时间（小时）", min_value=0.0, value=1.0, step=0.5
    )
    recent_cart_hours = st.sidebar.number_input(
        "最近一次加购距今时间（小时）", min_value=0.0, value=2.0, step=0.5
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("类别偏好（可选）")
    user_cat_breadth = st.sidebar.number_input(
        "历史涉猎的商品类别数（Breadth）", min_value=0.0, value=5.0, step=1.0
    )
    user_cat_concentration = st.sidebar.slider(
        "类别集中度（0~1，越高越集中）", min_value=0.0, max_value=1.0, value=0.4, step=0.05
    )

    inputs = {
        "sess_view_cnt": float(sess_view_cnt),
        "sess_cart_cnt": float(sess_cart_cnt),
        "sess_duration_sec": float(sess_duration_sec),
        "sess_unique_items": float(sess_unique_items),
        "user_recency_hours": float(user_recency_hours),
        "user_freq_total": float(user_freq_total),
        "user_cart_freq": float(user_cart_freq),
        "recent_view_hours": float(recent_view_hours),
        "recent_cart_hours": float(recent_cart_hours),
        "user_cat_breadth": float(user_cat_breadth),
        "user_cat_concentration": float(user_cat_concentration),
    }

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("实时购买意向预测")
        if st.button("立即预测", type="primary"):
            X_scaled = build_feature_vector_from_inputs(inputs, feature_names, scaler)
            prob = predict_proba_single(model, X_scaled)
            render_gauge(prob)

            st.markdown(f"**预测购买意向概率：{prob * 100:.2f}%**")
            st.markdown(f"**营销建议：** {marketing_suggestion(prob)}")
        else:
            st.info("在左侧填写用户的实时行为特征，然后点击上方“立即预测”按钮以获取结果。")

    with col_right:
        st.subheader("特征说明")
        st.markdown(
            "- **Session 行为特征**：基于当前会话的浏览次数、加购次数、停留时长等，"
            "反映本次访问的即时意图。\n"
            "- **RFM / 用户历史特征**：结合用户的历史活跃度与加购偏好，刻画长期价值。\n"
            "- **时间衰减特征**：使用公式  \n"
            r"  \( w = e^{-\lambda \Delta t} \)  "
            "  对近期行为赋予更高权重，与训练阶段完全同步。\n"
            "- **类别偏好特征**：刻画用户对不同商品类别的广度与集中程度。"
        )


def main():
    st.set_page_config(
        page_title="System for User Purchase Behavior Analysis and Prediction",
        page_icon=None,
        layout="wide",
    )

    st.title("Analysis and Prediction of User Purchase Behaviors")
    st.sidebar.markdown("### Modules")
    module = st.sidebar.radio(
        "Select",
        options=[
            "模块一：实时意图预测",
            "模块二：客户长期价值分析（BG/NBD）",
            "模块三：智能推荐引擎",
        ],
        index=0,
        label_visibility="collapsed",
    )

    if module == "模块一：实时意图预测":
        render_intent_prediction_module()
    elif module == "模块二：客户长期价值分析（BG/NBD）":
        render_bgnbd_module(data_dir=".")
    else:
        render_reco_module(data_dir=".")


if __name__ == "__main__":
    main()

