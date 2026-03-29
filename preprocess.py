# =============================================================================
# preprocess.py — 数据预处理、特征工程与样本均衡模块
# 基于 Retailrocket 电商数据集，结合论文《利用机器学习预测消费者购买意愿》
# =============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 1. 数据加载
# ──────────────────────────────────────────────────────────────────────────────

def load_data(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    从指定目录加载 Retailrocket 数据集三张表。
    返回 (events, item_props, category_tree) 三个 DataFrame。
    """
    print("[1/6] 正在加载原始数据...")

    # events.csv: 每行代表一次用户行为事件
    events = pd.read_csv(
        os.path.join(data_dir, "events.csv"),
        dtype={"visitorid": str, "itemid": str, "transactionid": str},
    )

    # item_properties_part1/2 需要合并；如果只有一份则直接读取
    prop_files = [
        os.path.join(data_dir, "item_properties_part1.csv"),
        os.path.join(data_dir, "item_properties_part2.csv"),
    ]
    existing_props = [f for f in prop_files if os.path.exists(f)]
    if existing_props:
        item_props = pd.concat(
            [pd.read_csv(f, dtype={"itemid": str}) for f in existing_props],
            ignore_index=True,
        )
    else:
        # 尝试单文件
        single = os.path.join(data_dir, "item_properties.csv")
        item_props = pd.read_csv(single, dtype={"itemid": str}) if os.path.exists(single) else pd.DataFrame()

    category_tree = pd.read_csv(
        os.path.join(data_dir, "category_tree.csv"),
        dtype={"categoryid": str, "parentid": str},
    ) if os.path.exists(os.path.join(data_dir, "category_tree.csv")) else pd.DataFrame()

    print(f"    events: {len(events):,} 行  |  item_props: {len(item_props):,} 行  |  category_tree: {len(category_tree):,} 行")
    return events, item_props, category_tree


# ──────────────────────────────────────────────────────────────────────────────
# 2. 适度清洗（保留 view / addtocart 信号）
# ──────────────────────────────────────────────────────────────────────────────

def clean_events(events: pd.DataFrame) -> pd.DataFrame:
    """
    清洗原则（适度）：
      - 去除 visitorid 为空的记录（无法追踪用户行为序列）
      - 去除 timestamp 明显异常（< 0 或超过合理上界）的记录
      - 保留 view / addtocart / transaction 全部事件类型
    """
    print("[2/6] 正在清洗数据...")
    n_before = len(events)

    # 2-a. 去除 visitorid 缺失
    events = events.dropna(subset=["visitorid"])
    events = events[events["visitorid"].str.strip() != ""]

    # 2-b. 时间戳合理性检查（Retailrocket 数据以毫秒为单位的 Unix 时间戳）
    #      合理区间：2015-01-01 ~ 2016-12-31（对应毫秒级别）
    ts_min = 1_420_000_000_000   # 约 2015-01-01
    ts_max = 1_490_000_000_000   # 约 2017-03-20
    events = events[(events["timestamp"] >= ts_min) & (events["timestamp"] <= ts_max)]

    # 2-c. 将毫秒时间戳转换为秒级，方便后续计算时差
    events["ts_sec"] = events["timestamp"] / 1000.0

    print(f"    清洗前: {n_before:,}  →  清洗后: {len(events):,}  (丢弃 {n_before - len(events):,} 行)")
    return events.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Session 切割
# ──────────────────────────────────────────────────────────────────────────────

SESSION_GAP_SEC = 30 * 60  # 30 分钟无活动则认定为新 Session（参考 Arif & Ayazuddin 论文）

def assign_sessions(events: pd.DataFrame) -> pd.DataFrame:
    """
    为每个用户的行为序列分配 Session ID。
    规则：同一用户相邻两次事件时间差 > SESSION_GAP_SEC，则开启新 Session。
    """
    print("[3/6] 正在切割 Session...")

    # 按用户和时间排序，确保行为序列有序
    events = events.sort_values(["visitorid", "ts_sec"]).reset_index(drop=True)

    # 计算同一用户内相邻事件的时间差（跨用户的时间差设为无穷大，强制开启新 Session）
    time_diff = events.groupby("visitorid")["ts_sec"].diff().fillna(np.inf)

    # 时间差超过阈值 → 新 Session 开始，累加得到全局唯一 Session 编号
    new_session_flag = (time_diff > SESSION_GAP_SEC).astype(int)
    events["session_id"] = (events["visitorid"] + "_" + new_session_flag.cumsum().astype(str))

    print(f"    共识别 {events['session_id'].nunique():,} 个 Session")
    return events


# ──────────────────────────────────────────────────────────────────────────────
# 4. 特征工程
# ──────────────────────────────────────────────────────────────────────────────

# 时间衰减参数 λ（越大表示历史衰减越快）
LAMBDA_DECAY = 1.0 / (7 * 24 * 3600)  # 以 7 天为半衰期

def build_features(events: pd.DataFrame, item_props: pd.DataFrame, category_tree: pd.DataFrame) -> pd.DataFrame:
    """
    构造机器学习所需的特征矩阵，每行对应一个 Session。
    特征体系覆盖论文中的三个维度：Session 维度、用户历史维度、类别偏好维度。
    """
    print("[4/6] 正在构造特征矩阵...")

    # ── 4-a. 辅助变量：参考时间点（全数据集最大时间戳，代表"当前时刻"）──
    t_now = events["ts_sec"].max()

    # ── 4-b. Session 级别基础统计（参考论文 Session 维度特征） ──
    # pandas 2.x 兼容写法：先在原始 DataFrame 上创建哑变量列，
    # 再用 groupby().sum() / .agg() 代替 apply(lambda df: ...)，
    # 避免新版 pandas 将整个子 DataFrame 传入 lambda 导致的 KeyError。

    # 为每个事件类型打上 0/1 标记列（向量化操作，无需 apply）
    events["_is_view"]        = (events["event"] == "view").astype(int)
    events["_is_cart"]        = (events["event"] == "addtocart").astype(int)
    events["_is_transaction"] = (events["event"] == "transaction").astype(int)

    sess_grp = events.groupby("session_id")

    # 浏览次数：view 标记列按 session 求和，反映用户的探索程度
    view_cnt = sess_grp["_is_view"].sum().rename("sess_view_cnt")

    # 加购次数：addtocart 标记列按 session 求和，加购是购买意愿的强烈信号
    cart_cnt = sess_grp["_is_cart"].sum().rename("sess_cart_cnt")

    # 会话持续时长（秒）：max(ts) - min(ts)，时长越长通常决策越深入
    duration = sess_grp["ts_sec"].agg(lambda x: x.max() - x.min()).rename("sess_duration_sec")

    # 会话内触达的商品数量：nunique 统计，触达商品越多说明浏览广度越大
    item_cnt = sess_grp["itemid"].nunique().rename("sess_unique_items")

    # 合并 session 级别特征，计算加购-浏览比率（衡量本次会话的决策转化率）
    sess_feat = pd.concat([view_cnt, cart_cnt, duration, item_cnt], axis=1)
    sess_feat["sess_cart_view_ratio"] = sess_feat["sess_cart_cnt"] / (sess_feat["sess_view_cnt"] + 1)

    # ── 4-c. 用户历史维度（BG/NBD 模型思想 + 时间衰减 FPMC） ──
    # 先将 session_id 映射回 visitorid（取每个 session 第一行的 visitorid 即可）
    sess_visitor = sess_grp["visitorid"].first().rename("visitorid").to_frame()
    user_events = events.copy()

    # Recency: 距离当前时刻最近一次行为的时间（秒），越小说明越活跃
    recency = (
        user_events.groupby("visitorid")["ts_sec"]
        .max()
        .rsub(t_now)           # t_now - max_ts = 离现在多久没动
        .rename("user_recency_sec")
    )

    # Frequency: 用户历史行为总次数（反映用户整体活跃度）
    frequency = user_events.groupby("visitorid")["event"].count().rename("user_freq_total")

    # 历史加购频率（纯频率信号）
    cart_freq = (
        user_events[user_events["event"] == "addtocart"]
        .groupby("visitorid")["event"]
        .count()
        .rename("user_cart_freq")
    )

    # ── 时间衰减特征（论文公式：w = exp(-λ * (t_now - t_event))）──
    # 对每个 view 事件赋予衰减权重，再按用户汇总
    user_events["decay_weight"] = np.exp(-LAMBDA_DECAY * (t_now - user_events["ts_sec"]))

    # 衰减加权后的 view 强度：近期浏览加权更高，体现时效性兴趣
    decayed_view = (
        user_events[user_events["event"] == "view"]
        .groupby("visitorid")["decay_weight"]
        .sum()
        .rename("user_decayed_view")
    )

    # 衰减加权后的 addtocart 强度：近期加购权重更高
    decayed_cart = (
        user_events[user_events["event"] == "addtocart"]
        .groupby("visitorid")["decay_weight"]
        .sum()
        .rename("user_decayed_cart")
    )

    # ── 4-d. 类别偏好特征（结合 category_tree） ──
    # 提取商品的类别属性
    cat_feature = pd.DataFrame()
    if not item_props.empty and "property" in item_props.columns and "value" in item_props.columns:
        # 筛选出 categoryid 属性行
        cat_rows = item_props[item_props["property"] == "categoryid"][["itemid", "value"]].copy()
        cat_rows.columns = ["itemid", "categoryid"]
        cat_rows = cat_rows.drop_duplicates("itemid")  # 每件商品取唯一类别

        # 将商品类别 merge 到行为事件中
        merged = user_events.merge(cat_rows, on="itemid", how="left")

        # 用户偏好类别数量（涉猎宽度，越多表示偏好越分散）
        cat_breadth = (
            merged.dropna(subset=["categoryid"])
            .groupby("visitorid")["categoryid"]
            .nunique()
            .rename("user_cat_breadth")
        )

        # 用户最常浏览类别的集中度（Herfindahl 指数 → 越高表示偏好越集中）
        def herfindahl(series):
            counts = series.value_counts(normalize=True)
            return (counts ** 2).sum()  # 类别集中度指数

        cat_conc = (
            merged.dropna(subset=["categoryid"])
            .groupby("visitorid")["categoryid"]
            .agg(herfindahl)
            .rename("user_cat_concentration")
        )
        cat_feature = pd.concat([cat_breadth, cat_conc], axis=1)

    # ── 4-e. 标签定义 (Target) ──
    # y=1：该 Session 内存在 transaction 事件
    # 直接对 _is_transaction 哑变量列求 max（有一个 1 则整个 session 为 1）
    has_transaction = sess_grp["_is_transaction"].max().rename("label")

    # ── 4-f. 合并所有特征到 Session 级别 ──
    df = sess_feat.copy()
    df = df.join(sess_visitor)  # 加入 visitorid

    # 将用户历史特征按 visitorid join 进来
    for feat in [recency, frequency, cart_freq, decayed_view, decayed_cart]:
        df = df.join(feat, on="visitorid")

    if not cat_feature.empty:
        df = df.join(cat_feature, on="visitorid")

    # 加入标签
    df = df.join(has_transaction)

    # 丢弃辅助列 visitorid 和哑变量标记列（非特征列），填充 NaN（部分用户无加购记录）
    drop_cols = ["visitorid", "_is_view", "_is_cart", "_is_transaction"]
    events.drop(columns=[c for c in drop_cols if c in events.columns], inplace=True, errors="ignore")
    df = df.drop(columns=["visitorid"], errors="ignore")
    df = df.fillna(0)

    print(f"    特征矩阵形状: {df.shape}  |  购买样本占比: {df['label'].mean():.4%}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 5. 数据集划分 + SMOTE 样本均衡
# ──────────────────────────────────────────────────────────────────────────────

def split_and_balance(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    将特征矩阵拆分为训练集（80%）和测试集（20%），
    并在训练集上使用 SMOTE 过采样解决严重的类别不均衡问题。

    返回: X_train_bal, X_test, y_train_bal, y_test, feature_names, scaler
    """
    print("[5/6] 正在划分数据集并应用 SMOTE 均衡...")

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values

    # 严格 80/20 划分，stratify 保证两个子集的正负样本比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 标准化（均值为 0，方差为 1），防止 KNN / Logistic Regression 受量纲影响
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SMOTE：在训练集中对少数类（购买=1）合成新样本，直到正负比例接近 1:1
    # k_neighbors=5 为 SMOTE 默认近邻数，sampling_strategy='minority' 仅对少数类过采样
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print(f"    训练集 SMOTE 前: {np.bincount(y_train)}  →  SMOTE 后: {np.bincount(y_train_bal)}")
    print(f"    测试集（不均衡，保持真实分布）: {np.bincount(y_test)}")

    return X_train_bal, X_test, y_train_bal, y_test, feature_cols, scaler


# ──────────────────────────────────────────────────────────────────────────────
# 6. 总入口函数
# ──────────────────────────────────────────────────────────────────────────────

def run_preprocessing(data_dir: str = "data"):
    """
    一键执行完整预处理流程，返回模型所需的所有数据。
    """
    events, item_props, category_tree = load_data(data_dir)
    events = clean_events(events)
    events = assign_sessions(events)
    df = build_features(events, item_props, category_tree)
    X_train, X_test, y_train, y_test, feature_names, scaler = split_and_balance(df)
    print("[6/6] 预处理完成 ✓\n")
    return X_train, X_test, y_train, y_test, feature_names, scaler


if __name__ == "__main__":
    run_preprocessing()


# ──────────────────────────────────────────────────────────────────────────────
# 7. 缓存加速版入口（推荐使用）
# ──────────────────────────────────────────────────────────────────────────────

CACHE_DIR = "cache"
CACHE_FILES = {
    "X_train":       os.path.join(CACHE_DIR, "X_train.npy"),
    "X_test":        os.path.join(CACHE_DIR, "X_test.npy"),
    "y_train":       os.path.join(CACHE_DIR, "y_train.npy"),
    "y_test":        os.path.join(CACHE_DIR, "y_test.npy"),
    "feature_names": os.path.join(CACHE_DIR, "feature_names.npy"),
    "scaler":        os.path.join(CACHE_DIR, "scaler.pkl"),
}


def _cache_exists() -> bool:
    """检查所有缓存文件是否齐全。"""
    return all(os.path.isfile(p) for p in CACHE_FILES.values())


def _save_cache(X_train, X_test, y_train, y_test, feature_names, scaler) -> None:
    """将预处理结果全部持久化到 cache/ 目录。"""
    import pickle
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(CACHE_FILES["X_train"],       X_train)
    np.save(CACHE_FILES["X_test"],        X_test)
    np.save(CACHE_FILES["y_train"],       y_train)
    np.save(CACHE_FILES["y_test"],        y_test)
    # feature_names 是字符串列表，用 allow_pickle 保存
    np.save(CACHE_FILES["feature_names"], np.array(feature_names, dtype=object))
    with open(CACHE_FILES["scaler"], "wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"    缓存已保存到 ./{CACHE_DIR}/ （共 {len(CACHE_FILES)} 个文件）")


def _load_cache():
    """从 cache/ 目录快速恢复预处理结果，通常只需 3~10 秒。"""
    import pickle
    print(f"[CACHE] 检测到缓存，直接加载（跳过全部特征工程）...")
    X_train       = np.load(CACHE_FILES["X_train"])
    X_test        = np.load(CACHE_FILES["X_test"])
    y_train       = np.load(CACHE_FILES["y_train"])
    y_test        = np.load(CACHE_FILES["y_test"])
    feature_names = np.load(CACHE_FILES["feature_names"], allow_pickle=True).tolist()
    with open(CACHE_FILES["scaler"], "rb") as f:
        scaler = pickle.load(f)
    print(f"    X_train: {X_train.shape}  X_test: {X_test.shape}")
    print(f"    y_train 分布: {np.bincount(y_train.astype(int))}")
    print(f"    y_test  分布: {np.bincount(y_test.astype(int))}")
    print("[CACHE] 加载完成 ✓\n")
    return X_train, X_test, y_train, y_test, feature_names, scaler


def run_preprocessing_cached(data_dir: str = "data", force_rebuild: bool = False):
    """
    缓存加速版预处理入口（推荐替代 run_preprocessing）。

    第一次运行：执行完整流程（约 5~15 分钟），结果自动保存到 cache/。
    后续运行：直接从 cache/ 加载 numpy 数组（约 3~10 秒），跳过所有计算。

    参数:
        data_dir      — 原始 CSV 数据目录
        force_rebuild — True 则忽略缓存强制重新计算（数据有更新时使用）
    """
    if not force_rebuild and _cache_exists():
        return _load_cache()

    # 首次运行：完整流程
    if force_rebuild:
        print("[CACHE] force_rebuild=True，忽略缓存，重新计算...")
    else:
        print("[CACHE] 未检测到缓存，开始完整预处理流程...")

    events, item_props, category_tree = load_data(data_dir)
    events = clean_events(events)
    events = assign_sessions(events)
    df     = build_features(events, item_props, category_tree)
    X_train, X_test, y_train, y_test, feature_names, scaler = split_and_balance(df)

    print("[6/6] 预处理完成，正在写入缓存...")
    _save_cache(X_train, X_test, y_train, y_test, feature_names, scaler)
    print("缓存写入完成 ✓\n")
    return X_train, X_test, y_train, y_test, feature_names, scaler
