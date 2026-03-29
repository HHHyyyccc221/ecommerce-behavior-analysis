# =============================================================================
# models.py — 五大分类模型定义、训练、持久化模块
# 包含: KNN / 决策树 / 随机森林 / 逻辑回归 / 梯度提升树(LightGBM)
# =============================================================================

import os
import pickle
import numpy as np
import lightgbm as lgb

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ──────────────────────────────────────────────────────────────────────────────
# 通用持久化工具
# ──────────────────────────────────────────────────────────────────────────────

def save_model(model, path: str) -> None:
    """将已训练的模型序列化保存到磁盘（pickle 格式）。"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"    ✔ 模型已保存 → {path}")


def load_model(path: str):
    """从磁盘加载已序列化的模型，返回模型对象。"""
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"    ✔ 已从缓存加载模型 ← {path}")
    return model


def _model_exists(path: str) -> bool:
    """检查模型文件是否已存在于磁盘。"""
    return os.path.isfile(path)


# ──────────────────────────────────────────────────────────────────────────────
# 1. K-最近邻 (KNN)
# ──────────────────────────────────────────────────────────────────────────────

def train_knn(
    X_train, y_train,
    model_path: str = "saved_models/knn.pkl",
    n_neighbors: int = 15,
) -> KNeighborsClassifier:
    """
    训练 KNN 分类器。
    - n_neighbors=15：经验值，购买意图数据中较大的 k 值有助于降低噪声。
    - weights='distance'：距离越近的邻居贡献越大，符合行为相似度的直觉。
    - metric='euclidean'：欧氏距离（数据已标准化，故适用）。
    - 非迭代模型：fit 后立即保存，无需 early stopping。
    """
    if _model_exists(model_path):
        return load_model(model_path)

    print("  [KNN] 开始训练...")
    # ⚠️ Windows 下 joblib 多线程与 ball_tree 存在兼容性问题，
    #    强制 n_jobs=1 + kd_tree 可彻底规避 parallel.py / _get_outputs 崩溃。
    #    数据集 Session 数量巨大（百万级），KNN 推理本身已很耗时，
    #    多线程反而因线程创建开销和内存竞争拖慢速度。
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights="distance",
        metric="euclidean",
        n_jobs=1,          # 单线程：Windows joblib 兼容性修复
        algorithm="kd_tree",  # kd_tree 在低维特征（<20）下比 ball_tree 更稳定
    )
    model.fit(X_train, y_train)
    save_model(model, model_path)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 2. 决策树 (Decision Tree)
# ──────────────────────────────────────────────────────────────────────────────

def train_decision_tree(
    X_train, y_train,
    model_path: str = "saved_models/decision_tree.pkl",
    max_depth: int = 8,
) -> DecisionTreeClassifier:
    """
    训练决策树分类器。
    - max_depth=8：控制树深度，防止在不均衡数据上过拟合。
    - class_weight='balanced'：自动按类别频率的倒数设置样本权重，
      配合 SMOTE 后的训练集进一步提升对少数类（购买）的关注。
    - min_samples_leaf=20：叶节点最少需含 20 个样本，剪枝防止过拟合。
    - criterion='gini'：基尼不纯度作为分裂准则（计算效率高）。
    """
    if _model_exists(model_path):
        return load_model(model_path)

    print("  [决策树] 开始训练...")
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion="gini",
        class_weight="balanced",
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(X_train, y_train)
    save_model(model, model_path)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 3. 随机森林 (Random Forest)
# ──────────────────────────────────────────────────────────────────────────────

def train_random_forest(
    X_train, y_train,
    model_path: str = "saved_models/random_forest.pkl",
    n_estimators: int = 300,
) -> RandomForestClassifier:
    """
    训练随机森林分类器。
    - n_estimators=300：300 棵树的集成，性能与训练时间之间的合理折衷。
    - max_features='sqrt'：每次分裂仅考虑 sqrt(n_features) 个特征，
      增加树之间的多样性，是随机森林的核心去相关机制。
    - max_depth=12：比单棵决策树允许更深，因为 Bagging 天然抑制过拟合。
    - class_weight='balanced_subsample'：每棵子树独立计算类别权重，
      在 Bootstrap 采样后仍保持对少数类的关注。
    - n_jobs=-1：全核并行训练。
    """
    if _model_exists(model_path):
        return load_model(model_path)

    print("  [随机森林] 开始训练（可能需要 1-3 分钟）...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    save_model(model, model_path)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 4. 逻辑回归 (Logistic Regression)
# ──────────────────────────────────────────────────────────────────────────────

def train_logistic_regression(
    X_train, y_train,
    model_path: str = "saved_models/logistic_regression.pkl",
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> LogisticRegression:
    """
    训练逻辑回归分类器（线性基准模型）。
    - solver='lbfgs'：L-BFGS 拟牛顿法，适合中等规模数据且支持 L2 正则。
    - C=1.0：正则化强度的倒数，C 越小正则化越强（此处保留默认值）。
    - max_iter=1000：最大迭代次数（early stopping 通过 tol 控制）。
    - tol=1e-4（容差）：当两次迭代间损失变化 < tol 时提前退出，
      等效于迭代型模型的早停机制。
    - class_weight='balanced'：应对类别不均衡。
    """
    if _model_exists(model_path):
        return load_model(model_path)

    print("  [逻辑回归] 开始训练...")
    model = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=max_iter,
        tol=tol,                  # ← early stopping 容差
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print(f"    实际迭代次数: {model.n_iter_[0]}")
    save_model(model, model_path)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 5. 梯度提升树 — LightGBM (GBT)
# ──────────────────────────────────────────────────────────────────────────────

def train_gbt(
    X_train, y_train,
    model_path: str = "saved_models/gbt_lgbm.pkl",
    n_estimators: int = 1000,
    early_stopping_rounds: int = 50,
    valid_fraction: float = 0.15,
) -> lgb.LGBMClassifier:
    """
    训练 LightGBM 梯度提升树（论文中的 GBT 实现）。

    Early Stopping 机制：
      - 从训练集中划分 15% 作为内部验证集（eval set），
        用于监控 AUC 变化；
      - 若连续 early_stopping_rounds=50 轮 AUC 不再提升，
        则自动终止训练，防止过拟合并节省时间。

    关键超参数说明：
      - learning_rate=0.05：较小的步长配合 1000 轮迭代，找到更细致的最优解。
      - num_leaves=63：叶子数量（2^6-1），控制模型复杂度。
      - subsample=0.8 / colsample_bytree=0.8：行/列子采样，增加随机性。
      - scale_pos_weight：正负样本比，自动补偿购买事件的稀少性。
      - is_unbalance=True：LightGBM 内置不均衡处理开关。
    """
    if _model_exists(model_path):
        return load_model(model_path)

    print("  [LightGBM GBT] 开始训练（含 Early Stopping）...")

    # 切分内部验证集用于 early stopping（不参与最终评估）
    from sklearn.model_selection import train_test_split as tts
    X_tr, X_val, y_tr, y_val = tts(
        X_train, y_train,
        test_size=valid_fraction,
        random_state=42,
        stratify=y_train,
    )

    # 正负样本比（用于 scale_pos_weight）
    neg, pos = np.bincount(y_tr)
    pos_weight = neg / max(pos, 1)

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,             # -1 表示不限深度，由 num_leaves 间接控制
        subsample=0.8,            # 每轮使用 80% 的样本
        colsample_bytree=0.8,     # 每轮使用 80% 的特征
        reg_alpha=0.1,            # L1 正则化
        reg_lambda=1.0,           # L2 正则化
        scale_pos_weight=pos_weight,
        is_unbalance=True,
        random_state=42,
        n_jobs=-1,
        verbose=-1,               # 关闭 LightGBM 自带的冗余日志
    )

    # 传入验证集，配置 early stopping 回调
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=100),   # 每 100 轮打印一次验证指标
    ]

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",         # 以 AUC 作为早停监控指标
        callbacks=callbacks,
    )

    print(f"    最优迭代轮次: {model.best_iteration_}")
    save_model(model, model_path)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 统一训练接口（main.py 调用入口）
# ──────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "KNN":              (train_knn,                "saved_models/knn.pkl"),
    "决策树":           (train_decision_tree,       "saved_models/decision_tree.pkl"),
    "随机森林":         (train_random_forest,       "saved_models/random_forest.pkl"),
    "逻辑回归":         (train_logistic_regression, "saved_models/logistic_regression.pkl"),
    "GBT (LightGBM)":  (train_gbt,                 "saved_models/gbt_lgbm.pkl"),
}


def train_all_models(X_train, y_train) -> dict:
    """
    依次训练（或从缓存加载）所有五个模型。
    返回 {模型名称: 模型对象} 字典。
    """
    trained = {}
    for name, (train_fn, path) in MODEL_REGISTRY.items():
        print(f"\n{'─'*50}\n▶ {name}")
        trained[name] = train_fn(X_train, y_train, model_path=path)
    return trained
