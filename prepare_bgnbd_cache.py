# =============================================================================
# prepare_bgnbd_cache.py — 模块二离线预计算脚本
# 运行一次即可；之后 app.py 模块二直接加载缓存，无需实时清洗/训练。
#
# 生成文件（保存在 bgnbd_cache/ 目录）：
#   events_cleaned.parquet   — 清洗后的事件表（含 ts_sec）
#   rfm.parquet              — BG/NBD 所需的 RFM 表
#   bgf_params.json          — BetaGeoFitter 拟合参数（4个浮点数）
#   heatmap_matrix.npy       — 热力图矩阵数据（max_frequency × max_recency）
#   heatmap_meta.json        — 热力图轴范围元信息
#   p_alive.parquet          — 每位用户的 P(Alive) 与 last_transaction_dt
#
# 用法：
#   python prepare_bgnbd_cache.py            # 默认 data_dir="."
#   python prepare_bgnbd_cache.py --data .   # 同上
# =============================================================================

import argparse
import json
import os

import numpy as np
import pandas as pd

CACHE_DIR = "bgnbd_cache"


def main(data_dir: str = ".") -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ── 1. 加载 & 清洗 ────────────────────────────────────────────────────────
    print("[1/5] 加载并清洗 events.csv …")
    from preprocess import load_data, clean_events
    events, _, _ = load_data(data_dir=data_dir)
    events = clean_events(events)

    parquet_path = os.path.join(CACHE_DIR, "events_cleaned.parquet")
    events.to_parquet(parquet_path, index=False)
    print(f"      已保存 → {parquet_path}  ({len(events):,} 行)")

    # ── 2. 构建 RFM ───────────────────────────────────────────────────────────
    print("[2/5] 构建 RFM 表 …")
    from lifetimes.utils import summary_data_from_transaction_data

    tx = events[events["event"] == "transaction"].copy()
    tx["transaction_dt"] = pd.to_datetime(
        tx["timestamp"], unit="ms", utc=True
    ).dt.tz_convert(None)

    dedup_cols = [c for c in ["visitorid", "transactionid", "transaction_dt"] if c in tx.columns]
    if dedup_cols:
        tx = tx.drop_duplicates(dedup_cols)

    rfm = summary_data_from_transaction_data(
        transactions=tx,
        customer_id_col="visitorid",
        datetime_col="transaction_dt",
        freq="D",
    )
    rfm["frequency"] = rfm["frequency"].clip(lower=0)

    rfm_path = os.path.join(CACHE_DIR, "rfm.parquet")
    rfm.to_parquet(rfm_path)
    print(f"      已保存 → {rfm_path}  ({len(rfm):,} 用户)")

    # ── 3. 拟合 BG/NBD 模型 ───────────────────────────────────────────────────
    print("[3/5] 拟合 BetaGeoFitter …")
    from lifetimes import BetaGeoFitter

    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(rfm["frequency"], rfm["recency"], rfm["T"])

    # BetaGeoFitter 内部含 lambda，无法直接 pickle。
    # 只保存四个拟合参数，app.py 加载时重建模型对象。
    model_params = {k: float(v) for k, v in bgf.params_.items()}
    model_path = os.path.join(CACHE_DIR, "bgf_params.json")
    with open(model_path, "w") as f:
        json.dump(model_params, f)
    print(f"      已保存 → {model_path}  params={model_params}")

    # ── 4. 预计算热力图矩阵 ───────────────────────────────────────────────────
    # 严格复现 lifetimes.plotting.plot_frequency_recency_matrix 逻辑：
    #   x 轴 = recency（最近一次购买距首次购买的天数 t_x，0 → max_T）
    #   y 轴 = frequency（重复购买次数，1 → max_frequency）
    #   值   = E[下一期购买次数 | frequency=f, recency=t_x, T=max_T]
    # 高频 + 高 recency（右上角）= 活跃核心客户，颜色最深。
    print("[4/5] 预计算 Recency × Frequency 热力图矩阵 …")
    max_frequency = int(rfm["frequency"].max())
    max_T         = int(rfm["T"].max())   # 观察窗口最大天数，作为 T 传入

    # recency 轴：0 → max_T（步长取整，控制矩阵大小不超过 200 列）
    step_r = max(1, max_T // 200)
    step_f = max(1, max_frequency // 100)
    freq_range    = np.arange(1, max_frequency + 1, step_f)   # 从 1 开始，0 次购买无意义
    recency_range = np.arange(0, max_T + 1,         step_r)

    matrix = np.zeros((len(freq_range), len(recency_range)), dtype=float)
    for i, freq in enumerate(freq_range):
        for j, t_x in enumerate(recency_range):
            # t_x <= T 才合法（recency 不能超过观察窗口）
            if t_x > max_T:
                matrix[i, j] = 0.0
                continue
            try:
                val = bgf.conditional_expected_number_of_purchases_up_to_time(
                    1,      # 预测未来 1 个单位时间内的购买次数
                    freq,   # 历史重复购买次数
                    t_x,    # 最近一次购买距首次购买的天数（recency）
                    max_T,  # 观察窗口长度（固定为数据集最大 T）
                )
                matrix[i, j] = float(val)
            except Exception:
                matrix[i, j] = 0.0

    heatmap_path = os.path.join(CACHE_DIR, "heatmap_matrix.npy")
    np.save(heatmap_path, matrix)

    meta = {
        "max_frequency": max_frequency,
        "max_T":         max_T,
        "freq_range":    freq_range.tolist(),
        "recency_range": recency_range.tolist(),
    }
    meta_path = os.path.join(CACHE_DIR, "heatmap_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"      已保存 → {heatmap_path}  矩阵尺寸 {matrix.shape}")
    print(f"      已保存 → {meta_path}")

    # ── 5. 预计算 P(Alive) + last_transaction_dt ──────────────────────────────
    print("[5/5] 预计算 P(Alive) …")
    palive = bgf.conditional_probability_alive(
        rfm["frequency"], rfm["recency"], rfm["T"]
    )
    rfm_full = rfm.copy()
    rfm_full["p_alive"] = pd.Series(palive, index=rfm.index)

    last_buy = (
        tx.groupby("visitorid")["transaction_dt"]
        .max()
        .rename("last_transaction_dt")
    )
    rfm_full = rfm_full.join(last_buy)

    palive_path = os.path.join(CACHE_DIR, "p_alive.parquet")
    rfm_full.to_parquet(palive_path)
    print(f"      已保存 → {palive_path}")

    print(f"\n✅ 全部完成，缓存目录：./{CACHE_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预计算模块二 BG/NBD 所需缓存")
    parser.add_argument("--data", default=".", help="events.csv 所在目录，默认当前目录")
    args = parser.parse_args()
    main(data_dir=args.data)
