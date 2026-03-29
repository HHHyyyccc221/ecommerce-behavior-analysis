# =============================================================================
# prepare_behavior_cache.py — 模块一行为类型离线预计算脚本
# 运行一次即可；app.py 模块一直接加载缓存，不产生额外加载时间。
#
# 生成文件（保存在 behavior_cache/ 目录）：
#   session_behavior.parquet  — 每个 session 的行为类型、统计指标
#   user_behavior.parquet     — 每个用户的行为类型汇总（取最近 session）
#
# 行为类型定义：
#   "focused"   — 类目内反复比较（session 内唯一类目数=1，浏览物品数≥3）
#   "explorer"  — 类目间跳跃型（session 内唯一类目数≥3）
#   "normal"    — 其他
#
# 用法：
#   python prepare_behavior_cache.py
#   python prepare_behavior_cache.py --data .
# =============================================================================

import argparse
import os

import numpy as np
import pandas as pd

CACHE_DIR = "behavior_cache"
SESSION_GAP_SEC = 30 * 60  # 与 preprocess.py 保持一致


def load_item_category(data_dir: str) -> pd.Series:
    """合并 item_properties_part1/2，提取每件物品最新的 categoryid。"""
    frames = []
    for fname in ("item_properties_part1.csv", "item_properties_part2.csv"):
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            frames.append(pd.read_csv(fpath, dtype={"itemid": str}))
    if not frames:
        print("  ⚠️  未找到 item_properties 文件，类目信息将缺失。")
        return pd.Series(dtype="object")

    props = pd.concat(frames, ignore_index=True)
    if not {"itemid", "property", "value"}.issubset(props.columns):
        return pd.Series(dtype="object")

    cat = props[props["property"] == "categoryid"][["itemid", "value"]].copy()
    if "timestamp" in props.columns:
        cat = props[props["property"] == "categoryid"][["itemid", "value", "timestamp"]].copy()
        cat["timestamp"] = pd.to_numeric(cat["timestamp"], errors="coerce")
        cat = cat.dropna(subset=["timestamp"]).sort_values("timestamp")
        cat = cat.drop_duplicates("itemid", keep="last")[["itemid", "value"]]

    cat.columns = ["itemid", "categoryid"]
    cat["itemid"] = cat["itemid"].astype(str)
    cat["categoryid"] = cat["categoryid"].astype(str)
    return cat.drop_duplicates("itemid").set_index("itemid")["categoryid"]


def assign_sessions(events: pd.DataFrame) -> pd.DataFrame:
    """为每个用户的行为序列分配 session_id（与 preprocess.py 逻辑一致）。"""
    events = events.sort_values(["visitorid", "ts_sec"]).reset_index(drop=True)
    time_diff = events.groupby("visitorid")["ts_sec"].diff().fillna(np.inf)
    new_flag = (time_diff > SESSION_GAP_SEC).astype(int)
    events["session_id"] = events["visitorid"] + "_" + new_flag.cumsum().astype(str)
    return events


def classify_session(n_cats: int, n_items: int) -> str:
    """根据类目数和物品数判定行为类型。"""
    if n_cats == 1 and n_items >= 3:
        return "focused"    # 类目内反复比较
    if n_cats >= 3:
        return "explorer"   # 类目间跳跃
    return "normal"


def main(data_dir: str = ".") -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ── 1. 加载并清洗 events ─────────────────────────────────────────────────
    print("[1/4] 加载并清洗 events.csv …")
    from preprocess import load_data, clean_events
    events, _, _ = load_data(data_dir=data_dir)
    events = clean_events(events)
    events["visitorid"] = events["visitorid"].astype(str)
    events["itemid"] = events["itemid"].astype(str)
    print(f"      清洗后事件数：{len(events):,}")

    # ── 2. 加载物品-类目映射 ─────────────────────────────────────────────────
    print("[2/4] 加载物品类目映射 …")
    item_to_cat = load_item_category(data_dir)
    events["categoryid"] = events["itemid"].map(item_to_cat)
    cat_coverage = events["categoryid"].notna().mean()
    print(f"      类目覆盖率：{cat_coverage:.1%}")

    # ── 3. 分配 session 并统计行为类型 ──────────────────────────────────────
    print("[3/4] 分配 Session，计算行为类型 …")
    events = assign_sessions(events)

    # 只看 view 事件（浏览行为是判断意图的核心信号）
    view_ev = events[events["event"] == "view"].copy()

    sess_stats = (
        view_ev.groupby("session_id")
        .agg(
            visitorid   =("visitorid",   "first"),
            ts_start    =("ts_sec",      "min"),
            ts_end      =("ts_sec",      "max"),
            n_items     =("itemid",      "nunique"),
            n_cats      =("categoryid",  lambda x: x.dropna().nunique()),
            n_views     =("event",       "count"),
        )
        .reset_index()
    )
    sess_stats["behavior_type"] = sess_stats.apply(
        lambda r: classify_session(int(r["n_cats"]), int(r["n_items"])), axis=1
    )
    sess_stats["duration_sec"] = sess_stats["ts_end"] - sess_stats["ts_start"]

    sess_path = os.path.join(CACHE_DIR, "session_behavior.parquet")
    sess_stats.to_parquet(sess_path, index=False)
    print(f"      已保存 → {sess_path}  ({len(sess_stats):,} sessions)")

    dist = sess_stats["behavior_type"].value_counts()
    print(f"      行为分布：{dist.to_dict()}")

    # ── 4. 汇总到用户粒度（取最近一个 session 的行为类型）──────────────────
    print("[4/4] 汇总到用户粒度 …")
    # 按 ts_start 降序，取每个用户最近的一个 session
    latest = (
        sess_stats.sort_values("ts_start", ascending=False)
        .drop_duplicates("visitorid")
        [["visitorid", "behavior_type", "n_cats", "n_items", "n_views", "ts_start"]]
        .rename(columns={"ts_start": "last_session_ts"})
    )

    user_path = os.path.join(CACHE_DIR, "user_behavior.parquet")
    latest.to_parquet(user_path, index=False)
    print(f"      已保存 → {user_path}  ({len(latest):,} 用户)")

    print(f"\n✅ 完成，缓存目录：./{CACHE_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预计算模块一行为类型缓存")
    parser.add_argument("--data", default=".", help="数据目录，默认当前目录")
    args = parser.parse_args()
    main(data_dir=args.data)
