# =============================================================================
# prepare_funnel_cache.py — 类目转化率漏斗离线预计算脚本
# 运行一次即可；app.py 模块二直接加载缓存，零实时计算。
#
# 生成文件（保存在 funnel_cache/ 目录）：
#   cat_funnel.parquet   — 每个类目的 view/addtocart/transaction 次数及转化率
#   item_funnel.parquet  — 每个物品的同类统计（用于下钻）
#
# 用法：
#   python prepare_funnel_cache.py
#   python prepare_funnel_cache.py --data .
# =============================================================================

import argparse
import os

import pandas as pd

CACHE_DIR = "funnel_cache"


def load_item_category(data_dir: str) -> pd.Series:
    frames = []
    for fname in ("item_properties_part1.csv", "item_properties_part2.csv"):
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            frames.append(pd.read_csv(fpath, dtype={"itemid": str}, encoding="latin-1"))
    if not frames:
        return pd.Series(dtype="object")

    props = pd.concat(frames, ignore_index=True)
    if not {"itemid", "property", "value"}.issubset(props.columns):
        return pd.Series(dtype="object")

    cat = props[props["property"] == "categoryid"][["itemid", "value"]].copy()
    if "timestamp" in props.columns:
        cat_ts = props[props["property"] == "categoryid"][
            ["itemid", "value", "timestamp"]
        ].copy()
        cat_ts["timestamp"] = pd.to_numeric(cat_ts["timestamp"], errors="coerce")
        cat_ts = cat_ts.dropna(subset=["timestamp"]).sort_values("timestamp")
        cat_ts = cat_ts.drop_duplicates("itemid", keep="last")[["itemid", "value"]]
        cat = cat_ts

    cat.columns = ["itemid", "categoryid"]
    cat["itemid"] = cat["itemid"].astype(str)
    cat["categoryid"] = cat["categoryid"].astype(str)
    return cat.drop_duplicates("itemid").set_index("itemid")["categoryid"]


def main(data_dir: str = ".") -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ── 1. 加载并清洗 events ─────────────────────────────────────────────────
    print("[1/3] 加载并清洗 events.csv …")
    from preprocess import load_data, clean_events
    events, _, _ = load_data(data_dir=data_dir)
    events = clean_events(events)
    events["itemid"] = events["itemid"].astype(str)
    print(f"      清洗后事件数：{len(events):,}")

    # ── 2. 映射类目 ──────────────────────────────────────────────────────────
    print("[2/3] 映射物品类目 …")
    item_to_cat = load_item_category(data_dir)
    events["categoryid"] = events["itemid"].map(item_to_cat)
    ev = events.dropna(subset=["categoryid"]).copy()
    print(f"      含类目事件数：{len(ev):,}  类目数：{ev['categoryid'].nunique():,}")

    # ── 3. 聚合类目级漏斗 ────────────────────────────────────────────────────
    print("[3/3] 聚合类目转化漏斗 …")

    def agg_events(df, level_col):
        grp = df.groupby(level_col)
        view  = grp.apply(lambda x: (x["event"] == "view").sum()).rename("view_cnt")
        cart  = grp.apply(lambda x: (x["event"] == "addtocart").sum()).rename("cart_cnt")
        tx    = grp.apply(lambda x: (x["event"] == "transaction").sum()).rename("tx_cnt")
        out = pd.concat([view, cart, tx], axis=1).reset_index()
        out["view_to_cart_rate"] = out["cart_cnt"] / out["view_cnt"].replace(0, float("nan"))
        out["cart_to_tx_rate"]   = out["tx_cnt"]  / out["cart_cnt"].replace(0, float("nan"))
        out["view_to_tx_rate"]   = out["tx_cnt"]  / out["view_cnt"].replace(0, float("nan"))
        return out

    # 类目级
    cat_funnel = agg_events(ev, "categoryid")
    cat_funnel = cat_funnel.sort_values("view_cnt", ascending=False)
    cat_path = os.path.join(CACHE_DIR, "cat_funnel.parquet")
    cat_funnel.to_parquet(cat_path, index=False)
    print(f"      已保存 → {cat_path}  ({len(cat_funnel):,} 类目)")

    # 物品级（用于下钻）
    item_funnel = agg_events(ev, "itemid")
    item_funnel["categoryid"] = item_funnel["itemid"].map(item_to_cat)
    item_funnel = item_funnel.sort_values("view_cnt", ascending=False)
    item_path = os.path.join(CACHE_DIR, "item_funnel.parquet")
    item_funnel.to_parquet(item_path, index=False)
    print(f"      已保存 → {item_path}  ({len(item_funnel):,} 物品)")

    print(f"\n✅ 完成，缓存目录：./{CACHE_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预计算类目转化漏斗缓存")
    parser.add_argument("--data", default=".", help="数据目录，默认当前目录")
    args = parser.parse_args()
    main(data_dir=args.data)
