# prepare_reco_data.py
import os
import pickle
import numpy as np
import pandas as pd

from preprocess import clean_events, SESSION_GAP_SEC
from app import build_transition_tables, recommend_td_multifaceted_fpmc, load_item_category_latest

DATA_DIR = "."
EVENTS_PATH = "events.csv"
RECO_CACHE_DIR = "reco_cache"
os.makedirs(RECO_CACHE_DIR, exist_ok=True)

TOP_USERS_PATH = os.path.join(RECO_CACHE_DIR, "top_users.npy")
TRANSITION_PATH = os.path.join(RECO_CACHE_DIR, "transitions.pkl")
ACTIVE_USER_IDS_PATH = os.path.join(RECO_CACHE_DIR, "active_users.npy")


def main():
    # 1) 只读需要的列
    print("Loading events header...")
    usecols = ["visitorid", "itemid", "timestamp", "event"]
    events = pd.read_csv(EVENTS_PATH, usecols=usecols,
                         dtype={"visitorid": str, "itemid": str})
    print(f"events loaded: {len(events):,}")

    # 2) 清洗 + 只保留最近 10% 或活跃前 N 用户
    events = clean_events(events)
    # 2.1 选活跃用户（行为数 > 5）
    vc = events["visitorid"].value_counts()
    active_users = vc[vc > 5].index.to_numpy()
    np.save(ACTIVE_USER_IDS_PATH, active_users)
    print(f"active users (events>5): {len(active_users):,}")

    # 2.2 可选：只保留这些活跃用户的行为（显著减小数据量）
    events = events[events["visitorid"].isin(active_users)].copy()

    # 3) 加载最新 category 映射
    item_to_cat = load_item_category_latest(DATA_DIR)

    # 4) 预计算转移表
    print("Building transition tables...")
    item_next, cat_next, cat_to_items, global_hot = build_transition_tables(events, item_to_cat)
    with open(TRANSITION_PATH, "wb") as f:
        pickle.dump(
            {"item_next": item_next,
             "cat_next": cat_next,
             "cat_to_items": cat_to_items,
             "global_hot": global_hot},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print("Transition tables saved.")

    # 5) 预计算高置信示例用户的完整推荐结果
    print("Computing results for top-confidence users...")
    precomputed_results = {}

    # 采样前 500 个活跃用户寻找最优解
    test_size = min(500, len(active_users))
    test_users = active_users[:test_size]

    scored = []
    for uid in test_users:
        recs = recommend_td_multifaceted_fpmc(
            visitorid=str(uid),
            events=events,
            item_to_cat=item_to_cat,
            item_next=item_next,
            cat_next=cat_next,
            cat_to_items=cat_to_items,
            global_hot=global_hot,
            k=5,
            alpha=0.25,
            beta=0.55,
            gamma=0.20,
            use_time_decay=True,
        )
        if not recs:
            continue
        top1_prob = float(recs[0][2])
        scored.append((str(uid), top1_prob, recs))

    # 选出概率最高的 10 个
    scored.sort(key=lambda x: x[1], reverse=True)
    top_10_samples = scored[:10]

    # 存储结果：{ '用户ID': [推荐列表] }
    final_data = {item[0]: item[2] for item in top_10_samples}
    np.save(TOP_USERS_PATH, np.array(list(final_data.keys()), dtype=object))

    # 保存完整推荐结果
    with open(os.path.join(RECO_CACHE_DIR, "precomputed_recs.pkl"), "wb") as f:
        pickle.dump(final_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("✅ 离线预计算完成！App 现在可以直接读取结果。")


if __name__ == "__main__":
    main()