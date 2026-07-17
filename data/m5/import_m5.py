"""
Builds data/m5/data.csv and data/m5/tags.csv from the raw M5 competition
file `sales_train_evaluation.csv` (30,490 leaf series x 1941 daily steps --
too large for PROFHiT to train on directly, since it keeps a separate
learned output layer per node and runs its attention encoder over every
node on every training step).

Subsamples to a manageable size in two dimensions:
  - Nodes: keeps all 3 states, all 10 stores, all 3 categories and 7
    departments (every store carries every department in M5), but samples
    a small, department-stratified subset of items (default 6 per
    department, ~42 items x 10 stores = ~420 leaves) rather than all 3,049.
  - Time: keeps only the most recent `HISTORY_DAYS` + `HORIZON` days rather
    than the full ~5.3 years, since M5's daily cadence would otherwise give
    far more training examples per epoch than any other dataset in this
    benchmark even after the node-count subsampling above.

Hierarchy nesting chosen: Total -> State -> Store -> Category -> Department
-> Item. Output format matches the sibling data/{labour,traffic,wiki2,...}
folders exactly (data.csv: one column per node, real values at every level;
tags.csv: one row per leaf, comma-separated cumulative ancestor path) so it
loads via the same generic hierarchy_data/tags_csv.py loader.

Usage:
    python data/m5/import_m5.py --raw-csv /path/to/sales_train_evaluation.csv
"""
import argparse

import numpy as np
import pandas as pd

HISTORY_DAYS = 300
HORIZON = 28
ITEMS_PER_DEPT = 6
SEED = 42


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-csv", required=True, help="path to sales_train_evaluation.csv")
    p.add_argument("--out-dir", default="data/m5")
    args = p.parse_args()

    raw = pd.read_csv(args.raw_csv)
    day_cols = [c for c in raw.columns if c.startswith("d_")]
    assert len(day_cols) >= HISTORY_DAYS + HORIZON, "not enough days in raw file"
    keep_days = day_cols[-(HISTORY_DAYS + HORIZON):]

    rng = np.random.default_rng(SEED)
    sampled_items = []
    for dept, grp in raw.groupby("dept_id"):
        items = sorted(grp["item_id"].unique())
        n = min(ITEMS_PER_DEPT, len(items))
        sampled_items.extend(rng.choice(items, size=n, replace=False).tolist())
    sampled_items = sorted(set(sampled_items))

    df = raw[raw["item_id"].isin(sampled_items)].copy()
    df = df[["item_id", "dept_id", "cat_id", "store_id", "state_id"] + keep_days]

    n_leaves = len(df)
    print(f"Sampled {len(sampled_items)} items x {df['store_id'].nunique()} stores = {n_leaves} leaf series")
    print(f"Using {HISTORY_DAYS} history days + {HORIZON} horizon days = {len(keep_days)} total timesteps")

    values = df[keep_days].to_numpy(dtype=np.float64)  # (n_leaves, T)
    T = values.shape[1]

    node_values = {}
    node_names_in_order = []

    def add_node(name, vec):
        if name not in node_values:
            node_values[name] = vec.copy()
            node_names_in_order.append(name)
        else:
            node_values[name] += vec

    leaf_paths = []  # tags.csv rows, one per leaf
    for i in range(n_leaves):
        row = df.iloc[i]
        state, store, cat, dept, item = (
            row["state_id"], row["store_id"], row["cat_id"], row["dept_id"], row["item_id"],
        )
        vec = values[i]

        add_node("Total", vec)
        add_node(state, vec)
        store_col = f"{state}-{store}"
        add_node(store_col, vec)
        cat_col = f"{state}-{store}-{cat}"
        add_node(cat_col, vec)
        dept_col = f"{state}-{store}-{cat}-{dept}"
        add_node(dept_col, vec)
        leaf_col = f"{state}-{store}-{cat}-{dept}-{item}"
        add_node(leaf_col, vec)

        leaf_paths.append(
            ["T", f"T-{state}", f"T-{store_col}", f"T-{cat_col}", f"T-{dept_col}", f"T-{leaf_col}"]
        )

    dates = pd.RangeIndex(T)  # no calendar.csv join for now; positional index is enough for training
    out_df = pd.DataFrame({name: node_values[name] for name in node_names_in_order}, index=dates)
    out_df.index.name = ""

    out_dir = args.out_dir
    out_df.to_csv(f"{out_dir}/data.csv")
    with open(f"{out_dir}/tags.csv", "w") as f:
        for path in leaf_paths:
            f.write(",".join(path) + "\n")

    print(f"Wrote {out_dir}/data.csv ({out_df.shape[0]} rows x {out_df.shape[1]} node columns)")
    print(f"Wrote {out_dir}/tags.csv ({len(leaf_paths)} leaf rows)")


if __name__ == "__main__":
    main()
