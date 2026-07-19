"""
Builds data/m5/data.csv and data/m5/tags.csv from the raw M5 competition
file `sales_train_evaluation.csv` (30,490 leaf series x 1941 daily steps --
too large for PROFHiT to train on directly, since it keeps a separate
learned output layer per node and runs its attention encoder over every
node on every training step).

Subsamples to a manageable size in two dimensions:
  - Nodes: keeps the `TOP_N_SERIES` bottom-level (item x store) series with
    the highest total sales over the history window, out of all 30,490 --
    not a stratified sample, so category/department representation isn't
    guaranteed even (a top-seller-heavy department can dominate the sample).
    Ranked using only the history window, not the horizon, so the selection
    itself doesn't leak information about the period being forecast.
  - Time: keeps only the most recent `HISTORY_DAYS` + `HORIZON` days rather
    than the full ~5.3 years, since M5's daily cadence would otherwise give
    far more training examples per epoch than any other dataset in this
    benchmark even after the node-count subsampling above.

Hierarchy nesting chosen: Total -> State -> Store -> Category -> Department
-> Item. Output format matches the sibling data/{labour,traffic,wiki2,...}
folders exactly (data.csv: one column per node, real values at every level;
tags.csv: one row per leaf, comma-separated cumulative ancestor path) so it
loads via the same generic hierarchy_data/agg_matrix.py loader.

Usage:
    python data/m5/import_m5.py --raw-csv /path/to/sales_train_evaluation.csv
"""
import argparse

import numpy as np
import pandas as pd

HISTORY_DAYS = 300
HORIZON = 28
TOP_N_SERIES = 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-csv", required=True, help="path to sales_train_evaluation.csv")
    p.add_argument("--out-dir", default="data/m5")
    args = p.parse_args()

    raw = pd.read_csv(args.raw_csv)
    day_cols = [c for c in raw.columns if c.startswith("d_")]
    assert len(day_cols) >= HISTORY_DAYS + HORIZON, "not enough days in raw file"
    keep_days = day_cols[-(HISTORY_DAYS + HORIZON):]
    history_cols = keep_days[:HISTORY_DAYS]

    total_sales = raw[history_cols].sum(axis=1)
    top_idx = total_sales.nlargest(TOP_N_SERIES).index
    df = raw.loc[top_idx, ["item_id", "dept_id", "cat_id", "store_id", "state_id"] + keep_days].copy()
    df = df.reset_index(drop=True)

    n_leaves = len(df)
    print(f"Sampled top {n_leaves} bottom-level series by history-window sales (out of {len(raw)})")
    print(f"  covering {df['item_id'].nunique()} unique items across {df['store_id'].nunique()} stores")
    print(f"  department counts: {df['dept_id'].value_counts().to_dict()}")
    print(f"Using {HISTORY_DAYS} history days + {HORIZON} horizon days = {len(keep_days)} total timesteps")

    values = df[keep_days].to_numpy(dtype=np.float64)  # (n_leaves, T)
    T = values.shape[1]

    node_values = {}
    node_names_in_order = []
    node_leaf_members = {}  # node_name -> set of leaf column names it aggregates

    def add_node(name, vec, leaf_col):
        if name not in node_values:
            node_values[name] = vec.copy()
            node_names_in_order.append(name)
            node_leaf_members[name] = set()
        else:
            node_values[name] += vec
        node_leaf_members[name].add(leaf_col)

    leaf_paths = []  # tags.csv rows, one per leaf (kept for reference/debugging)
    leaf_cols_in_order = []
    for i in range(n_leaves):
        row = df.iloc[i]
        state, store, cat, dept, item = (
            row["state_id"], row["store_id"], row["cat_id"], row["dept_id"], row["item_id"],
        )
        vec = values[i]
        leaf_col = f"{state}-{store}-{cat}-{dept}-{item}"
        leaf_cols_in_order.append(leaf_col)

        add_node("Total", vec, leaf_col)
        add_node(state, vec, leaf_col)
        store_col = f"{state}-{store}"
        add_node(store_col, vec, leaf_col)
        cat_col = f"{state}-{store}-{cat}"
        add_node(cat_col, vec, leaf_col)
        dept_col = f"{state}-{store}-{cat}-{dept}"
        add_node(dept_col, vec, leaf_col)
        add_node(leaf_col, vec, leaf_col)

        leaf_paths.append(
            ["T", f"T-{state}", f"T-{store_col}", f"T-{cat_col}", f"T-{dept_col}", f"T-{leaf_col}"]
        )

    dates = pd.RangeIndex(T)  # no calendar.csv join for now; positional index is enough for training
    out_df = pd.DataFrame({name: node_values[name] for name in node_names_in_order}, index=dates)
    out_df.index.name = ""

    # agg_mat.csv: one row per node, one column per leaf, 1 if that leaf
    # contributes to that node's total -- same convention as the other
    # datasets in this repo (labour/tourismsmall/traffic/wiki2), which is
    # what hierarchy_data/agg_matrix.py's loader reads directly.
    agg_df = pd.DataFrame(
        0,
        index=node_names_in_order,
        columns=leaf_cols_in_order,
        dtype=int,
    )
    for name, members in node_leaf_members.items():
        agg_df.loc[name, list(members)] = 1

    out_dir = args.out_dir
    out_df.to_csv(f"{out_dir}/data.csv")
    agg_df.to_csv(f"{out_dir}/agg_mat.csv")
    with open(f"{out_dir}/tags.csv", "w") as f:
        for path in leaf_paths:
            f.write(",".join(path) + "\n")

    print(f"Wrote {out_dir}/data.csv ({out_df.shape[0]} rows x {out_df.shape[1]} node columns)")
    print(f"Wrote {out_dir}/agg_mat.csv ({agg_df.shape[0]} rows x {agg_df.shape[1]} leaf columns)")
    print(f"Wrote {out_dir}/tags.csv ({len(leaf_paths)} leaf rows)")


if __name__ == "__main__":
    main()
