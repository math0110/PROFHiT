"""
Adapter for the zaai-ai/hierarchical_time_series_datasets HF datasets
(prison, tourism, police, m5).

Those datasets only ship leaf-level series plus, per grouping variable, a
category assignment for each leaf series -- a *crossed/grouped* hierarchy
(e.g. prison's leaves are crossed by state x gender x legal status), not a
single nested tree like PROFHiT's own data/*/data.csv files. There is no
aggregation matrix and no internal-node values provided; both are derived
here.

Each grouping variable is materialized as its own flat, single-level
hierarchy off "Total" (Total -> categories(g) -> leaves), generalizing the
two-hierarchy hmatrix1/hmatrix2 pattern already used for Tourism-L in
train_tourism.py to G independent hierarchies sharing the same node/data
array.
"""
import pickle

import numpy as np

from hierarchy_data import TSNode

HF_REPO_ID = "zaai-ai/hierarchical_time_series_datasets"
AVAILABLE_DATASETS = ("prison", "tourism", "police", "m5")


def _download_pickle(name: str, cache_dir=None):
    if name not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Unknown HF dataset '{name}', expected one of {AVAILABLE_DATASETS}"
        )
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename=f"{name}.pkl",
        cache_dir=cache_dir,
    )
    with open(path, "rb") as f:
        return pickle.load(f)


class GroupedHierarchyData:
    """
    self.data: (N, T) array, T = train_steps + h.
        Row 0 is "Total", rows 1..S are the leaf series (S = n_leaf),
        remaining rows are one aggregate per (grouping variable, category).
    self.hierarchies: dict[group_name -> list[TSNode]], length N each,
        one flat single-level tree per grouping variable covering ALL node
        indices (nodes belonging to other groups are inert self-nodes).
    """

    def __init__(self, name: str, cache_dir=None):
        self.name = name
        raw = _download_pickle(name, cache_dir=cache_dir)
        self.seasonality = raw["seasonality"]
        self.h = raw["h"]
        self.dates = raw["dates"]

        train = raw["train"]
        predict = raw["predict"]
        if predict["data"].shape[0] != train["data"].shape[0] + self.h:
            raise ValueError(
                f"[{name}] expected predict split to be train + h steps, got "
                f"train={train['data'].shape[0]} predict={predict['data'].shape[0]} "
                f"h={self.h}"
            )

        self.group_names = list(train["groups_names"].keys())
        self.groups_idx = {
            g: np.asarray(train["groups_idx"][g]) for g in self.group_names
        }
        self.groups_n = {g: int(train["groups_n"][g]) for g in self.group_names}
        self.category_names = {
            g: list(train["groups_names"][g]) for g in self.group_names
        }

        self.n_leaf = train["s"]
        self.train_steps = train["n"]
        self.total_steps = predict["n"]

        leaf_train = train["data"].T.astype(np.float64)  # (S, T_train)
        leaf_full = predict["data"].T.astype(np.float64)  # (S, T_train + h)
        if leaf_train.shape[0] != self.n_leaf:
            raise ValueError(f"[{name}] leaf series count mismatch")
        if not np.allclose(leaf_full[:, : self.train_steps], leaf_train):
            raise ValueError(
                f"[{name}] predict split does not prefix-match train split as expected"
            )

        self._build_nodes(leaf_full)

    def _build_nodes(self, leaf_full):
        S = self.n_leaf

        node_names = ["Total"]
        rows = [leaf_full.sum(axis=0)]

        leaf_idx = list(range(1, 1 + S))
        node_names += [f"leaf_{i}" for i in range(S)]
        rows += [leaf_full[i] for i in range(S)]

        group_cat_idx = {}  # group_name -> [node idx, ...] in category order
        next_idx = 1 + S
        for g in self.group_names:
            k = self.groups_n[g]
            idxs = self.groups_idx[g]
            cat_node_idx = []
            for c in range(k):
                mask = idxs == c
                rows.append(leaf_full[mask].sum(axis=0))
                cat_name = str(self.category_names[g][c])
                node_names.append(f"{g}={cat_name}")
                cat_node_idx.append(next_idx)
                next_idx += 1
            group_cat_idx[g] = cat_node_idx

        self.data = np.stack(rows, axis=0)  # (N, T)
        self.idx_dict = {name: i for i, name in enumerate(node_names)}
        self.node_names = node_names
        self.n_nodes = len(node_names)
        self._leaf_idx = leaf_idx
        self._group_cat_idx = group_cat_idx

        # sanity: Total == sum of leaves == sum of any group's category totals
        assert np.allclose(self.data[0], leaf_full.sum(axis=0))
        for g in self.group_names:
            cat_sum = self.data[group_cat_idx[g]].sum(axis=0)
            assert np.allclose(
                cat_sum, self.data[0], atol=1e-3
            ), f"[{self.name}] group {g} categories don't sum to Total"

        self.hierarchies = {}
        for g in self.group_names:
            nodes = [None] * self.n_nodes
            total_node = TSNode(0, "Total", None)
            nodes[0] = total_node
            cat_nodes = []
            for idx in group_cat_idx[g]:
                n = TSNode(idx, node_names[idx], total_node)
                nodes[idx] = n
                cat_nodes.append(n)
                total_node.children.append(n)
            for li, idx in enumerate(leaf_idx):
                cat_of_leaf = int(self.groups_idx[g][li])
                parent_node = cat_nodes[cat_of_leaf]
                leaf_node = TSNode(idx, node_names[idx], parent_node)
                nodes[idx] = leaf_node
                parent_node.children.append(leaf_node)
            # nodes belonging to OTHER groups: inert self-nodes (no-op in hmatrix)
            for idx in range(self.n_nodes):
                if nodes[idx] is None:
                    nodes[idx] = TSNode(idx, node_names[idx], None)
            self.hierarchies[g] = nodes

    def generate_hmatrices(self):
        """dict[group_name -> (N, N) 0/1 aggregation matrix], generalizing
        train_tourism.py's generate_hmatrix() to one matrix per grouping
        variable."""
        mats = {}
        for g, nodes in self.hierarchies.items():
            m = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
            for n in nodes:
                if len(n.children) == 0:
                    m[n.idx, n.idx] = 1.0
                else:
                    c_idx = [c.idx for c in n.children]
                    m[n.idx, c_idx] = 1.0
            mats[g] = m
        return mats

    @property
    def train_data(self):
        return self.data[:, : self.train_steps]

    @property
    def ground_truth_horizon(self):
        return self.data[:, self.train_steps : self.train_steps + self.h]
