"""
Loads a hierarchy dataset from its data.csv + agg_mat.csv files
(data/labour, data/tourismsmall, data/traffic, data/wiki2, data/m5).

agg_mat.csv is the aggregation/summing matrix: one row per node (including
leaves), one column per leaf series, 1 if that leaf contributes to that
node's total. Every dataset here already ships this file, so the tree can
be built directly from it instead of guessing at each dataset's own
data.csv column-naming convention.
"""
import numpy as np
import pandas as pd

from hierarchy_data import TSNode
from hierarchy_data.levels import compute_levels


class AggMatrixHierarchyData:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        df = pd.read_csv(f"{data_dir}/data.csv", index_col=0)
        agg = pd.read_csv(f"{data_dir}/agg_mat.csv", index_col=0)

        node_names = list(agg.index)
        n_nodes = len(node_names)

        missing = [n for n in node_names if n not in df.columns]
        if missing:
            raise ValueError(
                f"[{data_dir}] agg_mat.csv row {missing[0]!r} has no matching data.csv column"
            )

        agg_bool = agg.to_numpy() > 0.5  # (n_nodes, n_leaves)

        # agg_mat.csv gives each node's full leaf-set, not its immediate
        # parent -- recovered via set containment: a node's parent is
        # whichever other node has the smallest leaf-set that strictly
        # contains its own. Nodes with exactly one child share their
        # child's leaf-set exactly (tied, not a strict subset), so those
        # get grouped into a "class" and ordered by name length instead
        # (every dataset here builds deeper names by extending shallower
        # ones, e.g. wiki2's "de_DES" -> "de_DES_AAG").
        leafset_key = [tuple(np.nonzero(agg_bool[i])[0]) for i in range(n_nodes)]
        classes = {}
        for i, key in enumerate(leafset_key):
            classes.setdefault(key, []).append(i)
        for key in classes:
            classes[key].sort(key=lambda i: len(node_names[i]))

        class_keys = list(classes.keys())
        class_sizes = np.array([len(k) for k in class_keys])
        class_bool = np.array([agg_bool[classes[k][0]] for k in class_keys])

        parent_class_of = {}
        for ci, key in enumerate(class_keys):
            is_superset = ~(class_bool[ci] & ~class_bool).any(axis=1)
            is_strict = class_sizes > class_sizes[ci]
            candidates = np.where(is_superset & is_strict)[0]
            if len(candidates) == 0:
                parent_class_of[key] = None
                continue
            min_size = class_sizes[candidates].min()
            tied = candidates[class_sizes[candidates] == min_size]
            if len(tied) > 1:
                raise ValueError(
                    f"[{data_dir}] the class containing {node_names[classes[key][0]]!r} has "
                    f"{len(tied)} equally-small candidate parent classes -- not a clean tree"
                )
            parent_class_of[key] = class_keys[tied[0]]

        parent_of = np.full(n_nodes, -1, dtype=int)
        for key, members in classes.items():
            parent_key = parent_class_of[key]
            parent_of[members[0]] = -1 if parent_key is None else classes[parent_key][-1]
            for k in range(1, len(members)):
                parent_of[members[k]] = members[k - 1]

        # build TSNode tree, root first so parents exist before children
        # reference them (ties broken by name length, matching the order
        # established above)
        node_sizes = np.array([len(k) for k in leafset_key])
        nodes = [None] * n_nodes
        for i in sorted(range(n_nodes), key=lambda i: (-node_sizes[i], len(node_names[i]))):
            parent_node = nodes[parent_of[i]] if parent_of[i] != -1 else None
            node = TSNode(i, node_names[i], parent_node)
            nodes[i] = node
            if parent_node is not None:
                parent_node.children.append(node)

        self.data = df[node_names].to_numpy(dtype=np.float64).T  # (n_nodes, T)
        self.idx_dict = {name: i for i, name in enumerate(node_names)}
        self.nodes = nodes
        self.n_nodes = n_nodes
        self.dates = list(df.index)

    def levels(self):
        return compute_levels(self.nodes)

    def generate_hmatrix(self):
        m = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        for n in self.nodes:
            if len(n.children) == 0:
                m[n.idx, n.idx] = 1.0
            else:
                m[n.idx, [c.idx for c in n.children]] = 1.0
        return m
