"""
Loads a hierarchy dataset from its data.csv + agg_mat.csv files
(data/labour, data/tourismsmall, data/traffic, data/wiki2, data/m5).

agg_mat.csv is the aggregation/summing matrix ("S matrix" in the
hierarchical forecasting literature -- this is the standard output of
Nixtla's datasetsforecast/hierarchicalforecast tooling, and of the older R
`hts` package before it): one row per node (including the leaves
themselves), one column per leaf series, 1 if that leaf contributes to
that node's total, 0 otherwise. Every dataset here already ships this file
-- there's no need to reverse-engineer the tree from data.csv's column-
naming conventions (which differ across datasets) at all.

The only thing agg_mat.csv doesn't give directly is each node's *immediate*
parent (it tells you the full set of leaves under a node, not who its
direct children are). That's recovered from simple set containment: since
these are proper trees, a node's immediate parent is whichever other node
has the smallest leaf-set that still strictly contains this node's own
leaf-set. (Two nodes tied for smallest would mean the hierarchy isn't a
clean tree -- that's treated as an error rather than silently guessed.)
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

        # Phase 1: group nodes by *identical* leaf-set. A node with exactly
        # one child aggregates the exact same leaves as that child, so a
        # whole chain of single-child ancestors can share one leaf-set --
        # set containment alone can't order a chain like that (nothing is a
        # strict subset of anything else in it). Every dataset here builds
        # deeper names by extending a shallower one (e.g. wiki2's
        # "de_DES" -> "de_DES_AAG"), so sorting a chain by name length
        # (shortest = closest to root) recovers the right order.
        leafset_key = [tuple(np.nonzero(agg_bool[i])[0]) for i in range(n_nodes)]
        classes = {}
        for i, key in enumerate(leafset_key):
            classes.setdefault(key, []).append(i)
        for key in classes:
            classes[key].sort(key=lambda i: len(node_names[i]))

        class_keys = list(classes.keys())
        class_sizes = np.array([len(k) for k in class_keys])
        class_bool = np.array([agg_bool[classes[k][0]] for k in class_keys])

        # Phase 2: for each class (a single node, or a whole tied chain),
        # find its parent class = smallest strict superset among all other
        # *distinct* leaf-sets -- no more same-size ties here, since the
        # only reason for a tie was already resolved by grouping into
        # classes above.
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

        # Phase 3: assemble each node's immediate parent -- chain members
        # link to each other (shortest name first), and each chain's
        # topmost (shortest-named) member attaches to its parent class's
        # bottommost (longest-named, i.e. most specific) member.
        parent_of = np.full(n_nodes, -1, dtype=int)
        for key, members in classes.items():
            parent_key = parent_class_of[key]
            parent_of[members[0]] = -1 if parent_key is None else classes[parent_key][-1]
            for k in range(1, len(members)):
                parent_of[members[k]] = members[k - 1]

        # build TSNode tree, largest leaf-set (root) first so parents exist
        # before their children reference them -- within a tied chain
        # (same leaf-set, same size), shorter names come first since
        # that's the ancestor-to-descendant order established above
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
        """(N, N) 0/1 aggregation matrix marking each node's *immediate*
        children (self-identity row for leaves) -- matches
        train.py/train_tourism.py's generate_hmatrix() convention."""
        m = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        for n in self.nodes:
            if len(n.children) == 0:
                m[n.idx, n.idx] = 1.0
            else:
                m[n.idx, [c.idx for c in n.children]] = 1.0
        return m
