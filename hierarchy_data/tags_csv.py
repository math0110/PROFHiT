"""
Generic loader for the paper's own bundled datasets (data/labour,
data/tourismlarge, data/tourismsmall, data/traffic, data/wiki2, and
data/m5 once added) -- all of which ship the same two-file convention:

- data.csv: one column per hierarchy node (both leaf AND aggregate nodes
  already carry real values, unlike the HF grouped datasets which only had
  leaves), one row per timestep.
- tags.csv: one row per LEAF node, giving the cumulative ancestor path from
  root to that leaf as comma-separated strings (e.g.
  "T,T-hol,T-hol-nsw,T-hol-nsw-city").

The per-dataset existing loaders in hierarchy_data/__init__.py hardcode
fragile, dataset-specific string transforms to reconstruct the tree from
data.csv's column names alone (e.g. TourismHierarchyData's suffix-matching
on "Hol"/"Vis"/"Bus"/"Oth"). That breaks across datasets with different
naming conventions:
  - tourismsmall reorders tokens relative to its own tags.csv path
    ("T-hol-nsw-city" vs the actual column "nsw-hol-city").
  - wiki2 uses "_" as its data.csv separator vs tags.csv's "-"
    ("T-de-AAC" vs the actual column "de_AAC").
  - labour's data.csv columns are Python-list-repr strings with multi-word,
    space/hyphen-separated dimension values ("['Employed full-time',
    'Females', 'Australian Capital Territory']"), reordered relative to
    tags.csv's own per-level tokens ("Employedfull_time", "Females",
    "AustralianCapitalTerritory").

Instead, this walks each tags.csv row level by level, takes the newly-added
dimension token at each level (the suffix beyond the previous level), and
builds up a canonical *set* of normalized dimension tokens (all non-
alphanumeric characters stripped, lowercased) per node. data.csv columns
are parsed the same way -- as a Python list literal when they look like
one, otherwise split on "-"/"_" -- and matched by the same canonical set.
Set-based (not positional) matching is what makes this invariant to both
reordering and separator/spacing differences, without per-dataset
special-casing.
"""
import ast
import re

import numpy as np
import pandas as pd

from hierarchy_data import TSNode


def compute_levels(nodes):
    """dict[node_idx -> depth], root = 1, matching the paper's Table 6/7
    'Hierarchy Levels' convention. Works with any list of TSNode-like
    objects with .idx/.parent, including TourismHierarchyData's nodes1/
    nodes2 (not just TagsCSVHierarchyData's own .nodes)."""
    depths = {}

    def depth_of(node):
        if node.idx in depths:
            return depths[node.idx]
        d = 1 if node.parent is None else depth_of(node.parent) + 1
        depths[node.idx] = d
        return d

    for n in nodes:
        depth_of(n)
    return depths

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _normalize_token(tok):
    return _NON_ALNUM_RE.sub("", tok.lower())


def _column_canonical_key(col_name):
    """Parse a data.csv column name into a frozenset of normalized
    dimension tokens, handling both Python-list-repr columns (labour) and
    plain separator-joined columns (tourism/traffic/wiki2)."""
    stripped = col_name.strip()
    if stripped.startswith("["):
        try:
            parts = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            parts = re.split(r"[-_]", stripped)
    else:
        # tourismlarge pads every plain column name with a trailing "All"
        # wherever a dimension (e.g. purpose) hasn't been specified yet
        # ("AAll" = geo "A", purpose unspecified; "TotalAll" = root). It's
        # never a genuine dimension value in tags.csv, so strip it as a
        # suffix/prefix only (not a substring match anywhere) to avoid
        # false hits on e.g. a geo code that happens to contain "all".
        stripped = re.sub(r"(?i)all$", "", stripped)
        stripped = re.sub(r"(?i)^total", "", stripped)
        parts = re.split(r"[-_]", stripped)
    tokens = {_normalize_token(p) for p in parts}
    tokens.discard("total")
    tokens.discard("t")
    tokens.discard("")
    return frozenset(tokens)


class TagsCSVHierarchyData:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        df = pd.read_csv(f"{data_dir}/data.csv", index_col=0)

        # Exact column-name matching is tried first (needed for datasets we
        # generate ourselves with self-consistent naming, like m5, where
        # tokens can legitimately repeat across levels -- e.g. department
        # "HOBBIES_1" already contains the tokens "hobbies"+"1" that store
        # "CA_1" also contributed -- which breaks *set*-based canonical-key
        # matching since duplicate tokens collapse away). The canonical-key
        # fallback below is only for datasets with genuine reordering
        # (tourismsmall) or separator differences (wiki2) between tags.csv
        # and data.csv, where exact string matching can't work at all.
        exact_cols = set(df.columns)

        col_canon = {}
        for c in df.columns:
            key = _column_canonical_key(c)
            col_canon.setdefault(key, c)  # first occurrence wins; only a fallback

        if "Total" in exact_cols:
            root_col = "Total"
        else:
            root_col = col_canon.get(frozenset())
        if root_col is None:
            raise ValueError(f"[{data_dir}] could not find a Total/root column in data.csv")

        with open(f"{data_dir}/tags.csv") as f:
            leaf_paths = [line.strip().split(",") for line in f if line.strip()]

        nodes_by_col = {}
        idx_dict = {}
        nodes = []

        def get_or_create(col_name, parent_node):
            if col_name in nodes_by_col:
                return nodes_by_col[col_name]
            idx = len(nodes)
            node = TSNode(idx, col_name, parent_node)
            if parent_node is not None:
                parent_node.children.append(node)
            nodes_by_col[col_name] = node
            idx_dict[col_name] = idx
            nodes.append(node)
            return node

        root_node = get_or_create(root_col, None)

        for path in leaf_paths:
            parent = root_node
            seen_tokens = set()
            prev_level_str = path[0]
            for level_str in path[1:]:
                if not level_str.startswith(prev_level_str):
                    raise ValueError(
                        f"[{data_dir}] tags.csv path level {level_str!r} doesn't "
                        f"extend previous level {prev_level_str!r}"
                    )
                new_part = level_str[len(prev_level_str):].lstrip("-")
                seen_tokens.add(_normalize_token(new_part))

                full_path_name = level_str[2:]  # strip leading "T-"
                if full_path_name in exact_cols:
                    # datasets we generate ourselves with self-consistent,
                    # cumulative naming (m5)
                    col_name = full_path_name
                elif new_part in exact_cols:
                    # datasets whose aggregate labels are flat/independent
                    # rather than composed from their parent's label
                    # (traffic: "y11", "Bottom1" don't encode "y1")
                    col_name = new_part
                else:
                    # datasets whose column names encode the full
                    # cumulative path but reordered or with a different
                    # separator than tags.csv uses (tourismsmall, wiki2,
                    # labour)
                    key = frozenset(seen_tokens)
                    if key not in col_canon:
                        raise ValueError(
                            f"[{data_dir}] no data.csv column matches tags.csv path "
                            f"segment {level_str!r} (tried exact matches "
                            f"{full_path_name!r} and {new_part!r}, and canonical "
                            f"token set {key})"
                        )
                    col_name = col_canon[key]
                parent = get_or_create(col_name, parent)
                prev_level_str = level_str

        data = np.zeros((len(nodes), len(df)), dtype=np.float64)
        for col_name, node in nodes_by_col.items():
            data[node.idx, :] = df[col_name].values.astype(np.float64)

        self.data = data
        self.idx_dict = idx_dict
        self.nodes = nodes
        self.dates = list(df.index)
        self.n_nodes = len(nodes)

    def levels(self):
        return compute_levels(self.nodes)

    def generate_hmatrix(self):
        """(N, N) 0/1 aggregation matrix, matching train.py/train_tourism.py's
        generate_hmatrix() convention (self-identity row for leaves)."""
        m = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        for n in self.nodes:
            if len(n.children) == 0:
                m[n.idx, n.idx] = 1.0
            else:
                c_idx = [c.idx for c in n.children]
                m[n.idx, c_idx] = 1.0
        return m
