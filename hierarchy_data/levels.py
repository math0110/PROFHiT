"""
Shared hierarchy-tree utility used by both hierarchy_data/agg_matrix.py
(labour/tourismsmall/traffic/wiki2/m5) and the repository's original
TourismHierarchyData (tourismlarge), so per-level metrics work the same
way regardless of which loader built the tree.
"""


def compute_levels(nodes):
    """dict[node_idx -> depth], root = 1, matching the paper's Table 6/7
    'Hierarchy Levels' convention. Works with any list of TSNode-like
    objects with .idx/.parent, including TourismHierarchyData's nodes1/
    nodes2 (not just AggMatrixHierarchyData's own .nodes)."""
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
