"""
Evaluation metrics matching the PROFHiT paper's Section 5.1 definitions
(Kamarthi et al., KDD'23), for use across train.py / train_tourism.py /
train_hf.py so results are comparable to the paper's Table 3.

All functions take numpy arrays shaped (N, T) -- N nodes, T forecast steps
-- and return a single scalar averaged over both axes, matching how the
paper reports "average scores across all levels of hierarchy".

Notes on two definitions that are underspecified/ambiguous in the paper text:

- Log Score (LS): the paper defines it as the negative integral of the log
  *density* (not the log of the integral of the density) over a fixed-width
  window [y-L, y+L] around the ground truth. L is never given a concrete
  value in the main text, and computing it in raw units turns out to be a
  units-mismatch trap: the -log(sigma) term scales with a node's absolute
  magnitude, so within one hierarchy a large node (e.g. prison's Total,
  ~37,000) dominates the average versus small ones (leaves as low as 6)
  regardless of relative calibration quality. Here LS is computed in each
  node's own normalized (z-scored) units instead -- see `log_score`'s
  docstring for the full reasoning -- with `window` a fixed 0.005 (0.5% of
  one std) in that normalized space.

- Distributional Consistency Error (DCE, Eq. 7): as literally written, the
  paper's formula sums a per-node JSD-derived term over all non-leaf nodes
  and subtracts a single constant 1/2 at the end -- but Jensen-Shannon
  divergence (Eq. 8) contributes a "-1" (so "-0.5" after the outer 0.5
  factor) *per node*, and Table 3's reported magnitudes (0.02-0.42) are
  stated to be "average scores across all levels of hierarchy", so a raw
  sum growing with node count N cannot be what's plotted. We therefore
  compute the *mean* per-node JSD term (i.e. average, not sum, over
  non-leaf nodes) as the most reproducible reading consistent with how the
  other metrics in Table 3 are aggregated. Flag this if exact literal
  Eq. 7 semantics are required.
"""
import numpy as np
from scipy.stats import norm


def mape(y_true, y_pred, min_abs_actual=None):
    """Mean Absolute Percentage Error, excluding points where |y_true| is
    too small to divide by meaningfully. Hierarchical count data (crime,
    sales, visitor counts) routinely has zero/near-zero periods across
    hundreds of leaf nodes; dividing by those -- even after clamping the
    denominator to a tiny epsilon -- produces individual ratios in the
    thousands that dominate the mean and make MAPE meaningless (this is
    what happened before this fix: MAPE values in the millions). Points
    are excluded rather than epsilon-clamped, since a near-zero denominator
    still produces an enormous, average-dominating ratio either way.
    `min_abs_actual` defaults to 1% of the mean absolute ground truth;
    see also `wape`, which has no such issue and is the more robust
    headline metric for these datasets.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if min_abs_actual is None:
        min_abs_actual = max(0.01 * np.mean(np.abs(y_true)), 1e-8)
    mask = np.abs(y_true) >= min_abs_actual
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def wape(y_true, y_pred):
    """Weighted Absolute Percentage Error = sum|y-yhat| / sum|y|. Unlike
    MAPE, well-defined even with many zero/near-zero ground-truth points --
    the standard robust alternative for exactly this kind of intermittent
    hierarchical count data (used e.g. in the M5 competition)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.sum(np.abs(y_true))
    if denom < 1e-8:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def crps_gaussian(y_true, mu, sigma, eps=1e-6):
    """Closed-form CRPS assuming the fitted predictive distribution is
    Gaussian(mu, sigma) -- matches the paper's evaluation protocol
    ("We approximate F_y as a Gaussian distribution formed from samples
    of the model to derive CRPS")."""
    y_true = np.asarray(y_true, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.clip(np.asarray(sigma, dtype=np.float64), eps, None)
    z = (y_true - mu) / sigma
    crps = sigma * (
        z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)
    )
    return float(np.mean(crps))


def log_score(y_true, mu, sigma, node_scale=None, window=0.005, cap=10.0, eps=1e-6):
    """Eq. in Section 5.1: LS = -integral_{y-L}^{y+L} log p_y(y_hat) d y_hat,
    for Gaussian p_y(mu, sigma), evaluated in closed form. `cap` mirrors the
    paper's note (following Reich et al.) that per-point log-likelihood is
    floored at -10 -- implemented here as an upper bound on the resulting
    (already negated) LS value, so a single degenerate prediction can't
    dominate the average.

    Computed in each node's own normalized (z-scored) units, not raw ones.
    A first attempt at this used an absolute window sized off |y_true| --
    but the -log(sigma) normalizing term in a Gaussian's log-density scales
    with a node's *absolute* magnitude, so any window wide enough to be
    meaningful for a large node (e.g. prison's Total, ~37,000) dominates
    the average versus small nodes (leaves as low as 6) regardless of how
    well-calibrated either actually is -- an inherent units mismatch, not
    something a smarter constant can fix. Dividing y_true/mu/sigma by each
    node's own characteristic scale before integrating puts every node on
    the same footing (a Gaussian(mu, sigma) divided by a positive constant
    s is still Gaussian(mu/s, sigma/s)), so `window` (default 0.005, i.e.
    0.5% of one std) is uniformly meaningful regardless of a node's raw
    magnitude.

    `node_scale`: array broadcastable to y_true's shape giving each node's
    characteristic scale (e.g. the per-node std used to normalize the
    model's own training data -- the units it was actually calibrated in).
    Defaults to the std of y_true itself if not provided.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.clip(np.asarray(sigma, dtype=np.float64), eps, None)
    if node_scale is None:
        node_scale = np.maximum(np.std(y_true), eps)
    node_scale = np.clip(np.asarray(node_scale, dtype=np.float64), eps, None)

    y_n = y_true / node_scale
    mu_n = mu / node_scale
    sigma_n = np.clip(sigma / node_scale, eps, None)

    a = y_n - window - mu_n
    b = y_n + window - mu_n
    log_norm_const = np.log(sigma_n) + 0.5 * np.log(2 * np.pi)
    cubic_term = (b ** 3 - a ** 3) / (6 * sigma_n ** 2)
    ls = log_norm_const * (2 * window) + cubic_term
    ls = np.minimum(ls, cap)
    return float(np.mean(ls))


def calibration_score(y_true, mu, sigma, step=0.05, eps=1e-6):
    """Definition 3: CS = integral_0^1 |k_M(c) - c| dc, approximated via a
    Riemann sum with the step size (0.05) given in the paper."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.clip(np.asarray(sigma, dtype=np.float64).ravel(), eps, None)

    confidences = np.arange(step, 1.0 + 1e-9, step)
    total = 0.0
    for c in confidences:
        z = norm.ppf(0.5 + c / 2.0)
        lower = mu - z * sigma
        upper = mu + z * sigma
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        total += abs(coverage - c) * step
    return float(total)


def distributional_consistency_error(mu, sigma, hmatrices, eps=1e-6):
    """Eq. 5-7: for every non-leaf node i (aggregation weights assumed 1,
    matching the simple-summation convention used by all datasets here),
    compare node i's forecast distribution against the distribution implied
    by summing its children's forecasts, via the closed-form Gaussian JSD.
    See module docstring for the mean-vs-sum normalization caveat.

    mu, sigma: (N,) arrays -- forecast mean/std per node (e.g. averaged
        over the forecast horizon, or for a single horizon step).
    hmatrices: dict[group_name -> (N, N) 0/1 aggregation matrix], as
        returned by GroupedHierarchyData.generate_hmatrices() or the
        analogous generate_hmatrix() in train.py / train_tourism.py.

    Returns (overall, per_group) where overall is the mean over all
    groups' per-group means, and per_group is a dict[group_name -> float].
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.clip(np.asarray(sigma, dtype=np.float64), eps, None)
    var = sigma ** 2

    per_group = {}
    for g, m in hmatrices.items():
        m = np.asarray(m, dtype=np.float64)
        row_sum = m.sum(axis=1)
        is_leaf_or_inert = (np.diag(m) == 1) & (row_sum == 1)
        # a node can also be entirely absent from a *partial* hierarchy
        # (tourismlarge's geography/purpose hierarchies each cover only a
        # subset of nodes -- see hierarchy_data/levels.py's compute_levels
        # docstring), giving an all-zero row rather than a self-identity
        # one. Without excluding these too, agg_var gets clipped to the eps
        # floor and the JSD ratio explodes -- this is what caused
        # tourismlarge's DCE to come back in the hundreds of millions.
        is_absent = row_sum == 0
        not_scored = is_leaf_or_inert | is_absent
        parent_idx = np.where(~not_scored)[0]
        if len(parent_idx) == 0:
            continue

        agg_mu = m @ mu  # sum_j phi_ij * mu_j, phi_ij = 1
        agg_var = m @ var  # sum_j phi_ij^2 * sigma_j^2, phi_ij = 1
        agg_var = np.clip(agg_var, eps, None)

        diff_sq = (mu - agg_mu) ** 2
        term1 = (var + diff_sq) / (4 * agg_var)
        term2 = (agg_var + diff_sq) / (4 * var)
        per_node_jsd = term1 + term2 - 0.5

        per_group[g] = float(np.mean(per_node_jsd[parent_idx]))

    overall = float(np.mean(list(per_group.values()))) if per_group else 0.0
    return overall, per_group


def compute_all_metrics(y_true, mu, sigma, node_scale=None):
    """Convenience bundle of every accuracy/calibration metric (excludes
    DCE, which needs the full hmatrix rather than a node subset -- compute
    that separately over all nodes, not per level, since consistency is a
    property of a node's relation to its children, not something a level
    subset alone can measure)."""
    return {
        "rmse": rmse(y_true, mu),
        "mape": mape(y_true, mu),
        "wape": wape(y_true, mu),
        "crps": crps_gaussian(y_true, mu, sigma),
        "log_score": log_score(y_true, mu, sigma, node_scale=node_scale),
        "calibration_score": calibration_score(y_true, mu, sigma),
    }


def per_level_metrics(y_true, mu, sigma, level_of_node, node_scale=None):
    """Breaks out every metric in `compute_all_metrics` by hierarchy level,
    plus an 'overall' entry across all nodes -- matching the paper's
    Table 6/7 style per-level breakdown (root = level 1).

    y_true, mu, sigma, node_scale: (N, T) or (N,) arrays, N = number of
        nodes.
    level_of_node: (N,) array/list giving each node's hierarchy depth.

    Returns dict["overall" | "level_<k>" -> metrics dict].
    """
    y_true = np.asarray(y_true)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    level_of_node = np.asarray(level_of_node)
    if node_scale is not None:
        node_scale = np.asarray(node_scale)

    results = {"overall": compute_all_metrics(y_true, mu, sigma, node_scale)}
    for lvl in sorted(set(level_of_node.tolist())):
        mask = level_of_node == lvl
        ns = node_scale[mask] if node_scale is not None else None
        results[f"level_{lvl}"] = compute_all_metrics(y_true[mask], mu[mask], sigma[mask], ns)
    return results
