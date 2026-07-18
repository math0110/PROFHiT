"""
Evaluation metrics matching the PROFHiT paper's Section 5.1 definitions,
for use across train.py / train_tourism.py / train_hf.py / train_tags.py.

All functions take (N, T) arrays -- N nodes, T forecast steps -- and
return a single scalar averaged over both axes.
"""
import numpy as np
from scipy.stats import norm


def mape(y_true, y_pred, min_abs_actual=None):
    # excludes near-zero y_true instead of epsilon-dividing: hierarchical
    # count data has many zero/near-zero periods, and dividing by those
    # blows MAPE up into the millions. See wape() for the more robust
    # alternative.
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if min_abs_actual is None:
        min_abs_actual = max(0.01 * np.mean(np.abs(y_true)), 1e-8)
    mask = np.abs(y_true) >= min_abs_actual
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def wape(y_true, y_pred):
    # sum|err| / sum|actual| -- well-defined even with zero-heavy count data
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
    # closed-form CRPS for a fitted Gaussian(mu, sigma), matching the
    # paper's evaluation protocol
    y_true = np.asarray(y_true, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.clip(np.asarray(sigma, dtype=np.float64), eps, None)
    z = (y_true - mu) / sigma
    crps = sigma * (
        z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)
    )
    return float(np.mean(crps))


def log_score(y_true, mu, sigma, node_scale=None, window=0.005, cap=10.0, eps=1e-6):
    # LS = -integral_{y-L}^{y+L} log p_y(y_hat) dy_hat, closed form for
    # Gaussian p_y. Computed in each node's own normalized (z-scored) units
    # rather than raw ones -- the -log(sigma) term scales with a node's
    # absolute magnitude, so within one hierarchy a large node (e.g. a
    # Total in the tens of thousands) would dominate the average versus
    # small leaf nodes regardless of relative calibration. `node_scale` is
    # each node's characteristic scale (e.g. its training std); `cap`
    # floors degenerate predictions the same way the paper's log-likelihood
    # is floored at -10.
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
    # Definition 3: CS = integral_0^1 |k_M(c) - c| dc, Riemann sum, step=0.05
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
    # Eq. 5-7: for every non-leaf node, compare its forecast distribution
    # against the distribution implied by summing its children's
    # forecasts (closed-form Gaussian JSD), averaged rather than summed
    # per group so it stays comparable across datasets of different size.
    #
    # mu, sigma: (N,) forecast mean/std per node.
    # hmatrices: dict[group_name -> (N, N) 0/1 aggregation matrix].
    # Returns (overall, per_group).
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
        # subset of nodes), giving an all-zero row -- exclude those too or
        # agg_var hits the eps floor and the JSD ratio explodes.
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
    # DCE isn't included -- it needs the full hmatrix, not just a node
    # subset, so it's computed separately over all nodes, not per level.
    return {
        "rmse": rmse(y_true, mu),
        "mape": mape(y_true, mu),
        "wape": wape(y_true, mu),
        "crps": crps_gaussian(y_true, mu, sigma),
        "log_score": log_score(y_true, mu, sigma, node_scale=node_scale),
        "calibration_score": calibration_score(y_true, mu, sigma),
    }


def per_level_metrics(y_true, mu, sigma, level_of_node, node_scale=None):
    # every compute_all_metrics() metric broken out by hierarchy level
    # (root = level 1), plus an "overall" entry across all nodes
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
