"""
Train PROFHiT on the paper's own bundled datasets: labour, tourismsmall,
traffic, wiki2, m5 (via the generic hierarchy_data/agg_matrix.py loader), and
tourismlarge (via the existing TourismHierarchyData, which already handles
its crossed geography x purpose structure with two hierarchies).

Unlike train_hf.py's grouped/crossed HF datasets, these are proper nested
trees (tourismlarge excepted, which is crossed -- two trees sharing the
same leaves), so consistency uses a single hmatrix per dataset (or two, for
tourismlarge) rather than one per grouping variable.

Also reports metrics broken out per hierarchy level (root = level 1),
matching the paper's Table 6/7 style, since a single dataset-wide average
hides very different behavior at different levels.

Usage:
    python train_tags.py --dataset labour --epochs 20 --pretrain-epochs 10 --seeds 1
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch as th
from tqdm import tqdm

from hierarchy_data import TourismHierarchyData
from hierarchy_data.agg_matrix import AggMatrixHierarchyData
from hierarchy_data.levels import compute_levels
from metrics import distributional_consistency_error, per_level_metrics
from models.fnpmodels import Corem, EmbedMetaAttenSeq, RegressionSepFNP
from models.utils import float_tensor, long_tensor

TAGS_CSV_DATASETS = ("labour", "tourismsmall", "traffic", "wiki2", "m5")

# (ahead/horizon, seasonality) per dataset. Labour/tourismlarge/wiki2 match
# the paper's own Table 2 (tau column); m5's horizon matches the 28-day
# holdout already reserved by data/m5/import_m5.py. traffic/tourismsmall
# aren't in the paper's own table (traffic isn't one of its datasets at
# all; tourismsmall is a smaller cut of the same series as tourismlarge) --
# their horizon/seasonality are reasonable defaults, not paper-specified.
DATASET_CONFIG = {
    "labour": dict(ahead=8, seasonality=12),
    "tourismsmall": dict(ahead=12, seasonality=4),
    "traffic": dict(ahead=7, seasonality=7),
    "wiki2": dict(ahead=1, seasonality=7),
    "m5": dict(ahead=28, seasonality=7),
    "tourismlarge": dict(ahead=12, seasonality=4),
}


def parse_args():
    p = argparse.ArgumentParser(description="Train PROFHiT on a paper-native hierarchy dataset")
    p.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG.keys()))
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--pretrain-epochs", type=int, default=10)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--eval-samples", type=int, default=50)
    p.add_argument("--pretrain-lr", type=float, default=0.001)
    p.add_argument("--train-lr", type=float, default=0.001)
    p.add_argument("--pretrain-batch-size", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--lambda-", dest="lam", type=float, default=0.1)
    p.add_argument("--c", type=float, default=5.0)
    p.add_argument("--frac-val", type=float, default=0.1)
    p.add_argument("--backup-time", type=int, default=None)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def load_dataset(name):
    cfg = DATASET_CONFIG[name]
    if name == "tourismlarge":
        data_obj = TourismHierarchyData()
        levels1 = compute_levels(data_obj.nodes1)
        levels2 = compute_levels(data_obj.nodes2)

        def gen_hmatrices():
            n = data_obj.data.shape[0]
            ans1 = np.zeros((n, n), dtype=np.float32)
            for nd in data_obj.nodes1:
                if len(nd.children) == 0:
                    ans1[nd.idx, nd.idx] = 1.0
                else:
                    ans1[nd.idx, [c.idx for c in nd.children]] = 1.0
            ans2 = np.zeros((n, n), dtype=np.float32)
            for nd in data_obj.nodes2:
                if len(nd.children) == 0:
                    ans2[nd.idx, nd.idx] = 1.0
                else:
                    ans2[nd.idx, [c.idx for c in nd.children]] = 1.0
            return {"geography": ans1, "purpose": ans2}

        return data_obj, gen_hmatrices, {"geography": levels1, "purpose": levels2}, cfg
    else:
        data_obj = AggMatrixHierarchyData(f"data/{name}")
        levels = data_obj.levels()
        return data_obj, lambda: {"tree": data_obj.generate_hmatrix()}, {"tree": levels}, cfg


def pick_backup_time(seasonality, train_upto, override=None):
    if override is not None:
        return override
    return int(max(4, min(2 * seasonality, train_upto // 3)))


def jsd_norm(mu1, mu2, var1, var2):
    mu_diff = mu1 - mu2
    t1 = 0.5 * (mu_diff ** 2 + (var1) ** 2) / (2 * (var2) ** 2)
    t2 = 0.5 * (mu_diff ** 2 + (var2) ** 2) / (2 * (var1) ** 2)
    return t1 + t2 - 1.0


JSD_VAR_EPS = 1e-4  # floor on variance terms before they're divided by in jsd_norm


def jsd_loss(mu, logstd, hmatrix, train_means, train_std):
    lhs_mu = (((mu * train_std + train_means) * hmatrix).sum(1) - train_means) / (train_std)
    lhs_var = (((th.exp(2.0 * logstd) * (train_std ** 2)) * hmatrix).sum(1)) / (train_std ** 2)
    lhs_var = th.clamp(lhs_var, min=JSD_VAR_EPS)
    own_var = th.clamp((2.0 * logstd).exp(), min=JSD_VAR_EPS)
    ans = th.nan_to_num(jsd_norm(mu, lhs_mu, own_var, lhs_var))
    return ans.mean()


def run_one_seed(args, name, data_obj, hmatrices_np, levels_by_hierarchy, cfg, seed):
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    device = th.device("cuda") if (args.device == "cuda" and th.cuda.is_available()) else th.device("cpu")

    num_nodes = data_obj.data.shape[0]
    ahead = cfg["ahead"]
    total_steps = data_obj.data.shape[1]
    train_upto = total_steps - ahead
    backup_time = pick_backup_time(cfg["seasonality"], train_upto, args.backup_time)

    full_data = data_obj.data[:, :train_upto]
    train_means = np.mean(full_data, axis=1)
    train_std = np.std(full_data, axis=1)
    train_std = np.where(train_std < 1e-6, 1.0, train_std)
    train_data = (full_data - train_means[:, None]) / train_std[:, None]

    encoder = EmbedMetaAttenSeq(
        dim_seq_in=1, num_metadata=num_nodes, dim_metadata=1, dim_out=60, n_layers=2, bidirectional=True,
    ).to(device)
    decoder = RegressionSepFNP(
        dim_x=60, dim_y=1, dim_h=60, n_layers=3, dim_u=60, dim_z=60, nodes=num_nodes,
    ).to(device)

    pre_opt = th.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.pretrain_lr)

    perm = np.random.permutation(np.arange(backup_time, train_upto))
    n_val = max(1, int(len(perm) * args.frac_val))
    train_idx = perm[:-n_val]
    val_idx = perm[-n_val:]

    def pretrain_epoch():
        encoder.train()
        decoder.train()
        losses = []
        pre_opt.zero_grad()
        ref_x = float_tensor(train_data[:, :, None])
        meta_x = long_tensor(np.arange(num_nodes))
        for count, i in enumerate(train_idx):
            x = ref_x[:, : i - 1, :]
            y = ref_x[:, i, :]
            ref_out_x = encoder(ref_x, meta_x)
            out_x = encoder(x, meta_x)
            _, _, log_py, log_pqz, _ = decoder(ref_out_x, out_x, y)
            loss = -(log_py + log_pqz) / x.shape[0]
            loss.backward()
            losses.append(loss.detach().cpu().item())
            if (count + 1) % args.pretrain_batch_size == 0:
                pre_opt.step()
                pre_opt.zero_grad()
        if len(train_idx) % args.pretrain_batch_size != 0:
            pre_opt.step()
        return float(np.mean(losses))

    def pre_validate():
        encoder.eval()
        decoder.eval()
        losses = []
        ref_x = float_tensor(train_data[:, :, None])
        meta_x = long_tensor(np.arange(num_nodes))
        for i in val_idx:
            x = ref_x[:, : i - 1, :]
            y = ref_x[:, i, :]
            ref_out_x = encoder(ref_x, meta_x)
            out_x = encoder(x, meta_x)
            y_pred, _, _, _ = decoder.predict(ref_out_x, out_x, sample=False)
            losses.append(np.mean((y_pred.cpu().numpy() - y.cpu().numpy()) ** 2))
        return float(np.mean(losses))

    for ep in tqdm(range(args.pretrain_epochs), desc=f"[{name}] pretrain"):
        loss = pretrain_epoch()
        with th.no_grad():
            val_loss = pre_validate()
        tqdm.write(f"  pretrain epoch {ep}: loss={loss:.4f} val_loss={val_loss:.4f}")

    corem = Corem(nodes=num_nodes, c=args.c).to(device)
    all_params = list(encoder.parameters()) + list(decoder.parameters()) + list(corem.parameters())
    opt = th.optim.Adam(all_params, lr=args.train_lr)

    hmatrices = {g: float_tensor(m) for g, m in hmatrices_np.items()}
    th_means = float_tensor(train_means)
    th_std = float_tensor(train_std)

    def train_epoch():
        encoder.train()
        decoder.train()
        corem.train()
        losses = []
        opt.zero_grad()
        ref_x = float_tensor(train_data[:, :, None])
        meta_x = long_tensor(np.arange(num_nodes))
        for count, i in enumerate(train_idx):
            x = ref_x[:, : i - 1, :]
            y = ref_x[:, i, :]
            ref_out_x = encoder(ref_x, meta_x)
            out_x = encoder(x, meta_x)
            mean_sample1, logstd_sample1, _, log_pqz, _ = decoder(ref_out_x, out_x, y)
            mean_sample, logstd_sample, log_py, _ = corem(
                mean_sample1.squeeze(), logstd_sample1.squeeze(), y
            )
            loss1 = -(log_py + log_pqz) / x.shape[0]
            loss2 = sum(
                jsd_loss(mean_sample.squeeze(), logstd_sample.squeeze(), hm, th_means, th_std)
                for hm in hmatrices.values()
            ) / x.shape[0]
            loss = loss1 + args.lam * loss2
            if th.isnan(loss):
                continue
            loss.backward()
            losses.append(loss.detach().cpu().item())
            if (count + 1) % args.batch_size == 0:
                th.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
                opt.step()
                opt.zero_grad()
        if len(train_idx) % args.batch_size != 0:
            th.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
            opt.step()
        return float(np.mean(losses)) if losses else float("nan")

    def validate():
        encoder.eval()
        decoder.eval()
        corem.eval()
        losses = []
        ref_x = float_tensor(train_data[:, :, None])
        meta_x = long_tensor(np.arange(num_nodes))
        for i in val_idx:
            x = ref_x[:, : i - 1, :]
            y = ref_x[:, i, :]
            ref_out_x = encoder(ref_x, meta_x)
            out_x = encoder(x, meta_x)
            y_pred, mean_y, logstd_y, _ = decoder.predict(ref_out_x, out_x, sample=False)
            y_pred, _, _, _ = corem.predict(mean_y.squeeze(), logstd_y.squeeze(), sample=False)
            losses.append(np.mean((y_pred.cpu().numpy() - y.cpu().numpy()) ** 2))
        return float(np.mean(losses))

    for ep in tqdm(range(args.epochs), desc=f"[{name}] train"):
        loss = train_epoch()
        with th.no_grad():
            val_loss = validate()
        tqdm.write(f"  train epoch {ep}: loss={loss:.4f} val_loss={val_loss:.4f}")

    def sample_forecast():
        curr_data = train_data.copy()
        encoder.eval()
        decoder.eval()
        corem.eval()
        meta_x = long_tensor(np.arange(num_nodes))
        for _ in range(ahead):
            ref_x = float_tensor(train_data[:, :, None])
            x = float_tensor(curr_data[:, :, None])
            ref_out_x = encoder(ref_x, meta_x)
            out_x = encoder(x, meta_x)
            _, mean_y, logstd_y, _ = decoder.predict(ref_out_x, out_x, sample=False)
            y_pred, _, _, _ = corem.predict(mean_y.squeeze(), logstd_y.squeeze(), sample=True)
            curr_data = np.concatenate([curr_data, y_pred.cpu().numpy()], axis=1)
        return curr_data[:, -ahead:]

    with th.no_grad():
        preds = np.array(
            [sample_forecast() for _ in tqdm(range(args.eval_samples), desc=f"[{name}] sampling")]
        )
    preds = preds * train_std[:, None] + train_means[:, None]
    mean_preds = np.mean(preds, axis=0)
    std_preds = np.std(preds, axis=0)

    ground_truth = data_obj.data[:, train_upto : train_upto + ahead]

    overall_and_levels = {}
    for hier_name, levels in levels_by_hierarchy.items():
        # A hierarchy's own node list may only be a partial subset of all
        # nodes (tourismlarge's geography/purpose hierarchies each exclude
        # nodes that belong to the other), so score only the nodes that are
        # actually part of *this* hierarchy rather than assuming every node
        # index has a level everywhere.
        idxs = np.array(sorted(levels.keys()))
        level_arr = np.array([levels[i] for i in idxs])
        overall_and_levels[hier_name] = per_level_metrics(
            ground_truth[idxs], mean_preds[idxs], std_preds[idxs], level_arr,
            node_scale=train_std[idxs, None],
        )

    dce_overall, dce_per_hierarchy = distributional_consistency_error(
        mean_preds.mean(axis=1), std_preds.mean(axis=1), hmatrices_np
    )

    return {
        "seed": seed,
        "by_hierarchy": overall_and_levels,
        "dce": dce_overall,
        "dce_per_hierarchy": dce_per_hierarchy,
        "num_nodes": num_nodes,
        "backup_time": backup_time,
        "train_upto": train_upto,
        "ahead": ahead,
    }


def main():
    args = parse_args()
    data_obj, gen_hmatrices, levels_by_hierarchy, cfg = load_dataset(args.dataset)
    hmatrices_np = gen_hmatrices()
    print(
        f"Loaded {args.dataset}: n_nodes={data_obj.data.shape[0]} total_steps={data_obj.data.shape[1]} "
        f"ahead={cfg['ahead']} hierarchies={list(hmatrices_np.keys())}"
    )

    results = []
    for seed in range(args.seeds):
        t0 = time.time()
        res = run_one_seed(args, args.dataset, data_obj, hmatrices_np, levels_by_hierarchy, cfg, seed)
        res["elapsed_sec"] = time.time() - t0
        overall = res["by_hierarchy"][list(res["by_hierarchy"].keys())[0]]["overall"]
        print(
            f"[{args.dataset}] seed={seed} -> RMSE={overall['rmse']:.4f} WAPE={overall['wape']:.4f} "
            f"CRPS={overall['crps']:.4f} DCE={res['dce']:.4f} ({res['elapsed_sec']:.1f}s)"
        )
        results.append(res)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump({"dataset": args.dataset, "args": vars(args), "runs": results}, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
