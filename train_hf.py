"""
Train PROFHiT on the zaai-ai/hierarchical_time_series_datasets HF datasets
(prison, tourism, police, m5).

Generalizes train.py / train_tourism.py: instead of a fixed hierarchy (or a
hardcoded pair of hierarchies as in train_tourism.py), this loops over
GroupedHierarchyData's per-grouping-variable hmatrices, summing one JSD
consistency loss term per grouping variable.

Usage:
    python train_hf.py --dataset prison --epochs 20 --pretrain-epochs 10 --seeds 1
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import properscoring as ps
import torch as th
from tqdm import tqdm

from hierarchy_data.hf_grouped import AVAILABLE_DATASETS, GroupedHierarchyData
from models.fnpmodels import Corem, EmbedMetaAttenSeq, RegressionSepFNP
from models.utils import float_tensor, long_tensor
from utils import lag_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Train PROFHiT on an HF grouped-hierarchy dataset")
    p.add_argument("--dataset", required=True, choices=AVAILABLE_DATASETS)
    p.add_argument("--seeds", type=int, default=1, help="number of independent runs")
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
    p.add_argument("--backup-time", type=int, default=None,
                    help="override auto-picked history window length")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def pick_backup_time(data_obj, override=None):
    if override is not None:
        return override
    return int(max(4, min(2 * data_obj.seasonality, data_obj.train_steps // 3)))


def generate_hmatrices_tensor(data_obj):
    mats = data_obj.generate_hmatrices()
    return {g: float_tensor(m) for g, m in mats.items()}


def jsd_norm(mu1, mu2, var1, var2):
    mu_diff = mu1 - mu2
    t1 = 0.5 * (mu_diff ** 2 + (var1) ** 2) / (2 * (var2) ** 2)
    t2 = 0.5 * (mu_diff ** 2 + (var2) ** 2) / (2 * (var1) ** 2)
    return t1 + t2 - 1.0


def jsd_loss(mu, logstd, hmatrix, train_means, train_std):
    lhs_mu = (((mu * train_std + train_means) * hmatrix).sum(1) - train_means) / (
        train_std
    )
    lhs_var = (((th.exp(2.0 * logstd) * (train_std ** 2)) * hmatrix).sum(1)) / (
        train_std ** 2
    )
    ans = th.nan_to_num(jsd_norm(mu, lhs_mu, (2.0 * logstd).exp(), lhs_var))
    return ans.mean()


def run_one_seed(args, data_obj, seed):
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)

    device = th.device("cuda") if (args.device == "cuda" and th.cuda.is_available()) else th.device("cpu")

    num_nodes = data_obj.n_nodes
    ahead = data_obj.h
    train_upto = data_obj.train_steps
    backup_time = pick_backup_time(data_obj, args.backup_time)

    full_data = data_obj.train_data  # (N, train_upto)
    train_means = np.mean(full_data, axis=1)
    train_std = np.std(full_data, axis=1)
    train_std = np.where(train_std < 1e-6, 1.0, train_std)  # guard constant series
    train_data = (full_data - train_means[:, None]) / train_std[:, None]

    encoder = EmbedMetaAttenSeq(
        dim_seq_in=1,
        num_metadata=num_nodes,
        dim_metadata=1,
        dim_out=60,
        n_layers=2,
        bidirectional=True,
    ).to(device)
    decoder = RegressionSepFNP(
        dim_x=60,
        dim_y=1,
        dim_h=60,
        n_layers=3,
        dim_u=60,
        dim_z=60,
        nodes=num_nodes,
    ).to(device)

    pre_opt = th.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=args.pretrain_lr
    )

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

    for ep in tqdm(range(args.pretrain_epochs), desc=f"[{data_obj.name}] pretrain"):
        loss = pretrain_epoch()
        with th.no_grad():
            val_loss = pre_validate()
        tqdm.write(f"  pretrain epoch {ep}: loss={loss:.4f} val_loss={val_loss:.4f}")

    corem = Corem(nodes=num_nodes, c=args.c).to(device)
    opt = th.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + list(corem.parameters()),
        lr=args.train_lr,
    )

    hmatrices = generate_hmatrices_tensor(data_obj)
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
                jsd_loss(
                    mean_sample.squeeze(), logstd_sample.squeeze(), hm, th_means, th_std
                )
                for hm in hmatrices.values()
            ) / x.shape[0]
            loss = loss1 + args.lam * loss2
            if th.isnan(loss):
                continue
            loss.backward()
            losses.append(loss.detach().cpu().item())
            if (count + 1) % args.batch_size == 0:
                opt.step()
                opt.zero_grad()
        if len(train_idx) % args.batch_size != 0:
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

    for ep in tqdm(range(args.epochs), desc=f"[{data_obj.name}] train"):
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
            [sample_forecast() for _ in tqdm(range(args.eval_samples), desc=f"[{data_obj.name}] sampling")]
        )
    preds = preds * train_std[:, None] + train_means[:, None]  # (S, N, ahead)
    mean_preds = np.mean(preds, axis=0)  # (N, ahead)

    ground_truth = data_obj.ground_truth_horizon  # (N, ahead)
    rmse = float(np.sqrt(np.mean((ground_truth - mean_preds) ** 2)))
    mape = float(
        np.mean(np.abs((ground_truth - mean_preds) / np.where(ground_truth == 0, 1e-6, ground_truth)))
    )
    crps = float(
        ps.crps_ensemble(ground_truth, np.moveaxis(preds, [1, 2, 0], [0, 1, 2])).mean()
    )

    return {
        "seed": seed,
        "rmse": rmse,
        "mape": mape,
        "crps": crps,
        "num_nodes": num_nodes,
        "backup_time": backup_time,
    }


def main():
    args = parse_args()
    data_obj = GroupedHierarchyData(args.dataset)
    print(
        f"Loaded {args.dataset}: n_nodes={data_obj.n_nodes} n_leaf={data_obj.n_leaf} "
        f"train_steps={data_obj.train_steps} h={data_obj.h} groups={list(data_obj.groups_n.keys())}"
    )

    results = []
    for seed in range(args.seeds):
        t0 = time.time()
        res = run_one_seed(args, data_obj, seed)
        res["elapsed_sec"] = time.time() - t0
        print(f"[{args.dataset}] seed={seed} -> RMSE={res['rmse']:.4f} MAPE={res['mape']:.4f} CRPS={res['crps']:.4f} ({res['elapsed_sec']:.1f}s)")
        results.append(res)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump({"dataset": args.dataset, "args": vars(args), "runs": results}, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
