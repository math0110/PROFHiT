"""
Train PROFHiT on the paper's own bundled datasets: labour, tourismsmall,
traffic, wiki2, m5, tourismlarge. Reports metrics per hierarchy level too
(root = level 1), not just one dataset-wide average.

Edit DATASET below and rerun for a different dataset (same style as
train.py/train_tourism.py). Set PROFHIT_DATASET as an env var instead if
you don't want to edit the file, e.g. for scripting multiple runs.
"""
import json
import os
import time

import numpy as np
import torch as th
from tqdm import tqdm

from hierarchy_data import TourismHierarchyData
from hierarchy_data.agg_matrix import AggMatrixHierarchyData
from hierarchy_data.levels import compute_levels
from metrics import compute_all_metrics, distributional_consistency_error, per_level_metrics
from models.fnpmodels import Corem, EmbedMetaAttenSeq, RegressionSepFNP
from models.utils import float_tensor, long_tensor

SEED = 42
DEVICE = "cuda"
DATASET = os.environ.get("PROFHIT_DATASET", "labour")
PRE_TRAIN_EPOCHS = 20
TRAIN_EPOCHS = 40
EVAL_SAMPLES = 100
PRE_TRAIN_LR = 0.001
TRAIN_LR = 0.001
PRE_BATCH_SIZE = 10
BATCH_SIZE = 10
LAMBDA = 0.1
C = 5.0
FRAC_VAL = 0.1
OUT_DIR = "results"
OVERALL_ONLY = DATASET == "m5"  # m5 has 6 levels, more than any other dataset here -- skip the per-level breakdown for it

# (ahead/horizon, seasonality) per dataset -- labour/tourismlarge/wiki2
# match the paper's own Table 2 tau column; m5's horizon is the 28-day
# holdout reserved by data/m5/import_m5.py; traffic/tourismsmall aren't in
# the paper's table, so their values are reasonable defaults, not
# paper-specified.
DATASET_CONFIG = {
    "labour": dict(ahead=8, seasonality=12),
    "tourismsmall": dict(ahead=12, seasonality=4),
    "traffic": dict(ahead=7, seasonality=7),
    "wiki2": dict(ahead=1, seasonality=7),
    "m5": dict(ahead=28, seasonality=7),
    "tourismlarge": dict(ahead=12, seasonality=4),
}

np.random.seed(SEED)
th.manual_seed(SEED)
th.cuda.manual_seed(SEED)

if DEVICE == "cuda":
    device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
else:
    device = th.device("cpu")

cfg = DATASET_CONFIG[DATASET]

if DATASET == "tourismlarge":
    data_obj = TourismHierarchyData()
    levels_by_hierarchy = {
        "geography": compute_levels(data_obj.nodes1),
        "purpose": compute_levels(data_obj.nodes2),
    }

    def build_hmatrix(nodes, n):
        m = np.zeros((n, n), dtype=np.float32)
        for nd in nodes:
            if len(nd.children) == 0:
                m[nd.idx, nd.idx] = 1.0
            else:
                m[nd.idx, [c.idx for c in nd.children]] = 1.0
        return m

    n_all = data_obj.data.shape[0]
    hmatrices_np = {
        "geography": build_hmatrix(data_obj.nodes1, n_all),
        "purpose": build_hmatrix(data_obj.nodes2, n_all),
    }
else:
    data_obj = AggMatrixHierarchyData(f"data/{DATASET}")
    levels_by_hierarchy = {"tree": data_obj.levels()}
    hmatrices_np = {"tree": data_obj.generate_hmatrix()}

NUM_NODES = data_obj.data.shape[0]
AHEAD = cfg["ahead"]
TOTAL_STEPS = data_obj.data.shape[1]
TRAIN_UPTO = TOTAL_STEPS - AHEAD
BACKUP_TIME = int(max(4, min(2 * cfg["seasonality"], TRAIN_UPTO // 3)))

print(
    f"Loaded {DATASET}: n_nodes={NUM_NODES} total_steps={TOTAL_STEPS} "
    f"ahead={AHEAD} hierarchies={list(hmatrices_np.keys())}"
)

full_data = data_obj.data[:, :TRAIN_UPTO]
train_means = np.mean(full_data, axis=1)
train_std = np.std(full_data, axis=1)
train_std = np.where(train_std < 1e-6, 1.0, train_std)
train_data = (full_data - train_means[:, None]) / train_std[:, None]

encoder = EmbedMetaAttenSeq(
    dim_seq_in=1, num_metadata=NUM_NODES, dim_metadata=1, dim_out=60, n_layers=2, bidirectional=True,
).to(device)
decoder = RegressionSepFNP(
    dim_x=60, dim_y=1, dim_h=60, n_layers=3, dim_u=60, dim_z=60, nodes=NUM_NODES,
).to(device)

pre_opt = th.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=PRE_TRAIN_LR)

perm = np.random.permutation(np.arange(BACKUP_TIME, TRAIN_UPTO))
n_val = max(1, int(len(perm) * FRAC_VAL))
train_idx = perm[:-n_val]
val_idx = perm[-n_val:]


def pretrain_epoch():
    encoder.train()
    decoder.train()
    losses = []
    pre_opt.zero_grad()
    ref_x = float_tensor(train_data[:, :, None])
    meta_x = long_tensor(np.arange(NUM_NODES))
    for count, i in enumerate(train_idx):
        x = ref_x[:, : i - 1, :]
        y = ref_x[:, i, :]
        ref_out_x = encoder(ref_x, meta_x)
        out_x = encoder(x, meta_x)
        _, _, log_py, log_pqz, _ = decoder(ref_out_x, out_x, y)
        loss = -(log_py + log_pqz) / x.shape[0]
        loss.backward()
        losses.append(loss.detach().cpu().item())
        if (count + 1) % PRE_BATCH_SIZE == 0:
            pre_opt.step()
            pre_opt.zero_grad()
    if len(train_idx) % PRE_BATCH_SIZE != 0:
        pre_opt.step()
    return float(np.mean(losses))


def pre_validate():
    encoder.eval()
    decoder.eval()
    losses = []
    ref_x = float_tensor(train_data[:, :, None])
    meta_x = long_tensor(np.arange(NUM_NODES))
    for i in val_idx:
        x = ref_x[:, : i - 1, :]
        y = ref_x[:, i, :]
        ref_out_x = encoder(ref_x, meta_x)
        out_x = encoder(x, meta_x)
        y_pred, _, _, _ = decoder.predict(ref_out_x, out_x, sample=False)
        losses.append(np.mean((y_pred.cpu().numpy() - y.cpu().numpy()) ** 2))
    return float(np.mean(losses))


t0 = time.time()
print("Pretraining...")
for ep in tqdm(range(PRE_TRAIN_EPOCHS)):
    loss = pretrain_epoch()
    with th.no_grad():
        val_loss = pre_validate()
    print(f"Epoch {ep} loss: {loss:.4f} val_loss: {val_loss:.4f}")

corem = Corem(nodes=NUM_NODES, c=C).to(device)
all_params = list(encoder.parameters()) + list(decoder.parameters()) + list(corem.parameters())
opt = th.optim.Adam(all_params, lr=TRAIN_LR)

hmatrices = {g: float_tensor(m) for g, m in hmatrices_np.items()}
th_means = float_tensor(train_means)
th_std = float_tensor(train_std)

JSD_VAR_EPS = 1e-4  # floor on variance terms before they're divided by, avoids NaN blowups


def jsd_norm(mu1, mu2, var1, var2):
    mu_diff = mu1 - mu2
    t1 = 0.5 * (mu_diff ** 2 + (var1) ** 2) / (2 * (var2) ** 2)
    t2 = 0.5 * (mu_diff ** 2 + (var2) ** 2) / (2 * (var1) ** 2)
    return t1 + t2 - 1.0


def jsd_loss(mu, logstd, hmatrix, train_means, train_std):
    lhs_mu = (((mu * train_std + train_means) * hmatrix).sum(1) - train_means) / (train_std)
    lhs_var = (((th.exp(2.0 * logstd) * (train_std ** 2)) * hmatrix).sum(1)) / (train_std ** 2)
    lhs_var = th.clamp(lhs_var, min=JSD_VAR_EPS)
    own_var = th.clamp((2.0 * logstd).exp(), min=JSD_VAR_EPS)
    return th.nan_to_num(jsd_norm(mu, lhs_mu, own_var, lhs_var)).mean()


def train_epoch():
    encoder.train()
    decoder.train()
    corem.train()
    losses = []
    opt.zero_grad()
    ref_x = float_tensor(train_data[:, :, None])
    meta_x = long_tensor(np.arange(NUM_NODES))
    for count, i in enumerate(train_idx):
        x = ref_x[:, : i - 1, :]
        y = ref_x[:, i, :]
        ref_out_x = encoder(ref_x, meta_x)
        out_x = encoder(x, meta_x)
        mean_sample1, logstd_sample1, _, log_pqz, _ = decoder(ref_out_x, out_x, y)
        mean_sample, logstd_sample, log_py, _ = corem(mean_sample1.squeeze(), logstd_sample1.squeeze(), y)
        loss1 = -(log_py + log_pqz) / x.shape[0]
        loss2 = sum(
            jsd_loss(mean_sample.squeeze(), logstd_sample.squeeze(), hm, th_means, th_std)
            for hm in hmatrices.values()
        ) / x.shape[0]
        loss = loss1 + LAMBDA * loss2
        if th.isnan(loss):
            continue
        loss.backward()
        losses.append(loss.detach().cpu().item())
        if (count + 1) % BATCH_SIZE == 0:
            th.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
            opt.step()
            opt.zero_grad()
    if len(train_idx) % BATCH_SIZE != 0:
        th.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
        opt.step()
    return float(np.mean(losses)) if losses else float("nan")


def validate():
    encoder.eval()
    decoder.eval()
    corem.eval()
    losses = []
    ref_x = float_tensor(train_data[:, :, None])
    meta_x = long_tensor(np.arange(NUM_NODES))
    for i in val_idx:
        x = ref_x[:, : i - 1, :]
        y = ref_x[:, i, :]
        ref_out_x = encoder(ref_x, meta_x)
        out_x = encoder(x, meta_x)
        y_pred, mean_y, logstd_y, _ = decoder.predict(ref_out_x, out_x, sample=False)
        y_pred, _, _, _ = corem.predict(mean_y.squeeze(), logstd_y.squeeze(), sample=False)
        losses.append(np.mean((y_pred.cpu().numpy() - y.cpu().numpy()) ** 2))
    return float(np.mean(losses))


print("Training....")
for ep in tqdm(range(TRAIN_EPOCHS)):
    loss = train_epoch()
    with th.no_grad():
        val_loss = validate()
    print(f"Epoch {ep} loss: {loss:.4f} val_loss: {val_loss:.4f}")


def sample_forecast():
    curr_data = train_data.copy()
    encoder.eval()
    decoder.eval()
    corem.eval()
    meta_x = long_tensor(np.arange(NUM_NODES))
    for _ in range(AHEAD):
        ref_x = float_tensor(train_data[:, :, None])
        x = float_tensor(curr_data[:, :, None])
        ref_out_x = encoder(ref_x, meta_x)
        out_x = encoder(x, meta_x)
        _, mean_y, logstd_y, _ = decoder.predict(ref_out_x, out_x, sample=False)
        y_pred, _, _, _ = corem.predict(mean_y.squeeze(), logstd_y.squeeze(), sample=True)
        curr_data = np.concatenate([curr_data, y_pred.cpu().numpy()], axis=1)
    return curr_data[:, -AHEAD:]


with th.no_grad():
    preds = np.array([sample_forecast() for _ in tqdm(range(EVAL_SAMPLES))])
preds = preds * train_std[:, None] + train_means[:, None]
mean_preds = np.mean(preds, axis=0)
std_preds = np.std(preds, axis=0)

ground_truth = data_obj.data[:, TRAIN_UPTO : TRAIN_UPTO + AHEAD]

# A hierarchy's own node list may only cover a subset of all nodes
# (tourismlarge's geography/purpose hierarchies each exclude nodes
# belonging to the other), so score only the nodes actually part of it.
by_hierarchy = {}
for hier_name, levels in levels_by_hierarchy.items():
    idxs = np.array(sorted(levels.keys()))
    if OVERALL_ONLY:
        by_hierarchy[hier_name] = {
            "overall": compute_all_metrics(
                ground_truth[idxs], mean_preds[idxs], std_preds[idxs], train_std[idxs, None]
            )
        }
    else:
        level_arr = np.array([levels[i] for i in idxs])
        by_hierarchy[hier_name] = per_level_metrics(
            ground_truth[idxs], mean_preds[idxs], std_preds[idxs], level_arr, node_scale=train_std[idxs, None],
        )

dce_overall, dce_per_hierarchy = distributional_consistency_error(
    mean_preds.mean(axis=1), std_preds.mean(axis=1), hmatrices_np
)
elapsed_sec = time.time() - t0

overall = by_hierarchy[list(by_hierarchy.keys())[0]]["overall"]
print(
    f"[{DATASET}] RMSE={overall['rmse']:.4f} WAPE={overall['wape']:.4f} "
    f"CRPS={overall['crps']:.4f} DCE={dce_overall:.4f} ({elapsed_sec:.1f}s eval)"
)

os.makedirs(OUT_DIR, exist_ok=True)
out_path = f"{OUT_DIR}/{DATASET}.json"
with open(out_path, "w") as f:
    json.dump(
        {
            "dataset": DATASET,
            "seed": SEED,
            "pretrain_epochs": PRE_TRAIN_EPOCHS,
            "epochs": TRAIN_EPOCHS,
            "eval_samples": EVAL_SAMPLES,
            "num_nodes": NUM_NODES,
            "backup_time": BACKUP_TIME,
            "train_upto": TRAIN_UPTO,
            "ahead": AHEAD,
            "by_hierarchy": by_hierarchy,
            "dce": dce_overall,
            "dce_per_hierarchy": dce_per_hierarchy,
            "elapsed_sec": elapsed_sec,
        },
        f,
        indent=2,
    )
print(f"Saved results to {out_path}")
