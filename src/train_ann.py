"""
train_ann.py
============
Coarse relative-pose classifier (ANN) for the mini-project

    "Vision-Based Precision Pose Estimation For Autonomous Formation Flying"
    (reference: R. Punnoose, Stanford)

PAPER (Section IV-D): 42 inputs (x pixel, y pixel, occlusion flag for each of the 14
structural features), two 100-neuron hidden layers, 4800-way softmax output, ReLU,
Adam. The paper reports 54% validation accuracy (on its own ray-traced data).

THIS FILE: exactly that architecture, trained on the CSV made by generate_data.py.
Things the paper does NOT specify, so they are our choices: learning rate, batch
size, number of epochs, learning-rate schedule, input normalisation, the 80/20
stratified split. Our accuracy is NOT expected to equal 54% - it depends on our
simplified aircraft geometry, camera and noise, and is whatever the run prints.

INPUT PREPROCESSING
  * pixel coordinates are centred and scaled:  (u - cx)/(W/2), (v - cy)/(H/2)
  * features flagged as occluded/undetected have their coordinates set to 0
    (= image centre after centring) so the dummy (0,0) pixel in the CSV is not
    mistaken for a real measurement; the occlusion flag still tells the network
  * all 42 columns are then standardised with the TRAINING-set mean / std

CLASS ID (identical to generate_data.py)
    class = ((ix * 10 + iy) * 8 + iz) * 6 + ir

Run:  python src/train_ann.py [--epochs 60] [--hidden 100] [--batch-size 256] [--lr 0.001]
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CSV_PATH = DATA_DIR / "pose_dataset.csv"
INFO_PATH = DATA_DIR / "dataset_info.json"
MODEL_PATH = MODELS_DIR / "pose_ann.pt"
NORM_PATH = MODELS_DIR / "normalization.json"

# ----------------------------------------------------------------------------
# Pose discretisation (Table I of the paper) - MUST match generate_data.py
# ----------------------------------------------------------------------------
X_MIN, X_MAX, N_X = -100.0, 50.0, 10
Y_MIN, Y_MAX, N_Y = -50.0, 50.0, 10
Z_MIN, Z_MAX, N_Z = -20.0, 20.0, 8
P_MIN, P_MAX, N_PHI = -45.0, 45.0, 6
NUM_CLASSES = N_X * N_Y * N_Z * N_PHI            # 4800
NUM_FEATURES = 14
NUM_INPUTS = NUM_FEATURES * 3                    # 42

MINS = np.array([X_MIN, Y_MIN, Z_MIN, P_MIN])
MAXS = np.array([X_MAX, Y_MAX, Z_MAX, P_MAX])
NBINS = np.array([N_X, N_Y, N_Z, N_PHI])
WIDTHS = (MAXS - MINS) / NBINS
DIM_NAMES = ["x", "y", "z", "phi"]


def encode_class(ix, iy, iz, ir):
    return ((np.asarray(ix) * N_Y + np.asarray(iy)) * N_Z + np.asarray(iz)) * N_PHI + np.asarray(ir)


def decode_class(c):
    c = np.asarray(c)
    ir = c % N_PHI
    iz = (c // N_PHI) % N_Z
    iy = (c // (N_PHI * N_Z)) % N_Y
    ix = c // (N_PHI * N_Z * N_Y)
    return ix, iy, iz, ir


def pose_to_class(poses):
    idx = np.clip(np.floor((poses - MINS) / WIDTHS).astype(int), 0, NBINS - 1)
    return encode_class(idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3])


def class_to_center(c):
    """class id -> centre pose [x, y, z, phi_deg] of that cell, shape (N,4)."""
    ix, iy, iz, ir = decode_class(c)
    idx = np.stack([ix, iy, iz, ir], axis=1)
    return MINS + (idx + 0.5) * WIDTHS


# ----------------------------------------------------------------------------
# Model: 42 -> 100 -> 100 -> 4800 (ReLU), as in the paper
# ----------------------------------------------------------------------------
class PoseANN(nn.Module):
    def __init__(self, in_dim=NUM_INPUTS, hidden=100, n_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),      # logits; softmax is inside CrossEntropyLoss
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------------
# Data handling
# ----------------------------------------------------------------------------
def preprocess_raw(raw, img):
    """
    raw: (N,42) in CSV order [u1,v1,occ1,u2,v2,occ2,...]  ->  (N,42) float32, same order,
    centred / scaled, occluded coordinates zeroed. `img` = dict with width,height,cx,cy.
    """
    r = np.asarray(raw, dtype=np.float32).reshape(-1, NUM_FEATURES, 3)
    occ = r[:, :, 2]
    visible = occ < 0.5
    u = np.where(visible, (r[:, :, 0] - img["cx"]) / (img["width"] / 2.0), 0.0)
    v = np.where(visible, (r[:, :, 1] - img["cy"]) / (img["height"] / 2.0), 0.0)
    return np.stack([u, v, occ], axis=2).reshape(-1, NUM_INPUTS).astype(np.float32)


def load_dataset():
    if not CSV_PATH.exists() or not INFO_PATH.exists():
        sys.exit(f"Dataset not found. Run  python src/generate_data.py  first.\n(expected {CSV_PATH})")
    info = json.loads(INFO_PATH.read_text())

    # --- the two scripts must agree on the pose discretisation ---
    expected = {"x": [X_MIN, X_MAX, N_X], "y": [Y_MIN, Y_MAX, N_Y],
                "z": [Z_MIN, Z_MAX, N_Z], "phi_deg": [P_MIN, P_MAX, N_PHI]}
    if info["bins"] != expected or info["num_classes"] != NUM_CLASSES:
        sys.exit("Pose discretisation in dataset_info.json differs from train_ann.py - regenerate the data.")

    df = pd.read_csv(CSV_PATH)
    cols = info["feature_columns"]
    assert len(cols) == NUM_INPUTS, "expected 42 feature columns"
    raw = df[cols].values.astype(np.float32)
    labels = df["pose_class"].values.astype(np.int64)
    poses = df[["pose_x", "pose_y", "pose_z", "pose_phi_deg"]].values.astype(np.float64)

    # --- sanity: labels must equal the bin of the stored continuous pose ---
    if not np.array_equal(pose_to_class(poses), labels):
        sys.exit("Label / pose mismatch: pose_class is not the bin of (pose_x..pose_phi_deg). Regenerate the data.")
    if np.isnan(raw).any():
        sys.exit("NaNs found in the feature columns.")
    if labels.min() < 0 or labels.max() >= NUM_CLASSES:
        sys.exit("Labels outside 0..4799.")
    return raw, labels, poses, info


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def neighbourhood_metrics(pred, true):
    """
    Pose-neighbourhood metric (the paper notes wrong labels are usually adjacent bins).
    Compares predicted and true class in the 4 bin indices.
      within_k = fraction with |bin difference| <= k in ALL four dimensions.
    """
    p = decode_class(pred)
    t = decode_class(true)
    diffs = [np.abs(a - b) for a, b in zip(p, t)]
    out = {
        "top1_exact": float(np.mean(np.all([d == 0 for d in diffs], axis=0))),
        "within_1_bin_all_dims": float(np.mean(np.all([d <= 1 for d in diffs], axis=0))),
        "within_2_bins_all_dims": float(np.mean(np.all([d <= 2 for d in diffs], axis=0))),
    }
    for name, d in zip(DIM_NAMES, diffs):
        out[f"per_dim_exact_{name}"] = float(np.mean(d == 0))
        out[f"per_dim_within1_{name}"] = float(np.mean(d <= 1))
    return out


def physical_errors(pred, true_labels, true_poses):
    """
    Error of the predicted bin CENTRE w.r.t. the true continuous pose (paper Fig. 4 idea),
    and the irreducible quantisation error if the classifier were perfect.
    """
    cp = class_to_center(pred)
    co = class_to_center(true_labels)

    def stats(c):
        pos = np.linalg.norm(c[:, :3] - true_poses[:, :3], axis=1)
        rol = np.abs(c[:, 3] - true_poses[:, 3])
        return {"pos_median_m": float(np.median(pos)), "pos_mean_m": float(pos.mean()),
                "pos_p90_m": float(np.percentile(pos, 90)),
                "roll_median_deg": float(np.median(rol)), "roll_mean_deg": float(rol.mean())}
    return {"predicted_class_centre": stats(cp), "perfect_classifier_centre": stats(co)}


def evaluate(model, X, y, batch=8192):
    """Returns loss, top-1, top-5, predictions (numpy). X, y are torch tensors."""
    model.eval()
    crit_sum = nn.CrossEntropyLoss(reduction="sum")
    total, c1, c5 = 0.0, 0, 0
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb, yb = X[i:i + batch], y[i:i + batch]
            logits = model(xb)
            total += crit_sum(logits, yb).item()
            top5 = logits.topk(5, dim=1).indices
            c1 += (top5[:, 0] == yb).sum().item()
            c5 += (top5 == yb[:, None]).any(dim=1).sum().item()
            preds.append(top5[:, 0].cpu())
    n = len(X)
    return total / n, c1 / n, c5 / n, torch.cat(preds).numpy()


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def make_plots(hist, pred, true):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plots (pip install matplotlib).")
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ep = hist["epoch"]

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    ax[0].plot(ep, hist["train_loss"], label="training loss")
    ax[0].plot(ep, hist["val_loss"], label="validation loss")
    ax[0].set_title("Cross-entropy loss vs epoch")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(ep, [100 * a for a in hist["val_top1"]], label="top-1")
    ax[1].plot(ep, [100 * a for a in hist["val_top5"]], label="top-5")
    ax[1].plot(ep, [100 * a for a in hist["val_within1"]], label="within +-1 bin (all 4 dims)")
    ax[1].set_title("Validation accuracy vs epoch")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy [%]"); ax[1].legend(); ax[1].grid(alpha=0.3)

    ax[2].plot(ep, [100 * a for a in hist["val_top1"]], color="C0")
    ax[2].set_title("Validation top-1 accuracy vs epoch")
    ax[2].set_xlabel("epoch"); ax[2].set_ylabel("accuracy [%]"); ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_curves.png", dpi=130)
    plt.close(fig)

    # histogram of (predicted bin - true bin) per dimension, like paper Fig. 5
    p, t = decode_class(pred), decode_class(true)
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for a, name, pi, ti, n in zip(ax, DIM_NAMES, p, t, NBINS):
        d = pi - ti
        vals = np.arange(-(n - 1), n)
        counts = np.array([(d == v).sum() for v in vals])
        a.bar(vals, counts, color="C0")
        a.set_yscale("log")
        a.set_title(f"{name}: predicted bin - true bin")
        a.set_xlabel("bin difference"); a.set_ylabel("count (log)"); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "bin_error_histograms.png", dpi=130)
    plt.close(fig)
    print(f"Saved plots: {RESULTS_DIR / 'training_curves.png'}, {RESULTS_DIR / 'bin_error_histograms.png'}")


# ----------------------------------------------------------------------------
# Inference helpers (for the later particle-filter step)
# ----------------------------------------------------------------------------
def load_trained_model(model_path=MODEL_PATH, norm_path=NORM_PATH):
    """Returns (model, norm_dict). Use predict_topk() afterwards."""
    ckpt = torch.load(model_path, map_location="cpu")
    model = PoseANN(hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    norm = json.loads(Path(norm_path).read_text())
    return model, norm


def predict_topk(model, norm, raw_features, k=5):
    """
    raw_features: (N,42) in CSV format [u1,v1,occ1,...] (undetected: u=v=0, occ=1).
    Returns (class_ids (N,k), probabilities (N,k)).
    """
    x = preprocess_raw(raw_features, norm["image"])
    x = (x - np.array(norm["mean"], dtype=np.float32)) / np.array(norm["std"], dtype=np.float32)
    with torch.no_grad():
        prob = torch.softmax(model(torch.from_numpy(x.astype(np.float32))), dim=1)
        top = prob.topk(k, dim=1)
    return top.indices.numpy(), top.values.numpy()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=100, help="paper: 100")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--no-lr-decay", action="store_true", help="constant learning rate")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1-2. load data, separate inputs / labels --------------------------------------
    raw, labels, poses, info = load_dataset()
    print(f"Loaded {len(labels)} samples, {raw.shape[1]} input values, {len(np.unique(labels))} classes")
    X_all = preprocess_raw(raw, info["image"])

    # ---- 4. stratified split ------------------------------------------------------------
    idx = np.arange(len(labels))
    tr, va = train_test_split(idx, test_size=args.val_fraction, stratify=labels, random_state=args.seed)
    print(f"Stratified split: {len(tr)} train / {len(va)} validation "
          f"(min samples per class in validation: {np.bincount(labels[va], minlength=NUM_CLASSES).min()})")

    # ---- 3. standardise with TRAIN statistics -------------------------------------------
    mean = X_all[tr].mean(axis=0)
    std = X_all[tr].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    X_std = ((X_all - mean) / std).astype(np.float32)
    NORM_PATH.write_text(json.dumps({
        "mean": mean.tolist(), "std": std.tolist(), "image": info["image"],
        "input_order": "feature_k: u_norm, v_norm, occlusion (k=1..14)",
        "preprocessing": "u=(u-cx)/(W/2), v=(v-cy)/(H/2), zeroed if occluded; then (x-mean)/std",
    }, indent=2))

    Xtr = torch.from_numpy(X_std[tr]); ytr = torch.from_numpy(labels[tr])
    Xva = torch.from_numpy(X_std[va]); yva = torch.from_numpy(labels[va])

    # ---- 5-8. model, loss, optimiser ----------------------------------------------------
    model = PoseANN(hidden=args.hidden)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: 42 -> {args.hidden} -> {args.hidden} -> {NUM_CLASSES}  ({n_params:,} parameters)")
    if args.hidden != 100:
        print("NOTE: hidden size differs from the paper (100) - report this as a deviation.")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None if args.no_lr_decay else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"Adam lr={args.lr}, batch={args.batch_size}, epochs={args.epochs}, "
          f"cosine LR decay={'off' if scheduler is None else 'on'}")
    print(f"Chance level: top-1 = {100 / NUM_CLASSES:.3f}%, top-5 = {500 / NUM_CLASSES:.3f}%\n")

    hist = {k: [] for k in ["epoch", "train_loss", "val_loss", "val_top1", "val_top5", "val_within1"]}
    best = {"top1": -1.0, "state": None, "epoch": 0}
    n_train = len(Xtr)
    t_start = time.time()

    # ---- 9-11. mini-batch training ------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n_train)
        running = 0.0
        for i in range(0, n_train, args.batch_size):
            b = perm[i:i + args.batch_size]
            optimizer.zero_grad()
            loss = criterion(model(Xtr[b]), ytr[b])
            loss.backward()
            optimizer.step()
            running += loss.item() * len(b)
        train_loss = running / n_train
        if scheduler is not None:
            scheduler.step()

        val_loss, top1, top5, pred = evaluate(model, Xva, yva)
        within1 = neighbourhood_metrics(pred, labels[va])["within_1_bin_all_dims"]
        for k, v in zip(hist.keys(), [epoch, train_loss, val_loss, top1, top5, within1]):
            hist[k].append(v)
        if top1 > best["top1"]:
            best = {"top1": top1, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        print(f"epoch {epoch:3d}/{args.epochs} | train loss {train_loss:7.4f} | val loss {val_loss:7.4f} | "
              f"val top-1 {100 * top1:6.2f}% | top-5 {100 * top5:6.2f}% | within+-1bin {100 * within1:6.2f}% "
              f"| {time.time() - t0:4.1f}s")

    print(f"\nTraining finished in {(time.time() - t_start) / 60:.1f} min. "
          f"Best validation top-1 at epoch {best['epoch']}.")

    # ---- final evaluation with the best checkpoint --------------------------------------
    model.load_state_dict(best["state"])
    val_loss, top1, top5, pred = evaluate(model, Xva, yva)
    nb = neighbourhood_metrics(pred, labels[va])
    phys = physical_errors(pred, labels[va], poses[va])

    print("\n" + "=" * 62)
    print("FINAL VALIDATION RESULTS (best checkpoint)")
    print("=" * 62)
    print(f"Validation loss                      : {val_loss:.4f}")
    print(f"Top-1 accuracy                       : {100 * top1:.2f}%   (chance {100 / NUM_CLASSES:.3f}%)")
    print(f"Top-5 accuracy                       : {100 * top5:.2f}%")
    print(f"Within +-1 bin in ALL 4 dims         : {100 * nb['within_1_bin_all_dims']:.2f}%")
    print(f"Within +-2 bins in ALL 4 dims        : {100 * nb['within_2_bins_all_dims']:.2f}%")
    print("Per-dimension exact bin accuracy     : " +
          ", ".join(f"{n}={100 * nb[f'per_dim_exact_{n}']:.1f}%" for n in DIM_NAMES))
    print("Per-dimension within +-1 bin         : " +
          ", ".join(f"{n}={100 * nb[f'per_dim_within1_{n}']:.1f}%" for n in DIM_NAMES))
    pc, po = phys["predicted_class_centre"], phys["perfect_classifier_centre"]
    print(f"Position error of predicted centre   : median {pc['pos_median_m']:.1f} m, mean {pc['pos_mean_m']:.1f} m, "
          f"90th pct {pc['pos_p90_m']:.1f} m")
    print(f"  (a PERFECT classifier would give   : median {po['pos_median_m']:.1f} m, mean {po['pos_mean_m']:.1f} m "
          f"- quantisation floor)")
    print(f"Roll error of predicted centre       : median {pc['roll_median_deg']:.1f} deg, mean {pc['roll_mean_deg']:.1f} deg "
          f"(perfect: median {po['roll_median_deg']:.1f} deg)")
    print("Paper reference                      : ~54% top-1 (different data; not directly comparable)")

    if top1 < 0.10:
        print("\nWARNING: top-1 below 10%. Diagnostics to try:")
        print("  1. Re-run generate_data.py and confirm 'ALL CHECKS PASSED' and the sensitivity table.")
        print("  2. Check 'within +-1 bin' and per-dimension numbers - if those are high, the network learns")
        print("     the pose and only the exact 4800-way bin is hard (bin boundaries).")
        print("  3. Train longer (--epochs 120) or try --hidden 256 (state it as a deviation from the paper).")

    # ---- 13-15. save model, normalisation, metrics, plots -------------------------------
    torch.save({"state_dict": best["state"], "hidden": args.hidden, "in_dim": NUM_INPUTS,
                "num_classes": NUM_CLASSES, "best_epoch": best["epoch"]}, MODEL_PATH)
    pd.DataFrame(hist).to_csv(RESULTS_DIR / "training_history.csv", index=False)
    metrics = {"val_loss": val_loss, "top1": top1, "top5": top5, "neighbourhood": nb, "physical": phys,
               "best_epoch": best["epoch"], "args": vars(args), "n_train": int(len(tr)), "n_val": int(len(va))}
    (RESULTS_DIR / "ann_metrics.json").write_text(json.dumps(metrics, indent=2))
    make_plots(hist, pred, labels[va])
    print(f"\nSaved model         : {MODEL_PATH}")
    print(f"Saved normalisation : {NORM_PATH}")
    print(f"Saved metrics/hist  : {RESULTS_DIR / 'ann_metrics.json'}, {RESULTS_DIR / 'training_history.csv'}")


if __name__ == "__main__":
    main()