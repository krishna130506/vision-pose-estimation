"""
generate_data.py
================
Physically-based synthetic dataset for the mini-project

    "Vision-Based Precision Pose Estimation For Autonomous Formation Flying"
    (reference: R. Punnoose, Stanford)

PIPELINE (every one of the 42 inputs is a deterministic function of the pose,
plus a little measurement noise):

    relative pose (x, y, z, phi)
        -> camera placed at (x, y, z) in the leader frame, gimbaled at the leader
        -> 14 fixed 3-D structural feature points of a simplified aircraft
        -> pinhole camera projection  -> 2-D pixel coordinates (u, v)
        -> visibility test (behind camera / outside image / hidden by fuselage)
        -> small Gaussian pixel noise + small random detector drop-out
        -> 42 values = 14 x (u, v, occlusion_flag)

WHAT THE PAPER DOES vs WHAT THIS FILE DOES
  Paper : 14 features, 42 ANN inputs, 4800 pose classes (Table I), camera gimbaled
          to always point at the leader, features that are occluded are "not
          detected", images from a ray-traced aircraft model, features found by a
          CNN (CNN is out of the paper's scope).
  Here  : a simplified 14-point aircraft skeleton (NOT the paper's model - the
          paper does not publish its 14 feature locations), an analytic pinhole
          camera instead of ray tracing, and a simple ellipsoid-fuselage ray test
          for self-occlusion. The CNN feature detector is replaced by "perfect
          detection + small noise". This is a simplified student reproduction.

ASSUMPTIONS (ours, not stated in the paper)
  * Leader is straight-and-level and aligned with the follower (relative pitch/yaw
    = 0). The paper discretises only x, y, z and roll about the camera boresight.
  * Frame: leader body frame, x forward, y left, z up. The camera (follower) is at
    (x, y, z) in this frame, so x < 0 means the follower is BEHIND the leader.
    (Consistent with the paper's Fig. 6 where x goes from -90 m towards -40 m.)
  * phi = roll of the camera about its boresight (0 = leader's "up" is image-up).
  * Poses are sampled UNIFORMLY INSIDE each class cell (continuous pose), so the
    label is a quantisation of a continuous pose - like in the paper.
  * Minimum separation of 10 m between the aircraft (collision-avoidance), so a few
    cells near the origin are sampled only in the part of the cell with range>=10 m.

CLASS ID  (identical in train_ann.py)
    class = ((ix * N_Y + iy) * N_Z + iz) * N_PHI + ir
    ix in 0..9, iy in 0..9, iz in 0..7, ir in 0..5   ->   class in 0..4799
    decode: ir = c % 6; iz = (c // 6) % 8; iy = (c // 48) % 10; ix = c // 480

Run:  python src/generate_data.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths (relative to the project root, independent of the current directory)
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CSV_PATH = DATA_DIR / "pose_dataset.csv"
INFO_PATH = DATA_DIR / "dataset_info.json"

# ----------------------------------------------------------------------------
# Settings you may change
# ----------------------------------------------------------------------------
SEED = 42
SAMPLES_PER_CLASS = 30        # 30 * 4800 = 144,000 samples (use 25..50)
PIXEL_NOISE_STD = 0.5         # px, Gaussian measurement noise on detected features
DROPOUT_PROB = 0.02           # extra random detector misses per feature
MIN_RANGE_M = 10.0            # minimum distance camera <-> aircraft centre

# ----------------------------------------------------------------------------
# Pose discretisation: Table I of the paper (min, max, number of bins)
# ----------------------------------------------------------------------------
X_MIN, X_MAX, N_X = -100.0, 50.0, 10      # forward  [m]
Y_MIN, Y_MAX, N_Y = -50.0, 50.0, 10       # lateral  [m]
Z_MIN, Z_MAX, N_Z = -20.0, 20.0, 8        # vertical [m]
P_MIN, P_MAX, N_PHI = -45.0, 45.0, 6      # roll     [deg]
NUM_CLASSES = N_X * N_Y * N_Z * N_PHI     # = 4800
assert NUM_CLASSES == 4800

MINS = np.array([X_MIN, Y_MIN, Z_MIN, P_MIN])
MAXS = np.array([X_MAX, Y_MAX, Z_MAX, P_MAX])
NBINS = np.array([N_X, N_Y, N_Z, N_PHI])
WIDTHS = (MAXS - MINS) / NBINS            # 15 m, 10 m, 5 m, 15 deg


def encode_class(ix, iy, iz, ir):
    """4 bin indices -> single class id (0..4799). Works on arrays."""
    return ((np.asarray(ix) * N_Y + np.asarray(iy)) * N_Z + np.asarray(iz)) * N_PHI + np.asarray(ir)


def decode_class(c):
    """class id -> (ix, iy, iz, ir). Exact inverse of encode_class."""
    c = np.asarray(c)
    ir = c % N_PHI
    iz = (c // N_PHI) % N_Z
    iy = (c // (N_PHI * N_Z)) % N_Y
    ix = c // (N_PHI * N_Z * N_Y)
    return ix, iy, iz, ir


def pose_to_class(poses):
    """Continuous poses (N,4) = [x, y, z, phi_deg] -> class ids (N,)."""
    idx = np.floor((poses - MINS) / WIDTHS).astype(int)
    idx = np.clip(idx, 0, NBINS - 1)
    return encode_class(idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3])


# ----------------------------------------------------------------------------
# Simplified aircraft: 14 fixed structural feature points (leader body frame, metres)
# x forward, y left, z up. Roughly F-16 sized (15 m long, 10 m span). Our own
# choice - the paper does not list its feature locations. Feature order is fixed.
# ----------------------------------------------------------------------------
FEATURE_NAMES = [
    "nose_tip", "tail_exhaust", "left_wingtip", "right_wingtip", "fin_tip",
    "left_stab_tip", "right_stab_tip", "canopy_top", "left_wing_mid",
    "right_wing_mid", "left_wing_store", "right_wing_store",
    "intake_lip", "ventral_fin",
]
FEATURES = np.array([
    [ 7.5,  0.0,  0.0],   # 1  nose tip
    [-7.5,  0.0,  0.2],   # 2  tail / exhaust
    [-1.5,  5.0,  0.0],   # 3  left wingtip
    [-1.5, -5.0,  0.0],   # 4  right wingtip
    [-5.5,  0.0,  3.5],   # 5  vertical fin tip
    [-6.5,  2.8,  0.0],   # 6  left horizontal stabiliser tip
    [-6.5, -2.8,  0.0],   # 7  right horizontal stabiliser tip
    [ 2.5,  0.0,  1.4],   # 8  canopy top
    [ 0.5,  2.2,  0.0],   # 9  left wing mid leading edge
    [ 0.5, -2.2,  0.0],   # 10 right wing mid leading edge
    [ 0.5,  3.5, -0.5],   # 11 left wing store
    [ 0.5, -3.5, -0.5],   # 12 right wing store
    [ 3.0,  0.0, -0.9],   # 13 intake lip
    [-4.5,  0.0, -1.2],   # 14 ventral fin
])
NUM_FEATURES = len(FEATURES)
assert NUM_FEATURES == 14

# Fuselage ellipsoid used only as an occluder (centre, semi-axes). All 14 features
# lie outside it, so a feature is hidden iff the camera ray hits it first.
ELL_C = np.array([-0.5, 0.0, 0.0])
ELL_A = np.array([6.0, 0.6, 0.7])

# Skeleton lines (feature indices, 0-based) used only for plotting
SKELETON = [(0, 1), (2, 8), (8, 9), (9, 3), (5, 1), (1, 6), (1, 4), (0, 7), (7, 4), (12, 13)]

# ----------------------------------------------------------------------------
# Pinhole camera (follower camera). Square pixels, principal point at image centre.
# ----------------------------------------------------------------------------
IMG_W, IMG_H = 1280, 960
FOCAL_PX = 1500.0                          # ~46 deg horizontal field of view
CX, CY = IMG_W / 2.0, IMG_H / 2.0
MIN_DEPTH_M = 1.0                          # features closer than this to the image plane are lost


def project_features(poses, rng=None, add_noise=True):
    """
    Core physics. poses (N,4) = [x, y, z, phi_deg]  ->  u, v, visible (each (N,14)).

    1. Camera centre c = (x, y, z) in the leader frame. Leader attitude = identity.
    2. Gimbal: boresight f points from the camera to the leader origin (-c/|c|).
    3. Camera "up" = leader's +z projected perpendicular to f (level horizon),
       then the camera is rolled by phi about f.
    4. Pinhole: u = cx + F*Xc/Zc ,  v = cy + F*Yc/Zc  (Xc right, Yc down, Zc forward).
    5. Visibility: in front of camera, inside image, not hidden by the fuselage.
    """
    poses = np.asarray(poses, dtype=np.float64)
    c = poses[:, :3]
    phi = np.deg2rad(poses[:, 3])

    f = -c / np.linalg.norm(c, axis=1, keepdims=True)                # (n,3) boresight
    up0 = np.array([0.0, 0.0, 1.0])[None, :] - f[:, 2:3] * f         # project world-up
    norm = np.linalg.norm(up0, axis=1, keepdims=True)
    bad = (norm[:, 0] < 1e-6)                                        # looking straight up/down
    if np.any(bad):
        up0[bad] = np.array([1.0, 0.0, 0.0])[None, :] - f[bad, 0:1] * f[bad]
        norm = np.linalg.norm(up0, axis=1, keepdims=True)
    up0 = up0 / norm
    right0 = np.cross(f, up0)
    cp, sp = np.cos(phi)[:, None], np.sin(phi)[:, None]
    up = cp * up0 + sp * right0                                      # roll about boresight
    right = cp * right0 - sp * up0

    rel = FEATURES[None, :, :] - c[:, None, :]                       # (n,14,3) camera -> feature
    xc = np.einsum("nkj,nj->nk", rel, right)
    yc = -np.einsum("nkj,nj->nk", rel, up)                           # image y points down
    zc = np.einsum("nkj,nj->nk", rel, f)

    zs = np.where(zc > 1e-3, zc, 1e-3)
    u = CX + FOCAL_PX * xc / zs
    v = CY + FOCAL_PX * yc / zs

    # --- self-occlusion by the fuselage ellipsoid (ray / ellipsoid intersection) ---
    o = ((c - ELL_C) / ELL_A)[:, None, :]                            # ray origin, scaled space
    d = rel / ELL_A                                                  # ray direction, scaled space
    a = np.sum(d * d, axis=2)
    b = 2.0 * np.sum(o * d, axis=2)
    cc = np.sum(o * o, axis=2) - 1.0
    disc = b * b - 4.0 * a * cc
    t1 = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a)
    hidden = (disc > 0) & (t1 > 0) & (t1 < 1.0 - 1e-6)

    if add_noise:
        u = u + rng.normal(0.0, PIXEL_NOISE_STD, u.shape)
        v = v + rng.normal(0.0, PIXEL_NOISE_STD, v.shape)
        dropout = rng.random(u.shape) < DROPOUT_PROB
    else:
        dropout = np.zeros(u.shape, dtype=bool)

    in_image = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    visible = (zc > MIN_DEPTH_M) & in_image & (~hidden) & (~dropout)
    return u, v, visible


def sample_poses(class_ids, rng):
    """Uniform continuous pose inside each class cell (rejecting range < MIN_RANGE_M)."""
    ix, iy, iz, ir = decode_class(class_ids)
    lows = MINS + np.stack([ix, iy, iz, ir], axis=1) * WIDTHS
    poses = np.empty((len(class_ids), 4))
    todo = np.arange(len(class_ids))
    margin = 1e-3                      # keep samples strictly inside the cell (rounding-safe)
    for _ in range(500):
        poses[todo] = lows[todo] + margin + rng.random((len(todo), 4)) * (WIDTHS - 2 * margin)
        rng_m = np.linalg.norm(poses[todo, :3], axis=1)
        todo = todo[rng_m < MIN_RANGE_M]
        if len(todo) == 0:
            return poses
    raise RuntimeError("Could not sample valid poses for some class (MIN_RANGE_M too large?)")


def feature_columns():
    names = []
    for k in range(NUM_FEATURES):
        names += [f"feature_{k+1}_x", f"feature_{k+1}_y", f"feature_{k+1}_occlusion"]
    return names


def make_dataset(samples_per_class, rng):
    class_ids = np.repeat(np.arange(NUM_CLASSES), samples_per_class)
    poses = sample_poses(class_ids, rng)
    u, v, visible = project_features(poses, rng, add_noise=True)

    # undetected features are stored as (0, 0) with occlusion flag = 1
    u = np.where(visible, u, 0.0)
    v = np.where(visible, v, 0.0)
    occ = (~visible).astype(np.int8)

    cols = {}
    for k in range(NUM_FEATURES):                    # column order: f1_x, f1_y, f1_occ, f2_x, ...
        cols[f"feature_{k+1}_x"] = np.round(u[:, k], 3)
        cols[f"feature_{k+1}_y"] = np.round(v[:, k], 3)
        cols[f"feature_{k+1}_occlusion"] = occ[:, k]
    cols["pose_x"] = np.round(poses[:, 0], 4)
    cols["pose_y"] = np.round(poses[:, 1], 4)
    cols["pose_z"] = np.round(poses[:, 2], 4)
    cols["pose_phi_deg"] = np.round(poses[:, 3], 4)
    cols["pose_class"] = class_ids.astype(np.int32)
    df = pd.DataFrame(cols)
    return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)   # shuffle rows


# ----------------------------------------------------------------------------
# Validation and diagnostics
# ----------------------------------------------------------------------------
def validate_dataset(df):
    print("\n" + "=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)
    feats = feature_columns()
    ok = True

    print(f"1. Number of samples          : {len(df)}")
    n_unique = df["pose_class"].nunique()
    print(f"2. Unique pose classes        : {n_unique}  (expected {NUM_CLASSES})")
    ok &= (n_unique == NUM_CLASSES)
    counts = df["pose_class"].value_counts()
    print(f"3. Samples per class          : min {counts.min()}, max {counts.max()}, mean {counts.mean():.1f}")
    print(f"4. Number of input features   : {len(feats)}  (expected 42)")
    ok &= (len(feats) == 42)
    n_missing = int(df.isna().sum().sum())
    print(f"5. Missing (NaN) values       : {n_missing}")
    ok &= (n_missing == 0)

    vis = df[[f"feature_{k+1}_occlusion" for k in range(NUM_FEATURES)]].values == 0
    xs = df[[f"feature_{k+1}_x" for k in range(NUM_FEATURES)]].values
    ys = df[[f"feature_{k+1}_y" for k in range(NUM_FEATURES)]].values
    in_rng = (xs[vis].min() >= 0) and (xs[vis].max() < IMG_W) and (ys[vis].min() >= 0) and (ys[vis].max() < IMG_H)
    print(f"6. Detected-pixel ranges      : u in [{xs[vis].min():.1f}, {xs[vis].max():.1f}] (0..{IMG_W}), "
          f"v in [{ys[vis].min():.1f}, {ys[vis].max():.1f}] (0..{IMG_H})  -> {'OK' if in_rng else 'BAD'}")
    ok &= bool(in_rng)
    print(f"7. Occlusion percentage       : {100.0 * (~vis).mean():.2f}% of all feature slots")
    per_feat = 100.0 * (~vis).mean(axis=0)
    print("   per feature (%)            : " + ", ".join(f"{p:.1f}" for p in per_feat))
    many_hidden = int(((~vis).sum(axis=1) >= 12).sum())
    print(f"   samples with >=12 of 14 features undetected: {many_hidden} ({100.0 * many_hidden / len(df):.2f}%)")
    lab_ok = (df["pose_class"].min() == 0) and (df["pose_class"].max() == NUM_CLASSES - 1)
    print(f"8. Label range                : {df['pose_class'].min()}..{df['pose_class'].max()}  (expected 0..{NUM_CLASSES-1}) "
          f"-> {'OK' if lab_ok else 'BAD'}")
    ok &= bool(lab_ok)

    # label <-> continuous pose consistency + encode/decode round trip
    recomputed = pose_to_class(df[["pose_x", "pose_y", "pose_z", "pose_phi_deg"]].values)
    cons = bool(np.all(recomputed == df["pose_class"].values))
    ix, iy, iz, ir = decode_class(df["pose_class"].values)
    rt = bool(np.all(encode_class(ix, iy, iz, ir) == df["pose_class"].values))
    print(f"9. Label == bin(pose) check   : {'OK' if cons else 'BAD'};  encode(decode(c)) == c : {'OK' if rt else 'BAD'}")
    ok &= cons and rt
    return ok


def sensitivity_check(rng, n=3000):
    """
    Proof that the inputs depend on the pose: shift ONE pose coordinate by one class
    bin width (noise-free) and measure how far the visible features move in pixels.
    Compare with the measurement noise (PIXEL_NOISE_STD).
    """
    print("\n" + "=" * 60)
    print("POSE SENSITIVITY CHECK (noise-free, one-bin shift of one pose coordinate)")
    print("=" * 60)
    cls = rng.integers(0, NUM_CLASSES, n)
    base = sample_poses(cls, rng)
    u0, v0, vis0 = project_features(base, add_noise=False)
    for j, name in enumerate(["x (15 m)", "y (10 m)", "z (5 m)", "phi (15 deg)"]):
        shifted = base.copy()
        shifted[:, j] += WIDTHS[j]
        okp = np.linalg.norm(shifted[:, :3], axis=1) >= MIN_RANGE_M
        u1, v1, vis1 = project_features(shifted, add_noise=False)
        both = vis0 & vis1 & okp[:, None]
        shift = np.sqrt((u1 - u0) ** 2 + (v1 - v0) ** 2)[both]
        print(f"  +1 bin in {name:13s}: median feature shift = {np.median(shift):7.2f} px, "
              f"mean = {shift.mean():7.2f} px   (noise std = {PIXEL_NOISE_STD} px)")


def plot_configurations(rng):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping visualisation (pip install matplotlib).")
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base = np.array([-50.0, 15.0, 6.0, 0.0])
    sweeps = {
        "forward x [m]": (0, [-95, -75, -55, -35, -20]),
        "lateral y [m]": (1, [-45, -25, 0.5, 25, 45]),
        "vertical z [m]": (2, [-18, -9, 0.5, 9, 18]),
        "roll phi [deg]": (3, [-40, -20, 0, 20, 40]),
    }
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, 5))
    for ax, (title, (j, vals)) in zip(axes, sweeps.items()):
        for col, val in zip(colors, vals):
            pose = base.copy()
            pose[j] = val
            u, v, vis = project_features(pose[None, :], rng, add_noise=True)
            u, v, vis = u[0], v[0], vis[0]
            for a, b in SKELETON:
                if vis[a] and vis[b]:
                    ax.plot([u[a], u[b]], [v[a], v[b]], "-", color=col, lw=1, alpha=0.6)
            ax.scatter(u[vis], v[vis], s=18, color=col, label=f"{val:g}")
        ax.set_xlim(CX - 250, CX + 250)
        ax.set_ylim(CY + 250, CY - 250)           # image y axis points down
        ax.set_aspect("equal")
        ax.set_title(f"Only {title} changes")
        ax.set_xlabel("u [px]")
        ax.set_ylabel("v [px]")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Projected 14 features for different poses (others fixed at x={base[0]:g}, "
                 f"y={base[1]:g}, z={base[2]:g}, phi={base[3]:g})")
    fig.tight_layout()
    out = RESULTS_DIR / "feature_configurations.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nSaved visualisation: {out}")


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating {SAMPLES_PER_CLASS} samples x {NUM_CLASSES} classes = "
          f"{SAMPLES_PER_CLASS * NUM_CLASSES} samples ...")
    df = make_dataset(SAMPLES_PER_CLASS, rng)
    ok = validate_dataset(df)
    sensitivity_check(rng)
    plot_configurations(rng)

    df.to_csv(CSV_PATH, index=False)
    info = {
        "csv": CSV_PATH.name, "num_classes": NUM_CLASSES, "feature_columns": feature_columns(),
        "bins": {"x": [X_MIN, X_MAX, N_X], "y": [Y_MIN, Y_MAX, N_Y],
                 "z": [Z_MIN, Z_MAX, N_Z], "phi_deg": [P_MIN, P_MAX, N_PHI]},
        "class_formula": "((ix*N_Y+iy)*N_Z+iz)*N_PHI+ir",
        "image": {"width": IMG_W, "height": IMG_H, "focal_px": FOCAL_PX, "cx": CX, "cy": CY},
        "pixel_noise_std": PIXEL_NOISE_STD, "dropout_prob": DROPOUT_PROB,
        "samples_per_class": SAMPLES_PER_CLASS, "seed": SEED,
        "feature_names": FEATURE_NAMES, "features_xyz_m": FEATURES.tolist(),
    }
    INFO_PATH.write_text(json.dumps(info, indent=2))
    print(f"\nSaved dataset : {CSV_PATH}  ({CSV_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"Saved metadata: {INFO_PATH}")
    print(f"Validation    : {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED - read above'}")
    print(f"Done in {time.time() - t0:.1f} s")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()