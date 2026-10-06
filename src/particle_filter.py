"""
Phase 4: Particle-filter refinement for the pose-estimation mini-project.

This file is designed to live at:

    project_root/src/particle_filter.py

It reuses the exact physics/model definitions from generate_data.py and
the trained ANN/inference helpers from train_ann.py.

Pipeline:
    observed 42-D feature vector
        -> ANN top-k coarse pose hypotheses
        -> particle initialization around those hypotheses
        -> repeated likelihood weighting + systematic resampling + jitter
        -> refined continuous pose estimate

Important:
- The paper does not specify a particle-filter implementation. The PF
  likelihood, particle count, jitter schedule, and visibility penalty below
  are therefore implementation choices for this mini-project.
- The PF uses the same simplified aircraft/camera model as generate_data.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Import the exact project definitions.
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from generate_data import (  # noqa: E402
    MINS,
    MAXS,
    WIDTHS,
    NUM_CLASSES,
    NUM_FEATURES,
    project_features,
    sample_poses,
    pose_to_class,
)

from train_ann import (  # noqa: E402
    MODEL_PATH,
    NORM_PATH,
    PoseANN,
    preprocess_raw,
    predict_topk,
    class_to_center,
    decode_class,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_PARTICLES = 2000
DEFAULT_ITERATIONS = 8
DEFAULT_TOPK = 5

# Observation model. The generated observations use 0.5 px Gaussian noise,
# but 2% dropout and model mismatch make an exactly-0.5 px likelihood too
# brittle. We therefore use a small robust sigma for the PF likelihood.
PIXEL_SIGMA_PX = 2.0

# Penalty applied when the particle predicts a feature as visible/occluded
# differently from the observation.
VISIBILITY_LOG_PENALTY = 3.0

# Initial particle spread around each ANN top-k hypothesis.
INIT_SPREAD = np.array([0.50, 0.50, 0.50, 0.50], dtype=np.float64)

# Jitter standard deviation in units of the pose-bin width. It decreases
# over PF iterations to move from exploration to refinement.
JITTER_FRACTION_START = 0.30
JITTER_FRACTION_END = 0.04


def clamp_pose(pose: np.ndarray) -> np.ndarray:
    """Clamp pose(s) to the same ranges used by generate_data.py."""
    return np.clip(pose, MINS, MAXS)


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Systematic resampling.

    Returns indices with replacement, preserving high-weight particles while
    avoiding the extra variance of completely independent multinomial draws.
    """
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)


def observation_likelihood(
    particles: np.ndarray,
    obs_u: np.ndarray,
    obs_v: np.ndarray,
    obs_visible: np.ndarray,
) -> np.ndarray:
    """
    Compute log likelihood for every particle.

    The observation consists of 14 feature pixel coordinates and visibility.
    For observed visible features, use a Gaussian pixel-error model.
    For observed occluded features, no pixel-position error is used; instead
    a visibility mismatch receives a penalty.

    A robust clipped residual prevents one badly mismatched feature from
    completely destroying a particle's weight.
    """
    n = len(particles)

    # The projection function is deterministic when add_noise=False.
    pred_u, pred_v, pred_visible = project_features(
        particles,
        rng=None,
        add_noise=False,
    )

    visible_obs = obs_visible[None, :]
    pred_visible = pred_visible.astype(bool)

    # Visible observed features: squared pixel residual.
    du = pred_u - obs_u[None, :]
    dv = pred_v - obs_v[None, :]
    sq_error = du * du + dv * dv

    # Robustify very large residuals. A feature that is projected far away
    # should have low probability, but should not create numerical overflow.
    max_sq = (8.0 * PIXEL_SIGMA_PX) ** 2
    sq_error = np.minimum(sq_error, max_sq)

    visible_loglik = -0.5 * sq_error / (PIXEL_SIGMA_PX ** 2)

    # Visibility agreement. Visible observations require a visible prediction.
    # Occluded observations are allowed to be predicted occluded; predicting
    # them visible incurs a penalty.
    visibility_mismatch = pred_visible != visible_obs
    visibility_loglik = -VISIBILITY_LOG_PENALTY * visibility_mismatch

    # Only compare pixel coordinates when the real observation is visible.
    total = np.where(visible_obs, visible_loglik, 0.0)
    total += visibility_loglik

    return total.sum(axis=1)


def normalize_log_weights(logw: np.ndarray) -> np.ndarray:
    """Stable softmax for particle log weights."""
    m = np.max(logw)
    w = np.exp(logw - m)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full(len(logw), 1.0 / len(logw))
    return w / s


def effective_sample_size(weights: np.ndarray) -> float:
    return 1.0 / np.sum(weights * weights)


def initialise_particles(
    ann_classes: np.ndarray,
    ann_probs: np.ndarray,
    n_particles: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw particles around the ANN's top-k class centres.

    The number of particles allocated to each hypothesis is proportional to
    the ANN probability, so the PF can recover if the ANN top-1 is one of
    several nearby hypotheses.
    """
    k = len(ann_classes)
    probs = np.asarray(ann_probs, dtype=np.float64)
    probs = probs / probs.sum()

    counts = rng.multinomial(n_particles, probs)
    particles = []

    for class_id, count in zip(ann_classes, counts):
        if count == 0:
            continue

        centre = class_to_center(np.array([class_id]))[0].astype(np.float64)

        # Spread in physical units. The x/y/z/roll spreads are intentionally
        # tied to the class-bin widths so initialization respects the paper's
        # discretization.
        scale = WIDTHS * INIT_SPREAD
        p = centre[None, :] + rng.normal(
            0.0, scale, size=(count, 4)
        )

        particles.append(p)

    if not particles:
        # Defensive fallback.
        centre = class_to_center(np.array([ann_classes[0]]))[0]
        particles = [
            centre[None, :]
            + rng.normal(0.0, WIDTHS * INIT_SPREAD, size=(n_particles, 4))
        ]

    particles = np.concatenate(particles, axis=0)

    # Keep only physically allowed pose bounds.
    particles = clamp_pose(particles)

    # The data generator rejects aircraft separations below MIN_RANGE_M.
    # If a particle lands there, project it back to the nearest valid radial
    # distance without changing its direction.
    ranges = np.linalg.norm(particles[:, :3], axis=1)
    bad = ranges < 10.0
    if np.any(bad):
        safe = particles[bad, :3]
        safe_norm = np.linalg.norm(safe, axis=1, keepdims=True)
        safe_norm = np.maximum(safe_norm, 1e-9)
        particles[bad, :3] = safe / safe_norm * 10.01

        particles = clamp_pose(particles)

    return particles


def particle_filter(
    observed_raw: np.ndarray,
    ann_classes: np.ndarray,
    ann_probs: np.ndarray,
    n_particles: int = DEFAULT_PARTICLES,
    iterations: int = DEFAULT_ITERATIONS,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Run the static-state particle filter.

    Returns:
        refined_pose: weighted mean pose [x,y,z,roll]
        diagnostics: PF metrics/history
    """
    if rng is None:
        rng = np.random.default_rng(42)

    raw = np.asarray(observed_raw, dtype=np.float32).reshape(NUM_FEATURES, 3)
    obs_u = raw[:, 0].astype(np.float64)
    obs_v = raw[:, 1].astype(np.float64)
    obs_visible = raw[:, 2] < 0.5

    particles = initialise_particles(
        ann_classes,
        ann_probs,
        n_particles,
        rng,
    )

    history = []

    for it in range(iterations):
        logw = observation_likelihood(
            particles,
            obs_u,
            obs_v,
            obs_visible,
        )
        weights = normalize_log_weights(logw)

        estimate = np.sum(particles * weights[:, None], axis=0)
        map_pose = particles[np.argmax(weights)]

        ess = effective_sample_size(weights)

        history.append(
            {
                "iteration": it + 1,
                "effective_sample_size": float(ess),
                "weighted_pose": estimate.tolist(),
                "map_pose": map_pose.tolist(),
                "best_log_weight": float(np.max(logw)),
            }
        )

        # Resample.
        indices = systematic_resample(weights, rng)
        particles = particles[indices].copy()

        # Shrinking jitter: large early exploration, small late refinement.
        if iterations == 1:
            frac = JITTER_FRACTION_END
        else:
            alpha = it / (iterations - 1)
            frac = (
                JITTER_FRACTION_START * (1.0 - alpha)
                + JITTER_FRACTION_END * alpha
            )

        jitter = rng.normal(
            0.0,
            WIDTHS * frac,
            size=particles.shape,
        )
        particles += jitter
        particles = clamp_pose(particles)

    # Final weighting after the final jitter.
    logw = observation_likelihood(
        particles,
        obs_u,
        obs_v,
        obs_visible,
    )
    weights = normalize_log_weights(logw)

    refined_pose = np.sum(particles * weights[:, None], axis=0)

    diagnostics = {
        "history": history,
        "final_effective_sample_size": float(effective_sample_size(weights)),
        "final_map_pose": particles[np.argmax(weights)].tolist(),
        "final_weighted_pose": refined_pose.tolist(),
    }

    return refined_pose, diagnostics


def pose_errors(pred: np.ndarray, true: np.ndarray) -> dict:
    """Calculate the project's physical pose errors."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)

    position_error = float(np.linalg.norm(pred[:3] - true[:3]))
    roll_error = float(abs(pred[3] - true[3]))

    return {
        "position_error_m": position_error,
        "roll_error_deg": roll_error,
        "absolute_xyz_error_m": np.abs(pred[:3] - true[:3]).tolist(),
    }


def ann_class_to_continuous_pose(class_id: int) -> np.ndarray:
    """Convert ANN's class prediction into the centre of that pose cell."""
    return class_to_center(np.array([class_id]))[0].astype(np.float64)


def run_single_example(
    raw: np.ndarray,
    true_pose: np.ndarray,
    model,
    norm: dict,
    rng: np.random.Generator,
    n_particles: int,
    iterations: int,
    topk: int,
) -> dict:
    """Run ANN + PF on one existing dataset sample."""
    top_classes, top_probs = predict_topk(
        model,
        norm,
        raw[None, :],
        k=topk,
    )

    ann_class = int(top_classes[0, 0])
    ann_prob = float(top_probs[0, 0])
    ann_pose = ann_class_to_continuous_pose(ann_class)

    refined_pose, diagnostics = particle_filter(
        raw,
        top_classes[0],
        top_probs[0],
        n_particles=n_particles,
        iterations=iterations,
        rng=rng,
    )

    return {
        "true_pose": true_pose.tolist(),
        "ann_class": ann_class,
        "ann_probability": ann_prob,
        "ann_pose": ann_pose.tolist(),
        "pf_pose": refined_pose.tolist(),
        "ann_error": pose_errors(ann_pose, true_pose),
        "pf_error": pose_errors(refined_pose, true_pose),
        "pf_diagnostics": diagnostics,
    }


def evaluate_dataset(
    csv_path: Path,
    model,
    norm: dict,
    n_samples: int,
    n_particles: int,
    iterations: int,
    topk: int,
    seed: int,
) -> pd.DataFrame:
    """
    Evaluate ANN vs PF on randomly selected samples from the generated CSV.

    For speed, this defaults to 100 examples. Increase n_samples for the
    final experiment once the pipeline is confirmed to work.
    """
    df = pd.read_csv(csv_path)

    feature_cols = norm.get("input_order")
    # The actual column names are recovered directly from the CSV.
    cols = []
    for k in range(1, NUM_FEATURES + 1):
        cols.extend(
            [
                f"feature_{k}_x",
                f"feature_{k}_y",
                f"feature_{k}_occlusion",
            ]
        )

    if n_samples >= len(df):
        selected = df
    else:
        selected = df.sample(
            n=n_samples,
            random_state=seed,
        )

    rng = np.random.default_rng(seed)
    rows = []

    for count, (_, row) in enumerate(selected.iterrows(), start=1):
        raw = row[cols].values.astype(np.float32)
        true_pose = row[
            ["pose_x", "pose_y", "pose_z", "pose_phi_deg"]
        ].values.astype(np.float64)

        result = run_single_example(
            raw,
            true_pose,
            model,
            norm,
            rng,
            n_particles,
            iterations,
            topk,
        )

        ann_err = result["ann_error"]
        pf_err = result["pf_error"]

        rows.append(
            {
                "sample": count,
                "true_x": true_pose[0],
                "true_y": true_pose[1],
                "true_z": true_pose[2],
                "true_roll": true_pose[3],
                "ann_x": result["ann_pose"][0],
                "ann_y": result["ann_pose"][1],
                "ann_z": result["ann_pose"][2],
                "ann_roll": result["ann_pose"][3],
                "pf_x": result["pf_pose"][0],
                "pf_y": result["pf_pose"][1],
                "pf_z": result["pf_pose"][2],
                "pf_roll": result["pf_pose"][3],
                "ann_position_error_m": ann_err["position_error_m"],
                "pf_position_error_m": pf_err["position_error_m"],
                "ann_roll_error_deg": ann_err["roll_error_deg"],
                "pf_roll_error_deg": pf_err["roll_error_deg"],
                "ann_class": result["ann_class"],
                "ann_probability": result["ann_probability"],
            }
        )

        if count % 10 == 0 or count == len(selected):
            print(f"Evaluated {count}/{len(selected)} samples")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4: Particle-filter refinement of ANN pose estimates"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of samples for ANN-vs-PF evaluation (default: 20)",
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=DEFAULT_PARTICLES,
        help="Particles per sample (default: 2000)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="PF iterations (default: 8)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=DEFAULT_TOPK,
        help="ANN hypotheses used to initialize particles (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed",
    )
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "pose_dataset.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Run generate_data.py first."
        )

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Trained ANN not found: {MODEL_PATH}\n"
            "Run train_ann.py first."
        )

    if not Path(NORM_PATH).exists():
        raise FileNotFoundError(
            f"Normalization file not found: {NORM_PATH}\n"
            "Run train_ann.py first."
        )

    print("=" * 70)
    print("PHASE 4: ANN + PARTICLE FILTER POSE REFINEMENT")
    print("=" * 70)
    print(f"Samples       : {args.samples}")
    print(f"Particles     : {args.particles}")
    print(f"Iterations    : {args.iterations}")
    print(f"ANN top-k     : {args.topk}")
    print(f"Pixel sigma   : {PIXEL_SIGMA_PX} px")
    print(f"Seed          : {args.seed}")
    print()

    # Load trained ANN using the same checkpoint format as train_ann.py.
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model = PoseANN(hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    norm = json.loads(Path(NORM_PATH).read_text())

    results = evaluate_dataset(
        csv_path=csv_path,
        model=model,
        norm=norm,
        n_samples=args.samples,
        n_particles=args.particles,
        iterations=args.iterations,
        topk=args.topk,
        seed=args.seed,
    )

    out_csv = results_dir / "ann_vs_particle_filter.csv"
    results.to_csv(out_csv, index=False)

    # Aggregate comparison.
    ann_pos = results["ann_position_error_m"].to_numpy()
    pf_pos = results["pf_position_error_m"].to_numpy()
    ann_roll = results["ann_roll_error_deg"].to_numpy()
    pf_roll = results["pf_roll_error_deg"].to_numpy()

    def improvement(old, new):
        return 100.0 * (old.mean() - new.mean()) / max(old.mean(), 1e-12)

    print("\n" + "=" * 70)
    print("ANN vs PARTICLE FILTER")
    print("=" * 70)
    print(f"Position MAE / mean distance:")
    print(f"  ANN : {ann_pos.mean():.3f} m")
    print(f"  PF  : {pf_pos.mean():.3f} m")
    print(f"  Improvement: {improvement(ann_pos, pf_pos):.2f}%")

    print(f"\nPosition median error:")
    print(f"  ANN : {np.median(ann_pos):.3f} m")
    print(f"  PF  : {np.median(pf_pos):.3f} m")

    print(f"\nPosition 90th percentile:")
    print(f"  ANN : {np.percentile(ann_pos, 90):.3f} m")
    print(f"  PF  : {np.percentile(pf_pos, 90):.3f} m")

    print(f"\nRoll MAE / mean absolute error:")
    print(f"  ANN : {ann_roll.mean():.3f} deg")
    print(f"  PF  : {pf_roll.mean():.3f} deg")
    print(f"  Improvement: {improvement(ann_roll, pf_roll):.2f}%")

    # Save summary JSON.
    summary = {
        "phase": 4,
        "n_samples": int(len(results)),
        "particles": args.particles,
        "iterations": args.iterations,
        "ann_topk": args.topk,
        "pixel_sigma_px": PIXEL_SIGMA_PX,
        "visibility_log_penalty": VISIBILITY_LOG_PENALTY,
        "ann": {
            "position_mean_error_m": float(ann_pos.mean()),
            "position_median_error_m": float(np.median(ann_pos)),
            "position_p90_error_m": float(np.percentile(ann_pos, 90)),
            "roll_mean_error_deg": float(ann_roll.mean()),
            "roll_median_error_deg": float(np.median(ann_roll)),
        },
        "particle_filter": {
            "position_mean_error_m": float(pf_pos.mean()),
            "position_median_error_m": float(np.median(pf_pos)),
            "position_p90_error_m": float(np.percentile(pf_pos, 90)),
            "roll_mean_error_deg": float(pf_roll.mean()),
            "roll_median_error_deg": float(np.median(pf_roll)),
        },
        "improvement_percent": {
            "position_mean": float(improvement(ann_pos, pf_pos)),
            "roll_mean": float(improvement(ann_roll, pf_roll)),
        },
    }

    summary_path = results_dir / "particle_filter_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nSaved detailed results : {out_csv}")
    print(f"Saved summary          : {summary_path}")
    print("\nPhase 4 complete.")


if __name__ == "__main__":
    main()
