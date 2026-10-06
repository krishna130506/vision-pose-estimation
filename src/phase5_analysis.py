"""
Phase 5: ANN vs Particle Filter analysis.

Reads:
    results/ann_vs_particle_filter.csv

Produced by:
    src/particle_filter.py

Creates:
    results/phase5_error_distributions.png
    results/phase5_position_error_comparison.png
    results/phase5_roll_error_comparison.png
    results/phase5_samplewise_comparison.png
    results/phase5_pose_examples.png
    results/phase5_summary.json

This phase does NOT retrain the model or rerun the particle filter.
It analyzes the paired 100-sample experiment already completed in Phase 4.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

# Matplotlib is imported only when needed so the script can still produce
# the numerical summary if plotting is unavailable.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
INPUT_CSV = RESULTS_DIR / "ann_vs_particle_filter.csv"


def improvement_percent(ann, pf):
    ann = np.asarray(ann, dtype=float)
    pf = np.asarray(pf, dtype=float)
    return 100.0 * (ann - pf) / np.maximum(np.abs(ann), 1e-12)


def describe_pair(ann, pf):
    ann = np.asarray(ann, dtype=float)
    pf = np.asarray(pf, dtype=float)

    return {
        "ann_mean": float(np.mean(ann)),
        "pf_mean": float(np.mean(pf)),
        "ann_median": float(np.median(ann)),
        "pf_median": float(np.median(pf)),
        "ann_p90": float(np.percentile(ann, 90)),
        "pf_p90": float(np.percentile(pf, 90)),
        "mean_reduction_percent": float(
            100.0 * (np.mean(ann) - np.mean(pf)) / np.mean(ann)
        ),
        "median_reduction_percent": float(
            100.0 * (np.median(ann) - np.median(pf)) / np.median(ann)
        ),
        "p90_reduction_percent": float(
            100.0 * (np.percentile(ann, 90) - np.percentile(pf, 90))
            / np.percentile(ann, 90)
        ),
        "samples_pf_better": int(np.sum(pf < ann)),
        "samples_ann_better": int(np.sum(ann < pf)),
        "samples_equal": int(np.sum(np.isclose(ann, pf))),
    }


def make_error_distributions(df):
    if plt is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(
        df["ann_position_error_m"],
        bins=15,
        alpha=0.65,
        label="ANN",
    )
    axes[0].hist(
        df["pf_position_error_m"],
        bins=15,
        alpha=0.65,
        label="Particle Filter",
    )
    axes[0].set_title("Position Error Distribution")
    axes[0].set_xlabel("Euclidean position error [m]")
    axes[0].set_ylabel("Number of samples")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].hist(
        df["ann_roll_error_deg"],
        bins=15,
        alpha=0.65,
        label="ANN",
    )
    axes[1].hist(
        df["pf_roll_error_deg"],
        bins=15,
        alpha=0.65,
        label="Particle Filter",
    )
    axes[1].set_title("Roll Error Distribution")
    axes[1].set_xlabel("Absolute roll error [deg]")
    axes[1].set_ylabel("Number of samples")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "phase5_error_distributions.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_position_comparison(df):
    if plt is None:
        return

    ann = df["ann_position_error_m"].to_numpy()
    pf = df["pf_position_error_m"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 5))

    positions = np.array([1, 2])
    ax.boxplot(
        [ann, pf],
        positions=positions,
        tick_labels=["ANN", "ANN + PF"],
        showmeans=True,
    )

    ax.set_title("Position Error: ANN vs Particle Filter")
    ax.set_ylabel("Euclidean position error [m]")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "phase5_position_error_comparison.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_roll_comparison(df):
    if plt is None:
        return

    ann = df["ann_roll_error_deg"].to_numpy()
    pf = df["pf_roll_error_deg"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.boxplot(
        [ann, pf],
        positions=[1, 2],
        tick_labels=["ANN", "ANN + PF"],
        showmeans=True,
    )

    ax.set_title("Roll Error: ANN vs Particle Filter")
    ax.set_ylabel("Absolute roll error [deg]")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "phase5_roll_error_comparison.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_samplewise_comparison(df):
    if plt is None:
        return

    x = np.arange(1, len(df) + 1)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(
        x,
        df["ann_position_error_m"],
        marker="o",
        markersize=2.5,
        linewidth=1,
        label="ANN",
    )
    ax.plot(
        x,
        df["pf_position_error_m"],
        marker="o",
        markersize=2.5,
        linewidth=1,
        label="ANN + Particle Filter",
    )

    ax.set_title("Sample-by-Sample Position Error")
    ax.set_xlabel("Validation sample")
    ax.set_ylabel("Position error [m]")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "phase5_samplewise_comparison.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_pose_examples(df, n=6):
    """
    Visualizes the first six paired examples as x/y/z/roll estimates.

    This is deliberately a line plot rather than a 3-D aircraft rendering:
    it directly shows numerical agreement with the continuous ground truth.
    """
    if plt is None:
        return

    n = min(n, len(df))
    selected = df.iloc[:n]

    dimensions = [
        ("x", "true_x", "ann_x", "pf_x", "m"),
        ("y", "true_y", "ann_y", "pf_y", "m"),
        ("z", "true_z", "ann_z", "pf_z", "m"),
        ("roll", "true_roll", "ann_roll", "pf_roll", "deg"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    sample_ids = np.arange(1, n + 1)

    for ax, (name, true_col, ann_col, pf_col, unit) in zip(axes, dimensions):
        ax.plot(
            sample_ids,
            selected[true_col],
            marker="o",
            linewidth=2,
            label="Ground truth",
        )
        ax.plot(
            sample_ids,
            selected[ann_col],
            marker="x",
            linewidth=1.5,
            label="ANN",
        )
        ax.plot(
            sample_ids,
            selected[pf_col],
            marker="s",
            linewidth=1.5,
            label="ANN + PF",
        )
        ax.set_title(f"{name.upper()} estimate")
        ax.set_xlabel("Sample")
        ax.set_ylabel(f"{name} [{unit}]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Example Continuous Pose Estimates")
    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "phase5_pose_examples.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found.")
        print("Run Phase 4 first:")
        print(
            "python src/particle_filter.py "
            "--samples 100 --particles 2000 --iterations 8"
        )
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)

    required = [
        "ann_position_error_m",
        "pf_position_error_m",
        "ann_roll_error_deg",
        "pf_roll_error_deg",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(df) < 2:
        raise ValueError("At least two validation samples are required.")

    ann_pos = df["ann_position_error_m"].to_numpy()
    pf_pos = df["pf_position_error_m"].to_numpy()
    ann_roll = df["ann_roll_error_deg"].to_numpy()
    pf_roll = df["pf_roll_error_deg"].to_numpy()

    pos = describe_pair(ann_pos, pf_pos)
    roll = describe_pair(ann_roll, pf_roll)

    # Paired sample-wise statistics.
    pos_delta = ann_pos - pf_pos
    roll_delta = ann_roll - pf_roll

    summary = {
        "phase": 5,
        "input_file": str(INPUT_CSV),
        "n_samples": int(len(df)),
        "position": pos,
        "roll": roll,
        "paired_results": {
            "position_mean_delta_m": float(np.mean(pos_delta)),
            "position_median_delta_m": float(np.median(pos_delta)),
            "roll_mean_delta_deg": float(np.mean(roll_delta)),
            "roll_median_delta_deg": float(np.median(roll_delta)),
            "position_pf_better_fraction": float(np.mean(pf_pos < ann_pos)),
            "roll_pf_better_fraction": float(np.mean(pf_roll < ann_roll)),
        },
        "interpretation": {
            "primary_comparison":
                "ANN class-centre estimate versus continuous pose obtained "
                "after particle-filter refinement.",
            "important_caveat":
                "The particle filter uses the same synthetic camera, "
                "aircraft geometry, visibility model, and projection model "
                "used to generate the observations. Therefore the PF result "
                "demonstrates model-based refinement under the project's "
                "synthetic observation model; it is not an independent "
                "real-world benchmark.",
        },
    }

    summary_path = RESULTS_DIR / "phase5_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    make_error_distributions(df)
    make_position_comparison(df)
    make_roll_comparison(df)
    make_samplewise_comparison(df)
    make_pose_examples(df)

    # Print concise report.
    print("=" * 72)
    print("PHASE 5: ANN vs PARTICLE FILTER ANALYSIS")
    print("=" * 72)
    print(f"Validation samples: {len(df)}")

    print("\nPOSITION ERROR")
    print(f"  ANN mean       : {pos['ann_mean']:.3f} m")
    print(f"  PF mean        : {pos['pf_mean']:.3f} m")
    print(f"  Mean reduction : {pos['mean_reduction_percent']:.2f}%")
    print(f"  ANN median     : {pos['ann_median']:.3f} m")
    print(f"  PF median      : {pos['pf_median']:.3f} m")
    print(f"  ANN P90        : {pos['ann_p90']:.3f} m")
    print(f"  PF P90         : {pos['pf_p90']:.3f} m")
    print(f"  PF better      : {pos['samples_pf_better']}/{len(df)}")

    print("\nROLL ERROR")
    print(f"  ANN mean       : {roll['ann_mean']:.3f} deg")
    print(f"  PF mean        : {roll['pf_mean']:.3f} deg")
    print(f"  Mean reduction : {roll['mean_reduction_percent']:.2f}%")
    print(f"  ANN median     : {roll['ann_median']:.3f} deg")
    print(f"  PF median      : {roll['pf_median']:.3f} deg")
    print(f"  ANN P90        : {roll['ann_p90']:.3f} deg")
    print(f"  PF P90         : {roll['pf_p90']:.3f} deg")
    print(f"  PF better      : {roll['samples_pf_better']}/{len(df)}")

    print("\nGENERATED FILES")
    print(f"  {summary_path}")
    if plt is not None:
        for name in [
            "phase5_error_distributions.png",
            "phase5_position_error_comparison.png",
            "phase5_roll_error_comparison.png",
            "phase5_samplewise_comparison.png",
            "phase5_pose_examples.png",
        ]:
            print(f"  {RESULTS_DIR / name}")
    else:
        print("  matplotlib unavailable; plots were skipped.")

    print("\nPhase 5 complete.")


if __name__ == "__main__":
    main()
