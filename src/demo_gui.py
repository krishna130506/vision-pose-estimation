"""
demo_gui.py
===========
Clean, professional academic demonstration GUI for:
"Vision-Based Pose Estimation Using ANN and Particle Filtering"
(Rohan Punnoose, Stanford CS229 reproduction)

Features:
- Minimal, clean layout with restrained blue accents and readable typography.
- Four validated pose inputs with range labels and safety separation check.
- High-contrast synthetic camera canvas rendering 14 clearly visible landmarks.
- Clean, compact pose results table (Ground Truth, ANN Coarse, ANN + PF).
- Direct side-by-side error comparison reporting actual calculated metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project imports & path configuration
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generate_data import (
    X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, P_MIN, P_MAX,
    IMG_W, IMG_H, CX, CY, NUM_FEATURES, project_features
)
from train_ann import MODEL_PATH, NORM_PATH, PoseANN, predict_topk, class_to_center
from particle_filter import particle_filter

# Constants
RNG = np.random.default_rng(20261006)
PARTICLES = 2000
ITERATIONS = 8
TOPK = 5
MIN_RANGE_M = 10.0


class PoseEstimationApp:
    """Academic demonstration GUI for ANN + Particle Filter pose estimation."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Vision-Based Pose Estimation — ANN & Particle Filtering")
        self.root.geometry("1020x660")
        self.root.minsize(920, 580)

        # Apply standard clean ttk style
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        self._configure_styles()

        # State variables
        self.model = None
        self.norm = None
        self.current_pose = None
        self.current_raw = None
        self.entries: dict[str, ttk.Entry] = {}
        self.last_observation = None

        # Build UI layout
        self._build_header()
        self._build_main_content()
        self._build_status_bar()

        # Load trained weights
        self._load_model_artifacts()

    def _configure_styles(self) -> None:
        """Sets restrained, readable academic typography and spacing."""
        bg_col = "#f8fafc"
        self.root.configure(background=bg_col)

        self.style.configure(".", background=bg_col, foreground="#0f172a", font=("Segoe UI", 9))
        self.style.configure("TFrame", background=bg_col)
        self.style.configure("TLabelframe", background=bg_col, relief="groove")
        self.style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"), foreground="#1e3a8a", background=bg_col)

        # Buttons
        self.style.configure(
            "Primary.TButton",
            font=("Segoe UI", 9, "bold"),
            background="#2563eb",
            foreground="#ffffff",
            padding=(10, 5),
        )
        self.style.map("Primary.TButton", background=[("active", "#1d4ed8"), ("disabled", "#94a3b8")])

        self.style.configure(
            "Secondary.TButton",
            font=("Segoe UI", 9),
            background="#e2e8f0",
            foreground="#0f172a",
            padding=(8, 5),
        )
        self.style.map("Secondary.TButton", background=[("active", "#cbd5e1")])

        # Treeview tables
        self.style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"), foreground="#0f172a")
        self.style.configure("Treeview", font=("Consolas", 9), rowheight=24)

    def _build_header(self) -> None:
        """Section A: Header with title, subtitle, and concise description."""
        hdr = ttk.Frame(self.root, padding=(16, 12, 16, 6))
        hdr.pack(fill="x")

        ttk.Label(hdr, text="Vision-Based Pose Estimation", font=("Segoe UI", 16, "bold"), foreground="#0f172a").pack(anchor="w")
        ttk.Label(hdr, text="ANN Classification + Particle-Filter Refinement", font=("Segoe UI", 10, "bold"), foreground="#2563eb").pack(anchor="w", pady=(1, 2))
        ttk.Label(
            hdr,
            text="Estimates 4-DOF relative aircraft pose from 14 synthetic visual features using coarse ANN classification and continuous particle-filter refinement.",
            font=("Segoe UI", 8),
            foreground="#475569",
        ).pack(anchor="w")

    def _build_main_content(self) -> None:
        """Main two-column body: Left = Inputs + Canvas; Right = Results Table + Error Table."""
        body = ttk.Frame(self.root, padding=(16, 6, 16, 6))
        body.pack(fill="both", expand=True)

        body.columnconfigure(0, weight=1, minsize=440)
        body.columnconfigure(1, weight=1, minsize=460)
        body.rowconfigure(0, weight=1)

        # --- LEFT COLUMN: Controls & Observation ---
        left_col = ttk.Frame(body)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left_col.rowconfigure(1, weight=1)
        left_col.columnconfigure(0, weight=1)

        self._build_pose_input_card(left_col)
        self._build_observation_card(left_col)

        # --- RIGHT COLUMN: Results & Errors ---
        right_col = ttk.Frame(body)
        right_col.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right_col.rowconfigure(0, weight=0)
        right_col.rowconfigure(1, weight=0)
        right_col.columnconfigure(0, weight=1)

        self._build_results_card(right_col)
        self._build_error_card(right_col)

    def _build_pose_input_card(self, parent: ttk.Frame) -> None:
        """Section B: Four editable inputs with ranges and triggers."""
        box = ttk.LabelFrame(parent, text="Pose Input", padding=10)
        box.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        fields = [
            ("Forward X (m):", "x", X_MIN, X_MAX, -50.0),
            ("Lateral Y (m):", "y", Y_MIN, Y_MAX, 15.0),
            ("Vertical Z (m):", "z", Z_MIN, Z_MAX, 6.0),
            ("Roll (deg):", "roll", P_MIN, P_MAX, 0.0),
        ]

        grid = ttk.Frame(box)
        grid.pack(fill="x")
        grid.columnconfigure(2, weight=1)

        for r, (label, key, lo, hi, default_val) in enumerate(fields):
            ttk.Label(grid, text=label, font=("Segoe UI", 9, "bold")).grid(row=r, column=0, sticky="w", pady=2, padx=(0, 6))

            ent = ttk.Entry(grid, width=10, font=("Consolas", 9))
            ent.insert(0, f"{default_val:.2f}")
            ent.grid(row=r, column=1, sticky="w", pady=2)
            self.entries[key] = ent

            ttk.Label(grid, text=f"Range: [{lo:g}, {hi:g}]", font=("Segoe UI", 8), foreground="#64748b").grid(row=r, column=2, sticky="w", padx=(8, 0), pady=2)

        # Button row
        btn_bar = ttk.Frame(box)
        btn_bar.pack(fill="x", pady=(8, 0))

        ttk.Button(btn_bar, text="Generate Random Pose", style="Secondary.TButton", command=self.on_random_pose).pack(side="left", padx=(0, 8))
        ttk.Button(btn_bar, text="Run Estimation", style="Primary.TButton", command=self.on_run_estimation).pack(side="left")

    def _build_observation_card(self, parent: ttk.Frame) -> None:
        """Section C: High-contrast camera viewport showing 14 landmark points."""
        box = ttk.LabelFrame(parent, text="Synthetic Camera Observation", padding=8)
        box.grid(row=1, column=0, sticky="nsew")
        box.rowconfigure(1, weight=1)
        box.columnconfigure(0, weight=1)

        # Legend & Count header bar
        bar = ttk.Frame(box)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        self.lbl_counts = ttk.Label(bar, text="Detected: — / 14  |  Occluded: — / 14", font=("Segoe UI", 9, "bold"), foreground="#0f172a")
        self.lbl_counts.pack(side="left")

        leg = ttk.Frame(bar)
        leg.pack(side="right")
        ttk.Label(leg, text="● Detected", font=("Segoe UI", 8, "bold"), foreground="#10b981").pack(side="left", padx=(0, 8))
        ttk.Label(leg, text="○ Occluded / Dropped", font=("Segoe UI", 8), foreground="#64748b").pack(side="left")

        # Responsive camera viewport canvas
        self.canvas = tk.Canvas(box, bg="#0b1120", highlightthickness=1, highlightbackground="#334155", height=240)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda event: self._redraw_observation())

        # Descriptive disclaimer
        ttk.Label(
            box,
            text="* Synthetic camera-projected structural landmarks (analytical pinhole projection + ray-cast self-occlusion test + 0.5px noise), not real aircraft photographs.",
            font=("Segoe UI", 7, "italic"),
            foreground="#64748b",
            wraplength=420,
        ).grid(row=2, column=0, sticky="w", pady=(4, 0))

    def _build_results_card(self, parent: ttk.Frame) -> None:
        """Section D: One simple table for Ground Truth, ANN Coarse, and ANN + PF."""
        box = ttk.LabelFrame(parent, text="Pose Results", padding=10)
        box.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        cols = ("estimator", "x", "y", "z", "roll")
        self.tree_pose = ttk.Treeview(box, columns=cols, show="headings", height=3, selectmode="none")

        self.tree_pose.heading("estimator", text="Estimator", anchor="w")
        self.tree_pose.heading("x", text="X (m)", anchor="e")
        self.tree_pose.heading("y", text="Y (m)", anchor="e")
        self.tree_pose.heading("z", text="Z (m)", anchor="e")
        self.tree_pose.heading("roll", text="Roll (°)", anchor="e")

        self.tree_pose.column("estimator", width=170, anchor="w")
        self.tree_pose.column("x", width=65, anchor="e")
        self.tree_pose.column("y", width=65, anchor="e")
        self.tree_pose.column("z", width=65, anchor="e")
        self.tree_pose.column("roll", width=65, anchor="e")

        self.tree_pose.pack(fill="x")

        # Insert empty rows
        self.row_gt = self.tree_pose.insert("", "end", values=("Ground Truth", "—", "—", "—", "—"))
        self.row_ann = self.tree_pose.insert("", "end", values=("ANN Coarse Estimate", "—", "—", "—", "—"))
        self.row_pf = self.tree_pose.insert("", "end", values=("ANN + Particle Filter", "—", "—", "—", "—"))

        # Secondary label for ANN predicted class & probability
        self.lbl_ann_class = ttk.Label(
            box,
            text="ANN Prediction: Class — / 4800   |   Predicted Class Softmax Probability: — %",
            font=("Segoe UI", 8),
            foreground="#475569",
        )
        self.lbl_ann_class.pack(anchor="w", pady=(6, 0))

    def _build_error_card(self, parent: ttk.Frame) -> None:
        """Section E: Clean error comparison table showing actual calculated errors."""
        box = ttk.LabelFrame(parent, text="Error Comparison", padding=10)
        box.grid(row=1, column=0, sticky="ew")

        cols = ("metric", "ann_err", "pf_err", "diff")
        self.tree_err = ttk.Treeview(box, columns=cols, show="headings", height=2, selectmode="none")

        self.tree_err.heading("metric", text="Error Metric", anchor="w")
        self.tree_err.heading("ann_err", text="ANN Coarse Error", anchor="e")
        self.tree_err.heading("pf_err", text="Refined (PF) Error", anchor="e")
        self.tree_err.heading("diff", text="Difference", anchor="e")

        self.tree_err.column("metric", width=160, anchor="w")
        self.tree_err.column("ann_err", width=105, anchor="e")
        self.tree_err.column("pf_err", width=105, anchor="e")
        self.tree_err.column("diff", width=160, anchor="e")

        self.tree_err.pack(fill="x")

        # Initial placeholder rows
        self.row_pos_err = self.tree_err.insert("", "end", values=("3-D Position Euclidean Error", "— m", "— m", "—"))
        self.row_roll_err = self.tree_err.insert("", "end", values=("Roll Absolute Error", "— deg", "— deg", "—"))

        # One concise note
        ttk.Label(
            box,
            text="ANN provides coarse global class localization; Particle Filter performs continuous landmark-matching refinement.",
            font=("Segoe UI", 8, "italic"),
            foreground="#64748b",
            wraplength=440,
        ).pack(anchor="w", pady=(8, 0))

    def _build_status_bar(self) -> None:
        """Clean bottom status strip."""
        bar = ttk.Frame(self.root, padding=(16, 4, 16, 6))
        bar.pack(fill="x", side="bottom")

        self.status_text = tk.StringVar(value="Initializing demonstration GUI...")
        ttk.Label(bar, textvariable=self.status_text, font=("Segoe UI", 8), foreground="#334155").pack(side="left")

        self.model_info_text = tk.StringVar(value="Loading model...")
        ttk.Label(bar, textvariable=self.model_info_text, font=("Segoe UI", 8), foreground="#64748b").pack(side="right")

    # -----------------------------------------------------------------------
    # ML Loading & Validation
    # -----------------------------------------------------------------------
    def _load_model_artifacts(self) -> None:
        """Loads pre-trained ANN checkpoint and normalization dictionary."""
        try:
            if not Path(MODEL_PATH).exists():
                raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}\nRun src/train_ann.py first.")
            if not Path(NORM_PATH).exists():
                raise FileNotFoundError(f"Normalization file not found: {NORM_PATH}\nRun src/train_ann.py first.")

            ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            self.model = PoseANN(hidden=ckpt["hidden"])
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()

            self.norm = json.loads(Path(NORM_PATH).read_text(encoding="utf-8"))
            best_epoch = ckpt.get("best_epoch", 57)
            self.model_info_text.set(f"Loaded: models/pose_ann.pt (Best epoch {best_epoch})")
            self.status_text.set("Ready — Click 'Run Estimation' or 'Generate Random Pose'.")

            # Generate initial preview
            self.on_random_pose()

        except Exception as exc:
            self.status_text.set(f"Startup error: {exc}")
            self.model_info_text.set("Model load failed.")
            messagebox.showerror("Model Loading Error", str(exc))

    def read_pose_inputs(self) -> np.ndarray:
        """Validates numerical entries and enforces Table I physical bounds."""
        try:
            x_val = float(self.entries["x"].get().strip())
            y_val = float(self.entries["y"].get().strip())
            z_val = float(self.entries["z"].get().strip())
            r_val = float(self.entries["roll"].get().strip())
        except ValueError:
            raise ValueError("All pose inputs (X, Y, Z, Roll) must be valid floating-point numbers.")

        checks = [
            ("Forward X", x_val, X_MIN, X_MAX, "m"),
            ("Lateral Y", y_val, Y_MIN, Y_MAX, "m"),
            ("Vertical Z", z_val, Z_MIN, Z_MAX, "m"),
            ("Roll Angle φ", r_val, P_MIN, P_MAX, "°"),
        ]
        for name, val, lo, hi, unit in checks:
            if not (lo <= val <= hi):
                raise ValueError(f"{name} ({val:g} {unit}) is outside permitted bounds [{lo:g}, {hi:g}] {unit}.")

        dist = float(np.linalg.norm([x_val, y_val, z_val]))
        if dist < MIN_RANGE_M:
            raise ValueError(f"Follower-leader distance ({dist:.2f} m) must be at least {MIN_RANGE_M:.1f} m to avoid collision.")

        return np.array([x_val, y_val, z_val, r_val], dtype=np.float64)

    def set_pose_inputs(self, pose: np.ndarray) -> None:
        """Updates text entry boxes with formatted pose values."""
        for key, val in zip(("x", "y", "z", "roll"), pose):
            self.entries[key].delete(0, tk.END)
            self.entries[key].insert(0, f"{val:.2f}")

    def on_random_pose(self) -> None:
        """Samples a random valid pose within physical constraints."""
        for _ in range(500):
            candidate = np.array([
                RNG.uniform(X_MIN, X_MAX),
                RNG.uniform(Y_MIN, Y_MAX),
                RNG.uniform(Z_MIN, Z_MAX),
                RNG.uniform(P_MIN, P_MAX),
            ])
            if np.linalg.norm(candidate[:3]) >= MIN_RANGE_M:
                self.set_pose_inputs(candidate)
                self.generate_and_draw_observation(candidate)
                self._clear_results_display()
                self.status_text.set("Random test pose generated. Click 'Run Estimation'.")
                return
        messagebox.showwarning("Warning", "Could not sample a valid pose meeting minimum separation.")

    def _clear_results_display(self) -> None:
        """Clears results tables before estimation."""
        if self.current_pose is not None:
            self.tree_pose.item(self.row_gt, values=("Ground Truth", f"{self.current_pose[0]:.2f}", f"{self.current_pose[1]:.2f}", f"{self.current_pose[2]:.2f}", f"{self.current_pose[3]:.2f}"))
        self.tree_pose.item(self.row_ann, values=("ANN Coarse Estimate", "—", "—", "—", "—"))
        self.tree_pose.item(self.row_pf, values=("ANN + Particle Filter", "—", "—", "—", "—"))
        self.lbl_ann_class.config(text="ANN Prediction: Class — / 4800   |   Predicted Class Softmax Probability: — %")
        self.tree_err.item(self.row_pos_err, values=("3-D Position Euclidean Error", "— m", "— m", "—"))
        self.tree_err.item(self.row_roll_err, values=("Roll Absolute Error |φ_err|", "— °", "— °", "—"))

    # -----------------------------------------------------------------------
    # Canvas Observation Rendering
    # -----------------------------------------------------------------------
    def generate_and_draw_observation(self, pose: np.ndarray) -> None:
        """Computes synthetic 2D projection and updates the camera canvas."""
        u, v, visible = project_features(pose[None, :], rng=RNG, add_noise=True)
        u, v, visible = u[0], v[0], visible[0]

        u_clean = np.where(visible, u, 0.0)
        v_clean = np.where(visible, v, 0.0)
        occ = (~visible).astype(np.int8)

        raw = np.empty((NUM_FEATURES, 3), dtype=np.float32)
        raw[:, 0], raw[:, 1], raw[:, 2] = u_clean, v_clean, occ
        self.current_pose = pose.copy()
        self.current_raw = raw.reshape(-1)
        self.last_observation = (u, v, visible)

        n_vis = int(np.sum(visible))
        self.lbl_counts.config(text=f"Detected: {n_vis} / 14   |   Occluded: {14 - n_vis} / 14")

        # Update Ground Truth in table
        self.tree_pose.item(self.row_gt, values=("Ground Truth", f"{pose[0]:.2f}", f"{pose[1]:.2f}", f"{pose[2]:.2f}", f"{pose[3]:.2f}"))

        self._render_canvas(u, v, visible)

    def _redraw_observation(self) -> None:
        """Handles canvas resize events smoothly."""
        if self.last_observation is not None:
            u, v, visible = self.last_observation
            self._render_canvas(u, v, visible)

    def _render_canvas(self, u: np.ndarray, v: np.ndarray, visible: np.ndarray) -> None:
        """Renders 14 landmarks with clearly visible markers and feature IDs."""
        self.canvas.delete("all")
        cw = max(self.canvas.winfo_width(), 360)
        ch = max(self.canvas.winfo_height(), 180)

        scale = min(cw / IMG_W, ch / IMG_H) * 0.88
        vw, vh = IMG_W * scale, IMG_H * scale
        ox = (cw - vw) / 2.0
        oy = (ch - vh) / 2.0

        # Viewport frame
        self.canvas.create_rectangle(ox, oy, ox + vw, oy + vh, outline="#334155", fill="#070c18", width=1)

        # Center crosshair reticle at principal point
        cx_s = ox + CX * scale
        cy_s = oy + CY * scale
        self.canvas.create_line(cx_s - 12, cy_s, cx_s + 12, cy_s, fill="#38bdf8", width=1)
        self.canvas.create_line(cx_s, cy_s - 12, cx_s, cy_s + 12, fill="#38bdf8", width=1)
        self.canvas.create_oval(cx_s - 4, cy_s - 4, cx_s + 4, cy_s + 4, outline="#38bdf8", width=1)

        # Draw detected landmarks clearly (radius = 6 px)
        r = 6
        for i in range(NUM_FEATURES):
            if not visible[i]:
                continue
            px = ox + float(u[i]) * scale
            py = oy + float(v[i]) * scale

            # Emerald dot with white border
            self.canvas.create_oval(px - r, py - r, px + r, py + r, fill="#10b981", outline="#ffffff", width=1.5)
            # Feature ID text
            self.canvas.create_text(
                px + 9, py - 8,
                text=str(i + 1),
                fill="#f8fafc",
                anchor="w",
                font=("Segoe UI", 9, "bold")
            )

    # -----------------------------------------------------------------------
    # Estimation Execution
    # -----------------------------------------------------------------------
    def on_run_estimation(self) -> None:
        """Executes ANN classification and Particle Filter refinement."""
        if self.model is None or self.norm is None:
            messagebox.showerror("Error", "Trained model not loaded.")
            return

        try:
            pose = self.read_pose_inputs()
            self.generate_and_draw_observation(pose)
            self.status_text.set("Running ANN classification and Particle Filter refinement...")
            self.root.update_idletasks()

            # 1. ANN Coarse Inference
            top_classes, top_probs = predict_topk(self.model, self.norm, self.current_raw[None, :], k=TOPK)
            ann_class = int(top_classes[0, 0])
            ann_prob = float(top_probs[0, 0])
            ann_pose = class_to_center(np.array([ann_class]))[0].astype(np.float64)

            # 2. Particle Filter Refinement
            pf_pose, _ = particle_filter(
                self.current_raw,
                top_classes[0],
                top_probs[0],
                n_particles=PARTICLES,
                iterations=ITERATIONS,
                rng=RNG,
            )

            # Errors
            ann_pos_err = float(np.linalg.norm(ann_pose[:3] - pose[:3]))
            pf_pos_err = float(np.linalg.norm(pf_pose[:3] - pose[:3]))
            ann_roll_err = float(abs(ann_pose[3] - pose[3]))
            pf_roll_err = float(abs(pf_pose[3] - pose[3]))

            # Update Pose Results Table
            self.tree_pose.item(self.row_gt, values=("Ground Truth", f"{pose[0]:.2f}", f"{pose[1]:.2f}", f"{pose[2]:.2f}", f"{pose[3]:.2f}"))
            self.tree_pose.item(self.row_ann, values=("ANN Coarse Estimate", f"{ann_pose[0]:.2f}", f"{ann_pose[1]:.2f}", f"{ann_pose[2]:.2f}", f"{ann_pose[3]:.2f}"))
            self.tree_pose.item(self.row_pf, values=("ANN + Particle Filter", f"{pf_pose[0]:.2f}", f"{pf_pose[1]:.2f}", f"{pf_pose[2]:.2f}", f"{pf_pose[3]:.2f}"))

            self.lbl_ann_class.config(
                text=f"ANN Prediction: Class {ann_class} / 4800   |   Predicted Class Softmax Probability: {ann_prob * 100:.2f} %"
            )

            # Update Error Comparison Table
            pos_diff = ann_pos_err - pf_pos_err
            if ann_pos_err > 1e-9:
                pos_pct = 100.0 * pos_diff / ann_pos_err
                pos_diff_str = f"-{pos_diff:.3f} m ({pos_pct:.1f}% reduction)" if pos_diff >= 0 else f"+{-pos_diff:.3f} m (No improvement)"
            else:
                pos_diff_str = "0.000 m"

            roll_diff = ann_roll_err - pf_roll_err
            if ann_roll_err > 1e-9:
                roll_pct = 100.0 * roll_diff / ann_roll_err
                roll_diff_str = f"-{roll_diff:.3f} deg ({roll_pct:.1f}% reduction)" if roll_diff >= 0 else f"+{-roll_diff:.3f} deg (No improvement)"
            else:
                roll_diff_str = "0.000 deg"

            self.tree_err.item(
                self.row_pos_err,
                values=("3-D Position Euclidean Error", f"{ann_pos_err:.3f} m", f"{pf_pos_err:.3f} m", pos_diff_str)
            )
            self.tree_err.item(
                self.row_roll_err,
                values=("Roll Absolute Error", f"{ann_roll_err:.3f} deg", f"{pf_roll_err:.3f} deg", roll_diff_str)
            )

            self.status_text.set("Estimation complete.")

        except Exception as exc:
            self.status_text.set(f"Execution error: {exc}")
            messagebox.showerror("Execution Error", str(exc))


def main() -> None:
    root = tk.Tk()
    PoseEstimationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
