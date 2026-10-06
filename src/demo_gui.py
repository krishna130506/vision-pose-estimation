import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from generate_data import (
    X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, P_MIN, P_MAX,
    IMG_W, IMG_H, CX, CY, NUM_FEATURES, project_features
)
from train_ann import MODEL_PATH, NORM_PATH, PoseANN, predict_topk, class_to_center
from particle_filter import particle_filter

RNG = np.random.default_rng(20261006)
PARTICLES = 2000
ITERATIONS = 8
TOPK = 5


class App:
    def __init__(self, root):
        self.root = root
        root.title("Vision Pose Estimation — ANN + Particle Filter")
        root.geometry("1180x760")
        root.minsize(1050, 700)

        self.model = None
        self.norm = None
        self.pose = None
        self.raw = None
        self.entries = {}

        outer = ttk.Frame(root, padding=14)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer, text="Vision-Based Pose Estimation",
            font=("Segoe UI", 17, "bold")
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text="Synthetic observation → ANN coarse estimate → Particle-filter refinement"
        ).pack(anchor="w", pady=(2, 12))

        main = ttk.Frame(outer)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main, width=390)
        left.pack(side="left", fill="y", padx=(0, 14))
        left.pack_propagate(False)

        box = ttk.LabelFrame(
            left, text="Test pose used to generate the observation", padding=10
        )
        box.pack(fill="x")

        fields = [
            ("X position (m)", "x", X_MIN, X_MAX, -50.0),
            ("Y position (m)", "y", Y_MIN, Y_MAX, 15.0),
            ("Z position (m)", "z", Z_MIN, Z_MAX, 6.0),
            ("Roll (deg)", "roll", P_MIN, P_MAX, 0.0),
        ]

        for r, (label, key, lo, hi, default) in enumerate(fields):
            ttk.Label(box, text=label).grid(row=r, column=0, sticky="w", pady=5)
            e = ttk.Entry(box, width=15)
            e.insert(0, str(default))
            e.grid(row=r, column=1, padx=8, pady=5)
            ttk.Label(box, text=f"[{lo:g}, {hi:g}]").grid(row=r, column=2)
            self.entries[key] = e

        buttons = ttk.Frame(box)
        buttons.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10, 2))
        ttk.Button(
            buttons, text="Generate Random Pose",
            command=self.random_pose
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            buttons, text="Run Estimation",
            command=self.run
        ).pack(side="left")

        results = ttk.LabelFrame(left, text="Results", padding=8)
        results.pack(fill="x", pady=(14, 0))

        self.text = tk.Text(
            results, height=19, width=45, font=("Consolas", 9),
            state="disabled", relief="flat", background="#f7f7f7"
        )
        self.text.pack(fill="x")

        view = ttk.LabelFrame(main, text="14-feature camera observation", padding=8)
        view.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(
            view, background="#111111", highlightthickness=1,
            highlightbackground="#888888"
        )
        self.canvas.pack(fill="both", expand=True)

        ttk.Label(
            view,
            text="Red dots = detected features. Hidden/dropped features are not plotted."
        ).pack(anchor="w", pady=(7, 0))

        self.status = tk.StringVar(value="Loading trained ANN...")
        ttk.Label(outer, textvariable=self.status).pack(anchor="w", pady=(8, 0))

        self.load_model()

    def load_model(self):
        try:
            if not Path(MODEL_PATH).exists():
                raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}\nRun train_ann.py first.")
            if not Path(NORM_PATH).exists():
                raise FileNotFoundError(f"Normalization file not found:\n{NORM_PATH}\nRun train_ann.py first.")

            ckpt = torch.load(MODEL_PATH, map_location="cpu")
            self.model = PoseANN(hidden=ckpt["hidden"])
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()

            import json
            self.norm = json.loads(Path(NORM_PATH).read_text(encoding="utf-8"))
            self.status.set("Ready — trained ANN loaded.")
            self.random_pose()
        except Exception as e:
            self.status.set("Startup failed.")
            messagebox.showerror("Startup error", str(e))

    def read_pose(self):
        try:
            p = np.array([
                float(self.entries["x"].get()),
                float(self.entries["y"].get()),
                float(self.entries["z"].get()),
                float(self.entries["roll"].get())
            ], dtype=np.float64)
        except ValueError:
            raise ValueError("Enter valid numbers for X, Y, Z and Roll.")

        checks = [
            ("X", p[0], X_MIN, X_MAX),
            ("Y", p[1], Y_MIN, Y_MAX),
            ("Z", p[2], Z_MIN, Z_MAX),
            ("Roll", p[3], P_MIN, P_MAX),
        ]
        for name, value, lo, hi in checks:
            if not lo <= value <= hi:
                raise ValueError(f"{name} must be between {lo:g} and {hi:g}.")

        if np.linalg.norm(p[:3]) < 10:
            raise ValueError("Camera-aircraft distance must be at least 10 m.")
        return p

    def set_pose(self, p):
        for key, value in zip(("x", "y", "z", "roll"), p):
            self.entries[key].delete(0, tk.END)
            self.entries[key].insert(0, f"{value:.2f}")

    def random_pose(self):
        for _ in range(1000):
            p = np.array([
                RNG.uniform(X_MIN, X_MAX),
                RNG.uniform(Y_MIN, Y_MAX),
                RNG.uniform(Z_MIN, Z_MAX),
                RNG.uniform(P_MIN, P_MAX)
            ])
            if np.linalg.norm(p[:3]) >= 10:
                self.set_pose(p)
                self.make_observation(p)
                self.clear_results()
                self.status.set("Random test pose generated. Click Run Estimation.")
                return
        messagebox.showerror("Error", "Could not generate a valid pose.")

    def make_observation(self, pose):
        u, v, visible = project_features(
            pose[None, :], rng=RNG, add_noise=True
        )
        u, v, visible = u[0], v[0], visible[0]
        u = np.where(visible, u, 0.0)
        v = np.where(visible, v, 0.0)
        occ = (~visible).astype(np.int8)

        raw = np.empty((NUM_FEATURES, 3), dtype=np.float32)
        raw[:, 0], raw[:, 1], raw[:, 2] = u, v, occ
        self.pose = pose.copy()
        self.raw = raw.reshape(-1)
        self.draw(u, v, visible)

    def draw(self, u, v, visible):
        self.canvas.delete("all")
        self.root.update_idletasks()
        w = max(self.canvas.winfo_width(), 500)
        h = max(self.canvas.winfo_height(), 400)
        scale = min(w / IMG_W, h / IMG_H) * 0.92
        iw, ih = IMG_W * scale, IMG_H * scale
        ox, oy = (w - iw) / 2, (h - ih) / 2

        self.canvas.create_rectangle(
            ox, oy, ox + iw, oy + ih, outline="#555555"
        )
        cx, cy = ox + CX * scale, oy + CY * scale
        self.canvas.create_line(cx - 12, cy, cx + 12, cy, fill="#666666")
        self.canvas.create_line(cx, cy - 12, cx, cy + 12, fill="#666666")

        for i in range(NUM_FEATURES):
            if not visible[i]:
                continue
            px, py = ox + float(u[i]) * scale, oy + float(v[i]) * scale
            r = 5
            self.canvas.create_oval(
                px-r, py-r, px+r, py+r,
                fill="#ff5555", outline="white"
            )
            self.canvas.create_text(
                px+10, py-9, text=str(i+1), fill="white",
                anchor="w", font=("Segoe UI", 8, "bold")
            )

        n = int(np.sum(visible))
        self.canvas.create_text(
            ox+10, oy+10,
            text=f"Detected: {n}/14   Hidden/dropped: {14-n}/14",
            fill="white", anchor="nw",
            font=("Segoe UI", 10, "bold")
        )

    def clear_results(self):
        self.text.config(state="normal")
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, "Click  Run Estimation  to run the model.\n")
        self.text.config(state="disabled")

    def run(self):
        if self.model is None:
            messagebox.showerror("Error", "ANN model is not loaded.")
            return

        try:
            pose = self.read_pose()
            self.make_observation(pose)
            self.status.set("Running ANN + particle filter...")
            self.root.update_idletasks()

            classes, probs = predict_topk(
                self.model, self.norm, self.raw[None, :], k=TOPK
            )
            ann_class = int(classes[0, 0])
            ann_prob = float(probs[0, 0])
            ann_pose = class_to_center(np.array([ann_class]))[0].astype(float)

            pf_pose, _ = particle_filter(
                self.raw, classes[0], probs[0],
                n_particles=PARTICLES,
                iterations=ITERATIONS,
                rng=RNG
            )

            ann_pos = float(np.linalg.norm(ann_pose[:3] - pose[:3]))
            pf_pos = float(np.linalg.norm(pf_pose[:3] - pose[:3]))
            ann_roll = float(abs(ann_pose[3] - pose[3]))
            pf_roll = float(abs(pf_pose[3] - pose[3]))

            pos_imp = 100 * (ann_pos - pf_pos) / ann_pos if ann_pos else 0
            roll_imp = 100 * (ann_roll - pf_roll) / ann_roll if ann_roll else 0

            lines = [
                "GROUND TRUTH",
                f"  X     : {pose[0]:8.3f} m",
                f"  Y     : {pose[1]:8.3f} m",
                f"  Z     : {pose[2]:8.3f} m",
                f"  Roll  : {pose[3]:8.3f} deg",
                "",
                "ANN — COARSE ESTIMATE",
                f"  Class : {ann_class:4d} / 4800",
                f"  Prob. : {100*ann_prob:7.2f}%",
                f"  X     : {ann_pose[0]:8.3f} m",
                f"  Y     : {ann_pose[1]:8.3f} m",
                f"  Z     : {ann_pose[2]:8.3f} m",
                f"  Roll  : {ann_pose[3]:8.3f} deg",
                f"  Position error : {ann_pos:6.3f} m",
                f"  Roll error     : {ann_roll:6.3f} deg",
                "",
                "ANN + PARTICLE FILTER",
                f"  X     : {pf_pose[0]:8.3f} m",
                f"  Y     : {pf_pose[1]:8.3f} m",
                f"  Z     : {pf_pose[2]:8.3f} m",
                f"  Roll  : {pf_pose[3]:8.3f} deg",
                f"  Position error : {pf_pos:6.3f} m",
                f"  Roll error     : {pf_roll:6.3f} deg",
                "",
                "REFINEMENT",
                f"  Position improvement : {pos_imp:6.2f}%",
                f"  Roll improvement     : {roll_imp:6.2f}%",
            ]

            self.text.config(state="normal")
            self.text.delete("1.0", tk.END)
            self.text.insert(tk.END, "\n".join(lines))
            self.text.config(state="disabled")
            self.status.set("Complete — particle filter refined the ANN estimate.")

        except Exception as e:
            self.status.set("Estimation failed.")
            messagebox.showerror("Estimation error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
