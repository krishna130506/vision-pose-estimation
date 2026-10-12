# Vision-Based Aircraft Pose Estimation Using ANN and Particle Filtering

[![Course](https://img.shields.io/badge/Course-UE24CS352A%20Machine%20Learning-blue.svg)](https://github.com/krishna130506/vision-pose-estimation)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)](https://github.com/krishna130506/vision-pose-estimation)

An academic reproduction and extension of **Rohan Punnoose (Stanford CS229, 2020)**:  
> *"Vision-Based Precision Pose Estimation for Autonomous Formation Flying"*

---

## 1. Project Overview & Problem Statement

Autonomous formation flight of unmanned aerial vehicles (UAVs) requires relative localization of sub-meter precision to safely maintain aerodynamic formation geometry. While global navigation satellite systems (GNSS) are standard, they are susceptible to signal loss, jamming, and latency. Passive computer vision offers an autonomous, non-emitting alternative by tracking visual features on the leading aircraft using an onboard follower camera.

Estimating continuous 6-DOF pose directly from raw camera images via end-to-end regression is difficult due to extreme perspective variations, self-occlusions, and multi-modal pose ambiguities. Following the methodology of Punnoose (2020), this project formulates relative pose estimation across four primary dimensions:
- **Forward X position** ($X \in [-100, 50]$ m)
- **Lateral Y position** ($Y \in [-50, 50]$ m)
- **Vertical Z position** ($Z \in [-20, 20]$ m)
- **Boresight Roll angle** ($\phi \in [-45, 45]^\circ$)

*(Relative pitch and yaw are assumed zero during steady formation flight).*

To overcome the limitations of pure regression, this project implements a **two-stage hybrid framework**:
1. **Artificial Neural Network (ANN):** Rapidly maps 14 visual landmarks (42 inputs) into a discrete coarse pose class among **4,800 classes**.
2. **Particle Filter (PF):** Uses candidate hypotheses from the ANN to seed a continuous Monte Carlo filter that refines the coarse prediction to sub-meter accuracy by comparing projected and observed landmark coordinates.

---

## 2. System Pipeline Architecture

```text
       Synthetic Aircraft Pose [X, Y, Z, φ]
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│  Pinhole Projection & Fuselage Self-Occlusion Ray Test │
│  (1280×960 camera, f = 1500 px, σ = 0.5 px noise)      │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
        14 Aircraft Structural Landmarks (42-D)
             14 × (u, v, occlusion_flag)
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│  Stage 1: Artificial Neural Network (PoseANN)          │
│  42 Inputs ──► Linear(100) ──► Linear(100) ──► 4,800   │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
           Top-k Coarse Pose Hypotheses
           (Class centers & Softmax probabilities)
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│  Stage 2: Model-Based Particle Filter (PF)             │
│  • N = 2,000 particles initialized around top-5 ANN    │
│  • Analytical forward landmark projection              │
│  • Robust Gaussian observation likelihood              │
│  • Systematic resampling & 8-iteration jitter decay    │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
      Continuous Refined Aircraft Pose [X, Y, Z, φ]
              (Mean Position Error ~ 0.18 m)
```

---

## 3. Synthetic Dataset & Physical Observation Model

Because the reference paper did not publish CAD models or 3D coordinate listings, an F-16 scale simplified 3D airframe (15 m length, 10 m wingspan) comprising 14 key structural landmarks was modeled:
`nose_tip`, `tail_exhaust`, `left_wingtip`, `right_wingtip`, `fin_tip`, `left_stab_tip`, `right_stab_tip`, `canopy_top`, `left_wing_mid`, `right_wing_mid`, `left_wing_store`, `right_wing_store`, `intake_lip`, and `ventral_fin`.

### Observation Model
- **Camera Sensor:** Pinhole camera ($1280 \times 960$ px, principal point at center $c_x=640, c_y=480$, focal length $f=1500$ px $\approx 46^\circ$ horizontal FOV).
- **Gimbaling:** The follower camera at $(X, Y, Z)$ always points directly at the leader body origin $(-c / \|c\|)$.
- **Self-Occlusion Ray Casting:** An analytical fuselage ellipsoid (semi-axes $[6.0, 0.6, 0.7]$ m) performs ray-intersection testing. Features behind the camera, outside the viewport, or blocked by the fuselage are marked occluded ($occ = 1$) and have image coordinates zeroed ($u=0, v=0$).
- **Noise:** Detected features receive $\sigma = 0.5$ px Gaussian measurement noise and a $2\%$ random detector dropout rate.
- **Input Dimension:** $14 \times 3 = 42$ numerical values per sample.

### Pose Discretization (Table I of Punnoose 2020)
| Dimension | Range | Number of Bins | Bin Cell Width |
|---|---:|---:|---:|
| **Forward X** | $-100.0$ to $50.0$ m | 10 | 15.0 m |
| **Lateral Y** | $-50.0$ to $50.0$ m | 10 | 10.0 m |
| **Vertical Z** | $-20.0$ to $20.0$ m | 8 | 5.0 m |
| **Boresight Roll $\phi$** | $-45.0$ to $45.0^\circ$ | 6 | 15.0° |
| **Total Classes** | — | **4,800** | — |

The dataset contains exactly **144,000 samples** ($4,800 \times 30$ samples/class), with collision avoidance enforcing a minimum aircraft separation $\ge 10.0$ m. All 144,000 samples maintain 100% bijective class encode/decode consistency.

---

## 4. Stage 1: Artificial Neural Network (ANN)

Following Section IV-D of the Punnoose paper:
- **Architecture:** `42 -> Linear(100) -> ReLU -> Linear(100) -> ReLU -> Linear(4800)` ($534,400$ trainable parameters).
- **Preprocessing:** Pixel coordinates are centered and scaled: $(u - c_x)/(W/2), (v - c_y)/(H/2)$, with occluded coordinates zeroed. Standard $z$-score normalization is fitted strictly on the training set.
- **Split:** Stratified 80/20 train/validation split ($115,200$ training samples / $28,800$ validation samples; exactly 6 validation samples per class).
- **Training:** 60 epochs using Adam ($\text{lr} = 0.001$, batch size 256) with Cosine Annealing learning rate schedule. Best checkpoint saved at **Epoch 57**.

### Validation Checkpoint Performance (Best Epoch 57)
- **Validation Loss:** `1.4357` (Cross-Entropy)
- **Top-1 Exact Class Accuracy:** **62.85%** *(Chance level = 0.021%)*
- **Top-5 Candidate Accuracy:** **93.02%**
- **Within $\pm 1$ Bin (All 4 Dimensions):** **99.03%** *(Within $\pm 2$ bins: 99.76%)*
- **Per-Dimension Exact Bin Accuracy:** $X$: 91.6%, $Y$: 87.9%, $Z$: 82.5%, Roll: 92.0%
- **Physical Error (Predicted Cell Center vs. True Pose):**
  - Position Error: Median **5.53 m**, Mean **5.78 m**, 90th percentile **8.77 m**
  - Roll Error: Median **3.84°**, Mean **4.16°**

*(Punnoose reported ~54% Top-1 validation accuracy on synthetic ray-traced imagery).*

---

## 5. Stage 2: Particle Filter Refinement

The particle filter refines the coarse classification hypothesis to continuous coordinates:
1. **Hypothesis-Weighted Initialization:** 2,000 particles are sampled across the top-5 ANN class centers, distributed proportionally to their softmax probabilities with Gaussian spread scaled to bin widths.
2. **Measurement Likelihood:** For each particle, 14 landmarks are projected. A Gaussian log-likelihood with $\sigma = 2.0$ px is computed on visible landmarks, clipped at $8\sigma$ to handle dropouts. Visibility mismatches incur a log penalty of $3.0$.
3. **Resampling & Jitter Annealing:** Low-variance systematic resampling focuses particles around high-probability modes. Proposal jitter standard deviation anneals from $30\%$ of bin width down to $4\%$ across 8 iterations.
4. **Output:** Continuous pose is estimated via the weighted expectation across the particle distribution.

---

## 6. Evaluation Results & Performance Analysis

### Primary 100-Sample Paired Comparison (Phase 4 Validation)
| Metric | ANN Coarse Baseline | ANN + Particle Filter | Measured Improvement |
|---|---:|---:|---:|
| **Mean Position Error** | 5.017 m | **0.179 m** | **96.43% reduction** |
| **Median Position Error** | 5.137 m | **0.120 m** | **97.67% reduction** |
| **90th Percentile Position Error** | 7.285 m | **0.332 m** | **95.44% reduction** |
| **Mean Roll Error** | 4.108° | **0.414°** | **89.92% reduction** |
| **Median Roll Error** | 3.692° | **0.069°** | **98.13% reduction** |
| **90th Percentile Roll Error** | 6.360° | **0.173°** | **97.28% reduction** |

- **Position Success Rate:** The Particle Filter improved position accuracy on **99 out of 100** test samples.
- **Roll Success Rate:** The Particle Filter improved roll accuracy on **98 out of 100** test samples.
- **Outlier Sample 63:** On sample 63, true roll was $+20.5^\circ$ but the ANN predicted $-22.5^\circ$ (inverted roll mode). The particle filter converged to a local minimum at $-12.2^\circ$, causing an increase in position error ($2.16\text{ m} \to 2.32\text{ m}$).

### Scientific Transparency: Evaluation Split Disclosure
An in-depth audit of the Phase 4 evaluation script (`src/particle_filter.py`) revealed that the 100 test samples were drawn randomly from the full 144,000-sample dataset rather than solely from the validation split. Consequently:
- **81 samples** were from the ANN training split.
- **19 samples** were from the ANN validation split.

*Interpretation:* The Particle Filter does not access ground truth during estimation, so its landmark refinement mechanism is completely valid. However, the baseline ANN coarse class accuracy and probabilities on those 81 samples reflect training performance rather than a clean held-out validation benchmark.

---

## 7. Interactive Demonstration GUI

The project features a redesigned, modern scientific desktop GUI (`src/demo_gui.py`) built with pure standard Python `tkinter` and `ttk`.

```powershell
python src/demo_gui.py
```

### Dashboard Features:
1. **Ground-Truth Input Card:** Parameter inputs for $X, Y, Z, \text{Roll}$ with physical bound enforcement, input validation, and a **"Generate Random Pose"** trigger.
2. **Camera Viewport Canvas:** Dark $1280 \times 960$ scaled camera canvas displaying projected 2D landmarks, boresight reticle, detected/occluded count, and feature labels.
3. **Three Isolated Pose Panels:**
   - **Ground Truth Reference:** Exact simulated pose (no misleading errors).
   - **ANN Coarse Estimate:** Predicted class ID, predicted class softmax probability, cell center pose, and 3D Euclidean errors.
   - **ANN + Particle Filter Refined:** Final continuous metric pose and refined 3D Euclidean errors.
4. **Error Reduction & Visual Metrics:** Side-by-side numerical comparison of position and roll errors, proportional visual progress bars, and an explanatory 3-step pipeline guide.
5. **Robust Outlier Handling:** Truthfully reports degradation if an outlier hypothesis fails to improve.

---

## 8. Installation & Setup Guide

### Prerequisites
- Windows 10/11, macOS, or Linux
- Python 3.11 or 3.12 installed
- Standard Tkinter support (included with standard Windows Python)

### Windows PowerShell Setup
```powershell
# 1. Navigate to the repository root
cd "c:\Users\thiji\OneDrive\Documents\Projects\ML miniproject\vision-pose-estimation"

# 2. Create an isolated virtual environment
python -m venv .venv

# 3. Activate the virtual environment
.venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 9. Execution Guide (Verified Commands)

Run all commands from the repository root inside the activated `.venv`:

### 1. Launch the Demonstration GUI (Recommended)
```powershell
python src/demo_gui.py
```

### 2. Run Particle Filter Evaluation
```powershell
# Quick 5-sample smoke test
python src/particle_filter.py --samples 5 --particles 1000 --iterations 5

# Full 100-sample evaluation experiment
python src/particle_filter.py --samples 100 --particles 2000 --iterations 8
```

### 3. Generate Analytical Evaluation Plots (Phase 5)
```powershell
python src/phase5_analysis.py
```

### 4. Compile the Academic Report PDF (Strictly 2 Pages)
```powershell
python docs/generate_report_pdf.py
```
*Compiles: `docs/project_report.pdf` (verified exactly 2 pages).*

### 5. Compile the Academic Presentation Slide Deck (9 Slides)
```powershell
python docs/generate_presentation.py
```
*Compiles: `docs/project_presentation.pptx` (widescreen 16:9 layout).*

### 6. Dataset Synthesis & ANN Training (Optional — Checkpoints Pre-Supplied)
```powershell
# Regenerate the 144,000-sample dataset (takes ~2 minutes)
python src/generate_data.py

# Retrain the ANN for 60 epochs (takes ~15 minutes on CPU)
python src/train_ann.py --epochs 60 --batch-size 256 --lr 0.001
```

---

## 10. Repository Structure & Artifact Catalog

```text
vision-pose-estimation/
├── .venv/                         # Local Python virtual environment
├── data/
│   ├── dataset_info.json          # Camera intrinsics, landmark geometry, Table I bins
│   └── pose_dataset.csv           # 144,000 generated synthetic samples (39.9 MB)
├── docs/
│   ├── CONTRIBUTIONS.md           # Individual team contribution breakdown
│   ├── generate_presentation.py   # PPTX slide deck generator script
│   ├── generate_report_pdf.py     # Two-page academic report PDF generator script
│   ├── project_presentation.pptx  # 9-slide academic presentation deck
│   └── project_report.pdf         # Official two-page PDF mini-project write-up
├── models/
│   ├── normalization.json         # Active z-score standardization parameters (42-D)
│   └── pose_ann.pt                # Active PyTorch checkpoint (best epoch 57)
├── results/
│   ├── ann_metrics.json           # ANN validation evaluation metrics
│   ├── ann_vs_particle_filter.csv # 100-sample paired evaluation dataset
│   ├── bin_error_histograms.png   # Bin error distribution per dimension
│   ├── feature_configurations.png # 2D landmark sweeps across X, Y, Z, Roll
│   ├── particle_filter_summary.json # Summary metrics of Phase 4 PF run
│   ├── phase5_*.png (5 plots)     # Analytical error distributions, boxplots, examples
│   ├── phase5_summary.json        # Comprehensive Phase 5 statistical results
│   ├── training_curves.png        # Loss and Top-1 / Top-5 training curves
│   └── training_history.csv       # Epoch-by-epoch loss and accuracy log
├── src/
│   ├── demo_gui.py                # Redesigned interactive scientific Tkinter dashboard
│   ├── generate_data.py           # Synthetic pinhole & self-occlusion dataset generator
│   ├── particle_filter.py         # Multi-hypothesis particle filter refinement engine
│   ├── phase5_analysis.py         # Statistical paired analysis script
│   └── train_ann.py               # PyTorch PoseANN training and evaluation
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Comprehensive project documentation
```

---

## 11. Known Technical Limitations & Research Paper Comparison

To ensure academic and scientific integrity, the following limitations must be recognized:

1. **Static Snapshot vs. Dynamic Flight Trajectory:**  
   In the reference paper, Rohan Punnoose evaluated sequential tracking across a 10-second flight trajectory with nonlinear dynamics and unobserved leader acceleration disturbances, where the particle filter struggled to converge. In this student project, the particle filter is formulated as an **iterative static-snapshot continuous optimizer** with annealed jitter across 8 iterations on independent static frames.
2. **Inverse-Simulation Setup:**  
   The particle filter measurement likelihood uses the identical forward projection geometry, pinhole camera parameters, and fuselage occluder used to synthesize the dataset. The sub-meter convergence demonstrates mathematical correctness under the generative model, but does not capture real-world aerostructure flexing, lens distortion, or varying atmospheric illumination.
3. **Upstream CNN Feature Detection:**  
   As assumed in the Punnoose paper, the extraction of the 14 visual landmarks from raw RGB imagery is treated as an upstream CNN capability outside the project scope.

---

## 12. Academic Team & Individual Contributions

**Course:** UE24CS352A — Machine Learning  
**Team Allocation:** Team **G-19** (Section G), Serial No. **9**  
**Team Members:**  
- **Krishna G Pai** (SRN: `PES1UG24AM372`, GitHub: `@krishna130506`)
- **Thijil Thejaswi M N** (SRN: `PES1UG24AM412`)

*Detailed individual responsibilities, code provenance, and collaborative breakdown are documented in [docs/CONTRIBUTIONS.md](docs/CONTRIBUTIONS.md).*

---

## 13. References

1. **Punnoose, Rohan.** *"Vision-Based Precision Pose Estimation for Autonomous Formation Flying"*, Stanford University CS229: Machine Learning Project Report, Spring 2020. [PDF Report](https://cs229.stanford.edu/proj2020spr/report/Punnoose.pdf).
2. **Sharma, Sumant, et al.** *"Pose Estimation for Non-Cooperative Spacecraft Rendezvous Using Convolutional Neural Networks"*, 2018 IEEE Aerospace Conference, Big Sky, MT, 2018.
3. **Thrun, Sebastian, Wolfram Burgard, and Dieter Fox.** *"Probabilistic Robotics"*, MIT Press, Cambridge, MA, 2005.
