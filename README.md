# Vision-Based Pose Estimation Using ANN and Particle Filtering

## Overview

This mini-project implements a vision-based aircraft pose estimation pipeline inspired by the reference work **"Vision-Based Precision Pose Estimation for Autonomous Formation Flying."**

The system uses 14 structural aircraft features observed in a camera image. Each feature is represented by:

- image x-coordinate
- image y-coordinate
- visibility/occlusion indicator

This gives **42 input values**.

The project has two estimation stages:

1. **Artificial Neural Network (ANN)** for fast coarse pose classification.
2. **Particle Filter (PF)** for continuous model-based pose refinement.

The final system estimates:

- X position
- Y position
- Z position
- Roll angle

---

## Project Pipeline

```text
Synthetic aircraft pose
        |
        v
Camera projection + visibility
        |
        v
42 visual features
        |
        v
       ANN
        |
        v
4800 discrete pose classes
        |
        v
Coarse pose estimate
        |
        v
Particle Filter
        |
        v
Continuous refined pose
```

---

## Dataset

The project uses a physically based synthetic dataset.

### Input representation

There are 14 structural aircraft features.

Each feature contributes:

```text
(x_pixel, y_pixel, occlusion)
```

Therefore:

```text
14 × 3 = 42 input features
```

### Pose space

The pose space contains:

| Dimension | Range | Number of bins |
|---|---:|---:|
| X | -100 to 50 m | 10 |
| Y | -50 to 50 m | 10 |
| Z | -20 to 20 m | 8 |
| Roll | -45 to 45° | 6 |

Total classes:

```text
10 × 10 × 8 × 6 = 4800
```

The generated dataset contains **144,000 samples**, with 30 samples per pose class.

The aircraft geometry and camera model are simplified for this student reproduction because the exact geometry and feature coordinates of the reference implementation are not publicly specified in the paper.

---

## Neural Network

The ANN architecture follows the main architecture used in the reference work:

```text
Input: 42
    |
    v
Hidden layer: 100 neurons
    |
    v
Hidden layer: 100 neurons
    |
    v
Output: 4800 classes
```

Training uses:

- ReLU activation
- Adam optimizer
- Cross-entropy loss
- 60 training epochs
- train/validation split

---

## ANN Results

Best validation checkpoint:

```text
Validation loss          : 1.4357
Top-1 accuracy            : 62.85%
Top-5 accuracy            : 93.02%
Within ±1 bin, all dims   : 99.03%
Within ±2 bins, all dims  : 99.76%
```

Per-dimension exact-bin accuracy:

```text
X      : 91.6%
Y      : 87.9%
Z      : 82.5%
Roll   : 92.0%
```

Physical pose error:

```text
Position median error : 5.5 m
Position mean error   : 5.8 m
Position 90th pct.    : 8.8 m

Roll median error     : 3.8°
Roll mean error       : 4.2°
```

---

## Particle Filter Refinement

The particle filter takes the ANN's top pose hypotheses and performs continuous pose refinement.

For each particle:

1. Generate the expected feature projection.
2. Compare it with the observed feature coordinates.
3. Account for feature visibility.
4. Calculate the particle likelihood.
5. Normalize particle weights.
6. Resample.
7. Apply decreasing pose jitter.
8. Repeat for multiple iterations.

The PF therefore converts the ANN's discrete estimate into a continuous pose estimate.

---

## ANN vs Particle Filter

The Phase 4 validation used:

```text
Validation samples : 100
Particles/sample   : 2000
PF iterations      : 8
ANN hypotheses     : Top-5
```

Results:

| Metric | ANN | ANN + PF |
|---|---:|---:|
| Mean position error | 5.017 m | **0.179 m** |
| Median position error | 5.137 m | **0.120 m** |
| Position P90 | 7.285 m | **0.332 m** |
| Mean roll error | 4.108° | **0.414°** |
| Median roll error | 3.692° | **0.069°** |
| Roll P90 | 6.360° | **0.173°** |

The PF had lower position error on **99/100** validation samples and lower roll error on **98/100** samples.

The mean position error was reduced by **96.43%**, while mean roll error was reduced by **89.92%**.

### Important interpretation

The particle filter uses the same synthetic aircraft geometry, camera projection, and visibility model used to generate the synthetic observations. Therefore, these results demonstrate **model-based refinement under the project's synthetic observation model**. They should not be interpreted as an independent real-world benchmark.

---

## GUI Demonstration

The project includes a simple Tkinter GUI.

Run:

```powershell
python src/demo_gui.py
```

The GUI allows the user to:

1. Enter or generate a test pose.
2. Generate a noisy 14-feature camera observation.
3. Run the ANN.
4. Run particle-filter refinement.
5. Compare ground truth, ANN and refined estimates.
6. View position and roll errors.

Example demonstration:

```text
Ground truth position error:
ANN                  : 9.860 m
ANN + Particle Filter: 0.081 m

Ground truth roll error:
ANN                  : 2.410°
ANN + Particle Filter: 0.069°
```

---

## Repository Structure

```text
vision_pose_estimation/
|
├── data/
│   └── pose_dataset.csv
|
├── models/
│   ├── pose_ann.pt
│   └── normalization.json
|
├── results/
│   ├── training_curves.png
│   ├── bin_error_histograms.png
│   ├── ann_vs_particle_filter.csv
│   ├── particle_filter_summary.json
│   ├── phase5_summary.json
│   ├── phase5_error_distributions.png
│   ├── phase5_position_error_comparison.png
│   ├── phase5_roll_error_comparison.png
│   ├── phase5_samplewise_comparison.png
│   └── phase5_pose_examples.png
|
├── src/
│   ├── generate_data.py
│   ├── train_ann.py
│   ├── particle_filter.py
│   ├── phase5_analysis.py
│   └── demo_gui.py
|
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

Recommended Python version:

```text
Python 3.11
```

Create a virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Tkinter is normally included with the standard Windows Python installation and is not installed through `pip`.

---

## Running the Project

### 1. Generate the dataset

```powershell
python src/generate_data.py
```

### 2. Train the ANN

```powershell
python src/train_ann.py
```

This saves:

```text
models/pose_ann.pt
models/normalization.json
```

### 3. Run particle-filter validation

Small test:

```powershell
python src/particle_filter.py --samples 5 --particles 1000 --iterations 5
```

Main validation:

```powershell
python src/particle_filter.py --samples 100 --particles 2000 --iterations 8
```

### 4. Run Phase 5 analysis

```powershell
python src/phase5_analysis.py
```

### 5. Launch the GUI

```powershell
python src/demo_gui.py
```

---

## Reproducibility

The project uses fixed/random seeds in the main experiments where applicable.

The Phase 4 validation configuration is:

```text
Samples      : 100
Particles    : 2000
Iterations   : 8
ANN top-k    : 5
```

The reported results correspond to the generated CSV and analysis files in the `results/` directory.

---

## Limitations

1. The aircraft geometry is a simplified student implementation rather than the exact geometry used by the reference paper.
2. The dataset is synthetic rather than a real camera dataset.
3. The particle filter uses the same physical model used to generate the synthetic observations.
4. Therefore, the very low PF errors should not be interpreted as real-world aircraft pose accuracy.
5. The ANN's 4800 output classes introduce pose quantization because each class represents a discretized pose region.

---

## Reference

Punnoose, S. A., **Vision-Based Precision Pose Estimation for Autonomous Formation Flying**, Stanford CS229 Project Report.

The implementation in this repository is a student reproduction/extension of the core pose-estimation methodology, with a simplified synthetic data-generation model and an additional particle-filter refinement stage.

---

## Academic Project

Course:

**UE24CS352A — Machine Learning**

Mini-Project

The repository contains the implementation, experiments, evaluation and GUI demonstration used for the project review.
