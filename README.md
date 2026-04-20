# HDV Behavior Modeling on highD

Probabilistic short-horizon prediction of human driver behavior on the highD dataset using Dynamic Bayesian Networks (DBNs).

## Overview

This repository contains the code developed for a Master's thesis on human driver behavior modeling from highway trajectory data. The project focuses on learning probabilistic temporal patterns from the highD dataset and using them for behavior prediction, evaluation, and visualization.

The repository includes:
- preprocessing and exploratory analysis of the highD dataset
- DBN-based behavior modeling with latent style and action states
- training and evaluation pipelines
- trajectory simulation and visualization tools

The core implementation is located in `hdv/hdv_dbn`.

## Main Features

- preprocessing pipeline for highD trajectory data
- window-based sequence construction for temporal modeling
- DBN training with different emission-model settings
- evaluation scripts for generative and predictive analysis
- saved experiment artifacts and metrics
- interactive single-vehicle and multi-vehicle simulation tools

## Repository Structure

```text
.
├── hdv/
│   ├── data/
│   │   └── highd/              # Expected location of raw highD CSV files
│   ├── data_analysis/          # Dataset analysis and reporting scripts
│   ├── hdv_dbn/                # DBN model, training, inference, evaluation
│   └── models/                 # Saved experiments and trained model artifacts
├── simulation/                 # Simulation and visualization scripts
└── report/                     # Thesis LaTeX sources and related material
```

## Environment Setup

Create and activate a virtual environment, then install dependencies.

### Windows PowerShell

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

Expected dataset location:

- `hdv/data/highd/`

This folder should contain files such as:
- `01_tracks.csv`
- `01_tracksMeta.csv`
- `01_recordingMeta.csv`

The highD dataset is not included in this repository and must be obtained separately.

The code also supports cache files generated during preprocessing and training, such as Feather files and window caches.

## Quick Start Workflows

Run all commands from the repository root.

### 1. Dataset analysis report

```powershell
python hdv/data_analysis/report.py
```

This generates plots and tables for dataset analysis, including kinematics, lane-change statistics, lane-position features, and feasibility checks.

### 2. Train the DBN model

```powershell
python -m hdv.hdv_dbn.train_highd_dbn
```

Training will:
- load and preprocess highD trajectories
- build window-based sequences
- split the data into train, validation, and test sets at vehicle level
- fit scalers and train the model with EM
- save checkpoints and final model artifacts under `hdv/models/`

### 3. Evaluate a trained model

```powershell
python -m hdv.hdv_dbn.evaluate_highd_dbn
```

Evaluation writes metric artifacts into the experiment folder, for example:
- `eval_metrics.json`
- plots
- confusion matrices
- heatmaps

### 4. Run simulation and visualization

```powershell
python simulation/main.py
```

Edit the user options in `simulation/main.py` to choose:
- single-vehicle or multi-vehicle simulation
- recording and vehicle IDs
- selection filters such as lane change, acceleration, braking, or following

## Configuration

Main modeling and training settings are defined in:

- `hdv/hdv_dbn/config.py`

Important values to adjust include:
- latent state sizes (`DBN_STATES`)
- training iterations and stopping criteria (`TrainingConfig`)
- emission model mode (`poe` or `hierarchical`)
- scaling strategy (`use_classwise_scaling`)
- device (`cpu` or `cuda`)
- maximum number of recordings used (`max_highd_recordings`)

## Outputs

Common generated outputs include:
- model checkpoints and final models in `hdv/models/<experiment>/`
- split metadata in `split.json` inside the experiment directory
- evaluation summaries in `eval_metrics.json` inside the experiment directory
- analysis figures and reports created by scripts in `hdv/data_analysis/`

## Research Context

This repository was developed as part of a Master's thesis on probabilistic short-horizon prediction of human driver maneuvers from highway trajectory data.