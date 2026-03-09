# C-TeC

An independent re-implementation of **C-TeC: Curiosity-Driven Exploration via Temporal Contrastive Learning** ([paper](https://openreview.net/forum?id=gqjT7g5ZRa)), built entirely from scratch by reading the paper alone — no reference to the authors' original codebase was used at any point.

## Results

**All results, comparisons (C-TeC vs RND vs Random), and analysis are in the demo notebook:**



 [`demo.ipynb`](demo.ipynb) *(rendered inline — no setup needed)*


---

## Installation

### Option 1: Pixi (Recommended)

[Pixi](https://pixi.sh) is the recommended way to set up the environment. It automatically handles Python, CUDA, and all dependencies declared in [`pyproject.toml`](pyproject.toml).

**1. Install Pixi** (if not already installed):

```bash
# Linux / macOS
curl -fsSL https://pixi.sh/install.sh | bash

# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

**2. Clone the repository:**

```bash
git clone https://github.com/LSNoor/C-TeC.git
cd C-TeC
```

**3. Install the environment:**

```bash
pixi install
```

This will create a fully reproducible environment (pinned in [`pixi.lock`](pixi.lock)) with:
- Python 3.12
- PyTorch with CUDA 12.8 support
- All required packages (`minigrid`, `matplotlib`, `seaborn`, `pyyaml`, `tqdm`, `pydantic-settings`, `ipykernel`)

**4. Activate the environment:**

```bash
pixi shell
```

Or run a command directly without activating:

```bash
pixi run python c_tec/main.py --config configs/c-tec_config.yaml
```

---

### Option 2: Conda

A [`conda-environment.yaml`](conda-environment.yaml) file is provided (exported from the pixi workspace) for users who prefer Conda/Mamba.

> **Note:** PyTorch is installed via pip with the CUDA 12.8 index URL, so make sure your system has a compatible NVIDIA driver.

**1. Clone the repository:**

```bash
git clone https://github.com/LSNoor/C-TeC.git
cd C-TeC
```

**2. Create the environment from the provided file:**

```bash
conda env create -f conda-environment.yaml
```

**3. Activate the environment:**

```bash
conda activate default
```

That's it, all dependencies (including PyTorch with CUDA 12.8 and the project itself in editable mode) are installed automatically by the environment file.

> **Tip:** To regenerate `conda-environment.yaml` from the pixi workspace at any time, run:
> ```bash
> pixi workspace export conda-environment conda-environment.yaml
> ```

---

## Usage

### Training

To train the C-TeC method:

```bash
python c_tec/main.py --config configs/c-tec_config.yaml
```

To train the RND baseline:

```bash
python c_tec/main.py --config configs/rnd_config.yaml
```

### Evaluation

To evaluate a trained C-TeC model:

```bash
python c_tec/main.py --config configs/c-tec_config.yaml --mode evaluation --checkpoint checkpoints/checkpoint_final.pt
```

To evaluate a trained RND model:

```bash
python c_tec/main.py --config configs/rnd_config.yaml --mode evaluation --checkpoint checkpoints/checkpoint_final.pt
```

Evaluation results will be saved to `results/{method}/eval/`, including:
- `eval_metrics.json` - Episode-by-episode coverage and statistics
- `trajectory_buffer.pkl` - Collected trajectories for visualization

### Demo Notebook

Open [`demo.ipynb`](demo.ipynb) in Jupyter (the `ipykernel` package is included in the environment):

```bash
jupyter notebook demo.ipynb
```

---

## Requirements Summary

| Requirement | Version |
|---|---|
| Python | 3.12 |
| CUDA | 12.8 |
| PyTorch | ≥ 2.10 |
| MiniGrid | ≥ 3.0 |

---

## Citation

This repository is an independent reimplementation of the following paper:

```bibtex
@inproceedings{
  mohamed2025curiositydriven,
  title={Curiosity-Driven Exploration via Temporal Contrastive Learning},
  author={Faisal Mohamed and Catherine Ji and Benjamin Eysenbach and Glen Berseth},
  booktitle={Workshop on Reinforcement Learning Beyond Rewards @ Reinforcement Learning Conference 2025},
  year={2025},
  url={https://openreview.net/forum?id=gqjT7g5ZRa}
}
```
