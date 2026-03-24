# Deep LPPLS — Neural Network Calibration of Financial Bubbles

Python implementation of the **Deep LPPLS** model from:

> Nielsen, J., Sornette, D., & Raissi, M. (2024). *Deep LPPLS: Forecasting of temporal critical points in natural, engineering and financial systems.* [arXiv:2405.12803v1](https://arxiv.org/abs/2405.12803)

---

## Description / Overview

The **Log-Periodic Power Law Singularity (LPPLS)** model describes super-exponential price bubbles and predicts the critical time `tc` at which the bubble is most likely to burst. Traditional calibration relies on slow nonlinear optimisation; this project replaces it with two neural architectures that predict `tc`, `m`, and `ω` directly from a normalised 252-point price window:

| Model | Type | Training strategy |
|---|---|---|
| **M-LNN** (Mono-LPPLS-NN) | 2-layer feed-forward | Trained per series (unsupervised reconstruction loss) |
| **P-LNN** (Poly-LPPLS-NN) | 4-layer feed-forward | Pre-trained on 100 K synthetic series; generalises at inference time |

After calibration, the library builds a **probability density of `tc`** via kernel density estimation over all sliding-window predictions, and renders the dual-axis bubble analysis plot shown in Figures 4–6 of the paper.

---

## Quickstart

```python
import keras
import pandas as pd
from lppls import calibrate, compute_tc_pdf, plot_calibration, save_figure

# Load a trained P-LNN
model = keras.models.load_model("models/plnn_white.keras", compile=False)
model_fn = lambda x: model(x[None], training=False)[0].numpy()

# Load a price series (DatetimeIndex, positive raw prices)
prices = pd.read_csv("nasdaq.csv", index_col=0, parse_dates=True)["Close"]

# Sliding-window calibration (step=5 ≈ weekly resolution)
df = calibrate(prices, model_fn, step=5)

# Reproduce paper Figures 4–6
fig = plot_calibration(
    prices,
    {"P-LNN": df},
    tc_actual=pd.Timestamp("2000-03-10"),   # optional: known crash date
    tc_trough=pd.Timestamp("2002-10-09"),   # optional: known trough
)
save_figure(fig, "bubble_analysis.png")
```

---

## Installation

**Prerequisites:** Python ≥ 3.10, a CUDA-capable GPU (optional but recommended).

```bash
# Clone the repository
git clone https://github.com/your-user/lppls_py.git
cd lppls_py

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt` pins:

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
pandas>=2.0
tensorflow>=2.13
keras>=2.13
pytest>=7.4
```

> **GPU note:** TensorFlow will automatically use any available CUDA GPU.
> Mixed precision (`float16`) is enabled automatically during P-LNN training when a GPU is detected.

---

## Usage

### Train a P-LNN model

```bash
# White noise variant (paper default)
python -m lppls.train_plnn --noise white --output models/plnn_white.keras

# AR(1) noise variant
python -m lppls.train_plnn --noise ar1 --output models/plnn_ar1.keras

# Mixed noise (50 % white + 50 % AR(1))
python -m lppls.train_plnn --noise both --output models/plnn_both.keras
```

All CLI flags:

```
--noise        {white,ar1,both}   Noise variant (default: white)
--output       PATH               Output .keras path
--epochs       INT                Training epochs (default: 20)
--batch-size   INT                Batch size (default: 8)
--lr           FLOAT              Adam learning rate (default: 1e-5)
--n-train      INT                Training samples (default: 100 000)
--n-val        INT                Validation samples (default: 33 333)
--seed         INT                RNG seed (default: 42)
--no-mixed-precision              Disable float16 mixed precision
```

### Run calibration from Python

```python
from lppls import calibrate, calibrate_multi

# Single model
df = calibrate(prices, model_fn, step=5)

# Multiple models simultaneously
results = calibrate_multi(prices, {"LM": lm_fn, "M-LNN": mlnn_fn, "P-LNN": plnn_fn}, step=5)
```

Each row of the returned DataFrame contains:

| Column | Description |
|---|---|
| `window_start / window_end` | Calendar start/end of the 252-point window |
| `tc_norm / tc_calendar` | Predicted critical time (normalised and calendar) |
| `m`, `omega` | Predicted power-law exponent and log-frequency |
| `A`, `B`, `C1`, `C2` | Recovered linear LPPLS parameters |
| `series_norm` | Min-max normalised window array |
| `price_min / price_max` | Raw price range for scale inversion |
| `log_prices` | Whether log was applied before normalisation |

### Compute and plot the tc PDF

```python
from lppls import compute_tc_pdf, plot_calibration, save_figure
import pandas as pd

# Date grid over which to evaluate the density
grid = pd.date_range("1999-01-01", "2002-01-01", periods=500)
density = compute_tc_pdf(df, grid, bandwidth="scott")

# Full dual-axis figure (price + fits left, PDF fills right)
fig = plot_calibration(
    prices,
    {"M-LNN": mlnn_df, "P-LNN": plnn_df},
    show_fit_curves="median",   # "median" | "all" | "none"
    kde_bandwidth="scott",
    figsize=(14.0, 6.0),
)
save_figure(fig, "output/bubble.pdf", dpi=150)
```

### Run the test suite

```bash
pytest tests/ -v
```

80 tests total (44 inference/plotting + 36 core model tests).

---

## Features

- **M-LNN** — per-series unsupervised training; differentiable LPPLS reconstruction loss + parameter bound penalty (Eq. 2–3 of the paper)
- **P-LNN** — supervised pre-training on 100 K synthetic series; fast inference on any new series without retraining (Eq. 3 of the paper)
- **Synthetic dataset generation** — white Gaussian noise and AR(1) noise augmentation; parameter sampling from spec Table 1
- **Sliding-window calibration** — model-agnostic `calibrate()` accepts any `(252,) → (3,)` callable (P-LNN, M-LNN, LM wrapper, etc.)
- **KDE-based tc PDF** — `compute_tc_pdf()` via `scipy.stats.gaussian_kde` over calendar-ordinal tc predictions; numerically stable
- **Dual-axis bubble plot** — reproduces paper Figures 4–6: price line + LPPLS fit curves (left axis), filled PDF per model (right axis), calibration window shading, optional drawdown shading, vertical markers for `t1 / t2 / tc_actual / tc_trough`
- **GPU-ready training** — `MirroredStrategy` multi-GPU support, `mixed_float16` precision, `tf.data` pipeline with `prefetch(AUTOTUNE)` and pre-generated RAM arrays (eliminates Python-generator bottleneck)
- **LR scheduling** — `ReduceLROnPlateau(factor=0.5, patience=3)` prevents early plateau
- **Per-epoch diagnostics** — wall-clock throughput (samples/sec) and `nvidia-smi` GPU utilisation logging

---

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.10+ |
| ML framework | Keras (TensorFlow backend) |
| Numerics | NumPy, SciPy (`lfilter`, `gaussian_kde`) |
| Data | Pandas (`DatetimeIndex`, `DataFrame`) |
| Visualisation | Matplotlib (Agg-compatible) |
| Training infra | `tf.distribute.MirroredStrategy`, `tf.data`, `mixed_float16` |
| Testing | pytest |

---

## Project Structure

```
lppls_py/
├── lppls/
│   ├── __init__.py        # Public API re-exports (calibrate, plot_calibration, …)
│   ├── formula.py         # LPPLS formula (Eq. 1, 4/5, 8): lppls(), lppls_reformulated(),
│   │                      #   recover_linear_params()
│   ├── data_gen.py        # Synthetic series generation: clean LPPLS + white/AR(1) noise
│   ├── dataset.py         # P-LNN dataset pipeline: generate_plnn_sample(),
│   │                      #   pregenerate_arrays(), make_tf_dataset(); defines SERIES_LEN,
│   │                      #   _T_AXIS, _TC_MIN/_TC_MAX, NoiseType
│   ├── mlnn.py            # M-LNN architecture + loss + single-series training loop
│   ├── plnn.py            # P-LNN architecture (4 hidden layers) + MSE loss
│   ├── train_plnn.py      # Full P-LNN training pipeline + CLI entry-point
│   ├── inference.py       # Sliding-window calibration (calibrate / calibrate_multi)
│   │                      #   and KDE-based tc PDF (compute_tc_pdf)
│   └── plotting.py        # Dual-axis bubble analysis figure (plot_calibration,
│                          #   save_figure); reproduces paper Figs 4–6
├── tests/
│   ├── test_lppls.py      # Formula, data generation, M-LNN and P-LNN unit tests
│   ├── test_train_plnn.py # Dataset pipeline and training integration tests
│   ├── test_inference.py  # calibrate / calibrate_multi / compute_tc_pdf tests
│   └── test_plotting.py   # Figure structure and save_figure tests (Agg backend)
├── models/                # Saved .keras model files (git-ignored)
├── requirements.txt       # Python dependencies
└── CLAUDE.md              # Project and workflow instructions for Claude Code
```

---

## License

This project is released under the **MIT License**. You are free to use, modify, and distribute the code with attribution.

---

## Credits / Acknowledgments

- **Nielsen, J., Sornette, D., & Raissi, M.** — original *Deep LPPLS* paper ([arXiv:2405.12803v1](https://arxiv.org/abs/2405.12803))
- **Didier Sornette** — foundational LPPLS theory and the JLS model
- TensorFlow / Keras team for the distributed training infrastructure
