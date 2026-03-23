# Deep LPPLS Model Specification
**Source:** Nielsen, Sornette & Raissi (2024), arXiv:2405.12803v1

---

## 1. LPPLS Formula

### 1.1 Full form (Eq. 1)

```
O(t) = A + B(t_c - t)^m + C(t_c - t)^m * cos(ω ln(t_c - t) - φ)
```

| Symbol | Type | Description |
|--------|------|-------------|
| `O(t)` | observable | e.g. log-price of asset |
| `t_c`  | nonlinear param | critical time (singularity point) |
| `m`    | nonlinear param | power-law exponent, `0 < m < 1` |
| `ω`    | nonlinear param | log-periodic angular frequency |
| `A`    | linear param | value of `O(t)` at `t = t_c` |
| `B`    | linear param | amplitude of power-law growth |
| `C`    | linear param | amplitude of log-periodic oscillation |
| `φ`    | linear param | phase of log-periodic oscillation |

### 1.2 Reformulated form (Eq. 4, 5)

To reduce nonlinear parameters: replace `C, φ` with `C1 = C·cos(φ)`, `C2 = C·sin(φ)`.

```
O(t) = A + B·f + C1·g + C2·h
```

where:
```
f = (t_c - t)^m
g = (t_c - t)^m · cos(ω ln(t_c - t))
h = (t_c - t)^m · sin(ω ln(t_c - t))
```

Nonlinear parameters: `{t_c, m, ω}` (3 params)
Linear parameters: `{A, B, C1, C2}` (4 params)

---

## 2. Time Normalisation

- The time series is mapped to the unit segment `[0, 1]`.
- `t2` (end of series) is normalised to `1`.
- All `t` values are min-max scaled accordingly before any computation.

---

## 3. Parameter Constraints (Financial Bubble Domain)

| Parameter | Lower bound | Upper bound |
|-----------|------------|------------|
| `t_c` | `t2 - 0.2·t2` | `t2 + 0.2·t2` |
| `m` | `0.1` | `1.0` |
| `ω` | `6` | `13` |

For P-LNN synthetic data generation:

| Parameter | Range |
|-----------|-------|
| `t_c` | `t2` to `t2 + 50` (days, before normalisation) |
| `m` | `0.1` to `0.9` |
| `ω` | `6` to `13` |

---

## 4. Standard LM Calibration (Baseline, Appendix A.1)

### Loss function (Eq. 6)
```
F(t_c, m, ω, A, B, C1, C2) = (1/n) Σ_{i=1}^{n} [O(τ_i) - A - B·f_i - C1·g_i - C2·h_i]²
```
where `f_i = f(τ_i)`, `g_i = g(τ_i)`, `h_i = h(τ_i)`.

### Linear parameter solution (Eq. 7, 8)
At fixed `{t_c, m, ω}`, solve analytically:
```
{Â, B̂, Ĉ1, Ĉ2} = argmin_{A,B,C1,C2} F(t_c, m, ω, A, B, C1, C2)
```

Solved via the matrix equation (Eq. 8):
```
| N      Σf_i    Σg_i    Σh_i   | | Â  |   | Σ ln p_i      |
| Σf_i   Σf_i²  Σf_i·g_i Σf_i·h_i | | B̂  | = | Σ f_i·ln p_i  |
| Σg_i   Σf_i·g_i Σg_i²  Σg_i·h_i | | Ĉ1 |   | Σ g_i·ln p_i  |
| Σh_i   Σf_i·h_i Σg_i·h_i Σh_i² | | Ĉ2 |   | Σ h_i·ln p_i  |
```

### Reduced nonlinear loss (Eq. 9, 10)
```
F1(t_c, m, ω) = min_{A,B,C1,C2} F(t_c, m, ω, A, B, C1, C2)
{t̂_c, m̂, ω̂} = argmin_{t_c, m, ω} F1(t_c, m, ω)   [solved via LM algorithm]
```

---

## 5. M-LNN (Mono-LPPLS-NN) Model

### 5.1 Purpose
Trained **per time series** — a new M-LNN is fit for each empirical time series. Estimates nonlinear parameters `{t_c, m, ω}`.

### 5.2 Architecture (Eq. 2)
Feed-forward network with 2 hidden layers:
```
h1 = ReLU(W1·X + b1)
h2 = ReLU(W2·h1 + b2)
Y  = W_o·h2 + b_o
```

| Layer | Input dim | Output dim | Activation |
|-------|-----------|------------|------------|
| Input | `n` (series length, variable) | — | — |
| Hidden 1 | `n` | `n` | ReLU |
| Hidden 2 | `n` | `n` | ReLU |
| Output | `n` | `3` | Linear |

> Note: Hidden layer width equals input series length `n`. The output has 3 nodes for `[t_c, m, ω]`.

### 5.3 Loss Functions

**MSE loss** (reconstruction):
```
L_MSE = MSE(X, LPPLS(Ŷ))
      = (1/n) Σ [X_i - LPPLS(Ŷ)_i]²
```
where `LPPLS(Ŷ)_i` is the reconstructed time series from estimated parameters using Eq. 4/5.

**Penalty loss** (parameter bounds):
```
L_Penalty = α · Σ_{k=1}^{3} [max(0, θ_{k,min} - θ_k) + max(0, θ_k - θ_{k,max})]
```
where `θ1 = t_c`, `θ2 = m`, `θ3 = ω`, and `α` is the penalty coefficient.

**Total loss**:
```
L_Total = L_MSE + L_Penalty
```

### 5.4 Training Procedure
- Optimizer: **Adam**
- Learning rate: **1e-2**
- Preprocessing: **min-max scaling** of input series to `[0, 1]`
- Save best model: checkpoint at epoch with **lowest total loss**
- Terminate at convergence (fixed epoch count or early stopping)
- One network trained per time series (no shared weights across series)

---

## 6. P-LNN (Poly-LPPLS-NN) Model

### 6.1 Purpose
Trained **once** on a large synthetic dataset; generalises to unseen time series at inference. Fixed input length of **252** time points.

### 6.2 Architecture (Eq. 3)
Feed-forward network with 4 hidden layers:
```
h1 = ReLU(W1·X + b1)
h2 = ReLU(W2·h1 + b2)
h3 = ReLU(W3·h2 + b3)
h4 = ReLU(W4·h3 + b4)
Y  = W5·h4 + b5
```

| Layer | Input dim | Output dim | Activation |
|-------|-----------|------------|------------|
| Input | `252` | — | — |
| Hidden 1 | `252` | `252` | ReLU |
| Hidden 2 | `252` | `252` | ReLU |
| Hidden 3 | `252` | `252` | ReLU |
| Hidden 4 | `252` | `252` | ReLU |
| Output | `252` | `3` | Linear |

Output `Y = [t_c, m, ω]`.

### 6.3 Loss Function

Direct parameter MSE (Eq. in §2.2.4):
```
L = MSE(Y_true, Y_pred) = (1/3) Σ (θ_k_true - θ_k_pred)²
```
where the sum is over `{t_c, m, ω}`. Compares **predicted parameters** to **ground-truth parameters**, NOT reconstructed time series.

### 6.4 Training Procedure
- Optimizer: **Adam**
- Learning rate: **1e-5**
- Epochs: **20**
- Batch size: **8**
- Preprocessing: **min-max scaling** (dataset undergoes min-max scaling)
- Dataset: **100,000** training samples, **33,333** validation samples
- Hardware reference: Tesla V100-SXM2 GPUs (16GB), ~1.5h per P-LNN variant

---

## 7. Synthetic Data Generation

### 7.1 Clean LPPLS series
Generate `S = {s_1, ..., s_n}` using Eq. 4/5 with known parameters `{t_c, m, ω, A, B, C1, C2}`.
The function range is always rescaled to `[0, 1]`.

### 7.2 White noise augmentation
```
s'_i = s_i + η_i,   η_i ~ N(0, α²)
```
White noise amplitude α: **0.01 to 0.15** (as fraction of function range).

### 7.3 AR(1) noise augmentation
```
η_t = φ_ar · η_{t-1} + ε_t,   ε_t ~ N(0, σ²),   t = 1, ..., n
```
AR(1) variance: `σ²_η = σ² / (1 - φ_ar²)`
AR coefficient: `φ_ar = 0.9`
AR(1) amplitude: **0.01 to 0.05**.

### 7.4 P-LNN model variants

| Model | Noise |
|-------|-------|
| P-LNN-100K | White only |
| P-LNN-100K-AR1 | AR(1) only |
| P-LNN-100K-BOTH | White + AR(1), ~50/50 split |

---

## 8. Inference / Prediction

Given a fitted model and estimated `Ŷ = [t̂_c, m̂, ω̂]`:
1. Recover `{Â, B̂, Ĉ1, Ĉ2}` analytically using Eq. 8.
2. Reconstruct full LPPLS fit via Eq. 4/5.
3. Report `t̂_c` as the predicted critical time.

---

## 9. Open/Ambiguous Spec Items

The following details are **not fully specified** in the paper and require implementation choices:

| # | Item | Notes |
|---|------|-------|
| 1 | M-LNN hidden layer width | Paper says layers have width `n` (input length) — confirmed by architecture equation |
| 2 | M-LNN epoch count | Not stated; best model checkpoint implies training until convergence |
| 3 | Penalty coefficient α | Not given numerically; tunable hyperparameter |
| 4 | P-LNN hidden layer width | Paper says 252 nodes matching input size — confirmed |
| 5 | Output activation for `t_c`, `m`, `ω` | Linear output layer (no sigmoid/tanh clipping) — bounds enforced by penalty (M-LNN) or implicit in label distribution (P-LNN) |
| 6 | Linear parameter recovery at M-LNN inference | Paper implies Eq. 8 is used post-network; needs explicit implementation |
| 7 | Normalisation of `t` axis | Time is normalised to `[0,1]`; t2=1; requires mapping τ_i → [0,1] before feeding to LPPLS equations |
