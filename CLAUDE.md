# CLAUDE.md — lppls_py

## Project

Python implementation of the **Deep LPPLS** model from:

> Nielsen, J., Sornette, D., & Raissi, M. (2024). *Deep LPPLS: Forecasting of temporal critical points in natural, engineering and financial systems.* arXiv:2405.12803v1.

The project implements two neural network architectures for calibrating the Log-Periodic Power Law Singularity (LPPLS) model:

- **M-LNN (Mono-LPPLS-NN)** — feed-forward NN trained per time series to estimate the nonlinear LPPLS parameters (t_c, m, ω).
- **P-LNN (Poly-LPPLS-NN)** — supervised NN trained on a large corpus of synthetic LPPLS time series; can generalize to unseen series at inference time.

Core LPPLS formula: `O(t) = A + B(t_c - t)^m + C(t_c - t)^m * cos(ω ln(t_c - t) - φ)`

## Tech Stack

- **Language:** Python
- **ML framework:** Keras (use `keras` directly; compatible with TensorFlow backend)
- **GPU training:** All models must be GPU-ready (see GPU section below)

## GPU Training Requirements

- Enable mixed precision where appropriate: `keras.mixed_precision.set_global_policy("mixed_float16")`
- Use `tf.distribute.MirroredStrategy` for multi-GPU support
- Place data pipeline ops on CPU, compute on GPU — use `tf.data` with `prefetch(tf.data.AUTOTUNE)`
- Do not hardcode device strings; use strategy scopes instead

## Workflow Rules

### Implementing from a Research Paper
1. Read the relevant paper sections once to understand the architecture.
2. Produce a **concrete implementation plan** (architecture, data pipeline, loss, training loop).
3. **Immediately start writing code** — do not spend extended time on web searches or re-reading the paper once the plan is established.

### MCP Tools
- Before falling back to built-in tools (Glob, Grep, Bash), check available MCP tools first via `ToolSearch`.

### Git Operations
- Prefer `gh` CLI via Bash for all GitHub/git operations.
- Verify PATH includes `/usr/local/bin` before running `gh`.
