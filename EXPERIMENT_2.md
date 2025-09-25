# EXPERIMENT_2.md — Baselines & Evaluation to Demonstrate DISCO Superiority

This document tells a coding agent (e.g., Claude Code) exactly how to **implement, run, and evaluate** experiments that demonstrate **DISCO** is better than reasonable baselines. It is fully aligned with the repository structure and APIs defined in `README.md`, `DISCO.md`, and `EXPERIMENT.md`. The final workflow is designed to be executed from a **Jupyter notebook**.

> **Scope:** tabular data first (Adult Income, Bike Sharing, COMPAS, synthetic), extensible later.  
> **Modes:** primarily **P‑mode**; **S‑mode** supported on a subset for validation.  
> **Outcomes:** publication‑ready tables/figures + CSV summaries with statistical tests.

---

## 0) Repository alignment (recap)

Keep the repo tree consistent with `README.md`:

```
disco/
  data/               # loaders, splits, stack_prewindow()
  devices/            # Device adapters
  interventions/      # base + tabular + grids
  anchors/            # embed + knn
  matching/           # simplex + weights + metrics
  counterfactual/     # smode + pmode
  explain/            # te_curve + plotting + baselines (to implement)
  cli/                # main.py subcommands: train-devices, probe, explain, evaluate, synth
experiments/
  configs/            # YAMLs per study
  results/            # run_id folders
  figs/               # figures
  notebooks/          # Jupyter entry
```

**Command entry:** `python -m disco.cli.main <subcommand> [args]`.

---

## 1) What to implement (delta to earlier files)

### 1.1 New baselines (peer-based + local-only)

Create `disco/explain/baselines.py` with the following API:

```python
# disco/explain/baselines.py
from dataclasses import dataclass
import numpy as np
from typing import Dict, Any, List

@dataclass
class BaselineOutput:
    theta: np.ndarray
    y_t: np.ndarray
    y_syn: np.ndarray
    tau: np.ndarray
    auc_abs: float
    flip_theta: float | None
    meta: Dict[str, Any]

def baseline_avg(devices, x_star, grids, intervention, target_idx: int) -> BaselineOutput: ...
def baseline_topk_avg(devices, traj, theta0: np.ndarray, Kp: int,
                      x_star, grids, intervention, target_idx: int) -> BaselineOutput: ...
def baseline_glob_matching(traj, devices, x_star, grids, intervention,
                           target_idx: int, lam: float) -> BaselineOutput: ...
def baseline_unconstrained_ridge(y_t: np.ndarray, X_peers: np.ndarray, lam: float) -> np.ndarray: ...
def baseline_local_perturbation(device_t, x_star, grids, intervention) -> BaselineOutput: ...
# Optional (if shap installed): def baseline_kernelshap(...): ...
```

**Notes:**
- `baseline_avg`: `y_syn[θ] = mean_{j != t} g(f_j(T_d(θ, x*)))`. Requires device access at `x*`. If running strictly in P‑mode, use the same **anchor‑transfer** proxy as DISCO for fairness. Expose a flag in CLI: `--baseline-mode p|s`.
- `baseline_topk_avg`: select top‑`K'` peers by **pre‑window distance** from target using shared trajectories `traj[(device_id, d.name)]`. Distance = average L2 over `theta0` on **public probes**.
- `baseline_glob_matching`: same optimization as DISCO’s weight solver **but** use **all probes** (no anchor selection) to form `y_t` and `X_peers`.
- `baseline_unconstrained_ridge`: solve `argmin_w ||Xw - y||^2 + lam||w||^2` without simplex. Return `w` (may contain negatives); downstream, compute `y_syn = X_post w` analogous to DISCO’s post stage. When reporting, include a “clipped to nonnegative then renormalized” variant as sensitivity (store both in `meta`).
- `baseline_local_perturbation`: `y_syn` is **zeros** (or identical target baseline), thus `tau = y_t` — this is the “no peer” curve.

Provide helpers in this module to compute **AUC‑TE** (trapezoid on `|tau|`) and **flip_theta** if task is classification.

### 1.2 Evaluation utilities

Create `experiments/evaluate.py` to aggregate metrics across runs and baselines:

```python
def collect_run_metrics(run_dir: str) -> pd.DataFrame: ...
def compute_cv_across_devices(df: pd.DataFrame) -> pd.DataFrame: ...
def kendall_tau_feature_ranking(df: pd.DataFrame) -> pd.DataFrame: ...
def bound_coverage_and_tightness(df: pd.DataFrame) -> pd.DataFrame: ...
def stats_tests_pairwise(df: pd.DataFrame, metric: str, methods: list[str]) -> pd.DataFrame: ...
def save_tables_and_plots(df: pd.DataFrame, out_dir: str) -> None: ...
```

### 1.3 CLI additions

Extend `disco/cli/main.py`:
- `explain` subcommand: add `--baselines` list (e.g., `avg,topk,glob,lp,ur`), `--baseline-mode p|s`, and `--Kp` for TOP‑K. Save each baseline’s outputs under `experiments/results/<run_id>/baselines/<name>/...` mirroring DISCO layout.
- `evaluate` subcommand: point to a run directory; produce `summary.csv/.parquet`, four **tables** (see §6), and all figures into `experiments/figs/<run_id>/`.

---

## 2) Experimental design — what the code must produce

We compare **DISCO** against baselines on four quantitative axes:

1. **Pre‑window matching**: `ε_pre` (lower better).  
2. **Post‑window reliability**: bound coverage↑, tightness↓, P‑vs‑S gap↓ (subset).  
3. **Cross‑device comparability**: AUC‑TE coefficient of variation (CV)↓; **unit invariance** test (CV delta small).  
4. **Stability**: weight L1 distance under anchor bootstrap↓; TE variance↓.

All metrics are computed **per (dataset, target, x*, intervention d)** and then aggregated.

---

## 3) Datasets, interventions, and grids

- **Datasets**: Adult (classification), Bike (regression), COMPAS (classification), and Synthetic (ground truth). Implement loaders in `disco/data/loaders.py` (see `EXPERIMENT.md`).
- **Splits**: `split_across_devices(..., strategy="label-skew"| "covariate-shift"| "feature-shift")`.
- **Interventions**: `FeatureAdd` and `FeatureScale` with clamping where necessary.
- **Grids**:  
  - Pre‑window `theta0 = [0, ε, 2ε]` where `ε` is 5–10% of robust scale.  
  - Post‑grid `theta` from `γ = 3ε` up to `θ_max` with 6–12 points.  
  - Ensure `min(theta) > max(theta0)`.

YAML examples are already provided in `EXPERIMENT.md` §11.

---

## 4) Metrics — definitions & implementation details

> Implement all metric functions in `disco/explain/metrics.py` (create if missing). Save per‑sample metrics in JSON next to arrays; aggregate in `experiments/evaluate.py`.

### 4.1 Primary effectiveness
- **AUC‑TE**: trapezoidal rule on `|tau|` over `theta`.  
- **Flip intensity** (classification): first `theta` s.t. predicted class changes (use target raw model; scalarization must not hide argmax).

### 4.2 Pre‑window matching
- **ε_pre**: `(1/(Kq)) * ||Xw - y||_2`. Already computed in matching; persist in `eps_pre.json`.

### 4.3 Diagnostics
- **Anchor radius**: `sum_i α_i ||x_i - x*||_2`.  
- **Extrapolation distance**: `theta - theta0[-1]` for each post point.  
- **Effective peers**: `‖w‖_0` (count of weights > 1e-6).  
- **Weight entropy**: `-∑ w_j log(w_j + 1e-12)` (optional).

### 4.4 Bound coverage & tightness
Estimate local Lipschitz constants from **pre‑window** and **anchors**:
- `L_theta`: for each device and anchor, compute finite differences along `theta0` at fixed `x_i`, take max slope over devices & anchors.  
- `L_x`: for nearest‐neighbor anchors at `theta0[-1]`, compute slope in input space:  
  `|F_j(i,θ0[q])-F_j(i',θ0[q])| / ||x_i - x_{i'}||_2`, take max over devices & pairs within the anchor set (clip large outliers).

Define per sample & per θ:
```
RHS = eps_pre + L_theta * (theta - theta0[-1]) + L_x * anchor_radius + B_proxy
coverage = 1[ |tau(theta) - u_pre| <= RHS ]
tightness = |tau(theta) - u_pre| / RHS
```
where `u_pre = sum_i α_i ( F_t(i,θ0[q]) - sum_j w_j F_j(i,θ0[q]) )` (add helper in `te_curve.py`).  
Aggregate coverage as a fraction, and tightness by median/IQR.

### 4.5 Comparability across devices
For a fixed `(x*, d)`, **rotate target** across all devices (keeping same peer pool) and compute **AUC‑TE** per target → a vector.  
- **CV** = std/mean of that vector (skip cases with mean ≈0).  
- **Unit invariance**: rescale the intervened feature (e.g., ×60) and recompute CV; report `ΔCV`.

### 4.6 Stability
- **Weight stability**: bootstrap anchors (e.g., 30 resamples), solve `w^(b)`, compute median `‖w^(b) - w^(b')‖_1`.  
- **TE stability**: variance of `tau^(b)(theta)` across bootstraps (report mean over θ).

### 4.7 P‑mode vs S‑mode gap (subset)
If S‑mode simulator is available, compute `median_θ |tau^S - tau^P|` on matched runs.

---

## 5) Running: CLI recipes

**A) Train devices**

```bash
python -m disco.cli.main train-devices \
  --dataset adult --N 10 --non_iid label_skew --out ./experiments/devs_adult --seed 42
```

**B) Probe (public pre‑window trajectories)**

```bash
python -m disco.cli.main probe \
  --devices ./experiments/devs_adult \
  --dataset adult --probes 500 \
  --theta0 "[0.0,0.5,1.0]" \
  --interventions ./experiments/configs/tabular_interventions.yaml \
  --out ./experiments/results/adult_run2/probes
```

**C) Explain (DISCO + baselines)**

```bash
python -m disco.cli.main explain \
  --devices ./experiments/devs_adult \
  --probes ./experiments/results/adult_run2/probes \
  --dataset adult \
  --interventions ./experiments/configs/tabular_interventions.yaml \
  --K 50 --tau 0.5 --lam 0.01 --mode p \
  --baselines avg,topk,glob,lp,ur \
  --baseline-mode p \
  --Kp 5 \
  --queries 100 --seed 42 \
  --out ./experiments/results/adult_run2
```

**D) Evaluate (produce tables & figures)**

```bash
python -m disco.cli.main evaluate \
  --results ./experiments/results/adult_run2 \
  --figs ./experiments/figs/adult_run2
```

Output layout (mirror for baselines):
```
experiments/results/adult_run2/
  config.yaml
  probes/...
  disco/
    t=<idx>/x_id=<id>/d=<name>/
      y_t.npy, y_syn.npy, tau.npy, weights.npy
      eps_pre.json, diagnostics.json, plots/te_curve.png
  baselines/
    avg/... topk/... glob/... lp/... ur/...
  summary.csv  # aggregated per-sample metrics
```

---

## 6) Tables & figures to generate (automatically in `evaluate`)

**Table 1 — Pre‑window matching** (`ε_pre` ↓, effective peers, sparsity)
```
columns: dataset, method, eps_pre_mean, eps_pre_std, eff_peers_median, sparsity_rate
methods: disco, glob, ur, avg, topk
```

**Table 2 — Post‑window reliability** (coverage ↑, tightness ↓, P‑vs‑S gap ↓)
```
columns: dataset, method, coverage, tightness_median, tightness_iqr, p_vs_s_gap_median
methods: disco, glob, ur
```

**Table 3 — Cross‑device comparability** (CV ↓; ΔCV under unit scaling ↓)
```
columns: dataset, method, cv_mean, cv_std, delta_cv_unit_scale
methods: disco, lp, ig, shap
```

**Table 4 — Stability** (`||Δw||_1` ↓; Var(τ) ↓)
```
columns: dataset, method, l1_w_median, l1_w_iqr, var_tau_mean
methods: disco, glob, ur
```

**Figures** (Matplotlib only; one plot per figure, default colors):
1. TE curves with shaded `|τ|` area (representative samples).  
2. Boxplots of AUC‑TE across methods.  
3. CDFs of flip‑θ (classification).  
4. Scatter: `|τ - u_pre|` vs RHS bound terms; annotate coverage %.  
5. Scatter: P‑mode vs S‑mode gaps vs anchor radius.  
6. Stability: boxplots of `||Δw||_1` vs `K, τ, λ`.

---

## 7) Jupyter notebook workflow

Create `experiments/notebooks/StudyA_Adult.ipynb` with cells like below.  
**Important:** use `matplotlib` only; do not set custom styles/colors.

**Cell 1 — Environment & paths**:
```python
import os, json, numpy as np, pandas as pd
from pathlib import Path
RUN_ID = "adult_run2"
RES = Path("experiments/results")/RUN_ID
FIGS = Path("experiments/figs")/RUN_ID
FIGS.mkdir(parents=True, exist_ok=True)
```

**Cell 2 — (Optional) run CLI from notebook**:
```python
import subprocess, sys
def sh(cmd): 
    print(cmd); 
    subprocess.run(cmd, shell=True, check=True)

# Example: evaluate
sh(f"python -m disco.cli.main evaluate --results {RES} --figs {FIGS}")
```

**Cell 3 — Load summary and show headline tables**:
```python
df = pd.read_csv(RES/"summary.csv")
# Pivot Table 1: eps_pre
pivot1 = df.pivot_table(index="method", values="eps_pre", aggfunc=["mean","std"])
display(pivot1)

# Plot: AUC-TE by method
import matplotlib.pyplot as plt
vals = [df[df.method==m]["auc_te"].values for m in ["disco","glob","ur","avg","topk","lp"]]
labels = ["DISCO","GLOB","UR","AVG","TOPK","LP"]
plt.figure()
plt.boxplot(vals, labels=labels, showfliers=False)
plt.ylabel("AUC-TE")
plt.title("AUC-TE across methods")
plt.show()
```

**Cell 4 — Bound coverage & tightness**
```python
cov = pd.read_csv(RES/"coverage.csv")    # produced by evaluate.py
tight = pd.read_csv(RES/"tightness.csv")
print("Coverage (mean):", cov["coverage"].mean())
plt.figure(); plt.hist(tight["tightness"], bins=30); plt.xlabel("tightness"); plt.ylabel("count"); plt.title("Tightness histogram"); plt.show()
```

**Cell 5 — Comparability (CV)**:
```python
cv = pd.read_csv(RES/"comparability_cv.csv")
display(cv.groupby("method")["cv"].agg(["mean","std"]))
```

**Cell 6 — Stability**:
```python
stab = pd.read_csv(RES/"stability.csv")
display(stab.groupby("method")[["l1_w","var_tau"]].agg(["median","mean"]))
```

---

## 8) Statistical tests (in `evaluate.py`)

Provide utilities that:  
- Do **paired** tests (same `(x*, d)` across methods) for `eps_pre`, `auc_te`, `tightness`, etc. Use **paired t‑test** or **Wilcoxon**; report adjusted p‑values (Holm).  
- Save a `stats_tests.csv` with rows `(metric, method_a, method_b, p_value, significant@0.05)`.

Example usage inside `evaluate.py`:
```python
methods = ["disco","glob","ur","avg","topk"]
stats_eps = stats_tests_pairwise(df, metric="eps_pre", methods=methods)
stats_auc = stats_tests_pairwise(df, metric="auc_te", methods=methods)
stats_eps.to_csv(out_dir/"stats_eps.csv", index=False)
stats_auc.to_csv(out_dir/"stats_auc.csv", index=False)
```

---

## 9) Synthetic ground‑truth validation (optional but recommended)

Expose CLI `synth` (see `EXPERIMENT.md`) and parallel evaluation. Save `mse_tau.csv` with  
`columns = [dataset, method, mse_tau, corr_tau]`. Expect **DISCO < baselines** on MSE.

---

## 10) Reproducibility & logging

- Always write `config.yaml` under the run directory.  
- Save arrays (`y_t.npy`, `y_syn.npy`, `tau.npy`) and scalars (`eps_pre.json`, `diagnostics.json`).  
- Keep a `registry.csv` under `experiments/results` with columns:  
  `run_id, dataset, N, mode, K, tau, lambda, m, q, r, seed`.

---

## 11) Acceptance criteria (what “we win” looks like)

On Adult Income (N=10, m=500, K=50, τ=0.5, λ=0.01):  
- **Table 1**: `ε_pre(DISCO)` significantly **lower** than AVG/TOP‑K/GLOB/UR (p<0.01).  
- **Table 2**: coverage↑ and tightness↓ vs GLOB/UR; P‑vs‑S gap small on subset.  
- **Table 3**: comparability CV(DISCO) **lower** than LP/IG/SHAP; ΔCV under unit scaling small.  
- **Table 4**: stability metrics lower than GLOB/UR.

If these hold across Bike/COMPAS (with task‑appropriate metrics), we can claim DISCO **outperforms baselines** in matching, reliability, comparability, and stability.

---

## 12) Implementation tips

- Respect the **interfaces** (`Device`, `Intervention`, `AnchorSelector`, `WeightSolver`, `Explainer`).  
- Use **numpy**; cache repeated `T_d(θ, x)` evaluations.  
- Matplotlib only, one figure per plot, **no seaborn**, **no custom colors/styles**.  
- Use `numpy.random.Generator(seed)`; store seeds in outputs.  
- Keep modularity: evaluation code should not import training code internals.

---

## 13) Minimal development checklist for Claude Code

- [ ] Implement `disco/explain/baselines.py` APIs as specified.  
- [ ] Extend `disco/cli/main.py` to run baselines and save outputs.  
- [ ] Implement `experiments/evaluate.py` with aggregation, CV, Kendall τ, coverage/tightness, and stats tests.  
- [ ] Generate four tables and all figures into `experiments/figs/<run_id>/`.  
- [ ] Add `experiments/notebooks/StudyA_Adult.ipynb` with cells from §7.  
- [ ] Verify on a small run (N=5, queries=10) before scaling.

---

*End of EXPERIMENT_2.md.*
