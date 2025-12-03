# EXPERIMENT.md — Implementing and Running DISCO Experiments

This document instructs a coding agent (e.g., Claude Code) to implement **all experiments** for the DISCO project, aligned with the repository layout defined in `README.md`.

The goals are to:
- Implement the end‑to‑end DISCO pipeline for **tabular data**.
- Add **baselines** and **ablations**.
- Provide **evaluation**, **plots**, **reproducibility**, and **scaling** studies.
- Maintain **high cohesion** and **low coupling** via small interfaces and value objects.

> **Reminder**: keep private queries and labels on device; only public probe responses are shared.

---

## 0) Repository alignment and CLI entry points

**Confirm/adjust the repo tree** (see `README.md`). Add missing files as stubs, then fill functions:

```
disco/
  cli/
    main.py              # argparse subcommands: train-devices, probe, explain, evaluate, synth
    train_devices.py     # thin wrapper -> main.py (optional)
    probe.py             # thin wrapper -> main.py (optional)
    explain.py           # thin wrapper -> main.py (optional)
    evaluate.py          # thin wrapper -> main.py (optional)
experiments/
  configs/
    studyA_adult.yaml
    studyB_bike.yaml
    studyC_compas.yaml
    studyS_synth.yaml
  results/               # auto-created, contains run folders
  figs/                  # plots
```

**CLI design**: single entry `python -m disco.cli.main <subcommand> [args]` with subcommands:

- `train-devices`
- `probe`
- `explain`
- `evaluate`
- `synth` (synthetic ground‑truth generator/eval)

Thin wrappers (`train_devices.py` etc.) may call into `main.py` for backwards compatibility.

---

## 1) Datasets and device adapters

### 1.1 Loaders (`disco/data/loaders.py`)

Implement:
```python
def load_adult() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
def load_bike() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
def load_compas() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
```

Utilities:
```python
def standardize_numeric(X: np.ndarray, cols: list[int]) -> tuple[np.ndarray, dict]: ...
def stack_prewindow(traj: dict, target: Device, devices: list[Device],
                    anchor_sel: AnchorSelection, d: Intervention, theta0: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Return y_t [K*q], X_peers [K*q, N-1] in anchor-major, then theta0-major order."""
```

### 1.2 Splits (`disco/data/splits.py`)

Implement non‑IID partitioning:
```python
def split_across_devices(X: np.ndarray, y: np.ndarray, N: int, strategy: str, seed: int
                        ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Strategies: 'iid', 'label-skew', 'covariate-shift', 'feature-shift'."""
```

### 1.3 Device adapters (`disco/devices/*.py`)

Implement `SklearnDevice`, `TorchTabularDevice` (optional), honoring `devices/base.py` Protocol.  
Make sure `Device.g` supports scalarization `"prob"` and `"logit"` for classification, `"identity"` for regression.

---

## 2) Interventions and grids

### 2.1 Interventions (`disco/interventions/tabular.py`)

Implement:
```python
class FeatureAdd(Intervention):  # name=f"add_f{feat_idx}"
    def apply(self, X, theta): ...  # add clamp if provided

class FeatureScale(Intervention):  # name=f"scale_f{feat_idx}"
    def apply(self, X, theta): ...  # multiply (1+theta) with optional clamp

class ClampedShift(Intervention):
    def apply(self, X, theta): ...
```

### 2.2 Grids (`disco/interventions/grids.py`)

```python
@dataclass
class Grids:
    theta0: np.ndarray  # pre-window (ascending), include 0
    theta:  np.ndarray  # post-grid (min(theta) > theta0[-1])
```

---

## 3) Anchors and embeddings

### 3.1 Embeddings (`disco/anchors/embed.py`)

Provide `identity`, `pca(n_components)` helpers.

### 3.2 Anchor selection (`disco/anchors/knn.py`)

```python
class AnchorSelector:
    def __init__(self, emb, K: int, tau: float): ...
    def select(self, X_probe: np.ndarray, x_star: np.ndarray) -> AnchorSelection: ...
```
Compute KNN in embedding space; kernel weights `α_i ∝ exp(-‖z_i - z*‖^2 / (2τ^2))` normalized.

---

## 4) Matching (synthetic mixture)

### 4.1 Simplex projection (`disco/matching/simplex.py`)

Sorting‑based projection (Duchi et al., 2008). Unit test idempotence.

### 4.2 Solver (`disco/matching/weights.py`)

Projected gradient:
```python
class WeightSolver:
    def solve(self, mi: MatchInputs) -> MatchResult:
        # precompute XtX, Xty; iterate:
        #   grad = XtX @ w - Xty + lam * w
        #   w = project_simplex(w - η * grad)
        # stop by KKT residual or max iters
```

Small‑N projected Newton (optional): precompute `Q = XtX + lam I`, factorize, solve on active face.

Diagnostics in `matching/metrics.py`: compute `eps_pre`, sparsity (`‖w‖_0`), and bootstrap stability skeleton.

---

## 5) Public probe trajectories (pre‑window)

### 5.1 Function (`disco/cli/main.py`)

Implement a helper:
```python
def compute_probe_trajectories(devices: list[Device], P: np.ndarray,
                               interventions: list[Intervention], theta0: np.ndarray,
                               dp_noise_std: float = 0.0, seed: int = 0) -> dict:
    """
    Returns dict[(device_id, d.name)] -> np.ndarray [m, q] of scalar responses.
    If dp_noise_std>0, add Gaussian noise to each scalar before sharing.
    """
```
Cache trajectories to `experiments/results/<run_id>/probes/*.npz`.

### 5.2 CLI `probe`

```
python -m disco.cli.main probe \
  --devices ./experiments/devs \
  --dataset adult --probes 500 \
  --theta0 "[0,0.5,1.0]" \
  --interventions ./experiments/configs/tabular_interventions.yaml \
  --out ./experiments/results/$RUN_ID/probes \
  --dp-noise-std 0.0
```

---

## 6) Counterfactual builders

### 6.1 S‑mode (`disco/counterfactual/smode.py`)

Implement secure‑eval **simulator**:
```python
def build(self, w, devices, x_star, grids, intervention) -> np.ndarray:
    return np.array([ sum(w[j] * devices[j_idx].g(devices[j_idx].predict_raw(
                       intervention.apply(x_star, th)))) for th in grids.theta ])
```
(As a simulator, no real SMPC; in production replace with real secure channel.)

### 6.2 P‑mode (`disco/counterfactual/pmode.py`)

Implement anchor‑transfer proxy:
```python
def build(self, w, devices, X_probe, anchor_sel, x_star, grids, intervention) -> tuple[np.ndarray, float]:
    # Precompute Δ_j(i,θ) for anchors i and θ in post-grid using public trajectories
    # Estimate g(f_j(x*)) if allowed (θ=0 on device j), else use device-specific bias proxy=0 with a note
    # y_syn[θ] = Σ_j w_j ( proxy_j + Σ_i α_i Δ_j(i,θ) )
    # B_proxy: report empirical |y_syn^S - y_syn^P| if S-mode available in a subset; else 0 placeholder
```

---

## 7) Explanations and metrics

### 7.1 Explainer (`disco/explain/te_curve.py`)

Implement:
- `compute_y_t(device_t, x_star, grids, intervention)` → `[r]`
- `tau = y_t - y_syn`
- `auc_abs = trapezoid(|tau|, grids.theta)`
- `flip_theta` (classification): first θ where `argmax` changes when using raw model outputs (compute locally).

Plotting helpers in `explain/plotting.py`:
- TE curves with shaded area
- weight bars
- bound diagnostics (scatter)

---

## 8) Baselines and ablations

Create `disco/explain/baselines.py`:

```python
@dataclass
class BaselineOutput:
    theta: np.ndarray
    y_t: np.ndarray
    y_syn: np.ndarray
    tau: np.ndarray
    auc_abs: float
    flip_theta: float | None
    meta: dict

def baseline_avg(devices, x_star, grids, intervention, target_idx) -> BaselineOutput: ...
def baseline_topk_avg(devices, traj, anchor_sel, theta0, Kp, target_idx) -> BaselineOutput: ...
def baseline_glob_matching(devices, traj, theta0, x_star, grids, intervention, target_idx) -> BaselineOutput: ...
def baseline_unconstrained_ridge(devices, X_peers, y_t) -> np.ndarray: ...  # returns w (not on simplex)
def baseline_local_perturbation(device_t, x_star, grids, intervention) -> BaselineOutput: ...
def baseline_kernelshap(device_t, x_star, intervention, grids, background) -> dict:  # optional
```

**Notes**:
- `baseline_topk_avg`: compute pre‑window distance between target and each peer over all probes; pick `K'` nearest peers and average.  
- `baseline_glob_matching`: same optimization as DISCO but use **all probes** (no anchor selection).  
- `baseline_unconstrained_ridge`: solve ridge without simplex constraints; clip negative weights when comparing trajectories to avoid misleading behavior (report both).  
- `baseline_kernelshap`: optional, requires `shap`; if not installed, skip gracefully.

**Ablations** implemented via flags in `explain`:
- **LEAK**: allow using `theta` points in training (should be disabled in normal runs).  
- **Theta0={0}**: override grids to have pre‑window of size 1.  
- **No calibration**: set `scalarization="prob"` vs `"logit"`; measure impact.

---

## 9) Evaluation pipeline

### 9.1 Runner (`disco/cli/main.py` — `explain`)

Implement subcommand to run **one configuration** over a set of queries, target indices, and interventions:

Args:
```
--devices ./experiments/devs
--probes ./experiments/results/$RUN_ID/probes
--dataset adult
--targets 0,1,2,3,4      # or --n-targets 100
--queries 100            # number of x* to sample from test set
--interventions ./experiments/configs/tabular_interventions.yaml
--K 50 --tau 0.5 --lam 0.01
--mode p                 # p or s
--baselines avg,topk,glob,lp,ur
--out ./experiments/results/$RUN_ID
--seed 42
```

**Output structure**:
```
experiments/results/$RUN_ID/
  config.yaml
  probes/...
  disco/
    t=<idx>/
      x_id=<id>/
        d=<name>/
          y_t.npy
          y_syn.npy
          tau.npy
          weights.npy
          eps_pre.json
          diagnostics.json
          plots/te_curve.png
  baselines/
    <baseline_name>/... # mirror structure, with outputs + metrics
  summary.csv
```

### 9.2 Evaluator (`experiments/evaluate.py`)

Implement aggregation:
- Load all runs; compute per‑sample metrics (AUC‑TE, flip‑θ), diagnostics.
- Compute **comparability**: inter‑device dispersion of AUC‑TE; Kendall τ of feature rankings.
- Compute **stability**: weight variation under bootstraps (if available).
- Compute **P‑vs‑S gaps** for overlapping runs.
- Save `summary.csv` and `summary.parquet`.

### 9.3 Plots (`experiments/plots.py`)

- Boxplots of AUC‑TE across methods.
- CDFs of flip‑θ.
- Scatter plots: `|τ - u_pre|` vs bound terms; coverage & tightness tables.
- P‑mode vs S‑mode scatter vs anchor radius.
- Stability boxplots of `||Δw||_1` vs K, τ, λ.

---

## 10) Synthetic study (`disco/cli/main.py` — `synth`)

Implement a synthetic generator per EXPERIMENT PLAN:
- Devices share a base function `b(x)`; target adds a unique component `α s(θ) h(x)`.
- Generate **ground truth** τ and verify **recovery** and **bound coverage**.

Args:
```
python -m disco.cli.main synth --p 8 --N 10 --m 1000 \
  --alpha 0.7 --theta0 "[0,0.2,0.4]" --theta "[0.6,0.8,1.0,1.2]" \
  --K 50 --tau 0.5 --lam 0.01 --mode p --out ./experiments/results/synth1
```

Outputs: τ ground truth vs estimated; coverage and tightness stats.

---

## 11) Configs (YAML)

Create the following templates in `experiments/configs/`:

### 11.1 `studyA_adult.yaml`
```yaml
dataset: adult
N_devices: 10
non_iid: label_skew
probe:
  m: 500
  theta0: [0.0, 0.5, 1.0]
interventions:
  - type: FeatureAdd
    feat_idx: 2
    delta_max: 5.0
  - type: FeatureScale
    feat_idx: 5
    scale_max: 0.5
post_grid: [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
anchors: { K: 50, tau: 0.5 }
matching: { lambda: 0.01 }
mode: p
baselines: [avg, topk, glob, lp, ur]
queries: 100
seed: 42
```

### 11.2 `studyB_bike.yaml`
```yaml
dataset: bike
N_devices: 10
non_iid: covariate_shift
probe: { m: 500, theta0: [0.0, 0.3, 0.6] }
interventions:
  - type: FeatureScale
    feat_idx: 0
    scale_max: 0.5
post_grid: [0.9, 1.2, 1.5, 2.0, 2.5]
anchors: { K: 50, tau: 0.5 }
matching: { lambda: 0.01 }
mode: both   # run P and S in parallel to compare
baselines: [avg, glob, lp]
queries: 80
seed: 123
```

### 11.3 `studyC_compas.yaml`
```yaml
dataset: compas
N_devices: 10
non_iid: feature_shift
probe: { m: 400, theta0: [0.0, 0.4, 0.8] }
interventions:
  - type: FeatureAdd
    feat_idx: 1
    delta_max: 3.0
post_grid: [1.2, 1.6, 2.0, 2.4]
anchors: { K: 30, tau: 0.8 }
matching: { lambda: 0.03 }
mode: p
baselines: [avg, topk, lp]
queries: 60
seed: 7
```

### 11.4 `studyS_synth.yaml`
```yaml
dataset: synth
p: 8
N_devices: 10
probe: { m: 1000, theta0: [0.0, 0.2, 0.4] }
interventions:
  - type: FeatureAdd
    feat_idx: 3
    delta_max: 1.0
post_grid: [0.6, 0.8, 1.0, 1.2]
anchors: { K: 50, tau: 0.5 }
matching: { lambda: 0.01 }
alpha: 0.7
mode: p
queries: 50
seed: 2025
```

---

## 12) Reproducibility and logging

- Every subcommand writes `config.yaml` and `meta.json` (timestamp, git hash, libs).  
- Save arrays as `.npy` and tables as `.csv`/`.parquet`.  
- Use `numpy.random.Generator(seed)` everywhere.  
- Maintain a `experiments/results/registry.csv` with columns: `run_id, dataset, N, mode, K, tau, lambda, m, q, r, seed`.

---

## 13) Tests (quick sanity)

Implement in `tests/`:
- Unit tests for simplex projection, weight solver (KKT residuals), interventions, anchor selection.  
- Integration: tiny synthetic data (N=3, p=5) end‑to‑end produces non‑zero τ when a unique component is planted; LEAK ablation reduces τ spuriously.

---

## 14) Example end‑to‑end (Adult)

```bash
# 0) Train devices
python -m disco.cli.main train-devices --dataset adult --N 10 --non_iid label_skew --out ./experiments/devs --seed 42

# 1) Probe (public)
python -m disco.cli.main probe \
  --devices ./experiments/devs --dataset adult --probes 500 \
  --theta0 "[0.0,0.5,1.0]" --interventions ./experiments/configs/tabular_interventions.yaml \
  --out ./experiments/results/adult_run1/probes

# 2) Explain (DISCO + baselines)
python -m disco.cli.main explain \
  --devices ./experiments/devs \
  --probes ./experiments/results/adult_run1/probes \
  --dataset adult \
  --interventions ./experiments/configs/tabular_interventions.yaml \
  --K 50 --tau 0.5 --lam 0.01 --mode p \
  --baselines avg,topk,glob,lp,ur \
  --queries 100 --seed 42 \
  --out ./experiments/results/adult_run1

# 3) Evaluate and plot
python -m disco.cli.main evaluate --results ./experiments/results/adult_run1 --figs ./experiments/figs/adult_run1
```

---

## 15) Coding style and performance

- Type hints and `@dataclass` for all value containers.  
- Avoid global state; pass config objects explicitly.  
- Use vectorized numpy operations; cache repeated `T_d(θ, x)` when looping θ.  
- When `N` is large, favor projected gradient with precomputed `XtX` and `Xty`.

---

## 16) Extension hooks (images/text)

- Add new `Intervention` subclasses in `interventions/` and new `Device` adapters in `devices/`.  
- Rest of the pipeline (probes → anchors → matching → counterfactual → TE) remains unchanged.

---

**End of EXPERIMENT.md.**
