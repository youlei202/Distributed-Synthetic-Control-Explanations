# DISCO: Distributed Synthetic Control Explanations (Python)

This repository implements **DISCO** — a privacy‑preserving, **peer‑based treatment‑effect** explanation framework for heterogeneous devices.
The current release **focuses on tabular data**, with an interface that generalizes to images and text.

> **Core idea (1 sentence):** Learn a **synthetic peer mixture** on a small, public **pre‑intervention window**, then compute an explanation on a private sample as a **treatment‑effect curve** along interpretable interventions.

---

## ✨ Key features

- **Heterogeneous devices** (`sklearn`, `xgboost`, `torch` adapters) with a common scalarization `g`.
- **Public probe set** and **pre‑window** trajectories only — no private data leaves a device.
- **Intervention dictionary** for tabular features (additive/multiplicative, monotone, clamped).
- **Anchor selection** (KNN + kernels) with reproducible embeddings.
- **Synthetic control matching** (simplex‑constrained ridge; projected gradient / small‑N Newton).
- **Counterfactual construction** in two modes:
  - **S‑mode**: secure evaluation placeholder (simulated coordinator).
  - **P‑mode**: anchor‑transfer approximation with explicit bias term.
- **Explanations**: treatment‑effect curves, `AUC‑TE`, **flip intensity** (classification).
- **Diagnostics**: pre‑window fit `ε_pre`, weight sparsity, stability (bootstrap over anchors).
- **High cohesion, low coupling**: narrow, testable modules with language‑level interfaces.

---

## 🧱 Repository structure

```
disco/
  __init__.py
  config/
    schema.py            # Typed config, validation, defaults
  data/
    loaders.py           # Probe set, device-local splits, tabular utils
    splits.py            # Reproducible N-device partitioning
  devices/
    base.py              # Device interface (predict, g-scalarize)
    sklearn_tabular.py   # Adapters: LogisticRegression, RF, XGBoost, etc.
    torch_tabular.py     # Optional torch MLP adapter
  interventions/
    base.py              # Intervention interface: apply(theta, x)
    tabular.py           # FeatureAdd, FeatureScale, ClampedShift, etc.
    grids.py             # Pre-window & post-grid builders
  anchors/
    embed.py             # Phi_t embeddings (identity, PCA, autoenc placeholder)
    knn.py               # KNN selection, kernels, weights
  matching/
    weights.py           # Solve simplex ridge (projected grad / Newton)
    simplex.py           # Sorting-based simplex projection (O(n log n))
    metrics.py           # ε_pre, sparsity, stability diagnostics
  counterfactual/
    smode.py             # Secure-eval simulator (no x* leakage)
    pmode.py             # Anchor-transfer proxy builder (Δ responses)
  explain/
    te_curve.py          # τ(θ), AUC-TE, flip-θ
    plotting.py          # Matplotlib visualizations
  privacy/
    dp_noise.py          # (Optional) additive noise on public trajectories
  cli/
    main.py              # CLI entry: probe, match, explain, run-exp
experiments/
  configs/               # YAMLs for datasets, interventions, grids, N, K, τ, λ
  notebooks/             # Analysis & plotting (optional)
  results/               # Saved runs, logs, figures
tests/
  unit/                  # Unit tests per module
  integration/           # End-to-end smoke tests
README.md
pyproject.toml / setup.cfg
```

**Design note (High cohesion & Low coupling):**
- Each subpackage has a single responsibility (cohesion).
- Cross‑module communication uses **small, explicit interfaces**:
  `Device`, `Intervention`, `AnchorSelector`, `WeightSolver`, `CounterfactualBuilder`, `Explainer`.
- No module imports another’s internals — only interfaces (`base.py`) and value objects.

---

## 🔧 Installation

```bash
# Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate
pip install -U pip

# Core stack
pip install numpy scipy pandas scikit-learn matplotlib pyyaml

# Optional: xgboost and cvxpy (only if you need them)
pip install xgboost cvxpy

# For development
pip install pytest black mypy
```

---

## 🧩 Core interfaces (you will implement these)

### 1) Devices

```python
# disco/devices/base.py
from typing import Protocol, Literal, Any
import numpy as np

Scalarization = Literal["prob", "logit"]  # extend as needed

class Device(Protocol):
    id: str
    n_outputs: int  # C
    task: Literal["regression", "classification"]
    scalarization: Scalarization

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Return raw model outputs, shape [B, C] or [B,] for regression."""
        ...

    def g(self, raw: np.ndarray) -> np.ndarray:
        """Scalarize raw outputs to shape [B,]. E.g., prob[:,1] or logit."""
        ...
```

Adapters (e.g., `sklearn_tabular.py`) wrap models to expose this interface.  
**No device shares private `X*` — only public probe responses on the pre‑window.**

### 2) Interventions

```python
# disco/interventions/base.py
class Intervention(Protocol):
    name: str
    feat_idx: int  # tabular feature index (for per-feature interventions)
    theta_max: float

    def apply(self, X: np.ndarray, theta: float) -> np.ndarray:
        """Return X' with intervention along dimension d at intensity theta."""
        ...
```

Tabular implementations (`tabular.py`):
- `FeatureAdd(feat_idx, delta_max, clamp=None)`
- `FeatureScale(feat_idx, scale_max, clamp=None)`
- `ClampedShift(feat_idx, shift_max, lower, upper)`

### 3) Grids

```python
# disco/interventions/grids.py
@dataclass
class Grids:
    theta0: np.ndarray  # pre-window, e.g., [0, ε, 2ε]
    theta:  np.ndarray  # post-grid, e.g., [γ, …, θ_max] with γ > theta0[-1]
```

### 4) Anchors

```python
# disco/anchors/knn.py
@dataclass
class AnchorSelection:
    indices: np.ndarray  # (K,)
    weights: np.ndarray  # (K,), nonnegative, sum to 1

class AnchorSelector:
    def __init__(self, emb: Callable[[np.ndarray], np.ndarray], K: int, tau: float): ...
    def select(self, X_probe: np.ndarray, x_star: np.ndarray) -> AnchorSelection: ...
```

### 5) Matching (Synthetic control weights)

```python
# disco/matching/weights.py
@dataclass
class MatchInputs:
    y_t: np.ndarray         # [K*q]
    X_peers: np.ndarray     # [K*q, N-1]
    lam: float              # λ

@dataclass
class MatchResult:
    w: np.ndarray           # [N-1], on simplex
    eps_pre: float          # average pre-window fit

class WeightSolver:
    def solve(self, mi: MatchInputs) -> MatchResult:
        """Solve min ||X w - y||^2 + λ||w||^2 s.t. w≥0, 1^T w=1."""
        ...
```

- Implementation: **projected gradient** (default) + **small‑N projected Newton**.
- Reusable `simplex.project(v)` in `matching/simplex.py`.

### 6) Counterfactual builders

```python
# disco/counterfactual/smode.py
class SModeCounterfactual:
    def build(self, w: np.ndarray, devices: list[Device],
              x_star: np.ndarray, grids: Grids,
              intervention: Intervention) -> np.ndarray:
        """Return y_syn[θ] = Σ w_j g(f_j(T_d(θ, x*))) for θ in post-grid."""
        ...
```

```python
# disco/counterfactual/pmode.py
class PMModeCounterfactual:
    def build(self, w: np.ndarray, devices: list[Device],
              X_probe: np.ndarray, anchor_sel: AnchorSelection,
              x_star: np.ndarray, grids: Grids,
              intervention: Intervention) -> tuple[np.ndarray, float]:
        """
        Return (y_syn[θ], B_proxy). Uses Δ_j(i,θ) from public probes and
        α_i(x*). Optional g(f_j(x*)) proxy (θ=0) can be provided/estimated.
        """
        ...
```

### 7) Explanations & metrics

```python
# disco/explain/te_curve.py
@dataclass
class TEOutput:
    theta: np.ndarray        # post-grid
    y_t: np.ndarray          # target trajectory
    y_syn: np.ndarray        # synthetic trajectory
    tau: np.ndarray          # τ = y_t - y_syn
    auc_abs: float           # Σ |τ| Δθ
    flip_theta: float | None # minimal flip intensity (classification)

class Explainer:
    def run(self, device_t: Device, y_syn: np.ndarray,
            x_star: np.ndarray, grids: Grids,
            intervention: Intervention, task: str) -> TEOutput: ...
```

---

## ▶️ Quick Demo

Run the interactive Jupyter notebooks:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the basic demo
jupyter notebook experiments/notebooks/disco_demo.ipynb

# Run the full experiment pipeline
jupyter notebook experiments/notebooks/full_experiment_demo.ipynb
```

## 🚀 Command Line Interface

DISCO provides a complete CLI for running experiments:

```bash
# 1) Train heterogeneous devices
python -m disco.cli.main train-devices \
  --dataset adult --N 10 --non-iid label-skew \
  --out ./experiments/devices --seed 42

# 2) Compute probe trajectories
python -m disco.cli.main probe \
  --devices ./experiments/devices \
  --dataset adult --probes 500 \
  --theta0 "[0.0,0.5,1.0]" \
  --interventions ./experiments/configs/tabular_interventions.yaml \
  --out ./experiments/results/run1/probes

# 3) Run DISCO explanations
python -m disco.cli.main explain \
  --devices ./experiments/devices \
  --probes ./experiments/results/run1/probes \
  --dataset adult --targets 0,1,2 --queries 100 \
  --K 50 --tau 0.5 --lam 0.01 --mode p \
  --baselines avg,topk,glob,lp \
  --out ./experiments/results/run1

# 4) Run demo script
python experiments/run_demo.py
```

## 💻 Python API Quickstart

```python
from disco.data.loaders import load_adult
from disco.devices.sklearn_tabular import SklearnDevice
from disco.interventions.tabular import FeatureAdd
from disco.interventions.grids import Grids
from disco.anchors.knn import AnchorSelector
from disco.matching.weights import WeightSolver
from disco.counterfactual.pmode import PModeCounterfactual
from disco.explain.te_curve import Explainer
from disco.cli.main import run_disco_pipeline

# Load data and create devices
X_train, y_train, X_test, y_test = load_adult()
# ... train heterogeneous devices (see notebook for details)

# Define intervention and grids
intervention = FeatureAdd(feat_idx=2, delta_max=5.0, name="add_f2")
grids = Grids(theta0=np.array([0.0, 0.5, 1.0]),
              theta=np.linspace(1.5, 5.0, 8))

# Run complete DISCO pipeline
results = run_disco_pipeline(
    target_device=devices[0],
    devices=devices,
    x_star=X_test[100:101],  # private query
    X_probe=X_test[:500],    # public probes
    intervention=intervention,
    grids=grids,
    anchor_selector=AnchorSelector(emb=lambda X: X, K=50, tau=0.5),
    weight_solver=WeightSolver(),
    counterfactual_builder=PModeCounterfactual(),
    explainer=Explainer(),
    trajectories=trajectories,  # pre-computed
    mode="p"
)

print(f"AUC-TE: {results['te_output'].auc_abs:.4f}")
```

---

## 🧪 Experiments (designed for this repo)

We provide a **reproducible suite** to evaluate correctness, stability, and scalability.

### Datasets (tabular first)
- **Adult Income** (binary classification), **COMPAS** (risk score), **Bike Sharing** (regression).
  Extend with your domain data via a simple loader (no labels required for probes).

### Devices (heterogeneous)
- Logistic Regression, Random Forest, Gradient Boosted Trees (XGBoost), 1‑hidden‑layer MLP.
- Train per‑device on non‑IID splits (label skew, feature shift, covariate shift).
- Store *only* models locally; **no** private features leave device.

### Interventions & grids
- Per‑feature `FeatureAdd` (±k units), `FeatureScale` (×(1+θ)), clamped shifts.
- Pre‑window: `{0, ε, 2ε}`; Post‑grid: `[γ, …, θ_max]` with `γ > 2ε`.
- Choose ε by 5–10% of the feature’s robust scale (IQR/median‑abs‑dev).

### Baselines
1. **AVG**: equal weights on peers.  
2. **GLOB**: matching without anchors (use all probes) — tests locality benefit.  
3. **LP (local perturbation)**: target‑only forward perturbation (no peers).  
4. **LEAK**: (ablation) include post‑grid points in training — should look over‑optimistic.  
5. **RAND**: random simplex weights (sanity check).

### Metrics
- **Primary**: `AUC‑TE`, **flip intensity** (classification), τ curve shapes.
- **Reliability**: `ε_pre`, anchor radius term, `B_proxy` (P‑mode).
- **Stability**: weight variation `‖w(b)−w(b')‖₁` over anchor bootstraps; τ variance.
- **Scalability**: runtime per stage; communication volume `O(mq)` vs `O(mr)`.

### Protocol

1. **Prepare N devices** with heterogeneous models and non‑IID splits.  
2. **Publish pre‑window trajectories** on probes (optionally add DP noise).  
3. For each target and query `x*`:
   - Select **anchors** (`K, τ`).  
   - Solve **matching** (get `w`, `ε_pre`).  
   - Build **counterfactual** (S‑mode simulated or P‑mode).  
   - Compute **TE** (`τ`), `AUC‑TE`, flip‑θ, and diagnostics.
4. **Repeat** for multiple features (interventions), aggregate statistics.
5. **Compare** to baselines; **plot** TE curves with shaded `|τ|` area; report tables.

A ready CLI script (to be implemented) will look like:

```bash
# 1) Train N devices and save adapters
python -m disco.cli.train_devices --dataset adult --N 10 --non_iid label_skew --out ./experiments/devs

# 2) Publish pre-window trajectories on probes
python -m disco.cli.probe --devices ./experiments/devs --probes 500 --theta0 "[0,0.5,1.0]" --out ./experiments/probes

# 3) Run explanations for a set of queries
python -m disco.cli.explain --devices ./experiments/devs --probes ./experiments/probes        --interventions configs/tabular_interventions.yaml        --K 50 --tau 0.5 --lam 0.01 --mode p --results ./experiments/results

# 4) Evaluate and plot
python -m disco.cli.evaluate --results ./experiments/results --plot ./experiments/figs
```

---

## ⚙️ Configuration (YAML)

```yaml
dataset: adult
N_devices: 10
non_iid: label_skew
probe:
  m: 500
  seed: 42
interventions:
  - type: FeatureAdd
    feat_idx: 2
    delta_max: 5.0
  - type: FeatureScale
    feat_idx: 5
    scale_max: 0.5
grids:
  theta0: [0.0, 0.5, 1.0]
  theta:  [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
anchors:
  K: 50
  tau: 0.5
matching:
  lambda: 0.01
mode: p           # or s
privacy:
  dp_noise_std: 0.0
```

---

## 📐 Design notes (extensibility to images/text)

- The `Intervention` interface is data‑type agnostic. For images:
  - `PatchOcclusion(x, theta, mask)` or `BlurRegion`.
  - `phi_t` can be a penultimate‑layer embedding of the local CNN.
- For text:
  - `SpanDrop`, `SynonymReplace`, `PromptScale` with tokenization layers hidden inside the `Device` adapter.
- The **only contracts** used by downstream modules are:
  1) `apply(theta, x)` returns an input of the **same type/shape**;
  2) `Device.g(predict_raw(T_d(theta, x)))` returns a **scalar** per input.

Thus, high cohesion (each module focuses on its type) and low coupling (rest of the pipeline only sees scalars and numpy arrays).

---

## 🧪 Testing checklist

- **Unit**:
  - `interventions/tabular.py`: shape and bounds after `apply`.
  - `matching/simplex.py`: projection correctness and idempotence.
  - `matching/weights.py`: KKT residuals small, `w ≥ 0`, `1^T w = 1`.
  - `anchors/knn.py`: weights nonnegative, sum to 1; KNN indices correct.
  - `explain/te_curve.py`: `AUC-TE` equals trapezoidal area of `|τ|`.

- **Integration**:
  - End‑to‑end on a tiny synthetic dataset (N=3, p=5) validates `ε_pre → τ` monotonicity under controlled shifts.
  - P‑mode vs S‑mode consistency when proxy bias is set to zero.

---

## 🔒 Privacy knobs (optional)

- **Additive Gaussian noise** on public pre‑window trajectories before sharing:
  `F_j(i,θ) ← F_j(i,θ) + 𝒩(0, σ²)`.
- Propagate σ into diagnostics; study `AUC‑TE` degradation vs σ in experiments.

---

## 🧭 Coding style & quality

- Type hints everywhere; `mypy` clean.
- Black‑formatted; docstrings with examples.
- No cross‑module state; pass value objects (`dataclass`) explicitly.
- Randomness controlled by `numpy.random.Generator(seed)`; seeds stored in results.

---

## 📊 Expected figures

- TE curves per feature with `|τ|` shaded (`AUC‑TE`).
- Weight bar plots (active peers) and stability boxplots.
- Error‑bound diagnostics: scatter of `ε_pre` vs `AUC‑TE`, influence of anchor radius.
- Privacy/utility tradeoff: `AUC‑TE` vs DP noise σ.

---

## 📌 Roadmap

- [ ] CLI commands scaffolding (`probe`, `match`, `explain`, `evaluate`).
- [ ] Tabular adapters (sklearn, xgboost) + tests.
- [ ] S‑mode simulator and P‑mode proxy builder.
- [ ] Plotting utilities and paper‑quality figures.
- [ ] Image/text interventions and device adapters.

---

## License

MIT (to be set by the authors).

---

## Citation

