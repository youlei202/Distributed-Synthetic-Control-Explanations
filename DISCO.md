# DISCO Methodology: Peer-Based Treatment-Effect Explanations

This document presents the **core methodology** behind **DISCO** (DIstributed Synthetic COntrol explanations). It is self-contained and aligned with the repository layout. The focus is on **tabular data**, but every definition is data-type agnostic and extends to images/text via the same interfaces.

---

## 1. Problem Setup and Intuition

- We have **N devices** with heterogeneous models \( f_1,\dots,f_N \). Devices **cannot share private samples or labels**.
- There exists a **public unlabeled probe set** \( \mathcal{P}=\{\mathbf{x}_i\}_{i=1}^m \). It is used only to measure responses, never to train the models centrally.
- We maintain an **intervention dictionary** \( \mathcal{D} \). Each \( d\in\mathcal{D} \) defines a parametric transformation \( T_d(\theta,\mathbf{x}) \) that modifies a single interpretable dimension (e.g., scale feature \( j \), add a clamped offset, occlude an image patch, edit a text span).
- For a **target device** \( t \) and a **private query** \( \mathbf{x}^\ast \), we want an explanation that is:
  - **Directional**: along intervention \( d \) and intensity \( \theta \);
  - **Quantitative & comparable**: measured against the **synthetic peer mixture**;
  - **Privacy-preserving**: private \( \mathbf{x}^\ast \) never leaves device \( t \).

**Intuition.** DISCO follows synthetic control: learn peers **only on a small pre-window of intensities** using **public probes**, then evaluate the explanation **on a post grid** locally at the target. The explanation is a **treatment effect**—how much the target reacts differently from comparable peers when we intervene along \( d \).

---

## 2. Scalarization and Public Responses

Heterogeneous models produce different outputs (logits, probabilities, regression values). We standardize with a **scalarization** \( g:\mathbb{R}^C\to\mathbb{R} \) (e.g., class probability/logit, regression identity).

For device \( j \), intervention \( d \), probe \( \mathbf{x}_i\in\mathcal{P} \), and **pre-window** intensity \( \theta\in\Theta_0(d) \),
\[
F_j^{(d)}(i,\theta)=g\!\big(f_j(T_d(\theta,\mathbf{x}_i))\big)\in\mathbb{R}.
\]

- These are the **only scalars shared** (Line 2 of the algorithm).
- Optional per-device temperature calibration can be applied to make responses more comparable.

**Pre-window vs Post-grid.** For each \( d \), let \( \Theta_0(d)=\{\theta_0^{(1)},\dots,\theta_0^{(q)}\} \) with \( 0=\theta_0^{(1)}<\cdots<\theta_0^{(q)} \) and a **post grid** \( \Theta(d)=\{\theta^{(1)},\dots,\theta^{(r)}\} \) with \( \min\Theta(d)>\theta_0^{(q)} \). This enforces “**match before, evaluate after**”.

---

## 3. Anchor Selection near the Private Query

At the target device \( t \), we localize around \( \mathbf{x}^\ast \) with **anchors**:
1. Embed probes with \( \phi_t:\mathcal{X}\to\mathbb{R}^p \) (identity/PCA/penultimate layer).
2. Select \( \mathcal{A}(\mathbf{x}^\ast)=\mathrm{KNN}_K(\mathbf{x}^\ast;\mathcal{P}) \).
3. Compute nonnegative **kernel weights** \( \{\alpha_i(\mathbf{x}^\ast)\}_{i\in \mathcal{A}} \) with \( \sum_i \alpha_i=1 \).

**Why anchors?** They let us **transport** information from public probes to the private query while maintaining locality. The transport cost is controlled by the **anchor radius** \( \sum_i \alpha_i \|\mathbf{x}_i-\mathbf{x}^\ast\| \).

---

## 4. Pre‑Window Matching: Synthetic Peer Mixture

For each \( d\in\mathcal{D} \), **stack** responses over anchors and pre-window:
\[
\mathbf{y}^{(d)}_t=\bigl[F_t^{(d)}(i,\theta)\bigr]_{i\in\mathcal{A},\ \theta\in\Theta_0(d)}\in\mathbb{R}^{Kq},\quad
\mathbf{X}^{(d)}_{-t}=\bigl[\mathbf{x}^{(d)}_j\bigr]_{j\ne t}\in\mathbb{R}^{Kq\times (N-1)},
\]
where \( \mathbf{x}^{(d)}_j=\bigl[F_j^{(d)}(i,\theta)\bigr]_{i\in\mathcal{A},\ \theta\in\Theta_0(d)} \).

We learn **convex weights** \( \mathbf{w}^{(d)}(\mathbf{x}^\ast) \) by **simplex‑constrained ridge**:
\[
\min_{\mathbf{w}\in\mathbb{R}^{N-1}} \ \bigl\|\mathbf{X}^{(d)}_{-t}\mathbf{w}-\mathbf{y}^{(d)}_t\bigr\|_2^2 + \lambda \|\mathbf{w}\|_2^2
\ \ \text{s.t.}\ \ \mathbf{w}\ge \mathbf{0},\ \mathbf{1}^\top\mathbf{w}=1.
\]
- \( \lambda>0 \) improves conditioning and stability.
- Solver: **projected gradient** (default) or **projected Newton** for small \( N-1 \).

**Diagnostics.** The average pre-window fit
\[
\varepsilon_{\mathrm{pre}}^{(d)}=\frac{1}{Kq}\,\bigl\|\mathbf{X}^{(d)}_{-t}\mathbf{w}^{(d)}-\mathbf{y}^{(d)}_t\bigr\|_2
\]
is a **reliability score**; large values indicate weak matching.

**Interpretation.** In output space, \( \widehat{\mathbf{y}}^{(d)}_{\mathrm{syn}}=\mathbf{X}^{(d)}_{-t}\mathbf{w}^{(d)} \) is the Euclidean projection of \( \mathbf{y}^{(d)}_t \) onto the convex hull of peer columns; the residual \( \mathbf{r}^{(d)}_{\mathrm{pre}}=\mathbf{y}^{(d)}_t-\widehat{\mathbf{y}}^{(d)}_{\mathrm{syn}} \) is **orthogonal** to the active affine face and quantifies the **peer‑inexpressible component** at pre-window.

---

## 5. Counterfactual Construction at the Private Query

Given \( \mathbf{w}^{(d)} \), we need \( \hat{y}^{(d)}_{\mathrm{syn}}(\theta\mid \mathbf{x}^\ast) \) on the post grid.

### 5.1 S‑mode (secure evaluation; preferred when available)
Peers privately compute \( g(f_j(T_d(\theta,\mathbf{x}^\ast))) \) for \( \theta\in\Theta(d) \) via SMPC/TEE. The target forms
\[
\hat{y}^{(d)}_{\mathrm{syn}}(\theta\mid \mathbf{x}^\ast)=\sum_{j\ne t} w^{(d)}_j\, g\!\bigl(f_j(T_d(\theta,\mathbf{x}^\ast))\bigr).
\]
No private feature leaves device \( t \).

### 5.2 P‑mode (anchor‑transfer proxy; no secure channel)
We approximate peer responses at \( \mathbf{x}^\ast \) using **public anchor increments**:
\[
\Delta_j^{(d)}(i,\theta) = F_j^{(d)}(i,\theta)-F_j^{(d)}(i,0),\quad i\in\mathcal{A},\ \theta\in\Theta(d),
\]
and estimate
\[
g(f_j(T_d(\theta,\mathbf{x}^\ast))) \approx g(f_j(\mathbf{x}^\ast)) + \sum_{i\in\mathcal{A}} \alpha_i(\mathbf{x}^\ast)\,\Delta_j^{(d)}(i,\theta).
\]
This introduces a small **proxy bias** \( B_{\mathrm{proxy}} \) that is reported as a diagnostic. The synthetic counterfactual is the same convex mixture as in S‑mode.

---

## 6. Treatment‑Effect Explanation

The target evaluates its **own** post‑window trajectory locally:
\[
y_t^{(d)}(\theta)=g\!\big(f_t(T_d(\theta,\mathbf{x}^\ast))\big).
\]
The **treatment effect** is
\[
\tau_t^{(d)}(\mathbf{x}^\ast,\theta)=y_t^{(d)}(\theta)-\hat{y}^{(d)}_{\mathrm{syn}}(\theta\mid \mathbf{x}^\ast).
\]

**Summary metrics.**
- **AUC‑TE** (magnitude): \(\displaystyle \mathrm{AUC\!-\!TE}^{(d)}(\mathbf{x}^\ast)=\sum_{\theta\in\Theta(d)} |\tau_t^{(d)}(\mathbf{x}^\ast,\theta)|\,\Delta\theta\).
- **Flip intensity** (classification): smallest \( \theta \) on the grid that changes the argmax class.
- Optional: **peak deviation**, **signed area**, **local slope** for elasticity.

**Why this is meaningful.** Under standard smoothness and coverage assumptions, the **post‑window error** of the counterfactual/treatment effect is bounded by four interpretable terms: pre‑window fit \( \varepsilon_{\mathrm{pre}} \), intensity gap \( L_\theta(\theta-\theta_0^{(q)}) \), anchor radius term \( L_x\sum_i \alpha_i\|\mathbf{x}_i-\mathbf{x}^\ast\| \), and proxy bias \( B_{\mathrm{proxy}} \) (P‑mode only). Moreover, \( \tau \) equals the **peer‑inexpressible component** up to these terms, giving the explanation a clear semantics.

---

## 7. Practical Defaults and Hyperparameters

- **Pre‑window** \( \Theta_0(d) \): use \( \{0,\varepsilon,2\varepsilon\} \) with \( \varepsilon \) as 5–10% of a robust feature scale (IQR/MAD). Avoid overlap with the post grid.
- **Post grid** \( \Theta(d) \): start at \( \gamma>\theta_0^{(q)} \); choose 6–12 points up to \( \theta_{\max} \).
- **Anchors**: \( K\in[20,100] \), kernel bandwidth \( \tau\in[0.3,1.0] \) (tune by validation); monitor \( \max_i \alpha_i \) to avoid weight collapse.
- **Regularization** \( \lambda \): \( 10^{-3} \)–\( 10^{-1} \) for tabular; increase if the design is ill‑conditioned.
- **Scalarization** \( g \): probabilities (with temperature) for classification; identity for regression.
- **S‑mode vs P‑mode**: prefer S‑mode (no proxy bias); P‑mode is reliable when anchor radius is small and pre‑window fit is tight.

---

## 8. Diagnostics, Reliability, and Stability

Report alongside every explanation:
- **Pre‑window fit** \( \varepsilon_{\mathrm{pre}} \): large values flag low confidence.
- **Anchor radius** \( \sum_i \alpha_i\|\mathbf{x}_i-\mathbf{x}^\ast\| \): smaller is better.
- **Weight sparsity & identity of peers**: interpretability of the mixture.
- **Stability**: bootstrap anchors; measure \( \|\mathbf{w}^{(b)}-\mathbf{w}^{(b')}\|_1 \) and variance of \( \tau \).
- **Proxy bias** \( B_{\mathrm{proxy}} \) (P‑mode): should be small; otherwise prefer S‑mode.

These diagnostics map directly to the terms in the master error bound and justify conclusions drawn from \( \tau \) and AUC‑TE.

---

## 9. Complexity and Communication (Implementation-Ready)

- **Communication per device**: \( O(mq) \) scalars for pre‑window trajectories (and \( O(mr) \) if full trajectories are shared). No private \( \mathbf{x}^\ast \) is transmitted.
- **Matching**: projected Newton \( O((N\!-\!1)^3) \) for small peer counts; projected gradient \( O(Kq(N\!-\!1)) \) per iteration otherwise.
- **P‑mode**: build \( \Delta_j^{(d)}(i,\theta) \) once per peer in \( O(Kr) \); trivially parallel.

These figures inform experimental scaling and guide defaults for \( m,q,r,N,K \).

---

## 10. Extensions to Images and Text

- **Images**: interventions like `PatchOcclusion`, `BlurRegion`, `ColorShift`; \( \phi_t \) is a CNN embedding. The rest of the pipeline (scalarization, probes, anchors, matching, counterfactuals) is unchanged.
- **Text**: interventions like `SpanDrop`, `SynonymReplace`; \( \phi_t \) is a sentence embedding or model hidden state. The device adapter hides tokenization and model specifics; downstream modules see scalars only.

The type‑agnostic contracts are:
1) `apply(theta, x)` returns the **same type/shape** as `x`;
2) `g(f(T_d(theta, x)))` returns a **scalar**.

---

## 11. Minimal End-to-End Recipe (Tabular)

1. Train heterogeneous models on N devices (non‑IID splits).  
2. Publish pre‑window trajectories \( F_j^{(d)}(i,\theta) \) on public probes.  
3. For a target \( t \) and \( \mathbf{x}^\ast \), select anchors \( \mathcal{A} \) and weights \( \alpha \).  
4. Solve matching on the pre‑window to get \( \mathbf{w}^{(d)} \) and \( \varepsilon_{\mathrm{pre}} \).  
5. Build synthetic counterfactual \( \hat{y}^{(d)}_{\mathrm{syn}}(\theta\mid \mathbf{x}^\ast) \) (S‑mode or P‑mode).  
6. Compute target’s local trajectory \( y_t^{(d)}(\theta) \) and treatment effect \( \tau_t^{(d)}(\mathbf{x}^\ast,\theta) \).  
7. Summarize by **AUC‑TE**, flip‑\( \theta \), and diagnostics; visualize \( \tau \)-curves.

---

## 12. Limitations and Failure Modes

- **Poor pre‑window fit** (\( \varepsilon_{\mathrm{pre}} \) large): mixture unreliable; increase \( K \), refine \( g \), or enlarge pre‑window.
- **Large anchor radius**: anchors not representative; adjust \( K,\tau \) or improve \( \phi_t \).
- **Extrapolation too far**: \( \min\Theta(d) \) close to \( \theta_0^{(q)} \) is preferable; otherwise errors grow.
- **Highly non-smooth models**: Lipschitz assumptions may be weak; consider denser \( \Theta_0 \) or piecewise grids.
- **P‑mode bias**: when large, prefer S‑mode or better proxies for \( g(f_j(\mathbf{x}^\ast)) \).

---

## 13. Mapping to Code (High Cohesion, Low Coupling)

- `devices/`: scalarization and prediction.
- `interventions/` + `grids.py`: transformations and grids.
- `anchors/`: \( \phi_t \), KNN, and kernel weights.
- `matching/`: weight solver + diagnostics.
- `counterfactual/`: S‑mode and P‑mode builders.
- `explain/`: \( \tau \)-curves, AUC‑TE, flip‑\( \theta \), plots.

Each module exposes **small, typed interfaces**; downstream logic depends only on **scalars and numpy arrays**, making the pipeline easy to extend and test.

---

*End of DISCO methodology.*
