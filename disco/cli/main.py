"""Main CLI for DISCO experiments."""

import argparse
import ast
import json
import os
import shutil
import shutil
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split

from disco.data.loaders import load_adult, load_synthetic_tabular, stack_prewindow
from disco.data.splits import split_across_devices
from disco.devices.sklearn_tabular import SklearnDevice
from disco.interventions.tabular import FeatureAdd, FeatureScale, ClampedShift
from disco.interventions.grids import Grids
from disco.anchors.knn import AnchorSelector
from disco.matching.weights import WeightSolver, MatchInputs
from disco.counterfactual.smode import SModeCounterfactual
from disco.counterfactual.pmode import PModeCounterfactual
from disco.explain.te_curve import Explainer
from disco.synth.generator import create_scenario, SyntheticScenario


def save_config_and_meta(config: dict, run_dir: Path, git_hash: str = "unknown"):
    """Save configuration and metadata."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Save metadata
    meta = {
        "timestamp": time.time(),
        "git_hash": git_hash,
        "python_version": sys.version,
        "numpy_version": np.__version__,
    }

    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_interventions_from_config(config: dict) -> List:
    """Load interventions from configuration."""
    interventions = []

    for interv_config in config.get("interventions", []):
        name = interv_config.get("name")
        if interv_config["type"] == "FeatureAdd":
            intervention = FeatureAdd(
                feat_idx=interv_config["feat_idx"],
                delta_max=interv_config["delta_max"],
                name=name or f"add_f{interv_config['feat_idx']}"
            )
        elif interv_config["type"] == "FeatureScale":
            intervention = FeatureScale(
                feat_idx=interv_config["feat_idx"],
                scale_max=interv_config["scale_max"],
                name=name or f"scale_f{interv_config['feat_idx']}"
            )
        elif interv_config["type"] == "ClampedShift":
            intervention = ClampedShift(
                feat_idx=interv_config["feat_idx"],
                shift_max=interv_config["shift_max"],
                lower=interv_config.get("lower", -np.inf),
                upper=interv_config.get("upper", np.inf),
                name=name or f"clamped_f{interv_config['feat_idx']}"
            )
        else:
            raise ValueError(f"Unknown intervention type: {interv_config['type']}")

        interventions.append(intervention)

    return interventions


def train_devices_cmd(args):
    """Train heterogeneous devices with non-IID data splits."""
    if args.dataset == "scenario":
        print(f"Training {args.N} devices on synthetic scenario data...")

        if not args.scenario:
            raise ValueError("--scenario path must be provided when dataset is 'scenario'")

        scenario_path = Path(args.scenario)
        if not scenario_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {scenario_path}")

        scenario = SyntheticScenario.load(scenario_path)
        os.makedirs(args.out, exist_ok=True)
        rng = np.random.RandomState(args.seed)

        model_templates = [
            RandomForestRegressor(n_estimators=120, random_state=args.seed),
            GradientBoostingRegressor(random_state=args.seed),
            DecisionTreeRegressor(max_depth=6, random_state=args.seed),
            MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, random_state=args.seed),
            Ridge(alpha=1.0),
        ]

        devices = []
        registry_entries = []

        for i in range(args.N):
            X, y = scenario.sample_dataset(i, args.samples, rng)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=args.seed + i
            )

            template = model_templates[i % len(model_templates)]
            params = template.get_params()
            if "random_state" in params:
                params["random_state"] = args.seed + i
            model = template.__class__(**params)

            model.fit(X_train, y_train)
            mse = float(np.mean((model.predict(X_test) - y_test) ** 2))

            device = SklearnDevice(
                id=f"device_{i+1}",
                model=model,
                task="regression",
                scalarization="identity",
            )
            with open(Path(args.out) / f"device_{i+1}.pkl", "wb") as f:
                pickle.dump(device, f)
            devices.append(device)

            registry_entries.append(
                {
                    "device": device.id,
                    "model": model.__class__.__name__,
                    "mse": mse,
                }
            )

            print(f"Device {i+1}: {model.__class__.__name__} (MSE {mse:.4f})")

        registry = {
            "devices": [d.id for d in devices],
            "dataset": "scenario",
            "scenario": str(scenario_path),
            "samples": args.samples,
            "seed": args.seed,
            "models": registry_entries,
        }

        with open(Path(args.out) / "registry.json", "w") as f:
            json.dump(registry, f, indent=2)

        print(f"Saved {len(devices)} devices to {args.out}")
        return

    print(f"Training {args.N} devices on {args.dataset} with {args.non_iid} split...")

    # Load data
    if args.dataset == "adult":
        X_train, y_train, X_test, y_test = load_adult()
    elif args.dataset == "synthetic":
        X_train, y_train, X_test, y_test = load_synthetic_tabular(
            n_samples=2000, n_features=10, seed=args.seed
        )
    elif args.dataset == "scenario":
        if not args.scenario:
            raise ValueError("--scenario path must be provided for scenario dataset")
        scenario = SyntheticScenario.load(Path(args.scenario))
        X_train = scenario.X_probe
        X_test = scenario.X_probe
        y_train = None
        y_test = None
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Split data across devices
    splits = split_across_devices(
        X_train, y_train, N=args.N,
        strategy=args.non_iid.replace("-", "_"),
        seed=args.seed
    )

    # Define heterogeneous models
    model_types = [
        LogisticRegression(max_iter=200, random_state=args.seed),
        RandomForestClassifier(n_estimators=50, random_state=args.seed),
        DecisionTreeClassifier(max_depth=5, random_state=args.seed),
        GradientBoostingClassifier(n_estimators=30, random_state=args.seed),
        MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=args.seed)
    ]

    # Train devices
    devices = []
    os.makedirs(args.out, exist_ok=True)

    for i, ((X_i, y_i), model_template) in enumerate(zip(splits, model_types * (args.N // len(model_types) + 1))):
        print(f"Device {i+1}: Training {model_template.__class__.__name__}")

        # Clone model with different random state
        model = model_template.__class__(**{
            **model_template.get_params(),
            "random_state": args.seed + i
        })

        model.fit(X_i, y_i)

        device = SklearnDevice(
            id=f"device_{i+1}",
            model=model,
            task="classification",
            scalarization="prob"
        )
        devices.append(device)

        # Save device
        with open(f"{args.out}/device_{i+1}.pkl", "wb") as f:
            pickle.dump(device, f)

        # Test accuracy
        if len(X_test) > 100:
            acc = model.score(X_test[:100], y_test[:100])
            print(f"  Test accuracy: {acc:.3f}")

    # Save device registry
    registry = {
        "devices": [d.id for d in devices],
        "dataset": args.dataset,
        "N": args.N,
        "non_iid": args.non_iid,
        "seed": args.seed
    }

    with open(f"{args.out}/registry.json", "w") as f:
        json.dump(registry, f, indent=2)

    print(f"Saved {len(devices)} devices to {args.out}")




def load_devices(devices_dir: str) -> List:
    """Load trained devices from directory."""
    devices = []
    device_files = sorted([f for f in os.listdir(devices_dir) if f.startswith("device_") and f.endswith(".pkl")])

    for device_file in device_files:
        with open(os.path.join(devices_dir, device_file), "rb") as f:
            device = pickle.load(f)
        devices.append(device)

    return devices


def compute_probe_trajectories(
    devices: List,
    X_probe: np.ndarray,
    interventions: List,
    theta0: np.ndarray,
    dp_noise_std: float = 0.0,
    seed: int = 0,
    target_class: int = 1
) -> Dict[Tuple[str, str], np.ndarray]:
    """Compute pre-window trajectories on public probes."""
    rng = np.random.RandomState(seed)
    trajectories = {}

    for device in devices:
        for intervention in interventions:
            key = (device.id, intervention.name)
            m, q = len(X_probe), len(theta0)
            traj = np.zeros((m, q))

            for j, theta in enumerate(theta0):
                X_intervened = intervention.apply(X_probe, theta)
                raw = device.predict_raw(X_intervened)
                scalar = device.g(raw, target_class)
                traj[:, j] = scalar

                # Add DP noise if specified
                if dp_noise_std > 0:
                    traj[:, j] += rng.normal(0, dp_noise_std, m)

            trajectories[key] = traj

    return trajectories


def probe_cmd(args):
    """Compute and save probe trajectories."""
    print(f"Computing probe trajectories...")

    # Load devices
    devices = load_devices(args.devices)
    print(f"Loaded {len(devices)} devices")

    # Load dataset for probe selection
    if args.dataset == "adult":
        X_train, y_train, X_test, y_test = load_adult()
        X_probe = X_test[:args.probes]
        theta0 = np.array(eval(args.theta0))
    elif args.dataset == "synthetic":
        X_train, y_train, X_test, y_test = load_synthetic_tabular(
            n_samples=2000, n_features=10, seed=args.seed
        )
        X_probe = X_test[:args.probes]
        theta0 = np.array(eval(args.theta0))
    elif args.dataset == "scenario":
        if not args.scenario:
            raise ValueError("--scenario path must be provided for scenario dataset")
        scenario = SyntheticScenario.load(Path(args.scenario))
        X_probe = scenario.X_probe[:min(args.probes, len(scenario.X_probe))]
        theta0 = scenario.theta0
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Selected {len(X_probe)} probe samples")
    print(f"Pre-window: {theta0}")

    # Load interventions
    if args.interventions:
        if args.interventions.endswith('.yaml'):
            with open(args.interventions) as f:
                interv_config = yaml.safe_load(f)
            interventions = load_interventions_from_config(interv_config)
        else:
            # Non-YAML argument provided; fall back to defaults
            interventions = [
                FeatureAdd(feat_idx=0, delta_max=3.0, name="add_f0"),
                FeatureScale(feat_idx=1, scale_max=0.5, name="scale_f1")
            ]
    else:
        # Default interventions when no argument supplied
        interventions = [
            FeatureAdd(feat_idx=0, delta_max=3.0, name="add_f0"),
            FeatureScale(feat_idx=1, scale_max=0.5, name="scale_f1")
        ]

    print(f"Computing trajectories for {len(interventions)} interventions")

    # Compute trajectories
    target_class = 1 if getattr(devices[0], 'task', 'classification') == 'classification' else 0
    trajectories = compute_probe_trajectories(
        devices=devices,
        X_probe=X_probe,
        interventions=interventions,
        theta0=theta0,
        dp_noise_std=args.dp_noise_std,
        seed=args.seed,
        target_class=target_class
    )

    # Save trajectories
    os.makedirs(args.out, exist_ok=True)

    for (device_id, interv_name), traj in trajectories.items():
        filename = f"{device_id}_{interv_name}.npy"
        np.save(os.path.join(args.out, filename), traj)

    # Save probe metadata
    probe_meta = {
        "X_probe_shape": X_probe.shape,
        "theta0": theta0.tolist(),
        "interventions": [interv.name for interv in interventions],
        "dp_noise_std": args.dp_noise_std,
        "seed": args.seed
    }

    with open(os.path.join(args.out, "probe_meta.json"), "w") as f:
        json.dump(probe_meta, f, indent=2)

    # Save probe data
    np.save(os.path.join(args.out, "X_probe.npy"), X_probe)

    print(f"Saved trajectories to {args.out}")


def explain_cmd(args):
    """Run DISCO explanations and baselines."""
    print("Running DISCO explanations...")

    # Create output directory
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load devices
    devices = load_devices(args.devices)
    target_devices = [devices[i] for i in map(int, args.targets.split(","))]

    # Load probe data
    probe_dir = Path(args.probes)
    X_probe = np.load(probe_dir / "X_probe.npy")

    with open(probe_dir / "probe_meta.json") as f:
        probe_meta = json.load(f)

    theta0 = np.array(probe_meta["theta0"])

    # Load trajectories
    trajectories = {}
    for traj_file in probe_dir.glob("*.npy"):
        if traj_file.name == "X_probe.npy":
            continue

        # Parse filename: device_id_intervention_name.npy
        parts = traj_file.stem.split("_", 2)
        if len(parts) >= 3:
            device_id = f"{parts[0]}_{parts[1]}"  # device_1
            interv_name = "_".join(parts[2:])     # rest
        else:
            continue

        key = (device_id, interv_name)
        trajectories[key] = np.load(traj_file)

    print(f"Loaded {len(trajectories)} trajectory matrices")

    # Load dataset for test queries
    if args.dataset == "adult":
        X_train, y_train, X_test, y_test = load_adult()
    elif args.dataset == "synthetic":
        X_train, y_train, X_test, y_test = load_synthetic_tabular(
            n_samples=2000, n_features=10, seed=args.seed
        )
    elif args.dataset == "scenario":
        if not args.scenario:
            raise ValueError("--scenario path must be provided for scenario dataset")
        scenario = SyntheticScenario.load(Path(args.scenario))
        X_train = scenario.X_probe
        X_test = scenario.X_probe
        y_train = None
        y_test = None
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Load interventions from config
    if args.interventions:
        if args.interventions.endswith('.yaml'):
            with open(args.interventions) as f:
                interv_config = yaml.safe_load(f)
            interventions = load_interventions_from_config(interv_config)
        else:
            interventions = [
                FeatureAdd(feat_idx=0, delta_max=3.0, name="add_f0")
            ]
    else:
        interventions = [
            FeatureAdd(feat_idx=0, delta_max=3.0, name="add_f0")
        ]

    # Create grids
    post_grid = np.linspace(1.5, 3.0, 6)  # Default post grid
    grids = Grids(theta0=theta0, theta=post_grid)

    # Initialize DISCO components
    anchor_selector = AnchorSelector(emb=lambda X: X, K=args.K, tau=args.tau)
    weight_solver = WeightSolver(method="projected_grad", max_iter=1000)
    explainer = Explainer(compute_flip=True)

    # Sample query indices
    rng = np.random.RandomState(args.seed)
    query_indices = rng.choice(len(X_test), min(args.queries, len(X_test)), replace=False)

    print(f"Running explanations for {len(query_indices)} queries on {len(target_devices)} targets")

    # Run explanations
    for t_idx, target_device in enumerate(target_devices):
        target_dir = run_dir / f"t={t_idx}"
        target_dir.mkdir(exist_ok=True)

        for q_idx, x_idx in enumerate(query_indices):
            query_dir = target_dir / f"x_id={x_idx}"
            query_dir.mkdir(exist_ok=True)

            x_star = X_test[x_idx:x_idx+1]

            for intervention in interventions:
                interv_dir = query_dir / f"d={intervention.name}"
                interv_dir.mkdir(exist_ok=True)

                try:
                    # Run DISCO pipeline
                    result = run_disco_pipeline(
                        target_device=target_device,
                        devices=devices,
                        x_star=x_star,
                        X_probe=X_probe,
                        intervention=intervention,
                        grids=grids,
                        anchor_selector=anchor_selector,
                        weight_solver=weight_solver,
                        counterfactual_builder=PModeCounterfactual() if args.mode == "p" else SModeCounterfactual(),
                        explainer=explainer,
                        trajectories=trajectories,
                        target_class=1,
                        mode=args.mode,
                        lam=args.lam
                    )

                    # Save results
                    np.save(interv_dir / "y_t.npy", result["te_output"].y_t)
                    np.save(interv_dir / "y_syn.npy", result["te_output"].y_syn)
                    np.save(interv_dir / "tau.npy", result["te_output"].tau)
                    np.save(interv_dir / "weights.npy", result["weights"])

                    # Save diagnostics
                    diagnostics = {
                        "eps_pre": float(result["eps_pre"]),
                        "anchor_radius": float(result["anchor_radius"]),
                        "proxy_bias": float(result["proxy_bias"]),
                        "auc_te": float(result["te_output"].auc_abs),
                        "flip_theta": float(result["te_output"].flip_theta) if result["te_output"].flip_theta is not None else None,
                        "peak_tau": float(result["te_output"].peak_tau),
                        "n_effective_peers": int(result["n_effective_peers"]),
                        "weight_entropy": float(result["weight_entropy"])
                    }

                    with open(interv_dir / "diagnostics.json", "w") as f:
                        json.dump(diagnostics, f, indent=2)

                except Exception as e:
                    print(f"Error processing target {t_idx}, query {q_idx}, intervention {intervention.name}: {e}")
                    continue

            if q_idx % 10 == 0:
                print(f"  Completed {q_idx+1}/{len(query_indices)} queries for target {t_idx}")

    print("Explanation complete!")


def run_disco_pipeline(
    target_device,
    devices: List,
    x_star: np.ndarray,
    X_probe: np.ndarray,
    intervention,
    grids: Grids,
    anchor_selector,
    weight_solver,
    counterfactual_builder,
    explainer,
    trajectories: Dict,
    target_class: int = 1,
    mode: str = "p",
    lam: float = 0.01
) -> dict:
    """Run complete DISCO pipeline."""
    from disco.matching.metrics import (
        compute_weight_sparsity, compute_effective_peers,
        compute_weight_entropy
    )
    from disco.anchors.knn import compute_anchor_radius

    # Anchor selection
    anchor_sel = anchor_selector.select(X_probe, x_star)
    anchor_radius = compute_anchor_radius(anchor_sel, X_probe, x_star)

    # Stack pre-window data
    y_t, X_peers = stack_prewindow(
        trajectories, target_device, devices,
        anchor_sel, intervention, grids.theta0
    )

    # Solve for weights
    match_inputs = MatchInputs(y_t=y_t, X_peers=X_peers, lam=lam)
    match_result = weight_solver.solve(match_inputs)

    # Build counterfactual
    if mode == "s":
        y_syn = counterfactual_builder.build(
            match_result.w, devices, x_star, grids,
            intervention, target_device, target_class
        )
        proxy_bias = 0.0
    else:
        y_syn, proxy_bias = counterfactual_builder.build(
            match_result.w, devices, X_probe, anchor_sel,
            x_star, grids, intervention, target_device, target_class
        )

    # Compute treatment effect
    te_output = explainer.run(
        target_device, y_syn, x_star, grids,
        intervention, target_device.task, target_class
    )

    # Compute diagnostics
    weight_sparsity = compute_weight_sparsity(match_result.w)
    n_effective = compute_effective_peers(match_result.w)
    weight_entropy = compute_weight_entropy(match_result.w)

    return {
        "te_output": te_output,
        "weights": match_result.w,
        "eps_pre": match_result.eps_pre,
        "anchor_radius": anchor_radius,
        "proxy_bias": proxy_bias,
        "weight_sparsity": weight_sparsity,
        "n_effective_peers": n_effective,
        "weight_entropy": weight_entropy,
        "anchor_selection": anchor_sel
    }


def evaluate_cmd(args):
    """Evaluate and aggregate results."""
    print("Evaluating results...")

    results_dir = Path(args.results)

    # TODO: Implement result aggregation and evaluation
    print("Evaluation not yet implemented")


def synth_cmd(args):
    """Run synthetic ground-truth study."""
    print("Running synthetic study...")

    import json
    import numpy as np

    rng = np.random.RandomState(args.seed)

    def _parse_grid(raw: str, name: str) -> np.ndarray:
        try:
            values = ast.literal_eval(raw)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Could not parse {name} as list: {raw}") from exc
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError(f"{name} must be a 1D, non-empty list")
        return arr

    theta0 = _parse_grid(args.theta0, "theta0")
    theta_post = _parse_grid(args.theta, "theta")
    grids = Grids(theta0=theta0, theta=theta_post)

    scenario = create_scenario(
        p=args.p,
        N=args.N,
        alpha_target=args.alpha,
        theta0=theta0,
        theta_post=theta_post,
        m_probe=args.m,
        rng=rng,
        base_jitter=args.base_jitter,
        peer_alpha_std=args.peer_alpha_std,
        noise_std=args.noise_std,
    )

    devices = scenario.devices
    X_probe = scenario.X_probe
    x_star = scenario.x_star

    intervention = FeatureAdd(feat_idx=0, delta_max=float(theta_post.max()), name="intensity_shift")

    trajectories = compute_probe_trajectories(
        devices=devices,
        X_probe=X_probe,
        interventions=[intervention],
        theta0=theta0,
        dp_noise_std=0.0,
        seed=args.seed,
        target_class=0,
    )

    anchor_selector = AnchorSelector(emb=lambda X: X, K=min(args.K, len(X_probe)), tau=args.tau)
    weight_solver = WeightSolver(method="projected_grad", max_iter=1000)
    counterfactual_builder = PModeCounterfactual() if args.mode == "p" else SModeCounterfactual()
    explainer = Explainer(compute_flip=False)

    result = run_disco_pipeline(
        target_device=devices[0],
        devices=devices,
        x_star=x_star,
        X_probe=X_probe,
        intervention=intervention,
        grids=grids,
        anchor_selector=anchor_selector,
        weight_solver=weight_solver,
        counterfactual_builder=counterfactual_builder,
        explainer=explainer,
        trajectories=trajectories,
        target_class=0,
        mode=args.mode,
        lam=args.lam,
    )

    te_output = result["te_output"]

    _, _, tau_true = scenario.compute_tau(
        weights=result["weights"],
        intervention=intervention,
        x=x_star,
        theta=theta_post,
    )

    abs_error = np.abs(te_output.tau - tau_true)
    mse = float(np.mean(abs_error ** 2))
    mae = float(abs_error.mean())
    max_err = float(abs_error.max())

    eps_pre = float(result["eps_pre"])
    anchor_radius = float(result["anchor_radius"])
    proxy_bias = float(result["proxy_bias"])
    bound = eps_pre + anchor_radius + proxy_bias
    coverage = bool(np.all(abs_error <= bound + 1e-8))

    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "p": args.p,
        "N": args.N,
        "m": args.m,
        "alpha": args.alpha,
        "theta0": theta0.tolist(),
        "theta": theta_post.tolist(),
        "K": args.K,
        "tau": args.tau,
        "lam": args.lam,
        "mode": args.mode,
        "base_jitter": args.base_jitter,
        "peer_alpha_std": args.peer_alpha_std,
        "noise_std": args.noise_std,
        "seed": args.seed,
    }
    save_config_and_meta(run_config, out_dir)

    scenario.save(out_dir / "scenario.npz")

    probes_dir = out_dir / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)
    for (device_id, interv_name), traj in trajectories.items():
        np.save(probes_dir / f"{device_id}_{interv_name}.npy", traj)
    np.save(probes_dir / "X_probe.npy", X_probe)
    with (probes_dir / "probe_meta.json").open("w") as f:
        json.dump(
            {
                "theta0": theta0.tolist(),
                "interventions": sorted({name for _, name in trajectories.keys()}),
                "X_probe_shape": list(X_probe.shape),
                "dp_noise_std": 0.0,
                "seed": args.seed,
            },
            f,
            indent=2,
        )

    np.save(out_dir / "theta.npy", te_output.theta)
    np.save(out_dir / "tau_est.npy", te_output.tau)
    np.save(out_dir / "tau_true.npy", tau_true)
    np.save(out_dir / "y_target.npy", te_output.y_t)
    np.save(out_dir / "y_syn.npy", te_output.y_syn)
    np.save(out_dir / "weights.npy", result["weights"])
    np.save(out_dir / "x_star.npy", x_star)

    summary = {
        "mse": mse,
        "mae": mae,
        "max_abs_error": max_err,
        "coverage": coverage,
        "bound": bound,
        "eps_pre": eps_pre,
        "anchor_radius": anchor_radius,
        "proxy_bias": proxy_bias,
        "n_effective_peers": int(result["n_effective_peers"]),
        "weight_entropy": float(result["weight_entropy"]),
        "per_theta": [
            {
                "theta": float(theta_val),
                "tau_est": float(tau_est_val),
                "tau_true": float(tau_true_val),
                "abs_error": float(err_val),
            }
            for theta_val, tau_est_val, tau_true_val, err_val in zip(
                te_output.theta, te_output.tau, tau_true, abs_error
            )
        ],
    }

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Synthetic study complete. Results saved to {out_dir}")
    print(f"  MSE: {mse:.4e} | MAE: {mae:.4e} | coverage: {coverage}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="DISCO: Distributed Synthetic Control Explanations")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train devices
    train_parser = subparsers.add_parser("train-devices", help="Train heterogeneous devices")
    train_parser.add_argument("--dataset", choices=["adult", "synthetic", "scenario"], default="adult")
    train_parser.add_argument("--N", type=int, default=5, help="Number of devices")
    train_parser.add_argument("--non-iid", choices=["iid", "label-skew", "covariate-shift", "feature-shift"], default="label-skew")
    train_parser.add_argument("--out", required=True, help="Output directory for devices")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--scenario", help="Path to synthetic scenario for scenario dataset")
    train_parser.add_argument("--samples", type=int, default=2000, help="Samples per device for scenario dataset")

    # Probe
    probe_parser = subparsers.add_parser("probe", help="Compute probe trajectories")
    probe_parser.add_argument("--devices", required=True, help="Directory with trained devices")
    probe_parser.add_argument("--dataset", choices=["adult", "synthetic", "scenario"], default="adult")
    probe_parser.add_argument("--probes", type=int, default=500, help="Number of probe samples")
    probe_parser.add_argument("--theta0", default="[0.0,0.5,1.0]", help="Pre-window intensities")
    probe_parser.add_argument("--interventions", help="Interventions config file")
    probe_parser.add_argument("--out", required=True, help="Output directory")
    probe_parser.add_argument("--dp-noise-std", type=float, default=0.0, help="DP noise level")
    probe_parser.add_argument("--seed", type=int, default=42)
    probe_parser.add_argument("--scenario", help="Path to synthetic scenario for scenario dataset")

    # Explain
    explain_parser = subparsers.add_parser("explain", help="Run DISCO explanations")
    explain_parser.add_argument("--devices", required=True, help="Directory with trained devices")
    explain_parser.add_argument("--probes", required=True, help="Directory with probe trajectories")
    explain_parser.add_argument("--dataset", choices=["adult", "synthetic", "scenario"], default="adult")
    explain_parser.add_argument("--targets", default="0", help="Target device indices (comma-separated)")
    explain_parser.add_argument("--queries", type=int, default=100, help="Number of queries")
    explain_parser.add_argument("--interventions", help="Interventions config file")
    explain_parser.add_argument("--K", type=int, default=50, help="Number of anchors")
    explain_parser.add_argument("--tau", type=float, default=0.5, help="Kernel bandwidth")
    explain_parser.add_argument("--lam", type=float, default=0.01, help="Ridge regularization")
    explain_parser.add_argument("--mode", choices=["p", "s"], default="p", help="Mode: p (proxy) or s (secure)")
    explain_parser.add_argument("--baselines", default="", help="Baselines to run (comma-separated)")
    explain_parser.add_argument("--out", required=True, help="Output directory")
    explain_parser.add_argument("--seed", type=int, default=42)
    explain_parser.add_argument("--scenario", help="Path to synthetic scenario for scenario dataset")

    # Evaluate
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate results")
    evaluate_parser.add_argument("--results", required=True, help="Results directory")
    evaluate_parser.add_argument("--figs", help="Output directory for figures")

    # Synth
    synth_parser = subparsers.add_parser("synth", help="Run synthetic study")
    synth_parser.add_argument("--p", type=int, default=8, help="Number of features (including intensity slot)")
    synth_parser.add_argument("--N", type=int, default=10, help="Number of devices")
    synth_parser.add_argument("--m", type=int, default=1000, help="Number of probe samples")
    synth_parser.add_argument("--alpha", type=float, default=0.7, help="Treatment strength for target device")
    synth_parser.add_argument("--theta0", default="[0.0,0.2,0.4]", help="Pre-window intensities as list")
    synth_parser.add_argument("--theta", default="[0.6,0.8,1.0,1.2]", help="Post-grid intensities as list")
    synth_parser.add_argument("--K", type=int, default=50, help="Number of anchors for matching")
    synth_parser.add_argument("--tau", type=float, default=0.5, help="Kernel bandwidth for anchors")
    synth_parser.add_argument("--lam", type=float, default=0.01, help="Ridge regularization for matching")
    synth_parser.add_argument("--mode", choices=["p", "s"], default="p", help="DISCO mode: proxy (p) or secure (s)")
    synth_parser.add_argument("--base-jitter", type=float, default=0.3, help="Stddev of device-specific base offsets")
    synth_parser.add_argument("--peer-alpha-std", type=float, default=0.1, help="Stddev of peer treatment strengths")
    synth_parser.add_argument("--noise-std", type=float, default=0.05, help="Observation noise scale in generator")
    synth_parser.add_argument("--out", required=True, help="Output directory")
    synth_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.command == "train-devices":
        train_devices_cmd(args)
    elif args.command == "probe":
        probe_cmd(args)
    elif args.command == "explain":
        explain_cmd(args)
    elif args.command == "evaluate":
        evaluate_cmd(args)
    elif args.command == "synth":
        synth_cmd(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
