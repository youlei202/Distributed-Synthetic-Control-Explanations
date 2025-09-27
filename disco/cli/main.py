"""Main CLI for DISCO experiments."""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

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
        if interv_config["type"] == "FeatureAdd":
            intervention = FeatureAdd(
                feat_idx=interv_config["feat_idx"],
                delta_max=interv_config["delta_max"],
                name=f"add_f{interv_config['feat_idx']}"
            )
        elif interv_config["type"] == "FeatureScale":
            intervention = FeatureScale(
                feat_idx=interv_config["feat_idx"],
                scale_max=interv_config["scale_max"],
                name=f"scale_f{interv_config['feat_idx']}"
            )
        elif interv_config["type"] == "ClampedShift":
            intervention = ClampedShift(
                feat_idx=interv_config["feat_idx"],
                shift_max=interv_config["shift_max"],
                lower=interv_config.get("lower", -np.inf),
                upper=interv_config.get("upper", np.inf),
                name=f"clamped_f{interv_config['feat_idx']}"
            )
        else:
            raise ValueError(f"Unknown intervention type: {interv_config['type']}")

        interventions.append(intervention)

    return interventions


def train_devices_cmd(args):
    """Train heterogeneous devices with non-IID data splits."""
    print(f"Training {args.N} devices on {args.dataset} with {args.non_iid} split...")

    # Load data
    if args.dataset == "adult":
        X_train, y_train, X_test, y_test = load_adult()
    elif args.dataset == "synthetic":
        X_train, y_train, X_test, y_test = load_synthetic_tabular(
            n_samples=2000, n_features=10, seed=args.seed
        )
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
    elif args.dataset == "synthetic":
        X_train, y_train, X_test, y_test = load_synthetic_tabular(
            n_samples=2000, n_features=10, seed=args.seed
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Select probes
    X_probe = X_test[:args.probes]
    print(f"Selected {len(X_probe)} probe samples")

    # Parse theta0
    theta0 = np.array(eval(args.theta0))
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
    trajectories = compute_probe_trajectories(
        devices=devices,
        X_probe=X_probe,
        interventions=interventions,
        theta0=theta0,
        dp_noise_std=args.dp_noise_std,
        seed=args.seed
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
                        mode=args.mode
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
    mode: str = "p"
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
    match_inputs = MatchInputs(y_t=y_t, X_peers=X_peers, lam=0.01)
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

    # TODO: Implement synthetic study
    print("Synthetic study not yet implemented")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="DISCO: Distributed Synthetic Control Explanations")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train devices
    train_parser = subparsers.add_parser("train-devices", help="Train heterogeneous devices")
    train_parser.add_argument("--dataset", choices=["adult", "synthetic"], default="adult")
    train_parser.add_argument("--N", type=int, default=5, help="Number of devices")
    train_parser.add_argument("--non-iid", choices=["iid", "label-skew", "covariate-shift", "feature-shift"], default="label-skew")
    train_parser.add_argument("--out", required=True, help="Output directory for devices")
    train_parser.add_argument("--seed", type=int, default=42)

    # Probe
    probe_parser = subparsers.add_parser("probe", help="Compute probe trajectories")
    probe_parser.add_argument("--devices", required=True, help="Directory with trained devices")
    probe_parser.add_argument("--dataset", choices=["adult", "synthetic"], default="adult")
    probe_parser.add_argument("--probes", type=int, default=500, help="Number of probe samples")
    probe_parser.add_argument("--theta0", default="[0.0,0.5,1.0]", help="Pre-window intensities")
    probe_parser.add_argument("--interventions", help="Interventions config file")
    probe_parser.add_argument("--out", required=True, help="Output directory")
    probe_parser.add_argument("--dp-noise-std", type=float, default=0.0, help="DP noise level")
    probe_parser.add_argument("--seed", type=int, default=42)

    # Explain
    explain_parser = subparsers.add_parser("explain", help="Run DISCO explanations")
    explain_parser.add_argument("--devices", required=True, help="Directory with trained devices")
    explain_parser.add_argument("--probes", required=True, help="Directory with probe trajectories")
    explain_parser.add_argument("--dataset", choices=["adult", "synthetic"], default="adult")
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

    # Evaluate
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate results")
    evaluate_parser.add_argument("--results", required=True, help="Results directory")
    evaluate_parser.add_argument("--figs", help="Output directory for figures")

    # Synth
    synth_parser = subparsers.add_parser("synth", help="Run synthetic study")
    synth_parser.add_argument("--p", type=int, default=8, help="Number of features")
    synth_parser.add_argument("--N", type=int, default=10, help="Number of devices")
    synth_parser.add_argument("--m", type=int, default=1000, help="Number of probes")
    synth_parser.add_argument("--alpha", type=float, default=0.7, help="Treatment strength")
    synth_parser.add_argument("--out", required=True, help="Output directory")
    synth_parser.add_argument("--seed", type=int, default=42)

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
