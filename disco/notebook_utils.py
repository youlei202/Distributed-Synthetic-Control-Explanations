"""Notebook helpers to compute DISCO treatment effects."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import pickle
import numpy as np

from disco.anchors.knn import AnchorSelector
from disco.counterfactual.pmode import PModeCounterfactual
from disco.counterfactual.smode import SModeCounterfactual
from disco.devices.base import Device
from disco.explain.te_curve import Explainer, TEOutput
from disco.interventions.base import Intervention
from disco.interventions.grids import Grids
from disco.interventions.tabular import FeatureAdd
from disco.matching.weights import WeightSolver
from disco.cli.main import run_disco_pipeline, compute_probe_trajectories
from disco.synth.generator import SyntheticScenario


__all__ = [
    "NotebookArtifacts",
    "load_notebook_bundle",
    "build_default_grids",
    "compute_te_output",
    "load_scenario_artifacts",
    "compute_te_output_from_scenario",
]


@dataclass
class NotebookArtifacts:
    devices: List[Device]
    trajectories: Dict[Tuple[str, str], np.ndarray]
    X_probe: np.ndarray
    theta0: np.ndarray
    metadata: Dict[str, Any]


def load_notebook_bundle(devices_dir: str, probes_dir: str) -> NotebookArtifacts:
    devices_path = Path(devices_dir)
    probes_path = Path(probes_dir)

    if not devices_path.exists():
        raise FileNotFoundError(f"Devices directory not found: {devices_path}")
    if not probes_path.exists():
        raise FileNotFoundError(f"Probe directory not found: {probes_path}")

    devices = _load_devices(devices_path)
    X_probe = np.load(probes_path / "X_probe.npy")

    meta_path = probes_path / "probe_meta.json"
    if meta_path.exists():
        with meta_path.open() as f:
            metadata = json.load(f)
    else:
        metadata = {}

    theta0 = np.array(metadata.get("theta0", []), dtype=float)
    trajectories = _load_probe_trajectories(probes_path)

    return NotebookArtifacts(
        devices=devices,
        trajectories=trajectories,
        X_probe=X_probe,
        theta0=theta0,
        metadata=metadata,
    )


def build_default_grids(
    theta0: np.ndarray,
    *,
    n_post: int = 6,
    gap: float = 1.0,
    step: Optional[float] = None,
    theta_max: Optional[float] = None,
) -> Grids:
    theta0 = np.asarray(theta0, dtype=float)
    base = float(theta0[-1]) if theta0.size else 0.0

    if step is None:
        if theta0.size >= 2:
            diffs = np.diff(theta0)
            diffs = diffs[diffs > 0]
            if diffs.size:
                step = float(np.median(diffs))
            else:
                step = max(abs(base), 1.0) * 0.1
        else:
            scale = abs(base) if base != 0 else 1.0
            step = 0.1 * scale
        if step == 0:
            step = 0.5

    start = base + gap * step
    if theta_max is None:
        theta_max = start + step * max(n_post - 1, 1)

    theta = np.linspace(start, theta_max, n_post, dtype=float)
    return Grids(theta0=theta0, theta=theta)


def compute_te_output(
    bundle: NotebookArtifacts,
    *,
    target_device_index: int,
    x_star: np.ndarray,
    intervention: Intervention,
    grids: Optional[Grids] = None,
    mode: str = "p",
    anchor_selector: Optional[AnchorSelector] = None,
    weight_solver: Optional[WeightSolver] = None,
    counterfactual_builder: Optional[object] = None,
    explainer: Optional[Explainer] = None,
    target_class: int = 1,
    lam: float = 0.01,
) -> Tuple[TEOutput, Dict[str, Any]]:
    if grids is None:
        grids = build_default_grids(bundle.theta0)

    if anchor_selector is None:
        anchor_selector = AnchorSelector()
    if weight_solver is None:
        weight_solver = WeightSolver(method="projected_grad", max_iter=1000)

    if counterfactual_builder is None:
        if mode == "p":
            counterfactual_builder = PModeCounterfactual()
        elif mode == "s":
            counterfactual_builder = SModeCounterfactual()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    if explainer is None:
        explainer = Explainer()

    target_device = bundle.devices[target_device_index]

    x_star_array = np.asarray(x_star)
    if x_star_array.ndim == 1:
        x_star_array = x_star_array.reshape(1, -1)

    key = (target_device.id, intervention.name)
    if key not in bundle.trajectories:
        available = sorted({name for _, name in bundle.trajectories})
        raise KeyError(
            f"Missing trajectories for intervention '{intervention.name}'. "
            f"Available interventions: {available}"
        )

    result = run_disco_pipeline(
        target_device=target_device,
        devices=bundle.devices,
        x_star=x_star_array,
        X_probe=bundle.X_probe,
        intervention=intervention,
        grids=grids,
        anchor_selector=anchor_selector,
        weight_solver=weight_solver,
        counterfactual_builder=counterfactual_builder,
        explainer=explainer,
        trajectories=bundle.trajectories,
        target_class=target_class,
        mode=mode,
        lam=lam,
    )

    return result["te_output"], result


def load_scenario_artifacts(scenario_dir: str) -> Tuple[SyntheticScenario, Dict[Tuple[str, str], np.ndarray]]:
    scenario_path = Path(scenario_dir)
    scenario = SyntheticScenario.load(scenario_path / "scenario.npz")
    trajectories = _load_probe_trajectories(scenario_path / "probes")
    return scenario, trajectories


def compute_te_output_from_scenario(
    scenario: SyntheticScenario,
    *,
    intervention: Optional[Intervention] = None,
    trajectories: Optional[Dict[Tuple[str, str], np.ndarray]] = None,
    grids: Optional[Grids] = None,
    mode: str = "p",
    anchor_selector: Optional[AnchorSelector] = None,
    weight_solver: Optional[WeightSolver] = None,
    counterfactual_builder: Optional[object] = None,
    explainer: Optional[Explainer] = None,
    lam: float = 0.01,
) -> Tuple[TEOutput, Dict[str, Any]]:
    if intervention is None:
        intervention = FeatureAdd(feat_idx=0, delta_max=float(scenario.theta.max()), name="intensity_shift")

    if grids is None:
        grids = Grids(theta0=scenario.theta0, theta=scenario.theta)

    if anchor_selector is None:
        anchor_selector = AnchorSelector()
    if weight_solver is None:
        weight_solver = WeightSolver(method="projected_grad", max_iter=1000)

    if counterfactual_builder is None:
        counterfactual_builder = PModeCounterfactual() if mode == "p" else SModeCounterfactual()
    if explainer is None:
        explainer = Explainer()

    if trajectories is None:
        trajectories = compute_probe_trajectories(
            devices=scenario.devices,
            X_probe=scenario.X_probe,
            interventions=[intervention],
            theta0=scenario.theta0,
            dp_noise_std=0.0,
            seed=0,
            target_class=0,
        )

    result = run_disco_pipeline(
        target_device=scenario.devices[0],
        devices=scenario.devices,
        x_star=scenario.x_star,
        X_probe=scenario.X_probe,
        intervention=intervention,
        grids=grids,
        anchor_selector=anchor_selector,
        weight_solver=weight_solver,
        counterfactual_builder=counterfactual_builder,
        explainer=explainer,
        trajectories=trajectories,
        target_class=0,
        mode=mode,
        lam=lam,
    )

    return result["te_output"], result


def _load_devices(devices_path: Path) -> List[Device]:
    device_files = sorted(devices_path.glob("device_*.pkl"))
    if not device_files:
        raise FileNotFoundError(f"No device_*.pkl files found in {devices_path}")

    devices: List[Device] = []
    for file_path in device_files:
        with file_path.open("rb") as f:
            devices.append(pickle.load(f))  # type: ignore[arg-type]
    return devices


def _load_probe_trajectories(probes_path: Path) -> Dict[Tuple[str, str], np.ndarray]:
    trajectories: Dict[Tuple[str, str], np.ndarray] = {}

    for npy_path in probes_path.glob("*.npy"):
        if npy_path.name == "X_probe.npy":
            continue

        parts = npy_path.stem.split("_", 2)
        if len(parts) < 3:
            continue

        device_id = f"{parts[0]}_{parts[1]}"
        interv_name = "_".join(parts[2:])
        trajectories[(device_id, interv_name)] = np.load(npy_path)

    if not trajectories:
        raise FileNotFoundError(
            f"No trajectory matrices found in {probes_path}. Did you run the probe command?"
        )

    return trajectories
