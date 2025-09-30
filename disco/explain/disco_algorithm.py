"""DISCO algorithm helpers for consistency with baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from disco.anchors.knn import AnchorSelector
from disco.counterfactual.pmode import PModeCounterfactual
from disco.counterfactual.smode import SModeCounterfactual
from disco.devices.base import Device
from disco.explain.te_curve import TEOutput, Explainer
from disco.interventions.base import Intervention
from disco.interventions.grids import Grids
from disco.matching.weights import WeightSolver
from disco.cli.main import run_disco_pipeline


@dataclass
class DiscoOutput:
    """Wrapper for DISCO results to match baseline API."""

    te_output: TEOutput
    weights: np.ndarray
    diagnostics: Dict[str, Any]


class DiscoRunner:
    """Convenience runner for DISCO treatment-effect computation."""

    def __init__(
        self,
        mode: str = "p",
        lam: float = 0.01,
        anchor_selector: Optional[AnchorSelector] = None,
        weight_solver: Optional[WeightSolver] = None,
        counterfactual_builder: Optional[object] = None,
        explainer: Optional[Explainer] = None,
        anchor_space: str = "response",
    ) -> None:
        self.default_mode = mode
        self.default_lam = lam
        self.default_anchor_selector = anchor_selector
        self.default_weight_solver = weight_solver
        self.default_counterfactual_builder = counterfactual_builder
        self.default_explainer = explainer
        self.default_anchor_space = anchor_space

    def run(
        self,
        devices: List[Device],
        target_idx: int,
        x_star: np.ndarray,
        X_probe: np.ndarray,
        trajectories: Dict,
        grids: Grids,
        intervention: Intervention,
        *,
        target_class: int = 1,
        mode: Optional[str] = None,
        lam: Optional[float] = None,
        anchor_selector: Optional[AnchorSelector] = None,
        weight_solver: Optional[WeightSolver] = None,
        counterfactual_builder: Optional[object] = None,
        explainer: Optional[Explainer] = None,
        anchor_space: Optional[str] = None,
    ) -> DiscoOutput:
        """Run DISCO with explicit inputs."""
        if x_star.ndim == 1:
            x_star = x_star.reshape(1, -1)

        runner_mode = mode or self.default_mode
        runner_lam = self.default_lam if lam is None else lam
        anchor = anchor_selector or self.default_anchor_selector or AnchorSelector()
        solver = weight_solver or self.default_weight_solver or WeightSolver(method="projected_grad", max_iter=1000)
        if counterfactual_builder is None:
            if self.default_counterfactual_builder is not None:
                builder = self.default_counterfactual_builder
            else:
                builder = PModeCounterfactual() if runner_mode == "p" else SModeCounterfactual()
        else:
            builder = counterfactual_builder
        expl = explainer or self.default_explainer or Explainer()

        result = run_disco_pipeline(
            target_device=devices[target_idx],
            devices=devices,
            x_star=x_star,
            X_probe=X_probe,
            intervention=intervention,
            grids=grids,
            anchor_selector=anchor,
            weight_solver=solver,
            counterfactual_builder=builder,
            explainer=expl,
            trajectories=trajectories,
            target_class=target_class,
            mode=runner_mode,
            lam=runner_lam,
            anchor_space=anchor_space or self.default_anchor_space,
        )

        diagnostics = {k: v for k, v in result.items() if k not in ("te_output", "weights")}
        return DiscoOutput(te_output=result["te_output"], weights=result["weights"], diagnostics=diagnostics)


__all__ = ["DiscoRunner", "DiscoOutput"]
