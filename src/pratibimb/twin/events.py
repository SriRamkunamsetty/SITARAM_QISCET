"""Clinical event definitions evaluated on (ensembles of) glucose trajectories.

Hypoglycaemia follows the international CGM consensus (Battelino et al., Diabetes Care 2019):
level 1 < 70 mg/dL and level 2 < 54 mg/dL, sustained for >= 15 minutes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class EventKind(StrEnum):
    HYPO = "hypo"
    HYPER = "hyper"


@dataclass(frozen=True)
class RiskWindow:
    """A future interval [start, end) in twin minutes whose event probability we care about."""

    start_min: float
    end_min: float
    kind: EventKind = EventKind.HYPO
    threshold: float = 70.0
    min_duration_min: float = 15.0
    weight: float = 1.0
    label: str = ""
    decision_min: float | None = None  # when the risk call is made (defaults to start_min)

    @property
    def decision_time(self) -> float:
        return self.start_min if self.decision_min is None else self.decision_min


def sustained_crossing(traj: np.ndarray, threshold: float, min_steps: int, below: bool = True) -> np.ndarray:
    """True for rows of traj that stay beyond threshold for >= min_steps consecutive samples."""
    hit = traj < threshold if below else traj > threshold
    if min_steps <= 1:
        return np.asarray(hit.any(axis=1))
    if hit.shape[1] < min_steps:
        return np.zeros(hit.shape[0], dtype=bool)
    run = hit[:, : hit.shape[1] - min_steps + 1].copy()
    for k in range(1, min_steps):
        run &= hit[:, k : hit.shape[1] - min_steps + 1 + k]
    return np.asarray(run.any(axis=1))


def window_events(traj: np.ndarray, t0_min: float, dt: float, window: RiskWindow) -> np.ndarray:
    """Event indicator per trajectory for one window; traj[:, k] is glucose at t0 + k*dt."""
    k0 = max(int(np.ceil((window.start_min - t0_min) / dt - 1e-9)), 0)
    k1 = min(int(np.ceil((window.end_min - t0_min) / dt - 1e-9)), traj.shape[1])
    if k1 <= k0:
        return np.zeros(traj.shape[0], dtype=bool)
    min_steps = max(round(window.min_duration_min / dt), 1)
    return sustained_crossing(
        traj[:, k0:k1], window.threshold, min_steps, below=window.kind is EventKind.HYPO
    )
