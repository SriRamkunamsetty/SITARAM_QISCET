"""Fingerstick scheduling policies compared in the strip-budget experiment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Structured SMBG schedules used in routine Indian practice (FBS / PPBS-centred), by budget.
FIXED_SCHEDULES_HHMM: dict[int, tuple[str, ...]] = {
    1: ("07:00",),
    2: ("07:00", "10:30"),
    3: ("07:00", "10:30", "15:30"),
    4: ("07:00", "10:30", "15:30", "22:30"),
    5: ("07:00", "10:30", "15:30", "20:30", "22:30"),
    6: ("07:00", "10:30", "13:00", "15:30", "20:30", "22:30"),
}


def waking_slots(start_hhmm: str = "06:30", end_hhmm: str = "22:30", step_min: int = 30) -> list[float]:
    """Minutes-of-day at which a waking patient can be asked to test."""
    h0, m0 = (int(x) for x in start_hhmm.split(":"))
    h1, m1 = (int(x) for x in end_hhmm.split(":"))
    return [float(t) for t in range(h0 * 60 + m0, h1 * 60 + m1 + 1, step_min)]


def fixed_schedule(budget: int) -> list[float]:
    if budget not in FIXED_SCHEDULES_HHMM:
        raise ValueError(f"no fixed schedule for budget {budget}")
    out = []
    for hhmm in FIXED_SCHEDULES_HHMM[budget]:
        h, m = (int(x) for x in hhmm.split(":"))
        out.append(float(h * 60 + m))
    return out


def random_schedule(
    budget: int, slots: Sequence[float], rng: np.random.Generator, min_spacing_min: float = 60.0
) -> list[float]:
    """Uniformly random waking-hour times with a minimum spacing (rejection sampling)."""
    slots_arr = np.asarray(slots)
    for _ in range(1000):
        pick = np.sort(rng.choice(slots_arr, size=budget, replace=False))
        if budget <= 1 or np.min(np.diff(pick)) >= min_spacing_min:
            return pick.tolist()
    raise RuntimeError("could not satisfy spacing constraint")


@dataclass(frozen=True)
class PolicySpec:
    name: str  # "none" | "fixed" | "random" | "twin" | "cgm"
    budget: int = 0

    @property
    def key(self) -> str:
        return self.name if self.name in {"none", "cgm"} else f"{self.name}-{self.budget}"
