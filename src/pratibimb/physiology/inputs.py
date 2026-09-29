"""Exogenous inputs to the twin (meals, medications, activity) and their discretisation.

Times are expressed in minutes relative to the twin's anchor time. The engine layer converts
wall-clock datetimes to this representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from pratibimb.physiology.drugs import su_units


class FoodOrder(StrEnum):
    MIXED = "mixed"
    CARBS_LAST = "carbs_last"  # vegetables/protein first, carbohydrate last


@dataclass(frozen=True)
class Meal:
    t_min: float
    carbs_g: float
    glycemic_index: float = 70.0
    food_order: FoodOrder = FoodOrder.MIXED
    label: str = ""

    def split_mg(self) -> tuple[float, float]:
        """Split carbohydrate (mg) into fast- and slow-absorbing fractions.

        High-GI carbohydrate goes to the fast chain. Eating carbohydrate last slows gastric
        emptying (meta-analysis: t1/2 +28 min), modelled by moving a further 30% to the slow chain.
        """
        fast_frac = float(np.clip((self.glycemic_index - 35.0) / 60.0, 0.1, 0.95))
        if self.food_order is FoodOrder.CARBS_LAST:
            fast_frac *= 0.7
        total = self.carbs_g * 1000.0
        return total * fast_frac, total * (1.0 - fast_frac)


@dataclass(frozen=True)
class MedicationDose:
    t_min: float
    drug: str
    dose_mg: float

    @property
    def su_units(self) -> float:
        return su_units(self.drug, self.dose_mg)


@dataclass(frozen=True)
class ActivityBout:
    t_start_min: float
    t_end_min: float
    intensity: float  # 0..1 (1 ~ brisk walking, ~110 steps/min)


@dataclass
class EventLog:
    meals: list[Meal] = field(default_factory=list)
    medications: list[MedicationDose] = field(default_factory=list)
    activity: list[ActivityBout] = field(default_factory=list)

    def extend(self, other: EventLog) -> None:
        self.meals.extend(other.meals)
        self.medications.extend(other.medications)
        self.activity.extend(other.activity)

    def window(self, t0: float, t1: float) -> EventLog:
        return EventLog(
            meals=[m for m in self.meals if t0 <= m.t_min < t1],
            medications=[d for d in self.medications if t0 <= d.t_min < t1],
            activity=[a for a in self.activity if a.t_end_min > t0 and a.t_start_min < t1],
        )


@dataclass(frozen=True)
class GridInputs:
    """Inputs discretised on a uniform grid starting at t0 with step dt."""

    t0_min: float
    dt: float
    carbs_fast_mg: np.ndarray
    carbs_slow_mg: np.ndarray
    su_units: np.ndarray
    activity: np.ndarray
    meal_index: np.ndarray  # -1 where no meal starts, else index into the meals list

    @property
    def n_steps(self) -> int:
        return int(self.activity.shape[0])


def discretise(events: EventLog, t0_min: float, n_steps: int, dt: float) -> GridInputs:
    """Map events to grid steps. An event at time t is applied at the start of its step."""
    fast = np.zeros(n_steps)
    slow = np.zeros(n_steps)
    su = np.zeros(n_steps)
    act = np.zeros(n_steps)
    meal_index = np.full(n_steps, -1, dtype=np.int64)
    t_end = t0_min + n_steps * dt

    def step_of(t: float) -> int:
        return int(np.floor((t - t0_min) / dt + 1e-9))

    for idx, meal in enumerate(events.meals):
        if t0_min <= meal.t_min < t_end:
            k = step_of(meal.t_min)
            f, s = meal.split_mg()
            fast[k] += f
            slow[k] += s
            meal_index[k] = idx
    for dose in events.medications:
        if t0_min <= dose.t_min < t_end:
            su[step_of(dose.t_min)] += dose.su_units
    grid_starts = t0_min + dt * np.arange(n_steps)
    for bout in events.activity:
        overlap = np.clip(
            np.minimum(grid_starts + dt, bout.t_end_min) - np.maximum(grid_starts, bout.t_start_min),
            0.0,
            dt,
        )
        act += bout.intensity * overlap / dt
    np.clip(act, 0.0, 1.0, out=act)
    return GridInputs(t0_min, dt, fast, slow, su, act, meal_index)
