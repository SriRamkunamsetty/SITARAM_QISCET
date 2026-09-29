from __future__ import annotations

import numpy as np
import pytest

from pratibimb.physiology.drugs import su_units
from pratibimb.physiology.inputs import ActivityBout, EventLog, FoodOrder, Meal, MedicationDose, discretise
from pratibimb.physiology.model import (
    DEFAULT_CONSTANTS,
    N_PARAMS,
    G,
    PatientConstants,
    apply_impulses,
    rk4_step,
    steady_state,
)

PC = PatientConstants(body_weight_kg=65, on_metformin=False)


def theta(n: int = 1, **over: float) -> np.ndarray:
    base = {"si": 2.5e-4, "sg": 0.012, "phi": 0.07, "gb": 140.0, "kge": 0.016, "ksu": 2.0}
    base.update(over)
    return np.tile([base[k] for k in ("si", "sg", "phi", "gb", "kge", "ksu")], (n, 1))


def simulate(th: np.ndarray, events: EventLog, minutes: int = 600, t0_tod: float = 420.0) -> np.ndarray:
    dt = 5.0
    n = minutes // 5
    grid = discretise(events, 0.0, n, dt)
    y = steady_state(th, DEFAULT_CONSTANTS, PC)
    out = np.empty((th.shape[0], n))
    for k in range(n):
        out[:, k] = y[:, G]
        apply_impulses(y, grid.carbs_fast_mg[k], grid.carbs_slow_mg[k], grid.su_units[k])
        y = rk4_step(y, th, (t0_tod + k * dt) % 1440, grid.activity[k], dt, DEFAULT_CONSTANTS, PC)
    return out


def test_fasting_is_stable_and_finite() -> None:
    g = simulate(theta(), EventLog(), minutes=1440)[0]
    assert np.all(np.isfinite(g))
    assert g.min() > 120 and g.max() < 170  # basal 140 with dawn/circadian modulation


def test_meal_raises_then_returns() -> None:
    g = simulate(theta(), EventLog(meals=[Meal(0.0, 60.0, 75.0)]), minutes=480)[0]
    peak_k = int(np.argmax(g))
    assert 40 <= g.max() - g[0] <= 250
    assert 30 <= peak_k * 5 <= 180
    assert abs(g[-1] - g[0]) < 35


def test_sulfonylurea_lowers_glucose() -> None:
    base = simulate(theta(), EventLog(), minutes=480)[0]
    su = simulate(theta(), EventLog(medications=[MedicationDose(0.0, "glimepiride", 2.0)]), minutes=480)[0]
    assert su.min() < base.min() - 15


def test_carbs_last_blunts_early_excursion() -> None:
    mixed = simulate(theta(), EventLog(meals=[Meal(0.0, 70.0, 75.0)]), minutes=180)[0]
    last = simulate(theta(), EventLog(meals=[Meal(0.0, 70.0, 75.0, FoodOrder.CARBS_LAST)]), minutes=180)[0]
    assert last[12] < mixed[12]  # 60 min


def test_walk_after_meal_lowers_peak() -> None:
    meal = Meal(0.0, 70.0, 75.0)
    still = simulate(theta(), EventLog(meals=[meal]), minutes=240)[0]
    walk = simulate(theta(), EventLog(meals=[meal], activity=[ActivityBout(15, 45, 0.8)]), minutes=240)[0]
    assert walk.max() < still.max()


def test_vectorised_particles_are_independent_and_consistent() -> None:
    th = theta(3)
    th[2, 0] = 5e-4  # third particle: more insulin sensitive
    g = simulate(th, EventLog(meals=[Meal(0.0, 60.0)]), minutes=300)
    np.testing.assert_allclose(g[0], g[1])
    assert g[2].max() < g[0].max()
    assert th.shape[1] == N_PARAMS


def test_discretise_places_events_on_grid() -> None:
    ev = EventLog(meals=[Meal(12.0, 50.0)], activity=[ActivityBout(7.5, 12.5, 1.0)])
    grid = discretise(ev, 0.0, 6, 5.0)
    assert grid.carbs_fast_mg[2] + grid.carbs_slow_mg[2] == pytest.approx(50_000)
    np.testing.assert_allclose(grid.activity[:4], [0.0, 0.5, 0.5, 0.0])


def test_su_equivalents() -> None:
    assert su_units("Glimepiride", 2) == 2
    assert su_units("gliclazide", 80) == 2
    assert su_units("metformin", 1000) == 0
    with pytest.raises(ValueError):
        su_units("glimepiride", -1)
