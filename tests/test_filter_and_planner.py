from __future__ import annotations

import itertools

import numpy as np
import pytest

from pratibimb.cohort.endotypes import build_prior
from pratibimb.cohort.routine import routine_events
from pratibimb.cohort.virtual import cgm_trace, generate_day, sample_virtual_patient, simulate_truth
from pratibimb.physiology.inputs import EventLog, discretise
from pratibimb.physiology.model import PatientConstants
from pratibimb.sensing.planner import PlannerConfig, binary_entropy_bits, plan_readings
from pratibimb.twin.events import RiskWindow, sustained_crossing
from pratibimb.twin.particle_filter import FilterConfig, ParticleFilter


@pytest.fixture(scope="module")
def scenario():
    rng = np.random.default_rng(11)
    vp = sample_virtual_patient(rng, "T1", force_sulfonylurea=True, calibration_days=2)
    days = [generate_day(vp, d, rng) for d in range(2)]
    truth = simulate_truth(vp, days, rng)
    logged = EventLog()
    for d in days:
        logged.extend(d.logged_events)
    return vp, logged, truth


def _filter(vp, n=300, seed=1) -> ParticleFilter:
    return ParticleFilter(
        build_prior(vp.profile),
        PatientConstants(vp.profile.body_weight_kg, vp.profile.on_metformin),
        FilterConfig(n_particles=n),
        seed=seed,
    )


def _track(vp, logged, truth, readings) -> float:
    pf = _filter(vp)
    errs = []
    read = dict(readings)
    for k in range(len(truth.t_min) - 1):
        t = truth.t_min[k]
        if t in read:
            pf.assimilate(read[t], "cgm")
        errs.append(abs(pf.glucose_mean() - truth.glucose[k]))
        grid = discretise(logged.window(t, t + 5), t, 1, 5.0)
        pf.step(grid.carbs_fast_mg[0], grid.carbs_slow_mg[0], grid.su_units[0], grid.activity[0])
    return float(np.mean(errs[288:]))  # score day 2


def test_cgm_assimilation_beats_prior_only(scenario) -> None:
    vp, logged, truth = scenario
    t, y = cgm_trace(truth, np.random.default_rng(3))
    with_cgm = _track(vp, logged, truth, zip(t, y, strict=True))
    prior_only = _track(vp, logged, truth, [])
    assert with_cgm < prior_only
    assert with_cgm < 20.0


def test_rejects_implausible_reading(scenario) -> None:
    pf = _filter(scenario[0], n=200)
    with pytest.raises(ValueError):
        pf.assimilate(5.0)


def test_heavy_tailed_likelihood_limits_outlier_influence(scenario) -> None:
    pf = _filter(scenario[0], n=500)
    before = pf.glucose_mean()
    pf.assimilate(before + 300.0)  # e.g. sugary fingertip
    assert abs(pf.glucose_mean() - before) < 0.6 * 300.0


def test_binary_entropy_bounds() -> None:
    h = binary_entropy_bits(np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(h, [0.0, 1.0, 0.0], atol=1e-6)


def test_sustained_crossing_requires_duration() -> None:
    traj = np.array([[100, 65, 65, 100, 100], [100, 65, 65, 65, 100]], dtype=float)
    np.testing.assert_array_equal(sustained_crossing(traj, 70, 3), [False, True])


def test_planner_respects_budget_spacing_and_candidates(scenario) -> None:
    vp, _, _ = scenario
    pf = _filter(vp)
    horizon = 1440.0
    grid = discretise(routine_events(vp.profile, 0.0, horizon), 0.0, int(horizon / 5), 5.0)
    windows = [
        RiskWindow(390, 720, label="morning"),
        RiskWindow(1080, 1440, label="evening", weight=2.0),
    ]
    cands = [float(t) for t in range(360, 1350, 30)]
    plan = plan_readings(
        pf, grid, windows, cands, budget=3, config=PlannerConfig(n_ensemble=200, n_truths=100)
    )
    times = plan.times
    assert len(times) <= 3
    assert all(t in cands for t in times)
    assert all(b - a >= 60 for a, b in itertools.pairwise(times))
    assert all(r.gain_bits >= 0 for r in plan.readings)
    # anticipation mode: every reading precedes at least one decision it informs
    assert all(t <= 1080 for t in times)


def test_planner_returns_nothing_when_outcome_is_certain(scenario) -> None:
    vp, _, _ = scenario
    pf = _filter(vp, n=200)
    grid = discretise(EventLog(), 0.0, 100, 5.0)
    impossible = RiskWindow(0, 500, threshold=0.0, label="glucose below zero")
    plan = plan_readings(pf, grid, [impossible], [0.0, 60.0, 120.0], budget=2)
    assert plan.readings == []
    assert plan.prior_risk["glucose below zero"] == 0.0
