"""Active sensing: choose *when* to take a limited number of fingerstick readings.

For a budget of k strips, pick measurement times S maximising the weighted mutual information

    J(S) = sum_w  weight_w * I(Y_S ; E_w)

between the (noisy) readings Y_S and the binary events E_w "hypoglycaemia occurs in risk window
w". Everything is estimated from the twin's posterior-predictive ensemble:

    I(Y_S; E_w) = H(E_w) - E_{y_S}[ H(E_w | y_S) ]

where the inner posterior over trajectories is obtained by re-weighting ensemble members with
the reading likelihood (nested Monte Carlo). Sets are built greedily, the standard near-optimal
strategy for monotone information objectives. In deployment the plan is recomputed after every
reading, so each strip is chosen with everything learned so far.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from pratibimb.physiology.inputs import GridInputs
from pratibimb.twin.events import RiskWindow, window_events
from pratibimb.twin.particle_filter import SMBG_OBS, ObservationModel, ParticleFilter

LN2 = float(np.log(2.0))


def binary_entropy_bits(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p)) / LN2


@dataclass(frozen=True)
class PlannedReading:
    t_min: float
    gain_bits: float  # marginal information gain when added to the plan


@dataclass
class SensingPlan:
    readings: list[PlannedReading]
    total_gain_bits: float
    prior_risk: dict[str, float]  # P(event) per window before any new reading
    expected_residual_entropy_bits: dict[str, float]
    candidate_gain_bits: dict[float, float] = field(default_factory=dict)  # single-reading value

    @property
    def times(self) -> list[float]:
        return [r.t_min for r in self.readings]


@dataclass(frozen=True)
class PlannerConfig:
    n_ensemble: int = 400
    n_truths: int = 200
    min_spacing_min: float = 60.0
    min_gain_bits: float = 1e-3


def plan_readings(
    pf: ParticleFilter,
    future: GridInputs,
    windows: Sequence[RiskWindow],
    candidate_times: Sequence[float],
    budget: int,
    obs: ObservationModel = SMBG_OBS,
    config: PlannerConfig | None = None,
    rng: np.random.Generator | None = None,
    already_planned: Sequence[float] = (),
    precede_windows: bool = True,
) -> SensingPlan:
    """Greedy information-maximising schedule of up to ``budget`` readings.

    ``future`` must start at the filter's current time and cover every window and candidate.
    ``already_planned`` readings (e.g. a mandatory fasting check) count toward spacing only.
    With ``precede_windows`` (anticipation mode) a reading only informs windows whose risk call
    is made at or after the reading, e.g. the bedtime decision on overnight risk.
    """
    cfg = config or PlannerConfig()
    rng = rng or np.random.default_rng(0)
    dt = future.dt
    traj = pf.forecast(future, n_samples=cfg.n_ensemble, rng=rng)  # (M, T+1)
    m = traj.shape[0]

    events = np.stack([window_events(traj, pf.t, dt, w) for w in windows], axis=1).astype(float)
    weights = np.array([w.weight for w in windows])
    labels = [w.label or f"{w.kind}@{w.start_min:.0f}" for w in windows]
    p0 = events.mean(axis=0)
    h0 = binary_entropy_bits(p0)

    cand = np.array(sorted(t for t in candidate_times if pf.t - 1e-9 <= t <= pf.t + dt * future.n_steps))
    empty = SensingPlan(
        [], 0.0, dict(zip(labels, p0.tolist(), strict=True)), dict(zip(labels, h0.tolist(), strict=True))
    )
    if budget <= 0 or cand.size == 0 or weights @ h0 < cfg.min_gain_bits:
        return empty
    cidx = np.rint((cand - pf.t) / dt).astype(int)
    g_cand = traj[:, cidx]  # (M, C)

    # Nested Monte Carlo: J "true" futures, each producing a noisy reading at every candidate.
    truths = rng.choice(m, size=min(cfg.n_truths, m), replace=False)
    y = g_cand[truths] + obs.sd(g_cand[truths]) * rng.standard_normal((truths.size, cand.size))
    # loglik[c, j, i] = log p(y_jc | trajectory i)
    loglik = np.stack([obs.log_likelihood(y[:, c][:, None], g_cand[None, :, c]) for c in range(cand.size)])

    n_w = len(windows)
    decision = np.array([w.decision_time for w in windows])
    valid = (
        (cand[:, None] <= decision[None, :] + 1e-9) if precede_windows else np.ones((cand.size, n_w), bool)
    )
    validf = valid.astype(float)

    def expected_entropy(acc: np.ndarray) -> np.ndarray:
        """acc: (C, W, J, M) accumulated log-likelihood -> expected posterior entropy (C, W)."""
        logw = acc - acc.max(axis=3, keepdims=True)
        w = np.exp(logw)
        w /= w.sum(axis=3, keepdims=True)
        post = np.einsum("cwjm,mw->cwj", w, events)
        return binary_entropy_bits(post).mean(axis=2)

    def with_candidates(acc: np.ndarray, cs: np.ndarray) -> np.ndarray:
        return acc[None] + validf[cs][:, :, None, None] * loglik[cs][:, None, :, :]

    acc = np.zeros((n_w, truths.size, m))
    all_c = np.arange(cand.size)
    single = expected_entropy(with_candidates(acc, all_c))
    single_gain = (h0[None, :] - single) @ weights

    chosen: list[int] = []
    gains: list[float] = []
    current_obj = float(weights @ h0)
    blocked = ~valid.any(axis=1)  # readings too late to inform any decision
    for t_fixed in already_planned:
        blocked |= np.abs(cand - t_fixed) < cfg.min_spacing_min
    for _ in range(budget):
        avail = np.flatnonzero(~blocked)
        if avail.size == 0:
            break
        obj = expected_entropy(with_candidates(acc, avail)) @ weights
        best_local = int(np.argmin(obj))
        gain = current_obj - float(obj[best_local])
        if gain < cfg.min_gain_bits:
            break
        best = int(avail[best_local])
        chosen.append(best)
        gains.append(gain)
        current_obj = float(obj[best_local])
        acc = acc + validf[best][:, None, None] * loglik[best][None]
        blocked |= np.abs(cand - cand[best]) < cfg.min_spacing_min

    residual = expected_entropy(acc[None])[0] if chosen else h0
    readings = sorted(
        (PlannedReading(float(cand[c]), g) for c, g in zip(chosen, gains, strict=True)),
        key=lambda r: r.t_min,
    )
    return SensingPlan(
        readings=readings,
        total_gain_bits=float(sum(gains)),
        prior_risk=dict(zip(labels, p0.tolist(), strict=True)),
        expected_residual_entropy_bits=dict(zip(labels, residual.tolist(), strict=True)),
        candidate_gain_bits={float(t): float(g) for t, g in zip(cand, single_gain, strict=True)},
    )
