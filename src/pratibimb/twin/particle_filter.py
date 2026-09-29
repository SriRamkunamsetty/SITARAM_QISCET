"""Joint state-and-parameter particle filter: the data-assimilation core of the twin.

Each particle is one hypothesis about the patient: a physiological state (glucose, insulin
action, gut contents, drug on board) *and* a parameter vector (insulin sensitivity, beta-cell
function, basal glucose, gastric emptying, SU response, glucose effectiveness). Glucose readings
reweight the particles; resampling plus Liu–West kernel shrinkage keeps parameter diversity.
Slow random walks on S_I, basal glucose and SU response let the twin follow day-to-day drift.

Works with dense CGM or sparse fingersticks (SMBG): sparse data simply leaves the posterior wider.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from pratibimb.cohort.endotypes import ParameterPrior
from pratibimb.physiology.inputs import GridInputs
from pratibimb.physiology.model import (
    DEFAULT_CONSTANTS,
    GB,
    KSU,
    MINUTES_PER_DAY,
    N_PARAMS,
    PARAM_NAMES,
    SI,
    G,
    ModelConstants,
    PatientConstants,
    apply_impulses,
    rk4_step,
    steady_state,
)


@dataclass(frozen=True)
class ObservationModel:
    """Likelihood for a glucose reading: Student-t with sd = sqrt(abs^2 + (rel*G)^2).

    The sd is deliberately larger than the device error to absorb model discrepancy, and the
    heavy tail makes the twin robust to occasional bad readings (e.g. unwashed fingers).
    """

    abs_sd: float
    rel_sd: float
    nu: float = 4.0

    def sd(self, g: np.ndarray) -> np.ndarray:
        return np.sqrt(self.abs_sd**2 + (self.rel_sd * g) ** 2)

    def log_likelihood(self, y: float | np.ndarray, g: np.ndarray) -> np.ndarray:
        s = self.sd(g)
        r = (y - g) / s
        return -0.5 * (self.nu + 1.0) * np.log1p(r * r / self.nu) - np.log(s)


SMBG_OBS = ObservationModel(abs_sd=6.0, rel_sd=0.07)
CGM_OBS = ObservationModel(abs_sd=8.0, rel_sd=0.10)
OBS_MODELS = {"smbg": SMBG_OBS, "cgm": CGM_OBS}


@dataclass(frozen=True)
class FilterConfig:
    n_particles: int = 1000
    dt: float = 5.0
    g_process_sd: float = 1.5  # mg/dL per step: unmodelled glucose disturbances
    daily_log_sd: tuple[tuple[int, float], ...] = ((SI, 0.20), (GB, 0.08), (KSU, 0.20))
    meal_carb_log_sd: float = 0.25  # uncertainty of logged carbohydrate amounts
    resample_ess_frac: float = 0.5
    liu_west_h: float = 0.10
    warmup_min: float = 720.0


@dataclass
class ParticleFilter:
    prior: ParameterPrior
    patient: PatientConstants
    config: FilterConfig = field(default_factory=FilterConfig)
    constants: ModelConstants = DEFAULT_CONSTANTS
    tod_offset_min: float = 0.0  # time of day (min) at t = 0
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        n = self.config.n_particles
        self.theta_log = self.prior.sample(n, self.rng)
        self.state = steady_state(np.exp(self.theta_log), self.constants, self.patient)
        self.log_w = np.full(n, -math.log(n))
        self.t = 0.0
        steps_per_day = MINUTES_PER_DAY / self.config.dt
        self._rw_sd = np.zeros(N_PARAMS)
        for idx, sd in self.config.daily_log_sd:
            self._rw_sd[idx] = sd / math.sqrt(steps_per_day)
        self.n_assimilated = 0
        self._warm_up()

    # ------------------------------------------------------------------ propagation
    def _warm_up(self) -> None:
        """Settle each particle to its own fasting equilibrium before t = 0."""
        theta = np.exp(self.theta_log)
        steps = int(self.config.warmup_min / self.config.dt)
        for k in range(steps):
            tod = (self.tod_offset_min - self.config.warmup_min + k * self.config.dt) % MINUTES_PER_DAY
            self.state = rk4_step(self.state, theta, tod, 0.0, self.config.dt, self.constants, self.patient)

    def tod(self, t: float | None = None) -> float:
        return (self.tod_offset_min + (self.t if t is None else t)) % MINUTES_PER_DAY

    def _propagate(
        self,
        state: np.ndarray,
        theta_log: np.ndarray,
        t: float,
        carbs_fast: float,
        carbs_slow: float,
        su: float,
        activity: float,
        rng: np.random.Generator,
        stochastic: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = state.shape[0]
        if carbs_fast > 0 or carbs_slow > 0:
            mult = np.exp(self.config.meal_carb_log_sd * rng.standard_normal(n)) if stochastic else 1.0
            apply_impulses(state, carbs_fast * mult, carbs_slow * mult, 0.0)
        if su > 0:
            apply_impulses(state, 0.0, 0.0, su)
        state = rk4_step(
            state, np.exp(theta_log), self.tod(t), activity, self.config.dt, self.constants, self.patient
        )
        if stochastic:
            state[:, G] = np.maximum(
                state[:, G] + self.config.g_process_sd * rng.standard_normal(n), self.constants.g_floor
            )
            theta_log = theta_log + self._rw_sd * rng.standard_normal((n, N_PARAMS))
        return state, theta_log

    def step(
        self, carbs_fast: float = 0.0, carbs_slow: float = 0.0, su: float = 0.0, activity: float = 0.0
    ) -> None:
        self.state, self.theta_log = self._propagate(
            self.state, self.theta_log, self.t, carbs_fast, carbs_slow, su, activity, self.rng
        )
        self.t += self.config.dt

    def run(self, grid: GridInputs, record_mean: bool = False) -> np.ndarray | None:
        """Advance through a grid of inputs that starts at the filter's current time."""
        if abs(grid.t0_min - self.t) > 1e-6:
            raise ValueError(f"grid starts at {grid.t0_min}, filter is at {self.t}")
        means = np.empty(grid.n_steps) if record_mean else None
        for k in range(grid.n_steps):
            if means is not None:
                means[k] = self.glucose_mean()
            self.step(grid.carbs_fast_mg[k], grid.carbs_slow_mg[k], grid.su_units[k], grid.activity[k])
        return means

    # ------------------------------------------------------------------ assimilation
    def assimilate(self, value: float, source: str = "smbg") -> float:
        """Condition on a glucose reading taken now. Returns the log marginal likelihood."""
        if not (20.0 <= value <= 600.0):
            raise ValueError(f"glucose reading {value} mg/dL outside plausible range 20-600")
        obs = OBS_MODELS[source]
        ll = obs.log_likelihood(value, self.state[:, G])
        log_w = self.log_w + ll
        lse = float(np.logaddexp.reduce(log_w))
        self.log_w = log_w - lse
        self.n_assimilated += 1
        if self.ess() < self.config.resample_ess_frac * self.config.n_particles:
            self._resample()
        return lse

    def weights(self) -> np.ndarray:
        return np.exp(self.log_w)

    def ess(self) -> float:
        w = self.weights()
        return float(1.0 / np.sum(w * w))

    def _resample(self) -> None:
        n = self.config.n_particles
        w = self.weights()
        positions = (self.rng.random() + np.arange(n)) / n
        idx = np.searchsorted(np.cumsum(w), positions)
        idx = np.minimum(idx, n - 1)
        self.state = self.state[idx].copy()
        theta = self.theta_log[idx]
        # Liu–West kernel shrinkage: preserves mean and variance while restoring diversity.
        h = self.config.liu_west_h
        a = math.sqrt(1.0 - h * h)
        mean = theta.mean(axis=0)
        sd = theta.std(axis=0)
        self.theta_log = a * theta + (1.0 - a) * mean + h * sd * self.rng.standard_normal(theta.shape)
        self.log_w = np.full(n, -math.log(n))

    # ------------------------------------------------------------------ summaries
    def glucose_mean(self) -> float:
        return float(self.weights() @ self.state[:, G])

    def glucose_quantiles(self, qs: tuple[float, ...] = (0.1, 0.5, 0.9)) -> list[float]:
        return weighted_quantiles(self.state[:, G], self.weights(), qs)

    def parameter_summary(self) -> dict[str, dict[str, float]]:
        w = self.weights()
        out: dict[str, dict[str, float]] = {}
        for i, name in enumerate(PARAM_NAMES):
            vals = np.exp(self.theta_log[:, i])
            p05, p50, p95 = weighted_quantiles(vals, w, (0.05, 0.5, 0.95))
            out[name] = {"median": p50, "p05": p05, "p95": p95}
        return out

    def equal_weight_sample(self, m: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Draw m particles with probability proportional to weight (copies)."""
        idx = rng.choice(self.config.n_particles, size=m, replace=True, p=self.weights())
        return self.state[idx].copy(), self.theta_log[idx].copy()

    def forecast(
        self,
        grid: GridInputs,
        n_samples: int = 400,
        rng: np.random.Generator | None = None,
        stochastic: bool = True,
    ) -> np.ndarray:
        """Posterior-predictive glucose trajectories, shape (n_samples, n_steps + 1).

        Column 0 is the current time; column k is t + k*dt. The filter itself is not modified.
        """
        if abs(grid.t0_min - self.t) > 1e-6:
            raise ValueError(f"grid starts at {grid.t0_min}, filter is at {self.t}")
        rng = rng or np.random.default_rng(self.rng.integers(2**63))
        state, theta_log = self.equal_weight_sample(n_samples, rng)
        traj = np.empty((n_samples, grid.n_steps + 1))
        traj[:, 0] = state[:, G]
        t = self.t
        for k in range(grid.n_steps):
            state, theta_log = self._propagate(
                state,
                theta_log,
                t,
                grid.carbs_fast_mg[k],
                grid.carbs_slow_mg[k],
                grid.su_units[k],
                grid.activity[k],
                rng,
                stochastic,
            )
            t += self.config.dt
            traj[:, k + 1] = state[:, G]
        return traj

    def clone(self) -> ParticleFilter:
        return copy.deepcopy(self)


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, qs: tuple[float, ...]) -> list[float]:
    order = np.argsort(values)
    v = values[order]
    cw = np.cumsum(weights[order])
    cw /= cw[-1]
    return [float(np.interp(q, cw, v)) for q in qs]
