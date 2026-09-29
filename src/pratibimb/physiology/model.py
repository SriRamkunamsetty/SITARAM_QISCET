"""Vectorised glucose–insulin–gut–drug model for Type 2 Diabetes.

The model is an extended Bergman minimal model:

    dG/dt   = -S_G (G - G_b,eff) - X G + Ra(t) - e_ex A(t) G + c_cr [G_cr - G]_+
    dX/dt   = -p2 X + p2 S_I c(t) (1 + b_ex A_ex) I_e
    dI_e/dt = -n I_e + phi [G - G_b,eff]_+ + k_SU C_SU
    gut     : two parallel two-compartment chains (fast / slow carbohydrate)
    SU PK   : first-order absorption and elimination (glimepiride-equivalent units)
    A_ex    : low-pass filtered physical activity (prolonged insulin-sensitising effect)

G is plasma glucose (mg/dL), I_e insulin above basal (uU/mL), X remote insulin action (1/min).
Sulfonylurea (SU) drives insulin secretion independently of glucose, which is the mechanism
behind the fasting and nocturnal hypoglycaemia this project targets.

All functions operate on arrays shaped (n_particles, ...) so a particle filter can propagate
thousands of candidate "physiologies" in one call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# --- state vector layout -----------------------------------------------------------------
G, X, IE, Q1F, Q2F, Q1S, Q2S, ASU, CSU, AEX = range(10)
N_STATES = 10
STATE_NAMES = ("G", "X", "Ie", "Q1f", "Q2f", "Q1s", "Q2s", "Asu", "Csu", "Aex")

# --- patient-specific (estimated) parameter layout ----------------------------------------
SI, SG, PHI, GB, KGE, KSU = range(6)
N_PARAMS = 6
PARAM_NAMES = ("si", "sg", "phi", "gb", "kge", "ksu")
PARAM_LABELS = {
    "si": ("Insulin sensitivity", "1/min per uU/mL"),
    "sg": ("Glucose effectiveness", "1/min"),
    "phi": ("Beta-cell responsiveness", "uU/mL/min per mg/dL"),
    "gb": ("Basal glucose (untreated)", "mg/dL"),
    "kge": ("Gastric emptying rate", "1/min"),
    "ksu": ("Sulfonylurea secretagogue gain", "uU/mL/min per unit"),
}

MINUTES_PER_DAY = 1440.0


@dataclass(frozen=True)
class ModelConstants:
    """Population constants that are not estimated per patient."""

    p2: float = 0.025  # 1/min, insulin action dynamics
    n: float = 0.16  # 1/min, insulin clearance
    f_bio: float = 0.9  # carbohydrate bioavailability
    v_g: float = 1.7  # dL/kg glucose distribution volume
    slow_ratio: float = 0.4  # slow-chain rate relative to fast chain
    ka_su: float = 1.0 / 60.0  # 1/min SU absorption (Tmax ~2-3 h)
    ke_su: float = float(np.log(2.0) / 360.0)  # 1/min SU elimination (t1/2 ~6 h)
    cr_gain: float = 0.015  # 1/min counter-regulatory response below cr_threshold
    cr_threshold: float = 72.0  # mg/dL
    ex_uptake: float = 0.006  # 1/min insulin-independent uptake at full activity
    ex_si_boost: float = 0.8  # fractional S_I increase at sustained full activity
    ex_tau: float = 240.0  # min, persistence of the exercise insulin-sensitising effect
    circ_amp: float = 0.12  # circadian S_I amplitude
    circ_peak_min: float = 120.0  # S_I peaks ~02:00 (low overnight insulin need; dawn resistance)
    dawn_amp: float = 0.10  # dawn phenomenon: fractional rise in basal glucose
    dawn_center_min: float = 360.0  # 06:00
    dawn_width_min: float = 90.0
    sleep_cr_factor: float = 0.5  # counter-regulation blunted during sleep (23:00-06:00)
    met_effect: float = 0.12  # metformin: fractional reduction of basal glucose production
    g_floor: float = 20.0  # numerical floor, mg/dL

    def scaled(self, **factors: float) -> ModelConstants:
        """Return a copy with selected constants multiplied by the given factors."""
        values = {k: getattr(self, k) * v for k, v in factors.items()}
        return ModelConstants(**{**self.__dict__, **values})


DEFAULT_CONSTANTS = ModelConstants()


@dataclass(frozen=True)
class PatientConstants:
    """Per-patient quantities known from the EHR rather than estimated."""

    body_weight_kg: float = 65.0
    on_metformin: bool = False


def _circular_gaussian(tod: float, center: float, width: float) -> float:
    d = (tod - center + MINUTES_PER_DAY / 2) % MINUTES_PER_DAY - MINUTES_PER_DAY / 2
    return float(np.exp(-0.5 * (d / width) ** 2))


def time_modulators(tod_min: float, c: ModelConstants) -> tuple[float, float, float]:
    """Return (circadian S_I multiplier, dawn multiplier on basal glucose, counter-reg factor)."""
    circ = 1.0 + c.circ_amp * np.cos(2 * np.pi * (tod_min - c.circ_peak_min) / MINUTES_PER_DAY)
    dawn = 1.0 + c.dawn_amp * _circular_gaussian(tod_min, c.dawn_center_min, c.dawn_width_min)
    asleep = tod_min >= 23 * 60 or tod_min < 6 * 60
    return float(circ), dawn, c.sleep_cr_factor if asleep else 1.0


def rhs(
    y: FloatArray,
    theta: FloatArray,
    tod_min: float,
    activity: float,
    c: ModelConstants,
    pc: PatientConstants,
) -> FloatArray:
    """Time derivative of the state for every particle.

    y: (N, N_STATES) state; theta: (N, N_PARAMS) parameters on the natural scale.
    """
    g = y[:, G]
    x = y[:, X]
    ie = y[:, IE]
    si, sg, phi, gb, kge, ksu = (theta[:, i] for i in range(N_PARAMS))

    circ, dawn, cr_factor = time_modulators(tod_min, c)
    gb_eff = gb * (1.0 - c.met_effect * float(pc.on_metformin)) * dawn
    volume_dl = c.v_g * pc.body_weight_kg
    k_slow = kge * c.slow_ratio
    ra = c.f_bio * (kge * y[:, Q2F] + k_slow * y[:, Q2S]) / volume_dl

    dy = np.empty_like(y)
    dy[:, G] = (
        -sg * (g - gb_eff)
        - x * g
        + ra
        - c.ex_uptake * activity * g
        + cr_factor * c.cr_gain * np.maximum(c.cr_threshold - g, 0.0)
    )
    dy[:, X] = -c.p2 * x + c.p2 * si * circ * (1.0 + c.ex_si_boost * y[:, AEX]) * ie
    dy[:, IE] = -c.n * ie + phi * np.maximum(g - gb_eff, 0.0) + ksu * y[:, CSU]
    dy[:, Q1F] = -kge * y[:, Q1F]
    dy[:, Q2F] = kge * (y[:, Q1F] - y[:, Q2F])
    dy[:, Q1S] = -k_slow * y[:, Q1S]
    dy[:, Q2S] = k_slow * (y[:, Q1S] - y[:, Q2S])
    dy[:, ASU] = -c.ka_su * y[:, ASU]
    dy[:, CSU] = c.ka_su * y[:, ASU] - c.ke_su * y[:, CSU]
    dy[:, AEX] = (activity - y[:, AEX]) / c.ex_tau
    return dy


def rk4_step(
    y: FloatArray,
    theta: FloatArray,
    tod_min: float,
    activity: float,
    dt: float,
    c: ModelConstants,
    pc: PatientConstants,
) -> FloatArray:
    """One classical Runge–Kutta step of length dt minutes (inputs held constant)."""
    half = 0.5 * dt
    tod_mid = (tod_min + half) % MINUTES_PER_DAY
    tod_end = (tod_min + dt) % MINUTES_PER_DAY
    k1 = rhs(y, theta, tod_min, activity, c, pc)
    k2 = rhs(y + half * k1, theta, tod_mid, activity, c, pc)
    k3 = rhs(y + half * k2, theta, tod_mid, activity, c, pc)
    k4 = rhs(y + dt * k3, theta, tod_end, activity, c, pc)
    y_next = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    np.maximum(y_next, 0.0, out=y_next)
    np.maximum(y_next[:, G], c.g_floor, out=y_next[:, G])
    return y_next


def apply_impulses(
    y: FloatArray,
    carbs_fast_mg: float | FloatArray,
    carbs_slow_mg: float | FloatArray,
    su_units: float,
) -> None:
    """Add meal carbohydrate (mg) and SU dose (units) at the start of a step, in place."""
    y[:, Q1F] += carbs_fast_mg
    y[:, Q1S] += carbs_slow_mg
    y[:, ASU] += su_units


def steady_state(theta: FloatArray, c: ModelConstants, pc: PatientConstants) -> FloatArray:
    """Fasting state with no meal, drug or activity on board (dawn/circadian ignored)."""
    y = np.zeros((theta.shape[0], N_STATES))
    y[:, G] = theta[:, GB] * (1.0 - c.met_effect * float(pc.on_metformin))
    return y
