"""Synthetic Indian T2D cohort with known ground truth ("Synthea-IN" dynamic layer).

The truth simulator deliberately differs from the twin's own model so that evaluation is honest:
per-patient perturbations of fixed constants, day-to-day insulin-sensitivity drift, carbohydrate
counting error, unlogged snacks, missed doses and sensor noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from pratibimb.cohort.endotypes import (
    INSPIRED_ENDOTYPES,
    PRIOR_LOG_SD,
    Endotype,
    endotype_log_means,
)
from pratibimb.cohort.profile import DEFAULT_INDIAN_MEALS, HabitualMeal, PatientProfile, Regimen
from pratibimb.physiology.inputs import ActivityBout, EventLog, Meal, MedicationDose, discretise
from pratibimb.physiology.model import (
    DEFAULT_CONSTANTS,
    GB,
    KSU,
    MINUTES_PER_DAY,
    N_PARAMS,
    SI,
    G,
    ModelConstants,
    PatientConstants,
    apply_impulses,
    rk4_step,
    steady_state,
)


@dataclass(frozen=True)
class SensorModel:
    """Additive + multiplicative Gaussian error: sd = sqrt(abs^2 + (rel*G)^2)."""

    abs_sd: float
    rel_sd: float
    autocorr: float = 0.0  # AR(1) coefficient between consecutive readings

    def sd(self, g: np.ndarray | float) -> np.ndarray:
        return np.sqrt(self.abs_sd**2 + (self.rel_sd * np.asarray(g)) ** 2)


# Day-to-day log-scale variability of true parameters (assumptions; S_I day-to-day CV of
# ~20% is in the range reported for repeated clamps/meal tests in T2D).
DAILY_LOG_SD: dict[int, float] = {SI: 0.20, GB: 0.08, KSU: 0.20}

# Truth-side sensor error (what devices actually do).
SMBG_SENSOR = SensorModel(abs_sd=3.0, rel_sd=0.05)  # ISO 15197-class meter
CGM_SENSOR = SensorModel(abs_sd=5.0, rel_sd=0.08, autocorr=0.7)  # ~9% MARD class


@dataclass(frozen=True)
class VirtualPatient:
    patient_id: str
    endotype: str
    profile: PatientProfile
    true_theta: np.ndarray  # natural scale, shape (N_PARAMS,)
    true_constants: ModelConstants
    fasting_day_prob: float = 0.04
    unlogged_snack_prob: float = 0.15
    adherence: float = 0.92
    post_dinner_walk_prob: float = 0.3


@dataclass
class DayRecord:
    true_events: EventLog = field(default_factory=EventLog)
    logged_events: EventLog = field(default_factory=EventLog)
    fasting_day: bool = False


@dataclass(frozen=True)
class TruthTrajectory:
    t_min: np.ndarray  # grid times (start of each step), shape (T,)
    glucose: np.ndarray  # true plasma glucose at those times, shape (T,)
    dt: float

    def at(self, t: float) -> float:
        k = round((t - self.t_min[0]) / self.dt)
        return float(self.glucose[np.clip(k, 0, len(self.glucose) - 1)])


def _sample_endotype(rng: np.random.Generator, only: str | None) -> Endotype:
    if only is not None:
        return next(e for e in INSPIRED_ENDOTYPES if e.code == only)
    shares = np.array([e.share for e in INSPIRED_ENDOTYPES])
    return INSPIRED_ENDOTYPES[int(rng.choice(len(shares), p=shares / shares.sum()))]


# Assumed median untreated basal glucose per endotype (mg/dL); on-treatment HbA1c is then an
# *outcome* of the simulated physiology, not an input.
_UNTREATED_GB = {"SIDD": 205.0, "CIRDD": 180.0, "IROD": 170.0, "MARD": 150.0}


def sample_virtual_patient(
    rng: np.random.Generator,
    patient_id: str,
    endotype: str | None = None,
    force_sulfonylurea: bool = False,
    ksu_scale: float = 1.0,
    gb_scale: float = 1.0,
    calibration_days: int = 7,
) -> VirtualPatient:
    e = _sample_endotype(rng, endotype)
    age_dx = float(np.clip(rng.normal(e.age_dx, 8.0), 25, 75))
    duration = float(rng.uniform(1, 15))
    bmi = float(np.clip(rng.normal(e.bmi, 3.0), 18, 45))
    waist = float(np.clip(rng.normal(e.waist_cm, 8.0), 65, 140))
    height_m = float(rng.normal(1.62, 0.08))
    weight = bmi * height_m**2

    on_met = bool(rng.random() < 0.85)
    su_prob = {"SIDD": 0.7, "CIRDD": 0.6, "IROD": 0.45, "MARD": 0.4}[e.code]
    on_su = force_sulfonylurea or bool(rng.random() < su_prob)
    regimens: list[Regimen] = []
    if on_met:
        regimens.append(Regimen("metformin", 500.0, ("08:30", "21:00")))
    if on_su:
        u = rng.random()
        if u < 0.5:
            regimens.append(Regimen("glimepiride", float(rng.choice([1.0, 2.0, 3.0, 4.0])), ("08:00",)))
        elif u < 0.85:
            regimens.append(Regimen("gliclazide", float(rng.choice([40.0, 80.0])), ("08:00", "20:30")))
        else:
            regimens.append(Regimen("glibenclamide", float(rng.choice([2.5, 5.0])), ("08:00", "20:30")))

    # True parameters: endotype mean plus individual deviation.
    mu = endotype_log_means(e)
    mu[GB] = math.log(_UNTREATED_GB[e.code] * gb_scale)
    log_theta = mu + PRIOR_LOG_SD * rng.standard_normal(N_PARAMS)
    theta = np.exp(log_theta)
    theta[KSU] *= ksu_scale

    meals = tuple(
        HabitualMeal(
            hm.time_hhmm,
            float(np.clip(hm.carbs_g * rng.lognormal(0, 0.15), 10, 150)),
            hm.glycemic_index,
            hm.label,
        )
        for hm in DEFAULT_INDIAN_MEALS
    )
    profile = PatientProfile(
        age_years=age_dx + duration,
        sex=str(rng.choice(["F", "M"])),
        age_at_diagnosis=age_dx,
        bmi=bmi,
        waist_cm=waist,
        hba1c_pct=float("nan"),
        body_weight_kg=weight,
        fasting_glucose=None,
        resting_hr=float(rng.normal(74, 8)),
        daily_steps=float(np.clip(rng.normal(5500, 2000), 1000, 15000)),
        regimens=tuple(regimens),
        habitual_meals=meals,
    )
    constants = DEFAULT_CONSTANTS.scaled(
        p2=float(rng.lognormal(0, 0.15)),
        n=float(rng.lognormal(0, 0.15)),
        cr_gain=float(rng.lognormal(0, 0.3)),
        dawn_amp=float(rng.lognormal(0, 0.3)),
    )
    vp = VirtualPatient(patient_id, e.code, profile, theta, constants)

    # "Lab visit": derive HbA1c (ADAG) and fasting glucose from a pre-study simulation.
    cal_rng = np.random.default_rng(rng.integers(2**63))
    cal_days = [generate_day(vp, d, cal_rng) for d in range(calibration_days)]
    truth = simulate_truth(vp, cal_days, cal_rng)
    hba1c = float(np.clip((truth.glucose.mean() + 46.7) / 28.7, 5.0, 15.0))
    fasting_idx = (truth.t_min % MINUTES_PER_DAY) == 7 * 60
    fpg = float(truth.glucose[fasting_idx].mean() * rng.lognormal(0, 0.05))
    profile = replace(profile, hba1c_pct=round(hba1c, 1), fasting_glucose=round(fpg))
    return replace(vp, profile=profile)


def generate_day(vp: VirtualPatient, day: int, rng: np.random.Generator) -> DayRecord:
    """True and logged events for one day (day 0 starts at t = 0, local midnight)."""
    base = day * MINUTES_PER_DAY
    rec = DayRecord()
    rec.fasting_day = bool(rng.random() < vp.fasting_day_prob)
    for hm in vp.profile.habitual_meals:
        h, m = (int(x) for x in hm.time_hhmm.split(":"))
        t = base + h * 60 + m + rng.normal(0, 20)
        carbs = hm.carbs_g * rng.lognormal(0, 0.25)
        if rec.fasting_day and "snack" not in hm.label:
            carbs *= 0.25  # vrat/fasting-day meals: small, fruit or sabudana
        elif "dinner" in hm.label and rng.random() < 0.08:
            carbs *= 0.3  # light or skipped dinner
        gi = float(np.clip(hm.glycemic_index + rng.normal(0, 8), 40, 95))
        true_meal = Meal(t, float(carbs), gi, label=hm.label)
        rec.true_events.meals.append(true_meal)
        logged_carbs = float(carbs * rng.lognormal(0, 0.2))  # carb-counting error
        rec.logged_events.meals.append(
            Meal(5 * round(t / 5), logged_carbs, hm.glycemic_index, label=hm.label)
        )
    if rng.random() < vp.unlogged_snack_prob:
        rec.true_events.meals.append(Meal(base + rng.uniform(600, 1200), float(rng.uniform(10, 30)), 70.0))
    for reg in vp.profile.regimens:
        if not reg.is_sulfonylurea:
            continue
        for mod in reg.minutes_of_day():
            if rng.random() < vp.adherence:
                dose = MedicationDose(base + mod + rng.normal(0, 10), reg.drug, reg.dose_mg)
                rec.true_events.medications.append(dose)
                rec.logged_events.medications.append(dose)
    if rng.random() < 0.5:
        walk = ActivityBout(base + 390, base + 420, 0.6)  # morning walk
        rec.true_events.activity.append(walk)
        rec.logged_events.activity.append(walk)
    if rng.random() < vp.post_dinner_walk_prob:
        start = base + 21 * 60 + 40
        walk = ActivityBout(start, start + 15, 0.7)
        rec.true_events.activity.append(walk)
        rec.logged_events.activity.append(walk)
    return rec


def simulate_truth(
    vp: VirtualPatient,
    days: list[DayRecord],
    rng: np.random.Generator,
    dt: float = 5.0,
    daily_log_sd: dict[int, float] | None = None,
    g_noise_sd: float = 0.6,
) -> TruthTrajectory:
    """Simulate the 'real' patient over consecutive days starting at t = 0 (00:00).

    ``daily_log_sd`` gives the day-to-day (AR(1), coefficient 0.7) log-scale variability of
    selected parameters: insulin sensitivity, hepatic glucose output (basal level) and SU
    absorption/response, which drive night-to-night differences in hypoglycaemia risk.
    """
    daily_log_sd = DAILY_LOG_SD if daily_log_sd is None else daily_log_sd
    events = EventLog()
    for d in days:
        events.extend(d.true_events)
    n_steps = int(len(days) * MINUTES_PER_DAY / dt)
    grid = discretise(events, 0.0, n_steps, dt)
    pc = PatientConstants(vp.profile.body_weight_kg, vp.profile.on_metformin)
    theta = vp.true_theta[None, :].copy()
    y = steady_state(theta, vp.true_constants, pc)
    # warm-up one virtual day so the initial state is realistic
    for _ in range(int(MINUTES_PER_DAY / dt)):
        y = rk4_step(y, theta, 0.0, 0.0, dt, vp.true_constants, pc)
    log_base = np.log(vp.true_theta)
    deviation = {i: 0.0 for i in daily_log_sd}
    steps_per_day = int(MINUTES_PER_DAY / dt)
    g = np.empty(n_steps)
    for k in range(n_steps):
        if k % steps_per_day == 0:
            for i, sd in daily_log_sd.items():
                deviation[i] = 0.7 * deviation[i] + rng.normal(0, sd)
                theta[0, i] = math.exp(log_base[i] + deviation[i])
        g[k] = y[0, G]
        apply_impulses(y, grid.carbs_fast_mg[k], grid.carbs_slow_mg[k], grid.su_units[k])
        y = rk4_step(y, theta, (k * dt) % MINUTES_PER_DAY, grid.activity[k], dt, vp.true_constants, pc)
        y[0, G] = max(y[0, G] + rng.normal(0, g_noise_sd), vp.true_constants.g_floor)
    return TruthTrajectory(np.arange(n_steps) * dt, g, dt)


def sensor_reading(g_true: float, sensor: SensorModel, rng: np.random.Generator) -> float:
    return float(max(g_true + rng.normal(0.0, float(sensor.sd(g_true))), 20.0))


def cgm_trace(
    truth: TruthTrajectory,
    rng: np.random.Generator,
    every_min: float = 15.0,
    sensor: SensorModel = CGM_SENSOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulated CGM readings with AR(1)-correlated relative error."""
    stride = round(every_min / truth.dt)
    t = truth.t_min[::stride]
    g = truth.glucose[::stride]
    e = np.empty_like(g)
    prev = 0.0
    innov = math.sqrt(1 - sensor.autocorr**2)
    for i in range(len(g)):
        prev = sensor.autocorr * prev + innov * rng.standard_normal()
        e[i] = prev
    return t, np.maximum(g + e * sensor.sd(g), 20.0)
