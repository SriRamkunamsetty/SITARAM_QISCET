"""High-level, wall-clock digital twin: event-sourced, replayable, thread-safe.

The twin's state is a pure function of (profile, seed, event log). Every glucose reading, meal,
dose and activity bout is appended to an immutable log; the particle filter is advanced
incrementally and transparently rebuilt from the log when late (out-of-order) data arrive. This
gives an audit trail for every number the twin reports and makes persistence trivial.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from pratibimb import DISCLAIMER
from pratibimb.cohort.endotypes import INSPIRED_ENDOTYPES, build_prior
from pratibimb.cohort.profile import HabitualMeal, PatientProfile, Regimen
from pratibimb.cohort.routine import routine_events
from pratibimb.physiology.inputs import (
    ActivityBout,
    EventLog,
    FoodOrder,
    Meal,
    MedicationDose,
    discretise,
)
from pratibimb.physiology.model import PARAM_LABELS, PatientConstants
from pratibimb.sensing.planner import PlannerConfig, plan_readings
from pratibimb.twin.events import RiskWindow, sustained_crossing
from pratibimb.twin.particle_filter import FilterConfig, ParticleFilter

DT_MIN = 5.0


class TwinError(ValueError):
    """Invalid request against a twin (bad time, out-of-range value, ...)."""


@dataclass(frozen=True)
class GlucoseReading:
    t_min: float
    value_mg_dl: float
    source: str  # "smbg" | "cgm"


@dataclass
class TwinRecord:
    """Everything needed to reconstruct a twin exactly (the persistence format)."""

    twin_id: str
    anchor_iso: str
    seed: int
    n_particles: int
    profile: dict[str, Any]
    glucose: list[dict[str, Any]] = field(default_factory=list)
    meals: list[dict[str, Any]] = field(default_factory=list)
    medications: list[dict[str, Any]] = field(default_factory=list)
    activity: list[dict[str, Any]] = field(default_factory=list)


def _floor_to_grid(ts: datetime) -> datetime:
    return ts.replace(second=0, microsecond=0) - timedelta(minutes=ts.minute % int(DT_MIN))


def _profile_to_dict(p: PatientProfile) -> dict[str, Any]:
    d = asdict(p)
    d["regimens"] = [asdict(r) for r in p.regimens]
    d["habitual_meals"] = [asdict(m) for m in p.habitual_meals]
    return d


def _profile_from_dict(d: dict[str, Any]) -> PatientProfile:
    d = dict(d)
    d["regimens"] = tuple(
        Regimen(r["drug"], float(r["dose_mg"]), tuple(r["times_hhmm"])) for r in d.get("regimens", ())
    )
    d["habitual_meals"] = tuple(HabitualMeal(**m) for m in d.get("habitual_meals", ()))
    return PatientProfile(**d)


class DigitalTwin:
    def __init__(
        self,
        profile: PatientProfile,
        anchor: datetime,
        twin_id: str | None = None,
        seed: int | None = None,
        n_particles: int = 1000,
    ) -> None:
        if anchor.tzinfo is None:
            raise TwinError("anchor must be timezone-aware")
        self.twin_id = twin_id or uuid.uuid4().hex[:12]
        self.profile = profile
        self.anchor = _floor_to_grid(anchor)
        self.seed = int(seed) if seed is not None else int(np.random.default_rng().integers(2**31))
        self.n_particles = n_particles
        self.prior = build_prior(profile)
        self.events = EventLog()
        self.glucose: list[GlucoseReading] = []
        self._lock = threading.RLock()
        self._pf = self._new_filter()
        self._cursor = 0  # number of glucose readings already assimilated

    # ------------------------------------------------------------------ time helpers
    def to_min(self, ts: datetime) -> float:
        if ts.tzinfo is None:
            raise TwinError("timestamps must be timezone-aware (e.g. 2026-10-01T08:00:00+05:30)")
        return (ts - self.anchor).total_seconds() / 60.0

    def to_datetime(self, t_min: float) -> datetime:
        return self.anchor + timedelta(minutes=float(t_min))

    def _new_filter(self) -> ParticleFilter:
        tod = self.anchor.hour * 60 + self.anchor.minute
        return ParticleFilter(
            prior=self.prior,
            patient=PatientConstants(self.profile.body_weight_kg, self.profile.on_metformin),
            config=FilterConfig(n_particles=self.n_particles, dt=DT_MIN),
            tod_offset_min=float(tod),
            seed=self.seed,
        )

    def _check_time(self, t: float) -> None:
        if t < 0:
            raise TwinError("event precedes the twin's creation time")

    # ------------------------------------------------------------------ ingestion
    def add_glucose(self, ts: datetime, value: float, source: str = "smbg") -> None:
        if source not in {"smbg", "cgm"}:
            raise TwinError("source must be 'smbg' or 'cgm'")
        if not 20.0 <= value <= 600.0:
            raise TwinError("glucose must be within 20-600 mg/dL")
        t = self.to_min(ts)
        self._check_time(t)
        with self._lock:
            self.glucose.append(GlucoseReading(t, float(value), source))
            self.glucose.sort(key=lambda r: r.t_min)
            self._invalidate(t)

    def add_meal(
        self,
        ts: datetime,
        carbs_g: float,
        glycemic_index: float = 70.0,
        food_order: FoodOrder = FoodOrder.MIXED,
        label: str = "",
    ) -> None:
        if not 0 <= carbs_g <= 400:
            raise TwinError("carbs_g must be within 0-400 g")
        t = self.to_min(ts)
        self._check_time(t)
        with self._lock:
            self.events.meals.append(Meal(t, carbs_g, glycemic_index, food_order, label))
            self._invalidate(t)

    def add_medication(self, ts: datetime, drug: str, dose_mg: float) -> None:
        if dose_mg < 0:
            raise TwinError("dose_mg must be non-negative")
        t = self.to_min(ts)
        self._check_time(t)
        with self._lock:
            self.events.medications.append(MedicationDose(t, drug, dose_mg))
            self._invalidate(t)

    def add_activity(self, start: datetime, end: datetime, intensity: float) -> None:
        if not 0 <= intensity <= 1 or end <= start:
            raise TwinError("activity needs end > start and intensity in [0, 1]")
        t0, t1 = self.to_min(start), self.to_min(end)
        self._check_time(t0)
        with self._lock:
            self.events.activity.append(ActivityBout(t0, t1, intensity))
            self._invalidate(t0)

    def _invalidate(self, t: float) -> None:
        """New data at time t: rebuild the filter if it has already moved past t."""
        if t < self._pf.t - 1e-9:
            self._pf = self._new_filter()
            self._cursor = 0

    # ------------------------------------------------------------------ assimilation
    def _advance(self, t_target: float) -> None:
        n = int(np.floor((t_target - self._pf.t) / DT_MIN + 1e-9))
        if n <= 0:
            return
        grid = discretise(self.events.window(self._pf.t, self._pf.t + n * DT_MIN), self._pf.t, n, DT_MIN)
        self._pf.run(grid)

    def sync(self, now: datetime) -> None:
        """Bring the filter up to ``now``, assimilating every reading taken before it."""
        t_now = self.to_min(now)
        with self._lock:
            while self._cursor < len(self.glucose) and self.glucose[self._cursor].t_min <= t_now:
                r = self.glucose[self._cursor]
                self._advance(r.t_min)
                self._pf.assimilate(r.value_mg_dl, r.source)
                self._cursor += 1
            self._advance(t_now)

    # ------------------------------------------------------------------ outputs
    def state(self, now: datetime) -> dict[str, Any]:
        with self._lock:
            self.sync(now)
            q10, q50, q90 = self._pf.glucose_quantiles((0.1, 0.5, 0.9))
            params = self._pf.parameter_summary()
            prior = self.prior.summary()
            return {
                "twin_id": self.twin_id,
                "as_of": self.to_datetime(self._pf.t).isoformat(),
                "glucose_now": {"p10": q10, "median": q50, "p90": q90},
                "parameters": {
                    name: {
                        "label": PARAM_LABELS[name][0],
                        "unit": PARAM_LABELS[name][1],
                        **vals,
                        "prior_median": prior[name]["median"],
                        "change_vs_prior_pct": 100.0 * (vals["median"] / prior[name]["median"] - 1.0),
                    }
                    for name, vals in params.items()
                },
                "endotype_probabilities": self.prior.endotype_probs,
                "endotype_names": {e.code: e.name for e in INSPIRED_ENDOTYPES},
                "readings_assimilated": self._cursor,
                "effective_sample_size": self._pf.ess(),
                "disclaimer": DISCLAIMER,
            }

    def _future_events(self, t_start: float, t_end: float, planned: EventLog | None) -> EventLog:
        future = routine_events(self.profile, t_start, t_end, self._pf.tod_offset_min)
        logged = self.events.window(t_start, t_end)
        if planned is not None:  # explicit plans replace the habitual routine
            future = planned.window(t_start, t_end)
        # anything already logged in the future window (e.g. a scheduled dose) takes precedence
        if logged.meals:
            future.meals = logged.meals
        if logged.medications:
            future.medications = logged.medications
        future.activity = future.activity + logged.activity
        return future

    def forecast(
        self,
        now: datetime,
        horizon_min: float = 240.0,
        planned: EventLog | None = None,
        n_samples: int = 500,
    ) -> dict[str, Any]:
        if not 15 <= horizon_min <= 36 * 60:
            raise TwinError("horizon must be between 15 minutes and 36 hours")
        with self._lock:
            self.sync(now)
            t0 = self._pf.t
            n = int(horizon_min / DT_MIN)
            grid = discretise(self._future_events(t0, t0 + n * DT_MIN, planned), t0, n, DT_MIN)
            traj = self._pf.forecast(
                grid, n_samples=n_samples, rng=np.random.default_rng([self.seed, int(t0)])
            )
        qs = np.percentile(traj, [2.5, 10, 50, 90, 97.5], axis=0)
        return {
            "twin_id": self.twin_id,
            "start": self.to_datetime(t0).isoformat(),
            "step_minutes": DT_MIN,
            "times": [self.to_datetime(t0 + k * DT_MIN).isoformat() for k in range(n + 1)],
            "p2_5": qs[0].tolist(),
            "p10": qs[1].tolist(),
            "median": qs[2].tolist(),
            "p90": qs[3].tolist(),
            "p97_5": qs[4].tolist(),
            "risk": {
                "hypo_lt70_15min": float(sustained_crossing(traj, 70.0, 3).mean()),
                "hypo_lt54_15min": float(sustained_crossing(traj, 54.0, 3).mean()),
                "hyper_gt250": float(sustained_crossing(traj, 250.0, 1, below=False).mean()),
            },
            "disclaimer": DISCLAIMER,
        }

    def plan_fingersticks(
        self,
        now: datetime,
        strips: int,
        until: datetime | None = None,
        waking_start_hhmm: str | None = None,
        waking_end_hhmm: str | None = None,
        min_spacing_min: float = 60.0,
        include_night_window: bool = True,
    ) -> dict[str, Any]:
        """Recommend when to take the next ``strips`` fingerstick readings.

        Candidate times are every 30 minutes during waking hours between now and ``until``
        (default: the next wake-up). Risk windows are each remaining ~6-hour block, plus the
        coming night (weighted x2) whose risk call is made at bedtime.
        """
        if not 0 <= strips <= 12:
            raise TwinError("strips must be between 0 and 12")
        wake = waking_start_hhmm or self.profile.wake_hhmm
        sleep = waking_end_hhmm or self.profile.sleep_hhmm
        with self._lock:
            self.sync(now)
            t0 = self._pf.t
            tod0 = self._pf.tod_offset_min
            wake_m, sleep_m = (_hhmm(wake), _hhmm(sleep))
            day_start = t0 - ((tod0 + t0) % 1440.0)  # local midnight of 'now'
            tod_now = (tod0 + t0) % 1440.0
            bedtime = day_start + sleep_m if tod_now < sleep_m else day_start + 1440.0 + sleep_m
            next_wake = bedtime + (wake_m - sleep_m) % 1440.0
            t_end = self.to_min(until) if until else next_wake
            if t_end <= t0:
                raise TwinError("'until' must be in the future")
            windows = []
            block = 360.0
            s = t0
            while s < min(bedtime, t_end):
                e = min(s + block, bedtime, t_end)
                windows.append(
                    RiskWindow(s, e, label=f"day {self.to_datetime(s):%H:%M}-{self.to_datetime(e):%H:%M}")
                )
                s = e
            if include_night_window and bedtime < t_end:
                windows.append(
                    RiskWindow(bedtime, next_wake, weight=2.0, label="overnight", decision_min=bedtime)
                )
            cands = [
                t
                for t in np.arange(np.ceil(t0 / 30.0) * 30.0, min(bedtime, t_end) + 1e-9, 30.0)
                if _is_waking((tod0 + t) % 1440.0, wake_m, sleep_m)
            ]
            horizon = max(w.end_min for w in windows) if windows else t0 + DT_MIN
            n = int(np.ceil((horizon - t0) / DT_MIN))
            grid = discretise(self._future_events(t0, t0 + n * DT_MIN, None), t0, n, DT_MIN)
            plan = (
                plan_readings(
                    self._pf,
                    grid,
                    windows,
                    cands,
                    strips,
                    config=PlannerConfig(min_spacing_min=min_spacing_min),
                    rng=np.random.default_rng([self.seed, int(t0), strips]),
                )
                if windows
                else None
            )
        return {
            "twin_id": self.twin_id,
            "as_of": self.to_datetime(t0).isoformat(),
            "recommended_times": [
                {"time": self.to_datetime(r.t_min).isoformat(), "information_gain_bits": r.gain_bits}
                for r in (plan.readings if plan else [])
            ],
            "risk_before_readings": plan.prior_risk if plan else {},
            "candidate_value_bits": (
                {self.to_datetime(t).isoformat(): v for t, v in plan.candidate_gain_bits.items()}
                if plan
                else {}
            ),
            "total_information_gain_bits": plan.total_gain_bits if plan else 0.0,
            "strips_unused": strips - (len(plan.readings) if plan else 0),
            "rationale": (
                "Times are chosen to maximise information about hypoglycaemia risk in the remaining "
                "day blocks and the coming night; unused strips mean more readings would not "
                "change the risk estimate meaningfully."
            ),
            "disclaimer": DISCLAIMER,
        }

    # ------------------------------------------------------------------ persistence
    def to_record(self) -> TwinRecord:
        with self._lock:
            return TwinRecord(
                twin_id=self.twin_id,
                anchor_iso=self.anchor.isoformat(),
                seed=self.seed,
                n_particles=self.n_particles,
                profile=_profile_to_dict(self.profile),
                glucose=[asdict(g) for g in self.glucose],
                meals=[{**asdict(m), "food_order": str(m.food_order)} for m in self.events.meals],
                medications=[asdict(m) for m in self.events.medications],
                activity=[asdict(a) for a in self.events.activity],
            )

    @classmethod
    def from_record(cls, rec: TwinRecord) -> DigitalTwin:
        twin = cls(
            _profile_from_dict(rec.profile),
            datetime.fromisoformat(rec.anchor_iso),
            twin_id=rec.twin_id,
            seed=rec.seed,
            n_particles=rec.n_particles,
        )
        twin.glucose = sorted((GlucoseReading(**g) for g in rec.glucose), key=lambda r: r.t_min)
        twin.events.meals = [Meal(**{**m, "food_order": FoodOrder(m["food_order"])}) for m in rec.meals]
        twin.events.medications = [MedicationDose(**m) for m in rec.medications]
        twin.events.activity = [ActivityBout(**a) for a in rec.activity]
        return twin


def _hhmm(value: str) -> float:
    h, m = (int(x) for x in value.split(":"))
    return float(h * 60 + m)


def _is_waking(tod: float, wake: float, sleep: float) -> bool:
    return wake <= tod <= sleep if wake < sleep else (tod >= wake or tod <= sleep)
