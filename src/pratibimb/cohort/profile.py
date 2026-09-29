"""Static (EHR-derived) patient description used to initialise a twin."""

from __future__ import annotations

from dataclasses import dataclass, field

from pratibimb.physiology.drugs import is_sulfonylurea, normalise_drug


@dataclass(frozen=True)
class Regimen:
    """A standing prescription, e.g. glimepiride 2 mg at 08:00."""

    drug: str
    dose_mg: float
    times_hhmm: tuple[str, ...]

    @property
    def is_sulfonylurea(self) -> bool:
        return is_sulfonylurea(self.drug)

    @property
    def drug_key(self) -> str:
        return normalise_drug(self.drug)

    def minutes_of_day(self) -> list[float]:
        return [hhmm_to_minutes(t) for t in self.times_hhmm]


@dataclass(frozen=True)
class HabitualMeal:
    time_hhmm: str
    carbs_g: float
    glycemic_index: float = 70.0
    label: str = ""


DEFAULT_INDIAN_MEALS: tuple[HabitualMeal, ...] = (
    HabitualMeal("08:30", 50.0, 78.0, "breakfast (idli/dosa/poha)"),
    HabitualMeal("13:30", 85.0, 72.0, "lunch (rice, dal, sabzi)"),
    HabitualMeal("17:30", 20.0, 65.0, "tea + snack"),
    HabitualMeal("21:00", 65.0, 70.0, "dinner (chapati/rice)"),
)


@dataclass(frozen=True)
class PatientProfile:
    """Clinical summary; units: years, kg/m^2, cm, %, mg/dL, bpm, steps/day."""

    age_years: float
    sex: str
    age_at_diagnosis: float
    bmi: float
    waist_cm: float
    hba1c_pct: float
    body_weight_kg: float
    fasting_glucose: float | None = None
    triglycerides: float | None = None
    hdl: float | None = None
    resting_hr: float | None = None
    daily_steps: float | None = None
    regimens: tuple[Regimen, ...] = ()
    habitual_meals: tuple[HabitualMeal, ...] = DEFAULT_INDIAN_MEALS
    sleep_hhmm: str = "23:00"
    wake_hhmm: str = "06:30"
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def on_metformin(self) -> bool:
        return any(r.drug_key == "metformin" for r in self.regimens)

    @property
    def on_sulfonylurea(self) -> bool:
        return any(r.is_sulfonylurea for r in self.regimens)


def hhmm_to_minutes(value: str) -> float:
    hours, minutes = value.split(":")
    h, m = int(hours), int(minutes)
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ValueError(f"invalid time of day: {value!r}")
    return float(h * 60 + m)
