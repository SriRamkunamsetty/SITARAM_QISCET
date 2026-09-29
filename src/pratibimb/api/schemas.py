"""Request/response models for the Pratibimb REST API (pydantic v2)."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from pratibimb.physiology.inputs import FoodOrder

HHMM = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$", examples=["08:00"])


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RegimenIn(_Model):
    drug: str = Field(min_length=2, max_length=64, examples=["glimepiride"])
    dose_mg: float = Field(ge=0, le=5000)
    times: list[str] = Field(min_length=1, max_length=6, examples=[["08:00"]])

    @model_validator(mode="after")
    def _check_times(self) -> RegimenIn:
        import re

        for t in self.times:
            if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", t):
                raise ValueError(f"invalid time {t!r}, expected HH:MM")
        return self


class HabitualMealIn(_Model):
    time: str = HHMM
    carbs_g: float = Field(ge=0, le=400)
    glycemic_index: float = Field(default=70.0, ge=0, le=110)
    label: str = Field(default="", max_length=80)


class PatientProfileIn(_Model):
    age_years: float = Field(ge=18, le=110)
    sex: Literal["F", "M", "O"]
    age_at_diagnosis: float = Field(ge=5, le=110)
    bmi: float = Field(ge=12, le=70)
    waist_cm: float = Field(ge=40, le=200)
    hba1c_pct: float = Field(ge=4, le=18)
    body_weight_kg: float = Field(ge=25, le=250)
    fasting_glucose: float | None = Field(default=None, ge=40, le=500)
    triglycerides: float | None = Field(default=None, ge=20, le=3000)
    hdl: float | None = Field(default=None, ge=10, le=150)
    resting_hr: float | None = Field(default=None, ge=30, le=150)
    daily_steps: float | None = Field(default=None, ge=0, le=60000)
    regimens: list[RegimenIn] = Field(default_factory=list, max_length=12)
    habitual_meals: list[HabitualMealIn] | None = None
    wake_time: str = Field(default="06:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    sleep_time: str = Field(default="23:00", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")

    @model_validator(mode="after")
    def _check_ages(self) -> PatientProfileIn:
        if self.age_at_diagnosis > self.age_years:
            raise ValueError("age_at_diagnosis cannot exceed age_years")
        return self


class CreateTwinIn(_Model):
    profile: PatientProfileIn
    start: AwareDatetime = Field(description="Twin creation time; events before it are rejected.")
    n_particles: int = Field(default=1000, ge=200, le=5000)
    seed: int | None = Field(default=None, ge=0)


class GlucoseIn(_Model):
    time: AwareDatetime
    value_mg_dl: float = Field(ge=20, le=600)
    source: Literal["smbg", "cgm"] = "smbg"


class MealIn(_Model):
    time: AwareDatetime
    carbs_g: float = Field(ge=0, le=400)
    glycemic_index: float = Field(default=70.0, ge=0, le=110)
    food_order: FoodOrder = FoodOrder.MIXED
    label: str = Field(default="", max_length=80)


class MedicationIn(_Model):
    time: AwareDatetime
    drug: str = Field(min_length=2, max_length=64)
    dose_mg: float = Field(ge=0, le=5000)


class ActivityIn(_Model):
    start: AwareDatetime
    end: AwareDatetime
    intensity: float = Field(ge=0, le=1, description="0 = rest, 1 = brisk walking")


class EventsIn(_Model):
    glucose: list[GlucoseIn] = Field(default_factory=list, max_length=5000)
    meals: list[MealIn] = Field(default_factory=list, max_length=500)
    medications: list[MedicationIn] = Field(default_factory=list, max_length=500)
    activity: list[ActivityIn] = Field(default_factory=list, max_length=500)


class ForecastIn(_Model):
    now: AwareDatetime | None = None
    horizon_minutes: int = Field(default=240, ge=15, le=2160)
    planned_meals: list[MealIn] | None = Field(
        default=None, description="Replaces the habitual routine for this forecast (what-if)."
    )


class FingerstickPlanIn(_Model):
    now: AwareDatetime | None = None
    strips: int = Field(ge=0, le=12)
    until: AwareDatetime | None = None
    min_spacing_minutes: float = Field(default=60.0, ge=15, le=480)
    include_night_window: bool = True


class TwinCreatedOut(BaseModel):
    twin_id: str
    start: str
    most_likely_endotype: str
    endotype_probabilities: dict[str, float]
    prior: dict[str, dict[str, float]]
    disclaimer: str
