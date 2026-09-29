"""Expand a patient's habitual routine (meals + standing prescriptions) into timed events."""

from __future__ import annotations

import math

from pratibimb.cohort.profile import PatientProfile, hhmm_to_minutes
from pratibimb.physiology.inputs import EventLog, Meal, MedicationDose
from pratibimb.physiology.model import MINUTES_PER_DAY


def _occurrences(minute_of_day: float, t_start: float, t_end: float, tod_offset: float) -> list[float]:
    """Times t in [t_start, t_end) with (tod_offset + t) % 1440 == minute_of_day."""
    first_day = math.floor((t_start + tod_offset) / MINUTES_PER_DAY) - 1
    last_day = math.floor((t_end + tod_offset) / MINUTES_PER_DAY) + 1
    out = []
    for day in range(first_day, last_day + 1):
        t = day * MINUTES_PER_DAY + minute_of_day - tod_offset
        if t_start <= t < t_end:
            out.append(t)
    return out


def routine_events(
    profile: PatientProfile,
    t_start: float,
    t_end: float,
    tod_offset_min: float = 0.0,
    include_meals: bool = True,
    include_medications: bool = True,
) -> EventLog:
    """Habitual meals and scheduled doses within [t_start, t_end)."""
    log = EventLog()
    if include_meals:
        for hm in profile.habitual_meals:
            for t in _occurrences(hhmm_to_minutes(hm.time_hhmm), t_start, t_end, tod_offset_min):
                log.meals.append(Meal(t, hm.carbs_g, hm.glycemic_index, label=hm.label))
    if include_medications:
        for reg in profile.regimens:
            if not reg.is_sulfonylurea:
                continue  # non-secretagogues act through constant effects, not timed impulses
            for mod in reg.minutes_of_day():
                for t in _occurrences(mod, t_start, t_end, tod_offset_min):
                    log.medications.append(MedicationDose(t, reg.drug, reg.dose_mg))
    log.meals.sort(key=lambda m: m.t_min)
    log.medications.sort(key=lambda d: d.t_min)
    return log
