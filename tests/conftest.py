from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from pratibimb.cohort.profile import PatientProfile, Regimen

IST = timezone(timedelta(hours=5, minutes=30))


@pytest.fixture
def ist() -> timezone:
    return IST


@pytest.fixture
def su_profile() -> PatientProfile:
    """A MARD-like patient on metformin + glimepiride (typical hypo-risk profile)."""
    return PatientProfile(
        age_years=56,
        sex="M",
        age_at_diagnosis=49,
        bmi=25.5,
        waist_cm=92,
        hba1c_pct=7.1,
        body_weight_kg=68,
        fasting_glucose=128,
        resting_hr=74,
        daily_steps=6000,
        regimens=(
            Regimen("metformin", 500, ("08:30", "21:00")),
            Regimen("glimepiride", 2, ("08:00",)),
        ),
    )


@pytest.fixture
def start() -> datetime:
    return datetime(2026, 10, 1, 6, 0, tzinfo=IST)
