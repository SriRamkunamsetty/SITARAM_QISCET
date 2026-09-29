from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from pratibimb.cohort.endotypes import INSPIRED_ENDOTYPES, build_prior, endotype_probabilities
from pratibimb.cohort.profile import PatientProfile
from pratibimb.physiology.model import GB, SI


def _profile(**kw: float) -> PatientProfile:
    base = dict(
        age_years=50, sex="F", age_at_diagnosis=45, bmi=26, waist_cm=92, hba1c_pct=8.0, body_weight_kg=65
    )
    base.update(kw)
    return PatientProfile(**base)  # type: ignore[arg-type]


def test_published_shares_sum_to_one() -> None:
    assert math.isclose(sum(e.share for e in INSPIRED_ENDOTYPES), 1.0, abs_tol=1e-3)


def test_obese_insulin_resistant_profile_maps_to_irod() -> None:
    p = endotype_probabilities(_profile(bmi=33, waist_cm=110, hba1c_pct=8.3, age_at_diagnosis=46))
    assert max(p, key=p.__getitem__) == "IROD"
    assert math.isclose(sum(p.values()), 1.0)


def test_lean_young_high_a1c_profile_maps_to_sidd() -> None:
    p = endotype_probabilities(_profile(bmi=23, waist_cm=86, hba1c_pct=11.5, age_at_diagnosis=38))
    assert max(p, key=p.__getitem__) == "SIDD"


def test_prior_uses_fasting_glucose_and_treatment() -> None:
    plain = build_prior(_profile(fasting_glucose=130))
    on_met = build_prior(replace(_profile(fasting_glucose=130)))
    assert math.isclose(math.exp(plain.mean_log[GB]), 130, rel_tol=1e-6)
    assert np.all(on_met.sd_log > 0)


def test_wearables_shift_insulin_sensitivity_in_expected_direction() -> None:
    fit = build_prior(_profile(resting_hr=60, daily_steps=12000))
    unfit = build_prior(_profile(resting_hr=90, daily_steps=2000))
    assert fit.mean_log[SI] > unfit.mean_log[SI]
