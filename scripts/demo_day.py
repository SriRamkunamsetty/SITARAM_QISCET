"""End-to-end demo of the API on one virtual patient ("Ramesh"), using the real HTTP contract.

Simulates a day: the twin is created from an EHR summary, recommends when to test, receives
those fingerstick readings (from the virtual patient's true glucose), and at bedtime reports
overnight hypoglycaemia risk. Prints a clinician-style summary.

  uv run python scripts/demo_day.py
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
from fastapi.testclient import TestClient

from pratibimb.api.app import create_app
from pratibimb.api.store import InMemoryRepository
from pratibimb.cohort.virtual import (
    SMBG_SENSOR,
    generate_day,
    sample_virtual_patient,
    sensor_reading,
    simulate_truth,
)

IST = timezone(timedelta(hours=5, minutes=30))
DAY0 = datetime(2026, 10, 12, 0, 0, tzinfo=IST)


def iso(t_min: float) -> str:
    return (DAY0 + timedelta(minutes=float(t_min))).isoformat()


def main() -> None:
    rng = np.random.default_rng(7)
    vp = sample_virtual_patient(rng, "ramesh", endotype="MARD", force_sulfonylurea=True)
    days = [generate_day(vp, d, rng) for d in range(3)]
    truth = simulate_truth(vp, days, rng)
    p = vp.profile
    client = TestClient(create_app(InMemoryRepository(), api_key=""))

    profile = {
        "age_years": round(p.age_years),
        "sex": "M",
        "age_at_diagnosis": round(p.age_at_diagnosis),
        "bmi": round(p.bmi, 1),
        "waist_cm": round(p.waist_cm),
        "hba1c_pct": p.hba1c_pct,
        "body_weight_kg": round(p.body_weight_kg, 1),
        "fasting_glucose": p.fasting_glucose,
        "resting_hr": round(p.resting_hr or 72),
        "daily_steps": round(p.daily_steps or 5000),
        "regimens": [{"drug": r.drug, "dose_mg": r.dose_mg, "times": list(r.times_hhmm)} for r in p.regimens],
    }
    tid = client.post("/v1/twins", json={"profile": profile, "start": iso(0), "seed": 1}).json()["twin_id"]
    print(
        f"Twin {tid} | HbA1c {p.hba1c_pct}% | regimen: {[r.drug + ' ' + str(r.dose_mg) for r in p.regimens]}"
    )

    # Day 1: log meals/doses as they happen; test when the twin asks (3 strips).
    day = 1
    base = day * 1440
    rec = days[day].logged_events
    events = {
        "meals": [
            {
                "time": iso(m.t_min),
                "carbs_g": round(m.carbs_g),
                "glycemic_index": m.glycemic_index,
                "label": m.label,
            }
            for m in rec.meals
        ],
        "medications": [
            {"time": iso(d.t_min), "drug": d.drug, "dose_mg": d.dose_mg} for d in rec.medications
        ],
    }
    for prev in days[:day]:  # history
        events["meals"] += [
            {"time": iso(m.t_min), "carbs_g": round(m.carbs_g), "glycemic_index": m.glycemic_index}
            for m in prev.logged_events.meals
        ]
        events["medications"] += [
            {"time": iso(d.t_min), "drug": d.drug, "dose_mg": d.dose_mg}
            for d in prev.logged_events.medications
        ]
    client.post(f"/v1/twins/{tid}/events", json=events)

    now = base + 390  # 06:30 wake-up
    strips = 3
    while strips > 0:
        plan = client.post(
            f"/v1/twins/{tid}/fingerstick-plan", json={"now": iso(now), "strips": strips}
        ).json()
        if not plan["recommended_times"]:
            break
        nxt = plan["recommended_times"][0]["time"]
        t_next = (datetime.fromisoformat(nxt) - DAY0).total_seconds() / 60
        value = sensor_reading(truth.at(t_next), SMBG_SENSOR, rng)
        client.post(f"/v1/twins/{tid}/events", json={"glucose": [{"time": nxt, "value_mg_dl": round(value)}]})
        print(
            f"  twin asked for a strip at {nxt[11:16]} -> meter {value:.0f} mg/dL "
            f"(gain {plan['recommended_times'][0]['information_gain_bits']:.3f} bits)"
        )
        now, strips = t_next + 5, strips - 1

    bedtime = base + 22 * 60 + 30
    fc = client.post(f"/v1/twins/{tid}/forecast", json={"now": iso(bedtime), "horizon_minutes": 480}).json()
    state = client.get(f"/v1/twins/{tid}/state", params={"now": iso(bedtime)}).json()
    night = truth.glucose[(truth.t_min >= base + 1440) & (truth.t_min < base + 1440 + 390)]
    si = state["parameters"]["si"]
    print(
        f"\nBedtime twin estimate: {state['glucose_now']['median']:.0f} mg/dL "
        f"(80% band {state['glucose_now']['p10']:.0f}-{state['glucose_now']['p90']:.0f}); "
        f"true {truth.at(bedtime):.0f}"
    )
    print(f"Insulin sensitivity: {si['median']:.2e} ({si['change_vs_prior_pct']:+.0f}% vs EHR prior)")
    print(
        f"Overnight risk: P(glucose <70 for 15+ min) = {fc['risk']['hypo_lt70_15min']:.0%}, "
        f"P(<54) = {fc['risk']['hypo_lt54_15min']:.0%}"
    )
    print(f"What actually happened overnight: min {night.min():.0f} mg/dL")


if __name__ == "__main__":
    main()
