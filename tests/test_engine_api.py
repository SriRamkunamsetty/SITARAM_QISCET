from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from pratibimb.api.app import create_app
from pratibimb.api.store import InMemoryRepository, JsonFileRepository
from pratibimb.twin.engine import DigitalTwin, TwinError, TwinRecord


def _feed(twin: DigitalTwin, start, order: str) -> None:
    items = [
        ("meal", start + timedelta(hours=2, minutes=30), 55.0),
        ("glucose", start + timedelta(hours=1), 132.0),
        ("glucose", start + timedelta(hours=4, minutes=30), 188.0),
        ("med", start + timedelta(hours=2), 2.0),
    ]
    if order == "reversed":
        items = items[::-1]
    for kind, ts, v in items:
        if kind == "meal":
            twin.add_meal(ts, v, 78.0)
        elif kind == "glucose":
            twin.add_glucose(ts, v)
        else:
            twin.add_medication(ts, "glimepiride", v)


def test_event_sourcing_is_order_independent(su_profile, start) -> None:
    now = start + timedelta(hours=6)
    a = DigitalTwin(su_profile, start, seed=5, n_particles=300)
    _feed(a, start, "sorted")
    a.sync(start + timedelta(hours=3))  # partially advance, forcing a later rebuild
    b = DigitalTwin(su_profile, start, seed=5, n_particles=300)
    _feed(b, start, "reversed")
    assert a.state(now)["glucose_now"] == pytest.approx(b.state(now)["glucose_now"])


def test_record_round_trip_reproduces_state(su_profile, start) -> None:
    now = start + timedelta(hours=6)
    twin = DigitalTwin(su_profile, start, seed=9, n_particles=300)
    _feed(twin, start, "sorted")
    rec = TwinRecord(**asdict(twin.to_record()))
    clone = DigitalTwin.from_record(rec)
    assert clone.state(now)["glucose_now"] == pytest.approx(twin.state(now)["glucose_now"])


def test_rejects_naive_and_past_times(su_profile, start) -> None:
    twin = DigitalTwin(su_profile, start, n_particles=200)
    with pytest.raises(TwinError):
        twin.add_glucose(start.replace(tzinfo=None), 120)
    with pytest.raises(TwinError):
        twin.add_glucose(start - timedelta(hours=1), 120)
    with pytest.raises(TwinError):
        twin.add_glucose(start, 900)


def test_fingerstick_plan_is_within_waking_hours(su_profile, start) -> None:
    twin = DigitalTwin(su_profile, start, seed=3, n_particles=300)
    plan = twin.plan_fingersticks(start + timedelta(minutes=30), strips=3)
    assert len(plan["recommended_times"]) <= 3
    for r in plan["recommended_times"]:
        hhmm = r["time"][11:16]
        assert "06:30" <= hhmm <= "23:00"


# ------------------------------------------------------------------------------------ API
PROFILE = {
    "age_years": 56,
    "sex": "M",
    "age_at_diagnosis": 49,
    "bmi": 25.5,
    "waist_cm": 92,
    "hba1c_pct": 7.1,
    "body_weight_kg": 68,
    "fasting_glucose": 128,
    "regimens": [
        {"drug": "metformin", "dose_mg": 500, "times": ["08:30", "21:00"]},
        {"drug": "glimepiride", "dose_mg": 2, "times": ["08:00"]},
    ],
}
START = "2026-10-01T06:00:00+05:30"


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(InMemoryRepository(), api_key="secret"))


def _hdr() -> dict[str, str]:
    return {"X-API-Key": "secret"}


def test_health_is_public(client: TestClient) -> None:
    assert client.get("/healthz").json()["status"] == "ok"


def test_auth_required(client: TestClient) -> None:
    assert client.get("/v1/twins").status_code == 401
    assert client.get("/v1/twins", headers={"X-API-Key": "nope"}).status_code == 401


def test_full_workflow(client: TestClient) -> None:
    r = client.post(
        "/v1/twins", json={"profile": PROFILE, "start": START, "n_particles": 300, "seed": 1}, headers=_hdr()
    )
    assert r.status_code == 201, r.text
    tid = r.json()["twin_id"]
    assert r.json()["most_likely_endotype"] in {"SIDD", "IROD", "CIRDD", "MARD"}

    ev = {
        "glucose": [{"time": "2026-10-01T07:00:00+05:30", "value_mg_dl": 131}],
        "meals": [
            {"time": "2026-10-01T08:30:00+05:30", "carbs_g": 55, "glycemic_index": 80, "label": "idli"}
        ],
        "medications": [{"time": "2026-10-01T08:00:00+05:30", "drug": "glimepiride", "dose_mg": 2}],
    }
    r = client.post(f"/v1/twins/{tid}/events", json=ev, headers=_hdr())
    assert r.status_code == 202 and r.json()["accepted"]["glucose"] == 1

    r = client.get(f"/v1/twins/{tid}/state", params={"now": "2026-10-01T09:00:00+05:30"}, headers=_hdr())
    assert r.status_code == 200
    body = r.json()
    assert body["readings_assimilated"] == 1 and "si" in body["parameters"]

    r = client.post(
        f"/v1/twins/{tid}/forecast",
        json={"now": "2026-10-01T09:00:00+05:30", "horizon_minutes": 120},
        headers=_hdr(),
    )
    assert r.status_code == 200
    fc = r.json()
    assert len(fc["median"]) == len(fc["times"]) == 25
    assert all(lo <= mid <= hi for lo, mid, hi in zip(fc["p10"], fc["median"], fc["p90"], strict=True))
    assert 0 <= fc["risk"]["hypo_lt70_15min"] <= 1

    r = client.post(
        f"/v1/twins/{tid}/fingerstick-plan",
        json={"now": "2026-10-01T09:00:00+05:30", "strips": 2},
        headers=_hdr(),
    )
    assert r.status_code == 200 and len(r.json()["recommended_times"]) <= 2

    rec = client.get(f"/v1/twins/{tid}/record", headers=_hdr()).json()
    assert len(rec["glucose"]) == 1 and len(rec["meals"]) == 1

    assert client.delete(f"/v1/twins/{tid}", headers=_hdr()).status_code == 204
    assert client.get(f"/v1/twins/{tid}/state", headers=_hdr()).status_code == 404


def test_validation_errors(client: TestClient) -> None:
    bad_profile = {**PROFILE, "age_at_diagnosis": 70}
    r = client.post("/v1/twins", json={"profile": bad_profile, "start": START}, headers=_hdr())
    assert r.status_code == 422
    r = client.post("/v1/twins", json={"profile": PROFILE, "start": "2026-10-01T06:00:00"}, headers=_hdr())
    assert r.status_code == 422  # naive datetime rejected


def test_file_repository_persists_across_instances(tmp_path) -> None:
    c1 = TestClient(create_app(JsonFileRepository(tmp_path), api_key=""))
    tid = c1.post(
        "/v1/twins", json={"profile": PROFILE, "start": START, "n_particles": 200, "seed": 2}
    ).json()["twin_id"]
    c1.post(
        f"/v1/twins/{tid}/events",
        json={"glucose": [{"time": "2026-10-01T07:00:00+05:30", "value_mg_dl": 120}]},
    )
    c2 = TestClient(create_app(JsonFileRepository(tmp_path), api_key=""))
    assert tid in c2.get("/v1/twins").json()["twin_ids"]
    s = c2.get(f"/v1/twins/{tid}/state", params={"now": "2026-10-01T08:00:00+05:30"}).json()
    assert s["readings_assimilated"] == 1
