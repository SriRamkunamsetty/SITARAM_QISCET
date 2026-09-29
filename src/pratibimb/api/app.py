"""FastAPI service exposing the digital twin.

Configuration (environment variables):
  PRATIBIMB_API_KEY       if set, every /v1 request must send header ``X-API-Key``
  PRATIBIMB_DATA_DIR      if set, twins persist as JSON event logs in this directory
  PRATIBIMB_CORS_ORIGINS  comma-separated origins allowed to call the API (dashboard)

Run:  uv run uvicorn pratibimb.api.app:app --reload
"""

from __future__ import annotations

import hmac
import logging
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pratibimb import DISCLAIMER, __version__
from pratibimb.api.schemas import (
    CreateTwinIn,
    EventsIn,
    FingerstickPlanIn,
    ForecastIn,
    PatientProfileIn,
    TwinCreatedOut,
)
from pratibimb.api.store import InMemoryRepository, JsonFileRepository, TwinRepository
from pratibimb.cohort.profile import DEFAULT_INDIAN_MEALS, HabitualMeal, PatientProfile, Regimen
from pratibimb.physiology.inputs import EventLog, Meal
from pratibimb.twin.engine import DigitalTwin, TwinError

log = logging.getLogger("pratibimb.api")


def profile_from_schema(p: PatientProfileIn) -> PatientProfile:
    meals = (
        tuple(HabitualMeal(m.time, m.carbs_g, m.glycemic_index, m.label) for m in p.habitual_meals)
        if p.habitual_meals
        else DEFAULT_INDIAN_MEALS
    )
    return PatientProfile(
        age_years=p.age_years,
        sex=p.sex,
        age_at_diagnosis=p.age_at_diagnosis,
        bmi=p.bmi,
        waist_cm=p.waist_cm,
        hba1c_pct=p.hba1c_pct,
        body_weight_kg=p.body_weight_kg,
        fasting_glucose=p.fasting_glucose,
        triglycerides=p.triglycerides,
        hdl=p.hdl,
        resting_hr=p.resting_hr,
        daily_steps=p.daily_steps,
        regimens=tuple(Regimen(r.drug, r.dose_mg, tuple(r.times)) for r in p.regimens),
        habitual_meals=meals,
        sleep_hhmm=p.sleep_time,
        wake_hhmm=p.wake_time,
    )


def create_app(repository: TwinRepository | None = None, api_key: str | None = None) -> FastAPI:
    if repository is None:
        data_dir = os.environ.get("PRATIBIMB_DATA_DIR")
        repository = JsonFileRepository(Path(data_dir)) if data_dir else InMemoryRepository()
    api_key = api_key if api_key is not None else os.environ.get("PRATIBIMB_API_KEY") or None
    repo: TwinRepository = repository

    app = FastAPI(
        title="Pratibimb Digital Twin API",
        version=__version__,
        description=(
            "India-calibrated metabolic digital twin for Type 2 Diabetes with an active-sensing "
            f"fingerstick planner. {DISCLAIMER}"
        ),
    )
    origins = [o.strip() for o in os.environ.get("PRATIBIMB_CORS_ORIGINS", "").split(",") if o.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

    def require_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if api_key and not (x_api_key and hmac.compare_digest(x_api_key, api_key)):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or invalid X-API-Key")

    def load(twin_id: str) -> DigitalTwin:
        try:
            return repo.get(twin_id)
        except KeyError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"twin {twin_id} not found") from None

    def resolve_now(twin: DigitalTwin, now: datetime | None) -> datetime:
        return now or datetime.now(tz=twin.anchor.tzinfo)

    @app.exception_handler(TwinError)
    async def twin_error_handler(_: Request, exc: TwinError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    guarded = [Depends(require_key)]

    @app.post(
        "/v1/twins", status_code=201, response_model=TwinCreatedOut, dependencies=guarded, tags=["twins"]
    )
    def create_twin(body: CreateTwinIn) -> TwinCreatedOut:
        twin = DigitalTwin(
            profile_from_schema(body.profile), body.start, seed=body.seed, n_particles=body.n_particles
        )
        repo.save(twin)
        log.info("created twin %s (endotype %s)", twin.twin_id, twin.prior.most_likely_endotype)
        return TwinCreatedOut(
            twin_id=twin.twin_id,
            start=twin.anchor.isoformat(),
            most_likely_endotype=twin.prior.most_likely_endotype,
            endotype_probabilities=twin.prior.endotype_probs,
            prior=twin.prior.summary(),
            disclaimer=DISCLAIMER,
        )

    @app.get("/v1/twins", dependencies=guarded, tags=["twins"])
    def list_twins() -> dict[str, list[str]]:
        return {"twin_ids": repo.list_ids()}

    @app.delete("/v1/twins/{twin_id}", status_code=204, dependencies=guarded, tags=["twins"])
    def delete_twin(twin_id: str) -> None:
        try:
            repo.delete(twin_id)
        except KeyError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"twin {twin_id} not found") from None

    @app.post("/v1/twins/{twin_id}/events", status_code=202, dependencies=guarded, tags=["data"])
    def ingest(twin_id: str, body: EventsIn) -> dict[str, Any]:
        twin = load(twin_id)
        for g in body.glucose:
            twin.add_glucose(g.time, g.value_mg_dl, g.source)
        for m in body.meals:
            twin.add_meal(m.time, m.carbs_g, m.glycemic_index, m.food_order, m.label)
        for d in body.medications:
            twin.add_medication(d.time, d.drug, d.dose_mg)
        for a in body.activity:
            twin.add_activity(a.start, a.end, a.intensity)
        repo.save(twin)
        return {
            "accepted": {
                "glucose": len(body.glucose),
                "meals": len(body.meals),
                "medications": len(body.medications),
                "activity": len(body.activity),
            }
        }

    @app.get("/v1/twins/{twin_id}/state", dependencies=guarded, tags=["twin outputs"])
    def get_state(twin_id: str, now: Annotated[datetime | None, Query()] = None) -> dict[str, Any]:
        twin = load(twin_id)
        if now is not None and now.tzinfo is None:
            raise TwinError("now must include a timezone offset")
        return twin.state(resolve_now(twin, now))

    @app.post("/v1/twins/{twin_id}/forecast", dependencies=guarded, tags=["twin outputs"])
    def forecast(twin_id: str, body: ForecastIn) -> dict[str, Any]:
        twin = load(twin_id)
        planned = None
        if body.planned_meals is not None:
            planned = EventLog(
                meals=[
                    Meal(twin.to_min(m.time), m.carbs_g, m.glycemic_index, m.food_order, m.label)
                    for m in body.planned_meals
                ]
            )
        return twin.forecast(resolve_now(twin, body.now), body.horizon_minutes, planned)

    @app.post("/v1/twins/{twin_id}/fingerstick-plan", dependencies=guarded, tags=["twin outputs"])
    def fingerstick_plan(twin_id: str, body: FingerstickPlanIn) -> dict[str, Any]:
        twin = load(twin_id)
        return twin.plan_fingersticks(
            resolve_now(twin, body.now),
            body.strips,
            until=body.until,
            min_spacing_min=body.min_spacing_minutes,
            include_night_window=body.include_night_window,
        )

    @app.get("/v1/twins/{twin_id}/record", dependencies=guarded, tags=["audit"])
    def export_record(twin_id: str) -> dict[str, Any]:
        """Full event log (audit trail); replaying it reproduces the twin exactly."""
        return asdict(load(twin_id).to_record())

    return app


app = create_app()
