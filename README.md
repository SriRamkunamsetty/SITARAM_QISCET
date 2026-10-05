# Pratibimb (प्रतिबिंब) — a metabolic digital twin that decides when to test

**Digital Twin Challenge 2026 · Happiest Health**
India-calibrated Type 2 Diabetes twin that fuses an EHR summary with sparse fingerstick (or CGM)
and wearable data, forecasts glucose with uncertainty, and **tells the patient *when* to prick
their finger** so a small strip budget reveals the most about upcoming hypoglycaemia risk.

> ⚠️ Research prototype. Not a medical device. No dosing advice. See [MODEL_CARD.md](MODEL_CARD.md).

## ✅ Submission checklist (Digital Twin Challenge 2026)

| # | Requirement | Status | Where |
|---|---|---|---|
| 1 | Team details | ✅ Done | [Team details](#1-team-details) |
| 2 | College / incubator information | ✅ Done | [College / incubator](#2-college--incubator-information) |
| 3 | Project title | ✅ Done | [Project title](#3-project-title) |
| 4 | Problem statement | ✅ Done | [Problem statement](#4-problem-statement) |
| 5 | Healthcare use case | ✅ Done | [Healthcare use case](#5-healthcare-use-case) |
| 6 | Technical stack | ✅ Done | [Technical stack](#9-technical-stack) |
| 7 | AI/ML model or framework details | ✅ Done | [AI/ML model details](#8-aiml-model-and-framework-details) |
| 8 | 15–20 minute demo video (unlisted YouTube), link in README | ⬜ To do | [Demo video](#11-demo-video) |
| 9 | Open-source licence details | ✅ Done | [Licence](#15-open-source-licence) |
| 10 | Architecture diagram (PDF/PPT) | ✅ Done | [`docs/architecture.pdf`](docs/architecture.pdf) |
| 11 | Presentation (PDF/PPT) covering project details and outcomes | ✅ Done | [`docs/presentation.pdf`](docs/presentation.pdf), [`.pptx`](docs/presentation.pptx) |
| 12 | All files and links publicly accessible without extra permissions | ⬜ To do | [Accessibility](#16-public-accessibility) |

### To-do before submission (deadline: 20 Oct 2026, 7:00 PM IST)

- [ ] Record the **15–20 minute demo video**, upload it to YouTube as **Unlisted**, and paste the link in [Demo video](#11-demo-video)
- [x] ~~Create the **architecture diagram**~~ → [`docs/architecture.pdf`](docs/architecture.pdf)
- [x] ~~Create the **presentation**~~ → [`docs/presentation.pdf`](docs/presentation.pdf) and [`docs/presentation.pptx`](docs/presentation.pptx)
- [x] ~~Rename the repository to `TeamName_CollegeName`~~ → `SITARAM_QISCET`
- [ ] Make the repository **public** and check that every link (video, PDFs) opens in a logged-out/incognito browser
- [ ] Validate on real open data (ShanghaiT2DM) and update [Results](#10-results--strip-budget-experiment)
- [ ] Submit the team leader's name, phone number, email and the public repo link on the submission platform

---

## 1. Team details

**Team name:** SITARAM

| Name | Role | College |
|---|---|---|
| Mohan Sriram Kunamsetty | Team Leader | QIS College of Engineering and Technology |
| Mohana Lakshmi Srija Kunamsetty | Team Member | RISE Krishna Sai College of Engineering and Technology |

## 2. College / incubator information

This is an **inter-college team**, which the challenge rules allow:

- **QIS College of Engineering and Technology**: Mohan Sriram Kunamsetty (Team Leader)
- **RISE Krishna Sai College of Engineering and Technology**: Mohana Lakshmi Srija Kunamsetty (Team Member)

## 3. Project title

**Pratibimb: an India-calibrated metabolic digital twin for Type 2 Diabetes that decides when to test**

## 4. Problem statement

India has ~101 million people with diabetes (ICMR-INDIAB-17). Sulfonylureas are cheap and widely
prescribed, and they cause hypoglycaemia that is frequently **nocturnal and asymptomatic**
(Middleton et al., *Diabetes Care* 2017: 67% nocturnal, 73% asymptomatic). A CGM would reveal it,
but a sensor costs ₹4,200–5,250 per 14 days (≈ ₹9,000/month). Most patients therefore rely on a few
fingersticks taken at habitual times (fasting, 2-h post-breakfast), and these rarely coincide with risk.

**Question:** if a patient can afford only 1–4 strips a day, *when* should they test so that
dangerous lows are predicted before they happen?

## 5. Healthcare use case

1. **Enrol:** a doctor creates the patient's twin from the EHR summary (age, BMI, waist, HbA1c,
   fasting glucose, medication regimen) plus an optional smartwatch summary (resting HR, steps).
2. **Personalise:** the twin starts from an India-specific physiological prior (INSPIRED endotypes).
   It then learns the patient's own insulin sensitivity, basal glucose and sulfonylurea response
   from each fingerstick or CGM reading, together with the logged meals and doses.
3. **Plan:** each morning the twin recommends the **most informative testing times** for the
   patient's strip budget, and re-plans after every reading.
4. **Warn:** at bedtime it reports **overnight hypoglycaemia risk** with an uncertainty band, so the
   doctor can review the sulfonylurea dose or advise a bedtime snack.
5. **Audit:** every number the twin reports can be traced back to the exact readings and events
   behind it.

## 6. Why this is a digital twin (NASEM 2023 definition)

| NASEM element | Pratibimb |
|---|---|
| Virtual representation of a specific patient | 10-state glucose–insulin–gut–drug model with 6 patient-specific parameters |
| Dynamically updated from the physical twin | Joint state/parameter particle filter assimilates every reading (SMBG or CGM) |
| Predictive capability | Posterior-predictive glucose trajectories and event probabilities |
| Informs decisions | Active-sensing planner (when to test), overnight risk at bedtime |
| Two-way flow | The twin *requests* data from the patient (test at 16:30) and updates on what it receives |
| VVUQ | Unit-tested physics, known-truth evaluation, bootstrap CIs, heavy-tailed likelihood, full audit log |

## 7. How it works (architecture)

```mermaid
flowchart LR
    EHR["EHR summary<br/>age · BMI · waist · HbA1c · FPG · regimen"] --> P["Prior builder<br/>INSPIRED endotypes<br/>(SIDD / IROD / CIRDD / MARD)"]
    W["Wearable summary<br/>resting HR · steps"] --> P
    P --> PF["Particle-filter twin<br/>state + parameters"]
    LOG["Meal · dose · activity log"] --> PF
    G["Fingerstick / CGM readings"] --> PF
    PF --> F["Forecast<br/>glucose bands + hypo/hyper risk"]
    PF --> PL["Active-sensing planner<br/>max mutual information"]
    PL -->|"test at 16:30"| PT(("Patient"))
    PT -->|reading| G
    F --> DOC["Clinician API / dashboard"]
```

1. **Physiology** (`pratibimb/physiology/model.py`): extended Bergman minimal model with fast/slow
   carbohydrate absorption (GI- and food-order-aware), sulfonylurea PK/PD (glucose-independent
   insulin secretion, the hypoglycaemia mechanism), metformin, exercise, circadian insulin
   sensitivity, dawn phenomenon and sleep-blunted counter-regulation. Vectorised RK4.
2. **India-specific prior** (`cohort/endotypes.py`): soft assignment to the four INSPIRED Indian T2D
   endotypes (n = 19,084) → moment-matched log-normal prior; HOMA-IR and HOMA-B per endotype seed
   insulin sensitivity and beta-cell/SU response.
3. **Data assimilation** (`twin/particle_filter.py`): joint state/parameter particle filter,
   Student-t likelihood (robust to bad readings), Liu–West resampling, slow drift of insulin
   sensitivity, basal glucose and SU response.
4. **Active sensing** (`sensing/planner.py`): chooses reading times that maximise
   `Σ_w weight_w · I(Y_S ; hypo in window w)`, estimated by nested Monte Carlo over the twin's own
   forecast ensemble; greedy, re-planned after every reading; readings count only for risk calls
   made after them (e.g. the bedtime overnight decision).
5. **Event-sourced engine + API** (`twin/engine.py`, `api/`): every input is an immutable event;
   late data trigger deterministic replay; the full record is exportable for audit.

## 8. AI/ML model and framework details

Pratibimb is a **hybrid physiological + probabilistic AI** system. It does not use a black-box
classifier. Each component is interpretable and uncertainty-aware:

| Component | Method | Why |
|---|---|---|
| Mechanistic core | Extended Bergman minimal model: 10 states, 6 patient parameters, vectorised RK4 ODE solver | Physiologically meaningful parameters; valid what-if simulation |
| Personalised prior | Gaussian naive-Bayes soft assignment to the 4 INSPIRED Indian T2D endotypes, then a moment-matched log-normal prior; wearable covariates (resting HR, steps) shift insulin sensitivity | A sensible twin from day 1, specific to Indian patients |
| Data assimilation (learning) | Sequential Monte Carlo: joint state + parameter particle filter, 1,000 particles, Student-t likelihood, Liu–West kernel resampling, random-walk parameter drift | Learns each patient online from sparse, noisy data, with full posterior uncertainty |
| Prediction | Posterior-predictive ensemble forecasts; event probabilities (glucose < 70 mg/dL or < 54 mg/dL for ≥ 15 min; > 250 mg/dL) | Calibrated risk instead of a point guess |
| Decision AI (active sensing) | Bayesian experimental design: greedy maximisation of mutual information between future readings and hypoglycaemia events, estimated by nested Monte Carlo | Chooses **when** to test for the most information per strip |
| Evaluation | Known-truth virtual cohort, common random numbers across policies, AUROC, Brier score, patient-level paired bootstrap | Honest comparison against standard testing schedules |

**Frameworks:** NumPy (all modelling and inference, no GPU required), FastAPI + Pydantic v2 (service),
matplotlib (evaluation plots).

## 9. Technical stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Modelling, inference, planning | NumPy |
| API | FastAPI, Pydantic v2, Uvicorn |
| Persistence | Event-sourced JSON event logs (replayable audit trail) |
| Packaging | uv, hatchling |
| Quality | pytest (29 tests), ruff, mypy |
| CI/CD | GitHub Actions (Python 3.11 and 3.12, Docker build) |
| Deployment | Docker (non-root, health check, persistent volume) |

## 10. Results — strip-budget experiment

40 sulfonylurea-treated virtual Indian patients (HbA1c ≤ 8%), 7 evaluation days each, 1,120 risk
windows. At the start of each window (morning, afternoon, evening, and night at 22:30 bedtime) the twin
predicts P(glucose < 70 mg/dL for ≥ 15 min). Same truth, logs, meter noise and filter seed for
every policy. Full table: [`results/strip_budget/summary.md`](results/strip_budget/summary.md).

![strip budget](results/strip_budget/strip_budget.png)

| Strip budget | Policy | Strips actually used/day | AUROC | Brier ↓ | Sensitivity @ p≥0.2 |
|---|---|---|---|---|---|
| 1 | Fixed (fasting) | 1.0 | 0.954 | 0.0251 | 0.72 |
| 1 | **Twin-scheduled** | **0.8** | **0.985** | **0.0167** | **0.79** |
| 2 | Fixed (FBS + PPBS) | 2.0 | 0.974 | 0.0223 | 0.81 |
| 2 | **Twin-scheduled** | **1.4** | **0.987** | **0.0165** | **0.85** |
| — | CGM every 15 min (reference) | 65 | 0.990 | 0.0153 | 0.81 |
| — | No readings (prior + logs) | 0 | 0.856 | 0.0368 | 0.19 |

**Findings.** Twin-scheduled testing improves hypoglycaemia risk prediction over the standard
fasting / post-breakfast schedule at every budget. The paired, patient-level bootstrap Brier
improvement excludes zero at every budget: at 1 strip it is +0.008 [95% CI +0.002, +0.016]. The twin
also uses *fewer* strips: it skips readings that would not change the risk estimate. **Under one
strip a day, its risk prediction comes close to a CGM's.** At 3-4 strips a day a *random* schedule
does about as well as the twin (the Brier difference vs random includes zero), so the advantage is
specific to very small strip budgets, which is the realistic case for patients who cannot afford a CGM.

**Trade-off.** The planner optimises information about *hypoglycaemia*, not general glucose tracking.
Mean tracking error is higher than the fixed schedule's (e.g. 14.1 vs 10.4 mg/dL at a budget of 3).
The objective can be re-weighted toward time-in-range if a clinician wants that instead.

**Limitations.** Read these before quoting any number:
1. Hypoglycaemia was rare in this cohort: only **5 of 40 patients** had any low window, 47 events in
   total. AUROC therefore mainly reflects *who* is at risk, and the evidence rests on few patients.
2. The data are synthetic. The twin and the simulator share model structure, though the simulator
   deliberately adds mismatch.
3. Night-time lows are under-represented in the synthetic cohort (see DATA.md).
4. Real-data validation (ShanghaiT2DM) is the next milestone.

## 11. Demo video

**Length:** 15–20 minutes · **Platform:** YouTube (Unlisted)

> ⬜ **To do:** add the link here, e.g. `https://youtu.be/XXXXXXXXXXX`

## 12. Architecture diagram (PDF/PPT)

📄 **[`docs/architecture.pdf`](docs/architecture.pdf)** (vector PDF) · PNG: [`docs/figures/architecture.png`](docs/figures/architecture.png)

![architecture](docs/figures/architecture.png)

## 13. Presentation (PDF/PPT)

📄 **[`docs/presentation.pdf`](docs/presentation.pdf)** · editable: [`docs/presentation.pptx`](docs/presentation.pptx) (13 slides with speaker notes for the video)

Covers the problem, idea, healthcare use case, why it is a digital twin, architecture, AI/ML methods, evaluation design, results, limitations, a conceptual clinician dashboard, safety and regulation, and the roadmap.

**Conceptual clinician dashboard** (drawn from real twin output on a synthetic patient; the interactive dashboard is on the roadmap):

![dashboard mock-up](docs/figures/dashboard_mockup.png)

## 14. Quick start

```bash
uv sync --all-extras          # Python 3.11+, installs deps into .venv
uv run pytest                 # 29 tests: physiology, endotypes, filter, planner, engine, API
uv run python scripts/demo_day.py                     # one simulated day through the HTTP API
uv run uvicorn pratibimb.api.app:app --reload         # API + interactive docs at /docs
uv run pratibimb-strip-experiment --patients 40 --days 7 --workers 8   # reproduce results
uv run python scripts/build_docs.py                   # rebuild architecture PDF, dashboard mock-up, slides
powershell -File scripts/export_pdf.ps1               # slides -> PDF (Windows + PowerPoint)
```

Docker: `docker build -t pratibimb . && docker run -p 8000:8000 -v twins:/data/twins pratibimb`

### API (OpenAPI docs at `/docs`)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/twins` | Create a twin from an EHR profile (returns endotype probabilities + prior) |
| `POST` | `/v1/twins/{id}/events` | Ingest glucose (SMBG/CGM), meals, medications, activity |
| `GET` | `/v1/twins/{id}/state` | Current glucose estimate + posterior physiology vs prior |
| `POST` | `/v1/twins/{id}/forecast` | Glucose quantile bands and hypo/hyper risk; optional what-if meals |
| `POST` | `/v1/twins/{id}/fingerstick-plan` | Recommended testing times for a strip budget |
| `GET` | `/v1/twins/{id}/record` | Full event log (audit; replays deterministically) |
| `DELETE` | `/v1/twins/{id}` | Erase a twin (DPDP right to erasure) |

Configuration: `PRATIBIMB_API_KEY` (require `X-API-Key`), `PRATIBIMB_DATA_DIR` (persist event logs),
`PRATIBIMB_CORS_ORIGINS`. All timestamps must carry a timezone (e.g. `+05:30`).

## 15. Open-source licence

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
Anyone may use, modify and distribute the code, including commercially. The licence includes an
express patent grant and requires attribution. Third-party datasets are **not** redistributed;
each has its own licence (see [DATA.md](DATA.md)).

## 16. Public accessibility

> ⬜ **To do before submission:** make this repository public, and check that the video, the
> architecture diagram and the presentation all open without logging in.

## 17. Data, privacy and regulation

- Only synthetic and open data; no clinical data in the repo ([DATA.md](DATA.md)).
- Designed toward the DPDP Act 2023 / Rules 2025: data minimisation (no names or identifiers in
  the profile), erasure endpoint, full audit trail. Consent management is out of scope for v0.1.
- CDSCO's Draft Guidance on Medical Device Software (21 Oct 2025) would classify a clinical version
  as SaMD; this prototype makes no clinical claims.

## 18. Roadmap

- [ ] Real-data validation on ShanghaiT2DM (`scripts/download_data.py shanghai`) and CGMacros
- [ ] Conformal calibration of forecast bands; calibration plots
- [ ] Clinician dashboard (triage, AGP, forecast, strip planner, what-if thali simulator)
- [ ] Mechanistic DPP-4i / SGLT2i effects; fasting-day (vrat / Ramadan) scenario
- [ ] ABDM FHIR R4 import of the EHR profile

## References

Key sources are cited inline and in `research/Pratibimb_Research_Brief_v2.md`.
