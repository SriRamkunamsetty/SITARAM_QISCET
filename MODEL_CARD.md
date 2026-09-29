# Model card — Pratibimb metabolic twin v0.1

## Intended use
Research prototype for the Digital Twin Challenge 2026. It illustrates how a clinician-facing
tool could (1) maintain a personalised physiological model of a person with Type 2 Diabetes
from an EHR summary plus sparse fingerstick or CGM readings, (2) forecast glucose with
uncertainty, and (3) recommend *when* to test so a limited strip budget is most informative
about hypoglycaemia risk.

**Not intended for:** diagnosis, insulin or drug dosing, emergency detection, or any use without
clinician oversight. Not a medical device; not reviewed by CDSCO or any regulator.

## Model
- **Physiology:** extended Bergman minimal model with two-chain gut absorption, sulfonylurea
  PK/PD (glucose-independent secretion), metformin effect on basal glucose, exercise
  (acute uptake + prolonged insulin sensitisation), circadian insulin sensitivity, dawn
  phenomenon, sleep-blunted counter-regulation. 10 states, 6 patient-specific parameters.
- **Personalisation:** INSPIRED endotype soft-assignment from EHR → log-normal prior.
- **Data assimilation:** joint state/parameter particle filter (default 1,000 particles),
  Student-t likelihood (robust to bad readings), Liu–West kernel resampling, slow random walks on
  insulin sensitivity, basal glucose and SU response.
- **Active sensing:** greedy maximisation of weighted mutual information between future readings
  and hypoglycaemia events in upcoming risk windows (nested Monte Carlo on the posterior-predictive
  ensemble), re-planned after each reading.

## Evaluation
See `results/strip_budget/summary.md` (synthetic cohort, known ground truth, common random
numbers across policies, patient-level bootstrap CIs). Real-data validation on ShanghaiT2DM is
the next milestone.

## Known limitations and failure modes
- Requires meals and SU doses to be logged; unlogged large meals degrade forecasts until the next
  reading (the filter absorbs them only through glucose data).
- Structural model error: no explicit hepatic insulin sensitivity, glucagon, incretin, DPP-4i or
  SGLT2i dynamics yet (these drugs are accepted but have no mechanistic effect in v0.1).
- Uncertainty bands are model-based and **not yet conformally calibrated**; coverage must be
  checked on real data before bands are shown to clinicians as calibrated.
- Night-time hypoglycaemia is under-represented in the synthetic cohort (see DATA.md).
- Particle degeneracy is possible with very dense data and a poorly matched prior; the API
  exposes the effective sample size so it can be monitored.

## Safety design
No dosing outputs anywhere in the API; every response carries a disclaimer; the full event log
of every twin is exportable (`GET /v1/twins/{id}/record`) and replays deterministically, so every
number the twin has reported can be audited.
