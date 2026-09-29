# Pratibimb v2 — Deep Research Brief & Build Blueprint
**Digital Twin Challenge 2026 (Happiest Health)** · Prepared 23 Sep 2026 · Phase 1 deadline **20 Oct 2026, 7:00 PM IST** (27 days)

This builds on your two earlier briefs. It (1) fact-checks them against primary sources, (2) re-maps the competition as it stands today, and (3) adds six research-grade ideas from 2025–2026 work that no other team is using. Everything marked ✅ was checked against a primary source during this session.

---

## 0. TL;DR — the pitch in one paragraph

> **Pratibimb is a Bayesian metabolic twin for Indian Type 2 Diabetes that calibrates itself in milliseconds, runs on 3 fingersticks a day instead of a ₹4,000 CGM sensor, and tells the patient *when* to prick their finger so it learns the most.** A physiological glucose–insulin–drug model sits at its core, with its parameters (insulin sensitivity, β-cell function, gastric emptying and sulfonylurea effect) inferred by a neural network trained on a million simulated Indian patients. Doctors see a calibrated 2-hour forecast, a nocturnal-hypoglycaemia watch, and a Thali/Fasting/Heatwave simulator. The simulator's effect sizes are checked against published meta-analyses.

Why this wins: all six public competitor repos are **synthetic-data ML classifiers** (§1). Ours is a *twin* in the NASEM sense (data assimilation, VVUQ and decision support), validated on **real open data**, and aimed at India's actual constraint, which is that most patients can't afford a CGM.

---

## 1. Competitive landscape (re-checked today) ✅

Your earlier brief found 2 repos; there are now at least 6:

| Repo | What it is | Data | Mechanistic? | Uncertainty? | Real-data validation? |
|---|---|---|---|---|---|
| [ChipUP](https://github.com/reshureshumii/ChipUP_DigitalTwinChallenge) | T2D 2-h spike classifier, XGBoost, ROC-AUC ≈ 0.77 | Synthetic (200 pts) | No | No | No |
| [GlycoTwin (Sagnik1091)](https://github.com/Sagnik1091/glycotwin) | 7-day risk of mean glucose >180; 1–30-day counterfactual simulator; FastAPI | Synthetic | "Physiology-inspired constraints" | "Probabilistic estimates" (not calibrated) | No |
| [GlycoTwin-AI (shivamrajsr07)](https://github.com/shivamrajsr07/GlycoTwin-AI) | 2-h spike, XGBoost/LightGBM, React 19 + Recharts, what-if by re-running the classifier | Synthetic | No | No | No |
| [MediTwin (harsh23533-del)](https://github.com/harsh23533-del/digital-twin-) | Cardiac + metabolic + derm, XGBoost AUC 0.94 on Cleveland, Streamlit | Cleveland CSV + synthetic | No | No | Cleveland only (a static dataset with no time-series) |
| [PredictiTwin](https://github.com/dabhiriddhi001001-cmyk/PredictiTwin) | Deck + README only so far | ? | ? | ? | ? |
| Others ([hridaycode1119](https://github.com/hridaycode1119/digital-twin), [nikhitha-1604](https://github.com/nikhitha-1604/health_digital_twin), [siddhartha-kodipunjula](https://github.com/siddhartha-kodipunjula/human-digital-twin)) | Generic "human digital twin" repos | — | — | — | — |

**What this tells us:**
- The "T2D + XGBoost + synthetic + React/Streamlit dashboard" pattern is already common. Three teams are in the T2D-spike lane.
- **Nobody** has: real-data validation, calibrated uncertainty, parameter inference (a "twin" that learns physiology), drug pharmacology, hypoglycaemia, or a CGM-free mode.
- GlycoTwin-AI's what-if re-runs an ML classifier with edited inputs. Juries will spot that this is **not causal**, because a classifier is not a simulator. Our counterfactuals come from a mechanistic model, and we **validate their effect sizes against RCT meta-analyses** (§6.4). Say this out loud in the pitch.

---

## 2. Fact-check of your earlier briefs

| Claim in brief | Verdict | Correct value / source |
|---|---|---|
| INSPIRED: "SIDD 2%, IROD 25.9%, CIRDD 12.1%, MARD 35.8%" | ❌ **SIDD is wrong** | **SIDD 26.2% (n=5,009), IROD 25.9% (4,934), CIRDD 12.1% (2,313), MARD 35.8% (6,828)** of n=19,084 — [Anjana et al., BMJ Open DRC 2020, PMC7437708](https://pmc.ncbi.nlm.nih.gov/articles/PMC7437708/) |
| ICMR-INDIAB-17: diabetes 11.4%, prediabetes 15.3% | ✅ | [Lancet D&E 2023](https://pubmed.ncbi.nlm.nih.gov/37301218/) |
| Libre sensor ≈ ₹4,400–4,600 / 15 days | ⚠️ roughly | ₹4,200–5,250 per 14-day sensor in 2026 retail ([Moglix ₹4,349](https://www.moglix.com/abbott-freestyle-libre-glucometer-sensor/mp/msn858047xj492), [Colmed ₹4,448](https://www.colmed.in/freestyle-libre-sensor.html), [Flipkart ₹5,249](https://www.flipkart.com/freestyle-libre-sensor-flash-glucose-monitoring-system-glucometer/p/itm4b0460bf79bf1?pid=GLMFYHH5YMZFKQF6)). **≈ ₹8,500–10,500 / month** for continuous use |
| DPDP Rules notified 14 Nov 2025, 18-month phase-in | ✅ + detail | Three dates: **14 Nov 2025** (Board), **14 Nov 2026** (Consent Manager obligations), **14 May 2027** (Data Fiduciary obligations) — [PIB PDF](https://static.pib.gov.in/WriteReadData/specificdocs/documents/2025/nov/doc20251117695301.pdf). Consent-manager obligations start **6 days before the summit**, which makes a good pitch line |
| CDSCO regulates SaMD | ✅ + update | CDSCO issued a **Draft Guidance on Medical Device Software on 21 Oct 2025** ([Cyril Amarchand analysis](https://corporate.cyrilamarchandblogs.com/2026/01/medical-device-as-software-has-cdsco-guidance-changed-the-rules/)). Cite it in the Trust tab |
| AI-READI access ~2 weeks | ✅ | Needs a Data Access Committee review of your research proposal; the non-restricted tier is a click-through DLA; link "typically within 2 weeks" ([AI-READI FAQ](https://aireadi.org/faq), [licence docs](https://docs.aireadi.org/docs/3/preliminary/license)). **Apply today.** If it arrives after ~10 Oct, treat it as bonus validation |
| CGMacros licence CC BY-NC-SA 4.0 | ✅ | 45 people (15 healthy / 16 prediabetes / **14 T2D**), Libre Pro + Dexcom G6, Fitbit Sense, meal macros + photos, HbA1c, insulin, lipids, microbiome; 628 MB; v1.0.0 (Jan 2025) — [PhysioNet](https://physionet.org/content/cgmacros/1.0.0/) |
| ShanghaiT2DM CC BY 4.0 | ✅ | 100 T2D patients, 3–14 days CGM, clinical + labs + medications + daily diet — [Sci Data 2023](https://www.nature.com/articles/s41597-023-01940-7), [Figshare](https://figshare.com/articles/dataset/Diabetes_Datasets-ShanghaiT1DM_and_ShanghaiT2DM/20444397). ⚠️ No wearable HR/steps stream (check on download) |
| Carbs-last food order effect | ✅ upgraded | Use the **meta-analysis** instead of a single trial: 17 RCTs, n=389. Postprandial glucose **−42.7 mg/dL at 60 min**, **−13.0 at 120 min**, gastric-emptying half-time **+28 min**, HbA1c −0.16% — [Acta Diabetol 2025, doi:10.1007/s00592-025-02586-0](https://doi.org/10.1007/s00592-025-02586-0) |
| South Indian breakfast GI | ✅ | Rice idli **85**, rice dosa **76**, upma **71** (Chinese + Indian healthy men) — [Br J Nutr 2018, doi:10.1017/S0007114518000703](https://doi.org/10.1017/S0007114518000703) |

**Other upgrades from the INSPIRED paper.** It reports per-cluster physiology, which gives the twin's day-1 priors directly:

| Cluster | Share | Age dx | BMI | HbA1c | HOMA-IR | HOMA-B | Stim. C-peptide |
|---|---|---|---|---|---|---|---|
| SIDD | 26.2% | 42.5 | 24.9 | 10.7 | 2.8 | **38.8** | 1.7 |
| IROD | 25.9% | 46.5 | 32.6 | 8.3 | **4.1** | 100.8 | 3.3 |
| CIRDD | 12.1% | 42.1 | 26.5 | 9.1 | 3.8 | 64.5 | 2.6 |
| MARD | 35.8% | 50.2 | 25.9 | 7.2 | 2.6 | 94.1 | 3.0 |

HOMA-IR maps to the prior on insulin sensitivity S_I, and HOMA-B maps to the prior on β-cell responsiveness φ. The MDRF team's **DIANA** tool (PLOS Digit Health, Aug 2025) assigns an endotype from age at dx, BMI, waist, HbA1c, TG, HDL (C-peptide optional); its R code is at [gitlab.com/mdc_mdrf/T2D_cluster](https://gitlab.com/mdc_mdrf/T2D_cluster) ([PMC12324136](https://pmc.ncbi.nlm.nih.gov/articles/PMC12324136/)). Re-implementing its assignment step gives an **India-native, published** subtype classifier instead of an invented one.

---

## 3. Six boundary-pushing ideas, each backed by 2024–2026 research

Ranked by *impact ÷ effort*. Ideas 1–3 are the core; 4–6 add depth.

### Idea 1 — Amortised Bayesian twin via Simulation-Based Inference (SBI) ⭐ core
**What:** Instead of fitting each patient's ODE parameters with a slow UKF or MCMC, train a **neural posterior estimator** (normalising flow + time-series embedding net) on ~10⁶ simulated Indian virtual patients. For a new patient, one forward pass (milliseconds) returns the full posterior p(S_I, S_G, φ, k_gastric, EGP₀, SU-gain … | data).

**Why it's frontier:** [Hoang et al., "A Real-Time Digital Twin for Type 1 Diabetes using Simulation-Based Inference", arXiv:2507.01740 (Jul 2025)](https://arxiv.org/abs/2507.01740) showed NPE beats MCMC on speed, generalisation and UQ for T1D. **Nobody has done this for T2D, or for Indian phenotypes.** That is a legitimate research contribution.

**Why it fits the challenge:** it *is* data assimilation (NASEM element), it gives calibrated uncertainty by construction (VVUQ), and it makes "re-personalise every day" trivially cheap.

**Masking trick (the unifying design):** during training, randomly mask the glucose channel to mimic *CGM*, *8 fingersticks/day*, *4/day*, *2/day*. **The same network then serves CGM users and fingerstick users.** With sparse data its posterior is simply wider, and the dashboard shows that honestly.

**Tooling:** [`sbi`](https://github.com/sbi-dev/sbi) (Apache-2.0, PyTorch) + a vectorised NumPy/JAX simulator. Simulating 10⁶ × 72 h at 5-min steps is a few CPU-hours if vectorised.

**Supporting evidence the parameters are identifiable from free-living CGM:** [Healey & Kohane, EMBC 2024, doi:10.1109/EMBC53108.2024.10781685](https://doi.org/10.1109/EMBC53108.2024.10781685). S_I and β-cell estimates from daily CGM peaks correlate with OGTT ground truth. Also [Metwally et al., Nature BME 2025, PMC12183321](https://pmc.ncbi.nlm.nih.gov/articles/PMC12183321/): muscle IR, β-cell dysfunction and incretin deficiency predicted from the CGM curve shape.

### Idea 2 — Active-sensing twin: "it tells you *when* to prick your finger" ⭐ core, the headline innovation
**What:** Given a budget of *k* strips/day, the twin schedules each fingerstick at the time that maximally reduces uncertainty about the **next-24-h hypoglycaemia risk**. This is Bayesian experimental design / value of information. In practice: draw posterior samples, simulate the next day under the planned meals, and pick the time *t* where observing G(t) would most reduce the variance of P(hypo) (a sample-based expected-information-gain estimate).

**Why it matters in India:** strips cost ≈ ₹10–25 each vs ≈ ₹300/day for a CGM. The practical question is not *CGM or nothing*; it's *which 3 readings to take*. Prior art supports the premise: data assimilation works on sparse self-monitoring data ([Albers et al., PLOS Comp Bio 2017, PMC5409456](https://pmc.ncbi.nlm.nih.gov/articles/PMC5409456/); [offline+online DA for T2D, arXiv:1709.00163](https://arxiv.org/abs/1709.00163)); Bayesian estimation rescues reduced sampling ([Bayesian reduced-sampling minimal model, PMID 16144816](https://pubmed.ncbi.nlm.nih.gov/16144816/)); Bayesian design of OGTT sampling times ([arXiv:1903.11697](https://arxiv.org/abs/1903.11697)); SMBG→AGP metrics for LMICs ([DPA-Net, arXiv:2510.06623](https://arxiv.org/abs/2510.06623)). **Nobody has closed the loop into a twin that schedules its own measurements.**

**Headline experiment (one chart that wins the room):** hypo-detection sensitivity and 60-min RMSE vs *strips per day* (2, 3, 4, 6, CGM) under three policies: *fixed clinic schedule* (fasting + 2-h post-meal), *random*, and *twin-scheduled*. Run it on real CGM traces by subsampling, where the held-out CGM is ground truth. If twin-scheduled at 3/day approaches CGM at detecting nocturnal lows, that's the finding that sells the project.

### Idea 3 — Day-1 twin from an ordinary lab report + smartwatch (no CGM ever)
**What:** Build the prior from routine labs (FBG, HbA1c, TG, HDL) + smartwatch (resting HR, steps, HRV) + INSPIRED endotype.

**Evidence:** Google's **WEAR-ME** study ([Nature, Mar 2026, PMC13061641](https://pmc.ncbi.nlm.nih.gov/articles/PMC13061641/); [arXiv:2505.03784](https://arxiv.org/abs/2505.03784)), n=1,165, predicted insulin resistance (HOMA-IR) with AUROC **0.70** from wearables + demographics, **0.78** adding fasting glucose, and **0.80** with the full routine panel. The top wearable features were resting HR (r=0.27), steps (r=−0.25) and HRV-RMSSD (r=−0.14). Fold those published correlations into the S_I prior as a hierarchical regression, so the twin starts personalised, not generic.

### Idea 4 — Physics-conditioned foundation-model residual
**What:** The mechanistic forecast is the backbone. A **time-series foundation model** (Chronos-2 or TimesFM), fed the physics forecast as a covariate, predicts only the residual. The physics model captures what's known; the foundation model captures what isn't (stress, illness, sensor quirks).

**Evidence:** [GlucoFM-Bench, arXiv:2606.06881 (Jun 2026)](https://arxiv.org/abs/2606.06881), 15 datasets, 1,117 people: the best zero-shot TSFM is **within 5% of full supervised** models; a small LSTM is still best when there's lots of in-domain data. [arXiv:2609.11872 (Sep 2026)](https://arxiv.org/abs/2609.11872): fine-tuned Chronos-Bolt cut RMSE **8.6–18.2%** in non-diabetic/T2D cohorts; adding **dietary context (CGMacros) cut postprandial RMSE ≈ 15%**. Our hybrid adds physiology on top of that, which is exactly the gap those papers leave open.

**Known targets to beat on ShanghaiT2DM** ([Sci Rep, Dec 2025, PMC12819561](https://pmc.ncbi.nlm.nih.gov/articles/PMC12819561/)):

| Model | RMSE 30 min | RMSE 60 min | RMSE 90 min |
|---|---|---|---|
| XGBoost (competitors' approach) | 14.70 | 26.31 | 32.91 |
| LSTM | 13.38 | 23.94 | 31.57 |
| GPT-4.1 (best reported) | **12.57** | **22.45** | 32.19 |

Aim: beat XGBoost everywhere, match LSTM, and **beat all of them at 90–120 min and in the hypo range**, where physics helps most. Also report the numbers honestly if you don't.

**Other context:** [CGM-LSM (JHU + Welldoc, npj Health Syst 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13354170/) reports 1-h RMSE of 15.9 on OhioT1DM, but its weights are **not public**. [GluFormer (Nature)](https://pubmed.ncbi.nlm.nih.gov/41535468/) is a generative CGM foundation model. Mention both as related work; don't depend on them.

### Idea 5 — Indian pharmacology + India-specific stress scenarios
Model the drugs mechanistically, because that's what makes *dose-timing* counterfactuals and *hypo* prediction physiological rather than statistical:

| Drug class (common in India) | Mechanistic hook |
|---|---|
| Sulfonylurea (glimepiride, gliclazide) | **Glucose-independent** insulin secretion term → explains *nocturnal & fasting hypos* |
| Metformin | ↓ endogenous glucose production (EGP₀) |
| DPP-4i | ↑ incretin-mediated, glucose-dependent secretion (low hypo risk) |
| SGLT2i | Urinary glucose excretion above renal threshold |
| Basal insulin | Standard subcutaneous PK compartment |

Evidence for why hypoglycaemia matters: in SU-treated, *well-controlled* T2D, 30% had CGM hypos, **67% nocturnal and 73% asymptomatic**, with QTc prolongation in some patients ([Middleton et al., Diabetes Care 2017, doi:10.2337/dc16-1972](https://doi.org/10.2337/dc16-1972)).

**Scenario library** (each one is a toggle in the simulator):
1. **Thali builder:** INDB recipes ([Indian Nutrient Databank, 1,014 recipes, open, GitHub](https://github.com/lindsayjaacks/Indian-Nutrient-Databank-INDB-)) + published GI values.
2. **Food order (carbs-last):** implemented as slower gastric emptying. **Calibrate** so the simulated effect reproduces the meta-analysis (−42.7 mg/dL at 60 min; t½ +28 min).
3. **Post-meal walk:** insulin-independent uptake driven by steps/HR.
4. **Religious fasting (Navratri, Ekadashi, Ramadan, Karva Chauth):** a very common Indian clinical question. SU while fasting carries high hypo risk ([Bashir et al. 2018: 7.9 hypos/patient in 2 weeks, 50% asymptomatic, doi:10.1016/j.dsx.2018.06.005](https://doi.org/10.1016/j.dsx.2018.06.005)). The twin simulates the fasting day and flags when to review the SU dose before the fast. **No competitor has this, and every Indian diabetologist handles it.**
5. **Heatwave day:** higher ambient temperature is associated with serious hypoglycaemia in insulin users ([PMC10834387](https://pmc.ncbi.nlm.nih.gov/articles/PMC10834387/)). Use this as a risk-modifier flag driven by an IMD forecast, not as a mechanistic claim.
6. **Late dinner / night shift:** circadian S_I modulation.

### Idea 6 — Twin-grounded clinician copilot (LLM with tools, not free text)
**What:** The LLM (Claude or Gemini) **can only answer by calling twin tools**: `simulate(scenario)`, `explain_forecast()`, `get_agp()` and `posterior_summary()`. Every number it states must come from a tool result, which is checked automatically. It refuses dosing advice. A nurse or doctor **approves** each patient nudge before it's sent.

**Evidence:** WEAR-ME's ReAct "IR agent" (Gemini 2.0 Flash) was preferred by endocrinologists 80–92% of the time over a baseline LLM. The human-in-the-loop predictive twin RCT ([npj Health Syst, Jul 2026, PMC13354173](https://pmc.ncbi.nlm.nih.gov/articles/PMC13354173/)) had nurses review AI SMS feedback. This directly answers the Happiest Health jury's recurring worry that AI will damage the doctor–patient relationship: the doctor stays in the loop.

---

## 4. The twin model: concrete spec

**States:** plasma glucose G, remote insulin action X, plasma insulin I, gut compartments Q₁ and Q₂ (two-compartment absorption), SU and basal-insulin PK compartments.

```
dG/dt  = −(S_G + S_I·X)·G + EGP₀·(1 − m_met) + Ra_meal(t)/V − U_ex(steps,HR)·G − UGE_sglt2(G)
dX/dt  = −p₂·X + p₂·(I − I_b)
dI/dt  = −n·I + φ·[G − h]₊ ·(1+ε_dpp4)  +  σ_SU·SU(t)          ← SU term is glucose-independent → hypos
dQ₁/dt = −k_ge(meal)·Q₁ + D·δ(t−t_meal);   dQ₂/dt = k_ge·Q₁ − k_abs·Q₂;   Ra = f·k_abs·Q₂
S_I(t) = S_I⁰ · c(t_circadian) · exp(η_t),   η_t = daily random walk (sleep/stress/illness drift)
```

**Parameters inferred by SBI:** θ = {S_I⁰, S_G, φ, h, EGP₀, k_ge, σ_SU, m_met, η-volatility}.

**Priors:** hierarchical, with INSPIRED endotype (HOMA-IR → S_I, HOMA-B → φ), adjusted by the WEAR-ME covariates (RHR, steps, HRV) and labs.

**Observation model:** CGM = G(t−lag) + AR(1) sensor noise; fingerstick = G(t) + ISO-15197 noise (±15%).

**Twin-drift monitor:** if the posterior predictive coverage over the last 48 h falls below target, raise "recalibrate: wear one CGM sensor for 14 days".

**Start simple:** Bergman minimal model + gut + SU term is enough for a PoC. Add the full Dalla Man T2D model only if Week 2 goes well.

**Reference implementations (read, don't vendor):**
- [T2DSim_AI (OHSU)](https://github.com/mosqueralopez/T2DSim_AI): a physiology-guided NN digital twin for T2D with 40 pre-trained twins using HR, sleep and medications (SU, SGLT2i, GLP-1, metformin). Its **OHSU research licence is non-commercial and restricts redistribution**, so use it only as a comparison baseline and never copy it into your Apache-2.0 repo.
- [simglucose](https://github.com/jxx123/simglucose): the T1D UVA/Padova implementation, a good code-structure reference.
- [Physiologically-constrained NN twin (arXiv:2508.05705)](https://arxiv.org/abs/2508.05705): the stretch goal of turning the ODE into a neural state-space model.

---

## 5. Data plan (final)

| Role | Dataset | Streams | Licence handling |
|---|---|---|---|
| **Main real validation (meds + rice diet)** | ShanghaiT2DM (100 T2D) | EHR labs + meds + diet + CGM (15-min) | CC BY 4.0: OK to redistribute processed subsets with attribution |
| **Real two-stream fusion** | CGMacros (14 T2D + 16 prediabetes) | EHR labs + **Fitbit** + CGM + meal macros/photos | CC BY-NC-SA: **ship a download script, never commit raw files** |
| Wearable-physiology supplement | BIG IDEAs Lab (PhysioNet) | Empatica E4 HR/EDA/temp + CGM + food logs | Open (ODC-By) |
| Bonus if approved | AI-READI v3 (~2,280) | Dexcom G6 + Garmin HR/stress/sleep/steps + clinical | Controlled: never commit; report aggregate metrics only |
| **Synthetic Indian cohort ("Synthea-IN")** | Our simulator + ICMR-INDIAB prevalence + INSPIRED endotype mix + Indian drug regimens + INDB meals | ABDM FHIR R4 bundles + Apple-Health-style export | Ours, Apache-2.0 |

**Show the two-stream fusion literally.** The ingestion layer should parse (a) **ABDM/NRCeS FHIR R4 bundles** ([NRCeS IG](https://nrces.in/ndhm/fhir/r4/)) for the EHR, and (b) an **Apple Health `export.xml` / Google Health Connect-style JSON** for the wearables. Judges will recognise both formats, and it satisfies the challenge's "mimicking Apple Health, Google Fit" wording word for word.

---

## 6. Validation plan (VVUQ, following NASEM)

### 6.1 Forecast accuracy
- **Protocol:** leave-patients-out on ShanghaiT2DM and CGMacros.
- **Metrics:** RMSE/MAE at 30/60/90/120 min.
- **Clinical accuracy:** Clarke **and** Parkes (type 2) error grids.
- **Baselines:** persistence, ARIMA, XGBoost, LSTM, zero-shot Chronos-2, and the published ShanghaiT2DM numbers in §3.

### 6.2 Event detection (the challenge's actual ask)
- **Events:** spike >180 within 2 h after a meal; hypo <70 (and <54); nocturnal hypo (00:00–06:00).
- **Report for each:** sensitivity, precision, **median lead time**, and **false alerts per patient-day**.

### 6.3 Uncertainty
- Empirical coverage of the 80% and 95% intervals, plus a reliability diagram.
- Use **adaptive conformal inference** on top of the SBI posterior predictive, so coverage holds under drift. See the review [arXiv:2511.13608](https://arxiv.org/abs/2511.13608); the multi-step variant [AcMCP, arXiv:2410.13115](https://arxiv.org/abs/2410.13115) handles cross-horizon correlation.

### 6.4 Counterfactual validity (almost nobody does this)
Run the simulator's interventions on the virtual cohort and compare against published effect sizes:

| Intervention | Published effect | Pass criterion |
|---|---|---|
| Carbs-last | −42.7 mg/dL @60 min (95% CI −55.5 to −30.0) | Simulated mean within the CI |
| Idli vs dosa vs upma breakfast | GI 85 / 76 / 71 | Ranking preserved |
| Post-meal walk | Reduced postprandial excursion (direction and rough magnitude) | Direction correct; magnitude plausible |
| SU dose on a fasting day | Elevated hypo risk (Bashir 2018) | Direction correct |

### 6.5 Parameter recovery (verification)
On synthetic patients with known θ, show that posterior 90% intervals contain the true value about 90% of the time. This is simulation-based calibration (SBC), which `sbi` provides out of the box.

### 6.6 CGM-optional curve (the headline)
§3 Idea 2: accuracy and hypo sensitivity vs strips/day × sampling policy.

### 6.7 Ablations and fairness
- **Ablations:** remove the prior, wearables, drug model, or residual model one at a time.
- **Fairness:** break results down by endotype, sex, age band, and BMI (including the "thin-fat" band: BMI < 25 with high waist).

---

## 7. Architecture

```
┌──────────────── PHYSICAL TWIN ────────────────┐
│ ABDM FHIR R4 (labs, meds, dx) │ Apple Health / │
│                                │ Health Connect │
│ CGM  or  k fingersticks/day    │ HR, HRV, steps,│
│ Meals (INDB)                   │ sleep          │
└──────┬──────────────────────────────┬──────────┘
       ▼                              ▼
 [1] Ingest + harmonise (FHIR parser, 5-min grid, QC, DPDP consent check)
       ▼
 [2] Prior builder: DIANA-style endotype + WEAR-ME covariates → p(θ)
       ▼
 [3] SBI posterior network  q(θ | data)   ← trained on 10⁶ Synthea-IN twins (masked obs)
       ▼                                   (daily re-inference = data assimilation)
 [4] Mechanistic simulator (glucose–insulin–gut–drugs) → ensemble forecasts
       ▼
 [5] Residual learner (Chronos-2 w/ physics covariate  |  LightGBM fallback)
       ▼
 [6] Adaptive conformal calibration → 80/95% bands with coverage monitoring
       ▼
 [7] Event engine: P(spike), P(hypo), nocturnal watch, twin-drift flag
       ▼
 [8] Active-sensing planner: "next fingerstick at 02:30 and 07:10"
       ▼
 [9] Scenario simulator: Thali · food order · walk · fasting · heatwave · dose timing
       ▼
[10] Clinician dashboard + tool-grounded LLM copilot + audit/consent ledger
       ▼
   Approved nudge → patient (EN/HI/TE) → new data → [3]   (closed two-way loop)
```

**Stack:**
- **Modelling:** Python 3.11, NumPy + JAX (`diffrax` optional), `sbi` (PyTorch), LightGBM, `chronos-forecasting`, MAPIE/custom ACI, SHAP.
- **Interop and API:** `fhir.resources`, FastAPI.
- **Dashboard:** React + Vite + Recharts (your stated preference), or Streamlit if short on time.
- **Deployment:** Docker Compose; demo on Hugging Face Spaces or Render.
- **Licence:** Apache-2.0.

---

## 8. Dashboard (designed for a 5-minute OPD consult)

1. **Panel triage:** patients ranked by next-24-h risk, with chips such as 🌙 *Nocturnal hypo* and 🍚 *Post-lunch spike*.
2. **AGP report:** the standard page diabetologists already read.
3. **Twin vitals:** S_I, β-cell φ and gastric k_ge, each **with posterior bands**, plus the trend vs the patient's own baseline and the endotype tag (e.g., *IROD*). Example: "S_I ↓18% after 2 nights of <5 h sleep (90% CI 9–27%)".
4. **Forecast strip:** 4 h ahead with an 80% band, alert markers and lead time.
5. **"Why" panel:** mechanistic attribution (meal load / reduced S_I / SU effect / low activity), plus SHAP for the residual.
6. **Scenario simulator:** a drag-and-drop thali, food-order toggle, walk slider, fasting-day and heatwave toggles; before/after curves with bands.
7. **Sensing planner:** "Strips left today: 2. Best times: 02:30, 07:10." Also shows what the uncertainty will look like after each reading.
8. **Trust tab:** model card, calibration plot, validation numbers, the counterfactual-validity table (§6.4), the DPDP consent ledger, the CDSCO draft-guidance note, and a "research prototype, not a medical device" banner.
9. **Copilot:** answers questions like "Why is Ramesh at risk tonight?", with citations to the tool calls behind each figure.

---

## 9. 27-day plan with a hard MVP line

| Week | Dates | Must-have (MVP) | Stretch |
|---|---|---|---|
| 0 | **23 Sep (today)** | Register the team; **apply to AI-READI**; create repo `TeamName_CollegeName`; Apache-2.0 licence; download ShanghaiT2DM + CGMacros + BIG IDEAs | — |
| 1 | 24–30 Sep | Simulator (minimal model + gut + SU + metformin) with unit tests; Synthea-IN generator + FHIR export; data loaders; baselines (persistence, ARIMA, XGBoost, LSTM) on ShanghaiT2DM | Chronos-2 zero-shot baseline |
| 2 | 1–7 Oct | SBI posterior (CGM + masked sparse obs); SBC parameter-recovery check; event engine; conformal bands; evaluation harness with error grids | Physics-covariate Chronos residual |
| **MVP gate: 8 Oct** | | A forecast with calibrated bands + hypo/spike alerts, validated on real data. **If this isn't done, cut ideas 5–6 and finish this.** | |
| 3 | 8–14 Oct | Active-sensing planner + **strips/day experiment**; FastAPI; dashboard (triage, AGP, twin vitals, forecast, simulator) | Fasting / heatwave scenarios; LLM copilot |
| 4 | 15–19 Oct | Counterfactual-validity table; `VALIDATION.md`, `MODEL_CARD.md`, `DATA.md`; architecture PDF; deck; 3–5-min video; hosted demo; **submit 19 Oct** | Hindi/Telugu nudges |
| — | 20 Oct | Buffer only | — |

**Roles for a team of 4:**
- **(A) Physiology + SBI:** simulator, priors, posterior.
- **(B) Evaluation + ML:** baselines, residual model, conformal, error grids, strips/day experiment.
- **(C) Full-stack:** FHIR/Apple Health ingestion, API, dashboard, deploy.
- **(D) Clinical + story:** ideally a medical or pharmacy student. They validate the scenarios, write the model card, and do the deck and video.

---

## 10. Pitch arc (for the jury and the summit)

1. **The patient.** Ramesh, 52, Ongole, on metformin + glimepiride, can't afford ₹9,000 a month for a CGM. On Navratri he fasts. At 3 AM his glucose will fall to 58 and nobody will know.
2. **The definition.** Quote NASEM's definition of a digital twin: dynamically updated from the patient, predictive, informs decisions. Show a checklist of how Pratibimb meets each element, and note that none of the other repos do.
3. **Live demo.**
   - One routine lab report + a smartwatch produce a day-1 twin (IROD endotype).
   - Three fingersticks, **timed by the twin**, give a 90-min warning of the 3 AM low.
   - The doctor tests "halve the glimepiride on fasting days" and a dal-first dinner in the simulator.
   - The copilot drafts a Telugu nudge and the doctor approves it.
4. **The numbers.**
   - RMSE vs published ShanghaiT2DM baselines.
   - Hypo lead time.
   - Calibration plot.
   - Simulated food-order effect vs the meta-analysis confidence interval.
   - **The strips/day curve.**
5. **The closer.** *"Three fingersticks, chosen by the twin, instead of a sensor every two weeks."* Then mention DPDP consent-manager readiness (in force 14 Nov 2026), ABDM FHIR interoperability, and the CDSCO SaMD pathway.

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| SBI posterior mis-calibrated on real data (sim-to-real gap) | SBC on synthetic; posterior-predictive checks on real traces; fall back to UKF (`filterpy`) if needed. Report both |
| Strips/day result is weak | Still publishable as an honest negative; the planner still beats *fixed* schedules on hypo detection, which is the likely outcome |
| AI-READI not approved in time | Not on the critical path; ShanghaiT2DM + CGMacros + BIG IDEAs suffice |
| Scope creep | MVP gate on 8 Oct; ideas 5–6 are explicitly optional |
| Licence contamination | Never commit CGMacros/AI-READI raw data; no T2DSim_AI code in the repo; `DATA.md` provenance table |
| Clinical-safety optics | No dosing advice anywhere; human approval of nudges; "not a medical device" banner; model card lists failure modes |

---

## 12. Source list (all opened or checked this session)

**Competitors:** [ChipUP](https://github.com/reshureshumii/ChipUP_DigitalTwinChallenge) · [GlycoTwin](https://github.com/Sagnik1091/glycotwin) · [GlycoTwin-AI](https://github.com/shivamrajsr07/GlycoTwin-AI) · [MediTwin](https://github.com/harsh23533-del/digital-twin-) · [PredictiTwin](https://github.com/dabhiriddhi001001-cmyk/PredictiTwin)

**India epidemiology & subtypes:** [ICMR-INDIAB-17](https://pubmed.ncbi.nlm.nih.gov/37301218/) · [INSPIRED (PMC7437708)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7437708/) · [DIANA (PMC12324136)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12324136/) · [INDB](https://github.com/lindsayjaacks/Indian-Nutrient-Databank-INDB-)

**Datasets:** [ShanghaiT2DM](https://www.nature.com/articles/s41597-023-01940-7) · [CGMacros](https://physionet.org/content/cgmacros/1.0.0/) · [AI-READI FAQ](https://aireadi.org/faq) · [Glucose-ML collection](https://arxiv.org/html/2507.14077v1) · [Awesome-CGM](https://github.com/IrinaStatsLab/Awesome-CGM)

**Methods:**
- [SBI real-time T1D twin (arXiv:2507.01740)](https://arxiv.org/abs/2507.01740)
- [Healey & Kohane EMBC 2024](https://doi.org/10.1109/EMBC53108.2024.10781685)
- [CGM subphenotypes (PMC12183321)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12183321/)
- [WEAR-ME (PMC13061641)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13061641/)
- [GlucoFM-Bench (arXiv:2606.06881)](https://arxiv.org/abs/2606.06881)
- [TSFM + dietary context (arXiv:2609.11872)](https://arxiv.org/abs/2609.11872)
- [CGM-LSM (PMC13354170)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13354170/)
- [GluFormer (PMID 41535468)](https://pubmed.ncbi.nlm.nih.gov/41535468/)
- [ShanghaiT2DM LLM/ML benchmark (PMC12819561)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12819561/)
- [Albers 2017 (PMC5409456)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5409456/)
- [DPA-Net (arXiv:2510.06623)](https://arxiv.org/abs/2510.06623)
- [Conformal TS review (arXiv:2511.13608)](https://arxiv.org/abs/2511.13608)
- [AcMCP (arXiv:2410.13115)](https://arxiv.org/abs/2410.13115)
- [T2DSim_AI](https://github.com/mosqueralopez/T2DSim_AI)
- [Physiologically-constrained NN twin (arXiv:2508.05705)](https://arxiv.org/abs/2508.05705)
- [SSM-CGM on AI-READI (arXiv:2510.04386)](https://arxiv.org/abs/2510.04386)
- [Human-in-the-loop twin RCT (PMC13354173)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13354173/)

**Clinical evidence** (DOIs via PubMed): [Carbs-last meta-analysis](https://doi.org/10.1007/s00592-025-02586-0) · [Indian breakfast GI](https://doi.org/10.1017/S0007114518000703) · [SU nocturnal hypos + QTc](https://doi.org/10.2337/dc16-1972) · [Hypos during Ramadan fasting](https://doi.org/10.1016/j.dsx.2018.06.005) · [Heat & hypoglycaemia (PMC10834387)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10834387/) · [Premeal pistachio CGM RCT, MDRF](https://doi.org/10.1016/j.tjnut.2024.12.005)

**Regulation:** [DPDP Rules 2025 (PIB)](https://static.pib.gov.in/WriteReadData/specificdocs/documents/2025/nov/doc20251117695301.pdf) · [CDSCO draft SaMD guidance analysis](https://corporate.cyrilamarchandblogs.com/2026/01/medical-device-as-software-has-cdsco-guidance-changed-the-rules/) · [ABDM FHIR IG](https://nrces.in/ndhm/fhir/r4/)

**Pricing:** [Moglix](https://www.moglix.com/abbott-freestyle-libre-glucometer-sensor/mp/msn858047xj492) · [Colmed](https://www.colmed.in/freestyle-libre-sensor.html) · [Flipkart](https://www.flipkart.com/freestyle-libre-sensor-flash-glucose-monitoring-system-glucometer/p/itm4b0460bf79bf1?pid=GLMFYHH5YMZFKQF6)

*Not re-verified this session (carried over from brief v1, check before quoting): Twin Health India RCT figures, the Albers "converges in ~1 week" claim, the Sci Rep 2025 post-meal-walk effect size, and fingerstick strip prices (₹10–25 is a typical retail range; confirm with a local pharmacy).*
