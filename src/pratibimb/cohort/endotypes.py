"""Indian T2D endotypes (INSPIRED study) and the EHR -> physiological prior mapping.

Source: Anjana RM et al. BMJ Open Diabetes Res Care 2020;8:e001506 (PMC7437708), n = 19,084.
Cluster shares and means below are quoted from that paper. Per-cluster standard deviations and
the HOMA -> model-parameter scalings are modelling assumptions, documented as such, and are
meant to be refit once real paired data are available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pratibimb.cohort.profile import PatientProfile
from pratibimb.physiology.model import (
    DEFAULT_CONSTANTS,
    GB,
    KGE,
    KSU,
    N_PARAMS,
    PARAM_NAMES,
    PHI,
    SG,
    SI,
)


@dataclass(frozen=True)
class Endotype:
    code: str
    name: str
    share: float
    age_dx: float
    bmi: float
    waist_cm: float
    hba1c: float
    homa_ir: float
    homa_b: float


INSPIRED_ENDOTYPES: tuple[Endotype, ...] = (
    Endotype("SIDD", "Severe insulin-deficient diabetes", 0.262, 42.5, 24.9, 90.0, 10.7, 2.8, 38.8),
    Endotype("IROD", "Insulin-resistant obese diabetes", 0.259, 46.5, 32.6, 108.0, 8.3, 4.1, 100.8),
    Endotype("CIRDD", "Combined insulin-resistant & deficient", 0.121, 42.1, 26.5, 94.9, 9.1, 3.8, 64.5),
    Endotype("MARD", "Mild age-related diabetes", 0.358, 50.2, 25.9, 92.4, 7.2, 2.6, 94.1),
)
ENDOTYPE_CODES = tuple(e.code for e in INSPIRED_ENDOTYPES)

# Assumed within-cluster SDs for (age_dx, bmi, waist, hba1c) used for soft assignment.
_FEATURE_SD = np.array([9.0, 3.5, 9.0, 1.5])

# Parameter scalings (assumptions): S_I inversely proportional to HOMA-IR, beta-cell
# responsiveness and SU response proportional to HOMA-B.
_SI_PER_INV_HOMA_IR = 7.5e-4
_PHI_PER_HOMA_B = 0.08 / 100.0
_KSU_PER_HOMA_B = 2.5 / 90.0
_SG_T2D = 0.012
_KGE_T2D = 0.016
# Assumed fractional lowering of fasting glucose by a sulfonylurea (typical FPG reductions are
# 20-30%); used to back out the untreated basal level from an on-treatment FPG.
_SU_FPG_LOWERING = 0.20

# Log-scale prior SDs (assumptions) for si, sg, phi, gb, kge, ksu.
PRIOR_LOG_SD = np.array([0.35, 0.25, 0.40, 0.18, 0.30, 0.45])


def fasting_glucose_from_hba1c(hba1c_pct: float) -> float:
    """Approximate fasting glucose: 0.9 x ADAG estimated average glucose (28.7*A1c - 46.7)."""
    return 0.9 * (28.7 * hba1c_pct - 46.7)


def endotype_log_means(e: Endotype) -> np.ndarray:
    """Log-scale prior means for the non-glycaemic parameters of an endotype (gb set later)."""
    mu = np.empty(N_PARAMS)
    mu[SI] = math.log(_SI_PER_INV_HOMA_IR / e.homa_ir)
    mu[SG] = math.log(_SG_T2D)
    mu[PHI] = math.log(_PHI_PER_HOMA_B * e.homa_b)
    mu[GB] = math.log(fasting_glucose_from_hba1c(e.hba1c))
    mu[KGE] = math.log(_KGE_T2D)
    mu[KSU] = math.log(_KSU_PER_HOMA_B * e.homa_b)
    return mu


def endotype_probabilities(profile: PatientProfile) -> dict[str, float]:
    """Soft INSPIRED endotype membership (Gaussian naive Bayes on the published centroids)."""
    x = np.array([profile.age_at_diagnosis, profile.bmi, profile.waist_cm, profile.hba1c_pct])
    logp = []
    for e in INSPIRED_ENDOTYPES:
        mu = np.array([e.age_dx, e.bmi, e.waist_cm, e.hba1c])
        z = (x - mu) / _FEATURE_SD
        logp.append(math.log(e.share) - 0.5 * float(z @ z))
    arr = np.array(logp)
    arr = np.exp(arr - arr.max())
    arr /= arr.sum()
    return {e.code: float(p) for e, p in zip(INSPIRED_ENDOTYPES, arr, strict=True)}


@dataclass(frozen=True)
class ParameterPrior:
    """Independent log-normal prior over the six estimated parameters."""

    mean_log: np.ndarray
    sd_log: np.ndarray
    endotype_probs: dict[str, float]

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n parameter vectors on the log scale, shape (n, N_PARAMS)."""
        return self.mean_log + self.sd_log * rng.standard_normal((n, N_PARAMS))

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for i, name in enumerate(PARAM_NAMES):
            out[name] = {
                "median": math.exp(self.mean_log[i]),
                "p05": math.exp(self.mean_log[i] - 1.645 * self.sd_log[i]),
                "p95": math.exp(self.mean_log[i] + 1.645 * self.sd_log[i]),
            }
        return out

    @property
    def most_likely_endotype(self) -> str:
        return max(self.endotype_probs, key=self.endotype_probs.__getitem__)


def build_prior(profile: PatientProfile) -> ParameterPrior:
    """Map an EHR profile (+ optional wearable summary) to a physiological prior.

    1. Soft endotype membership -> moment-matched mixture of endotype priors.
    2. Basal glucose centred on the measured fasting glucose (or HbA1c-derived estimate),
       corrected for metformin so ``gb`` is the untreated basal level.
    3. Optional wearable covariates nudge S_I (direction from WEAR-ME, Nature 2026: higher
       resting HR and fewer steps associate with higher HOMA-IR). Magnitudes are provisional.
    """
    probs = endotype_probabilities(profile)
    means = np.stack([endotype_log_means(e) for e in INSPIRED_ENDOTYPES])
    w = np.array([probs[e.code] for e in INSPIRED_ENDOTYPES])
    mix_mean = w @ means
    mix_var = w @ (PRIOR_LOG_SD**2 + means**2) - mix_mean**2
    sd = np.sqrt(np.maximum(mix_var, 1e-12))

    fpg = profile.fasting_glucose or fasting_glucose_from_hba1c(profile.hba1c_pct)
    met = DEFAULT_CONSTANTS.met_effect if profile.on_metformin else 0.0
    su = _SU_FPG_LOWERING if profile.on_sulfonylurea else 0.0
    mix_mean[GB] = math.log(fpg / ((1.0 - met) * (1.0 - su)))
    sd[GB] = 0.12 if profile.fasting_glucose is not None and not su else PRIOR_LOG_SD[GB]

    shift = 0.0
    if profile.resting_hr is not None:
        shift -= 0.10 * (profile.resting_hr - 72.0) / 10.0
    if profile.daily_steps is not None:
        shift += 0.08 * (profile.daily_steps - 6000.0) / 3000.0
    mix_mean[SI] += float(np.clip(shift, -0.3, 0.3))

    return ParameterPrior(mean_log=mix_mean, sd_log=sd, endotype_probs=probs)
