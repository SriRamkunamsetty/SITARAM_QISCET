"""Drug catalogue for regimens common in India.

Sulfonylureas are converted to glimepiride-equivalent "SU units" (1 unit = 1 mg glimepiride).
Equivalences are approximate maximal-dose ratios used for simulation only, never for dosing.
"""

from __future__ import annotations

# drug name -> mg per SU unit
_SU_MG_PER_UNIT: dict[str, float] = {
    "glimepiride": 1.0,
    "gliclazide_mr": 30.0,
    "gliclazide": 40.0,
    "glibenclamide": 2.5,
    "glipizide": 5.0,
}

SULFONYLUREAS = frozenset(_SU_MG_PER_UNIT)
OTHER_SUPPORTED = frozenset({"metformin", "dpp4i", "sglt2i"})


def normalise_drug(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def is_sulfonylurea(name: str) -> bool:
    return normalise_drug(name) in SULFONYLUREAS


def su_units(name: str, dose_mg: float) -> float:
    """Glimepiride-equivalent units for an SU dose; 0 for non-secretagogues."""
    key = normalise_drug(name)
    if key not in _SU_MG_PER_UNIT:
        return 0.0
    if dose_mg < 0:
        raise ValueError("dose_mg must be non-negative")
    return dose_mg / _SU_MG_PER_UNIT[key]
