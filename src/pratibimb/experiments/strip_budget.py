"""Strip-budget experiment: does a twin that chooses *when* to test beat standard schedules?

Setting: sulfonylurea-treated virtual Indian T2D patients with known ground truth. Each day has
four risk windows (morning 06:30-12:00, afternoon 12:00-18:00, evening 18:00-24:00, night
00:00-06:30). At the start of each window the twin predicts P(hypoglycaemia in window) from the
readings taken so far. Night risk is predicted at bedtime (22:30).

Policies, at k strips per day:
  none     prior + meal/medication log only (no glucose readings)
  fixed    structured FBS/PPBS schedule used in routine practice
  random   random waking-hour times
  twin     active sensing: information-maximising times, re-planned after every reading
  cgm      reference: a CGM reading every 15 minutes, day and night

The same truth, logs, meter noise and filter seed are shared across policies (common random
numbers), so differences come from *when* readings are taken.

Run:  uv run pratibimb-strip-experiment --patients 40 --days 7 --workers 8
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pratibimb.cohort.endotypes import build_prior
from pratibimb.cohort.routine import routine_events
from pratibimb.cohort.virtual import (
    SMBG_SENSOR,
    DayRecord,
    TruthTrajectory,
    VirtualPatient,
    cgm_trace,
    generate_day,
    sample_virtual_patient,
    sensor_reading,
    simulate_truth,
)
from pratibimb.physiology.inputs import EventLog, discretise
from pratibimb.physiology.model import MINUTES_PER_DAY, PatientConstants
from pratibimb.sensing.planner import PlannerConfig, plan_readings
from pratibimb.sensing.policies import PolicySpec, fixed_schedule, random_schedule, waking_slots
from pratibimb.twin.events import RiskWindow, sustained_crossing
from pratibimb.twin.particle_filter import FilterConfig, ParticleFilter

WINDOWS_MOD = {  # label -> (start, end) minutes relative to the day's midnight
    "morning": (390.0, 720.0),
    "afternoon": (720.0, 1080.0),
    "evening": (1080.0, 1440.0),
    "night": (1440.0, 1830.0),
}
DECISION_MOD = {"morning": 390.0, "afternoon": 720.0, "evening": 1080.0, "night": 1350.0}
WINDOW_WEIGHT = {"morning": 1.0, "afternoon": 1.0, "evening": 1.0, "night": 2.0}
DAY_START_MOD, DAY_END_MOD = 390.0, 1350.0  # waking window 06:30-22:30


@dataclass
class WindowPrediction:
    patient_id: str
    policy: str
    day: int
    window: str
    p_hypo: float
    truth_hypo: bool


@dataclass
class DayTracking:
    patient_id: str
    policy: str
    day: int
    mae: float
    tir_true: float
    tir_est: float
    n_readings: int


@dataclass(frozen=True)
class ExperimentConfig:
    n_days: int = 7
    run_in_days: int = 1
    n_particles: int = 500
    budgets: tuple[int, ...] = (1, 2, 3, 4)
    seed: int = 2026
    planner_ensemble: int = 300
    planner_truths: int = 150


# ----------------------------------------------------------------------------- helpers
def _truth_hypo(truth: TruthTrajectory, start: float, end: float) -> bool:
    mask = (truth.t_min >= start) & (truth.t_min < end)
    return bool(sustained_crossing(truth.glucose[mask][None, :], 70.0, 3)[0])


def _meter(truth: TruthTrajectory, t: float, seed: int) -> float:
    rng = np.random.default_rng([seed, round(t)])
    return sensor_reading(truth.at(t), SMBG_SENSOR, rng)


class _Runner:
    """Advances a filter along the logged event stream of one virtual patient."""

    def __init__(self, vp: VirtualPatient, logged: EventLog, cfg: ExperimentConfig, seed: int):
        self.vp = vp
        self.logged = logged
        self.pf = ParticleFilter(
            prior=build_prior(vp.profile),
            patient=PatientConstants(vp.profile.body_weight_kg, vp.profile.on_metformin),
            config=FilterConfig(n_particles=cfg.n_particles),
            seed=seed,
        )
        self.means: list[tuple[float, float]] = []

    def advance_to(self, t_target: float, record: bool = False) -> None:
        dt = self.pf.config.dt
        n = round((t_target - self.pf.t) / dt)
        if n <= 0:
            return
        grid = discretise(self.logged.window(self.pf.t, self.pf.t + n * dt), self.pf.t, n, dt)
        t0 = self.pf.t
        means = self.pf.run(grid, record_mean=record)
        if record and means is not None:
            self.means.extend((t0 + k * dt, float(g)) for k, g in enumerate(means))

    def future_grid(self, t_end: float) -> tuple[EventLog, int]:
        """Planning inputs: habitual routine for the future (the twin cannot see future logs)."""
        dt = self.pf.config.dt
        n = round((t_end - self.pf.t) / dt)
        future = routine_events(self.vp.profile, self.pf.t, self.pf.t + n * dt)
        return future, n

    def predict(self, start: float, end: float) -> float:
        dt = self.pf.config.dt
        events, n = self.future_grid(end)
        grid = discretise(events, self.pf.t, n, dt)
        traj = self.pf.forecast(grid, n_samples=400, rng=np.random.default_rng(int(self.pf.t)))
        k0 = round((start - self.pf.t) / dt)
        return float(sustained_crossing(traj[:, k0:], 70.0, 3).mean())


# ----------------------------------------------------------------------------- one run
def run_policy(
    vp: VirtualPatient,
    days: list[DayRecord],
    truth: TruthTrajectory,
    policy: PolicySpec,
    cfg: ExperimentConfig,
    seed: int,
) -> tuple[list[WindowPrediction], list[DayTracking]]:
    logged = EventLog()
    for d in days:
        logged.extend(d.logged_events)
    runner = _Runner(vp, logged, cfg, seed)
    preds: list[WindowPrediction] = []
    tracks: list[DayTracking] = []
    slots = waking_slots()
    policy_rng = np.random.default_rng([seed, 7])
    planner_cfg = PlannerConfig(n_ensemble=cfg.planner_ensemble, n_truths=cfg.planner_truths)
    cgm_t, cgm_y = cgm_trace(truth, np.random.default_rng([seed, 99])) if policy.name == "cgm" else ((), ())
    cgm_iter = iter(zip(cgm_t, cgm_y, strict=True))
    next_cgm = next(cgm_iter, None)

    def take_cgm_until(t_stop: float) -> int:
        nonlocal next_cgm
        count = 0
        while next_cgm is not None and next_cgm[0] <= t_stop:
            runner.advance_to(next_cgm[0], record=True)
            runner.pf.assimilate(float(next_cgm[1]), "cgm")
            count += 1
            next_cgm = next(cgm_iter, None)
        return count

    n_total_days = cfg.run_in_days + cfg.n_days
    for day in range(n_total_days):
        base = day * MINUTES_PER_DAY
        n_read = 0
        runner.means = []
        if policy.name == "fixed":
            schedule = [base + m for m in fixed_schedule(policy.budget)]
        elif policy.name == "random":
            schedule = [base + m for m in random_schedule(policy.budget, slots, policy_rng)]
        else:
            schedule = []
        budget_left = policy.budget if policy.name == "twin" else 0

        decision_points = sorted((base + t, w) for w, t in DECISION_MOD.items())
        slot_times = [base + s for s in slots]
        for s in slot_times:
            if policy.name == "cgm":
                n_read += take_cgm_until(s)
            runner.advance_to(s, record=True)
            # --- twin: (re)plan at wake-up and whenever the next planned time is passed
            if policy.name == "twin" and budget_left > 0:
                remaining = [
                    RiskWindow(
                        base + a,
                        base + b,
                        weight=WINDOW_WEIGHT[w],
                        label=w,
                        decision_min=base + DECISION_MOD[w],
                    )
                    for w, (a, b) in WINDOWS_MOD.items()
                    if base + DECISION_MOD[w] >= s
                ]
                if remaining:
                    horizon_end = max(w.end_min for w in remaining)
                    events, n = runner.future_grid(horizon_end)
                    grid = discretise(events, runner.pf.t, n, runner.pf.config.dt)
                    cands = [t for t in slot_times if t >= s]
                    plan = plan_readings(
                        runner.pf,
                        grid,
                        remaining,
                        cands,
                        budget_left,
                        config=planner_cfg,
                        rng=np.random.default_rng([seed, int(s)]),
                        precede_windows=True,
                    )
                    schedule = [plan.times[0]] if plan.readings else []
            if schedule and abs(schedule[0] - s) < 1e-6:
                runner.pf.assimilate(_meter(truth, s, seed), "smbg")
                schedule.pop(0)
                n_read += 1
                budget_left -= 1
            for t_dec, w in decision_points:
                if abs(t_dec - s) < 1e-6 and day >= cfg.run_in_days:
                    a, b = WINDOWS_MOD[w]
                    p = runner.predict(base + a, base + b)
                    preds.append(
                        WindowPrediction(
                            vp.patient_id, policy.key, day, w, p, _truth_hypo(truth, base + a, base + b)
                        )
                    )
        # tracking accuracy over the waking day
        if day >= cfg.run_in_days and runner.means:
            t_arr = np.array([m[0] for m in runner.means])
            g_est = np.array([m[1] for m in runner.means])
            g_true = np.array([truth.at(t) for t in t_arr])
            tracks.append(
                DayTracking(
                    vp.patient_id,
                    policy.key,
                    day,
                    float(np.mean(np.abs(g_est - g_true))),
                    float(np.mean((g_true >= 70) & (g_true <= 180))),
                    float(np.mean((g_est >= 70) & (g_est <= 180))),
                    n_read,
                )
            )
        # overnight
        next_morning = base + MINUTES_PER_DAY + DAY_START_MOD
        if day < n_total_days - 1:
            if policy.name == "cgm":
                n_read += take_cgm_until(next_morning - runner.pf.config.dt)
            runner.advance_to(next_morning)
    return preds, tracks


def run_patient(args: tuple[int, ExperimentConfig, float]) -> dict:
    index, cfg, max_hba1c = args
    rng = np.random.default_rng([cfg.seed, index])
    for _attempt in range(200):
        vp = sample_virtual_patient(rng, f"VP{index:03d}", force_sulfonylurea=True)
        if vp.profile.hba1c_pct <= max_hba1c:
            break
    else:  # pragma: no cover
        raise RuntimeError("inclusion criterion never met")
    n_total = cfg.run_in_days + cfg.n_days + 1
    days = [generate_day(vp, d, rng) for d in range(n_total)]
    truth = simulate_truth(vp, days, rng)
    policies = [PolicySpec("none"), PolicySpec("cgm")]
    for k in cfg.budgets:
        policies += [PolicySpec("fixed", k), PolicySpec("random", k), PolicySpec("twin", k)]
    seed = int(rng.integers(2**31))
    preds: list[WindowPrediction] = []
    tracks: list[DayTracking] = []
    for pol in policies:
        p, t = run_policy(vp, days, truth, pol, cfg, seed)
        preds += p
        tracks += t
    return {
        "patient": {
            "id": vp.patient_id,
            "endotype": vp.endotype,
            "hba1c": vp.profile.hba1c_pct,
            "regimens": [f"{r.drug} {r.dose_mg:g}mg x{len(r.times_hhmm)}" for r in vp.profile.regimens],
        },
        "predictions": [asdict(p) for p in preds],
        "tracking": [asdict(t) for t in tracks],
    }


# ----------------------------------------------------------------------------- metrics
def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels], scores[~labels]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = np.empty(scores.size)
    order = np.argsort(scores, kind="mergesort")
    ranks[order] = np.arange(1, scores.size + 1)
    for v in np.unique(scores):  # average ranks for ties
        tie = scores == v
        ranks[tie] = ranks[tie].mean()
    return float((ranks[labels].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def bootstrap_ci(
    scores: np.ndarray, labels: np.ndarray, groups: np.ndarray, n: int = 500, seed: int = 0
) -> tuple[float, float]:
    """Patient-level (cluster) bootstrap 95% CI for AUROC."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        a = auroc(scores[idx], labels[idx])
        if not np.isnan(a):
            vals.append(a)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarise(results: list[dict]) -> dict[str, dict]:
    by_policy: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        for p in r["predictions"]:
            by_policy[p["policy"]].append(p)
    track_by: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        for t in r["tracking"]:
            track_by[t["policy"]].append(t)
    summary = {}
    for pol, rows in sorted(by_policy.items()):
        s = np.array([r["p_hypo"] for r in rows])
        y = np.array([r["truth_hypo"] for r in rows])
        g = np.array([r["patient_id"] for r in rows])
        night = np.array([r["window"] == "night" for r in rows])
        lo, hi = bootstrap_ci(s, y, g)
        alert = s >= 0.2
        tr = track_by[pol]
        summary[pol] = {
            "n_windows": int(y.size),
            "n_hypo_windows": int(y.sum()),
            "auroc": auroc(s, y),
            "auroc_ci95": [lo, hi],
            "auroc_night": auroc(s[night], y[night]),
            "brier": float(np.mean((s - y) ** 2)),
            "sensitivity@0.2": float(alert[y].mean()) if y.any() else float("nan"),
            "alert_rate@0.2": float(alert.mean()),
            "tracking_mae": float(np.mean([t["mae"] for t in tr])) if tr else float("nan"),
            "tir_abs_error_pct": float(np.mean([abs(t["tir_est"] - t["tir_true"]) for t in tr]) * 100)
            if tr
            else float("nan"),
            "readings_per_day": float(np.mean([t["n_readings"] for t in tr])) if tr else 0.0,
        }
    return summary


def to_markdown(summary: dict[str, dict], meta: dict) -> str:
    lines = [
        "# Strip-budget experiment",
        "",
        f"Patients: {meta['n_patients']} SU-treated virtual patients (HbA1c <= {meta['max_hba1c']}%), "
        f"{meta['n_days']} evaluation days each, seed {meta['seed']}. Runtime {meta['runtime_s']:.0f} s.",
        "",
        "| Policy | Readings/day | AUROC (95% CI) | AUROC night | Brier | Sens@p>=0.2 | Alert rate | Tracking MAE (mg/dL) | TIR error (pp) |",  # noqa: E501
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def order(k: str) -> tuple[int, int]:
        name, _, b = k.partition("-")
        return (
            int(b) if b else (0 if name == "none" else 99),
            ["none", "fixed", "random", "twin", "cgm"].index(name),
        )

    for pol in sorted(summary, key=order):
        s = summary[pol]
        lines.append(
            f"| {pol} | {s['readings_per_day']:.1f} | {s['auroc']:.3f} ({s['auroc_ci95'][0]:.3f}-{s['auroc_ci95'][1]:.3f}) "  # noqa: E501
            f"| {s['auroc_night']:.3f} | {s['brier']:.4f} | {s['sensitivity@0.2']:.2f} | {s['alert_rate@0.2']:.2f} "  # noqa: E501
            f"| {s['tracking_mae']:.1f} | {s['tir_abs_error_pct']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def plot(summary: dict[str, dict], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, style in [("fixed", "s--"), ("random", "^:"), ("twin", "o-")]:
        ks = sorted(int(k.split("-")[1]) for k in summary if k.startswith(name + "-"))
        axes[0].plot(ks, [summary[f"{name}-{k}"]["auroc"] for k in ks], style, label=name, lw=2)
        axes[1].plot(ks, [summary[f"{name}-{k}"]["tracking_mae"] for k in ks], style, label=name, lw=2)
    for ax, key in zip(axes, ["auroc", "tracking_mae"], strict=True):
        for ref, ls in [("cgm", "-"), ("none", "-.")]:
            if ref in summary:
                ax.axhline(summary[ref][key], color="grey", ls=ls, lw=1, label=f"{ref} (reference)")
        ax.set_xlabel("Fingerstick strips per day")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AUROC: predicting hypoglycaemia per risk window")
    axes[1].set_ylabel("Glucose tracking error, MAE (mg/dL)")
    axes[0].legend()
    fig.suptitle("Pratibimb: value of choosing *when* to test")
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", type=int, default=40)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--particles", type=int, default=500)
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--max-hba1c", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=Path("results/strip_budget"))
    args = ap.parse_args(argv)

    cfg = ExperimentConfig(
        n_days=args.days, n_particles=args.particles, budgets=tuple(args.budgets), seed=args.seed
    )
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs = [(i, cfg, args.max_hba1c) for i in range(args.patients)]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            results = list(ex.map(run_patient, jobs))
    else:
        results = [run_patient(j) for j in jobs]
    summary = summarise(results)
    meta = {
        "n_patients": args.patients,
        "n_days": args.days,
        "max_hba1c": args.max_hba1c,
        "seed": args.seed,
        "particles": args.particles,
        "runtime_s": time.time() - t0,
    }
    (args.out / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=2))
    (args.out / "raw.json").write_text(json.dumps(results))
    md = to_markdown(summary, meta)
    (args.out / "summary.md").write_text(md, encoding="utf-8")
    plot(summary, args.out / "strip_budget.png")
    print(md)


if __name__ == "__main__":
    main()
