"""Experiments and baseline for EC Assignment 2 (Team 15).

Research question: how does tournament selection pressure (k=2 vs k=6) affect
an EA evolving a neural-network controller for targeted locomotion?

This script:
  1. runs both EA variants (ea.py) for several independent seeds,
  2. runs an equal-budget random-search baseline with the same seeds,
  3. evaluates a fixed non-evolved controller (all-zero weights) as a reference,
  4. saves every run as JSON and produces the convergence plot, a boxplot,
     a summary table and significance tests for the report.

Fitness = distance to target, so LOWER is better.

Usage (run from the folder with ea.py / simulation.py / controller.py):
    uv run python experiments.py --time-eval          # time evaluations, estimate total runtime
    uv run python experiments.py --quick              # smoke test (tiny pop, few gens, 1 seed)
    uv run python experiments.py                      # full experiment: 5 seeds x 3 methods
    uv run python experiments.py --generations 60     # more generations (baseline budget follows)
    uv run python experiments.py --workers 4          # run several runs in parallel
    uv run python experiments.py --plot-only          # re-make plots/tables from saved runs
    uv run python experiments.py --video high_pressure  # record the best controller of a method

Finished runs are skipped on restart (same seed, same budget), so an
interrupted experiment can simply be started again. Use --overwrite to redo.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from ea import VARIANTS, EAConfig, run_ea
from simulation import evaluate as evaluate_genotype
from simulation import get_problem_size, run_experiment

# --- OUTPUT LOCATIONS --- #
RESULTS_ROOT = Path.cwd() / "__data__" / "experiments"
BASELINE = "random_search"
DEFAULT_SEEDS = [1, 2, 3, 4, 5]

LABELS = {
    "low_pressure": "EA, tournament k=2 (low pressure)",
    "high_pressure": "EA, tournament k=6 (high pressure)",
    BASELINE: "Random search (same budget)",
}
COLORS = {
    "low_pressure": "tab:blue",
    "high_pressure": "tab:red",
    BASELINE: "tab:gray",
}


def _to_builtin(obj: object) -> object:
    """JSON fallback for numpy types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot serialise {type(obj)}")


# --------------------------------------------------------------------------- #
# Baseline: random search with exactly the EA's evaluation budget
# --------------------------------------------------------------------------- #
def run_random_search(seed: int, config: EAConfig) -> dict[str, object]:
    """Sample genotypes uniformly in the EA's init range, keep the best.

    Evaluations are grouped into the same "generations" as the EA
    (population_size first, then n_offspring per generation), so the curves
    line up on the same x-axis and the total budget is identical.
    With the same seed, the first batch equals the EA's initial population.
    """
    rng = np.random.default_rng(seed)
    n_genes = get_problem_size()[2]

    best_fitness = np.inf
    best_genotype: list[float] = []
    evaluations = 0
    history = []

    batch_sizes = [config.population_size] + [config.n_offspring] * config.num_generations
    for generation, size in enumerate(batch_sizes):
        batch = []
        for _ in range(size):
            genotype = rng.uniform(config.init_low, config.init_high, size=n_genes)
            fitness = float(evaluate_genotype(genotype))
            batch.append(fitness)
            if fitness < best_fitness:
                best_fitness, best_genotype = fitness, genotype.tolist()
        evaluations += size
        history.append(
            {
                "generation": generation,
                "evaluations": evaluations,
                "best": best_fitness,  # best-so-far
                "mean": float(np.mean(batch)),  # mean of this batch
                "worst": float(np.max(batch)),
            }
        )

    return {
        "variant": BASELINE,
        "seed": seed,
        "tournament_size": None,
        "history": history,
        "best_fitness": best_fitness,
        "best_genotype": best_genotype,
        "evaluations": evaluations,
    }


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #
def run_path(results_dir: Path, method: str, seed: int) -> Path:
    return results_dir / "runs" / f"{method}_seed{seed}.json"


def _job(method: str, seed: int, config_dict: dict, results_dir: str) -> tuple[str, int, float, float]:
    """One independent run. Top-level so it can run in a worker process."""
    config = EAConfig(**config_dict)
    (Path.cwd() / "__data__" / "ea").mkdir(parents=True, exist_ok=True)  # ea.py's db folder

    start = time.perf_counter()
    if method == BASELINE:
        result = run_random_search(seed, config)
    else:
        result = run_ea(method, seed=seed, config=config)
    wall = time.perf_counter() - start

    result["wall_time_s"] = wall
    result["config"] = config_dict
    result["evaluation_budget"] = config.evaluation_budget

    path = run_path(Path(results_dir), method, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, default=_to_builtin, indent=1))
    return method, seed, float(result["best_fitness"]), wall


def already_done(path: Path, budget: int) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("evaluations") == budget
    except json.JSONDecodeError:
        return False


def run_all(
    methods: list[str],
    seeds: list[int],
    config: EAConfig,
    results_dir: Path,
    workers: int,
    overwrite: bool,
) -> None:
    budget = config.evaluation_budget
    print(
        f"Budget per run: {budget} evaluations "
        f"(pop {config.population_size} + {config.num_generations} gens x {config.n_offspring} offspring)"
    )

    jobs = []
    for seed in seeds:
        for method in methods:
            if not overwrite and already_done(run_path(results_dir, method, seed), budget):
                print(f"  skip {method:14s} seed {seed} (already done)")
                continue
            jobs.append((method, seed))

    if not jobs:
        print("Nothing to run.")
        return
    print(f"Running {len(jobs)} runs with {workers} worker(s)...")

    config_dict = asdict(config)
    t0 = time.perf_counter()
    if workers <= 1:
        for i, (method, seed) in enumerate(jobs, 1):
            m, s, best, wall = _job(method, seed, config_dict, str(results_dir))
            print(f"  [{i}/{len(jobs)}] {m:14s} seed {s}: best {best:.4f}  ({wall / 60:.1f} min)")
    else:
        # "spawn" gives each worker a clean MuJoCo state
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            futures = [pool.submit(_job, m, s, config_dict, str(results_dir)) for m, s in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                m, s, best, wall = fut.result()
                print(f"  [{i}/{len(jobs)}] {m:14s} seed {s}: best {best:.4f}  ({wall / 60:.1f} min)")
    print(f"All runs finished in {(time.perf_counter() - t0) / 60:.1f} min.")


def zero_controller_reference(results_dir: Path) -> float:
    """Fixed non-evolved controller: all weights 0 -> all joints held at 0."""
    path = results_dir / "zero_controller.json"
    if path.exists():
        return json.loads(path.read_text())["fitness"]
    n_genes = get_problem_size()[2]
    fitness = float(evaluate_genotype(np.zeros(n_genes)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fitness": fitness}))
    return fitness


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def load_runs(results_dir: Path) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    for path in sorted((results_dir / "runs").glob("*.json")):
        run = json.loads(path.read_text())
        runs.setdefault(run["variant"], []).append(run)
    return runs


def curves(runs: list[dict], key: str) -> np.ndarray:
    """(n_runs, n_generations+1) array; truncates to the shortest run."""
    length = min(len(r["history"]) for r in runs)
    if any(len(r["history"]) != length for r in runs):
        print(f"  warning: runs have different lengths, truncating to {length} points")
    arr = np.array([[row[key] for row in r["history"][:length]] for r in runs], dtype=float)
    if key == "best":
        arr = np.minimum.accumulate(arr, axis=1)  # best-so-far
    return arr


def plot_convergence(runs: dict[str, list[dict]], zero_fit: float | None, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_best, ax_mean) = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for method in [*VARIANTS, BASELINE]:
        if method not in runs:
            continue
        gens = np.arange(curves(runs[method], "best").shape[1])
        n = len(runs[method])
        for ax, key in ((ax_best, "best"), (ax_mean, "mean")):
            if key == "mean" and method == BASELINE:
                continue  # a random-search batch is not a population
            arr = curves(runs[method], key)
            mu, sd = arr.mean(axis=0), arr.std(axis=0)
            ax.plot(gens, mu, color=COLORS[method], label=f"{LABELS[method]} (n={n})")
            ax.fill_between(gens, mu - sd, mu + sd, color=COLORS[method], alpha=0.2)

    if zero_fit is not None:
        for ax in (ax_best, ax_mean):
            ax.axhline(zero_fit, color="black", ls="--", lw=1, label="Zero-weight controller")

    ax_best.set_title("Best fitness so far (mean ± std over runs)")
    ax_mean.set_title("Population mean fitness (mean ± std over runs)")
    for ax in (ax_best, ax_mean):
        ax.set_xlabel("Generation")
        ax.set_ylabel("Distance to target (lower is better)")
        ax.grid(alpha=0.3)
    ax_best.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=200)
    fig.savefig(out / "convergence.pdf")
    plt.close(fig)


def plot_final_box(runs: dict[str, list[dict]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [m for m in [*VARIANTS, BASELINE] if m in runs]
    data = [[r["best_fitness"] for r in runs[m]] for m in methods]
    fig, ax = plt.subplots(figsize=(6, 4))
    names = [LABELS[m].split(" (")[0] for m in methods]
    try:
        ax.boxplot(data, tick_labels=names)  # matplotlib >= 3.9
    except TypeError:
        ax.boxplot(data, labels=names)
    for i, vals in enumerate(data, 1):
        ax.scatter(np.full(len(vals), i), vals, color=COLORS[methods[i - 1]], zorder=3, s=15)
    ax.set_ylabel("Final best distance to target (lower is better)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "final_best_boxplot.png", dpi=200)
    fig.savefig(out / "final_best_boxplot.pdf")
    plt.close(fig)


def summarise(runs: dict[str, list[dict]], zero_fit: float | None, out: Path) -> None:
    rows = []
    for method in [m for m in [*VARIANTS, BASELINE] if m in runs]:
        finals = np.array([r["best_fitness"] for r in runs[method]], dtype=float)
        walls = [r.get("wall_time_s", np.nan) for r in runs[method]]
        rows.append(
            {
                "method": method,
                "n_runs": len(finals),
                "evaluations": runs[method][0]["evaluations"],
                "mean_best": finals.mean(),
                "std_best": finals.std(ddof=1) if len(finals) > 1 else 0.0,
                "median_best": np.median(finals),
                "min_best": finals.min(),
                "max_best": finals.max(),
                "mean_wall_min": np.nanmean(walls) / 60,
            }
        )

    with open(out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Final best fitness (distance to target, lower is better) ===")
    for r in rows:
        print(
            f"{r['method']:14s} n={r['n_runs']}  mean {r['mean_best']:.4f} ± {r['std_best']:.4f}  "
            f"median {r['median_best']:.4f}  [{r['min_best']:.4f}, {r['max_best']:.4f}]  "
            f"evals {r['evaluations']}"
        )
    if zero_fit is not None:
        print(f"{'zero weights':14s} fitness {zero_fit:.4f}  (fixed non-evolved controller)")

    # Plateau check (assignment: stop when the curve plateaus, not at a fixed gen count)
    print("\n=== Plateau check (mean best-so-far over last 20% of generations) ===")
    for method in [m for m in VARIANTS if m in runs]:
        mu = curves(runs[method], "best").mean(axis=0)
        cut = max(1, int(0.8 * (len(mu) - 1)))
        drop = (mu[cut] - mu[-1]) / max(mu[cut], 1e-12) * 100
        verdict = "still improving -> add generations" if drop > 2 else "plateaued"
        print(f"{method:14s} improved {drop:.1f}% from gen {cut} to {len(mu) - 1}: {verdict}")

    # Significance tests on final best fitness
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        print("\n(scipy not installed: skipping Mann-Whitney U tests)")
        return
    pairs = [("low_pressure", "high_pressure"), ("low_pressure", BASELINE), ("high_pressure", BASELINE)]
    lines = ["pair,U,p_value"]
    print("\n=== Mann-Whitney U, two-sided, on final best fitness ===")
    for a, b in pairs:
        if a in runs and b in runs and len(runs[a]) > 1 and len(runs[b]) > 1:
            fa = [r["best_fitness"] for r in runs[a]]
            fb = [r["best_fitness"] for r in runs[b]]
            res = mannwhitneyu(fa, fb, alternative="two-sided")
            print(f"{a} vs {b}: U={res.statistic:.1f}, p={res.pvalue:.4f}")
            lines.append(f"{a} vs {b},{res.statistic},{res.pvalue}")
    (out / "stats_tests.csv").write_text("\n".join(lines) + "\n")


def analyse(results_dir: Path) -> None:
    runs = load_runs(results_dir)
    if not runs:
        print(f"No runs found in {results_dir / 'runs'}")
        return
    zero_path = results_dir / "zero_controller.json"
    zero_fit = json.loads(zero_path.read_text())["fitness"] if zero_path.exists() else None

    plot_convergence(runs, zero_fit, results_dir)
    plot_final_box(runs, results_dir)
    summarise(runs, zero_fit, results_dir)
    print(f"\nPlots and tables written to {results_dir}")


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def time_evaluations(config: EAConfig, n_methods: int, n_seeds: int, workers: int, n: int = 5) -> None:
    n_genes = get_problem_size()[2]
    rng = np.random.default_rng(0)
    start = time.perf_counter()
    for _ in range(n):
        evaluate_genotype(rng.uniform(config.init_low, config.init_high, n_genes))
    per_eval = (time.perf_counter() - start) / n
    per_run = per_eval * config.evaluation_budget
    total = per_run * n_methods * n_seeds / max(workers, 1)
    print(f"{per_eval:.2f} s per evaluation  ->  {per_run / 60:.1f} min per run")
    print(f"{n_methods} methods x {n_seeds} seeds with {workers} worker(s): ~{total / 60:.0f} min total")


def record_best(method: str, results_dir: Path) -> None:
    runs = load_runs(results_dir).get(method, [])
    if not runs:
        print(f"No runs for {method} in {results_dir}")
        return
    best = min(runs, key=lambda r: r["best_fitness"])
    print(f"Recording {method} seed {best['seed']} (fitness {best['best_fitness']:.4f})")
    run_experiment(best["best_genotype"], mode="video", verbose=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="EC Assignment 2 experiments: tournament size vs random search.")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--generations", type=int, default=EAConfig.num_generations)
    p.add_argument("--population", type=int, default=EAConfig.population_size)
    p.add_argument("--methods", nargs="+", default=[*VARIANTS, BASELINE], choices=[*VARIANTS, BASELINE])
    p.add_argument("--workers", type=int, default=1, help="parallel runs (check CPU count first)")
    p.add_argument("--overwrite", action="store_true", help="redo runs that already exist")
    p.add_argument("--quick", action="store_true", help="smoke test: pop 10, 3 gens, seed 1, separate folder")
    p.add_argument("--plot-only", action="store_true", help="only re-make plots/tables from saved runs")
    p.add_argument("--time-eval", action="store_true", help="time a few evaluations and estimate runtime")
    p.add_argument("--video", choices=[*VARIANTS, BASELINE], help="record the best controller of a method")
    args = p.parse_args()

    if args.quick:
        config = EAConfig(population_size=10, num_generations=3)
        seeds = [1]
        results_dir = RESULTS_ROOT.with_name("experiments_quick")
    else:
        config = EAConfig(population_size=args.population, num_generations=args.generations)
        seeds = args.seeds
        results_dir = RESULTS_ROOT
    # Results of different budgets go to separate folders so they never mix.
    if not args.quick and (args.generations, args.population) != (EAConfig.num_generations, EAConfig.population_size):
        results_dir = RESULTS_ROOT.with_name(f"experiments_pop{args.population}_gen{args.generations}")
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.time_eval:
        time_evaluations(config, len(args.methods), len(seeds), args.workers)
        return
    if args.video:
        record_best(args.video, results_dir)
        return
    if not args.plot_only:
        zero_controller_reference(results_dir)
        run_all(args.methods, seeds, config, results_dir, args.workers, args.overwrite)
    analyse(results_dir)


if __name__ == "__main__":
    main()
