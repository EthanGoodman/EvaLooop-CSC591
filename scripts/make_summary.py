"""Build summary.csv (one row per run) from EvaLoop results files.

Usage:
    python scripts/make_summary.py results/*_results.json --out summary.csv \
        [--extended summary_extended.csv]

summary.csv uses the assignment's header:
    model,configuration,n_tasks,success_rate,asl_score
where success_rate = pass@1 / T and asl_score = ASL_pow = sum_i n_i * i^2 / (T * M), i.e. the
leaderboard's `aslScore` (no similarity judge). Metrics come from evaloop.analysis.metrics.

The optional extended CSV adds ASL_bias, pass_drop, the loop histogram and the number of
tasks stopped by API errors (which are kept in T and counted at the loops they had sustained).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaloop.analysis import metrics  # noqa: E402

CONFIGURATION_NAMES = {
    "code_generation_summarization": "standard-mbpp-plus",
    "code_generation_pseudocode": "pseudocode-mbpp-plus",
}


def summarize(path: Path) -> dict:
    """Compute the metrics of one results file (a single-model run)."""
    (run,) = json.loads(path.read_text(encoding="utf-8"))
    if not run.get("complete", True):
        raise ValueError(f"{path} is an incomplete (aborted) run")
    if run.get("config", {}).get("limit"):
        raise ValueError(f"{path} is a --limit smoke run, not a full run")

    cycles = [task["successful_cycles"] for task in run["prompt_results"]]
    max_cycles = run["max_cycles"]
    return {
        "model": run["config"]["model"],
        "configuration": CONFIGURATION_NAMES[run["loop"]],
        "n_tasks": len(cycles),
        "success_rate": metrics.pass_at_k(cycles, 1) / len(cycles),
        "asl_score": metrics.asl_pow(cycles, max_cycles),
        "asl_bias": metrics.asl_bias(cycles, max_cycles),
        "pass_drop": metrics.pass_drop(cycles, max_cycles),
        "pass_at_max": metrics.pass_at_k(cycles, max_cycles) / len(cycles),
        "histogram": " ".join(map(str, metrics.cycle_histogram(cycles, max_cycles))),
        "api_error_tasks": len(run.get("api_error_tasks", [])),
        "results_file": path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("summary.csv"))
    parser.add_argument("--extended", type=Path)
    args = parser.parse_args()

    rows = sorted((summarize(p) for p in args.results), key=lambda r: (r["model"], r["configuration"]))

    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "configuration", "n_tasks", "success_rate", "asl_score"])
        for r in rows:
            writer.writerow([r["model"], r["configuration"], r["n_tasks"],
                             f"{r['success_rate']:.3f}", f"{r['asl_score']:.3f}"])
    print(f"wrote {args.out} ({len(rows)} runs)")

    if args.extended:
        with args.extended.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            for r in rows:
                writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()})
        print(f"wrote {args.extended}")

    for r in rows:
        if r["api_error_tasks"]:
            print(f"WARNING: {r['results_file']}: {r['api_error_tasks']} task(s) stopped by API errors")


if __name__ == "__main__":
    main()
