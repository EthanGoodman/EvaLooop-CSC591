"""Compare the code <-> pseudocode loop (B1) against the standard loop on the same models/tasks.

Usage:
    python scripts/analyze_b1.py --results results --out results/analysis \
        [--package <assignment4-data>/loops] [--sample 15] [--seed 0]

Inputs are the full-run results files written by `evaloop evaluate`
(<model>_code_generation_{summarization,pseudocode}_results.json). Outputs in --out:
    b1_stats.json     every number below, per model
    b1_report.md      readable tables: metrics, paired comparison, ranking, drift checks
    b1_failures.md    seeded sample of failed loops for manual inspection, plus
                      "broken by pseudocode only" / "broken by standard only" case lists
"""

import argparse
import ast
import json
import random
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaloop.analysis import metrics  # noqa: E402

STD, PSEUDO = "code_generation_summarization", "code_generation_pseudocode"


# ----------------------------------------------------------------------------- loading

def load_runs(results_dir: Path) -> dict:
    """{model: {loop: run}} for complete, full (non --limit) runs."""
    runs = {}
    for path in sorted(results_dir.glob("*_results.json")):
        (run,) = json.loads(path.read_text(encoding="utf-8"))
        if not run.get("complete") or run["config"].get("limit"):
            continue
        runs.setdefault(run["config"]["model"], {})[run["loop"]] = run
    return {m: r for m, r in runs.items() if STD in r and PSEUDO in r}


def by_task(run: dict) -> dict:
    return {t["task_id"]: t for t in run["prompt_results"]}


def inverse_body(wrapped: str) -> str:
    """The model's summary/pseudocode without the docstring wrapper and assert line."""
    lines = wrapped.strip().splitlines()
    return "\n".join(lines[1:-2]).strip()


def parses_as_python(text: str) -> bool:
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False


# ----------------------------------------------------------------------------- statistics

def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided sign test p-value (ties excluded)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def run_metrics(run: dict) -> dict:
    cycles = [t["successful_cycles"] for t in run["prompt_results"]]
    m = run["max_cycles"]
    return {
        "n_tasks": len(cycles),
        "pass@1": metrics.pass_at_k(cycles, 1) / len(cycles),
        "pass@M": metrics.pass_at_k(cycles, m) / len(cycles),
        "ASL_pow": metrics.asl_pow(cycles, m),
        "ASL_bias": metrics.asl_bias(cycles, m),
        "pass_drop": metrics.pass_drop(cycles, m),
        "histogram": metrics.cycle_histogram(cycles, m),
        "api_error_tasks": len(run.get("api_error_tasks", [])),
    }


def paired(std: dict, pseudo: dict) -> dict:
    """Per-task comparison of sustained loops on the same tasks."""
    s, p = by_task(std), by_task(pseudo)
    ids = sorted(set(s) & set(p))
    diff = [p[i]["successful_cycles"] - s[i]["successful_cycles"] for i in ids]
    wins, losses = sum(d > 0 for d in diff), sum(d < 0 for d in diff)
    # Cycle 1 runs the identical generation prompt in both loops: disagreement here is
    # run-to-run nondeterminism, not an effect of the loop type.
    c1 = [(s[i]["successful_cycles"] >= 1, p[i]["successful_cycles"] >= 1) for i in ids]
    both_c1 = [i for i in ids if s[i]["successful_cycles"] >= 1 and p[i]["successful_cycles"] >= 1]
    d2 = [p[i]["successful_cycles"] - s[i]["successful_cycles"] for i in both_c1]
    w2, l2 = sum(d > 0 for d in d2), sum(d < 0 for d in d2)
    return {
        "tasks": len(ids),
        "pseudo_longer": wins,
        "std_longer": losses,
        "ties": len(ids) - wins - losses,
        "mean_diff": sum(diff) / len(ids),
        "sign_test_p": sign_test(wins, losses),
        "cycle1_agreement": sum(a == b for a, b in c1) / len(ids),
        "cycle1_pass_std_only": sum(a and not b for a, b in c1),
        "cycle1_pass_pseudo_only": sum(b and not a for a, b in c1),
        "given_both_pass_cycle1": {
            "tasks": len(both_c1),
            "pseudo_longer": w2,
            "std_longer": l2,
            "mean_diff": sum(d2) / len(d2) if d2 else 0.0,
            "sign_test_p": sign_test(w2, l2),
        },
    }


def inverse_checks(run: dict) -> dict:
    """Format checks on the inverse task's outputs, and length drift across cycles."""
    texts = [c["task_2_output"] for t in run["prompt_results"] for c in t["cycles"] if "task_2_output" in c]
    bodies = [inverse_body(x) for x in texts]
    by_cycle = {}
    for t in run["prompt_results"]:
        for c in t["cycles"]:
            if "task_2_output" in c:
                by_cycle.setdefault(c["cycle"], []).append(len(inverse_body(c["task_2_output"])))
    return {
        "outputs": len(texts),
        "assert_anchor_kept": sum(x.strip().splitlines()[-2].lstrip().startswith("assert") for x in texts) / len(texts),
        "valid_python": sum(map(parses_as_python, bodies)) / len(bodies),
        "contains_def": sum("def " in b for b in bodies) / len(bodies),
        "mean_chars": sum(map(len, bodies)) / len(bodies),
        "mean_chars_by_cycle": {k: round(sum(v) / len(v), 1) for k, v in sorted(by_cycle.items())},
    }


def package_agreement(run: dict, package_dir: Path, package_id: str) -> dict:
    """Agreement of our standard-loop run with the published run of the same model."""
    path = package_dir / f"{package_id}.json"
    if not path.exists():
        return {}
    published = {t["task_id"]: t["successful_cycles"] for t in json.loads(path.read_text(encoding="utf-8"))["tasks"]}
    ours = {t["task_id"]: t["successful_cycles"] for t in run["prompt_results"]}
    ids = sorted(set(ours) & set(published))
    pub_cycles = [published[i] for i in ids]
    return {
        "package_model_id": package_id,
        "tasks": len(ids),
        "same_sustained_loops": sum(ours[i] == published[i] for i in ids) / len(ids),
        "same_cycle1_outcome": sum((ours[i] >= 1) == (published[i] >= 1) for i in ids) / len(ids),
        "published_ASL_pow": metrics.asl_pow(pub_cycles, 10),
        "published_pass@1": metrics.pass_at_k(pub_cycles, 1) / len(ids),
    }


# ----------------------------------------------------------------------------- failure dossier

def failure_entry(task: dict) -> dict:
    cycles = task["cycles"]
    failing = cycles[-1]
    previous = cycles[-2] if len(cycles) > 1 else None
    tests = failing.get("test_results", {})
    return {
        "task_id": task["task_id"],
        "failed_cycle": failing["cycle"],
        "initial_prompt": task["initial_prompt"].strip(),
        "prompt_of_failing_cycle": (previous or {}).get("task_2_output", task["initial_prompt"]).strip(),
        "last_passing_code": (previous or {}).get("task_1_output"),
        "failing_code": failing.get("task_1_output"),
        "base_passed": tests.get("base_success"),
        "plus_passed": tests.get("plus_success"),
    }


def failed_loops(run: dict) -> list:
    """Loops that broke after at least one successful cycle (cycle-1 failures are plain
    generation failures, identical in both loop types)."""
    return [t for t in run["prompt_results"]
            if 1 <= t["successful_cycles"] < run["max_cycles"] and "api_error_cycle" not in t]


def write_dossier(path: Path, runs: dict, sample: int, seed: int) -> dict:
    rng = random.Random(seed)
    out = ["# B1 failed-loop dossier\n",
           f"Seeded sample (seed={seed}) of loops that failed after >= 1 successful cycle. "
           "For each: the prompt that fed the failing cycle, the code that last passed, and the "
           "failing code. Classify each manually (e.g. requirement dropped, cannot implement own "
           "description, extraction/format issue, over-reliance on the assert line).\n"]
    sampled = {}
    for model, loops in runs.items():
        for loop, label, k in ((PSEUDO, "pseudocode loop", sample), (STD, "standard loop", max(5, sample // 3))):
            pool = failed_loops(loops[loop])
            picks = sorted(rng.sample(pool, min(k, len(pool))), key=lambda t: t["task_id"])
            sampled[f"{model} | {label}"] = [t["task_id"] for t in picks]
            out.append(f"\n## {model} - {label} ({len(picks)} of {len(pool)} mid-loop failures)\n")
            for t in picks:
                e = failure_entry(t)
                out.append(f"\n### {e['task_id']} - failed at cycle {e['failed_cycle']} "
                           f"(base={e['base_passed']}, plus={e['plus_passed']})\n")
                out.append("**Original prompt**\n```\n" + e["initial_prompt"] + "\n```\n")
                out.append("**Prompt of the failing cycle**\n```\n" + e["prompt_of_failing_cycle"] + "\n```\n")
                out.append("**Last passing code**\n```python\n" + str(e["last_passing_code"]) + "\n```\n")
                out.append("**Failing code**\n```python\n" + str(e["failing_code"]) + "\n```\n")
                out.append("**My Classification:** \n")

        s, p = by_task(loops[STD]), by_task(loops[PSEUDO])
        broke_pseudo = [i for i in s if s[i]["successful_cycles"] == 10 and 1 <= p[i]["successful_cycles"] < 10]
        broke_std = [i for i in s if p[i]["successful_cycles"] == 10 and 1 <= s[i]["successful_cycles"] < 10]
        out.append(f"\n## {model} - representative-case candidates\n")
        out.append(f"- Broken by pseudocode only (std survived 10): {', '.join(broke_pseudo) or 'none'}\n")
        out.append(f"- Broken by standard only (pseudocode survived 10): {', '.join(broke_std) or 'none'}\n")
    path.write_text("".join(out), encoding="utf-8")
    return sampled


# ----------------------------------------------------------------------------- report

def write_report(path: Path, stats: dict) -> None:
    lines = ["# B1 results: code <-> pseudocode vs standard loop\n",
             "\n## Metrics per run\n",
             "| model | loop | n | pass@1 | pass@10 | ASL_pow | ASL_bias | pass_drop | API-error tasks |\n",
             "|---|---|---|---|---|---|---|---|---|\n"]
    for model, s in stats["models"].items():
        for loop in (STD, PSEUDO):
            m = s["metrics"][loop]
            lines.append(f"| {model} | {loop} | {m['n_tasks']} | {m['pass@1']:.3f} | {m['pass@M']:.3f} | "
                         f"{m['ASL_pow']:.3f} | {m['ASL_bias']:.3f} | {m['pass_drop']:.3f} | {m['api_error_tasks']} |\n")
    lines += ["\n## Paired per-task comparison (pseudocode minus standard, sustained loops)\n",
              "| model | pseudo longer | std longer | ties | mean diff | sign-test p | cycle-1 agreement | "
              "given both pass c1: pseudo/std longer, p |\n|---|---|---|---|---|---|---|---|\n"]
    for model, s in stats["models"].items():
        p = s["paired"]
        g = p["given_both_pass_cycle1"]
        lines.append(f"| {model} | {p['pseudo_longer']} | {p['std_longer']} | {p['ties']} | {p['mean_diff']:+.3f} | "
                     f"{p['sign_test_p']:.3g} | {p['cycle1_agreement']:.3f} | "
                     f"{g['pseudo_longer']}/{g['std_longer']}, p={g['sign_test_p']:.3g} |\n")
    lines += ["\n## Ranking by ASL_pow\n"]
    for key, order in stats["ranking"].items():
        lines.append(f"- {key}: {' > '.join(order)}\n")
    lines += ["\n## Inverse-task output checks\n",
              "| model | loop | outputs | assert kept | valid Python | contains `def` | mean chars | mean chars by cycle |\n",
              "|---|---|---|---|---|---|---|---|\n"]
    for model, s in stats["models"].items():
        for loop in (STD, PSEUDO):
            c = s["inverse_checks"][loop]
            lines.append(f"| {model} | {loop} | {c['outputs']} | {c['assert_anchor_kept']:.3f} | {c['valid_python']:.3f} | "
                         f"{c['contains_def']:.3f} | {c['mean_chars']:.0f} | {c['mean_chars_by_cycle']} |\n")
    if any(s.get("package") for s in stats["models"].values()):
        lines += ["\n## Standard loop vs published data package (same model)\n",
                  "| model | package id | same sustained loops | same cycle-1 outcome | published ASL_pow | "
                  "published pass@1 |\n|---|---|---|---|---|---|\n"]
        for model, s in stats["models"].items():
            g = s.get("package")
            if g:
                lines.append(f"| {model} | {g['package_model_id']} | {g['same_sustained_loops']:.3f} | "
                             f"{g['same_cycle1_outcome']:.3f} | {g['published_ASL_pow']:.3f} | {g['published_pass@1']:.3f} |\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/analysis"))
    parser.add_argument("--package", type=Path, help="data package loops/ directory (optional cross-check)")
    parser.add_argument("--package_ids", default="", help="model=package_id pairs, comma-separated")
    parser.add_argument("--sample", type=int, default=15, help="failed pseudocode loops sampled per model")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    runs = load_runs(args.results)
    if not runs:
        sys.exit(f"No model has both complete full runs in {args.results}")
    package_ids = dict(pair.split("=", 1) for pair in args.package_ids.split(",") if pair)

    stats = {"models": {}, "ranking": {}}
    for model, loops in runs.items():
        entry = {
            "metrics": {loop: run_metrics(loops[loop]) for loop in (STD, PSEUDO)},
            "paired": paired(loops[STD], loops[PSEUDO]),
            "inverse_checks": {loop: inverse_checks(loops[loop]) for loop in (STD, PSEUDO)},
        }
        if args.package and model in package_ids:
            entry["package"] = package_agreement(loops[STD], args.package, package_ids[model])
        stats["models"][model] = entry
    for loop in (STD, PSEUDO):
        stats["ranking"][loop] = sorted(runs, key=lambda m: -stats["models"][m]["metrics"][loop]["ASL_pow"])
    if all("package" in s for s in stats["models"].values()):
        stats["ranking"]["published (data package)"] = sorted(
            runs, key=lambda m: -stats["models"][m]["package"]["published_ASL_pow"])

    args.out.mkdir(parents=True, exist_ok=True)
    stats["dossier_sample"] = write_dossier(args.out / "b1_failures.md", runs, args.sample, args.seed)
    (args.out / "b1_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_report(args.out / "b1_report.md", stats)
    print(f"wrote {args.out}/b1_report.md, b1_failures.md, b1_stats.json")


if __name__ == "__main__":
    main()
