# EvaLoop extension: code ↔ pseudocode loop

CSC 591/791 Assignment 4, Option B (new attack).

- **Public fork:** https://github.com/EthanGoodman/EvaLooop-CSC591 (branch `feature/pseudocode-loop`)
- **Upstream:** [evalooop/EvaLooop](https://github.com/evalooop/EvaLooop) at `ce22fe2` (`dev`); paper: [arXiv:2505.12185](https://arxiv.org/abs/2505.12185)
- **Patch against upstream:** `b1-pseudocode-loop.patch` (code changes only)

## The extension

`--loop code_generation_pseudocode` replaces the standard loop's code → one-sentence summary step
with code → pseudocode. Everything else is identical to the standard loop
(`--loop code_generation_summarization`), so the format of the intermediate description is the
only difference between the two.

| | Standard loop (baseline) | Pseudocode loop (extension) |
|---|---|---|
| Generation prompt | `Generate Python code for the following task: \n{description}\nReturn only the code, without explanations or comments.` | identical |
| Inverse prompt | ``Use one sentence to summarize the following code and start with 'write a python function to':\n\n```\n{code}\n```\n\n```\nwrite a python function to\n``` `` | ``Convert the following Python code into pseudocode:\n\n```\n{code}\n```\n\nWrite language-agnostic pseudocode using plain structured keywords (FUNCTION, SET, IF, ELSE, FOR EACH, WHILE, RETURN). Do not use Python syntax and do not copy the Python code verbatim. Return only the pseudocode, without explanations or comments.`` |
| Output extraction | whole response | first fenced block if present, else whole response |
| Carried to the next cycle | `"""\n{summary}\n{assert line}\n"""` | `"""\n{pseudocode}\n{assert line}\n"""` |
| Failure condition | the generated code fails the MBPP+ `base` or `plus` tests (EvalPlus) | identical |
| Signature stability | the original prompt's final `assert` line is re-attached every cycle | identical (same `postprocess`, inherited) |

The model's own output is the only source of perturbation: apart from the fixed prompt templates,
the only text not written by the model is the original task's assert line, as in the standard loop.

## What changed in the existing framework, and why

New files:

| File | Purpose |
|---|---|
| `evaloop/tasks/code_pseudocode.py` | `CodePseudocodeTask`: the pseudocode prompt and extraction; inherits the assert-line `postprocess` from `CodeSummarizationTask` |
| `evaloop/models/openai_compatible.py` | `OpenAICompatibleLLM`: runs OpenRouter models (see below) |
| `Dockerfile`, `.dockerignore`, `scripts/run_in_docker.sh` | sandbox for executing generated code |
| `scripts/make_summary.py`, `scripts/analyze_b1.py` | `summary.csv` and the analysis outputs |
| `tests/test_pseudocode.py`, `tests/test_openai_compatible.py`, `tests/test_evaluator_robustness.py` | unit tests (offline, mocked models) |

Modified files:

| File | Change | Why |
|---|---|---|
| `evaloop/tasks/__init__.py` | import the new task | task registration (the framework's extension point) |
| `evaloop/core/config.py` | `loop` setting; `code_generation_pseudocode` task configs; generation config shared by both loops | the loop type was hard-coded; sharing guarantees an identical generation step. The standard loop's prompts are unchanged (checked by a test) |
| `evaloop/core/evaluator.py` | `run()` uses `config.loop`; results record `loop`, `config`, `complete`, `api_error_tasks` | the loop and results filename were hard-coded; provenance for reproducibility |
| `evaloop/core/evaluator.py` | prompts whose model call failed are re-run (2 retries); a task still failing is stopped and marked `api_error_cycle` instead of being scored as a model failure; abort if > 5% of a cycle fails; checkpoint after every cycle | the API client returns error text on failure, which was tested as code (or became the next prompt) and silently lowered ASL |
| `evaloop/core/evaluator.py` | `test_results.get("plus_details", [])` | a tester error crashed the whole run with a `KeyError` |
| `evaloop/core/config.py`, `evaloop/models/factory.py`, `evaloop/cli.py` | `--api_base_url` routes a model to `OpenAICompatibleLLM`; `--provider_order`, `--allow_fallbacks`, `--max_workers`, `--limit`, `--loop` | model ids such as `qwen/...` were routed to local vLLM/HuggingFace; OpenRouter providers serve different quantizations, so each run is pinned to one provider |
| `evaloop/models/factory.py` | vLLM/HuggingFace backends imported lazily | `open_source_vllm.py` raises `ImportError` without vLLM, so no model of any type could be created without it |
| `.gitignore` | ignore `*.key` | keep API keys out of git |

`OpenAICompatibleLLM` reuses the existing `OpenAILLM` unchanged and adds: raising on API errors,
OpenRouter provider pinning, `max_tokens` = 1024 (the published protocol; the existing `openai` path
sends 4096) and concurrent requests.

## Models, tools and dependencies

| Model (OpenRouter id) | Pinned provider | Precision |
|---|---|---|
| `google/gemma-3-12b-it` | `deepinfra/bf16` | bf16 |
| `qwen/qwen3-30b-a3b-instruct-2507` | `nebius/fp8` | fp8 (no bf16 endpoint available) |

Both models are in the assignment data package, which was used only as a cross-check of the
standard-loop runs. No LLM other than the two evaluated models was used (no similarity judge).
API cost: $0.51 for the four full runs (OpenRouter account usage).

- **Docker** (tested with Docker Desktop 29.7.2). The image is `python:3.10-slim` with CPU-only
  `torch==2.6.0`, `evalplus==0.3.1` (MBPP+ v0.2.0, 378 tasks, downloaded at build time),
  `openai==3.42.1` and `requirements.txt`; `pip freeze` is saved in the image at `/app/pip-freeze.txt`.
- **Bash** to run `scripts/run_in_docker.sh` (Git Bash on Windows).
- **Python ≥ 3.9** on the host for `scripts/make_summary.py` and `scripts/analyze_b1.py`
  (standard library plus this repo's `evaloop.analysis.metrics`).
- An **OpenRouter API key**.

**Sandbox.** Generated code is untrusted, and EvalPlus's guards are Unix-only: on native Windows,
EvalPlus scores even correct code as `timeout` because `import resource` fails. All runs therefore
execute inside the container, as an unprivileged user with all capabilities dropped and memory/PID
limits; only `results/` and `logs/` are mounted from the host.

## How to run

```bash
docker build -t evaloop-b1 .
echo "<openrouter key>" > openrouter.key     # ignored by git and docker

# usage: scripts/run_in_docker.sh <model id> <provider> <loop>
bash scripts/run_in_docker.sh google/gemma-3-12b-it            deepinfra/bf16 code_generation_summarization
bash scripts/run_in_docker.sh google/gemma-3-12b-it            deepinfra/bf16 code_generation_pseudocode
bash scripts/run_in_docker.sh qwen/qwen3-30b-a3b-instruct-2507 nebius/fp8     code_generation_summarization
bash scripts/run_in_docker.sh qwen/qwen3-30b-a3b-instruct-2507 nebius/fp8     code_generation_pseudocode
```

Each run writes `results/<model>_<loop>_results.json` and `logs/<model>_<loop>/evaloop.log`.
The four runs above used `--max_workers 24` (concurrency only; it does not affect outputs).

Settings of every run, recorded in each results file's `config`: temperature 0.0, top-p 1.0, at most
1024 new tokens, at most 10 cycles, one run per configuration, all 378 MBPP+ tasks (no sampling),
OpenRouter provider pinned with fallbacks disabled.

Unit tests: `docker run --rm --entrypoint python evaloop-b1 -m pytest`

## Reproducing the reported numbers

| Output | Command | Definition |
|---|---|---|
| `summary.csv` | `python scripts/make_summary.py results/*_results.json --out summary.csv --extended summary_extended.csv` | `success_rate` = pass@1 / 378; `asl_score` = `ASL_pow` = Σᵢ nᵢ·i² / (378·10), the leaderboard's `aslScore` (no similarity judge). Computed with `evaloop.analysis.metrics`, which reproduces the leaderboard's `aslScore` and `successRate` for both models when applied to the data package |
| `summary_extended.csv` | same command | adds ASL_bias, pass_drop, pass@10, the loop histogram and API-error counts |
| `results/analysis/b1_report.md`, `b1_stats.json` | `python scripts/analyze_b1.py --results results --out results/analysis --package <assignment4-data>/loops --package_ids "google/gemma-3-12b-it=gemma-3-12b-it,qwen/qwen3-30b-a3b-instruct-2507=qwen3-30b-a3b-instruct-2507"` | per-run metrics, paired per-task comparison (exact sign test), ranking, pseudocode format checks, agreement with the data package |
| `results/analysis/b1_failures.md` | same command (`--seed 0 --sample 15` by default) | seeded sample of failed loops used for the manual failure analysis |

Results files: `results/gemma-3-12b-it_code_generation_{summarization,pseudocode}_results.json`
and `results/qwen3-30b-a3b-instruct-2507_code_generation_{summarization,pseudocode}_results.json`.

## License

MIT, as upstream (see `LICENSE`).
