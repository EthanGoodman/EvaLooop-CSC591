"""Offline tests of EvaLoopEvaluator's handling of failed model calls (no evalplus/torch/API)."""

import json
import sys
import types

import pytest

from evaloop.core.config import EvaluationConfig
from evaloop.tasks import TaskFactory

from conftest import GENERATION_CONFIG, MBPP_STYLE_PROMPT, SUMMARIZATION_CONFIG

CODE = "def similar_elements(a, b):\n    return tuple(set(a) & set(b))"


@pytest.fixture
def evaluator_cls(monkeypatch):
    """Import the evaluator with its heavy dependencies (evalplus, torch) stubbed out."""
    py_testing = types.ModuleType("evaloop.evaluation.py_testing")
    py_testing.EvalPlusCodeTester = object
    py_testing.load_mbpp_dataset = lambda: ({}, {})
    factory = types.ModuleType("evaloop.models.factory")
    factory.ModelFactory = object
    monkeypatch.setitem(sys.modules, "evaloop.evaluation.py_testing", py_testing)
    monkeypatch.setitem(sys.modules, "evaloop.models.factory", factory)
    monkeypatch.delitem(sys.modules, "evaloop.core.evaluator", raising=False)
    from evaloop.core.evaluator import EvaLoopEvaluator

    yield EvaLoopEvaluator
    monkeypatch.delitem(sys.modules, "evaloop.core.evaluator", raising=False)


class FlakyModel:
    """Returns passing code/summaries; raises for prompts while ``failures[marker]`` > 0."""

    max_workers = 4

    def __init__(self, failures=None):
        self.failures = dict(failures or {})

    def generate(self, prompt):
        for marker, remaining in self.failures.items():
            if marker in prompt and remaining != 0:
                self.failures[marker] = remaining - 1
                raise RuntimeError(f"API down for {marker}")
        if prompt.startswith("Generate Python code"):
            return f"```python\n{CODE}\n```"
        return "write a python function to find shared elements"

    def batch_generate(self, prompts):
        return [self.generate(p) for p in prompts]


class PassingTester:
    """Every generated code passes, without plus_details for one task (tester error shape)."""

    def test_code(self, code, task_id, entry_point):
        if task_id == "Mbpp/no-details":
            return {"success": True}
        return {"success": True, "plus_details": [True, True]}


def _run(evaluator_cls, tmp_path, model, task_ids, **config_kwargs):
    config = EvaluationConfig(
        model="gpt-4", output_dir=str(tmp_path), experiment_name="exp", max_cycles=3, **config_kwargs
    )
    evaluator = evaluator_cls(config)
    tasks = [TaskFactory().create_task(GENERATION_CONFIG), TaskFactory().create_task(SUMMARIZATION_CONFIG)]
    prompts = [{"task_id": t, "prompt": MBPP_STYLE_PROMPT.replace("shared", t)} for t in task_ids]
    problems = {t: {"entry_point": "similar_elements"} for t in task_ids}
    return evaluator._run_evaluation_cycles(model, tasks, prompts, problems, PassingTester())


def _by_id(result):
    return {r["task_id"]: r for r in result["prompt_results"]}


def test_clean_run(evaluator_cls, tmp_path):
    result = _run(evaluator_cls, tmp_path, FlakyModel(), ["Mbpp/1", "Mbpp/2"])
    assert [r["successful_cycles"] for r in result["prompt_results"]] == [3, 3]
    assert result["complete"] is True
    assert result["api_error_tasks"] == []


def test_transient_failure_is_retried(evaluator_cls, tmp_path):
    # Mbpp/1's prompts fail twice (the first attempt and one retry), then succeed
    result = _run(evaluator_cls, tmp_path, FlakyModel({"Mbpp/1": 2}), ["Mbpp/1", "Mbpp/2"])
    assert _by_id(result)["Mbpp/1"]["successful_cycles"] == 3
    assert result["api_error_tasks"] == []
    assert all(c["success"] is True for c in _by_id(result)["Mbpp/1"]["cycles"])


def test_persistent_failure_is_not_a_model_failure(evaluator_cls, tmp_path):
    task_ids = [f"Mbpp/{i}" for i in range(30)]  # 1/30 failing is below the 5% abort rate
    result = _run(evaluator_cls, tmp_path, FlakyModel({"Mbpp/7 ": -1}), task_ids)
    failed = _by_id(result)["Mbpp/7"]
    assert failed["api_error_cycle"] == 1
    assert failed["cycles"][-1]["success"] is None
    assert "test_results" not in failed["cycles"][-1]
    assert result["api_error_tasks"] == ["Mbpp/7"]
    assert all(_by_id(result)[t]["successful_cycles"] == 3 for t in task_ids if t != "Mbpp/7")


def test_summary_failure_does_not_count_as_pass_then_vanish(evaluator_cls, tmp_path):
    # Only the summarization prompt (which contains the code) fails, and only in cycle 2
    class SummaryFlaky(FlakyModel):
        def __init__(self):
            super().__init__()
            self.summary_calls = 0

        def generate(self, prompt):
            if not prompt.startswith("Generate Python code"):
                self.summary_calls += 1
                if self.summary_calls >= 2:
                    raise RuntimeError("summary call failed")
            return super().generate(prompt)

    task_ids = [f"Mbpp/{i}" for i in range(1)]
    with pytest.raises(RuntimeError, match="aborting"):
        _run(evaluator_cls, tmp_path, SummaryFlaky(), task_ids)
    saved = json.loads((tmp_path / "exp_code_generation_summarization_results.json").read_text())[0]
    (task,) = saved["prompt_results"]
    assert task["successful_cycles"] == 1
    assert task["api_error_cycle"] == 2


def test_systemic_failure_aborts_with_checkpoint(evaluator_cls, tmp_path):
    with pytest.raises(RuntimeError, match="aborting"):
        _run(evaluator_cls, tmp_path, FlakyModel({"Generate": -1}), ["Mbpp/1", "Mbpp/2"])
    saved = json.loads((tmp_path / "exp_code_generation_summarization_results.json").read_text())
    assert saved[0]["complete"] is False


def test_missing_plus_details_does_not_crash(evaluator_cls, tmp_path):
    result = _run(evaluator_cls, tmp_path, FlakyModel(), ["Mbpp/no-details"])
    assert result["prompt_results"][0]["successful_cycles"] == 3


def test_checkpoint_written_each_cycle(evaluator_cls, tmp_path):
    _run(evaluator_cls, tmp_path, FlakyModel(), ["Mbpp/1"], limit=1)
    saved = json.loads((tmp_path / "exp_code_generation_summarization_results.json").read_text())
    assert saved[0]["loop"] == "code_generation_summarization"
    assert saved[0]["config"]["limit"] == 1
