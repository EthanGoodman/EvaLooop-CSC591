"""Unit tests for the code <-> pseudocode loop task."""

import pytest

from evaloop.core.config import SUPPORTED_LOOPS, EvaluationConfig
from evaloop.evaluation.cycle import BatchEvaluationCycle
from evaloop.tasks import CodePseudocodeTask, TaskFactory

from conftest import ASSERT_LINE, GENERATION_CONFIG, MBPP_STYLE_PROMPT, SUMMARIZATION_CONFIG

PSEUDOCODE = "FUNCTION similar_elements(a, b)\n    RETURN items present in both a and b"


class TestCodePseudocodeTask:
    def test_registered(self):
        task = TaskFactory().create_task({"type": "code_pseudocode"})
        assert isinstance(task, CodePseudocodeTask)

    def test_create_prompt_default_template(self):
        task = CodePseudocodeTask({"type": "code_pseudocode"})
        prompt = task.create_prompt("def f(x): return x")
        assert "def f(x): return x" in prompt
        assert "pseudocode" in prompt
        assert "Do not use Python syntax" in prompt

    def test_create_prompt_uses_configured_template(self):
        task = CodePseudocodeTask({"prompt_template": "P: {code}"})
        assert task.create_prompt("x = 1") == "P: x = 1"

    def test_extract_output_fenced_block(self):
        task = CodePseudocodeTask({})
        response = f"Here is the pseudocode:\n```\n{PSEUDOCODE}\n```\nDone."
        assert task.extract_output(response) == PSEUDOCODE

    def test_extract_output_no_fence_returns_stripped_response(self):
        task = CodePseudocodeTask({})
        assert task.extract_output(f"  {PSEUDOCODE}  \n") == PSEUDOCODE

    def test_postprocess_retains_assert_line(self):
        task = CodePseudocodeTask({})
        result = task.postprocess(PSEUDOCODE, {"cycle_input": MBPP_STYLE_PROMPT})
        assert result == f'"""\n{PSEUDOCODE}\n{ASSERT_LINE}\n"""\n'

    def test_assert_line_survives_repeated_cycles(self):
        # Each cycle's input is the previous cycle's wrapped output, so the
        # anchor must still be found at [-2] after many cycles.
        task = CodePseudocodeTask({})
        cycle_input = MBPP_STYLE_PROMPT
        for _ in range(10):
            cycle_input = task.postprocess(PSEUDOCODE, {"cycle_input": cycle_input})
        assert cycle_input.splitlines()[-2] == ASSERT_LINE

    def test_is_not_testable(self):
        assert CodePseudocodeTask({}).produces_testable_code is False


def test_cycle_with_pseudocode_task():
    code = "def similar_elements(a, b):\n    return tuple(set(a) & set(b))"

    class Model:
        def generate(self, prompt):
            if prompt.startswith("Generate Python code"):
                return f"```python\n{code}\n```"
            return f"```\n{PSEUDOCODE}\n```"

    factory = TaskFactory()
    tasks = [factory.create_task(GENERATION_CONFIG), factory.create_task({"type": "code_pseudocode"})]
    result = BatchEvaluationCycle(Model(), tasks).run_single_cycle(
        [{"task_id": "Mbpp/2", "prompt": MBPP_STYLE_PROMPT}], cycle_number=1
    )
    (res,) = result["cycle_results"]
    assert res["task_1_output"] == code
    assert res["task_2_output"] == f'"""\n{PSEUDOCODE}\n{ASSERT_LINE}\n"""\n'


class TestLoopSelection:
    def _config(self, tmp_path, **kwargs):
        return EvaluationConfig(model="gpt-4", output_dir=str(tmp_path), **kwargs)

    def test_default_loop_is_standard(self, tmp_path):
        assert self._config(tmp_path).loop == "code_generation_summarization"

    def test_standard_loop_prompts_unchanged(self, tmp_path):
        # The baseline must use exactly the prompts that produced the published data
        configs = self._config(tmp_path).get_task_configs("code_generation_summarization")
        assert configs == [GENERATION_CONFIG, SUMMARIZATION_CONFIG]

    def test_pseudocode_loop_shares_generation_step(self, tmp_path):
        config = self._config(tmp_path, loop="code_generation_pseudocode")
        configs = config.get_task_configs(config.loop)
        assert configs == [GENERATION_CONFIG, {"type": "code_pseudocode"}]

    def test_every_supported_loop_builds_tasks(self, tmp_path):
        factory = TaskFactory()
        for loop in SUPPORTED_LOOPS:
            tasks = [factory.create_task(c) for c in self._config(tmp_path).get_task_configs(loop)]
            assert [t.produces_testable_code for t in tasks] == [True, False]

    def test_unknown_loop_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="loop must be one of"):
            self._config(tmp_path, loop="nope")

    def test_loop_recorded_in_config_dict(self, tmp_path):
        assert self._config(tmp_path, loop="code_generation_pseudocode").to_dict()["loop"] == (
            "code_generation_pseudocode"
        )
