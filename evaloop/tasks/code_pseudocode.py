import re

from .code_summarization import CodeSummarizationTask
from .factory import register_task


@register_task("code_pseudocode")
class CodePseudocodeTask(CodeSummarizationTask):
    """Task for converting code into language-agnostic pseudocode.

    The inverse of code generation in the code <-> pseudocode loop. It only
    replaces the natural-language summary with pseudocode: postprocess is
    inherited unchanged, so the pseudocode is wrapped as a docstring with the
    cycle's assert line re-attached, exactly like the standard loop. This keeps
    the function name/signature anchored across cycles and makes the summary
    format the only difference from the generation <-> summarization loop.
    """

    DEFAULT_PROMPT_TEMPLATE = (
        "Convert the following Python code into pseudocode:\n\n"
        "```\n{code}\n```\n\n"
        "Write language-agnostic pseudocode using plain structured keywords "
        "(FUNCTION, SET, IF, ELSE, FOR EACH, WHILE, RETURN). Do not use Python syntax "
        "and do not copy the Python code verbatim. "
        "Return only the pseudocode, without explanations or comments."
    )

    def create_prompt(self, input_text: str) -> str:
        """Create a prompt for code-to-pseudocode conversion."""
        prompt_template = self.task_config.get("prompt_template", self.DEFAULT_PROMPT_TEMPLATE)
        return prompt_template.format(code=input_text)

    def extract_output(self, raw_response: str) -> str:
        """Extract pseudocode from LLM response.

        Models often wrap pseudocode in a fenced block; the fence is formatting,
        not content, so the first fenced block is used when present (same rule
        as code extraction). Otherwise the whole response is used.
        """
        pattern = r"```(?:\w+)?\s*([\s\S]*?)```"
        matches = re.findall(pattern, raw_response)

        if matches:
            return matches[0].strip()

        return raw_response.strip()
