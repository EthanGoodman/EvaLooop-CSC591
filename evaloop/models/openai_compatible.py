"""OpenAI-compatible API models (e.g. OpenRouter) with fail-loud errors and concurrency."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from .closed_source import OpenAILLM

# Prefixes OpenAILLM.generate returns instead of raising when a request fails
_ERROR_PREFIXES = ("Error generating response", "Error: Still received empty response")


class APICallError(RuntimeError):
    """An API request failed; the model never produced an output for this prompt."""


class OpenAICompatibleLLM(OpenAILLM):
    """OpenAILLM for any OpenAI-compatible endpoint, such as OpenRouter.

    Differences from OpenAILLM, all needed to run API models inside the loop:
      - API failures raise APICallError instead of returning an error string, which the
        loop would otherwise test as generated code (or carry forward as the next prompt)
        and count as a model failure.
      - Optional OpenRouter provider routing (``provider`` config) so that every request
        of a run is served by the same provider/quantization.
      - batch_generate runs requests concurrently; BatchEvaluationCycle uses it automatically.
    """

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        # OpenRouter provider routing preferences, e.g. {"order": ["deepinfra"], "allow_fallbacks": False}
        self.provider = model_config.get("provider")
        self.max_workers = model_config.get("max_workers", 8)
        # Retries on top of OpenAILLM's own, for failures it does not retry (e.g. connection errors)
        self.call_attempts = model_config.get("call_attempts", 3)
        self.call_backoff = model_config.get("call_backoff", 10.0)
        self.logger = logging.getLogger(__name__)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response, raising APICallError if every attempt fails."""
        if self.provider:
            kwargs.setdefault("extra_body", {"provider": self.provider})

        for attempt in range(1, self.call_attempts + 1):
            response = super().generate(prompt, **kwargs)
            if not response.startswith(_ERROR_PREFIXES):
                return response
            self.logger.warning(f"API call failed (attempt {attempt}/{self.call_attempts}): {response}")
            if attempt < self.call_attempts:
                time.sleep(self.call_backoff * attempt)
        raise APICallError(response)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        """Generate responses for prompts concurrently, preserving order."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self.generate, prompts))
