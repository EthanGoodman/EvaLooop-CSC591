"""Tests for OpenAI-compatible endpoint routing and the OpenAICompatibleLLM wrapper (no API calls)."""

import pytest

from evaloop.core.config import EvaluationConfig


class TestRouting:
    def _model_config(self, tmp_path, **kwargs):
        return EvaluationConfig(output_dir=str(tmp_path), **kwargs).get_model_config()

    def test_base_url_routes_to_openai_compatible(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        config = self._model_config(
            tmp_path,
            model="qwen/qwen-2.5-coder-32b-instruct",
            api_base_url="https://openrouter.ai/api/v1",
        )
        assert config["type"] == "openai_compatible"
        assert config["model_id"] == "qwen/qwen-2.5-coder-32b-instruct"
        assert config["base_url"] == "https://openrouter.ai/api/v1"
        assert config["api_key"] == "sk-test"
        assert config["max_tokens"] == 1024
        assert config["temperature"] == 0.0
        assert "provider" not in config

    def test_provider_pinning(self, tmp_path):
        config = self._model_config(
            tmp_path,
            model="qwen/qwen-2.5-coder-32b-instruct",
            api_base_url="https://openrouter.ai/api/v1",
            provider_order=["deepinfra"],
        )
        assert config["provider"] == {"order": ["deepinfra"], "allow_fallbacks": False}

    def test_without_base_url_routing_is_unchanged(self, tmp_path):
        assert self._model_config(tmp_path, model="gpt-4")["type"] == "openai"


class TestOpenAICompatibleLLM:
    @pytest.fixture
    def llm_cls(self, monkeypatch):
        pytest.importorskip("openai")
        from evaloop.models import closed_source
        from evaloop.models.openai_compatible import OpenAICompatibleLLM

        calls = []
        responses = []

        def fake_generate(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return responses.pop(0) if responses else f"out:{prompt}"

        monkeypatch.setattr(closed_source.OpenAILLM, "generate", fake_generate)
        OpenAICompatibleLLM.calls = calls
        OpenAICompatibleLLM.responses = responses
        return OpenAICompatibleLLM

    def _make(self, llm_cls, **kwargs):
        config = {"name": "m", "model_id": "m", "api_key": "sk-test", "call_backoff": 0}
        config.update(kwargs)
        return llm_cls(config)

    def test_passes_provider_routing(self, llm_cls):
        provider = {"order": ["deepinfra"], "allow_fallbacks": False}
        assert self._make(llm_cls, provider=provider).generate("p") == "out:p"
        assert llm_cls.calls[0][1] == {"extra_body": {"provider": provider}}

    def test_no_provider_sends_no_extra_body(self, llm_cls):
        self._make(llm_cls).generate("p")
        assert llm_cls.calls[0][1] == {}

    def test_error_string_raises_after_retries(self, llm_cls):
        from evaloop.models.openai_compatible import APICallError

        llm_cls.responses.extend(["Error generating response: 402 insufficient credits"] * 3)
        with pytest.raises(APICallError, match="402"):
            self._make(llm_cls, call_attempts=3).generate("p")
        assert len(llm_cls.calls) == 3

    def test_empty_response_error_raises(self, llm_cls):
        from evaloop.models.openai_compatible import APICallError

        llm_cls.responses.append("Error: Still received empty response after multiple attempts.")
        with pytest.raises(APICallError):
            self._make(llm_cls, call_attempts=1).generate("p")

    def test_recovers_after_transient_error(self, llm_cls):
        llm_cls.responses.extend(["Error generating response: Connection error.", "def f(): pass"])
        assert self._make(llm_cls).generate("p") == "def f(): pass"

    def test_batch_generate_preserves_order(self, llm_cls):
        prompts = [f"p{i}" for i in range(20)]
        outputs = self._make(llm_cls, max_workers=4).batch_generate(prompts)
        assert outputs == [f"out:{p}" for p in prompts]


def test_factory_creates_api_model_without_torch_or_vllm(monkeypatch):
    # Regression: the factory used to import the vLLM backend eagerly, and that module
    # raises ImportError without vLLM, so no model type could be created.
    pytest.importorskip("openai")
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "vllm", None)
    for name in ("evaloop.models.factory", "evaloop.models.open_source", "evaloop.models.open_source_vllm"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    from evaloop.models.factory import ModelFactory
    from evaloop.models.openai_compatible import OpenAICompatibleLLM

    model = ModelFactory().create_model(
        {"type": "openai_compatible", "name": "m", "model_id": "m", "api_key": "sk-test"}
    )
    assert isinstance(model, OpenAICompatibleLLM)
