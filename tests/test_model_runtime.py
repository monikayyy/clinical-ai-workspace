"""Runtime observability and hard prompt-budget tests for the local model adapter."""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import httpx
import pytest

import app.model as module
from app.model import LocalModel, PromptBudgetExceeded


def content_characters(messages: list[dict]) -> int:
    total = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    total += len(part.get("text") or "")
                elif part.get("type") == "image_url":
                    total += 300
    return total


def synthetic_token_count(messages: list[dict]) -> int:
    return math.ceil(content_characters(messages) / 4) + 8 * len(messages)


class RuntimeServer:
    def __init__(self, n_ctx=4096, token_endpoint=True):
        self.n_ctx = n_ctx
        self.token_endpoint = token_endpoint
        self.requests: list[tuple[str, str, dict | None]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, payload))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{
                "id": "smolvlm", "object": "model", "multimodal": True,
                "capabilities": ["completion", "vision"],
            }]})
        if request.url.path == "/props":
            return httpx.Response(200, json={
                "default_generation_settings": {
                    "n_ctx": self.n_ctx,
                    "params": {
                        "temperature": 0.8, "top_k": 40, "top_p": 0.95,
                        "min_p": 0.05, "repeat_last_n": 64,
                        "repeat_penalty": 1.0, "presence_penalty": 0.0,
                        "frequency_penalty": 0.0, "seed": 4294967295,
                        "samplers": ["penalties", "top_k", "top_p", "min_p", "temperature"],
                    },
                }
            })
        if request.url.path == "/v1/chat/completions/input_tokens":
            if not self.token_endpoint:
                return httpx.Response(404, json={"error": {"message": "not supported"}})
            return httpx.Response(200, json={
                "object": "response.input_tokens",
                "input_tokens": synthetic_token_count(payload["messages"]),
            })
        if request.url.path == "/v1/chat/completions":
            prompt_tokens = synthetic_token_count(payload["messages"])
            # This assertion validates the acceptance criterion at the actual
            # inference boundary, not just in a helper method.
            max_allowed = min(math.floor(self.n_ctx * 0.80), self.n_ctx - payload["max_tokens"] - 64)
            assert prompt_tokens <= max_allowed
            return httpx.Response(200, json={
                "id": "chatcmpl-runtime-test",
                "object": "chat.completion",
                "created": 0,
                "model": "smolvlm",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "A concise grounded answer."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 6, "total_tokens": prompt_tokens + 6},
                "timings": {"prompt_n": prompt_tokens, "prompt_ms": 20.0, "prompt_per_second": 500.0, "predicted_n": 6, "predicted_ms": 30.0, "predicted_per_second": 200.0},
            })
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")



class FakeCompletions:
    def __init__(self, server: RuntimeServer):
        self.server = server

    def create(self, *, model, messages, temperature, max_tokens, extra_body):
        payload = {
            "model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, **extra_body,
        }
        self.server.requests.append(("POST", "/v1/chat/completions", payload))
        prompt_tokens = synthetic_token_count(messages)
        max_allowed = min(
            math.floor(self.server.n_ctx * 0.80),
            self.server.n_ctx - max_tokens - 64,
        )
        assert prompt_tokens <= max_allowed
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="A concise grounded answer."))],
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens, completion_tokens=6, total_tokens=prompt_tokens + 6
            ),
            model_extra={
                "timings": {
                    "prompt_n": prompt_tokens, "prompt_ms": 20.0,
                    "prompt_per_second": 500.0, "predicted_n": 6,
                    "predicted_ms": 30.0, "predicted_per_second": 200.0,
                }
            },
        )


class FakeOpenAI:
    def __init__(self, server: RuntimeServer):
        self.chat = SimpleNamespace(completions=FakeCompletions(server))

    def close(self):
        pass

def model_for(server: RuntimeServer, monkeypatch) -> LocalModel:
    monkeypatch.setenv("MODEL_PROMPT_FRACTION", "0.80")
    monkeypatch.setenv("MODEL_TOKEN_SAFETY_MARGIN", "64")
    monkeypatch.setenv("MODEL_CONTEXT_FALLBACK", "4096")
    monkeypatch.setenv("MODEL_TEMPERATURE", "0")
    monkeypatch.setenv("MODEL_TOP_K", "1")
    monkeypatch.setenv("MODEL_TOP_P", "1")
    monkeypatch.setenv("MODEL_MIN_P", "0")
    monkeypatch.setenv("MODEL_REPEAT_PENALTY", "1.05")
    monkeypatch.setenv("MODEL_SEED", "42")
    client = httpx.Client(transport=httpx.MockTransport(server.handler), trust_env=False)
    return LocalModel(http_client=client, openai_client=FakeOpenAI(server))


def test_runtime_info_reports_actual_context_and_sampler_defaults(monkeypatch):
    server = RuntimeServer(n_ctx=8192)
    model = model_for(server, monkeypatch)
    try:
        info = model.runtime_info(force=True)
        assert info["available"] is True
        assert info["context_tokens"] == 8192
        assert info["context_source"] == "llama_cpp_props"
        assert info["server_defaults"]["top_k"] == 40
        assert info["application_generation"] == {
            "temperature": 0.0, "top_k": 1, "top_p": 1.0,
            "min_p": 0.0, "repeat_penalty": 1.05, "seed": 42, "answer_max_tokens": 320,
        }
        assert info["loaded_models"][0]["id"] == "smolvlm"
        status = model.status()
        assert status["available"] is True
        assert status["runtime"]["context_tokens"] == 8192
    finally:
        model.close()


def test_long_assistant_prompt_is_trimmed_before_inference(monkeypatch):
    server = RuntimeServer(n_ctx=4096)
    model = model_for(server, monkeypatch)
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": (f"turn {index} " + "conversation " * 150)}
        for index in range(8)
    ]
    context = "\n\n".join(
        f"[R{index+1}] Relevant reviewed record {index+1}\n" + ("clinical evidence sentence. " * 100)
        for index in range(4)
    )
    try:
        answer = model.answer_with_history("Please explain the documented situation.", context, history)
        assert answer == "A concise grounded answer."
        metrics = model.last_request_metrics()
        assert metrics is not None
        assert metrics["within_budget"] is True
        assert metrics["prompt_tokens"] <= metrics["prompt_limit_tokens"]
        assert metrics["prompt_fraction"] <= 0.80
        assert metrics["history_turns_dropped"] > 0 or metrics["source_blocks_dropped"] > 0 or metrics["source_characters_dropped"] > 0
        assert metrics["headroom_tokens"] >= metrics["safety_margin_tokens"]
        chat_request = next(payload for method, path, payload in server.requests if path == "/v1/chat/completions")
        assert chat_request["temperature"] == 0.0
        assert chat_request["top_k"] == 1
        assert chat_request["top_p"] == 1.0
        assert chat_request["min_p"] == 0.0
        assert chat_request["repeat_penalty"] == 1.05
        assert chat_request["seed"] == 42
    finally:
        model.close()


def test_token_endpoint_failure_uses_conservative_estimate(monkeypatch):
    server = RuntimeServer(n_ctx=4096, token_endpoint=False)
    model = model_for(server, monkeypatch)
    try:
        messages = [{"role": "system", "content": "instructions"}, {"role": "user", "content": "synthetic input" * 20}]
        inspection = model.inspect_request(messages, 100)
        assert inspection["count_method"] == "conservative_estimate"
        assert inspection["prompt_tokens"] > 0
        assert inspection["within_budget"] is True
    finally:
        model.close()


def test_irreducible_prompt_is_rejected_before_chat_completion(monkeypatch):
    server = RuntimeServer(n_ctx=900)
    model = model_for(server, monkeypatch)
    try:
        with pytest.raises(PromptBudgetExceeded, match="could not fit"):
            model.answer_with_history(
                "Q" * 1500,
                "[R1] reviewed record\n" + "evidence " * 300,
                [],
            )
        assert not any(path == "/v1/chat/completions" for _, path, _ in server.requests)
        metrics = model.last_request_metrics()
        assert metrics["rejected_before_inference"] is True
        assert metrics["within_budget"] is False
    finally:
        model.close()


def test_summary_is_shortened_to_budget(monkeypatch):
    server = RuntimeServer(n_ctx=1800)
    model = model_for(server, monkeypatch)
    try:
        answer = model.summarize("synthetic reviewed content. " * 1000)
        assert answer == "A concise grounded answer."
        metrics = model.last_request_metrics()
        assert metrics["within_budget"] is True
        assert metrics["source_characters_dropped"] > 0
    finally:
        model.close()


def test_image_request_budget_uses_token_counter(monkeypatch):
    server = RuntimeServer(n_ctx=4096)
    model = model_for(server, monkeypatch)
    try:
        result = model.complete("Transcribe visible text.", "Synthetic image.", b"png-bytes", max_tokens=100)
        assert result == "A concise grounded answer."
        metrics = model.last_request_metrics()
        assert metrics["count_method"] == "llama_cpp_input_tokens"
        assert metrics["within_budget"] is True
    finally:
        model.close()


