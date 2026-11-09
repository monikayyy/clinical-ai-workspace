"""Local-only decision-model adapters for retrieval evaluation.

Decision models return typed probabilities rather than generated prose. They are
used only after patient scope has already been enforced. No adapter in this file
allows remote inference or implicit model downloads.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
from typing import Mapping, Protocol


DECISION_MODEL_REVISION = "decision-models-v1"
DECISION_RAG_REVISION = "decision-rag-eval-v1"
DECISION_RESCUE_REVISION = "decision-rescue-eval-v1"


class DecisionModelError(RuntimeError):
    pass


class DecisionBackend(Protocol):
    name: str
    available: bool
    error: str | None

    def choose(self, state_text: str, options: Mapping[str, str], *, instruction: str) -> dict[str, float]: ...


@dataclass(frozen=True)
class LocalDecisionModelConfig:
    model_path: str
    backend: str = "arc"
    device: str = "mps"
    dtype: str = "float16"
    allow_local_model_code: bool = False

    def resolved_path(self) -> Path:
        return Path(self.model_path).expanduser().resolve()


def _tracing_enabled() -> bool:
    truthy = {"1", "true", "yes", "on"}
    return any(
        os.getenv(name, "").strip().casefold() in truthy
        for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")
    )


def _normalise_probabilities(raw: Mapping[str, float], expected: Mapping[str, str]) -> dict[str, float]:
    values = {key: max(0.0, float(raw.get(key, 0.0))) for key in expected}
    total = sum(values.values())
    if total <= 0.0:
        return {key: 1.0 / max(1, len(values)) for key in values}
    return {key: value / total for key, value in values.items()}


class ArcDecideBackend:
    """Local Arc53 decision-model adapter.

    The model repository is prepared separately. The repository's own
    ``arc_decide.py`` client is imported from the *local* model directory, then
    instantiated with that same local path. Clinical runtime never resolves a
    Hugging Face repository name.
    """

    name = "arc_decide_local"

    def __init__(self, config: LocalDecisionModelConfig):
        self.config = config
        self._decider = None
        self.error: str | None = None
        if _tracing_enabled():
            self.error = "external LangChain/LangSmith tracing is enabled"
            return
        path = config.resolved_path()
        if not path.is_dir():
            self.error = f"local decision model directory does not exist: {path}"
            return
        client = path / "arc_decide.py"
        if not client.is_file():
            self.error = f"arc_decide.py not found in local model directory: {path}"
            return
        try:
            # Enforce offline model resolution before importing the local client.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            spec = importlib.util.spec_from_file_location("clinical_workspace_arc_decide", client)
            if spec is None or spec.loader is None:
                raise DecisionModelError("could not load local arc_decide.py")
            module = importlib.util.module_from_spec(spec)
            import sys
            sys.path.insert(0, str(path))
            try:
                spec.loader.exec_module(module)
            finally:
                if sys.path and sys.path[0] == str(path):
                    sys.path.pop(0)
            decider_cls = getattr(module, "Decider")
            self._decider = decider_cls(
                str(path),
                device=config.device,
                dtype=config.dtype,
            )
        except Exception as exc:  # optional model dependency / local model files
            self.error = f"{type(exc).__name__}: {exc}"
            self._decider = None

    @property
    def available(self) -> bool:
        return self._decider is not None

    def choose(self, state_text: str, options: Mapping[str, str], *, instruction: str) -> dict[str, float]:
        if not self._decider:
            raise DecisionModelError(self.error or "Arc decision model unavailable")
        if not 2 <= len(options) <= 16:
            raise DecisionModelError("Arc choice decisions require 2-16 options")
        question = {
            "type": "choice",
            "instructions": instruction,
            "criteria": dict(options),
        }
        result = self._decider.decide({"message": state_text}, {"choice": question})
        answer = result.get("choice") or {}
        probabilities = answer.get("probabilities") or {}
        if isinstance(probabilities, list):
            # Tolerate clients that return [{option, probability}, ...].
            probabilities = {
                str(item.get("option")): float(item.get("probability", 0.0))
                for item in probabilities if isinstance(item, dict)
            }
        if not isinstance(probabilities, dict):
            probabilities = {}
        # Some clients return only the winning choice. Preserve deterministic
        # behavior but mark it as a one-hot distribution.
        if not probabilities and answer.get("choice") in options:
            probabilities = {str(answer["choice"]): 1.0}
        return _normalise_probabilities(probabilities, options)


class D1DecisionBackend:
    """Optional local Liquid d1 adapter.

    d1 currently requires repository-provided model code. For that reason this
    adapter refuses to load unless ``allow_local_model_code`` is explicitly set.
    The model path must be local and ``local_files_only=True`` is mandatory.
    """

    name = "d1_local"

    def __init__(self, config: LocalDecisionModelConfig):
        self.config = config
        self._model = None
        self.error: str | None = None
        if _tracing_enabled():
            self.error = "external LangChain/LangSmith tracing is enabled"
            return
        if not config.allow_local_model_code:
            self.error = "d1 local repository code is disabled; pass allow_local_model_code explicitly"
            return
        path = config.resolved_path()
        if not path.is_dir():
            self.error = f"local decision model directory does not exist: {path}"
            return
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            import torch
            from transformers import AutoModel

            if config.device == "mps" and not torch.backends.mps.is_available():
                device = "cpu"
            else:
                device = config.device
            dtype = torch.float32 if device == "cpu" else (
                torch.float16 if config.dtype == "float16" else torch.bfloat16
            )
            self._model = AutoModel.from_pretrained(
                str(path),
                trust_remote_code=True,
                local_files_only=True,
                dtype=dtype,
            ).to(device)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def choose(self, state_text: str, options: Mapping[str, str], *, instruction: str) -> dict[str, float]:
        if not self._model:
            raise DecisionModelError(self.error or "d1 decision model unavailable")
        if not 2 <= len(options) <= 16:
            raise DecisionModelError("d1 choice decisions require 2-16 options")
        result = self._model.system_one(
            state_text,
            {
                "choice": {
                    "type": "choice",
                    "instructions": instruction,
                    "criteria": dict(options),
                }
            },
        )
        answer = result.get("choice") or {}
        probabilities = answer.get("probabilities") or {}
        if not isinstance(probabilities, dict):
            probabilities = {}
        if not probabilities and answer.get("choice") in options:
            probabilities = {str(answer["choice"]): 1.0}
        return _normalise_probabilities(probabilities, options)


class HeuristicDecisionBackend:
    """Dependency-free test backend.

    It scores an option by lexical overlap between the state and option text.
    This is deliberately *not* a production candidate; it exists to verify
    patient scoping, navigation, ranking, and benchmark accounting.
    """

    name = "heuristic_test_decider"
    available = True
    error = None

    def choose(self, state_text: str, options: Mapping[str, str], *, instruction: str) -> dict[str, float]:
        import re

        q = set(re.findall(r"[a-z0-9]+", state_text.casefold()))
        scores: dict[str, float] = {}
        for key, description in options.items():
            d = set(re.findall(r"[a-z0-9]+", description.casefold()))
            overlap = len(q & d)
            # non-zero floor keeps all branches traversable in tests
            scores[key] = 0.01 + float(overlap)
        return _normalise_probabilities(scores, options)


def load_decision_backend(config: LocalDecisionModelConfig) -> DecisionBackend:
    backend = config.backend.strip().casefold()
    if backend == "arc":
        return ArcDecideBackend(config)
    if backend == "d1":
        return D1DecisionBackend(config)
    raise DecisionModelError(f"unsupported decision backend: {config.backend}")
