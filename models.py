from typing import Any

from pydantic import BaseModel

from config import CODEX_MODELS


MODEL_METADATA: dict[str, dict[str, Any]] = {
    "gpt-5.5": {
        "context_window": 1_000_000,
        "max_output_tokens": 128_000,
        "reasoning": True,
        "input_modalities": ["text", "image"],
    },
    "gpt-5.4": {
        "context_window": 1_050_000,
        "max_output_tokens": 128_000,
        "reasoning": True,
        "input_modalities": ["text", "image"],
    },
    "gpt-5.4-mini": {
        "context_window": 400_000,
        "max_output_tokens": 128_000,
        "reasoning": True,
        "input_modalities": ["text", "image"],
    },
    "gpt-5.3-codex": {
        "context_window": 272_000,
        "max_output_tokens": 128_000,
        "reasoning": True,
        "input_modalities": ["text", "image"],
    },
    "gpt-5.3-codex-spark": {
        "context_window": 128_000,
        "max_output_tokens": 128_000,
        "reasoning": True,
        "input_modalities": ["text"],
    },
    "gpt-5.2": {
        "context_window": 1_050_000,
        "max_output_tokens": 128_000,
        "reasoning": True,
        "input_modalities": ["text", "image"],
    },
    "gpt-5.2-codex": {
        "context_window": 272_000,
        "max_output_tokens": 128_000,
        "reasoning": True,
        "input_modalities": ["text", "image"],
    },
    "gpt-5.1-codex": {
        "max_output_tokens": 128_000,
        "reasoning": True,
        "input_modalities": ["text", "image"],
    },
}


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 1700000000
    owned_by: str = "openai"
    permission: list[Any] = []
    root: str = ""
    parent: str | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    reasoning: bool | None = None
    input_modalities: list[str] | None = None


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


def _build_model_info(model_id: str) -> ModelInfo:
    return ModelInfo(id=model_id, root=model_id, **MODEL_METADATA.get(model_id, {}))


def get_model_list() -> ModelListResponse:
    models = [_build_model_info(model_id) for model_id in CODEX_MODELS]
    return ModelListResponse(data=models)


def get_model_info(model_id: str) -> ModelInfo | None:
    if model_id in CODEX_MODELS:
        return _build_model_info(model_id)
    return None
