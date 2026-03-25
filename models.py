import time
from typing import Any

from pydantic import BaseModel

from config import CODEX_MODELS


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 1700000000
    owned_by: str = "openai"
    permission: list[Any] = []
    root: str = ""
    parent: str | None = None


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


def get_model_list() -> ModelListResponse:
    models = [
        ModelInfo(id=model_id, root=model_id)
        for model_id in CODEX_MODELS
    ]
    return ModelListResponse(data=models)


def get_model_info(model_id: str) -> ModelInfo | None:
    if model_id in CODEX_MODELS:
        return ModelInfo(id=model_id, root=model_id)
    return None
