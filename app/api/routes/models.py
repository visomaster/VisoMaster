"""
GET    /api/models          — list all currently loaded models
DELETE /api/models/{name}   — unload a specific model by name
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_models_processor
from app.api.schemas import LoadedModelInfo, LoadedModelsResponse, OkResponse

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=LoadedModelsResponse)
def list_loaded_models(mp=Depends(get_models_processor)):
    """Return every model that is currently loaded in memory."""
    entries = mp.get_loaded_models()
    return LoadedModelsResponse(
        models=[LoadedModelInfo(**e) for e in entries]
    )


@router.delete("/{model_name:path}", response_model=OkResponse)
def unload_model(model_name: str, mp=Depends(get_models_processor)):
    """Unload a single model from memory by its registered name.

    The ``model_name`` path segment may contain slashes (e.g. DFM model
    paths), so ``:path`` is used to capture the full name.
    """
    unloaded = mp.unload_model(model_name)
    if not unloaded:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' is not currently loaded.",
        )
    return OkResponse(message=f"Model '{model_name}' unloaded.")
