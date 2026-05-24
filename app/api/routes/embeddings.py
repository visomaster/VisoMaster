"""
GET    /api/embeddings
POST   /api/embeddings/merge
GET    /api/embeddings/export
POST   /api/embeddings/import
DELETE /api/embeddings/{embedding_id}
POST   /api/embeddings/clear
"""
from __future__ import annotations

import json
import uuid
from typing import List

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from app.api.deps import get_app_state
from app.api.schemas import (
    EmbeddingCard,
    EmbeddingListResponse,
    MergeEmbeddingRequest,
    OkResponse,
)
from app.core.state import AppState, EmbeddingStore, MergedEmbedding

router = APIRouter(prefix="/api/embeddings", tags=["embeddings"])


@router.get("", response_model=EmbeddingListResponse)
def list_embeddings(state: AppState = Depends(get_app_state)):
    return EmbeddingListResponse(
        embeddings=[
            EmbeddingCard(embedding_id=eid, name=e.name)
            for eid, e in state.embeddings.items()
        ]
    )


@router.post("/merge", response_model=EmbeddingCard)
def merge_embeddings(
    body: MergeEmbeddingRequest,
    state: AppState = Depends(get_app_state),
):
    """
    Average the embeddings of the listed input faces into a single merged embedding.
    """
    if not body.input_face_ids:
        raise HTTPException(status_code=400, detail="input_face_ids must not be empty")

    merge_method = state.control.get("EmbMergeMethodSelection", "Mean")
    stores: List[EmbeddingStore] = []
    for fid in body.input_face_ids:
        face = state.input_faces.get(fid)
        if face is None:
            raise HTTPException(status_code=404, detail=f"Input face '{fid}' not found")
        stores.append(face.embedding_store)

    all_models: set[str] = set()
    for s in stores:
        all_models.update(s.store.keys())

    merged: dict[str, np.ndarray] = {}
    for model in all_models:
        vecs = [s.store[model] for s in stores if model in s.store]
        if not vecs:
            continue
        stacked = np.stack(vecs, axis=0)
        if merge_method == "Median":
            merged[model] = np.median(stacked, axis=0).astype(np.float32)
        else:
            merged[model] = np.mean(stacked, axis=0).astype(np.float32)

    embedding_id = str(uuid.uuid1().int)
    state.embeddings[embedding_id] = MergedEmbedding(
        embedding_id=embedding_id,
        name=body.name,
        embedding_store=EmbeddingStore(store=merged),
    )
    return EmbeddingCard(embedding_id=embedding_id, name=body.name)


@router.get("/export")
def export_embeddings(state: AppState = Depends(get_app_state)):
    """Download all merged embeddings as a JSON file."""
    data = [
        {
            "name": e.name,
            "embedding_store": e.embedding_store.to_json(),
        }
        for e in state.embeddings.values()
    ]
    return JSONResponse(
        content=data,
        headers={"Content-Disposition": 'attachment; filename="embeddings.json"'},
    )


@router.post("/import", response_model=EmbeddingListResponse)
async def import_embeddings(
    file: UploadFile = File(...),
    state: AppState = Depends(get_app_state),
):
    """Upload a previously exported embeddings JSON file."""
    try:
        content = await file.read()
        data = json.loads(content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="Expected a JSON array")

    imported: List[EmbeddingCard] = []
    for item in data:
        embedding_id = str(uuid.uuid1().int)
        state.embeddings[embedding_id] = MergedEmbedding(
            embedding_id=embedding_id,
            name=item.get("name", "Imported"),
            embedding_store=EmbeddingStore.from_json(item.get("embedding_store", {})),
        )
        imported.append(EmbeddingCard(embedding_id=embedding_id, name=item.get("name", "Imported")))

    return EmbeddingListResponse(embeddings=imported)


@router.delete("/{embedding_id}", response_model=OkResponse)
def delete_embedding(embedding_id: str, state: AppState = Depends(get_app_state)):
    if embedding_id not in state.embeddings:
        raise HTTPException(status_code=404, detail=f"Embedding '{embedding_id}' not found")
    del state.embeddings[embedding_id]
    # Remove from any target face assignments
    for tf in state.target_faces.values():
        if embedding_id in tf.assigned_embedding_ids:
            tf.assigned_embedding_ids.remove(embedding_id)
    return OkResponse(message=f"Removed embedding {embedding_id}")


@router.post("/clear", response_model=OkResponse)
def clear_embeddings(state: AppState = Depends(get_app_state)):
    state.embeddings.clear()
    for tf in state.target_faces.values():
        tf.assigned_embedding_ids.clear()
    return OkResponse(message="All embeddings cleared")
