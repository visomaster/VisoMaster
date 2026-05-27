"""
app/api/schemas.py
──────────────────
Pydantic models for all API request/response bodies.
Run `python -m app.api.schemas` to dump the OpenAPI JSON for codegen.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ── Primitives ────────────────────────────────────────────────────────────────

class OkResponse(BaseModel):
    ok: bool = True
    message: str = ""


class ErrorResponse(BaseModel):
    ok: bool = False
    error: str
    detail: Optional[str] = None


# ── System ────────────────────────────────────────────────────────────────────

class GpuInfo(BaseModel):
    index: int
    name: str
    total_mb: int
    free_mb: int
    used_mb: int


class SystemInfoResponse(BaseModel):
    platform: str
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    ort_version: str
    ort_providers: List[str]
    trt_available: bool
    trt_version: Optional[str]
    ffmpeg_available: bool
    gpus: List[GpuInfo]


class GpuMemoryResponse(BaseModel):
    used_mb: int
    total_mb: int


class ProviderSwitchRequest(BaseModel):
    provider: Literal["CUDA", "TensorRT", "TensorRT-Engine", "CPU"]


class ProviderResponse(BaseModel):
    active_provider: str


# ── Schema (layout-data) ──────────────────────────────────────────────────────

class WidgetDescriptor(BaseModel):
    """Mirrors one entry in *_layout_data.py (exec_function stripped out)."""
    widget_name: str
    section: str
    level: int = 1
    label: str = ""
    widget_type: Literal["toggle", "slider", "decimal_slider", "selection", "text"] = "toggle"
    default: Any = None
    options: Optional[List[str]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    step: Optional[Union[int, float]] = None
    decimals: Optional[int] = None
    help: str = ""
    parent_toggle: Optional[str] = None
    required_toggle_value: Optional[bool] = None
    parent_selection: Optional[str] = None
    required_selection_value: Optional[str] = None
    width: Optional[int] = None


class SchemaResponse(BaseModel):
    widgets: List[WidgetDescriptor]


# ── State ─────────────────────────────────────────────────────────────────────

class ControlPatchRequest(BaseModel):
    """Patch one or more control values."""
    updates: Dict[str, Any]


class ParameterPatchRequest(BaseModel):
    """Patch one or more per-face parameter values."""
    updates: Dict[str, Any]


class StateResponse(BaseModel):
    """Snapshot of the full AppState (JSON-serialisable subset)."""
    selected_media_id: Optional[str]
    selected_face_id: Optional[str]
    control: Dict[str, Any]
    target_media: List[Dict[str, Any]]
    target_faces: Dict[str, Any]
    input_faces: Dict[str, Any]
    embeddings: Dict[str, Any]
    markers: Dict[str, Any]
    webcam_transform: Dict[str, Any]
    webrtc_transform: Dict[str, Any]
    last_target_media_folder: str
    last_input_media_folder: str
    output_media_folder: str


# ── Target media ──────────────────────────────────────────────────────────────

class MediaCard(BaseModel):
    media_id: str
    media_path: str
    file_type: str
    thumbnail_url: str


class ScanFolderRequest(BaseModel):
    path: str
    recursive: bool = False


class ScanFolderResponse(BaseModel):
    items: List[MediaCard]


class AddFilesRequest(BaseModel):
    paths: List[str]


class SelectMediaRequest(BaseModel):
    media_id: str


# ── Target faces ──────────────────────────────────────────────────────────────

class FaceCard(BaseModel):
    face_id: str
    thumbnail_url: str
    assigned_input_face_ids: List[str] = []
    assigned_embedding_ids: List[str] = []


class FindFacesResponse(BaseModel):
    found: int
    faces: List[FaceCard]


class AssignFaceRequest(BaseModel):
    input_face_id: str


class AssignEmbeddingRequest(BaseModel):
    embedding_id: str


# ── Input faces ───────────────────────────────────────────────────────────────

class InputFaceCard(BaseModel):
    face_id: str
    media_path: str
    thumbnail_url: str


class ScanInputFolderResponse(BaseModel):
    items: List[InputFaceCard]


# ── Embeddings ────────────────────────────────────────────────────────────────

class EmbeddingCard(BaseModel):
    embedding_id: str
    name: str


class MergeEmbeddingRequest(BaseModel):
    name: str
    input_face_ids: List[str]


class EmbeddingListResponse(BaseModel):
    embeddings: List[EmbeddingCard]


# ── Playback ──────────────────────────────────────────────────────────────────

class PlaybackState(BaseModel):
    file_type: Optional[str]
    fps: float
    current_frame: int
    max_frame: int
    is_playing: bool
    is_recording: bool
    swap_enabled: bool
    edit_enabled: bool
    loop_enabled: bool = False


class SeekRequest(BaseModel):
    frame: int


class StepRequest(BaseModel):
    n: int = 30   # positive = forward, negative = rewind


class RecordStartRequest(BaseModel):
    output_folder: Optional[str] = None


class RecordStopResponse(BaseModel):
    output_path: str


# ── Markers ───────────────────────────────────────────────────────────────────

class MarkerInfo(BaseModel):
    frame_number: int


class MarkerListResponse(BaseModel):
    markers: List[int]


# ── Streaming sources ─────────────────────────────────────────────────────────

class WebcamInfo(BaseModel):
    index: int
    label: str
    thumbnail_url: Optional[str] = None


class WebcamListResponse(BaseModel):
    webcams: List[WebcamInfo]


class WebRTCStartResponse(BaseModel):
    http_url: str
    https_url: str
    ws_url: str
    wss_url: str


class StreamTransformRequest(BaseModel):
    rotation: Literal[0, 90, 180, 270] = 0
    flip_h: bool = False
    flip_v: bool = False


# ── Workspace ─────────────────────────────────────────────────────────────────

class WorkspaceSaveRequest(BaseModel):
    filename: str


class WorkspaceLoadRequest(BaseModel):
    filename: str


# ── WebSocket events (server → client) ───────────────────────────────────────

class WsEvent(BaseModel):
    type: str
    payload: Dict[str, Any] = {}


# ── Folder browser ───────────────────────────────────────────────────────────

class FolderEntry(BaseModel):
    name: str
    path: str
    is_dir: bool


class BrowseFolderResponse(BaseModel):
    path: str          # resolved absolute path that was listed
    parent: Optional[str]  # parent directory, or None if at root
    entries: List[FolderEntry]


class QuickFolder(BaseModel):
    label: str
    path: str


class QuickFoldersResponse(BaseModel):
    folders: List[QuickFolder]


# ── Preview ───────────────────────────────────────────────────────────────────

class PreviewSnapshotResponse(BaseModel):
    """URL to the latest processed frame as JPEG."""
    url: str
    frame_number: int
    width: int
    height: int


# ── Models ────────────────────────────────────────────────────────────────────

class LoadedModelInfo(BaseModel):
    name: str
    store: Literal["onnx", "trt", "dfm"]
    device: str
    vram_mb: int = 0


class LoadedModelsResponse(BaseModel):
    models: List[LoadedModelInfo]
