"""
app/core/state.py
─────────────────
Central application state that lives independently of any UI framework.
The Qt MainWindow and the FastAPI server both hold a reference to the same
AppState instance; they never share state through any other channel.
"""
from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from app.helpers.miscellaneous import ParametersDict


# ── Embedded face data ────────────────────────────────────────────────────────

@dataclass
class EmbeddingStore:
    """Per-recognition-model 512-dim ArcFace vectors for one face."""
    store: Dict[str, np.ndarray] = field(default_factory=dict)

    def to_json(self) -> Dict[str, List[float]]:
        return {k: v.tolist() for k, v in self.store.items()}

    @classmethod
    def from_json(cls, d: Dict[str, List[float]]) -> "EmbeddingStore":
        return cls(store={k: np.array(v, dtype=np.float32) for k, v in d.items()})


@dataclass
class TargetFace:
    face_id: str
    cropped_face: np.ndarray                    # uint8 BGR, ~82×82
    embedding_store: EmbeddingStore             # embeddings for all recognition models
    assigned_input_face_ids: List[str] = field(default_factory=list)
    assigned_embedding_ids: List[str] = field(default_factory=list)
    # Merged embedding used as swap source (computed from assigned faces/embeddings)
    assigned_input_embedding: EmbeddingStore = field(default_factory=EmbeddingStore)

    def to_json(self) -> dict:
        return {
            "face_id": self.face_id,
            "cropped_face": self.cropped_face.tolist(),
            "embedding_store": self.embedding_store.to_json(),
            "assigned_input_face_ids": self.assigned_input_face_ids,
            "assigned_embedding_ids": self.assigned_embedding_ids,
            "assigned_input_embedding": self.assigned_input_embedding.to_json(),
        }

    @classmethod
    def from_json(cls, d: dict) -> "TargetFace":
        return cls(
            face_id=d["face_id"],
            cropped_face=np.array(d["cropped_face"], dtype=np.uint8),
            embedding_store=EmbeddingStore.from_json(d["embedding_store"]),
            assigned_input_face_ids=d.get("assigned_input_face_ids", []),
            assigned_embedding_ids=d.get("assigned_embedding_ids", []),
            assigned_input_embedding=EmbeddingStore.from_json(
                d.get("assigned_input_embedding", {})
            ),
        )


@dataclass
class InputFace:
    face_id: str
    media_path: str
    embedding_store: EmbeddingStore = field(default_factory=EmbeddingStore)
    # Cropped thumbnail (optional, not always loaded)
    cropped_face: Optional[np.ndarray] = None

    def to_json(self) -> dict:
        return {
            "face_id": self.face_id,
            "media_path": self.media_path,
            "embedding_store": self.embedding_store.to_json(),
        }

    @classmethod
    def from_json(cls, d: dict) -> "InputFace":
        return cls(
            face_id=d["face_id"],
            media_path=d["media_path"],
            embedding_store=EmbeddingStore.from_json(d.get("embedding_store", {})),
        )


@dataclass
class MergedEmbedding:
    embedding_id: str
    name: str
    embedding_store: EmbeddingStore = field(default_factory=EmbeddingStore)

    def to_json(self) -> dict:
        return {
            "embedding_id": self.embedding_id,
            "name": self.name,
            "embedding_store": self.embedding_store.to_json(),
        }

    @classmethod
    def from_json(cls, d: dict) -> "MergedEmbedding":
        return cls(
            embedding_id=d["embedding_id"],
            name=d.get("name") or d.get("embedding_name", ""),
            embedding_store=EmbeddingStore.from_json(d.get("embedding_store", {})),
        )


@dataclass
class MediaRef:
    media_id: str
    media_path: str
    file_type: str  # 'video' | 'image' | 'webcam' | 'webrtc'

    def to_json(self) -> dict:
        return {"media_id": self.media_id, "media_path": self.media_path, "file_type": self.file_type}

    @classmethod
    def from_json(cls, d: dict) -> "MediaRef":
        return cls(
            media_id=d["media_id"],
            media_path=d["media_path"],
            file_type=d.get("file_type", "video"),
        )


@dataclass
class Marker:
    frame_number: int
    parameters: Dict[str, Any]   # face_id → param dict
    control: Dict[str, Any]

    def to_json(self) -> dict:
        return {
            "frame_number": self.frame_number,
            "parameters": self.parameters,
            "control": self.control,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Marker":
        return cls(
            frame_number=int(d["frame_number"]),
            parameters=d.get("parameters", {}),
            control=d.get("control", {}),
        )


# ── Streaming transform state ─────────────────────────────────────────────────

@dataclass
class StreamTransform:
    rotation: int = 0       # 0 | 90 | 180 | 270 (clockwise)
    flip_h: bool = False
    flip_v: bool = False

    def to_json(self) -> dict:
        return {"rotation": self.rotation, "flip_h": self.flip_h, "flip_v": self.flip_v}

    @classmethod
    def from_json(cls, d: dict) -> "StreamTransform":
        return cls(
            rotation=d.get("rotation", 0),
            flip_h=d.get("flip_h", False),
            flip_v=d.get("flip_v", False),
        )


# ── Top-level AppState ────────────────────────────────────────────────────────

@dataclass
class AppState:
    """
    Single source of truth for all session data.

    Both the Qt MainWindow and the FastAPI server hold a reference to the
    same instance.  The Qt layer reads/writes it directly; the API layer
    reads/writes it through route handlers.  No other shared state exists.
    """

    # ── Global settings (mirrors main_window.control) ─────────────────────
    control: Dict[str, Any] = field(default_factory=dict)

    # ── Per-face parameters (mirrors main_window.parameters) ──────────────
    # face_id → ParametersDict (falls back to default_parameters on missing keys)
    parameters: Dict[str, Any] = field(default_factory=dict)
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    current_widget_parameters: Dict[str, Any] = field(default_factory=dict)

    # ── Working set ────────────────────────────────────────────────────────
    target_media: Dict[str, MediaRef] = field(default_factory=dict)
    target_faces: Dict[str, TargetFace] = field(default_factory=dict)
    input_faces: Dict[str, InputFace] = field(default_factory=dict)
    embeddings: Dict[str, MergedEmbedding] = field(default_factory=dict)

    # ── Selection ──────────────────────────────────────────────────────────
    selected_media_id: Optional[str] = None
    selected_face_id: Optional[str] = None

    # ── Markers ────────────────────────────────────────────────────────────
    markers: Dict[int, Marker] = field(default_factory=dict)

    # ── Streaming transforms ───────────────────────────────────────────────
    webcam_transform: StreamTransform = field(default_factory=StreamTransform)
    webrtc_transform: StreamTransform = field(default_factory=StreamTransform)
    media_transform:  StreamTransform = field(default_factory=StreamTransform)

    # ── Folder memory ──────────────────────────────────────────────────────
    last_target_media_folder: str = ""
    last_input_media_folder: str = ""
    loaded_embedding_filename: str = ""
    output_media_folder: str = ""

    # ── Playback options ───────────────────────────────────────────────────
    loop_enabled: bool = False

    # ── Helpers ────────────────────────────────────────────────────────────

    def new_face_id(self) -> str:
        return str(uuid.uuid1().int)

    def get_parameters(self, face_id: str) -> ParametersDict:
        """Return a ParametersDict for the given face, creating it if absent."""
        if face_id not in self.parameters:
            self.parameters[face_id] = ParametersDict(
                copy.deepcopy(self.default_parameters), self.default_parameters
            )
        raw = self.parameters[face_id]
        if isinstance(raw, ParametersDict):
            return raw
        return ParametersDict(raw, self.default_parameters)

    def set_parameter(self, face_id: str, name: str, value: Any) -> None:
        params = self.get_parameters(face_id)
        params[name] = value
        self.parameters[face_id] = params

    def set_control(self, name: str, value: Any) -> None:
        self.control[name] = value
        # Keep the dedicated dataclass field in sync with the matching
        # control entry so both code paths (the Qt VP, which reads
        # state.control, and the headless play loop, which reads
        # state.loop_enabled) see the same value.
        if name == "loop_enabled":
            self.loop_enabled = bool(value)

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_json(self) -> dict:
        """Produce a dict compatible with last_workspace.json."""
        target_faces_data = {}
        for fid, tf in self.target_faces.items():
            raw_params = self.parameters.get(fid, {})
            if isinstance(raw_params, ParametersDict):
                raw_params = raw_params.data
            target_faces_data[fid] = {
                **tf.to_json(),
                "parameters": raw_params,
                "control": self.control.copy(),
            }

        markers_data = {}
        for pos, m in self.markers.items():
            markers_data[str(pos)] = m.to_json()

        return {
            "selected_media_id": self.selected_media_id,
            "target_medias_data": [m.to_json() for m in self.target_media.values()],
            "target_faces_data": target_faces_data,
            "input_faces_data": {
                fid: f.to_json() for fid, f in self.input_faces.items()
            },
            "embeddings_data": {
                eid: e.to_json() for eid, e in self.embeddings.items()
            },
            "markers": markers_data,
            "control": self.control,
            "current_widget_parameters": (
                self.current_widget_parameters.data
                if isinstance(self.current_widget_parameters, ParametersDict)
                else self.current_widget_parameters
            ),
            "last_target_media_folder_path": self.last_target_media_folder,
            "last_input_media_folder_path": self.last_input_media_folder,
            "loaded_embedding_filename": self.loaded_embedding_filename,
            "webcam_transform": self.webcam_transform.to_json(),
            "webrtc_transform": self.webrtc_transform.to_json(),
        }

    @classmethod
    def from_json(cls, d: dict, default_parameters: dict) -> "AppState":
        """Reconstruct from a last_workspace.json dict."""
        state = cls(
            control=d.get("control", {}),
            default_parameters=default_parameters,
            selected_media_id=d.get("selected_media_id"),
            last_target_media_folder=d.get("last_target_media_folder_path", ""),
            last_input_media_folder=d.get("last_input_media_folder_path", ""),
            loaded_embedding_filename=d.get("loaded_embedding_filename", ""),
        )

        # Target media
        for m in d.get("target_medias_data", []):
            ref = MediaRef.from_json(m)
            state.target_media[ref.media_id] = ref

        # Input faces
        for fid, fd in d.get("input_faces_data", {}).items():
            state.input_faces[fid] = InputFace.from_json({**fd, "face_id": fid})

        # Embeddings
        for eid, ed in d.get("embeddings_data", {}).items():
            state.embeddings[eid] = MergedEmbedding.from_json({**ed, "embedding_id": eid})

        # Target faces + parameters
        for fid, tfd in d.get("target_faces_data", {}).items():
            state.target_faces[fid] = TargetFace.from_json({**tfd, "face_id": fid})
            raw_params = tfd.get("parameters", {})
            state.parameters[fid] = ParametersDict(raw_params, default_parameters)

        # Markers
        for pos_str, md in d.get("markers", {}).items():
            pos = int(pos_str)
            state.markers[pos] = Marker.from_json({**md, "frame_number": pos})

        # Current widget parameters
        cwp = d.get("current_widget_parameters", {})
        state.current_widget_parameters = ParametersDict(cwp, default_parameters)

        # Streaming transforms
        if "webcam_transform" in d:
            state.webcam_transform = StreamTransform.from_json(d["webcam_transform"])
        if "webrtc_transform" in d:
            state.webrtc_transform = StreamTransform.from_json(d["webrtc_transform"])

        return state
