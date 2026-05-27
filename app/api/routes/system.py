"""
GET  /api/system/info
GET  /api/system/gpu-memory
POST /api/system/clear-memory
GET  /api/system/providers
POST /api/system/providers
GET  /api/system/browse-folder
GET  /api/system/quick-folders
"""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

import torch
import onnxruntime

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_models_processor
from app.api.schemas import (
    BrowseFolderResponse,
    FolderEntry,
    GpuInfo,
    GpuMemoryResponse,
    OkResponse,
    ProviderResponse,
    ProviderSwitchRequest,
    QuickFolder,
    QuickFoldersResponse,
    SystemInfoResponse,
)
from app.helpers.miscellaneous import is_ffmpeg_in_path

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/info", response_model=SystemInfoResponse)
def system_info(mp=Depends(get_models_processor)):
    """Return hardware / runtime information."""
    gpus: list[GpuInfo] = []
    if torch.cuda.is_available():
        try:
            import subprocess as sp
            total_out = sp.check_output(
                "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits".split()
            ).decode().strip().splitlines()
            free_out = sp.check_output(
                "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits".split()
            ).decode().strip().splitlines()
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total = int(total_out[i]) if i < len(total_out) else props.total_memory // (1024 ** 2)
                free = int(free_out[i]) if i < len(free_out) else 0
                gpus.append(GpuInfo(
                    index=i,
                    name=props.name,
                    total_mb=total,
                    free_mb=free,
                    used_mb=total - free,
                ))
        except Exception:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append(GpuInfo(
                    index=i,
                    name=props.name,
                    total_mb=props.total_memory // (1024 ** 2),
                    free_mb=0,
                    used_mb=0,
                ))

    trt_version: str | None = None
    trt_available = False
    try:
        import tensorrt as trt  # type: ignore
        trt_version = trt.__version__
        trt_available = True
    except ModuleNotFoundError:
        pass

    return SystemInfoResponse(
        platform=platform.platform(),
        python_version=sys.version,
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
        ort_version=onnxruntime.__version__,
        ort_providers=onnxruntime.get_available_providers(),
        trt_available=trt_available,
        trt_version=trt_version,
        ffmpeg_available=is_ffmpeg_in_path(),
        gpus=gpus,
    )


@router.get("/gpu-memory", response_model=GpuMemoryResponse)
def gpu_memory(mp=Depends(get_models_processor)):
    """Return current GPU memory usage (primary GPU)."""
    try:
        used, total = mp.get_gpu_memory()
        return GpuMemoryResponse(used_mb=used, total_mb=total)
    except Exception:
        return GpuMemoryResponse(used_mb=0, total_mb=0)


@router.post("/clear-memory", response_model=OkResponse)
def clear_memory(mp=Depends(get_models_processor)):
    """Unload all models and flush GPU memory."""
    mp.clear_gpu_memory()
    return OkResponse(message="GPU memory cleared")


@router.get("/providers", response_model=ProviderResponse)
def get_providers(mp=Depends(get_models_processor)):
    """Return the currently active execution provider."""
    return ProviderResponse(active_provider=mp.provider_name)


@router.post("/providers", response_model=ProviderResponse)
def set_providers(body: ProviderSwitchRequest, mp=Depends(get_models_processor)):
    """Switch the active execution provider and reload all models."""
    mp.switch_providers_priority(body.provider)
    mp.delete_models()
    return ProviderResponse(active_provider=mp.provider_name)


@router.get("/browse-folder", response_model=BrowseFolderResponse)
def browse_folder(path: str = Query(default=""), show_files: bool = Query(default=False)):
    """
    List the contents of a directory for the folder-browser UI.

    - `path`       — absolute path to list; defaults to the user home directory.
    - `show_files` — when True, include non-directory entries as well.

    Returns the resolved path, its parent (None at filesystem root), and a
    sorted list of entries (directories first, then files if requested).
    """
    # Default to home directory when no path is given
    target = Path(path).resolve() if path.strip() else Path.home()

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {target}")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {target}")

    # Determine parent — None when we're at a filesystem root
    parent_path: str | None = None
    parent = target.parent
    if parent != target:  # at root, parent == self
        parent_path = str(parent)

    entries: list[FolderEntry] = []
    try:
        for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if item.name.startswith("."):
                continue  # skip hidden entries
            if item.is_dir():
                entries.append(FolderEntry(name=item.name, path=str(item), is_dir=True))
            elif show_files and item.is_file():
                entries.append(FolderEntry(name=item.name, path=str(item), is_dir=False))
    except PermissionError:
        pass  # return whatever we managed to collect

    return BrowseFolderResponse(path=str(target), parent=parent_path, entries=entries)


@router.get("/quick-folders", response_model=QuickFoldersResponse)
def quick_folders():
    """
    Return a list of convenient starting points for the folder browser:
    home directory, desktop, common media locations, and the VisoMaster
    data folder (passed via the VM_DATA_FOLDER env var or auto-detected).
    """
    folders: list[QuickFolder] = []

    home = Path.home()
    folders.append(QuickFolder(label="Home", path=str(home)))

    # Desktop
    desktop = home / "Desktop"
    if desktop.is_dir():
        folders.append(QuickFolder(label="Desktop", path=str(desktop)))

    # Documents / Videos / Pictures — common on Windows and Linux
    for label, rel in [("Documents", "Documents"), ("Videos", "Videos"), ("Pictures", "Pictures")]:
        candidate = home / rel
        if candidate.is_dir():
            folders.append(QuickFolder(label=label, path=str(candidate)))

    # VisoMaster data folder — prefer env var, fall back to CWD
    data_folder_env = os.environ.get("VM_DATA_FOLDER", "").strip()
    if data_folder_env and Path(data_folder_env).is_dir():
        folders.append(QuickFolder(label="Data Folder", path=data_folder_env))
    else:
        # Fall back to the working directory (where VisoMaster was launched from)
        cwd = Path.cwd()
        folders.append(QuickFolder(label="Launch Directory", path=str(cwd)))

    # Windows drive roots (C:\, D:\, …)
    if platform.system() == "Windows":
        import string
        for letter in string.ascii_uppercase:
            drive = Path(f"{letter}:\\")
            if drive.exists():
                folders.append(QuickFolder(label=f"{letter}:\\", path=str(drive)))

    return QuickFoldersResponse(folders=folders)
