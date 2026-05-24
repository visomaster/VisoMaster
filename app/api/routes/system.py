"""
GET  /api/system/info
GET  /api/system/gpu-memory
POST /api/system/clear-memory
GET  /api/system/providers
POST /api/system/providers
"""
from __future__ import annotations

import platform
import sys

import torch
import onnxruntime

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_models_processor
from app.api.schemas import (
    GpuInfo,
    GpuMemoryResponse,
    OkResponse,
    ProviderResponse,
    ProviderSwitchRequest,
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
