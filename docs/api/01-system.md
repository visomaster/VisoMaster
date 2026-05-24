# System & Hardware

Endpoints for inspecting the runtime environment and managing GPU resources.

---

## GET /api/system/info

Returns hardware and runtime version information. Call this on startup to check what's available before configuring providers.

**Response**

```json
{
  "platform": "Windows-11-10.0.22631-SP0",
  "python_version": "3.10.13 ...",
  "torch_version": "2.4.1+cu124",
  "cuda_available": true,
  "cuda_version": "12.4",
  "ort_version": "1.20.0",
  "ort_providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
  "trt_available": true,
  "trt_version": "10.6.0",
  "ffmpeg_available": true,
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "total_mb": 24564,
      "free_mb": 20100,
      "used_mb": 4464
    }
  ]
}
```

---

## GET /api/system/gpu-memory

Returns current GPU memory usage for the primary GPU (index 0). Useful for polling a VRAM bar in the UI.

**Response**

```json
{ "used_mb": 4464, "total_mb": 24564 }
```

Returns `{ "used_mb": 0, "total_mb": 0 }` if `nvidia-smi` is unavailable.

---

## POST /api/system/clear-memory

Unloads all loaded ONNX/TensorRT/DFM model sessions and calls `torch.cuda.empty_cache()`. Use this when switching between heavy workloads or if VRAM is exhausted.

**No request body.**

**Response**

```json
{ "ok": true, "message": "GPU memory cleared" }
```

> Models will be lazily reloaded on the next inference call.

---

## GET /api/system/providers

Returns the currently active ONNX Runtime execution provider.

**Response**

```json
{ "active_provider": "CUDA" }
```

Possible values: `"CUDA"`, `"TensorRT"`, `"TensorRT-Engine"`, `"CPU"`.

---

## POST /api/system/providers

Switches the active execution provider. All loaded model sessions are deleted and will be reloaded on next use.

**Request body**

```json
{ "provider": "TensorRT" }
```

| Field | Type | Values |
|---|---|---|
| `provider` | string | `"CUDA"` · `"TensorRT"` · `"TensorRT-Engine"` · `"CPU"` |

**Provider guide**

| Provider | When to use |
|---|---|
| `CUDA` | Default GPU path. Fast, no build step. |
| `TensorRT` | Builds TRT engines on first use (slow first run, fast after). Best throughput. |
| `TensorRT-Engine` | Like TensorRT but uses cached ORT EP context files. Requires TRT ≥ 10.2. |
| `CPU` | No GPU. Slow but works everywhere. |

**Response**

```json
{ "active_provider": "TensorRT" }
```
