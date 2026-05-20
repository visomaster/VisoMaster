# Design Document

## Overview

This design document describes the GPU-accelerated H.264 decoding pipeline for WebRTC streaming in VisoMaster. It introduces NVDEC hardware decoding via PyAV's h264_cuvid codec, GPU-resident NV12→RGB color conversion via PyTorch CUDA, and a NAL unit queue for lightweight IPC between the WebRTC subprocess and the main process.

## Architecture

The GPU-accelerated streaming feature introduces a parallel decoding path alongside the existing CPU-based WebRTC pipeline. The architecture splits into two modes selected at startup by a capability probe:

**GPU Path:** WebRTC Subprocess (signaling + NAL extraction) → `multiprocessing.Queue` → Main Process GPU Decoder (PyAV h264_cuvid) → GPU Color Converter (NV12→RGB via PyTorch CUDA) → FrameWorker (CUDA tensor input)

**CPU Path (fallback):** WebRTC Subprocess (signaling + full decode) → Shared Memory (BGR) → Main Process numpy read → FrameWorker (numpy input → GPU upload)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Main Process                                 │
│                                                                     │
│  ┌──────────────┐    ┌─────────────┐    ┌──────────────────┐       │
│  │ Capability   │───▶│ Video       │───▶│ FrameWorker      │       │
│  │ Probe        │    │ Processor   │    │ (accepts tensor  │       │
│  └──────────────┘    └──────┬──────┘    │  or numpy)       │       │
│                             │           └──────────────────┘       │
│                    ┌────────┴────────┐                              │
│                    │                 │                              │
│              [GPU Path]        [CPU Path]                           │
│                    │                 │                              │
│         ┌─────────▼──────┐   ┌──────▼───────┐                     │
│         │ NAL Queue      │   │ Shared Memory │                     │
│         │ Consumer       │   │ Poller        │                     │
│         └─────────┬──────┘   └──────────────-┘                     │
│                   │                                                 │
│         ┌─────────▼──────┐                                         │
│         │ GPU Decoder    │                                         │
│         │ (h264_cuvid)   │                                         │
│         └─────────┬──────┘                                         │
│                   │                                                 │
│         ┌─────────▼──────┐                                         │
│         │ Color Converter│                                         │
│         │ (NV12→RGB CUDA)│                                         │
│         └────────────────┘                                         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    WebRTC Subprocess                                 │
│                                                                     │
│  ┌──────────────┐    ┌─────────────────────────────────┐           │
│  │ Signaling    │    │ Track Handler                    │           │
│  │ (SDP/ICE)    │    │  GPU mode: extract NAL → Queue  │           │
│  │              │    │  CPU mode: decode → SharedMem    │           │
│  └──────────────┘    └─────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. CapabilityProbe

**Location:** `app/processors/external/nvdec_probe.py`

Responsible for detecting NVDEC hardware decoding support at startup. Runs once during VideoProcessor initialization and caches the result.

```python
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_nvdec_available: Optional[bool] = None


def probe_nvdec() -> bool:
    """
    Test NVDEC availability by attempting to open a PyAV codec context
    with the h264_cuvid decoder. Returns True if available.
    """
    global _nvdec_available
    if _nvdec_available is not None:
        return _nvdec_available

    try:
        import av
        codec = av.codec.Codec('h264_cuvid', 'r')
        ctx = codec.create()
        ctx.close()
        _nvdec_available = True
        logger.info("GPU-accelerated decoding enabled (NVDEC h264_cuvid available)")
    except Exception as e:
        _nvdec_available = False
        logger.warning(f"NVDEC unavailable, falling back to CPU decoding: {e}")

    return _nvdec_available


def is_nvdec_available() -> bool:
    """Return cached probe result. Runs probe if not yet executed."""
    if _nvdec_available is None:
        return probe_nvdec()
    return _nvdec_available
```

### 2. NALQueueHandler (Subprocess Side)

**Location:** Modifications to `app/processors/external/webrtc_server.py`

When GPU mode is active, the `VideoStreamTrack` handler extracts raw H.264 NAL units from the incoming RTP stream instead of decoding to BGR. NAL units are placed into a bounded `multiprocessing.Queue`.

```python
import multiprocessing
from collections import deque
import logging

logger = logging.getLogger(__name__)

NAL_QUEUE_MAX_SIZE = 120  # ~4 seconds at 30fps


class NALExtractorTrack:
    """
    Replaces VideoStreamTrack when GPU path is active.
    Extracts H.264 NAL units from the incoming track and enqueues them.
    """

    def __init__(self, track, nal_queue: multiprocessing.Queue):
        self._track = track
        self._nal_queue = nal_queue
        self._task = None

    def start(self):
        import asyncio
        self._task = asyncio.ensure_future(self._run())

    def stop(self):
        if self._task:
            self._task.cancel()

    async def _run(self):
        while True:
            try:
                packet = await self._track.recv()
            except Exception:
                break

            # Extract raw NAL bytes from the RTP packet/frame
            # aiortc provides VideoFrame; we need the encoded packet
            # In GPU mode, we intercept at the codec level before decode
            nal_bytes = self._extract_nal_from_packet(packet)
            if nal_bytes is None:
                continue

            try:
                if self._nal_queue.full():
                    # Drop oldest to prevent unbounded growth
                    try:
                        self._nal_queue.get_nowait()
                    except Exception:
                        pass
                    logger.debug("NAL queue full, dropped oldest unit")
                self._nal_queue.put_nowait(nal_bytes)
            except Exception as e:
                logger.debug(f"Failed to enqueue NAL unit: {e}")

    def _extract_nal_from_packet(self, packet) -> bytes | None:
        """Extract raw H.264 NAL unit bytes from an RTP video packet."""
        # Implementation depends on aiortc internals for accessing
        # the encoded payload before software decode
        if hasattr(packet, 'to_bytes'):
            return bytes(packet.to_bytes())
        return None
```

### 3. GPUDecoder

**Location:** `app/processors/external/gpu_decoder.py`

Manages a persistent PyAV `CodecContext` using the `h264_cuvid` decoder. Accepts NAL unit bytes and produces decoded NV12 frames.

```python
import logging
from typing import Optional, Generator

import av
import numpy as np

logger = logging.getLogger(__name__)


class GPUDecoder:
    """
    Decodes H.264 NAL units on the GPU using PyAV with h264_cuvid (NVDEC).
    Maintains a single persistent codec context per session.
    """

    def __init__(self):
        self._codec_ctx: Optional[av.codec.CodecContext] = None
        self._initialized = False

    def initialize(self) -> None:
        """Create the h264_cuvid codec context."""
        if self._initialized:
            return
        codec = av.codec.Codec('h264_cuvid', 'r')
        self._codec_ctx = codec.create()
        self._codec_ctx.open()
        self._initialized = True
        logger.info("GPUDecoder initialized with h264_cuvid")

    def decode(self, nal_bytes: bytes) -> list:
        """
        Feed a NAL unit to the decoder and return any decoded frames.
        Returns a list of decoded frames (may be empty if decoder is buffering).
        """
        if not self._initialized:
            self.initialize()

        frames = []
        try:
            packet = av.Packet(nal_bytes)
            decoded = self._codec_ctx.decode(packet)
            for frame in decoded:
                frames.append(frame)
        except av.error.InvalidDataError as e:
            logger.error(f"GPUDecoder: invalid NAL unit, skipping: {e}")
        except Exception as e:
            logger.error(f"GPUDecoder: decoding error, skipping: {e}")

        return frames

    def flush(self) -> list:
        """Flush remaining frames from the decoder."""
        if not self._initialized or self._codec_ctx is None:
            return []

        frames = []
        try:
            decoded = self._codec_ctx.decode(None)
            for frame in decoded:
                frames.append(frame)
        except Exception as e:
            logger.error(f"GPUDecoder: error during flush: {e}")
        return frames

    def close(self) -> None:
        """Close the codec context and release GPU resources."""
        if self._codec_ctx is not None:
            try:
                self.flush()
                self._codec_ctx.close()
            except Exception as e:
                logger.error(f"GPUDecoder: error during close: {e}")
            finally:
                self._codec_ctx = None
                self._initialized = False
                logger.info("GPUDecoder closed and resources released")

    @property
    def is_active(self) -> bool:
        return self._initialized and self._codec_ctx is not None
```

### 4. ColorConverter

**Location:** `app/processors/external/color_converter.py`

Performs NV12-to-RGB color space conversion entirely on the GPU using PyTorch CUDA tensor operations. The conversion follows BT.601 coefficients.

```python
import torch


class NV12ToRGBConverter:
    """
    Converts NV12 frames to RGB using PyTorch CUDA operations.
    All computation stays on GPU — no CPU memory copies.

    NV12 layout:
      - Y plane: H × W (uint8)
      - UV plane: H/2 × W (uint8, interleaved U and V)
    """

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device)

    def convert(self, y_plane: torch.Tensor, uv_plane: torch.Tensor) -> torch.Tensor:
        """
        Convert NV12 planes to RGB tensor.

        Args:
            y_plane: (H, W) uint8 tensor on CUDA — the luma plane
            uv_plane: (H//2, W) uint8 tensor on CUDA — interleaved chroma

        Returns:
            (H, W, 3) uint8 tensor on CUDA in RGB order
        """
        h, w = y_plane.shape

        # Convert to float for arithmetic
        y = y_plane.float()  # (H, W)

        # Separate U and V from interleaved UV plane
        u = uv_plane[:, 0::2].float()  # (H//2, W//2)
        v = uv_plane[:, 1::2].float()  # (H//2, W//2)

        # Upsample U and V to full resolution via nearest-neighbor
        u = u.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)  # (H, W)
        v = v.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)  # (H, W)

        # BT.601 conversion
        # R = Y + 1.402 * (V - 128)
        # G = Y - 0.344136 * (U - 128) - 0.714136 * (V - 128)
        # B = Y + 1.772 * (U - 128)
        c = y
        d = u - 128.0
        e = v - 128.0

        r = c + 1.402 * e
        g = c - 0.344136 * d - 0.714136 * e
        b = c + 1.772 * d

        # Clamp and convert to uint8
        rgb = torch.stack([r, g, b], dim=-1)  # (H, W, 3)
        rgb = rgb.clamp(0, 255).to(torch.uint8)

        return rgb

    def convert_from_frame(self, av_frame) -> torch.Tensor:
        """
        Convert a PyAV VideoFrame (NV12 format) to an RGB CUDA tensor.

        Args:
            av_frame: PyAV VideoFrame in NV12 format

        Returns:
            (H, W, 3) uint8 CUDA tensor in RGB order
        """
        # Extract planes as numpy arrays, then upload to GPU
        # Note: For true zero-copy with NVDEC hardware surfaces,
        # we'd use CUDA-mapped memory. This path uploads the NV12
        # planes which are much smaller than full RGB frames.
        import numpy as np

        y_np = av_frame.planes[0].to_ndarray()  # H × W
        uv_np = av_frame.planes[1].to_ndarray()  # H//2 × W

        y_tensor = torch.from_numpy(y_np).to(self.device)
        uv_tensor = torch.from_numpy(uv_np).to(self.device)

        return self.convert(y_tensor, uv_tensor)
```

### 5. Modified FrameWorker

**Location:** Modifications to `app/processors/workers/frame_worker.py`

The FrameWorker constructor and `process_frame` method are updated to accept either a numpy ndarray or a CUDA tensor. Type detection is implicit via `isinstance` checks.

```python
import torch
import numpy as np


class FrameWorker(threading.Thread):
    def __init__(self, frame, main_window, frame_number, frame_queue, is_single_frame=False):
        super().__init__()
        # frame can be numpy.ndarray OR torch.Tensor (CUDA)
        self.frame = frame
        # ... rest of existing init ...

    def run(self):
        try:
            # ... existing parameter setup ...

            if self.main_window.swapfacesButton.isChecked() or ...:
                self.frame = self.process_frame()
            else:
                self.frame = self._to_bgr_numpy(self.frame)

            self.frame = np.ascontiguousarray(self.frame)
            # ... rest of existing run ...

    def process_frame(self):
        # Load frame into VRAM — handle both input types
        if isinstance(self.frame, torch.Tensor) and self.frame.is_cuda:
            # GPU path: frame is already (H, W, 3) uint8 on CUDA
            img = self.frame.permute(2, 0, 1)  # (3, H, W)
        else:
            # CPU path: numpy array, upload to GPU
            img = torch.from_numpy(self.frame.astype('uint8')).to(self.models_processor.device)
            img = img.permute(2, 0, 1)  # (3, H, W)

        # ... rest of existing process_frame unchanged ...

    def _to_bgr_numpy(self, frame) -> np.ndarray:
        """Convert frame to BGR numpy for non-processing display path."""
        if isinstance(frame, torch.Tensor):
            if frame.is_cuda:
                frame = frame.cpu()
            frame = frame.numpy()
        # RGB to BGR
        return frame[..., ::-1]
```

### 6. Modified VideoProcessor

**Location:** Modifications to `app/processors/video_processor.py`

VideoProcessor gains a NAL queue consumer loop for the GPU path, alongside the existing shared memory poller for the CPU path.

```python
import multiprocessing
from app.processors.external.nvdec_probe import is_nvdec_available
from app.processors.external.gpu_decoder import GPUDecoder
from app.processors.external.color_converter import NV12ToRGBConverter


class VideoProcessor(QObject):
    def __init__(self, main_window, num_threads=2):
        super().__init__()
        # ... existing init ...

        # GPU decoding components
        self._nvdec_available = False
        self._gpu_decoder: GPUDecoder | None = None
        self._color_converter: NV12ToRGBConverter | None = None
        self._nal_queue: multiprocessing.Queue | None = None
        self._gpu_decode_active = False

    def initialize_webrtc_gpu_path(self):
        """Initialize GPU decoding components if NVDEC is available."""
        self._nvdec_available = is_nvdec_available()
        if self._nvdec_available:
            self._gpu_decoder = GPUDecoder()
            self._color_converter = NV12ToRGBConverter(device='cuda')

    def process_next_webrtc_frame_gpu(self):
        """GPU path: consume NAL units from queue, decode, convert, dispatch."""
        if self.frame_queue.qsize() >= self.num_threads:
            return
        if self._nal_queue is None:
            return

        try:
            nal_bytes = self._nal_queue.get_nowait()
        except Exception:
            return  # No NAL unit available

        try:
            frames = self._gpu_decoder.decode(nal_bytes)
            for frame in frames:
                rgb_tensor = self._color_converter.convert_from_frame(frame)
                rgb_tensor = self._apply_streaming_transforms_tensor(rgb_tensor)
                self.frame_queue.put(self.current_frame_number)
                self.start_frame_worker(self.current_frame_number, rgb_tensor)
        except Exception as e:
            print(f"[WebRTC GPU] Decode error, falling back to CPU: {e}")
            self._switch_to_cpu_fallback()

    def _switch_to_cpu_fallback(self):
        """Switch from GPU path to CPU fallback mid-session."""
        self._gpu_decode_active = False
        if self._gpu_decoder:
            self._gpu_decoder.close()
            self._gpu_decoder = None
        # Reconnect to shared memory path
        self.frame_read_timer.stop()
        self.frame_read_timer.timeout.disconnect()
        self.frame_read_timer.timeout.connect(self.process_next_webrtc_frame)
        self.frame_read_timer.start(int(1000 / 30 * 0.8))

    def stop_gpu_decoder(self):
        """Drain queue and close GPU decoder on session end."""
        if self._nal_queue is not None:
            # Drain remaining items
            while not self._nal_queue.empty():
                try:
                    nal_bytes = self._nal_queue.get_nowait()
                    if self._gpu_decoder and self._gpu_decoder.is_active:
                        self._gpu_decoder.decode(nal_bytes)
                except Exception:
                    break

        if self._gpu_decoder:
            self._gpu_decoder.close()
            self._gpu_decoder = None
        self._gpu_decode_active = False
```

### Interfaces

#### NAL Queue Interface

```python
# Created in Main Process, passed to subprocess
nal_queue = multiprocessing.Queue(maxsize=120)

# Subprocess enqueues:
nal_queue.put_nowait(nal_bytes: bytes)

# Main Process dequeues:
nal_bytes = nal_queue.get_nowait()  # raises queue.Empty if none available
```

#### GPUDecoder Interface

```python
decoder = GPUDecoder()
decoder.initialize()                    # Opens h264_cuvid context
frames = decoder.decode(nal_bytes)      # Returns list of av.VideoFrame
frames = decoder.flush()                # Flush buffered frames
decoder.close()                         # Release resources
decoder.is_active                       # Property: bool
```

#### ColorConverter Interface

```python
converter = NV12ToRGBConverter(device='cuda')
rgb_tensor = converter.convert(y_plane, uv_plane)       # From raw planes
rgb_tensor = converter.convert_from_frame(av_frame)     # From PyAV frame
# Returns: torch.Tensor shape (H, W, 3), dtype uint8, device cuda
```

#### FrameWorker Input Contract

```python
# Accepts either:
worker = FrameWorker(frame=numpy_array, ...)   # numpy.ndarray (H, W, 3) uint8 RGB
worker = FrameWorker(frame=cuda_tensor, ...)   # torch.Tensor (H, W, 3) uint8 cuda RGB
```

## Data Models

### NAL Unit Message

```python
# Raw bytes passed through multiprocessing.Queue
# No wrapper class needed — just bytes objects
nal_unit: bytes  # Raw H.264 NAL unit payload
```

### Capability State

```python
# Stored as module-level cached boolean in nvdec_probe.py
_nvdec_available: Optional[bool] = None  # None = not yet probed
```

### GPU Decoder State

```python
class GPUDecoderState:
    codec_ctx: av.codec.CodecContext | None  # Persistent across session
    initialized: bool                         # Whether context is open
```

### Frame Data Flow Types

```python
from typing import Union
import numpy as np
import torch

# Frame type accepted by FrameWorker
FrameInput = Union[np.ndarray, torch.Tensor]
# np.ndarray: shape (H, W, 3), dtype uint8, RGB order
# torch.Tensor: shape (H, W, 3), dtype uint8, device cuda, RGB order
```

## Error Handling

### Decoder Errors

| Error Condition | Handling Strategy |
|---|---|
| Invalid NAL unit (corrupt data) | Log error, skip NAL unit, continue with next |
| NVDEC hardware failure mid-session | Log error, close decoder, switch to CPU fallback |
| PyAV codec context crash | Catch exception, close decoder, switch to CPU fallback |
| Queue closed (subprocess died) | Detect `EOFError`/`BrokenPipeError`, stop decoder gracefully |

### Queue Overflow

| Condition | Handling |
|---|---|
| NAL queue full (producer faster than consumer) | Drop oldest NAL unit, log debug warning |
| Main process queue full (frame_queue) | Skip frame dispatch, wait for next timer tick |

### Session Lifecycle Errors

| Condition | Handling |
|---|---|
| New session while old decoder active | Close old decoder first, then initialize new |
| Subprocess unexpected termination | Detect via queue error, stop decoder, clean up |
| GPU OOM during decode | Catch RuntimeError, close decoder, switch to CPU |

## Testing Strategy

### Unit Tests
- CapabilityProbe: Mock PyAV to test both success and failure paths
- GPUDecoder: Mock codec context to verify lifecycle (init, decode, flush, close)
- ColorConverter: Test with known NV12 values against expected RGB output
- FrameWorker: Verify both numpy and CUDA tensor inputs produce correct results
- NAL queue overflow: Verify bounded behavior and oldest-drop semantics

### Property-Based Tests
- NV12→RGB conversion correctness across random valid pixel values
- FrameWorker path equivalence (numpy vs CUDA tensor input)
- Decoder resilience to random/corrupt byte sequences
- Queue overflow preserves most recent N items

### Integration Tests
- End-to-end GPU path: NAL enqueue → decode → color convert → FrameWorker
- CPU fallback activation when NVDEC is unavailable
- Mid-session fallback when GPU decoder fails
- Session lifecycle: start, process frames, stop, resource cleanup

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: NAL Queue Overflow Preserves Most Recent Data

*For any* sequence of NAL units enqueued that exceeds the queue capacity N, the queue SHALL contain exactly the N most recently enqueued NAL units, with all older units having been discarded.

**Validates: Requirements 2.3, 2.4**

### Property 2: Decoder Resilience to Malformed Input

*For any* byte sequence fed to the GPU_Decoder (including random/corrupt data), the decoder SHALL either produce valid decoded frames or log an error and continue accepting subsequent NAL units — it SHALL NOT raise an unhandled exception or leave the decoder in an unusable state.

**Validates: Requirements 3.4**

### Property 3: NV12 to RGB Conversion Correctness and Output Format

*For any* valid NV12 frame with Y values in [0, 255] and UV values in [0, 255] and dimensions H×W (where H and W are even and positive), the Color_Converter SHALL produce a CUDA tensor of shape (H, W, 3) with dtype uint8, where each pixel's RGB values match the BT.601 conversion formula within ±1 of the expected value (due to integer rounding).

**Validates: Requirements 4.1, 4.2**

### Property 4: FrameWorker Path Equivalence

*For any* valid RGB frame data (H×W×3, uint8), processing the frame through FrameWorker as a numpy ndarray (CPU path) and as an equivalent CUDA tensor (GPU path) SHALL produce pixel-identical output arrays.

**Validates: Requirements 5.3, 5.4**

### Property 5: GPU Failure Triggers CPU Fallback

*For any* runtime exception raised by the GPU_Decoder during an active session, the VideoProcessor SHALL transition to the CPU_Fallback_Path and continue processing subsequent frames without interruption.

**Validates: Requirements 6.4**

### Property 6: Queue Drain Completeness on Session Stop

*For any* number of NAL units remaining in the NAL_Queue when a session stop is initiated, the VideoProcessor SHALL dequeue all remaining items and signal the GPU_Decoder to flush before releasing resources.

**Validates: Requirements 8.2, 8.3**

### Property 7: Graceful Subprocess Termination Handling

*For any* unexpected termination of the WebRTC_Subprocess (simulated via queue closure, broken pipe, or EOF), the VideoProcessor SHALL detect the failure condition and stop the GPU_Decoder without raising an unhandled exception in the Main_Process.

**Validates: Requirements 8.5**
