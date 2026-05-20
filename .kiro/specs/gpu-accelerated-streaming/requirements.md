# Requirements Document

## Introduction

This feature introduces GPU-accelerated H.264 decoding for the WebRTC streaming pipeline in VisoMaster. The current CPU-based path (aiortc → shared memory → numpy → torch GPU upload) is preserved as a fallback. The new GPU path moves decoding into the main process using NVDEC (via PyAV/h264_cuvid), eliminates the shared memory transfer for decoded frames, and delivers frames directly to FrameWorker as CUDA tensors. The WebRTC subprocess is reduced to signaling-only duties, passing compressed H.264 NAL units to the main process via a lightweight queue.

## Glossary

- **WebRTC_Subprocess**: The child process launched via `multiprocessing.Process` that runs the aiohttp/aiortc WebRTC signaling server (`webrtc_server.py`).
- **Main_Process**: The primary VisoMaster process hosting the Qt event loop, VideoProcessor, and FrameWorker threads.
- **NAL_Queue**: A `multiprocessing.Queue` (or equivalent lightweight IPC mechanism) used to pass compressed H.264 NAL units from the WebRTC_Subprocess to the Main_Process.
- **GPU_Decoder**: The component in the Main_Process that uses PyAV with the `h264_cuvid` FFmpeg decoder to decode H.264 NAL units on the GPU via NVDEC.
- **NVDEC**: NVIDIA's hardware video decoder engine accessed through FFmpeg's `h264_cuvid` codec.
- **Color_Converter**: The component that performs NV12-to-RGB color space conversion on the GPU using PyTorch/CUDA operations.
- **FrameWorker**: The existing threaded worker (`frame_worker.py`) that receives frames and runs face-processing inference on the GPU.
- **VideoProcessor**: The existing QObject (`video_processor.py`) that orchestrates frame reading, dispatching to FrameWorker, and display.
- **CPU_Fallback_Path**: The existing pipeline: aiortc decodes on CPU → BGR frame written to shared memory → Main_Process reads via numpy → uploads to GPU via `torch.from_numpy().to(device)`.
- **Capability_Probe**: A startup check that determines whether NVDEC hardware decoding is available on the current system.
- **CUDA_Tensor**: A `torch.Tensor` residing in GPU memory (device='cuda').

## Requirements

### Requirement 1: NVDEC Capability Detection

**User Story:** As a user, I want the application to automatically detect whether my GPU supports hardware video decoding, so that I get the best performance without manual configuration.

#### Acceptance Criteria

1. WHEN the Main_Process initializes the WebRTC streaming subsystem, THE Capability_Probe SHALL test for NVDEC availability by attempting to open a PyAV codec context with the `h264_cuvid` decoder.
2. WHEN the Capability_Probe determines that NVDEC is available, THE Capability_Probe SHALL log an informational message indicating GPU-accelerated decoding is enabled.
3. WHEN the Capability_Probe determines that NVDEC is unavailable, THE Capability_Probe SHALL log a warning message indicating fallback to CPU decoding.
4. THE Capability_Probe SHALL complete the detection within the Main_Process startup sequence before any WebRTC stream is accepted.
5. THE Capability_Probe SHALL store the detection result as a boolean flag accessible to the VideoProcessor.

### Requirement 2: WebRTC Subprocess Signaling-Only Mode

**User Story:** As a developer, I want the WebRTC subprocess to handle only signaling and pass compressed NAL units to the main process, so that GPU decoding can happen without shared memory overhead.

#### Acceptance Criteria

1. WHILE the GPU_Decoder path is active, THE WebRTC_Subprocess SHALL extract compressed H.264 NAL units from incoming RTP packets instead of decoding frames to BGR.
2. WHILE the GPU_Decoder path is active, THE WebRTC_Subprocess SHALL enqueue each H.264 NAL unit (as raw bytes) into the NAL_Queue.
3. THE NAL_Queue SHALL use a bounded capacity to prevent unbounded memory growth when the Main_Process consumer is slower than the producer.
4. IF the NAL_Queue reaches its capacity limit, THEN THE WebRTC_Subprocess SHALL drop the oldest enqueued NAL unit and log a debug-level warning.
5. WHILE the GPU_Decoder path is active, THE WebRTC_Subprocess SHALL continue to perform WebRTC signaling (SDP offer/answer, ICE negotiation) as it does today.
6. WHILE the CPU_Fallback_Path is active, THE WebRTC_Subprocess SHALL continue to decode frames and write BGR data to shared memory using the existing pipeline.

### Requirement 3: GPU Decoding via PyAV h264_cuvid

**User Story:** As a user, I want H.264 video frames decoded directly on my GPU, so that CPU load is reduced and frames reach the face-processing pipeline faster.

#### Acceptance Criteria

1. WHEN a NAL unit is dequeued from the NAL_Queue, THE GPU_Decoder SHALL feed the NAL unit to a PyAV `CodecContext` configured with the `h264_cuvid` decoder.
2. WHEN the GPU_Decoder produces a decoded frame, THE GPU_Decoder SHALL output the frame in the hardware surface format provided by NVDEC (NV12).
3. THE GPU_Decoder SHALL maintain a single persistent `CodecContext` instance for the duration of a WebRTC session to avoid repeated initialization overhead.
4. IF the GPU_Decoder encounters a decoding error on a NAL unit, THEN THE GPU_Decoder SHALL log the error and skip to the next available NAL unit without crashing the Main_Process.
5. WHEN a WebRTC session ends, THE GPU_Decoder SHALL flush and close the `CodecContext` and release associated GPU resources.

### Requirement 4: GPU Color Conversion (NV12 to RGB)

**User Story:** As a developer, I want decoded NV12 frames converted to RGB on the GPU, so that the data stays in GPU memory and avoids a CPU round-trip before face processing.

#### Acceptance Criteria

1. WHEN the GPU_Decoder outputs an NV12 frame, THE Color_Converter SHALL convert the frame from NV12 color space to RGB color space using PyTorch CUDA tensor operations.
2. THE Color_Converter SHALL produce a CUDA_Tensor with shape (H, W, 3) and dtype uint8 in RGB channel order.
3. THE Color_Converter SHALL perform the conversion entirely on the GPU without copying pixel data to CPU memory.
4. WHEN the Color_Converter produces an RGB CUDA_Tensor, THE Color_Converter SHALL pass the tensor directly to the FrameWorker for processing.

### Requirement 5: FrameWorker CUDA Tensor Input Support

**User Story:** As a developer, I want FrameWorker to accept frames that are already on the GPU as CUDA tensors, so that the GPU path avoids redundant CPU-to-GPU transfers.

#### Acceptance Criteria

1. WHEN FrameWorker receives a frame as a CUDA_Tensor, THE FrameWorker SHALL skip the `torch.from_numpy().to(device)` upload step and use the tensor directly.
2. WHEN FrameWorker receives a frame as a numpy ndarray (CPU_Fallback_Path), THE FrameWorker SHALL continue to upload the frame to GPU via `torch.from_numpy().to(device)` as it does today.
3. THE FrameWorker SHALL accept both input types (CUDA_Tensor and numpy ndarray) without requiring callers to specify the type explicitly.
4. THE FrameWorker SHALL produce identical processing results regardless of whether the input frame arrived as a CUDA_Tensor or a numpy ndarray.

### Requirement 6: CPU Fallback Path Preservation

**User Story:** As a user without a supported NVIDIA GPU, I want the existing CPU-based WebRTC pipeline to continue working unchanged, so that the application remains functional on my hardware.

#### Acceptance Criteria

1. WHILE the Capability_Probe has determined NVDEC is unavailable, THE VideoProcessor SHALL use the CPU_Fallback_Path for all WebRTC frame processing.
2. THE CPU_Fallback_Path SHALL operate identically to the current implementation: aiortc decodes frames on CPU, writes BGR to shared memory, and the Main_Process reads via numpy.
3. THE CPU_Fallback_Path SHALL remain fully functional and unmodified in its behavior when the GPU path is not active.
4. IF the GPU_Decoder fails during an active session (runtime error), THEN THE VideoProcessor SHALL log the error and fall back to the CPU_Fallback_Path for the remainder of that session.

### Requirement 7: Shared Memory Elimination for GPU Path

**User Story:** As a developer, I want the GPU decoding path to bypass shared memory entirely, so that the pipeline has lower latency and simpler resource management.

#### Acceptance Criteria

1. WHILE the GPU_Decoder path is active, THE Main_Process SHALL NOT attach to or read from the shared memory block (`visomaster_webrtc_frame`).
2. WHILE the GPU_Decoder path is active, THE WebRTC_Subprocess SHALL NOT create or write to the shared memory block.
3. WHILE the CPU_Fallback_Path is active, THE WebRTC_Subprocess SHALL create and write to the shared memory block as it does today.
4. THE VideoProcessor SHALL select between the NAL_Queue consumer (GPU path) and the shared memory poller (CPU path) based on the Capability_Probe result.

### Requirement 8: Session Lifecycle Management

**User Story:** As a user, I want WebRTC streaming sessions to start and stop cleanly regardless of which decoding path is active, so that resources are properly managed.

#### Acceptance Criteria

1. WHEN a WebRTC session starts and the GPU path is active, THE VideoProcessor SHALL begin consuming NAL units from the NAL_Queue and feeding them to the GPU_Decoder.
2. WHEN a WebRTC session stops, THE VideoProcessor SHALL drain remaining items from the NAL_Queue and signal the GPU_Decoder to flush.
3. WHEN a WebRTC session stops, THE GPU_Decoder SHALL release all GPU memory associated with the decoding context.
4. WHEN a new WebRTC session starts while a previous session's GPU_Decoder is still active, THE VideoProcessor SHALL close the previous GPU_Decoder before initializing a new one.
5. IF the WebRTC_Subprocess terminates unexpectedly, THEN THE VideoProcessor SHALL detect the closed NAL_Queue and stop the GPU_Decoder gracefully.
