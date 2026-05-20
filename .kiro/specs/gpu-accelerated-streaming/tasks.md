# Implementation Plan: GPU-Accelerated Streaming

## Overview

This plan implements a GPU-accelerated H.264 decoding pipeline for WebRTC streaming. The work introduces NVDEC hardware decoding via PyAV's `h264_cuvid` codec, GPU-resident NV12→RGB color conversion via PyTorch CUDA, and a NAL unit queue for lightweight IPC — all while preserving the existing CPU fallback path unchanged.

## Tasks

- [ ] 1. Create NVDEC capability probe module
  - [ ] 1.1 Implement `nvdec_probe.py` with `probe_nvdec()` and `is_nvdec_available()` functions
    - Create `app/processors/external/nvdec_probe.py`
    - Implement probe that attempts to open a PyAV codec context with `h264_cuvid`
    - Cache result as module-level boolean
    - Log informational message on success, warning on failure
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 1.2 Write unit tests for capability probe
    - Mock PyAV to test success path (NVDEC available)
    - Mock PyAV to test failure path (NVDEC unavailable)
    - Verify caching behavior (probe runs only once)
    - _Requirements: 1.1, 1.2, 1.3, 1.5_

- [ ] 2. Implement GPU decoder module
  - [ ] 2.1 Create `gpu_decoder.py` with `GPUDecoder` class
    - Create `app/processors/external/gpu_decoder.py`
    - Implement `initialize()`, `decode()`, `flush()`, `close()` methods
    - Maintain persistent `CodecContext` for session duration
    - Handle decoding errors gracefully (log and skip)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [ ]* 2.2 Write property test for decoder resilience
    - **Property 2: Decoder Resilience to Malformed Input**
    - Feed random/corrupt byte sequences to the decoder
    - Verify decoder never raises unhandled exceptions
    - Verify decoder remains usable after malformed input
    - **Validates: Requirements 3.4**

  - [ ]* 2.3 Write unit tests for GPU decoder lifecycle
    - Test initialize/close cycle
    - Test decode with valid NAL bytes (mocked codec context)
    - Test flush returns buffered frames
    - Test error handling on invalid data
    - _Requirements: 3.1, 3.3, 3.4, 3.5_

- [ ] 3. Implement NV12-to-RGB color converter
  - [ ] 3.1 Create `color_converter.py` with `NV12ToRGBConverter` class
    - Create `app/processors/external/color_converter.py`
    - Implement BT.601 NV12→RGB conversion using PyTorch CUDA tensor ops
    - Implement `convert()` for raw Y/UV planes and `convert_from_frame()` for PyAV frames
    - Output shape (H, W, 3), dtype uint8, on CUDA device
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [ ]* 3.2 Write property test for NV12→RGB conversion correctness
    - **Property 3: NV12 to RGB Conversion Correctness and Output Format**
    - Generate random valid NV12 frames (even H×W, Y/UV in [0,255])
    - Verify output shape is (H, W, 3), dtype uint8, device cuda
    - Verify pixel values match BT.601 formula within ±1
    - **Validates: Requirements 4.1, 4.2**

  - [ ]* 3.3 Write unit tests for color converter
    - Test with known NV12 values against expected RGB output
    - Test edge cases (all-black, all-white, pure colors)
    - Verify no CPU memory copies during conversion
    - _Requirements: 4.1, 4.2, 4.3_

- [ ] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Modify WebRTC subprocess for dual-mode operation
  - [ ] 5.1 Add NAL extractor track handler to `webrtc_server.py`
    - Implement `NALExtractorTrack` class that extracts H.264 NAL units from incoming RTP packets
    - Enqueue NAL bytes into a `multiprocessing.Queue` instead of decoding to BGR
    - Implement bounded queue with oldest-drop overflow policy
    - Log debug warning on queue overflow
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [ ] 5.2 Add mode flag and NAL queue parameter to `run_server()`
    - Accept `gpu_mode: bool` and `nal_queue: multiprocessing.Queue | None` parameters
    - When `gpu_mode=True`, use `NALExtractorTrack` instead of `VideoStreamTrack`
    - When `gpu_mode=False`, preserve existing shared memory behavior unchanged
    - _Requirements: 2.5, 2.6, 7.2, 7.3_

  - [ ]* 5.3 Write property test for NAL queue overflow behavior
    - **Property 1: NAL Queue Overflow Preserves Most Recent Data**
    - Enqueue more than N items into a bounded queue of size N
    - Verify queue contains exactly the N most recent items
    - **Validates: Requirements 2.3, 2.4**

- [ ] 6. Modify VideoProcessor for GPU path integration
  - [ ] 6.1 Add GPU path initialization to `VideoProcessor`
    - Import and call `is_nvdec_available()` during WebRTC setup
    - Initialize `GPUDecoder` and `NV12ToRGBConverter` when NVDEC is available
    - Create `multiprocessing.Queue(maxsize=120)` for NAL units
    - Store `_nvdec_available` flag for path selection
    - _Requirements: 1.4, 1.5, 7.4_

  - [ ] 6.2 Implement `process_next_webrtc_frame_gpu()` method
    - Consume NAL units from queue via `get_nowait()`
    - Decode via `GPUDecoder`, convert via `NV12ToRGBConverter`
    - Pass resulting CUDA tensor to `start_frame_worker()`
    - _Requirements: 3.1, 4.4, 7.1_

  - [ ] 6.3 Implement CPU fallback switch and session lifecycle
    - Implement `_switch_to_cpu_fallback()` for mid-session GPU failure
    - Implement `stop_gpu_decoder()` to drain queue and close decoder on session end
    - Handle subprocess termination detection via queue errors
    - Close old decoder before initializing new one on session restart
    - _Requirements: 6.4, 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ] 6.4 Wire GPU/CPU path selection into `process_video()` for WebRTC
    - When `_nvdec_available` is True, pass `nal_queue` to subprocess and connect GPU frame reader
    - When False, use existing shared memory path unchanged
    - Skip shared memory attachment when GPU path is active
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [ ]* 6.5 Write property test for GPU failure triggers CPU fallback
    - **Property 5: GPU Failure Triggers CPU Fallback**
    - Simulate GPU decoder exception during active session
    - Verify VideoProcessor transitions to CPU fallback path
    - Verify subsequent frames continue processing
    - **Validates: Requirements 6.4**

  - [ ]* 6.6 Write property test for queue drain on session stop
    - **Property 6: Queue Drain Completeness on Session Stop**
    - Enqueue N NAL units, then trigger session stop
    - Verify all items are dequeued and decoder is flushed
    - **Validates: Requirements 8.2, 8.3**

  - [ ]* 6.7 Write property test for graceful subprocess termination
    - **Property 7: Graceful Subprocess Termination Handling**
    - Simulate queue closure / broken pipe / EOF
    - Verify VideoProcessor stops decoder without unhandled exceptions
    - **Validates: Requirements 8.5**

- [ ] 7. Modify FrameWorker for dual-input support
  - [ ] 7.1 Update `FrameWorker` to accept CUDA tensor or numpy array
    - Add `isinstance` check at start of `process_frame()` to detect input type
    - When input is CUDA tensor: skip `torch.from_numpy().to(device)`, permute directly
    - When input is numpy: preserve existing upload path unchanged
    - Add `_to_bgr_numpy()` helper for non-processing display path
    - _Requirements: 5.1, 5.2, 5.3_

  - [ ]* 7.2 Write property test for FrameWorker path equivalence
    - **Property 4: FrameWorker Path Equivalence**
    - Generate random RGB frames (H×W×3, uint8)
    - Process same frame as numpy and as equivalent CUDA tensor
    - Verify pixel-identical output
    - **Validates: Requirements 5.3, 5.4**

- [ ] 8. Integration wiring and end-to-end validation
  - [ ] 8.1 Wire all components together in the WebRTC startup flow
    - In `process_video()` WebRTC branch: probe NVDEC, create queue, launch subprocess with mode flag
    - Connect appropriate timer callback based on path selection
    - Ensure `stop_processing()` calls `stop_gpu_decoder()` when GPU path was active
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ]* 8.2 Write integration tests for end-to-end GPU path
    - Test NAL enqueue → decode → color convert → FrameWorker dispatch
    - Test CPU fallback activation when NVDEC is unavailable
    - Test mid-session fallback when GPU decoder fails
    - Test session lifecycle: start, process, stop, resource cleanup
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 8.1, 8.2, 8.3_

- [ ] 9. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The CPU fallback path (Requirement 6) is preserved by not modifying existing shared memory logic — only adding the new GPU path alongside it

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1", "3.1"] },
    { "id": 1, "tasks": ["1.2", "2.2", "2.3", "3.2", "3.3"] },
    { "id": 2, "tasks": ["5.1", "5.2", "7.1"] },
    { "id": 3, "tasks": ["5.3", "7.2", "6.1"] },
    { "id": 4, "tasks": ["6.2", "6.3", "6.4"] },
    { "id": 5, "tasks": ["6.5", "6.6", "6.7"] },
    { "id": 6, "tasks": ["8.1"] },
    { "id": 7, "tasks": ["8.2"] }
  ]
}
```
