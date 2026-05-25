# VisoMaster — Product Overview

VisoMaster is a desktop AI face-swap and face-editing application. It processes images, video files, webcam streams, and WebRTC streams in real time, applying face swaps, expression editing, masking, and enhancement using GPU-accelerated ONNX/TensorRT models.

## Core Capabilities

- **Face Swap** — multiple swapper models (Inswapper, InStyleSwapper, SimSwap, GhostFace, CSCS, DeepFaceLab DFM)
- **Face Editor** — LivePortrait-based expression/pose control and color makeup adjustments
- **Face Restoration** — GFPGAN, CodeFormer, GPEN, VQFR, RestoreFormer
- **Frame Enhancement** — RealESRGAN, BSRGAN, DDColor, DeOldify
- **Masking** — Occluder, DFL XSeg, FaceParser, CLIPSeg, per-part mouth/eye restore
- **Live Playback** — real-time preview before saving; virtual camera output (OBS/Zoom/Twitch)
- **WebRTC Streaming** — WHIP protocol ingestion from phones/OBS via a bundled `streamrelay` subprocess
- **Video Markers** — per-frame parameter overrides for precise editing
- **TensorRT Acceleration** — engines auto-built and cached under `tensorrt-engines/`

## Target Users

Casual creators and professionals who need real-time or batch face-swap/editing on Windows (primary) and Linux (RunPod/server).

## Ethical Scope

Intended for creative, entertainment, and research use only. Users are responsible for consent and legal compliance.
