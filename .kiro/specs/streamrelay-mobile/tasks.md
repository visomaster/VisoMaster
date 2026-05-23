# Implementation Plan: StreamRelay Mobile

## Overview

This implementation plan creates a Flutter mobile application that streams live camera frames to a StreamRelay server using hardware-accelerated H.264 encoding. The architecture uses native code (Swift/Kotlin) for the streaming pipeline to achieve 30-60 fps performance. iOS implementation is completed first to validate the architecture before Android implementation.

## Tasks

- [ ] 1. Set up Flutter project structure and core interfaces
  - [ ] 1.1 Create Flutter project with required dependencies
    - Initialize Flutter project `streamrelay_mobile`
    - Add dependencies: `shared_preferences`, `provider`
    - Configure iOS and Android platform settings
    - _Requirements: 9.1, 9.2_

  - [ ] 1.2 Create Dart data models and interfaces
    - Create `Resolution` class with width, height, aspectRatio
    - Create `StreamingStats` class with fps, bitrate
    - Create `StreamRelayException` class
    - _Requirements: 5.1, 8.1, 8.2_

  - [ ] 1.3 Implement StreamRelayController with MethodChannel
    - Create controller with MethodChannel `com.streamrelay.mobile/streaming`
    - Implement initialize, startStreaming, stopStreaming methods
    - Implement setHost, setPort, setResolution, setBitrate methods
    - Handle native callbacks: onStatsUpdate, onError, onConnectionStateChanged
    - _Requirements: 9.3, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [ ]* 1.4 Write unit tests for StreamRelayController
    - Test state management and notifyListeners behavior
    - Test validation of host/port inputs
    - _Requirements: 7.7_

- [ ] 2. Implement settings persistence
  - [ ] 2.1 Implement settings save/load in StreamRelayController
    - Save host, port, resolution, bitrate to SharedPreferences
    - Load settings on controller initialization
    - Default port to 9090
    - _Requirements: 3.2, 3.3, 3.4, 5.4, 5.5, 6.4, 6.5_

  - [ ]* 2.2 Write property test for settings persistence round-trip
    - **Property 3: Settings Persistence Round-Trip**
    - **Validates: Requirements 3.3, 3.4, 5.4, 5.5, 6.4, 6.5**

- [ ] 3. Implement Flutter UI widgets
  - [ ] 3.1 Create CameraPreviewWidget
    - Display live camera feed using Texture widget
    - Accept textureId and aspectRatio parameters
    - Handle texture not available state
    - _Requirements: 1.5_

  - [ ] 3.2 Create SettingsPanel widget
    - Text input for server host
    - Text input for port with numeric keyboard
    - Resolution picker dropdown
    - Bitrate slider control (500 kbps to 20 Mbps range)
    - _Requirements: 3.1, 3.2, 5.1, 5.2, 6.1, 6.2_

  - [ ]* 3.3 Write property test for bitrate range validation
    - **Property 8: Bitrate Range Validation**
    - **Validates: Requirements 6.2**

  - [ ] 3.4 Create StatsDisplay widget
    - Show current FPS value
    - Show current bitrate in human-readable format (kbps/Mbps)
    - Update at least once per second
    - Hide or show zero when not streaming
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [ ]* 3.5 Write property test for bitrate formatting
    - **Property 9: Stats Display Format**
    - **Validates: Requirements 8.2**

  - [ ] 3.6 Create StreamButton widget
    - Toggle between Start and Stop states
    - Validate server configuration before starting
    - _Requirements: 7.1, 7.4, 7.7_

  - [ ] 3.7 Create main app screen with all widgets
    - Compose CameraPreviewWidget, SettingsPanel, StatsDisplay, StreamButton
    - Wire up StreamRelayController with Provider
    - Handle permission denied and error states
    - _Requirements: 1.6, 1.7_

- [ ] 4. Checkpoint - Ensure Flutter layer compiles
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Implement iOS native plugin foundation
  - [ ] 5.1 Create StreamRelayPlugin.swift with MethodChannel handler
    - Register plugin with Flutter
    - Handle initialize, startStreaming, stopStreaming methods
    - Handle updateResolution, updateBitrate methods
    - Send callbacks to Flutter: onStatsUpdate, onError
    - _Requirements: 9.3_

  - [ ] 5.2 Create iOS data models (Models.swift)
    - StreamingConfig struct
    - StreamingStats struct
    - Resolution struct
    - InitInfo struct
    - StreamingError enum
    - _Requirements: 9.1_

- [ ] 6. Implement iOS camera capture and preview
  - [ ] 6.1 Create StreamingPipeline.swift with AVFoundation camera setup
    - Request camera permission
    - Initialize AVCaptureSession with front camera
    - Configure AVCaptureVideoDataOutput for BGRA format
    - Register texture with FlutterTextureRegistry
    - Implement FlutterTexture protocol for preview
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 9.4_

  - [ ] 6.2 Implement camera frame interception
    - Implement AVCaptureVideoDataOutputSampleBufferDelegate
    - Intercept CMSampleBuffer frames before Dart
    - Update texture registry for preview
    - Pass frames to encoder when streaming
    - _Requirements: 1.3, 9.4_

- [ ] 7. Implement iOS H.264 hardware encoder
  - [ ] 7.1 Create H264Encoder.swift with VideoToolbox
    - Create VTCompressionSession with H.264 codec
    - Configure real-time encoding, baseline profile
    - Set bitrate, frame rate, keyframe interval
    - _Requirements: 2.1, 2.5_

  - [ ] 7.2 Implement frame encoding and Annex B conversion
    - Encode CVPixelBuffer frames
    - Convert AVCC format to Annex B with start codes
    - Extract and prepend SPS/PPS for keyframes
    - _Requirements: 2.3, 2.4_

  - [ ]* 7.3 Write property test for Annex B NALU format
    - **Property 1: Annex B NALU Format**
    - **Validates: Requirements 2.3**

  - [ ]* 7.4 Write property test for keyframe SPS/PPS inclusion
    - **Property 2: Keyframe SPS/PPS Inclusion**
    - **Validates: Requirements 2.4**

  - [ ] 7.5 Implement dynamic bitrate and resolution updates
    - Update bitrate via VTSessionSetProperty
    - Recreate encoder session for resolution changes
    - _Requirements: 6.3, 5.3_

- [ ] 8. Implement iOS WebSocket client
  - [ ] 8.1 Create WebSocketClient.swift with URLSessionWebSocketTask
    - Connect to wss://{host}:{port}/ws
    - Implement URLSessionDelegate for self-signed certificate handling
    - Send text and binary messages
    - Handle connection, disconnection, and errors
    - _Requirements: 3.5, 3.6, 10.1, 10.2_

  - [ ]* 8.2 Write property test for WebSocket URL format
    - **Property 4: WebSocket URL Format**
    - **Validates: Requirements 3.5, 10.2**

  - [ ] 8.3 Implement codec negotiation protocol
    - Send JSON codec message on connection: {"type": "codec", "codec": "h264", "width": W, "height": H}
    - Handle server fallback response
    - Send binary frames for encoded data
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ]* 8.4 Write property test for codec negotiation message format
    - **Property 5: Codec Negotiation Message Format**
    - **Validates: Requirements 4.1, 4.2, 4.3**

  - [ ]* 8.5 Write property test for binary frame transmission
    - **Property 6: Binary Frame Transmission**
    - **Validates: Requirements 4.5**

  - [ ] 8.6 Implement auto-reconnection on connection loss
    - Detect WebSocket disconnection
    - Attempt reconnection after 1 second delay
    - _Requirements: 3.8_

- [ ] 9. Wire iOS streaming pipeline together
  - [ ] 9.1 Integrate encoder and WebSocket in StreamingPipeline
    - Connect camera output to encoder
    - Connect encoder output to WebSocket
    - Implement startStreaming and stopStreaming
    - _Requirements: 7.2, 7.3, 7.5, 7.6, 9.5_

  - [ ] 9.2 Implement statistics collection and reporting
    - Track frame count and bytes sent
    - Calculate FPS and bitrate every second
    - Send stats to Flutter via MethodChannel
    - _Requirements: 8.1, 8.2, 8.3_

  - [ ] 9.3 Implement resolution change during streaming
    - Update encoder resolution
    - Send new codec negotiation message
    - _Requirements: 5.6_

  - [ ]* 9.4 Write property test for resolution change triggers codec negotiation
    - **Property 7: Resolution Change Triggers Codec Negotiation**
    - **Validates: Requirements 5.6**

- [ ] 10. Checkpoint - iOS implementation complete
  - Ensure all tests pass, ask the user if questions arise.
  - Test end-to-end streaming on iOS device/simulator

- [ ] 11. Implement Android native plugin foundation
  - [ ] 11.1 Create StreamRelayPlugin.kt with MethodChannel handler
    - Register plugin with Flutter
    - Handle initialize, startStreaming, stopStreaming methods
    - Handle updateResolution, updateBitrate methods
    - Send callbacks to Flutter: onStatsUpdate, onError
    - _Requirements: 9.2, 9.3_

  - [ ] 11.2 Create Android data models (Models.kt)
    - StreamingConfig data class
    - StreamingStats data class
    - Resolution data class
    - InitInfo data class
    - _Requirements: 9.2_

- [ ] 12. Implement Android camera capture and preview
  - [ ] 12.1 Create StreamingPipeline.kt with CameraX setup
    - Initialize ProcessCameraProvider
    - Configure Preview use case with SurfaceTexture
    - Configure ImageAnalysis for frame capture
    - Register texture with FlutterTextureRegistry
    - _Requirements: 1.2, 1.4, 1.5, 9.4_

  - [ ] 12.2 Implement camera frame interception
    - Set ImageAnalysis analyzer
    - Intercept ImageProxy frames before Dart
    - Pass frames to encoder when streaming
    - _Requirements: 1.4, 9.4_

- [ ] 13. Implement Android H.264 hardware encoder
  - [ ] 13.1 Create H264Encoder.kt with MediaCodec
    - Create MediaCodec encoder for video/avc
    - Configure bitrate, frame rate, color format
    - Set baseline profile and level
    - _Requirements: 2.2, 2.5_

  - [ ] 13.2 Implement frame encoding and Annex B conversion
    - Convert ImageProxy to YUV format
    - Encode frames via MediaCodec
    - Convert to Annex B format with start codes
    - Extract and prepend SPS/PPS for keyframes
    - _Requirements: 2.3, 2.4_

  - [ ] 13.3 Implement dynamic bitrate and resolution updates
    - Update bitrate via MediaCodec.setParameters
    - Recreate encoder for resolution changes
    - _Requirements: 6.3, 5.3_

- [ ] 14. Implement Android WebSocket client
  - [ ] 14.1 Create WebSocketClient.kt with OkHttp
    - Connect to wss://{host}:{port}/ws
    - Configure TrustManager for self-signed certificates
    - Send text and binary messages
    - Handle connection, disconnection, and errors
    - _Requirements: 3.5, 3.6, 10.1, 10.2_

  - [ ] 14.2 Implement codec negotiation and auto-reconnection
    - Send JSON codec message on connection
    - Handle server fallback response
    - Implement auto-reconnection after 1 second
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 3.8_

- [ ] 15. Wire Android streaming pipeline together
  - [ ] 15.1 Integrate encoder and WebSocket in StreamingPipeline
    - Connect camera output to encoder
    - Connect encoder output to WebSocket
    - Implement startStreaming and stopStreaming
    - _Requirements: 7.2, 7.3, 7.5, 7.6, 9.5_

  - [ ] 15.2 Implement statistics collection and reporting
    - Track frame count and bytes sent
    - Calculate FPS and bitrate every second
    - Send stats to Flutter via MethodChannel
    - _Requirements: 8.1, 8.2, 8.3_

  - [ ] 15.3 Implement resolution change during streaming
    - Update encoder resolution
    - Send new codec negotiation message
    - _Requirements: 5.6_

- [ ] 16. Final checkpoint - Full implementation complete
  - Ensure all tests pass, ask the user if questions arise.
  - Test end-to-end streaming on both iOS and Android devices

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- iOS implementation (tasks 5-10) must be completed before Android implementation (tasks 11-15) per Requirement 9.1
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- Native code uses Swift for iOS and Kotlin for Android as specified in the design

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "5.1", "5.2"] },
    { "id": 2, "tasks": ["1.3", "2.1", "6.1"] },
    { "id": 3, "tasks": ["1.4", "2.2", "3.1", "3.2", "6.2"] },
    { "id": 4, "tasks": ["3.3", "3.4", "7.1"] },
    { "id": 5, "tasks": ["3.5", "3.6", "7.2"] },
    { "id": 6, "tasks": ["3.7", "7.3", "7.4", "7.5", "8.1"] },
    { "id": 7, "tasks": ["8.2", "8.3"] },
    { "id": 8, "tasks": ["8.4", "8.5", "8.6"] },
    { "id": 9, "tasks": ["9.1"] },
    { "id": 10, "tasks": ["9.2", "9.3"] },
    { "id": 11, "tasks": ["9.4", "11.1", "11.2"] },
    { "id": 12, "tasks": ["12.1"] },
    { "id": 13, "tasks": ["12.2", "13.1"] },
    { "id": 14, "tasks": ["13.2", "13.3", "14.1"] },
    { "id": 15, "tasks": ["14.2", "15.1"] },
    { "id": 16, "tasks": ["15.2", "15.3"] }
  ]
}
```
