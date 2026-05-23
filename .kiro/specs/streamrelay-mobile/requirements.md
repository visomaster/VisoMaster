# Requirements Document

## Introduction

StreamRelay Mobile is a native Flutter application that streams live camera frames from iOS and Android devices to a StreamRelay server. The app bypasses browser-based limitations by implementing hardware-accelerated H.264 encoding at the native layer, achieving 30-60 fps performance compared to the 22 fps browser limit on iOS. The architecture intercepts raw camera buffers before they reach Dart, encodes them using platform-native hardware encoders (VideoToolbox on iOS, MediaCodec on Android), and transmits H.264 Annex B NALUs over WebSocket to the existing StreamRelay server infrastructure.

## Glossary

- **StreamRelay_Mobile_App**: The Flutter application that captures camera frames and streams them to a StreamRelay server
- **Native_Streaming_Pipeline**: Platform-specific code (Swift/Kotlin) that intercepts camera buffers and performs hardware encoding
- **Camera_Preview_Widget**: The Flutter Texture widget that displays the live camera feed via FlutterTextureRegistry
- **Connection_Manager**: The component responsible for establishing and maintaining WebSocket connections to the server
- **Settings_Persistence**: The component that stores and retrieves user preferences including server connection details
- **Hardware_Encoder**: Platform-native H.264 encoder (VideoToolbox on iOS, MediaCodec on Android)
- **Stats_Display**: The UI component showing real-time streaming statistics
- **CMSampleBuffer**: iOS camera frame buffer format from AVFoundation
- **ImageProxy**: Android camera frame buffer format from CameraX
- **Annex_B_NALU**: H.264 Network Abstraction Layer Unit in Annex B byte stream format with start codes
- **SPS**: Sequence Parameter Set - H.264 metadata describing video stream properties
- **PPS**: Picture Parameter Set - H.264 metadata describing picture encoding parameters
- **FlutterTextureRegistry**: Flutter's mechanism for zero-copy GPU texture sharing with native code

## Requirements

### Requirement 1: Camera Capture and Preview

**User Story:** As a user, I want to see a live preview of my camera feed so that I can verify what is being streamed.

#### Acceptance Criteria

1. WHEN the StreamRelay_Mobile_App launches, THE Camera_Preview_Widget SHALL request camera permission from the operating system.
2. WHEN camera permission is granted, THE Native_Streaming_Pipeline SHALL initialize the platform camera (AVFoundation on iOS, CameraX on Android).
3. WHILE the camera is active, THE Native_Streaming_Pipeline SHALL intercept CMSampleBuffer frames on iOS before they reach Dart.
4. WHILE the camera is active, THE Native_Streaming_Pipeline SHALL intercept ImageProxy frames on Android before they reach Dart.
5. WHILE the camera is active, THE Camera_Preview_Widget SHALL display the live camera feed using FlutterTextureRegistry for zero-copy GPU rendering.
6. IF camera permission is denied, THEN THE StreamRelay_Mobile_App SHALL display an error message explaining that camera access is required.
7. IF the camera fails to initialize, THEN THE StreamRelay_Mobile_App SHALL display an error message with the failure reason.

### Requirement 2: Hardware H.264 Encoding

**User Story:** As a user, I want my camera feed to be efficiently encoded so that streaming uses minimal battery and achieves high frame rates.

#### Acceptance Criteria

1. WHEN streaming is started on iOS, THE Hardware_Encoder SHALL encode frames using VideoToolbox with H.264 codec.
2. WHEN streaming is started on Android, THE Hardware_Encoder SHALL encode frames using MediaCodec with H.264 codec.
3. WHILE encoding, THE Hardware_Encoder SHALL output frames in Annex B NALU format with start codes.
4. WHEN a keyframe is generated, THE Hardware_Encoder SHALL prepend SPS and PPS NALUs to the keyframe data.
5. WHILE streaming, THE Hardware_Encoder SHALL maintain a frame rate between 30 and 60 fps.
6. IF hardware encoding fails, THEN THE StreamRelay_Mobile_App SHALL display an error message indicating encoding failure.

### Requirement 3: Server Connection Management

**User Story:** As a user, I want to connect to my StreamRelay server by entering its address so that I can stream to my own infrastructure.

#### Acceptance Criteria

1. THE StreamRelay_Mobile_App SHALL provide text input fields for server host and port configuration.
2. THE StreamRelay_Mobile_App SHALL default the port field to 9090.
3. WHEN the user enters server connection details, THE Settings_Persistence SHALL save the host and port values to device storage.
4. WHEN the StreamRelay_Mobile_App launches, THE Settings_Persistence SHALL restore the last-used server host and port values.
5. WHEN the user initiates a connection, THE Connection_Manager SHALL establish a WebSocket connection to wss://{host}:{port}/ws.
6. WHEN connecting to a server with a self-signed certificate, THE Connection_Manager SHALL automatically trust the certificate without user intervention.
7. IF the WebSocket connection fails, THEN THE Connection_Manager SHALL display an error message with the connection failure reason.
8. IF the WebSocket connection is lost during streaming, THEN THE Connection_Manager SHALL attempt to reconnect automatically.

### Requirement 4: Codec Negotiation Protocol

**User Story:** As a user, I want the app to automatically negotiate the video format with the server so that streaming works without manual configuration.

#### Acceptance Criteria

1. WHEN a WebSocket connection is established, THE Connection_Manager SHALL send a codec negotiation message in JSON format: {"type": "codec", "codec": "h264", "width": W, "height": H}.
2. THE Connection_Manager SHALL include the actual video width in the codec negotiation message.
3. THE Connection_Manager SHALL include the actual video height in the codec negotiation message.
4. WHEN the server responds with a fallback message, THE StreamRelay_Mobile_App SHALL display a warning that H.264 is not supported by the server.
5. WHILE streaming, THE Native_Streaming_Pipeline SHALL send encoded frames as binary WebSocket messages containing Annex B NALUs.

### Requirement 5: Resolution Selection

**User Story:** As a user, I want to choose my streaming resolution so that I can balance quality and bandwidth usage.

#### Acceptance Criteria

1. THE StreamRelay_Mobile_App SHALL provide a resolution picker UI control.
2. THE StreamRelay_Mobile_App SHALL offer resolution options supported by the device camera.
3. WHEN the user selects a resolution, THE Native_Streaming_Pipeline SHALL configure the camera to capture at the selected resolution.
4. WHEN the user selects a resolution, THE Settings_Persistence SHALL save the selected resolution to device storage.
5. WHEN the StreamRelay_Mobile_App launches, THE Settings_Persistence SHALL restore the last-used resolution setting.
6. WHEN the resolution changes during an active stream, THE Connection_Manager SHALL send an updated codec negotiation message with the new dimensions.

### Requirement 6: Bitrate Control

**User Story:** As a user, I want to adjust the streaming bitrate so that I can optimize for my network conditions.

#### Acceptance Criteria

1. THE StreamRelay_Mobile_App SHALL provide a bitrate control UI element.
2. THE StreamRelay_Mobile_App SHALL allow bitrate selection in a range appropriate for H.264 video streaming.
3. WHEN the user adjusts the bitrate, THE Hardware_Encoder SHALL configure the encoder to target the selected bitrate.
4. WHEN the user adjusts the bitrate, THE Settings_Persistence SHALL save the selected bitrate to device storage.
5. WHEN the StreamRelay_Mobile_App launches, THE Settings_Persistence SHALL restore the last-used bitrate setting.

### Requirement 7: Stream Control

**User Story:** As a user, I want to start and stop streaming with a single button so that I have clear control over when data is transmitted.

#### Acceptance Criteria

1. THE StreamRelay_Mobile_App SHALL provide a Start/Stop streaming button.
2. WHEN the Start button is pressed and no server connection exists, THE Connection_Manager SHALL establish a WebSocket connection before starting the stream.
3. WHEN the Start button is pressed and a server connection exists, THE Native_Streaming_Pipeline SHALL begin encoding and transmitting frames.
4. WHILE streaming is active, THE StreamRelay_Mobile_App SHALL display the button in a Stop state.
5. WHEN the Stop button is pressed, THE Native_Streaming_Pipeline SHALL stop encoding and transmitting frames.
6. WHEN the Stop button is pressed, THE Connection_Manager SHALL close the WebSocket connection.
7. IF streaming cannot start due to missing server configuration, THEN THE StreamRelay_Mobile_App SHALL prompt the user to enter server details.

### Requirement 8: Live Statistics Display

**User Story:** As a user, I want to see real-time streaming statistics so that I can monitor the quality and performance of my stream.

#### Acceptance Criteria

1. WHILE streaming is active, THE Stats_Display SHALL show the current frames per second (fps).
2. WHILE streaming is active, THE Stats_Display SHALL show the current bitrate in a human-readable format.
3. THE Stats_Display SHALL update statistics at least once per second.
4. WHEN streaming is not active, THE Stats_Display SHALL hide or show zero values for statistics.

### Requirement 9: Platform Priority and Architecture

**User Story:** As a developer, I want the iOS implementation completed first so that we can validate the architecture on the primary target platform.

#### Acceptance Criteria

1. THE Native_Streaming_Pipeline SHALL implement iOS support using Swift and AVFoundation before implementing Android support.
2. THE Native_Streaming_Pipeline SHALL implement Android support using Kotlin and CameraX after iOS implementation is complete.
3. THE StreamRelay_Mobile_App SHALL communicate between Flutter and native code exclusively through MethodChannel for control operations.
4. THE Native_Streaming_Pipeline SHALL use FlutterTextureRegistry for passing preview frames to Flutter without copying through Dart.
5. THE Native_Streaming_Pipeline SHALL use native WebSocket clients to send encoded frames directly from the native layer.

### Requirement 10: TLS Certificate Handling

**User Story:** As a user, I want the app to connect to my local server without certificate warnings so that setup is simple.

#### Acceptance Criteria

1. WHEN connecting to a server, THE Connection_Manager SHALL accept self-signed TLS certificates without prompting the user.
2. THE Connection_Manager SHALL use HTTPS (wss://) for all WebSocket connections.
3. IF a certificate validation error occurs for reasons other than self-signing, THEN THE Connection_Manager SHALL display a security warning to the user.
