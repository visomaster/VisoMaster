# Technical Design Document: StreamRelay Mobile

## Overview

StreamRelay Mobile is a native Flutter application that streams live camera frames from iOS and Android devices to a StreamRelay server. The app bypasses browser-based limitations by implementing hardware-accelerated H.264 encoding at the native layer, achieving 30-60 fps performance compared to the 22 fps browser limit on iOS.

The architecture intercepts raw camera buffers before they reach Dart, encodes them using platform-native hardware encoders (VideoToolbox on iOS, MediaCodec on Android), and transmits H.264 Annex B NALUs over WebSocket to the existing StreamRelay server infrastructure.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Flutter Application                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  Camera Preview │  │  Settings UI    │  │  Stats Display              │  │
│  │  (Texture)      │  │  (Host/Port/    │  │  (FPS, Bitrate)             │  │
│  │                 │  │   Resolution/   │  │                             │  │
│  │                 │  │   Bitrate)      │  │                             │  │
│  └────────┬────────┘  └────────┬────────┘  └──────────────┬──────────────┘  │
│           │                    │                          │                  │
│           │         ┌──────────┴──────────┐               │                  │
│           │         │   MethodChannel     │               │                  │
│           │         │   (Control Plane)   │◄──────────────┘                  │
│           │         └──────────┬──────────┘                                  │
└───────────┼────────────────────┼────────────────────────────────────────────┘
            │                    │
┌───────────┼────────────────────┼────────────────────────────────────────────┐
│           │     Native Layer   │                                             │
│           ▼                    ▼                                             │
│  ┌─────────────────┐  ┌─────────────────────────────────────────────────┐   │
│  │ FlutterTexture  │  │           Native Streaming Pipeline              │   │
│  │ Registry        │  │  ┌─────────────┐  ┌──────────────┐  ┌─────────┐ │   │
│  │ (Preview)       │◄─┤  │   Camera    │─▶│   Hardware   │─▶│ Native  │ │   │
│  │                 │  │  │   Capture   │  │   Encoder    │  │ WebSocket│ │   │
│  └─────────────────┘  │  │ (AVFoundation│  │ (VideoToolbox│  │ Client  │ │   │
│                       │  │  / CameraX) │  │  / MediaCodec)│  │         │ │   │
│                       │  └─────────────┘  └──────────────┘  └────┬────┘ │   │
│                       └──────────────────────────────────────────┼──────┘   │
└──────────────────────────────────────────────────────────────────┼──────────┘
                                                                   │
                                                                   ▼
                                                    ┌──────────────────────────┐
                                                    │   StreamRelay Server     │
                                                    │   wss://{host}:{port}/ws │
                                                    └──────────────────────────┘
```


### Key Design Decisions

1. **Forked Camera Plugins**: Fork `camera_avfoundation` (iOS) and `camera_android_camerax` (Android) to intercept camera buffers at the native layer before they reach Dart.

2. **Native Streaming Pipeline**: All frame processing (capture → encode → transmit) happens in native code to avoid Dart overhead and achieve maximum performance.

3. **Hardware Encoding**: Use platform-native hardware encoders (VideoToolbox on iOS, MediaCodec on Android) for efficient H.264 encoding with minimal battery impact.

4. **Native WebSocket Clients**: Use `URLSessionWebSocketTask` (iOS) and OkHttp (Android) to send encoded frames directly from native code, avoiding Dart serialization overhead.

5. **FlutterTextureRegistry for Preview**: Camera preview is rendered via Flutter's texture registry for zero-copy GPU rendering, while the streaming pipeline operates independently.

6. **MethodChannel for Control**: All control operations (start/stop, settings changes) flow through MethodChannel, keeping the control plane in Dart while the data plane stays native.

## Components and Interfaces

### 1. Flutter UI Layer

#### 1.1 CameraPreviewWidget

Displays the live camera feed using a `Texture` widget backed by `FlutterTextureRegistry`.

```dart
class CameraPreviewWidget extends StatefulWidget {
  final int textureId;
  final double aspectRatio;
  
  const CameraPreviewWidget({
    required this.textureId,
    required this.aspectRatio,
    super.key,
  });
  
  @override
  State<CameraPreviewWidget> createState() => _CameraPreviewWidgetState();
}

class _CameraPreviewWidgetState extends State<CameraPreviewWidget> {
  @override
  Widget build(BuildContext context) {
    return AspectRatio(
      aspectRatio: widget.aspectRatio,
      child: Texture(textureId: widget.textureId),
    );
  }
}
```

#### 1.2 SettingsPanel

Provides UI controls for server configuration, resolution, and bitrate.

```dart
class SettingsPanel extends StatefulWidget {
  final StreamRelayController controller;
  
  const SettingsPanel({required this.controller, super.key});
  
  @override
  State<SettingsPanel> createState() => _SettingsPanelState();
}

class _SettingsPanelState extends State<SettingsPanel> {
  final _hostController = TextEditingController();
  final _portController = TextEditingController(text: '9090');
  
  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // Server configuration
        TextField(
          controller: _hostController,
          decoration: const InputDecoration(labelText: 'Server Host'),
          onChanged: (value) => widget.controller.setHost(value),
        ),
        TextField(
          controller: _portController,
          decoration: const InputDecoration(labelText: 'Port'),
          keyboardType: TextInputType.number,
          onChanged: (value) => widget.controller.setPort(int.tryParse(value) ?? 9090),
        ),
        // Resolution picker
        ResolutionPicker(
          resolutions: widget.controller.availableResolutions,
          selected: widget.controller.selectedResolution,
          onChanged: widget.controller.setResolution,
        ),
        // Bitrate slider
        BitrateSlider(
          value: widget.controller.bitrate,
          onChanged: widget.controller.setBitrate,
        ),
      ],
    );
  }
}
```

#### 1.3 StatsDisplay

Shows real-time streaming statistics.

```dart
class StatsDisplay extends StatelessWidget {
  final StreamingStats stats;
  
  const StatsDisplay({required this.stats, super.key});
  
  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: [
        StatItem(label: 'FPS', value: '${stats.fps}'),
        StatItem(label: 'Bitrate', value: _formatBitrate(stats.bitrate)),
      ],
    );
  }
  
  String _formatBitrate(int bitsPerSecond) {
    if (bitsPerSecond >= 1000000) {
      return '${(bitsPerSecond / 1000000).toStringAsFixed(1)} Mbps';
    }
    return '${(bitsPerSecond / 1000).toStringAsFixed(0)} kbps';
  }
}

class StreamingStats {
  final int fps;
  final int bitrate;
  
  const StreamingStats({required this.fps, required this.bitrate});
  
  static const zero = StreamingStats(fps: 0, bitrate: 0);
}
```

#### 1.4 StreamRelayController

Main controller managing the streaming lifecycle via MethodChannel.

```dart
class StreamRelayController extends ChangeNotifier {
  static const _channel = MethodChannel('com.streamrelay.mobile/streaming');
  
  String _host = '';
  int _port = 9090;
  Resolution _resolution = Resolution.hd720;
  int _bitrate = 4000000; // 4 Mbps default
  bool _isStreaming = false;
  int? _textureId;
  StreamingStats _stats = StreamingStats.zero;
  List<Resolution> _availableResolutions = [];
  
  // Getters
  String get host => _host;
  int get port => _port;
  Resolution get selectedResolution => _resolution;
  int get bitrate => _bitrate;
  bool get isStreaming => _isStreaming;
  int? get textureId => _textureId;
  StreamingStats get stats => _stats;
  List<Resolution> get availableResolutions => _availableResolutions;

  StreamRelayController() {
    _channel.setMethodCallHandler(_handleMethodCall);
    _loadSettings();
  }
  
  Future<void> _handleMethodCall(MethodCall call) async {
    switch (call.method) {
      case 'onStatsUpdate':
        _stats = StreamingStats(
          fps: call.arguments['fps'] as int,
          bitrate: call.arguments['bitrate'] as int,
        );
        notifyListeners();
        break;
      case 'onError':
        // Handle error from native layer
        break;
      case 'onConnectionStateChanged':
        // Handle connection state changes
        break;
    }
  }
  
  Future<void> initialize() async {
    final result = await _channel.invokeMethod<Map>('initialize');
    _textureId = result?['textureId'] as int?;
    _availableResolutions = (result?['resolutions'] as List?)
        ?.map((r) => Resolution.fromMap(r as Map))
        .toList() ?? [];
    notifyListeners();
  }
  
  Future<void> startStreaming() async {
    if (_host.isEmpty) {
      throw StreamRelayException('Server host is required');
    }
    await _channel.invokeMethod('startStreaming', {
      'host': _host,
      'port': _port,
      'width': _resolution.width,
      'height': _resolution.height,
      'bitrate': _bitrate,
    });
    _isStreaming = true;
    notifyListeners();
  }
  
  Future<void> stopStreaming() async {
    await _channel.invokeMethod('stopStreaming');
    _isStreaming = false;
    _stats = StreamingStats.zero;
    notifyListeners();
  }
  
  void setHost(String host) {
    _host = host;
    _saveSettings();
    notifyListeners();
  }
  
  void setPort(int port) {
    _port = port;
    _saveSettings();
    notifyListeners();
  }

  Future<void> setResolution(Resolution resolution) async {
    _resolution = resolution;
    _saveSettings();
    if (_isStreaming) {
      await _channel.invokeMethod('updateResolution', {
        'width': resolution.width,
        'height': resolution.height,
      });
    }
    notifyListeners();
  }
  
  Future<void> setBitrate(int bitrate) async {
    _bitrate = bitrate;
    _saveSettings();
    if (_isStreaming) {
      await _channel.invokeMethod('updateBitrate', {'bitrate': bitrate});
    }
    notifyListeners();
  }
  
  Future<void> _loadSettings() async {
    final prefs = await SharedPreferences.getInstance();
    _host = prefs.getString('streamrelay_host') ?? '';
    _port = prefs.getInt('streamrelay_port') ?? 9090;
    _bitrate = prefs.getInt('streamrelay_bitrate') ?? 4000000;
    final resWidth = prefs.getInt('streamrelay_resolution_width');
    final resHeight = prefs.getInt('streamrelay_resolution_height');
    if (resWidth != null && resHeight != null) {
      _resolution = Resolution(width: resWidth, height: resHeight);
    }
    notifyListeners();
  }
  
  Future<void> _saveSettings() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('streamrelay_host', _host);
    await prefs.setInt('streamrelay_port', _port);
    await prefs.setInt('streamrelay_bitrate', _bitrate);
    await prefs.setInt('streamrelay_resolution_width', _resolution.width);
    await prefs.setInt('streamrelay_resolution_height', _resolution.height);
  }
}

class Resolution {
  final int width;
  final int height;
  
  const Resolution({required this.width, required this.height});
  
  static const hd720 = Resolution(width: 1280, height: 720);
  static const hd1080 = Resolution(width: 1920, height: 1080);
  
  factory Resolution.fromMap(Map map) => Resolution(
    width: map['width'] as int,
    height: map['height'] as int,
  );
  
  double get aspectRatio => width / height;
}

class StreamRelayException implements Exception {
  final String message;
  StreamRelayException(this.message);
}
```

### 2. iOS Native Layer (Swift)

#### 2.1 StreamRelayPlugin

Main plugin class handling MethodChannel communication.

```swift
import Flutter
import AVFoundation

public class StreamRelayPlugin: NSObject, FlutterPlugin {
    private var streamingPipeline: StreamingPipeline?
    private var textureRegistry: FlutterTextureRegistry?
    private var methodChannel: FlutterMethodChannel?
    
    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(
            name: "com.streamrelay.mobile/streaming",
            binaryMessenger: registrar.messenger()
        )
        let instance = StreamRelayPlugin()
        instance.textureRegistry = registrar.textures()
        instance.methodChannel = channel
        registrar.addMethodCallDelegate(instance, channel: channel)
    }
    
    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "initialize":
            handleInitialize(result: result)
        case "startStreaming":
            handleStartStreaming(call: call, result: result)
        case "stopStreaming":
            handleStopStreaming(result: result)
        case "updateResolution":
            handleUpdateResolution(call: call, result: result)
        case "updateBitrate":
            handleUpdateBitrate(call: call, result: result)
        default:
            result(FlutterMethodNotImplemented)
        }
    }
    
    private func handleInitialize(result: @escaping FlutterResult) {
        guard let textureRegistry = textureRegistry else {
            result(FlutterError(code: "NO_TEXTURE_REGISTRY", 
                               message: "Texture registry not available", 
                               details: nil))
            return
        }
        
        streamingPipeline = StreamingPipeline(
            textureRegistry: textureRegistry,
            onStats: { [weak self] stats in
                self?.methodChannel?.invokeMethod("onStatsUpdate", arguments: [
                    "fps": stats.fps,
                    "bitrate": stats.bitrate
                ])
            },
            onError: { [weak self] error in
                self?.methodChannel?.invokeMethod("onError", arguments: [
                    "message": error.localizedDescription
                ])
            }
        )

        streamingPipeline?.initialize { [weak self] initResult in
            switch initResult {
            case .success(let info):
                result([
                    "textureId": info.textureId,
                    "resolutions": info.resolutions.map { [
                        "width": $0.width,
                        "height": $0.height
                    ]}
                ])
            case .failure(let error):
                result(FlutterError(code: "INIT_FAILED",
                                   message: error.localizedDescription,
                                   details: nil))
            }
        }
    }
    
    private func handleStartStreaming(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any],
              let host = args["host"] as? String,
              let port = args["port"] as? Int,
              let width = args["width"] as? Int,
              let height = args["height"] as? Int,
              let bitrate = args["bitrate"] as? Int else {
            result(FlutterError(code: "INVALID_ARGS", message: "Missing arguments", details: nil))
            return
        }
        
        let config = StreamingConfig(
            host: host,
            port: port,
            width: width,
            height: height,
            bitrate: bitrate
        )
        
        streamingPipeline?.startStreaming(config: config) { startResult in
            switch startResult {
            case .success:
                result(nil)
            case .failure(let error):
                result(FlutterError(code: "START_FAILED",
                                   message: error.localizedDescription,
                                   details: nil))
            }
        }
    }
    
    private func handleStopStreaming(result: @escaping FlutterResult) {
        streamingPipeline?.stopStreaming()
        result(nil)
    }
    
    private func handleUpdateResolution(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any],
              let width = args["width"] as? Int,
              let height = args["height"] as? Int else {
            result(FlutterError(code: "INVALID_ARGS", message: "Missing arguments", details: nil))
            return
        }
        streamingPipeline?.updateResolution(width: width, height: height)
        result(nil)
    }
    
    private func handleUpdateBitrate(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any],
              let bitrate = args["bitrate"] as? Int else {
            result(FlutterError(code: "INVALID_ARGS", message: "Missing arguments", details: nil))
            return
        }
        streamingPipeline?.updateBitrate(bitrate)
        result(nil)
    }
}
```

#### 2.2 StreamingPipeline (iOS)

Core pipeline managing camera capture, encoding, and transmission.

```swift
import AVFoundation
import VideoToolbox

class StreamingPipeline: NSObject {
    private let textureRegistry: FlutterTextureRegistry
    private let onStats: (StreamingStats) -> Void
    private let onError: (Error) -> Void
    
    private var captureSession: AVCaptureSession?
    private var videoOutput: AVCaptureVideoDataOutput?
    private var textureId: Int64 = -1
    private var pixelBufferRef: CVPixelBuffer?
    
    private var encoder: H264Encoder?
    private var webSocketClient: WebSocketClient?
    private var isStreaming = false
    
    private var frameCount = 0
    private var bytesSent = 0
    private var lastStatsTime = Date()
    
    init(textureRegistry: FlutterTextureRegistry,
         onStats: @escaping (StreamingStats) -> Void,
         onError: @escaping (Error) -> Void) {
        self.textureRegistry = textureRegistry
        self.onStats = onStats
        self.onError = onError
        super.init()
    }
    
    func initialize(completion: @escaping (Result<InitInfo, Error>) -> Void) {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            guard let self = self else { return }
            
            if !granted {
                completion(.failure(StreamingError.cameraPermissionDenied))
                return
            }
            
            do {
                try self.setupCaptureSession()
                let resolutions = self.getAvailableResolutions()
                completion(.success(InitInfo(
                    textureId: self.textureId,
                    resolutions: resolutions
                )))
            } catch {
                completion(.failure(error))
            }
        }
    }

    private func setupCaptureSession() throws {
        let session = AVCaptureSession()
        session.sessionPreset = .hd1280x720
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, 
                                                    for: .video, 
                                                    position: .front) else {
            throw StreamingError.cameraNotAvailable
        }
        
        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else {
            throw StreamingError.cameraConfigurationFailed
        }
        session.addInput(input)
        
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera.capture"))
        
        guard session.canAddOutput(output) else {
            throw StreamingError.cameraConfigurationFailed
        }
        session.addOutput(output)
        
        self.captureSession = session
        self.videoOutput = output
        
        // Register texture with Flutter
        textureId = textureRegistry.register(self)
        
        session.startRunning()
    }
    
    func startStreaming(config: StreamingConfig, 
                        completion: @escaping (Result<Void, Error>) -> Void) {
        // Initialize encoder
        encoder = H264Encoder(
            width: config.width,
            height: config.height,
            bitrate: config.bitrate,
            onEncodedFrame: { [weak self] data, isKeyframe in
                self?.sendEncodedFrame(data: data, isKeyframe: isKeyframe)
            }
        )
        
        // Connect WebSocket
        let urlString = "wss://\(config.host):\(config.port)/ws"
        webSocketClient = WebSocketClient(
            url: urlString,
            onConnected: { [weak self] in
                self?.sendCodecNegotiation(width: config.width, height: config.height)
                self?.isStreaming = true
                self?.startStatsTimer()
                completion(.success(()))
            },
            onDisconnected: { [weak self] in
                self?.handleDisconnection()
            },
            onError: { [weak self] error in
                self?.onError(error)
                completion(.failure(error))
            }
        )
        webSocketClient?.connect()
    }

    private func sendCodecNegotiation(width: Int, height: Int) {
        let message: [String: Any] = [
            "type": "codec",
            "codec": "h264",
            "width": width,
            "height": height
        ]
        if let data = try? JSONSerialization.data(withJSONObject: message) {
            webSocketClient?.sendText(String(data: data, encoding: .utf8) ?? "")
        }
    }
    
    private func sendEncodedFrame(data: Data, isKeyframe: Bool) {
        guard isStreaming else { return }
        webSocketClient?.sendBinary(data)
        frameCount += 1
        bytesSent += data.count
    }
    
    func stopStreaming() {
        isStreaming = false
        webSocketClient?.disconnect()
        webSocketClient = nil
        encoder = nil
        frameCount = 0
        bytesSent = 0
    }
    
    func updateResolution(width: Int, height: Int) {
        encoder?.updateResolution(width: width, height: height)
        if isStreaming {
            sendCodecNegotiation(width: width, height: height)
        }
    }
    
    func updateBitrate(_ bitrate: Int) {
        encoder?.updateBitrate(bitrate)
    }
    
    private func handleDisconnection() {
        // Attempt reconnection
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
            self?.webSocketClient?.connect()
        }
    }
    
    private func startStatsTimer() {
        lastStatsTime = Date()
        frameCount = 0
        bytesSent = 0
        
        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] timer in
            guard let self = self, self.isStreaming else {
                timer.invalidate()
                return
            }
            
            let elapsed = Date().timeIntervalSince(self.lastStatsTime)
            let fps = Int(Double(self.frameCount) / elapsed)
            let bitrate = Int(Double(self.bytesSent * 8) / elapsed)
            
            self.onStats(StreamingStats(fps: fps, bitrate: bitrate))
            
            self.frameCount = 0
            self.bytesSent = 0
            self.lastStatsTime = Date()
        }
    }
    
    private func getAvailableResolutions() -> [Resolution] {
        return [
            Resolution(width: 640, height: 480),
            Resolution(width: 1280, height: 720),
            Resolution(width: 1920, height: 1080)
        ]
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension StreamingPipeline: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // Update texture for preview
        pixelBufferRef = pixelBuffer
        textureRegistry.textureFrameAvailable(textureId)
        
        // Encode frame if streaming
        if isStreaming {
            encoder?.encode(pixelBuffer: pixelBuffer)
        }
    }
}

// MARK: - FlutterTexture
extension StreamingPipeline: FlutterTexture {
    func copyPixelBuffer() -> Unmanaged<CVPixelBuffer>? {
        guard let pixelBuffer = pixelBufferRef else { return nil }
        return Unmanaged.passRetained(pixelBuffer)
    }
}
```

#### 2.3 H264Encoder (iOS - VideoToolbox)

Hardware H.264 encoder using VideoToolbox.

```swift
import VideoToolbox
import CoreMedia

class H264Encoder {
    private var compressionSession: VTCompressionSession?
    private var width: Int
    private var height: Int
    private var bitrate: Int
    private let onEncodedFrame: (Data, Bool) -> Void
    
    private var spsData: Data?
    private var ppsData: Data?
    
    init(width: Int, height: Int, bitrate: Int, 
         onEncodedFrame: @escaping (Data, Bool) -> Void) {
        self.width = width
        self.height = height
        self.bitrate = bitrate
        self.onEncodedFrame = onEncodedFrame
        setupEncoder()
    }

    private func setupEncoder() {
        let encoderCallback: VTCompressionOutputCallback = { 
            outputCallbackRefCon, sourceFrameRefCon, status, infoFlags, sampleBuffer in
            guard let refCon = outputCallbackRefCon,
                  let sampleBuffer = sampleBuffer,
                  status == noErr else { return }
            
            let encoder = Unmanaged<H264Encoder>.fromOpaque(refCon).takeUnretainedValue()
            encoder.handleEncodedFrame(sampleBuffer: sampleBuffer)
        }
        
        var session: VTCompressionSession?
        let status = VTCompressionSessionCreate(
            allocator: kCFAllocatorDefault,
            width: Int32(width),
            height: Int32(height),
            codecType: kCMVideoCodecType_H264,
            encoderSpecification: nil,
            imageBufferAttributes: nil,
            compressedDataAllocator: nil,
            outputCallback: encoderCallback,
            refcon: Unmanaged.passUnretained(self).toOpaque(),
            compressionSessionOut: &session
        )
        
        guard status == noErr, let session = session else { return }
        
        // Configure encoder
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_RealTime, value: kCFBooleanTrue)
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_ProfileLevel, 
                            value: kVTProfileLevel_H264_Baseline_AutoLevel)
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_AverageBitRate, 
                            value: bitrate as CFNumber)
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_MaxKeyFrameInterval, 
                            value: 60 as CFNumber)
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_ExpectedFrameRate, 
                            value: 30 as CFNumber)
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_AllowFrameReordering, 
                            value: kCFBooleanFalse)
        
        VTCompressionSessionPrepareToEncodeFrames(session)
        compressionSession = session
    }

    func encode(pixelBuffer: CVPixelBuffer) {
        guard let session = compressionSession else { return }
        
        let presentationTime = CMTime(value: Int64(CACurrentMediaTime() * 1000), timescale: 1000)
        
        VTCompressionSessionEncodeFrame(
            session,
            imageBuffer: pixelBuffer,
            presentationTimeStamp: presentationTime,
            duration: .invalid,
            frameProperties: nil,
            sourceFrameRefcon: nil,
            infoFlagsOut: nil
        )
    }
    
    private func handleEncodedFrame(sampleBuffer: CMSampleBuffer) {
        guard let dataBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { return }
        
        var length: Int = 0
        var dataPointer: UnsafeMutablePointer<Int8>?
        CMBlockBufferGetDataPointer(dataBuffer, atOffset: 0, lengthAtOffsetOut: nil, 
                                    totalLengthOut: &length, dataPointerOut: &dataPointer)
        
        guard let pointer = dataPointer else { return }
        
        let isKeyframe = isKeyFrame(sampleBuffer: sampleBuffer)
        var outputData = Data()
        
        // For keyframes, prepend SPS and PPS
        if isKeyframe {
            extractParameterSets(sampleBuffer: sampleBuffer)
            if let sps = spsData, let pps = ppsData {
                outputData.append(contentsOf: [0x00, 0x00, 0x00, 0x01])
                outputData.append(sps)
                outputData.append(contentsOf: [0x00, 0x00, 0x00, 0x01])
                outputData.append(pps)
            }
        }
        
        // Convert AVCC to Annex B format
        var offset = 0
        while offset < length - 4 {
            var naluLength: UInt32 = 0
            memcpy(&naluLength, pointer.advanced(by: offset), 4)
            naluLength = CFSwapInt32BigToHost(naluLength)
            
            // Add Annex B start code
            outputData.append(contentsOf: [0x00, 0x00, 0x00, 0x01])
            outputData.append(Data(bytes: pointer.advanced(by: offset + 4), 
                                   count: Int(naluLength)))
            
            offset += 4 + Int(naluLength)
        }
        
        onEncodedFrame(outputData, isKeyframe)
    }

    private func isKeyFrame(sampleBuffer: CMSampleBuffer) -> Bool {
        guard let attachments = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, 
                                                                         createIfNecessary: false) as? [[CFString: Any]],
              let first = attachments.first else { return false }
        return !(first[kCMSampleAttachmentKey_NotSync] as? Bool ?? false)
    }
    
    private func extractParameterSets(sampleBuffer: CMSampleBuffer) {
        guard let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer) else { return }
        
        // Extract SPS
        var spsSize: Int = 0
        var spsCount: Int = 0
        var spsPointer: UnsafePointer<UInt8>?
        CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
            formatDescription, parameterSetIndex: 0, parameterSetPointerOut: &spsPointer,
            parameterSetSizeOut: &spsSize, parameterSetCountOut: &spsCount, nalUnitHeaderLengthOut: nil
        )
        if let spsPointer = spsPointer {
            spsData = Data(bytes: spsPointer, count: spsSize)
        }
        
        // Extract PPS
        var ppsSize: Int = 0
        var ppsPointer: UnsafePointer<UInt8>?
        CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
            formatDescription, parameterSetIndex: 1, parameterSetPointerOut: &ppsPointer,
            parameterSetSizeOut: &ppsSize, parameterSetCountOut: nil, nalUnitHeaderLengthOut: nil
        )
        if let ppsPointer = ppsPointer {
            ppsData = Data(bytes: ppsPointer, count: ppsSize)
        }
    }
    
    func updateResolution(width: Int, height: Int) {
        self.width = width
        self.height = height
        teardown()
        setupEncoder()
    }
    
    func updateBitrate(_ bitrate: Int) {
        self.bitrate = bitrate
        if let session = compressionSession {
            VTSessionSetProperty(session, key: kVTCompressionPropertyKey_AverageBitRate, 
                                value: bitrate as CFNumber)
        }
    }
    
    private func teardown() {
        if let session = compressionSession {
            VTCompressionSessionInvalidate(session)
        }
        compressionSession = nil
    }
    
    deinit {
        teardown()
    }
}
```

#### 2.4 WebSocketClient (iOS)

Native WebSocket client using URLSessionWebSocketTask with self-signed certificate support.

```swift
import Foundation

class WebSocketClient: NSObject {
    private var webSocketTask: URLSessionWebSocketTask?
    private var urlSession: URLSession?
    private let url: String
    private let onConnected: () -> Void
    private let onDisconnected: () -> Void
    private let onError: (Error) -> Void
    
    init(url: String,
         onConnected: @escaping () -> Void,
         onDisconnected: @escaping () -> Void,
         onError: @escaping (Error) -> Void) {
        self.url = url
        self.onConnected = onConnected
        self.onDisconnected = onDisconnected
        self.onError = onError
        super.init()
    }
    
    func connect() {
        guard let url = URL(string: url) else {
            onError(WebSocketError.invalidURL)
            return
        }
        
        // Create session with delegate for self-signed cert handling
        let config = URLSessionConfiguration.default
        urlSession = URLSession(configuration: config, delegate: self, delegateQueue: nil)
        
        webSocketTask = urlSession?.webSocketTask(with: url)
        webSocketTask?.resume()
        
        receiveMessage()
        onConnected()
    }
    
    func disconnect() {
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil
        urlSession?.invalidateAndCancel()
        urlSession = nil
    }
    
    func sendText(_ text: String) {
        let message = URLSessionWebSocketTask.Message.string(text)
        webSocketTask?.send(message) { [weak self] error in
            if let error = error {
                self?.onError(error)
            }
        }
    }
    
    func sendBinary(_ data: Data) {
        let message = URLSessionWebSocketTask.Message.data(data)
        webSocketTask?.send(message) { [weak self] error in
            if let error = error {
                self?.onError(error)
            }
        }
    }

    private func receiveMessage() {
        webSocketTask?.receive { [weak self] result in
            switch result {
            case .success(let message):
                switch message {
                case .string(let text):
                    self?.handleTextMessage(text)
                case .data(let data):
                    self?.handleBinaryMessage(data)
                @unknown default:
                    break
                }
                self?.receiveMessage()
            case .failure(let error):
                self?.onError(error)
                self?.onDisconnected()
            }
        }
    }
    
    private func handleTextMessage(_ text: String) {
        // Handle server messages (e.g., fallback requests)
        if let data = text.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let type = json["type"] as? String,
           type == "fallback" {
            // Server requested fallback - notify error handler
            onError(WebSocketError.serverRequestedFallback)
        }
    }
    
    private func handleBinaryMessage(_ data: Data) {
        // Server doesn't send binary messages to client
    }
}

// MARK: - URLSessionDelegate (Self-signed certificate handling)
extension WebSocketClient: URLSessionDelegate {
    func urlSession(_ session: URLSession, 
                    didReceive challenge: URLAuthenticationChallenge,
                    completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {
        // Accept self-signed certificates
        if challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust,
           let serverTrust = challenge.protectionSpace.serverTrust {
            let credential = URLCredential(trust: serverTrust)
            completionHandler(.useCredential, credential)
        } else {
            completionHandler(.performDefaultHandling, nil)
        }
    }
}

enum WebSocketError: Error {
    case invalidURL
    case connectionFailed
    case serverRequestedFallback
}
```

### 3. Android Native Layer (Kotlin)

#### 3.1 StreamRelayPlugin (Android)

Main plugin class handling MethodChannel communication.

```kotlin
package com.streamrelay.mobile

import android.content.Context
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.view.TextureRegistry

class StreamRelayPlugin : FlutterPlugin, MethodChannel.MethodCallHandler {
    private lateinit var channel: MethodChannel
    private lateinit var context: Context
    private lateinit var textureRegistry: TextureRegistry
    private var streamingPipeline: StreamingPipeline? = null

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel = MethodChannel(binding.binaryMessenger, "com.streamrelay.mobile/streaming")
        channel.setMethodCallHandler(this)
        context = binding.applicationContext
        textureRegistry = binding.textureRegistry
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
        streamingPipeline?.release()
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "initialize" -> handleInitialize(result)
            "startStreaming" -> handleStartStreaming(call, result)
            "stopStreaming" -> handleStopStreaming(result)
            "updateResolution" -> handleUpdateResolution(call, result)
            "updateBitrate" -> handleUpdateBitrate(call, result)
            else -> result.notImplemented()
        }
    }

    private fun handleInitialize(result: MethodChannel.Result) {
        streamingPipeline = StreamingPipeline(
            context = context,
            textureRegistry = textureRegistry,
            onStats = { stats ->
                channel.invokeMethod("onStatsUpdate", mapOf(
                    "fps" to stats.fps,
                    "bitrate" to stats.bitrate
                ))
            },
            onError = { error ->
                channel.invokeMethod("onError", mapOf(
                    "message" to error.message
                ))
            }
        )

        streamingPipeline?.initialize { initResult ->
            initResult.fold(
                onSuccess = { info ->
                    result.success(mapOf(
                        "textureId" to info.textureId,
                        "resolutions" to info.resolutions.map { mapOf(
                            "width" to it.width,
                            "height" to it.height
                        )}
                    ))
                },
                onFailure = { error ->
                    result.error("INIT_FAILED", error.message, null)
                }
            )
        }
    }

    private fun handleStartStreaming(call: MethodCall, result: MethodChannel.Result) {
        val host = call.argument<String>("host") ?: return result.error("INVALID_ARGS", "Missing host", null)
        val port = call.argument<Int>("port") ?: return result.error("INVALID_ARGS", "Missing port", null)
        val width = call.argument<Int>("width") ?: return result.error("INVALID_ARGS", "Missing width", null)
        val height = call.argument<Int>("height") ?: return result.error("INVALID_ARGS", "Missing height", null)
        val bitrate = call.argument<Int>("bitrate") ?: return result.error("INVALID_ARGS", "Missing bitrate", null)

        val config = StreamingConfig(host, port, width, height, bitrate)
        streamingPipeline?.startStreaming(config) { startResult ->
            startResult.fold(
                onSuccess = { result.success(null) },
                onFailure = { error -> result.error("START_FAILED", error.message, null) }
            )
        }
    }

    private fun handleStopStreaming(result: MethodChannel.Result) {
        streamingPipeline?.stopStreaming()
        result.success(null)
    }

    private fun handleUpdateResolution(call: MethodCall, result: MethodChannel.Result) {
        val width = call.argument<Int>("width") ?: return result.error("INVALID_ARGS", "Missing width", null)
        val height = call.argument<Int>("height") ?: return result.error("INVALID_ARGS", "Missing height", null)
        streamingPipeline?.updateResolution(width, height)
        result.success(null)
    }

    private fun handleUpdateBitrate(call: MethodCall, result: MethodChannel.Result) {
        val bitrate = call.argument<Int>("bitrate") ?: return result.error("INVALID_ARGS", "Missing bitrate", null)
        streamingPipeline?.updateBitrate(bitrate)
        result.success(null)
    }
}
```

#### 3.2 StreamingPipeline (Android)

Core pipeline managing camera capture, encoding, and transmission using CameraX.

```kotlin
package com.streamrelay.mobile

import android.content.Context
import android.graphics.SurfaceTexture
import android.util.Size
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import io.flutter.view.TextureRegistry
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.Timer
import kotlin.concurrent.fixedRateTimer

class StreamingPipeline(
    private val context: Context,
    private val textureRegistry: TextureRegistry,
    private val onStats: (StreamingStats) -> Unit,
    private val onError: (Exception) -> Unit
) {
    private var cameraProvider: ProcessCameraProvider? = null
    private var preview: Preview? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var textureEntry: TextureRegistry.SurfaceTextureEntry? = null
    
    private var encoder: H264Encoder? = null
    private var webSocketClient: WebSocketClient? = null
    private var isStreaming = false
    
    private var frameCount = 0
    private var bytesSent = 0
    private var statsTimer: Timer? = null
    
    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    fun initialize(callback: (Result<InitInfo>) -> Unit) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        
        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
                setupCamera()
                
                val resolutions = getAvailableResolutions()
                callback(Result.success(InitInfo(
                    textureId = textureEntry?.id() ?: -1,
                    resolutions = resolutions
                )))
            } catch (e: Exception) {
                callback(Result.failure(e))
            }
        }, ContextCompat.getMainExecutor(context))
    }

    private fun setupCamera() {
        textureEntry = textureRegistry.createSurfaceTexture()
        val surfaceTexture = textureEntry?.surfaceTexture()
        surfaceTexture?.setDefaultBufferSize(1280, 720)
        
        preview = Preview.Builder()
            .setTargetResolution(Size(1280, 720))
            .build()
            .also {
                it.setSurfaceProvider { request ->
                    val surface = android.view.Surface(surfaceTexture)
                    request.provideSurface(surface, cameraExecutor) { }
                }
            }
        
        imageAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(Size(1280, 720))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor) { imageProxy ->
                    processFrame(imageProxy)
                }
            }
        
        val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
        
        cameraProvider?.unbindAll()
        cameraProvider?.bindToLifecycle(
            context as androidx.lifecycle.LifecycleOwner,
            cameraSelector,
            preview,
            imageAnalysis
        )
    }
    
    private fun processFrame(imageProxy: ImageProxy) {
        if (isStreaming) {
            encoder?.encode(imageProxy)
        }
        imageProxy.close()
    }

    fun startStreaming(config: StreamingConfig, callback: (Result<Unit>) -> Unit) {
        encoder = H264Encoder(
            width = config.width,
            height = config.height,
            bitrate = config.bitrate,
            onEncodedFrame = { data, isKeyframe ->
                sendEncodedFrame(data, isKeyframe)
            }
        )
        
        val url = "wss://${config.host}:${config.port}/ws"
        webSocketClient = WebSocketClient(
            url = url,
            onConnected = {
                sendCodecNegotiation(config.width, config.height)
                isStreaming = true
                startStatsTimer()
                callback(Result.success(Unit))
            },
            onDisconnected = { handleDisconnection() },
            onError = { error ->
                onError(error)
                callback(Result.failure(error))
            }
        )
        webSocketClient?.connect()
    }

    private fun sendCodecNegotiation(width: Int, height: Int) {
        val message = """{"type":"codec","codec":"h264","width":$width,"height":$height}"""
        webSocketClient?.sendText(message)
    }
    
    private fun sendEncodedFrame(data: ByteArray, isKeyframe: Boolean) {
        if (!isStreaming) return
        webSocketClient?.sendBinary(data)
        frameCount++
        bytesSent += data.size
    }
    
    fun stopStreaming() {
        isStreaming = false
        statsTimer?.cancel()
        statsTimer = null
        webSocketClient?.disconnect()
        webSocketClient = null
        encoder?.release()
        encoder = null
        frameCount = 0
        bytesSent = 0
    }
    
    fun updateResolution(width: Int, height: Int) {
        encoder?.updateResolution(width, height)
        if (isStreaming) {
            sendCodecNegotiation(width, height)
        }
    }
    
    fun updateBitrate(bitrate: Int) {
        encoder?.updateBitrate(bitrate)
    }
    
    private fun handleDisconnection() {
        // Attempt reconnection after delay
        android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
            webSocketClient?.connect()
        }, 1000)
    }
    
    private fun startStatsTimer() {
        frameCount = 0
        bytesSent = 0
        
        statsTimer = fixedRateTimer(period = 1000L) {
            val fps = frameCount
            val bitrate = bytesSent * 8
            
            onStats(StreamingStats(fps = fps, bitrate = bitrate))
            
            frameCount = 0
            bytesSent = 0
        }
    }
    
    private fun getAvailableResolutions(): List<Resolution> {
        return listOf(
            Resolution(640, 480),
            Resolution(1280, 720),
            Resolution(1920, 1080)
        )
    }
    
    fun release() {
        stopStreaming()
        cameraProvider?.unbindAll()
        textureEntry?.release()
        cameraExecutor.shutdown()
    }
}
```

#### 3.3 H264Encoder (Android - MediaCodec)

Hardware H.264 encoder using MediaCodec.

```kotlin
package com.streamrelay.mobile

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer

class H264Encoder(
    private var width: Int,
    private var height: Int,
    private var bitrate: Int,
    private val onEncodedFrame: (ByteArray, Boolean) -> Unit
) {
    private var mediaCodec: MediaCodec? = null
    private var spsData: ByteArray? = null
    private var ppsData: ByteArray? = null
    
    private val startCode = byteArrayOf(0x00, 0x00, 0x00, 0x01)
    
    init {
        setupEncoder()
    }
    
    private fun setupEncoder() {
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height).apply {
            setInteger(MediaFormat.KEY_BIT_RATE, bitrate)
            setInteger(MediaFormat.KEY_FRAME_RATE, 30)
            setInteger(MediaFormat.KEY_COLOR_FORMAT, 
                       MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 2)
            setInteger(MediaFormat.KEY_PROFILE, MediaCodecInfo.CodecProfileLevel.AVCProfileBaseline)
            setInteger(MediaFormat.KEY_LEVEL, MediaCodecInfo.CodecProfileLevel.AVCLevel31)
        }
        
        mediaCodec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC).apply {
            setCallback(object : MediaCodec.Callback() {
                override fun onInputBufferAvailable(codec: MediaCodec, index: Int) {
                    // Input handled in encode()
                }
                
                override fun onOutputBufferAvailable(codec: MediaCodec, index: Int, 
                                                      info: MediaCodec.BufferInfo) {
                    handleEncodedFrame(codec, index, info)
                }
                
                override fun onError(codec: MediaCodec, e: MediaCodec.CodecException) {
                    // Handle error
                }
                
                override fun onOutputFormatChanged(codec: MediaCodec, format: MediaFormat) {
                    extractParameterSets(format)
                }
            })
            configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            start()
        }
    }

    fun encode(imageProxy: ImageProxy) {
        val codec = mediaCodec ?: return
        
        val inputBufferIndex = codec.dequeueInputBuffer(0)
        if (inputBufferIndex >= 0) {
            val inputBuffer = codec.getInputBuffer(inputBufferIndex) ?: return
            
            // Convert ImageProxy to YUV and copy to input buffer
            val yuvData = imageProxyToYuv(imageProxy)
            inputBuffer.clear()
            inputBuffer.put(yuvData)
            
            val presentationTimeUs = System.nanoTime() / 1000
            codec.queueInputBuffer(inputBufferIndex, 0, yuvData.size, presentationTimeUs, 0)
        }
    }
    
    private fun imageProxyToYuv(imageProxy: ImageProxy): ByteArray {
        val yPlane = imageProxy.planes[0]
        val uPlane = imageProxy.planes[1]
        val vPlane = imageProxy.planes[2]
        
        val ySize = yPlane.buffer.remaining()
        val uSize = uPlane.buffer.remaining()
        val vSize = vPlane.buffer.remaining()
        
        val nv21 = ByteArray(ySize + uSize + vSize)
        
        yPlane.buffer.get(nv21, 0, ySize)
        vPlane.buffer.get(nv21, ySize, vSize)
        uPlane.buffer.get(nv21, ySize + vSize, uSize)
        
        return nv21
    }
    
    private fun handleEncodedFrame(codec: MediaCodec, index: Int, info: MediaCodec.BufferInfo) {
        val outputBuffer = codec.getOutputBuffer(index) ?: return
        
        val isKeyframe = (info.flags and MediaCodec.BUFFER_FLAG_KEY_FRAME) != 0
        val outputData = mutableListOf<Byte>()
        
        // For keyframes, prepend SPS and PPS
        if (isKeyframe) {
            spsData?.let { sps ->
                outputData.addAll(startCode.toList())
                outputData.addAll(sps.toList())
            }
            ppsData?.let { pps ->
                outputData.addAll(startCode.toList())
                outputData.addAll(pps.toList())
            }
        }
        
        // Convert AVCC to Annex B format
        outputBuffer.position(info.offset)
        outputBuffer.limit(info.offset + info.size)
        
        while (outputBuffer.remaining() >= 4) {
            val naluLength = outputBuffer.int
            if (naluLength > 0 && outputBuffer.remaining() >= naluLength) {
                outputData.addAll(startCode.toList())
                val naluData = ByteArray(naluLength)
                outputBuffer.get(naluData)
                outputData.addAll(naluData.toList())
            }
        }
        
        codec.releaseOutputBuffer(index, false)
        onEncodedFrame(outputData.toByteArray(), isKeyframe)
    }

    private fun extractParameterSets(format: MediaFormat) {
        format.getByteBuffer("csd-0")?.let { spsBuffer ->
            spsData = ByteArray(spsBuffer.remaining()).also { spsBuffer.get(it) }
            // Remove start code if present
            if (spsData?.take(4) == startCode.toList()) {
                spsData = spsData?.drop(4)?.toByteArray()
            }
        }
        
        format.getByteBuffer("csd-1")?.let { ppsBuffer ->
            ppsData = ByteArray(ppsBuffer.remaining()).also { ppsBuffer.get(it) }
            // Remove start code if present
            if (ppsData?.take(4) == startCode.toList()) {
                ppsData = ppsData?.drop(4)?.toByteArray()
            }
        }
    }
    
    fun updateResolution(width: Int, height: Int) {
        this.width = width
        this.height = height
        release()
        setupEncoder()
    }
    
    fun updateBitrate(bitrate: Int) {
        this.bitrate = bitrate
        mediaCodec?.setParameters(android.os.Bundle().apply {
            putInt(MediaCodec.PARAMETER_KEY_VIDEO_BITRATE, bitrate)
        })
    }
    
    fun release() {
        mediaCodec?.stop()
        mediaCodec?.release()
        mediaCodec = null
    }
}
```

#### 3.4 WebSocketClient (Android - OkHttp)

Native WebSocket client using OkHttp with self-signed certificate support.

```kotlin
package com.streamrelay.mobile

import okhttp3.*
import okio.ByteString
import okio.ByteString.Companion.toByteString
import java.security.cert.X509Certificate
import javax.net.ssl.*

class WebSocketClient(
    private val url: String,
    private val onConnected: () -> Unit,
    private val onDisconnected: () -> Unit,
    private val onError: (Exception) -> Unit
) {
    private var webSocket: WebSocket? = null
    private var client: OkHttpClient? = null

    fun connect() {
        val trustAllCerts = arrayOf<TrustManager>(object : X509TrustManager {
            override fun checkClientTrusted(chain: Array<X509Certificate>, authType: String) {}
            override fun checkServerTrusted(chain: Array<X509Certificate>, authType: String) {}
            override fun getAcceptedIssuers(): Array<X509Certificate> = arrayOf()
        })
        
        val sslContext = SSLContext.getInstance("TLS").apply {
            init(null, trustAllCerts, java.security.SecureRandom())
        }
        
        client = OkHttpClient.Builder()
            .sslSocketFactory(sslContext.socketFactory, trustAllCerts[0] as X509TrustManager)
            .hostnameVerifier { _, _ -> true }
            .build()
        
        val request = Request.Builder().url(url).build()
        
        webSocket = client?.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) { onConnected() }
            override fun onMessage(webSocket: WebSocket, text: String) { handleTextMessage(text) }
            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                webSocket.close(1000, null)
                onDisconnected()
            }
            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                onError(Exception(t.message))
            }
        })
    }
    
    fun disconnect() {
        webSocket?.close(1000, "Client disconnected")
        webSocket = null
        client?.dispatcher?.executorService?.shutdown()
        client = null
    }
    
    fun sendText(text: String) { webSocket?.send(text) }
    fun sendBinary(data: ByteArray) { webSocket?.send(data.toByteString()) }
    
    private fun handleTextMessage(text: String) {
        if (text.contains("\"type\":\"fallback\"")) {
            onError(Exception("Server requested fallback to JPEG"))
        }
    }
}
```

## Data Models

```kotlin
// Kotlin
data class StreamingConfig(val host: String, val port: Int, val width: Int, val height: Int, val bitrate: Int)
data class StreamingStats(val fps: Int, val bitrate: Int)
data class Resolution(val width: Int, val height: Int)
data class InitInfo(val textureId: Long, val resolutions: List<Resolution>)
```

```swift
// Swift
struct StreamingConfig { let host: String; let port: Int; let width: Int; let height: Int; let bitrate: Int }
struct StreamingStats { let fps: Int; let bitrate: Int }
struct Resolution { let width: Int; let height: Int }
struct InitInfo { let textureId: Int64; let resolutions: [Resolution] }
enum StreamingError: Error { case cameraPermissionDenied, cameraNotAvailable, cameraConfigurationFailed, encodingFailed }
```

## Error Handling

| Error Type | Cause | User Message | Recovery |
|------------|-------|--------------|----------|
| Camera Permission Denied | User denied camera access | "Camera access is required to stream video" | Prompt to open settings |
| Camera Not Available | No camera found | "No camera available on this device" | None |
| Camera Configuration Failed | Camera setup error | "Failed to initialize camera" | Retry initialization |
| Encoding Failed | Hardware encoder error | "Video encoding failed" | Stop streaming, show error |
| Connection Failed | WebSocket connection error | "Could not connect to server: {reason}" | Retry connection |
| Connection Lost | WebSocket disconnected | "Connection lost, reconnecting..." | Auto-reconnect |
| Server Fallback | Server doesn't support H.264 | "Server does not support H.264 encoding" | Show warning |

## Correctness Properties

*Properties define universal behaviors that should hold across all valid inputs.*


### Property 1: Annex B NALU Format

*For any* encoded H.264 frame output by the Hardware_Encoder, the frame data SHALL contain valid Annex B start codes (0x00 0x00 0x00 0x01) preceding each NALU.

**Validates: Requirements 2.3**

### Property 2: Keyframe SPS/PPS Inclusion

*For any* keyframe generated by the Hardware_Encoder, the frame data SHALL be prepended with SPS and PPS NALUs, each preceded by an Annex B start code.

**Validates: Requirements 2.4**

### Property 3: Settings Persistence Round-Trip

*For any* valid settings values (host, port, resolution, bitrate), saving the settings and then loading them SHALL return identical values.

**Validates: Requirements 3.3, 3.4, 5.4, 5.5, 6.4, 6.5**

### Property 4: WebSocket URL Format

*For any* valid host and port combination, the Connection_Manager SHALL construct a WebSocket URL in the format `wss://{host}:{port}/ws`.

**Validates: Requirements 3.5, 10.2**

### Property 5: Codec Negotiation Message Format

*For any* valid width and height values, the codec negotiation message SHALL be valid JSON containing exactly the fields: type="codec", codec="h264", width={actual_width}, height={actual_height}.

**Validates: Requirements 4.1, 4.2, 4.3**

### Property 6: Binary Frame Transmission

*For any* encoded frame, the Native_Streaming_Pipeline SHALL transmit it as a binary WebSocket message (not text).

**Validates: Requirements 4.5**

### Property 7: Resolution Change Triggers Codec Negotiation

*For any* resolution change during an active stream, the Connection_Manager SHALL send a new codec negotiation message with the updated dimensions.

**Validates: Requirements 5.6**

### Property 8: Bitrate Range Validation

*For any* bitrate value, the StreamRelay_Mobile_App SHALL accept values within the range 500,000 to 20,000,000 bps and reject values outside this range.

**Validates: Requirements 6.2**

### Property 9: Stats Display Format

*For any* bitrate value in bits per second, the Stats_Display SHALL format it as "{value} kbps" for values < 1,000,000 or "{value} Mbps" for values >= 1,000,000.

**Validates: Requirements 8.2**


## Testing Strategy

### Unit Tests
- Settings persistence (save/load round-trip)
- Bitrate formatting logic
- URL construction
- Codec negotiation message generation
- Annex B conversion logic

### Integration Tests
- Camera initialization and permission flow
- WebSocket connection with self-signed certificates
- End-to-end streaming to local server
- Resolution change during streaming
- Bitrate change during streaming
- Reconnection after connection loss

### Property-Based Tests
- Annex B format validation across random frame data
- Settings persistence round-trip with random values
- Codec negotiation message format with random dimensions
- Bitrate formatting with random values

## Project Structure

```
streamrelay_mobile/
├── lib/
│   ├── main.dart
│   ├── src/
│   │   ├── controller/
│   │   │   └── stream_relay_controller.dart
│   │   ├── models/
│   │   │   ├── resolution.dart
│   │   │   ├── streaming_stats.dart
│   │   │   └── streaming_config.dart
│   │   └── widgets/
│   │       ├── camera_preview_widget.dart
│   │       ├── settings_panel.dart
│   │       ├── stats_display.dart
│   │       └── stream_button.dart
│   └── streamrelay_mobile.dart
├── ios/
│   └── Classes/
│       ├── StreamRelayPlugin.swift
│       ├── StreamingPipeline.swift
│       ├── H264Encoder.swift
│       ├── WebSocketClient.swift
│       └── Models.swift
├── android/
│   └── src/main/kotlin/com/streamrelay/mobile/
│       ├── StreamRelayPlugin.kt
│       ├── StreamingPipeline.kt
│       ├── H264Encoder.kt
│       ├── WebSocketClient.kt
│       └── Models.kt
├── test/
│   ├── controller_test.dart
│   ├── models_test.dart
│   └── widgets_test.dart
└── pubspec.yaml
```

## Dependencies

### Flutter (pubspec.yaml)
```yaml
dependencies:
  flutter:
    sdk: flutter
  shared_preferences: ^2.2.0
  provider: ^6.1.0

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^3.0.0
```

### iOS (Podfile)
- AVFoundation (system framework)
- VideoToolbox (system framework)

### Android (build.gradle)
```gradle
dependencies {
    implementation 'androidx.camera:camera-core:1.3.0'
    implementation 'androidx.camera:camera-camera2:1.3.0'
    implementation 'androidx.camera:camera-lifecycle:1.3.0'
    implementation 'com.squareup.okhttp3:okhttp:4.12.0'
}
```
