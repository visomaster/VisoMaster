# 06 · End-to-end Data Flows

This document traces concrete user actions through the codebase. Each flow lists every function call in order so the corresponding REST/WS endpoints are obvious.

## Flow 1 — Load a target video

```
User clicks "Browse Folder" (buttonTargetVideosPath)
  → list_view_actions.select_target_medias(main_window, 'folder')
        QFileDialog → folder_name
  → list_view_actions.select_target_medias creates TargetMediaLoaderWorker(folder_name=folder_name)
  → worker.thumbnail_ready signal → list_view_actions.add_media_thumbnail_to_target_videos_list
        creates a TargetMediaCardButton, adds to self.targetVideosList
        registers in main_window.target_videos[media_id]

User clicks a card (TargetMediaCardButton.clicked)
  → TargetMediaCardButton.load_media()
        if file_type == 'video':
            self.media_capture = cv2.VideoCapture(media_path)
            video_processor.media_capture = self.media_capture
            video_processor.fps = ...
            video_processor.max_frame_number = ...
            ret, frame = misc_helpers.read_frame(media_capture)
        elif file_type == 'image':
            frame = misc_helpers.read_image_file(media_path)
        elif file_type == 'webcam':
            self.media_capture = cv2.VideoCapture(webcam_index, webcam_backend)
            ...
        elif file_type == 'webrtc':
            shm = SharedMemory(name="visomaster_webrtc_frame", create=False)
            video_processor.webrtc_shm = shm

        common_widget_actions.refresh_frame(main_window)
            → video_processor.process_current_frame() (single-frame preview path)
```

## Flow 2 — Detect faces in current frame

```
User clicks "Find Target Faces" (findTargetFacesButton)
  → card_actions.find_target_faces(main_window)
        Read the current frame from cv2 capture / shm / image file (always BGR)
        frame = frame[..., ::-1]                                 # → RGB
        img = torch.from_numpy(frame).to(device).permute(2,0,1)  # CHW
        if ManualRotationEnableToggle: rotate img

        bboxes, kpss_5, _ = models_processor.run_detect(img, DetectorModelSelection,
                                                       max_num=MaxFacesToDetectSlider,
                                                       score=DetectorScoreSlider/100,
                                                       use_landmark_detection=...,
                                                       rotation_angles=[0,90,180,270] if AutoRotationToggle else [0])

        for face_kps in kpss_5:
            face_emb, cropped_img = models_processor.run_recognize_direct(
                img, face_kps, SimilarityTypeSelection, RecognitionModelSelection)

            # check de-duplication against existing target_faces
            # else compute embeddings for ALL recognition models so future model-switches work
            for option in SETTINGS_LAYOUT_DATA['Face Recognition']['RecognitionModelSelection']['options']:
                target_emb, _ = models_processor.run_recognize_direct(img, face_kps, ..., option)
                embedding_store[option] = target_emb

            list_view_actions.add_media_thumbnail_to_target_faces_list(
                main_window, face_img, embedding_store, pixmap, face_id=uuid.uuid1().int)

        if main_window.target_faces and not selected_target_face_id:
            list(main_window.target_faces.values())[0].click()    # select first
```

## Flow 3 — Add a source face

```
User drags an image into inputFacesList
  → list_view_actions.select_input_face_images(main_window, files_list=[...])
  → InputFacesLoaderWorker.run()
        for file in files_list:
            frame = misc_helpers.read_image_file(file)
            kpss_5 = models_processor.run_detect(frame, ...)[1][0]      # take first face
            for option in RecognitionModelSelection.options:
                emb, _ = run_recognize_direct(img, kpss_5, ..., option)
                embedding_store[option] = emb
            emit thumbnail_ready

  → InputFaceCardButton created and added to inputFacesList

User clicks an input face button while a target face is selected
  → InputFaceCardButton.load_input_face()
        target_face = main_window.target_faces[main_window.selected_target_face_id]
        target_face.assigned_input_faces[self.face_id] = self.embedding_store
        target_face.calculate_assigned_input_embedding()
            # combine assigned_input_faces + assigned_merged_embeddings
            # via mean or median per recognition model
            # store result in target_face.assigned_input_embedding[model] : np.ndarray
        common_widget_actions.refresh_frame(main_window)
```

## Flow 4 — Process a single frame (preview)

```
User clicks "Swap Faces" toggle (swapfacesButton)
  → video_control_actions.process_swap_faces(main_window)
  → video_processor.process_current_frame()
        read frame from media_capture / image file / shm
        frame_queue.put(current_frame_number)
        FrameWorker(frame, main_window, frame_number, frame_queue, is_single_frame=True)
            .run()                                  # synchronous in current thread

FrameWorker.run() → process_frame() (see 04-backend-pipeline.md)
        emit single_frame_processed_signal(frame_number, pixmap, frame)

main thread slot: VideoProcessor.display_current_frame
  → graphics_view_actions.update_graphics_view(main_window, pixmap, frame_number)
  → _send_frame_to_output_window(frame)
```

## Flow 5 — Play & record a video

```
User clicks Play (buttonMediaPlay → toggled True)
  → video_control_actions.play_video(main_window, True)
  → video_processor.process_video()
        recording = False (unless Record was clicked first)
        self.start_time = perf_counter()
        if recording: create_ffmpeg_subprocess()
        compute interval = 1000/fps * 0.8
        frame_read_timer.start(interval)
        frame_display_timer.start()
        gpu_memory_update_timer.start(5000)

(loop)
  frame_read_timer → process_next_frame
       → start_frame_worker(frame_number, frame, is_single_frame=False)
  frame_display_timer → display_next_frame
       → if recording: recording_sp.stdin.write(frame.tobytes())
       → update_graphics_view

User clicks Record (buttonMediaRecord → toggled True)
  → video_control_actions.record_video(main_window, True)
        check OutputMediaFolder set + ffmpeg available
        video_processor.recording = True
        buttonMediaPlay.setChecked(True)            # cascades into play_video

User clicks Stop
  → play_video(False) or record_video(False)
  → video_processor.stop_processing()
        stop timers, join threads
        if recording:
            recording_sp.stdin.close(); wait()
            ffmpeg <temp_output.mp4> + audio from <media_path> → <output_folder>/<basename>_<datetime>.mp4
            os.remove(temp_output.mp4)
            print average FPS
        torch.cuda.empty_cache()
        gc.collect()
        reset_media_buttons
```

## Flow 6 — Webcam mode

```
User clicks Streaming tab → Webcam sub-tab
  → on_input_source_tab_changed(1) + on_streaming_sub_tab_changed(0)
        list_view_actions.clear_stop_loading_target_media_streaming(self, 'webcam')
        list_view_actions.load_target_webcams(main_window)
            TargetMediaLoaderWorker(webcam_mode=True).run()
                for i in range(WebcamMaxNoSelection):
                    cv2.VideoCapture(i, CAMERA_BACKENDS[WebcamBackendSelection])
                    extract a thumbnail
                    emit webcam_thumbnail_ready

User clicks a webcam card
  → TargetMediaCardButton.load_media() (file_type == 'webcam')
        media_capture = cv2.VideoCapture(webcam_index, webcam_backend)
        media_capture.set(CAP_PROP_FRAME_WIDTH, ...)  # from WebcamMaxResSelection

User clicks Play
  → video_processor.process_video() (webcam branch)
        frame_read_timer.timeout = process_next_webcam_frame
        frame_display_timer.timeout = display_next_webcam_frame

(loop)
  process_next_webcam_frame → media_capture.read() → BGR→RGB
       → _apply_streaming_transforms (rotation/flip + FPS)
       → start_frame_worker (file_type='webcam' → emits webcam_frame_processed_signal)
  display_next_webcam_frame → pop from webcam_frames_to_display queue
       → send_frame_to_virtualcam (if enabled)
       → update_graphics_view
```

## Flow 7 — WebRTC mode

```
User clicks Streaming tab → WebRTC sub-tab
  → on_streaming_sub_tab_changed(1)
        list_view_actions.load_target_webrtc(main_window)
            TargetMediaLoaderWorker(webrtc_mode=True).run() → load_webrtc()
                multiprocessing.Process(target=streamrelay.run_server,
                                        kwargs={
                                            http_port: WebRTCHttpPortText (9091),
                                            https_port: WebRTCHttpsPortText (9090),
                                            cert_file: app/ui/external/certificates/cert.pem,
                                            key_file:  app/ui/external/certificates/key.pem,
                                            host: WebRTCBindAddressText (0.0.0.0),
                                            shm_name: 'visomaster_webrtc_frame',
                                        }, daemon=True).start()
                main_window.webrtc_server_process = p
                emit webrtc_thumbnail_ready (placeholder pixmap)

(External device — phone browser or Larix Broadcaster)
  Connects to http://<host>:9091/ (web client) or POSTs WHIP offer to /whip
  StreamServer:
    - Negotiates RTCPeerConnection via aiortc
    - Receives video track, decodes to BGR ndarray
    - Writes [counter|w|h|raw_bgr] into SharedMemory("visomaster_webrtc_frame")

User clicks the WebRTC card
  → TargetMediaCardButton.load_media() (file_type == 'webrtc')
        attach SharedMemory(name="visomaster_webrtc_frame", create=False)
        video_processor.webrtc_shm = shm
        buttonMediaPlay.setChecked(True)            # auto-play

(loop)
  video_processor.process_video() (webrtc branch)
        if shm not yet there → poll every 500ms via _try_attach_webrtc_shm
        once attached → frame_read_timer.timeout = process_next_webrtc_frame
        process_next_webrtc_frame
            counter = shm.buf[0:4]
            if counter same as last → return
            read w/h, copy bytes, BGR → RGB
            apply streaming transforms
            start_frame_worker
```

## Flow 8 — Save / load workspace

```
On close:
  closeEvent → save_load_actions.save_current_workspace(main_window, 'last_workspace.json')
        snapshot all card lists, parameters, control, markers → JSON

On launch:
  load_last_workspace() → if last_workspace.json exists, show LoadLastWorkspaceDialog
  user accepts → save_load_actions.load_saved_workspace(main_window, 'last_workspace.json')
        clear lists, replay TargetMediaLoaderWorker / InputFacesLoaderWorker with stored ids
        restore parameters, embeddings, target faces, markers, control
```

## Flow 9 — Frame markers (per-frame parameter overrides)

```
User clicks "Add Marker" (or presses F)
  → video_control_actions.add_video_slider_marker(main_window)
        position = current videoSeekSlider value
        markers[position] = {'parameters': deepcopy(parameters),
                              'control':    control.copy()}
        videoSeekSlider.add_marker_and_paint(position)

While playing:
  FrameWorker.run() → update_parameters_and_control_from_marker(main_window, frame_number)
        if markers[frame_number]:
            main_window.parameters = deepcopy(markers[frame_number]['parameters'])
            main_window.control.update(markers[frame_number]['control'])

While displaying:
  display_next_frame → update_widget_values_from_markers(main_window, next_frame_to_display)
        if not recording: refresh widget UI to show the marker's values
```

## Flow 10 — Embeddings save / load

```
User clicks "Save Embedding As"
  → save_load_actions.save_embeddings_to_file(main_window, save_as=True)
        QFileDialog → JSON path
        for each EmbeddingCardButton:
            { 'name': embedding_name,
              'embedding_store': { recognition_model: list[float], ... } }
        json.dump(...)

User clicks "Open Embedding"
  → save_load_actions.open_embeddings_from_file(main_window)
        json.load → list of dicts
        for each:
            embedding_store = { model: np.array(values) for model, values in ... }
            list_view_actions.create_and_add_embed_button_to_list(...)
```
