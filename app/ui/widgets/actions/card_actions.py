
from typing import TYPE_CHECKING, Dict
import uuid

import numpy
import cv2
import torch
from torchvision.transforms import v2

import app.ui.widgets.actions.common_actions as common_widget_actions
from app.ui.widgets.actions import list_view_actions
import app.helpers.miscellaneous as misc_helpers
from app.ui.widgets.settings_layout_data import SETTINGS_LAYOUT_DATA

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

def clear_target_faces(main_window: 'MainWindow', refresh_frame=True):
    if main_window.video_processor.processing:
        main_window.video_processor.stop_processing()
    main_window.targetFacesList.clear()
    for _, target_face in main_window.target_faces.items():
        target_face.deleteLater()
    main_window.target_faces = {}
    main_window.parameters = {}

    main_window.selected_target_face_id = False
    # Set Parameter widget values to default
    common_widget_actions.set_widgets_values_using_face_id_parameters(main_window=main_window, face_id=False)
    if refresh_frame:
        common_widget_actions.refresh_frame(main_window=main_window)

    
def clear_input_faces(main_window: 'MainWindow'):
    main_window.inputFacesList.clear()
    for _, input_face in main_window.input_faces.items():
        input_face.deleteLater()
    main_window.input_faces = {}

    for _, target_face in main_window.target_faces.items():
        target_face.assigned_input_faces = {}
        target_face.calculate_assigned_input_embedding()
    common_widget_actions.refresh_frame(main_window=main_window)

def clear_merged_embeddings(main_window: 'MainWindow'):
    main_window.inputEmbeddingsList.clear()
    for _, embed_button in main_window.merged_embeddings.items():
        embed_button.deleteLater()
    main_window.merged_embeddings = {}

    for _, target_face in main_window.target_faces.items():
        target_face.assigned_merged_embeddings = {}
        target_face.calculate_assigned_input_embedding()
    common_widget_actions.refresh_frame(main_window=main_window)

def uncheck_all_input_faces(main_window: 'MainWindow'):
    # Uncheck All other input faces. Skip buttons whose underlying C++
    # widget has already been deleted — happens when a previous scan/clear
    # left zombie entries in the dict.
    for face_id in list(main_window.input_faces.keys()):
        input_face_button = main_window.input_faces.get(face_id)
        if input_face_button is None:
            continue
        try:
            input_face_button.setChecked(False)
        except RuntimeError:
            # Underlying C++ widget already deleted — drop the stale entry.
            main_window.input_faces.pop(face_id, None)

def uncheck_all_merged_embeddings(main_window: 'MainWindow'):
    for embedding_id in list(main_window.merged_embeddings.keys()):
        embed_button = main_window.merged_embeddings.get(embedding_id)
        if embed_button is None:
            continue
        try:
            embed_button.setChecked(False)
        except RuntimeError:
            main_window.merged_embeddings.pop(embedding_id, None)

def find_target_faces(main_window: 'MainWindow'):
    control = main_window.control.copy()
    video_processor = main_window.video_processor
    if video_processor.media_path:
        frame = None
        media_capture = video_processor.media_capture

        if video_processor.file_type=='image':
            frame = misc_helpers.read_image_file(video_processor.media_path)
        elif video_processor.file_type=='video' and media_capture:
            # If the play loop is running, grab the last delivered frame
            # instead of touching the capture object from this thread —
            # concurrent reads trigger FFmpeg's async_lock assertion and
            # also race with the play loop's frame counter.
            if video_processor.processing and isinstance(video_processor.current_frame, numpy.ndarray) and video_processor.current_frame.size > 0:
                frame = video_processor.current_frame  # already BGR
            else:
                ret, frame = misc_helpers.read_frame(media_capture)
                if ret:
                    media_capture.set(cv2.CAP_PROP_POS_FRAMES, video_processor.current_frame_number)
                else:
                    frame = None
        elif video_processor.file_type=='webcam' and media_capture:
            # Same story for webcam — prefer the last delivered frame while
            # the loop is reading from the device.
            if video_processor.processing and isinstance(video_processor.current_frame, numpy.ndarray) and video_processor.current_frame.size > 0:
                frame = video_processor.current_frame
            else:
                ret, frame = misc_helpers.read_frame(media_capture)
                if not ret:
                    frame = None
        elif video_processor.file_type=='webrtc' and video_processor.webrtc_shm is not None:
            # Read the latest frame written to shared memory by the WebRTC server
            try:
                from streamrelay.protocol import SHM_HEADER_BYTES
                import struct
                shm = video_processor.webrtc_shm
                w = struct.unpack_from("<I", shm.buf, 4)[0]
                h = struct.unpack_from("<I", shm.buf, 8)[0]
                if w > 0 and h > 0:
                    raw = bytes(shm.buf[SHM_HEADER_BYTES: SHM_HEADER_BYTES + w * h * 3])
                    frame = numpy.frombuffer(raw, dtype=numpy.uint8).reshape((h, w, 3)).copy()
                    # Apply the same streaming transforms (rotation/flip) used during playback
                    # so detection sees the frame as the user does.
                    if hasattr(video_processor, '_apply_streaming_transforms'):
                        # _apply_streaming_transforms expects RGB; convert temporarily
                        frame_rgb = frame[..., ::-1]
                        frame_rgb = video_processor._apply_streaming_transforms(frame_rgb)
                        frame = frame_rgb[..., ::-1].copy()  # back to BGR for the shared post-processing below
                else:
                    print("[WebRTC] No frame available yet for Find Faces.")
            except Exception as e:
                print(f"[WebRTC] Error reading frame for Find Faces: {e}")

        # Final fallback — if nothing produced a frame above, use the last
        # frame the FrameWorker delivered. Covers cases where the capture
        # object was released or the play loop is paused mid-stream.
        if frame is None and isinstance(getattr(video_processor, 'current_frame', None), numpy.ndarray) and video_processor.current_frame.size > 0:
            frame = video_processor.current_frame

        if frame is not None:
        # Frame must be in RGB format
            frame = frame[..., ::-1]  # Swap the channels from BGR to RGB

            # Always re-detect against the current frame so users can press
            # Find Faces again to refresh the face list (e.g. after seeking).
            # Without this clear step, faces that match an existing one within
            # SimilarityThresholdSlider get skipped and the result feels stale.
            clear_target_faces(main_window, refresh_frame=False)

            # print(frame)
            img = torch.from_numpy(frame.astype('uint8')).to(main_window.models_processor.device)
            img = img.permute(2,0,1)
            if control['ManualRotationEnableToggle']:
                img = v2.functional.rotate(img, angle=control['ManualRotationAngleSlider'], interpolation=v2.InterpolationMode.BILINEAR, expand=True)

            _, kpss_5, _ = main_window.models_processor.run_detect(img, control['DetectorModelSelection'], max_num=control['MaxFacesToDetectSlider'], score=control['DetectorScoreSlider']/100.0, input_size=(512, 512), use_landmark_detection=control['LandmarkDetectToggle'], landmark_detect_mode=control['LandmarkDetectModelSelection'], landmark_score=control["LandmarkDetectScoreSlider"]/100.0, from_points=control["DetectFromPointsToggle"], rotation_angles=[0] if not control["AutoRotationToggle"] else [0, 90, 180, 270])

            ret = []
            for face_kps in kpss_5:
                face_emb, cropped_img = main_window.models_processor.run_recognize_direct(img, face_kps, control['SimilarityTypeSelection'], control['RecognitionModelSelection'])
                ret.append([face_kps, face_emb, cropped_img, img])

            if ret:
                # Loop through every detected face. The duplicate-check is
                # gone now that we clear before detecting — every detection
                # produces a fresh card.
                for face in ret:
                    face_img = face[2].cpu().numpy()
                    face_img = face_img[..., ::-1]  # Swap the channels from RGB to BGR
                    face_img = numpy.ascontiguousarray(face_img)
                    pixmap = common_widget_actions.get_pixmap_from_frame(main_window, face_img)

                    embedding_store: Dict[str, numpy.ndarray] = {}
                    # Only embed for the currently selected recognition
                    # model. Other models are loaded on demand when the
                    # user actually switches to them — pre-computing them
                    # here would crash if any model file is missing
                    # (see KeyError: 'GhostArcFace' for users without all
                    # ArcFace variants downloaded).
                    recognition_model = control['RecognitionModelSelection']
                    embedding_store[recognition_model] = face[1]

                    face_id = str(uuid.uuid1().int)

                    list_view_actions.add_media_thumbnail_to_target_faces_list(main_window, face_img, embedding_store, pixmap, face_id)
            # Select the first target face if no target face is already selected
        if main_window.target_faces and not main_window.selected_target_face_id:
            list(main_window.target_faces.values())[0].click()

    # For static media (image/video) the original behavior was to stop playback
    # so the just-detected faces are visible on a frozen frame. For live streams
    # (webcam/webrtc) we want playback to keep flowing — otherwise the user has
    # to manually re-click play after every Find Faces press.
    if main_window.video_processor.processing and \
       main_window.video_processor.file_type not in ('webcam', 'webrtc'):
        main_window.video_processor.stop_processing()
    common_widget_actions.refresh_frame(main_window)

    common_widget_actions.update_gpu_memory_progressbar(main_window)