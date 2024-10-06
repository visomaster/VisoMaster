import cv2
import threading
import queue
from App.UI.Core.MainWindow import Ui_MainWindow
from PySide6 import QtWidgets, QtGui
from PySide6 import QtCore
import App.UI.Widgets.WidgetActions as widget_actions
from functools import partial
from App.Processors.VideoProcessor import VideoProcessor
from App.Processors.ModelsProcessor import ModelsProcessor
from App.UI.Widgets.WidgetComponents import GraphicsViewEventFilter, ParametersWidget, TargetFaceCardButton, InputFaceCardButton, TargetMediaCardButton, EmbeddingCardButton
from App.UI.Widgets.SwapperLayoutData import SWAPPER_LAYOUT_DATA
from App.UI.Widgets.SettingsLayoutData import SETTINGS_LAYOUT_DATA
from typing import Dict, List

class FrameProcessorWorker(QtCore.QObject):
    # Signal to update the UI with the processed frame
    frame_processed = QtCore.Signal(int, QtGui.QPixmap)
    processing_complete = QtCore.Signal()  # Signal when processing all frames is complete

    def __init__(self, frame_queue):
        super().__init__()
        self.queue: queue.Queue = frame_queue  # Queue for storing frames to process
        self._is_processing = threading.Event()  # Event to signal when processing is active
        self.running = True

    def process_frames(self):
        """Process all frames from the queue."""
        if self._is_processing.is_set():
            return  # Avoid re-entering if already processing

        self._is_processing.set()  # Start processing

        while self.running and not self.queue.empty():
            try:
                # Get frame from the queue (this will block if the queue is empty)
                frame_number, pixmap = self.queue.get(timeout=1)
                print(f"Processing frame {frame_number} in worker thread.")

                # Emit signal to update the UI with the processed frame
                self.frame_processed.emit(frame_number, pixmap)

                # Mark the task as done
                self.queue.task_done()
            except queue.Empty:
                break  # If the queue is empty, exit the loop

        self._is_processing.clear()  # Finished processing
        self.processing_complete.emit()  # Emit signal when done

    def stop(self):
        """Stop the worker gracefully."""
        self.running = False
        if self._is_processing.is_set():
            self._is_processing.clear()

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    update_frame_signal = QtCore.Signal(int, QtGui.QPixmap)

    def initialize_variables(self):
        self.video_loader_worker = False
        self.input_faces_loader_worker = False
        self.video_processor = VideoProcessor(self)
        self.models_processor = ModelsProcessor(self)
        self.target_videos: List[TargetMediaCardButton] = [] #Contains button objects of target videos (Set as list instead of single video to support batch processing in future)
        self.target_faces: List[TargetFaceCardButton] = [] #Contains button objects of target faces
        self.input_faces: List[InputFaceCardButton] = [] #Contains button objects of source faces (images)
        self.merged_embeddings: List[EmbeddingCardButton] = []
        self.cur_selected_target_face_button: TargetFaceCardButton = False
        self.selected_video_buttons: List[TargetMediaCardButton] = [] #Contains list of buttons linked to videos/images
        self.selected_target_face_id = 0
        '''
            self.parameters dict have the following structure:
            {
                face_id (int): 
                {
                    parameter_name: parameter_value,
                    ------
                }
                -----
            }
        '''
        self.parameters: Dict[int, Dict[str, bool|int|float|str]] = {} 

        self.default_parameters: Dict[str, bool|int|float|str] = {}
        self.copied_parameters: Dict[str, bool|int|float|str] = {}

        self.parameters_list = {}
        self.control = {}
        self.parameter_widgets: Dict[str, ParametersWidget | QtWidgets.QWidget] = {}
        self.loaded_embedding_filename: str = ''
        self.processed_frames = {}
        self.next_frame_to_display = None  # Index of the next frame to display
        self._is_slider_pressed = threading.Event()

    def initialize_widgets(self):
        # Initialize QListWidget for target media
        self.targetVideosList.setFlow(QtWidgets.QListWidget.LeftToRight)
        self.targetVideosList.setWrapping(True)
        self.targetVideosList.setResizeMode(QtWidgets.QListWidget.Adjust)

        # Initialize QListWidget for face images
        self.inputFacesList.setFlow(QtWidgets.QListWidget.LeftToRight)
        self.inputFacesList.setWrapping(True)
        self.inputFacesList.setResizeMode(QtWidgets.QListWidget.Adjust)

        # Initialize graphics frame to view frames
        self.scene = QtWidgets.QGraphicsScene()
        self.graphicsViewFrame.setScene(self.scene)
        # Event filter to start playing when clicking on frame
        graphics_event_filter = GraphicsViewEventFilter(self, self.graphicsViewFrame,)
        self.graphicsViewFrame.installEventFilter(graphics_event_filter)

        self.buttonSelectTargetVideos.clicked.connect(partial(widget_actions.onClickSelectTargetVideos, self, 'folder'))
        self.buttonSelectTargetVideoFiles.clicked.connect(partial(widget_actions.onClickSelectTargetVideos, self, 'files'))
        self.buttonSelectInputFaces.clicked.connect(partial(widget_actions.onClickSelectInputImages, self, 'folder'))
        self.buttonSelectInputFacesFiles.clicked.connect(partial(widget_actions.onClickSelectInputImages, self, 'files'))

        self.videoSeekSlider.valueChanged.connect(partial(widget_actions.OnChangeSlider, self))
        self.videoSeekSlider.sliderPressed.connect(partial(widget_actions.on_slider_pressed, self))
        self.videoSeekSlider.sliderReleased.connect(partial(widget_actions.on_slider_released, self))

        # Connect the Play/Stop button to the OnClickPlayButton method
        self.buttonMediaPlay.toggled.connect(partial(widget_actions.OnClickPlayButton, self))
        # self.buttonMediaStop.clicked.connect(partial(self.video_processor.stop_processing))
        self.findTargetFacesButton.clicked.connect(partial(widget_actions.find_target_faces, self))
        self.clearTargetFacesButton.clicked.connect(partial(widget_actions.clear_target_faces, self))
        self.targetVideosSearchBox.textChanged.connect(partial(widget_actions.filterTargetVideos, self))
        self.filterImagesCheckBox.clicked.connect(partial(widget_actions.filterTargetVideos, self))
        self.filterVideosCheckBox.clicked.connect(partial(widget_actions.filterTargetVideos, self))
        self.inputFacesSearchBox.textChanged.connect(partial(widget_actions.filterInputFaces, self))
        self.inputEmbeddingsSearchBox.textChanged.connect(partial(widget_actions.filterMergedEmbeddings, self))
        self.openEmbeddingButton.clicked.connect(partial(widget_actions.open_embeddings_from_file, self))
        self.saveEmbeddingButton.clicked.connect(partial(widget_actions.save_embeddings_to_file, self))
        self.saveEmbeddingAsButton.clicked.connect(partial(widget_actions.save_embeddings_to_file, self, True))


        widget_actions.initializeModelLoadDialog(self)
        widget_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=SWAPPER_LAYOUT_DATA, layoutWidget=self.swapWidgetsLayout, data_type='parameter')
        widget_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=SETTINGS_LAYOUT_DATA, layoutWidget=self.settingsWidgetsLayout, data_type='control')

        # Initialize the button states
        widget_actions.resetMediaButtons(self)

        # widget_actions.add_groupbox_and_widgets_from_layout_map(self)
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.initialize_variables()
        self.initialize_widgets()

        # Create an event to signal when processing is complete
        self.processing_finished_event = threading.Event()
        self.processing_finished_event.set()

        # Queue to store frames to process
        self.frame_queue = queue.Queue()

        # Create a QThread for frame processing
        self.frame_processor_thread = QtCore.QThread()
        self.frame_processor_worker = FrameProcessorWorker(self.frame_queue)

        # Move the worker to the thread
        self.frame_processor_worker.moveToThread(self.frame_processor_thread)

        # Connect the signal to enqueue frames for processing
        self.update_frame_signal.connect(self.enqueue_frame)

        # Connect the worker signal to update the UI
        self.frame_processor_worker.frame_processed.connect(self.handle_processed_frame, QtCore.Qt.QueuedConnection)

        # Connect the processing complete signal
        self.frame_processor_worker.processing_complete.connect(self.on_processingcomplete)

        # Start the thread
        self.frame_processor_thread.start()

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        # Call the method to fit the image to the view whenever the window resizes
        if self.scene.items():
            widget_actions.fit_image_to_view(self, self.scene.items()[0])

    def enqueue_frame(self, frame_number, pixmap):
        """Enqueue frame to be processed by the worker."""
        # Clear the event every time new frames are enqueued
        self.processing_finished_event.clear()

        print(f"Enqueueing frame {frame_number} in worker thread.")
        self.frame_queue.put((frame_number, pixmap))

        # Start frame processing in the worker
        self.frame_processor_worker.process_frames()

    @QtCore.Slot(int, QtGui.QPixmap)
    def handle_processed_frame(self, frame_number, pixmap):
        self.processed_frames[frame_number] = pixmap
        self.display_frames_in_order()

    @QtCore.Slot()
    def on_processingcomplete(self):
        """Handle completion of frame processing from the worker."""
        self.processing_finished_event.set()  # Now the entire process is complete
        print("Worker finished processing all frames.")

    def wait_for_processing_to_finish(self):
        """Wait for both processing and displaying of frames to finish using threading.Event."""
        print("Waiting for processing and display of frames to finish...")
        self.processing_finished_event.wait()  # This will block until the event is set
        print("Processing and display of frames complete.")

    def display_frames_in_order(self):
        if self.next_frame_to_display is None:
            if self.processed_frames:
                self.next_frame_to_display = min(self.processed_frames.keys())
            else:
                return  # No frames processed yet

        while self.next_frame_to_display in self.processed_frames:
            pixmap = self.processed_frames.pop(self.next_frame_to_display)
            widget_actions.update_graphics_view(self, pixmap, self.next_frame_to_display)
            self.next_frame_to_display += 1

    def reset_frame_counter(self):
        self.processed_frames.clear()
        self.next_frame_to_display = None

    def closeEvent(self, event):
        print("MainWindow: closeEvent called.")

        # Stop the frame processor worker
        self.frame_processor_worker.stop()

        # Wait for the thread to finish
        self.frame_processor_thread.quit()
        self.frame_processor_thread.wait()

        self.video_processor.stop_processing()
        widget_actions.clear_stop_loading_input_media(self)
        widget_actions.clear_stop_loading_target_media(self)

        # Optionally handle the event if needed
        event.accept()