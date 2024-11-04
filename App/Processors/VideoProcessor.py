import threading
import cv2
import queue
from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtGui import QPixmap
from App.Processors.Workers.Frame_Worker import FrameWorker
from App.UI.Widgets import WidgetActions as widget_actions
from typing import TYPE_CHECKING, Dict
import time

if TYPE_CHECKING:
    from App.UI.MainUI import MainWindow

class VideoProcessor(QObject):
    processing_complete = Signal()
    frame_processed_signal = Signal(int, QPixmap)
    single_frame_processed_signal = Signal(int, QPixmap)
    def __init__(self, main_window: 'MainWindow', num_threads=5):
        super().__init__()
        self.main_window = main_window
        self.frame_queue = queue.Queue(maxsize=num_threads)
        self.media_capture = None
        self.file_type = None
        self.processing = False
        self.current_frame_number = 0
        self.max_frame_number = 0
        self.media_path = None
        self.num_threads = num_threads
        self.threads: Dict[int, threading.Thread] = {}

        self.start_time = 0
        self.end_time = 0

        # Timer to manage frame reading intervals
        self.frame_read_timer = QTimer()
        self.frame_read_timer.timeout.connect(self.process_next_frame)

        self.next_frame_to_display = 0
        self.frame_processed_signal.connect(self.store_frame_to_display)
        self.frame_display_timer = QTimer()
        self.frame_display_timer.timeout.connect(self.display_next_frame)
        self.frames_to_display = {}

        self.single_frame_processed_signal.connect(self.display_current_frame)

    Slot(int, QPixmap)
    def store_frame_to_display(self, frame_number, pixmap):
        self.frames_to_display[frame_number] = pixmap

    Slot(int, QPixmap)
    def display_current_frame(self, frame_number, pixmap):
        widget_actions.update_graphics_view(self.main_window, pixmap, frame_number)

    def display_next_frame(self):
        if not self.processing or (self.next_frame_to_display > self.max_frame_number):
            self.stop_processing()
        if self.next_frame_to_display not in self.frames_to_display:
            return
        else:
            pixmap = self.frames_to_display.pop(self.next_frame_to_display)
            widget_actions.update_graphics_view(self.main_window, pixmap, self.next_frame_to_display)
            self.threads.pop(self.next_frame_to_display)
            self.next_frame_to_display += 1

    def set_number_of_threads(self, value):
        self.stop_processing()
        self.main_window.models_processor.set_number_of_threads(value)
        self.num_threads = value
        self.frame_queue = queue.Queue(maxsize=self.num_threads)
        print(f"Max Threads set as {value} ")

    def process_video(self):
        """Start video processing by reading frames and enqueueing them."""
        if self.processing:
            print("Processing already in progress. Ignoring start request.")
            return

        print("Starting video processing.")
        self.start_time = time.time()
        self.processing = True

        if self.file_type == 'video':

            if self.media_capture and self.media_capture.isOpened():
                fps = self.media_capture.get(cv2.CAP_PROP_FPS)
                interval = 1000 / fps if fps > 0 else 30
                print(f"Starting frame_read_timer with an interval of {interval} ms.")
                self.frame_read_timer.start()
                self.frame_display_timer.start()

            else:
                print("Error: Unable to open the video.")
                self.processing = False
                self.frame_read_timer.stop()
                widget_actions.setPlayButtonIconToPlay(self.main_window)

        elif self.file_type == 'image':
            self.process_current_frame()

    def process_next_frame(self):
        """Read the next frame and add it to the queue for processing."""

        if self.current_frame_number > self.max_frame_number:
            print("Stopping frame_read_timer as all frames have been read!")
            self.frame_read_timer.stop()

        if self.frame_queue.qsize() >= self.num_threads:
            print(f"Queue is full ({self.frame_queue.qsize()} frames). Throttling frame reading.")
            return

        if self.file_type == 'video' and self.media_capture:
            ret, frame = self.media_capture.read()
            if ret:
                frame = frame[..., ::-1]  # Convert BGR to RGB
                print(f"Enqueuing frame {self.current_frame_number}")
                self.frame_queue.put((self.current_frame_number, frame))
                self.start_frame_worker(self.current_frame_number, frame)
                self.current_frame_number += 1
            else:
                print("Cannot read frame!.")
                # self.stop_processing()

    def start_frame_worker(self, frame_number, frame, is_single_frame=False):
        """Start a FrameWorker to process the given frame."""
        worker = FrameWorker(frame, self.main_window, frame_number, self.frame_queue, is_single_frame)
        self.threads[frame_number] = worker
        worker.start()

    def process_current_frame(self):

        print("Called process_current_frame()")
        # self.main_window.processed_frames.clear()

        self.next_frame_to_display = self.current_frame_number
        if self.file_type == 'video' and self.media_capture:
            ret, frame = self.media_capture.read()
            if ret:
                frame = frame[..., ::-1]  # Convert BGR to RGB
                print(f"Enqueuing frame {self.current_frame_number}")
                self.frame_queue.put((self.current_frame_number, frame))
                self.start_frame_worker(self.current_frame_number, frame, is_single_frame=True)
                
                self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
            else:
                print("Cannot read frame!")

        """Process a single image frame directly without queuing."""
        if self.file_type == 'image':
            frame = cv2.imread(self.media_path)
            if frame is not None:

                frame = frame[..., ::-1]  # Convert BGR to RGB
                print("Processing current frame as image.")
                self.start_frame_worker(self.current_frame_number, frame, is_single_frame=True)
            else:
                print("Error: Unable to read image file.")

    def stop_processing(self):
        """Stop video processing and signal completion."""
        if not self.processing:
            print("Processing not active. No action to perform.")
            return False

        self.frame_read_timer.stop()
        self.frame_display_timer.stop()


        for ind, thread in self.threads.items():
            thread.join()

        self.end_time = time.time()

        print(f"Processing completed in {self.end_time - self.start_time} seconds")

        self.frames_to_display.clear()

        print("Stopping video processing.")
        self.processing = False
        self.processing_complete.emit()

        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()

        self.current_frame_number = self.main_window.videoSeekSlider.value()
        self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)

        widget_actions.resetMediaButtons(self.main_window)
        return True