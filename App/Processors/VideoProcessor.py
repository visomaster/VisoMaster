import threading
import cv2
import queue
from functools import partial
from PySide6.QtCore import QObject, QTimer, Signal, QThread
from App.Processors.Workers.Frame_Worker import FrameWorker
from App.UI.Widgets import WidgetActions as widget_actions
# Lock for synchronizing thread-safe operations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from App.UI.MainUI import MainWindow

class VideoProcessingWorker(threading.Thread):
    def __init__(self, frame_queue, main_window):
        super().__init__()
        self.frame_queue = frame_queue
        self.main_window = main_window
        self._running = True

    def run(self):
        while self._running:
            try:
                current_frame_number, frame = self.frame_queue.get(timeout=1)
                print(f"VideoProcessingWorker: Worker has obtained frame {current_frame_number}")

                # Process the frame with FrameWorker
                worker = FrameWorker(frame, self.main_window, current_frame_number)
                worker.start()

                self.frame_queue.task_done()
            except queue.Empty:
                if not self._running:
                    break
                continue
            except Exception as e:
                print(f"Error in worker: {e}")
                self._running = False

    def stop(self):
        self._running = False
        self.join()

class VideoProcessor(QObject):
    processing_complete = Signal()

    def __init__(self, main_window, num_threads=5):
        super().__init__()
        self.main_window = main_window
        self.frame_queue = queue.Queue()
        self.threads = []
        self.media_capture = None
        self.file_type = None
        self.processing = False
        self.current_frame_number = 0
        self.max_frame_number = 0
        self.media_path = None
        self.num_threads = num_threads
        self.allow_frame_display = True  # Flag to track if frame display is allowed

        # QTimer managed by the main thread
        self.frame_read_timer = QTimer()  # This timer must only be started from the UI thread
        self.frame_read_timer.timeout.connect(self.process_next_frame)

    def create_threads(self, threads_count=False):
        """Create and start the worker threads."""
        for _ in range(threads_count or self.num_threads):
            thread = VideoProcessingWorker(self.frame_queue, self.main_window)
            thread.start()
            self.threads.append(thread)
            print("VideoProcessingWorker created and started.")

    def process_video(self):
        """Start video processing by reading frames and enqueueing them."""
        if self.processing:
            print("Processing already in progress. Ignoring start request.")
            return  # Avoid restarting processing if it's already in progress

        print("Starting video processing.")
        self.processing = True

        if self.file_type == 'video':
            self.allow_frame_display = True  # Allow frame display for new processing
            self.reset_frame_counter()

            if self.media_capture and self.media_capture.isOpened():
                # Create the processing threads
                self.create_threads()
                fps = self.media_capture.get(cv2.CAP_PROP_FPS)
                interval = 1000 / fps if fps > 0 else 30
                print(f"Starting frame_read_timer with an interval of {interval} ms.")
                # Ensure the timer is started in the main thread
                self.frame_read_timer.start(interval)
            else:
                print("Error: Unable to open the video.")
                self.processing = False
                widget_actions.setPlayButtonIconToPlay(self.main_window)

        elif self.file_type == 'image':
            self.max_frame_number = 0
            self.process_current_frame(ignore_processing=True)

    def process_next_frame(self):
        """Read the next frame and add it to the queue for processing."""
        if not self.processing:
            return

        if self.file_type == 'video' and self.media_capture:
            if self.current_frame_number >= self.max_frame_number:
                self.stop_processing()
                self.current_frame_number = self.max_frame_number
                return

            ret, frame = self.media_capture.read()

            if not ret:
                print("End of video. Attempting to read from the beginning.")
                self.media_capture.release()
                self.media_capture = cv2.VideoCapture(self.media_path)
                self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart from the first frame position
                self.current_frame_number = 0
                ret, frame = self.media_capture.read()

            if ret:
                print(f"Enqueuing frame {self.current_frame_number}")
                self.frame_queue.put((self.current_frame_number, frame))
            else:
                print(f"Error reading frame at position {self.current_frame_number}")

        self.current_frame_number += 1

    def process_current_frame(self, ignore_processing=False):
        """Read the current frame and process it immediately."""
        if self.processing and not ignore_processing:
            print("Processing already in progress. Ignoring start request.")
            return

        self.processing = True
        self.allow_frame_display = True  # Allow frame display for new processing
        self.reset_frame_counter()

        if self.file_type == 'video' and self.media_capture:
            # restore the last frame position if necessary
            max_frame_number = int(self.media_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.current_frame_number > max_frame_number - 1:
                self.current_frame_number = max_frame_number - 1

            self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)

            # Read the current frame
            ret, frame = self.media_capture.read()

            if not ret:
                print("End of video. Attempting to read from the beginning.")
                self.media_capture.release()
                self.media_capture = cv2.VideoCapture(self.media_path)
                self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart from the first frame position
                self.current_frame_number = 0
                ret, frame = self.media_capture.read()

            if ret:
                print(f"Enqueuing frame {self.current_frame_number}")
                worker = FrameWorker(frame, self.main_window, self.current_frame_number)
                worker.start()

                # After reading, reset the position to avoid advancing
                self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
            else:
                print(f"Error reading frame at position {self.current_frame_number}")

        elif self.file_type == 'image':
            frame = cv2.imread(self.media_path)
            if frame is not None:
                print(f"Enqueuing frame {self.current_frame_number}")
                worker = FrameWorker(frame, self.main_window, self.current_frame_number)
                worker.start()
            else:
                print(f"Error reading image at path {self.media_path}")

        self.processing = False

    def stop_processing(self):
        """Stop video processing and signal completion."""
        if not self.processing:
            print("Processing not active. No action to perform.")
            return

        print("Stopping video processing.")
        self.processing = False
        self.allow_frame_display = False  # Prevent further frame display

        # Stop the QTimer only from the main thread
        self.frame_read_timer.stop()

        # Stop all worker threads
        for thread in self.threads:
            print("Stopping VideoProcessingWorker.")
            thread.stop()
        self.threads.clear()
        print("All VideoProcessingWorkers have been stopped.")

        self.processing_complete.emit()
        print("Signal processing_complete emitted.")

        # Clear the frame queue immediately
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
        print("Frame queue cleared.")

        # Reset the media control buttons
        widget_actions.resetMediaButtons(self.main_window)

    def reset_frame_counter(self):
        self.main_window.reset_frame_counter()