import threading
import concurrent.futures
import cv2
import queue
import torch
from PySide6.QtCore import QObject, QTimer, Signal

from App.Processors.Workers.Frame_Worker import FrameWorker
from App.UI.Widgets import WidgetActions as widget_actions

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from App.UI.MainUI import MainWindow

class VideoProcessor(QObject):
    processing_complete = Signal()

    def __init__(self, main_window: 'MainWindow', num_threads=5):
        super().__init__()
        self.main_window = main_window
        self.frame_queue = queue.Queue(maxsize=num_threads)  # Queue to limit the number of pending frames
        self.executor = None
        self.media_capture = None
        self.file_type = None
        self.processing = False
        self.current_frame_number = 0
        self.max_frame_number = 0
        self.media_path = None
        self.num_threads = num_threads

        # QTimer managed by the main thread
        self.frame_read_timer = QTimer()
        self.frame_read_timer.timeout.connect(self.process_next_frame)

    def process_video(self):
        """Start video processing by reading frames and enqueuing them."""
        if self.processing:
            print("Processing already in progress. Ignoring start request.")
            return

        print("Starting video processing.")
        self.processing = True

        if self.file_type == 'video':
            self.reset_frame_counter()

            if self.media_capture and self.media_capture.isOpened():
                fps = self.media_capture.get(cv2.CAP_PROP_FPS)
                interval = 1000 / fps if fps > 0 else 30
                print(f"Starting frame_read_timer with an interval of {interval} ms.")

                # Create thread pool limited to num_threads
                self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads)

                # Start the timer to read frames
                self.frame_read_timer.start(interval)
            else:
                print("Error: unable to open video.")
                self.processing = False
                widget_actions.setPlayButtonIconToPlay(self.main_window)

        elif self.file_type == 'image':
            self.process_current_frame(ignore_processing=True)

    def process_next_frame(self):
        """Read the next frame and add it to the queue for processing."""
        if not self.processing:
            return

        # Check if the queue is full and throttle the reading of new frames
        if self.frame_queue.full():
            print(f"Queue full ({self.frame_queue.qsize()} frames). Throttling frame reading.")
            return  # Skip reading a new frame

        if self.file_type == 'video' and self.media_capture:
            if self.current_frame_number > self.max_frame_number:
                self.stop_processing()
                return

            ret, frame = self.media_capture.read()

            if ret:
                # Frame must be in RGB format
                frame = frame[..., ::-1]  # Swap the channels from BGR to RGB

                print(f"Enqueueing frame {self.current_frame_number}")
                try:
                    # Put the frame in the queue and submit the task for processing
                    self.frame_queue.put_nowait((self.current_frame_number, frame))
                    self.executor.submit(self.process_frame)
                except queue.Full:
                    print("Frame queue full, frame discarded.")
            else:
                print(f"Error reading frame at position {self.current_frame_number}")

            self.current_frame_number = int(self.media_capture.get(cv2.CAP_PROP_POS_FRAMES))

    def process_frame(self):
        """Process the current frame by retrieving it from the queue."""
        try:
            frame_number, frame = self.frame_queue.get()
            print(f"Processing frame {frame_number}")
            worker = FrameWorker(frame, self.main_window, frame_number)
            worker.start()

            # Wait for the worker to finish (join waits for the thread to terminate)
            worker.join()  # This blocks until the worker completes
        
            # Mark the task as done
            self.frame_queue.task_done()
        except queue.Empty:
            print("Queue empty, no frame to process.")
        except Exception as e:
            print(f"Error processing frame: {e}")

    def process_current_frame(self, ignore_processing=False):
        """Read and process the current frame immediately."""
        if self.processing and not ignore_processing:
            print("Processing already in progress. Ignoring start request.")
            return

        self.processing = True
        self.reset_frame_counter()

        if self.file_type == 'video' and self.media_capture:
            # Restore the last frame position if necessary
            if self.current_frame_number > self.max_frame_number:
                self.current_frame_number = self.max_frame_number
                self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)

            # Read the current frame
            ret, frame = self.media_capture.read()

            if ret:
                # Frame must be in RGB format
                frame = frame[..., ::-1]  # Swap the channels from BGR to RGB

                print(f"Enqueueing frame {self.current_frame_number}")
                worker = FrameWorker(frame, self.main_window, self.current_frame_number)
                worker.start()

                # Wait for the worker to finish (join waits for the thread to terminate)
                worker.join()  # This blocks until the worker completes

                # After reading, reset the position to avoid advancing
                self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
            else:
                print(f"Error reading frame at position {self.current_frame_number}")

        elif self.file_type == 'image':
            frame = cv2.imread(self.media_path)
            if frame is not None:
                print(f"Enqueueing frame {self.current_frame_number}")
                worker = FrameWorker(frame, self.main_window, self.current_frame_number)
                worker.start()

                # Wait for the worker to finish (join waits for the thread to terminate)
                worker.join()  # This blocks until the worker completes

            else:
                print(f"Error reading image at path {self.media_path}")

        self.processing = False

    def stop_processing(self):
        """Stop video processing and signal completion."""
        if not self.processing:
            print("Processing not active. No action to perform.")
            return False

        print("Stopping video processing.")
        self.processing = False

        # Stop the QTimer only from the main thread
        self.frame_read_timer.stop()

        # Stop all running threads
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

        # Immediately clear the frame queue
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
        print("Frame queue cleared.")

        # Reset multimedia control buttons
        widget_actions.resetMediaButtons(self.main_window)

        self.processing_complete.emit()
        print("Signal processing_complete emitted.")

        torch.cuda.empty_cache()

        return True

    def reset_frame_counter(self):
        self.main_window.reset_frame_counter()