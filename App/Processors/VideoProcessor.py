import threading
import cv2
import queue
from PySide6.QtCore import QObject, QTimer, Signal, QThread
from App.Processors.Workers.Frame_Worker import FrameWorker
from App.UI.Widgets import WidgetActions as widget_actions
# Lock for synchronizing thread-safe operations
lock = threading.Lock()
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from App.UI.MainUI import MainWindow
class VideoProcessingWorker(threading.Thread):
    def __init__(self, frame_queue: queue.Queue, main_window, single_frame=False):
        super().__init__()
        self.frame_queue = frame_queue
        self.main_window = main_window
        self.single_frame = single_frame
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                # Get the next frame to process
                current_frame_number, frame = self.frame_queue.get(timeout=1)
                # Process the frame
                worker = FrameWorker(frame, self.main_window, current_frame_number, self.single_frame)
                worker.run()  # Instead of starting a new thread, we process it directly
                self.frame_queue.task_done()
            except queue.Empty:
                continue

    def stop(self):
        self._stop_event.set()

class VideoProcessor(QObject):
    processing_complete = Signal()

    def __init__(self, main_window: 'MainWindow', num_threads=2):
        super().__init__()
        self.main_window = main_window
        self.frame_queue = queue.Queue()  # Queue for storing frames to be processed
        self.threads = []
        self.media_capture = None
        self.file_type = None
        self.processing = False
        self.current_frame_number = 0 # This is used to store the count of the last frame read and processed from the cv capture object
        self.max_frame_number = 0
        self.media_path = None
        self.frame_read_timer = QTimer()
        self.frame_read_timer.timeout.connect(self.read_next_frame)
        self.current_frame_data = ()
        self.num_threads = num_threads  # Number of threads for processing

        self.processed_frames = []
        self.last_displayed_frame_number = 0 #This is used to store the count of the last frame that was emited and displayed in the graphicsviewframe 
        self.frame_emit_timer = QTimer() #Timer used to display the frames in the periodically according to the video fps
        self.frame_emit_timer.timeout.connect(self.emit_lowest_frame)

    def create_and_run_frame_workers(self, threads_count=False, single_frame=False):
        """Create and start the worker threads."""
        for _ in range(threads_count or self.num_threads):
            thread = VideoProcessingWorker(self.frame_queue, self.main_window, single_frame)
            thread.start()
            self.threads.append(thread)

    def process_video(self):
        """Start video processing by reading frames and enqueueing them."""
        if self.processing:
            self.stop_processing()
            return
        
        if self.file_type == 'video':
            if self.media_capture:
                if not self.media_capture.isOpened():
                    print("Error: Cannot open video")
                    return
                self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
                self.processing = True
                self.create_and_run_frame_workers()  # Start the worker threads
                # self.frame_read_timer.start(5)
                video_fps = self.media_capture.get(cv2.CAP_PROP_FPS)
                self.frame_read_timer.start(20)  # frame_read_timer to control frame reading pace
                self.frame_emit_timer.start(1000/video_fps)
        elif self.file_type == 'image':
            self.processing = True
            self.max_frame_number = 1
            self.create_and_run_frame_workers(threads_count=1)  # Start worker threads for image processing
            self.frame_read_timer.start(10)

    def read_next_frame(self):
        """Read the next frame and add it to the queue for processing."""
        if not self.processing:
            return

        if self.file_type == 'video' and self.media_capture:
            if self.current_frame_number >= self.max_frame_number and self.last_displayed_frame_number >= self.max_frame_number:
                self.stop_processing()
                return

            with lock:
                ret, frame = self.media_capture.read()

                if not ret:
                    self.media_capture.release()
                    self.media_capture = cv2.VideoCapture(self.media_path)
                    ret, frame = self.media_capture.read()

                if ret:
                    print(f"Enqueuing frame {self.current_frame_number}")
                    self.frame_queue.put((self.current_frame_number, frame))
                else:
                    print(f"Error reading frame at position {self.current_frame_number}")

        elif self.file_type == 'image':
            frame = cv2.imread(self.media_path)
            self.frame_queue.put((self.current_frame_number, frame))
            self.processing = False
        self.current_frame_number += 1

    def emit_lowest_frame(self):
        if self.processed_frames:
            lowest_index = 0
            lowest_frame_number = self.processed_frames[0]['frame_number']
            for index, frame_data in enumerate(self.processed_frames):
                if frame_data['frame_number']<lowest_frame_number:
                    lowest_frame_number = frame_data['frame_number']
                    lowest_index=index
            lowest_frame_data = self.processed_frames.pop(lowest_index)
            self.main_window.update_frame_signal.emit( self.main_window, lowest_frame_data['scaled_pixmap'], lowest_frame_data['frame_number'])
            self.last_displayed_frame_number = lowest_frame_data['frame_number']
    def stop_processing(self):
        """Stop video processing and signal completion."""
        self.processing = False
        self.frame_read_timer.stop()
        self.frame_emit_timer.stop()
        self.processed_frames.clear()
        self.current_frame_number = self.last_displayed_frame_number

        # Stop all worker threads
        for thread in self.threads:
            thread.stop()

        self.threads.clear()

        self.processing_complete.emit()  # Emit signal when processing is complete
        
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
        widget_actions.resetMediaButtons(self.main_window)

