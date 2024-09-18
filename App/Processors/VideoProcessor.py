import threading
import cv2
import queue
from PySide6.QtCore import QObject, QTimer, Signal, QThread
from App.Processors.Workers.Frame_Worker import FrameWorker

# Lock for synchronizing thread-safe operations
lock = threading.Lock()

class VideoProcessingWorker(threading.Thread):
    def __init__(self, frame_queue, main_window):
        super().__init__()
        self.frame_queue = frame_queue
        self.main_window = main_window
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                # Get the next frame to process
                current_frame_number, frame = self.frame_queue.get(timeout=1)
                # Process the frame
                worker = FrameWorker(frame, self.main_window, current_frame_number)
                worker.run()  # Instead of starting a new thread, we process it directly
                self.frame_queue.task_done()
            except queue.Empty:
                continue

    def stop(self):
        self._stop_event.set()

class VideoProcessor(QObject):
    processing_complete = Signal()

    def __init__(self, main_window, num_threads=5):
        super().__init__()
        self.main_window = main_window
        self.frame_queue = queue.Queue()  # Queue for storing frames to be processed
        self.threads = []
        self.media_capture = None
        self.file_type = None
        self.processing = False
        self.current_frame_number = 0
        self.max_frame_number = 0
        self.media_path = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_next_frame)
        self.num_threads = num_threads  # Number of threads for processing

    def create_threads(self, threads_count=False):
        """Create and start the worker threads."""
        for _ in range(threads_count or self.num_threads):
            thread = VideoProcessingWorker(self.frame_queue, self.main_window)
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
                
                self.processing = True
                self.create_threads()  # Start the worker threads
                # self.timer.start(5)
                self.timer.start(1000/self.media_capture.get(cv2.CAP_PROP_FPS))  # Timer to control frame reading pace
                
        elif self.file_type == 'image':
            self.processing = True
            self.max_frame_number = 1
            self.create_threads()  # Start worker threads for image processing
            self.timer.start(10)

    def process_next_frame(self):
        """Read the next frame and add it to the queue for processing."""
        if not self.processing:
            return

        if self.file_type == 'video' and self.media_capture:
            if self.current_frame_number >= self.max_frame_number:
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

        self.current_frame_number += 1

    def stop_processing(self):
        """Stop video processing and signal completion."""
        self.processing = False
        self.timer.stop()

        # Stop all worker threads
        for thread in self.threads:
            thread.stop()

        self.threads.clear()

        self.processing_complete.emit()  # Emit signal when processing is complete
        
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
