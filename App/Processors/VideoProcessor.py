from PySide6.QtCore import QRunnable, QThreadPool, Signal, QObject
import cv2
from App.Workers.Frame_Worker import FrameWorker

class VideoProcessingWorker(QRunnable):
    def __init__(self, frame, main_window, current_frame_number):
        super().__init__()
        self.current_frame_number = current_frame_number
        self.frame = frame
        self.main_window = main_window

    def run(self):
        # Process the frame
        runnable = FrameWorker(self.frame, self.main_window, self.current_frame_number)
        self.main_window.thread_pool.start(runnable)

class VideoProcessor(QObject):
    # Signal to indicate when processing is complete
    processing_complete = Signal()

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.thread_pool = QThreadPool()
        # self.thread_pool.setMaxThreadCount(1)  # Adjust as needed
        self.media_capture = None
        self.processing = False
        self.current_frame_number = 0


    def process_video(self):
        if self.processing:
            self.stop_processing()
            return
        if not self.media_capture.isOpened():
            print("Error: Cannot open video")
            return

        self.processing = True
        self.max_frame_number = int(self.media_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.process_next_frame()

    def process_next_frame(self):
        if not self.processing:
            return
        
        if self.current_frame_number >= self.max_frame_number:
            self.stop_processing()
            return
        ret, frame = self.media_capture.read()
        if ret:
            worker = VideoProcessingWorker(frame, self.main_window, self.current_frame_number)
            self.thread_pool.start(worker)

            # Process the next frame after the current frame is processed
            self.current_frame_number += 1
            self.main_window.thread_pool.start(self.process_next_frame)  # Add to QThreadPool to avoid recursion
        else:
            print("Error reading frame")

    def stop_processing(self):
        self.processing = False
        self.thread_pool.waitForDone()  # Wait for all threads to finish
        # if self.media_capture:
        #     self.media_capture.release()
        self.processing_complete.emit()  # Emit signal when processing is complete