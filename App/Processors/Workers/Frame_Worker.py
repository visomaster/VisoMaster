from PySide6.QtCore import QRunnable,QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor
from PySide6.QtWidgets import QGraphicsPixmapItem
import cv2
import App.UI.Widgets.WidgetActions as widget_actions
import numpy
import torch
from torchvision.transforms import v2
import numpy

class FrameWorker(QRunnable):
    def __init__(self, frame, main_window, current_frame_number):
        super().__init__()
        self.current_frame_number = current_frame_number
        self.frame = frame
        self.main_window = main_window
        self.models_processor = main_window.models_processor
        # self.graphicsViewFrame = graphicsViewFrame

    def run(self):
        self.frame = self.process_swap_on_frame()
        # Convert the frame (which is a NumPy array) to QImage
        scaled_pixmap = widget_actions.get_pixmap_from_frame(self.main_window, self.frame)
        self.main_window.update_frame_signal.emit(self.main_window, scaled_pixmap, self.current_frame_number)

    def process_swap_on_frame(self):
        # Load frame into VRAM
        img = torch.from_numpy(self.frame.astype('uint8')).to('cuda') #HxWxc
        img = img.permute(2,0,1)#cxHxW

        #Scale up frame if it is smaller than 512
        img_x = img.size()[2]
        img_y = img.size()[1]

        det_scale = 1.0
        if img_x<512 and img_y<512:
            # if x is smaller, set x to 512
            if img_x <= img_y:
                new_height = int(512*img_y/img_x)
                tscale = v2.Resize((new_height, 512), antialias=True)
            else:
                new_height = 512
                tscale = v2.Resize((new_height, int(512*img_x/img_y)), antialias=True)

            img = tscale(img)

            det_scale = torch.div(new_height, img_y)

        elif img_x<512:
            new_height = int(512*img_y/img_x)
            tscale = v2.Resize((new_height, 512), antialias=True)
            img = tscale(img)

            det_scale = torch.div(new_height, img_y)

        elif img_y<512:
            new_height = 512
            tscale = v2.Resize((new_height, int(512*img_x/img_y)), antialias=True)
            img = tscale(img)

            det_scale = torch.div(new_height, img_y)
        # print(self.models_processor.run_detect(img))
        img = img.permute(1,2,0)
        img = img.cpu().numpy()
        return numpy.ascontiguousarray(img)