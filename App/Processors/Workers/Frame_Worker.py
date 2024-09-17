from PySide6.QtCore import QRunnable,QTimer, QThread
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor
from PySide6.QtWidgets import QGraphicsPixmapItem
import cv2
import App.UI.Widgets.WidgetActions as widget_actions
import torch
from torchvision.transforms import v2
from skimage import transform as trans
from math import floor, ceil

import numpy as np
from App.Processors.Utils import FaceUtil as faceutil
import threading
lock = threading.Lock()
class FrameWorker(QThread):
    def __init__(self, frame, main_window, current_frame_number):
        super().__init__()
        self.current_frame_number = current_frame_number
        self.frame = frame
        self.main_window = main_window
        self.models_processor = main_window.models_processor
        # self.graphicsViewFrame = graphicsViewFrame

    def run(self):
        self.frame = self.process_swap()
        # Convert the frame (which is a NumPy array) to QImage
        scaled_pixmap = widget_actions.get_pixmap_from_frame(self.main_window, self.frame)
        self.main_window.update_frame_signal.emit(self.main_window, scaled_pixmap, self.current_frame_number)

    def process_swap(self):
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

        bboxes, kpss_5, kpss = self.models_processor.run_detect(img,max_num=2)
        ret = []
        if len(kpss_5)>0:
            for i in range(kpss_5.shape[0]):
                face_kps_5 = kpss_5[i]
                face_kps = kpss[i]
                face_emb, _ = self.models_processor.run_recognize(img, face_kps_5)
                ret.append([face_kps_5, face_kps, face_emb])
        if ret and self.main_window.selected_target_face_buttons and self.main_window.selected_input_face_buttons:
            # Loop through target faces to see if they match our found face embeddings
            for i, fface in enumerate(ret):
                    for target_face in self.main_window.selected_target_face_buttons:
                        sim = self.models_processor.findCosineDistance(fface[2], target_face.embedding)
                        if sim>=60:
                            s_e = self.main_window.selected_input_face_buttons[0].embedding
                            img = self.swap_core(img, fface[0],  s_e=s_e)
        img = img.permute(1,2,0)
        img = img.cpu().numpy()
        return np.ascontiguousarray(img)
    

    def swap_core(self, img, kps_5, kps=False, s_e=[], t_e=[], dfl_model=False): # img = RGB
        swapper_model = 'Inswapper128'
        dst = faceutil.get_arcface_template(image_size=512, mode='arcfacemap')
        M, _ = faceutil.estimate_norm_arcface_template(kps_5, src=dst)
        tform = trans.SimilarityTransform()
        tform.params[0:2] = M

        # Scaling Transforms
        t512 = v2.Resize((512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        t256 = v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        t128 = v2.Resize((128, 128), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)

        # Grab 512 face from image and create 256 and 128 copys
        original_face_512 = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0), interpolation=v2.InterpolationMode.BILINEAR )
        original_face_512 = v2.functional.crop(original_face_512, 0,0, 512, 512)# 3, 512, 512
        original_face_256 = t256(original_face_512)
        original_face_128 = t128(original_face_256)
        if swapper_model == 'Inswapper128':
            latent = torch.from_numpy(self.models_processor.calc_swapper_latent(s_e)).float().to('cuda')
            dim = 1
            input_face_affined = original_face_128

        itex = 1
        output_size = int(128 * dim)
        output = torch.zeros((output_size, output_size, 3), dtype=torch.float32, device='cuda')
        input_face_affined = input_face_affined.permute(1, 2, 0)
        input_face_affined = torch.div(input_face_affined, 255.0)

        if swapper_model == 'Inswapper128':
            with torch.no_grad():  # Disabilita il calcolo del gradiente se è solo per inferenza
                for k in range(itex):
                    for j in range(dim):
                        for i in range(dim):
                            input_face_disc = input_face_affined[j::dim,i::dim]
                            input_face_disc = input_face_disc.permute(2, 0, 1)
                            input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()

                            swapper_output = torch.empty((1,3,128,128), dtype=torch.float32, device='cuda').contiguous()
                            self.models_processor.run_swapper(input_face_disc, latent, swapper_output)

                            swapper_output = torch.squeeze(swapper_output)
                            swapper_output = swapper_output.permute(1, 2, 0)

                            output[j::dim, i::dim] = swapper_output.clone()
                    prev_face = input_face_affined.clone()
                    input_face_affined = output.clone()
                    output = torch.mul(output, 255)
                    output = torch.clamp(output, 0, 255)

        output = output.permute(2, 0, 1)
        swap = t512(output)

        # Cslculate the area to be mergerd back to the original frame
        IM512 = tform.inverse.params[0:2, :]
        corners = np.array([[0,0], [0,511], [511, 0], [511, 511]])

        x = (IM512[0][0]*corners[:,0] + IM512[0][1]*corners[:,1] + IM512[0][2])
        y = (IM512[1][0]*corners[:,0] + IM512[1][1]*corners[:,1] + IM512[1][2])

        left = floor(np.min(x))
        if left<0:
            left=0
        top = floor(np.min(y))
        if top<0:
            top=0
        right = ceil(np.max(x))
        if right>img.shape[2]:
            right=img.shape[2]
        bottom = ceil(np.max(y))
        if bottom>img.shape[1]:
            bottom=img.shape[1]

        # Untransform the swap
        swap = v2.functional.pad(swap, (0,0,img.shape[2]-512, img.shape[1]-512))
        swap = v2.functional.affine(swap, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0,interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )
        swap = swap[0:3, top:bottom, left:right]
        swap = swap.permute(1, 2, 0)

        #Add the cropped areas and place them back into the original image
        swap = swap.type(torch.uint8)
        swap = swap.permute(2,0,1)
        img[0:3, top:bottom, left:right] = swap
        return img