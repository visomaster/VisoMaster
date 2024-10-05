from PySide6.QtCore import QRunnable,QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor
from PySide6.QtWidgets import QGraphicsPixmapItem
import cv2
import App.UI.Widgets.WidgetActions as widget_actions
import torch
from torchvision.transforms import v2
from skimage import transform as trans
from math import floor, ceil

import torchvision
from torchvision import transforms
torchvision.disable_beta_transforms_warning()

import numpy as np
from App.Processors.Utils import FaceUtil as faceutil
import threading

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from App.UI.MainUI import MainWindow

class FrameWorker(threading.Thread):
    def __init__(self, frame, main_window: 'MainWindow', frame_number):
        super().__init__()
        self.frame = frame
        self.main_window = main_window
        self.frame_number = frame_number
        self.models_processor = main_window.models_processor
        # self.graphicsViewFrame = graphicsViewFrame

    def run(self):
        try:
            self.parameters = self.main_window.parameters.copy()
            # Process the frame with model inference
            self.frame = self.process_swap()

            # Check if processing is still allowed before displaying the frame
            print(f"Displaying frame {self.frame_number}")
            # Convert the frame (which is a NumPy array) to QImage
            pixmap = widget_actions.get_pixmap_from_frame(self.main_window, self.frame)
            self.main_window.update_frame_signal.emit(self.frame_number, pixmap)

        except Exception as e:
            print(f"Error in FrameWorker: {e}")

    def process_swap(self):
        parameters = self.parameters
        # Load frame into VRAM
        img = torch.from_numpy(self.frame.astype('uint8')).to(self.models_processor.device) #HxWxc
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

        bboxes, kpss_5, kpss = self.models_processor.run_detect(img, parameters['DetectorModelSelection'], max_num=20, score=parameters['DetectorScoreSlider']/100.0, use_landmark_detection=parameters['LandmarkDetectToggle'], landmark_detect_mode=parameters['LandmarkDetectModelSelection'], landmark_score=parameters["LandmarkDetectScoreSlider"]/100.0, from_points=parameters["DetectFromPointsToggle"], rotation_angles=[0] if not parameters["AutoRotationToggle"] else [0, 90, 180, 270])
        ret = []
        if len(kpss_5)>0:
            for i in range(kpss_5.shape[0]):
                face_kps_5 = kpss_5[i]
                face_kps = kpss[i]
                face_emb, _ = self.models_processor.run_recognize(img, face_kps_5)
                ret.append([face_kps_5, face_kps, face_emb])
        if ret:
            # Loop through target faces to see if they match our found face embeddings
            for i, fface in enumerate(ret):
                    for target_face in self.main_window.target_faces:
                        sim = self.models_processor.findCosineDistance(fface[2], target_face.embedding)
                        if sim>=parameters['SimilarityThresholdSlider']:
                            s_e = target_face.assigned_input_embedding
                            img = self.swap_core(img, fface[0], s_e=s_e, t_e=fface[2])
        img = img.permute(1,2,0)
        img = img.cpu().numpy()
        # Img must be in BGR format
        img = img[..., ::-1]  # Swap the channels from RGB to BGR
        return np.ascontiguousarray(img)

    def swap_core(self, img, kps_5, kps=False, s_e=[], t_e=[], dfl_model=False): # img = RGB
        parameters = self.parameters
        swapper_model = parameters['SwapModelSelection']

        dst = faceutil.get_arcface_template(image_size=512, mode='arcfacemap')
        # Change the ref points
        if parameters['FaceAdjEnableToggle']:
            for k in dst:
                k[:,0] += parameters['KpsXSlider']
                k[:,1] += parameters['KpsYSlider']
                k[:,0] -= 255
                k[:,0] *= (1+parameters['KpsScaleSlider']/100)
                k[:,0] += 255
                k[:,1] -= 255
                k[:,1] *= (1+parameters['KpsScaleSlider']/100)
                k[:,1] += 255
                    
        M, _ = faceutil.estimate_norm_arcface_template(kps_5, src=dst)
        tform = trans.SimilarityTransform()
        tform.params[0:2] = M

        # Scaling Transforms
        t512 = v2.Resize((512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        t384 = v2.Resize((384, 384), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        t256 = v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        t128 = v2.Resize((128, 128), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)

        # Grab 512 face from image and create 256 and 128 copys
        original_face_512 = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0), interpolation=v2.InterpolationMode.BILINEAR )
        original_face_512 = v2.functional.crop(original_face_512, 0,0, 512, 512)# 3, 512, 512
        original_face_384 = t384(original_face_512)
        original_face_256 = t256(original_face_512)
        original_face_128 = t128(original_face_256)
        if s_e is not None and len(s_e) > 0:
            if swapper_model == 'Inswapper128':
                latent = torch.from_numpy(self.models_processor.calc_swapper_latent(s_e)).float().to(self.models_processor.device)
                if parameters['FaceLikenessEnableToggle']:
                    factor = parameters['FaceLikenessFactorDecimalSlider']
                    dst_latent = torch.from_numpy(self.models_processor.calc_swapper_latent(t_e)).float().to(self.models_processor.device)
                    latent = latent - (factor * dst_latent)

                dim = 1
                if parameters['SwapperResSelection'] == '128':
                    dim = 1
                    input_face_affined = original_face_128
                elif parameters['SwapperResSelection'] == '256':
                    dim = 2
                    input_face_affined = original_face_256
                elif parameters['SwapperResSelection'] == '384':
                    dim = 3
                    input_face_affined = original_face_384
                elif parameters['SwapperResSelection'] == '512':
                    dim = 4
                    input_face_affined = original_face_512

            # Optional Scaling # change the transform matrix scaling from center
            if parameters['FaceAdjEnableToggle']:
                input_face_affined = v2.functional.affine(input_face_affined, 0, (0, 0), 1 + parameters['FaceScaleAmountSlider'] / 100, 0, center=(dim*128/2, dim*128/2), interpolation=v2.InterpolationMode.BILINEAR)

            itex = 1
            if parameters['StrengthEnableToggle']:
                itex = ceil(parameters['StrengthAmountSlider'] / 100.)

            output_size = int(128 * dim)
            output = torch.zeros((output_size, output_size, 3), dtype=torch.float32, device=self.models_processor.device)
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

                                swapper_output = torch.empty((1,3,128,128), dtype=torch.float32, device=self.models_processor.device).contiguous()
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
        else:
            swap = original_face_512
            if parameters['StrengthEnableToggle']:
                itex = ceil(parameters['StrengthAmountSlider'] / 100.)
                prev_face = torch.div(swap, 255.)
                prev_face = prev_face.permute(1, 2, 0)

        if parameters['StrengthEnableToggle']:
            if itex == 0:
                swap = original_face_512.clone()
            else:
                alpha = np.mod(parameters['StrengthAmountSlider'], 100)*0.01
                if alpha==0:
                    alpha=1

                # Blend the images
                prev_face = torch.mul(prev_face, 255)
                prev_face = torch.clamp(prev_face, 0, 255)
                prev_face = prev_face.permute(2, 0, 1)
                prev_face = t512(prev_face)
                swap = torch.mul(swap, alpha)
                prev_face = torch.mul(prev_face, 1-alpha)
                swap = torch.add(swap, prev_face)

        # Create border mask
        border_mask = torch.ones((128, 128), dtype=torch.float32, device=self.models_processor.device)
        border_mask = torch.unsqueeze(border_mask,0)

        # if parameters['BorderState']:
        top = parameters['BorderTopSlider']
        left = parameters['BorderLeftSlider']
        right = 128 - parameters['BorderRightSlider']
        bottom = 128 - parameters['BorderBottomSlider']

        border_mask[:, :top, :] = 0
        border_mask[:, bottom:, :] = 0
        border_mask[:, :, :left] = 0
        border_mask[:, :, right:] = 0

        gauss = transforms.GaussianBlur(parameters['BorderBlurSlider']*2+1, (parameters['BorderBlurSlider']+1)*0.2)
        border_mask = gauss(border_mask)

        # Create image mask
        swap_mask = torch.ones((128, 128), dtype=torch.float32, device=self.models_processor.device)
        swap_mask = torch.unsqueeze(swap_mask,0)

        # Restorer
        if parameters["FaceRestorerEnableToggle"]:
            swap = self.models_processor.apply_facerestorer(swap, parameters['FaceRestorerDetTypeSelection'], parameters['FaceRestorerTypeSelection'], parameters["FaceRestorerBlendSlider"], parameters['FaceFidelityWeightDecimalSlider'], parameters['DetectorScoreSlider'])

        # Restorer2
        if parameters["FaceRestorerEnable2Toggle"]:
            swap = self.models_processor.apply_facerestorer(swap, parameters['FaceRestorerDetType2Selection'], parameters['FaceRestorerType2Selection'], parameters["FaceRestorerBlend2Slider"], parameters['FaceFidelityWeight2DecimalSlider'], parameters['DetectorScoreSlider'])

        # Occluder
        if parameters["OccluderEnableToggle"]:
            mask = self.models_processor.apply_occlusion(original_face_256, parameters["OccluderSizeSlider"])
            mask = t128(mask)
            swap_mask = torch.mul(swap_mask, mask)
            gauss = transforms.GaussianBlur(parameters['OccluderXSegBlurSlider']*2+1, (parameters['OccluderXSegBlurSlider']+1)*0.2)
            swap_mask = gauss(swap_mask)

        if parameters["DFLXSegEnableToggle"]:
            img_mask = self.models_processor.apply_dfl_xseg(original_face_256, -parameters["DFLXSegSizeSlider"])
            img_mask = t128(img_mask)
            swap_mask = torch.mul(swap_mask, 1 - img_mask)
            gauss = transforms.GaussianBlur(parameters['OccluderXSegBlurSlider']*2+1, (parameters['OccluderXSegBlurSlider']+1)*0.2)
            swap_mask = gauss(swap_mask)

        if parameters["FaceParserEnableToggle"]:
            mask = self.models_processor.apply_face_parser(swap, parameters)
            mask = t128(mask)
            swap_mask = torch.mul(swap_mask, mask)

        # CLIPs
        if parameters["ClipEnableToggle"]:
            mask = self.models_processor.run_CLIPs(original_face_512, parameters["ClipText"], parameters["ClipAmountSlider"])
            mask = t128(mask)
            swap_mask *= mask

        if parameters['RestoreMouthEnableToggle'] or parameters['RestoreEyesEnableToggle']:
            M = tform.params[0:2]
            ones_column = np.ones((kps_5.shape[0], 1), dtype=np.float32)
            homogeneous_kps = np.hstack([kps_5, ones_column])
            dst_kps_5 = np.dot(homogeneous_kps, M.T)

            img_swap_mask = torch.ones((1, 512, 512), dtype=torch.float32, device=self.models_processor.device).contiguous()
            img_orig_mask = torch.zeros((1, 512, 512), dtype=torch.float32, device=self.models_processor.device).contiguous()

            if parameters['RestoreMouthEnableToggle']:
                img_swap_mask = self.models_processor.restore_mouth(img_orig_mask, img_swap_mask, dst_kps_5, parameters['RestoreMouthBlendAmountSlider']/100, parameters['RestoreMouthFeatherBlendSlider'], parameters['RestoreMouthSizeFactorSlider']/100, parameters['RestoreXMouthRadiusFactorDecimalSlider'], parameters['RestoreYMouthRadiusFactorDecimalSlider'], parameters['RestoreXMouthOffsetSlider'], parameters['RestoreYMouthOffsetSlider'])
                img_swap_mask = torch.clamp(img_swap_mask, 0, 1)

            if parameters['RestoreEyesEnableToggle']:
                img_swap_mask = self.models_processor.restore_eyes(img_orig_mask, img_swap_mask, dst_kps_5, parameters['RestoreEyesBlendAmountSlider']/100, parameters['RestoreEyesFeatherBlendSlider'], parameters['RestoreEyesSizeFactorDecimalSlider'],  parameters['RestoreXEyesRadiusFactorDecimalSlider'], parameters['RestoreYEyesRadiusFactorDecimalSlider'], parameters['RestoreXEyesOffsetSlider'], parameters['RestoreYEyesOffsetSlider'], parameters['RestoreEyesSpacingOffsetSlider'])
                img_swap_mask = torch.clamp(img_swap_mask, 0, 1)

            gauss = transforms.GaussianBlur(parameters['RestoreEyesMouthBlurSlider']*2+1, (parameters['RestoreEyesMouthBlurSlider']+1)*0.2)
            img_swap_mask = gauss(img_swap_mask)

            img_swap_mask = t128(img_swap_mask)
            swap_mask = torch.mul(swap_mask, img_swap_mask)

        # Face Diffing
        if parameters["DifferencingEnableToggle"]:
            mask = self.models_processor.apply_fake_diff(swap, original_face_512, parameters["DifferencingAmountSlider"])
            gauss = transforms.GaussianBlur(parameters['DifferencingBlendAmountSlider']*2+1, (parameters['DifferencingBlendAmountSlider']+1)*0.2)
            mask = gauss(mask.type(torch.float32))
            swap = swap * mask + original_face_512*(1-mask)

        if parameters["AutoColorEnableToggle"]:
            # Histogram color matching original face on swapped face
            if parameters['AutoColorTransferTypeSelection'] == 'Test':
                swap = faceutil.histogram_matching(original_face_512, swap, parameters["AutoColorBlendAmountSlider"])

            elif parameters['AutoColorTransferTypeSelection'] == 'Test_Mask':
                swap = faceutil.histogram_matching_withmask(original_face_512, swap, t512(swap_mask), parameters["AutoColorBlendAmountSlider"])

            elif parameters['AutoColorTransferTypeSelection'] == 'DFL_Test':
                swap = faceutil.histogram_matching_DFL_test(original_face_512, swap, parameters["AutoColorBlendAmountSlider"])

            elif parameters['AutoColorTransferTypeSelection'] == 'DFL_Orig':
                swap = faceutil.histogram_matching_DFL_Orig(original_face_512, swap, t512(swap_mask), parameters["AutoColorBlendAmountSlider"])

        # Apply color corrections
        if parameters['ColorEnableToggle']:
            swap = torch.unsqueeze(swap,0).contiguous()
            swap = v2.functional.adjust_gamma(swap, parameters['ColorGammaDecimalSlider'], 1.0)
            swap = torch.squeeze(swap)
            swap = swap.permute(1, 2, 0).type(torch.float32)

            del_color = torch.tensor([parameters['ColorRedSlider'], parameters['ColorGreenSlider'], parameters['ColorBlueSlider']], device=self.models_processor.device)
            swap += del_color
            swap = torch.clamp(swap, min=0., max=255.)
            swap = swap.permute(2, 0, 1).type(torch.uint8)

            swap = v2.functional.adjust_brightness(swap, parameters['ColorBrightnessDecimalSlider'])
            swap = v2.functional.adjust_contrast(swap, parameters['ColorContrastDecimalSlider'])
            swap = v2.functional.adjust_saturation(swap, parameters['ColorSaturationDecimalSlider'])
            swap = v2.functional.adjust_sharpness(swap, parameters['ColorSharpnessDecimalSlider'])
            swap = v2.functional.adjust_hue(swap, parameters['ColorHueDecimalSlider'])

            if parameters['ColorNoiseDecimalSlider'] > 0:
                swap = swap.permute(1, 2, 0).type(torch.float32)
                swap = swap + parameters['ColorNoiseDecimalSlider']*torch.randn(512, 512, 3, device=self.models_processor.device)
                swap = torch.clamp(swap, 0, 255)
                swap = swap.permute(2, 0, 1)

        if parameters['FinalBlendAdjEnableToggle'] and parameters['FinalBlendAdjEnableToggle'] > 0:
            final_blur_strength = parameters['FinalBlendAmountSlider']  # Ein Parameter steuert beides
            # Bestimme kernel_size und sigma basierend auf dem Parameter
            kernel_size = 2 * final_blur_strength + 1  # Ungerade Zahl, z.B. 3, 5, 7, ...
            sigma = final_blur_strength * 0.1  # Sigma proportional zur Stärke
            # Gaussian Blur anwenden
            gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
            swap = gaussian_blur(swap)

            # Add blur to swap_mask results
            gauss = transforms.GaussianBlur(parameters['OverallMaskBlendAmountSlider'] * 2 + 1, (parameters['OverallMaskBlendAmountSlider'] + 1) * 0.2)
            swap_mask = gauss(swap_mask)

        # Combine border and swap mask, scale, and apply to swap
        swap_mask = torch.mul(swap_mask, border_mask)
        swap_mask = t512(swap_mask)

        swap = torch.mul(swap, swap_mask)

        # Calculate the area to be mergerd back to the original frame
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

        # Untransform the swap mask
        swap_mask = v2.functional.pad(swap_mask, (0,0,img.shape[2]-512, img.shape[1]-512))
        swap_mask = v2.functional.affine(swap_mask, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )
        swap_mask = swap_mask[0:1, top:bottom, left:right]
        swap_mask = swap_mask.permute(1, 2, 0)
        swap_mask = torch.sub(1, swap_mask)

        # Apply the mask to the original image areas
        img_crop = img[0:3, top:bottom, left:right]
        img_crop = img_crop.permute(1,2,0)
        img_crop = torch.mul(swap_mask,img_crop)
            
        #Add the cropped areas and place them back into the original image
        swap = torch.add(swap, img_crop)
        swap = swap.type(torch.uint8)
        swap = swap.permute(2,0,1)
        img[0:3, top:bottom, left:right] = swap
        return img