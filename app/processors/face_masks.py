from typing import TYPE_CHECKING

import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import v2

from app.processors.external.clipseg import CLIPDensePredT
from app.processors.models_data import models_dir
if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor

class FaceMasks:
    def __init__(self, models_processor: 'ModelsProcessor'):
        self.models_processor = models_processor

    def apply_occlusion(self, img, amount):
        img = torch.div(img, 255)
        img = torch.unsqueeze(img, 0).contiguous()
        outpred = torch.ones((256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()

        self.models_processor.run_occluder(img, outpred)

        outpred = torch.squeeze(outpred)
        outpred = (outpred > 0)
        outpred = torch.unsqueeze(outpred, 0).type(torch.float32)

        if amount >0:
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=self.models_processor.device)

            for _ in range(int(amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)

            outpred = torch.squeeze(outpred)

        if amount <0:
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=self.models_processor.device)

            for _ in range(int(-amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)

            outpred = torch.squeeze(outpred)
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)

        outpred = torch.reshape(outpred, (1, 256, 256))
        return outpred

    def run_occluder(self, image, output):
        if not self.models_processor.models['Occluder']:
            self.models_processor.models['Occluder'] = self.models_processor.load_model('Occluder')

        io_binding = self.models_processor.models['Occluder'].io_binding()
        io_binding.bind_input(name='img', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,1,256,256), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['Occluder'].run_with_iobinding(io_binding)

    def apply_dfl_xseg(self, img, amount):
        img = img.type(torch.float32)
        img = torch.div(img, 255)
        img = torch.unsqueeze(img, 0).contiguous()
        outpred = torch.ones((256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()

        self.run_dfl_xseg(img, outpred)

        outpred = torch.clamp(outpred, min=0.0, max=1.0)
        outpred[outpred < 0.1] = 0
        # invert values to mask areas to keep
        outpred = 1.0 - outpred
        outpred = torch.unsqueeze(outpred, 0).type(torch.float32)

        if amount > 0:
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=self.models_processor.device)

            for _ in range(int(amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)

            outpred = torch.squeeze(outpred)

        if amount < 0:
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=self.models_processor.device)

            for _ in range(int(-amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)

            outpred = torch.squeeze(outpred)
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)

        outpred = torch.reshape(outpred, (1, 256, 256))
        return outpred

    def run_dfl_xseg(self, image, output):
        if not self.models_processor.models['XSeg']:
            self.models_processor.models['XSeg'] = self.models_processor.load_model('XSeg')

        io_binding = self.models_processor.models['XSeg'].io_binding()
        io_binding.bind_input(name='in_face:0', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='out_mask:0', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,1,256,256), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['XSeg'].run_with_iobinding(io_binding)
        
    def apply_face_parser(self, img, parameters):
        # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        FaceAmount = parameters["BackgroundParserSlider"]

        img = torch.div(img, 255)
        img = v2.functional.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        img = torch.reshape(img, (1, 3, 512, 512))
        outpred = torch.empty((1,19,512,512), dtype=torch.float32, device=self.models_processor.device).contiguous()

        self.run_faceparser(img, outpred)

        outpred = torch.squeeze(outpred)
        outpred = torch.argmax(outpred, 0)

        face_attributes = {
            1: parameters['FaceParserSlider'], #Face
            2: parameters['LeftEyebrowParserSlider'], #Left Eyebrow
            3: parameters['RightEyebrowParserSlider'], #Right Eyebrow
            4: parameters['LeftEyeParserSlider'], #Left Eye
            5: parameters['RightEyeParserSlider'], #Right Eye
            6: parameters['EyeGlassesParserSlider'], #EyeGlasses
            10: parameters['NoseParserSlider'], #Nose
            11: parameters['MouthParserSlider'], #Mouth
            12: parameters['UpperLipParserSlider'], #Upper Lip
            13: parameters['LowerLipParserSlider'], #Lower Lip
            14: parameters['NeckParserSlider'], #Neck
            17: parameters['HairParserSlider'], #Hair
        }
        
        # Pre-calculated kernel for dilation (3x3 kernel to reduce iterations)
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=self.models_processor.device)  # Kernel 3x3

        face_parses = []
        for attribute, attribute_value in face_attributes.items():
            if attribute_value > 0:
                attribute_idxs = torch.tensor( [attribute], device=self.models_processor.device)
                iters = int(attribute_value)

                attribute_parse = torch.isin(outpred, attribute_idxs)
                attribute_parse = torch.clamp(~attribute_parse, 0, 1).type(torch.float32)
                attribute_parse = torch.reshape(attribute_parse, (1,1,512,512))
                attribute_parse = torch.neg(attribute_parse)
                attribute_parse = torch.add(attribute_parse, 1)

                for _ in range(iters):
                    attribute_parse = torch.nn.functional.conv2d(attribute_parse, kernel, padding=(1, 1))
                    attribute_parse = torch.clamp(attribute_parse, 0, 1)

                attribute_parse = torch.squeeze(attribute_parse)
                attribute_parse = torch.neg(attribute_parse)
                attribute_parse = torch.add(attribute_parse, 1)
                attribute_parse = torch.reshape(attribute_parse, (1, 512, 512))

                # Apply Gaussian blur if needed
                blur_kernel_size = parameters['FaceBlurParserSlider'] * 2 + 1
                if blur_kernel_size > 1:
                    gauss = transforms.GaussianBlur(blur_kernel_size, (parameters['FaceBlurParserSlider'] + 1) * 0.2)
                    attribute_parse = gauss(attribute_parse)
            else:
                attribute_parse = torch.ones((1, 512, 512), dtype=torch.float32, device=self.models_processor.device)
            face_parses.append(attribute_parse)

        # BG Parse
        bg_idxs = torch.tensor([0, 14, 15, 16, 17, 18], device=self.models_processor.device)

        # For bg_parse, invert the mask so that the black background expands
        bg_parse = 1 - torch.isin(outpred, bg_idxs).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 512)

        if FaceAmount > 0:
            for _ in range(int(FaceAmount)):
                bg_parse = torch.nn.functional.conv2d(bg_parse, kernel, padding=(1, 1))  # Padding (1, 1) for 3x3 kernel
                bg_parse = torch.clamp(bg_parse, 0, 1)

            blur_kernel_size = parameters['BackgroundBlurParserSlider'] * 2 + 1
            if blur_kernel_size > 1:
                gauss = transforms.GaussianBlur(blur_kernel_size, (parameters['BackgroundBlurParserSlider'] + 1) * 0.2)
                bg_parse = gauss(bg_parse)

            bg_parse = torch.clamp(bg_parse, 0, 1)

        elif FaceAmount < 0:
            bg_parse = 1 - bg_parse  # Invert mask back
            for _ in range(int(-FaceAmount)):
                bg_parse = torch.nn.functional.conv2d(bg_parse, kernel, padding=(1, 1))  # Padding (1, 1) for 3x3 kernel
                bg_parse = torch.clamp(bg_parse, 0, 1)

            bg_parse = 1 - bg_parse  # Re-invert back
            blur_kernel_size = parameters['BackgroundBlurParserSlider'] * 2 + 1
            if blur_kernel_size > 1:
                gauss = transforms.GaussianBlur(blur_kernel_size, (parameters['BackgroundBlurParserSlider'] + 1) * 0.2)
                bg_parse = gauss(bg_parse)

            bg_parse = torch.clamp(bg_parse, 0, 1)

        else:
            # If FaceAmount is 0, use a fully white mask
            bg_parse = torch.ones((1, 512, 512), dtype=torch.float32, device=self.models_processor.device)

        out_parse = bg_parse.squeeze(0)
        for face_parse in face_parses:
            out_parse = torch.mul(out_parse, face_parse)

        # Final clamping to ensure the output parse is valid
        out_parse = torch.clamp(out_parse, 0, 1)

        return out_parse

    # https://github.com/yakhyo/face-parsing
    def run_faceparser(self, image, output):
        if not self.models_processor.models['FaceParser']:
            self.models_processor.models['FaceParser'] = self.models_processor.load_model('FaceParser')

        image = image.contiguous()
        io_binding = self.models_processor.models['FaceParser'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,19,512,512), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['FaceParser'].run_with_iobinding(io_binding)

    def run_CLIPs(self, img, CLIPText, CLIPAmount):
        # Ottieni il dispositivo su cui si trova l'immagine
        device = img.device

        # Controllo se la sessione CLIP è già stata inizializzata
        if not self.models_processor.clip_session:
            self.models_processor.clip_session = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
            self.models_processor.clip_session.eval()
            self.models_processor.clip_session.load_state_dict(torch.load(f'{models_dir}/rd64-uni-refined.pth', weights_only=True), strict=False)
            self.models_processor.clip_session.to(device)  # Sposta il modello sul dispositivo dell'immagine

        # Crea un mask tensor direttamente sul dispositivo dell'immagine
        clip_mask = torch.ones((352, 352), device=device)

        # L'immagine è già un tensore, quindi la converto a float32 e la normalizzo nel range [0, 1]
        img = img.float() / 255.0  # Conversione in float32 e normalizzazione

        # Rimuovi la parte ToTensor(), dato che img è già un tensore.
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((352, 352))
        ])

        # Applica la trasformazione all'immagine
        CLIPimg = transform(img).unsqueeze(0).contiguous().to(device)

        # Se ci sono prompt CLIPText, esegui la predizione
        if CLIPText != "":
            prompts = CLIPText.split(',')

            with torch.no_grad():
                # Esegui la predizione sulla sessione CLIP
                preds = self.models_processor.clip_session(CLIPimg.repeat(len(prompts), 1, 1, 1), prompts)[0]

            # Calcola la maschera CLIP usando la sigmoid e tieni tutto sul dispositivo
            clip_mask = 1 - torch.sigmoid(preds[0][0])
            for i in range(len(prompts) - 1):
                clip_mask *= 1 - torch.sigmoid(preds[i + 1][0])

            # Applica la soglia sulla maschera
            thresh = CLIPAmount / 100.0
            clip_mask = (clip_mask > thresh).float()

        return clip_mask.unsqueeze(0)  # Ritorna il tensore torch direttamente

    def soft_oval_mask(self, height, width, center, radius_x, radius_y, feather_radius=None):
        """
        Create a soft oval mask with feathering effect using integer operations.

        Args:
            height (int): Height of the mask.
            width (int): Width of the mask.
            center (tuple): Center of the oval (x, y).
            radius_x (int): Radius of the oval along the x-axis.
            radius_y (int): Radius of the oval along the y-axis.
            feather_radius (int): Radius for feathering effect.

        Returns:
            torch.Tensor: Soft oval mask tensor of shape (H, W).
        """
        if feather_radius is None:
            feather_radius = max(radius_x, radius_y) // 2  # Integer division

        # Calculating the normalized distance from the center
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')

        # Calculating the normalized distance from the center
        normalized_distance = torch.sqrt(((x - center[0]) / radius_x) ** 2 + ((y - center[1]) / radius_y) ** 2)

        # Creating the oval mask with a feathering effect
        mask = torch.clamp((1 - normalized_distance) * (radius_x / feather_radius), 0, 1)

        return mask

    def restore_mouth(self, img_orig, img_swap, kpss_orig, blend_alpha=0.5, feather_radius=10, size_factor=0.5, radius_factor_x=1.0, radius_factor_y=1.0, x_offset=0, y_offset=0):
        """
        Extract mouth from img_orig using the provided keypoints and place it in img_swap.

        Args:
            img_orig (torch.Tensor): The original image tensor of shape (C, H, W) from which mouth is extracted.
            img_swap (torch.Tensor): The target image tensor of shape (C, H, W) where mouth is placed.
            kpss_orig (list): List of keypoints arrays for detected faces. Each keypoints array contains coordinates for 5 keypoints.
            radius_factor_x (float): Factor to scale the horizontal radius. 1.0 means circular, >1.0 means wider oval, <1.0 means narrower.
            radius_factor_y (float): Factor to scale the vertical radius. 1.0 means circular, >1.0 means taller oval, <1.0 means shorter.
            x_offset (int): Horizontal offset for shifting the mouth left (negative value) or right (positive value).
            y_offset (int): Vertical offset for shifting the mouth up (negative value) or down (positive value).

        Returns:
            torch.Tensor: The resulting image tensor with mouth from img_orig placed on img_swap.
        """
        left_mouth = np.array([int(val) for val in kpss_orig[3]])
        right_mouth = np.array([int(val) for val in kpss_orig[4]])

        mouth_center = (left_mouth + right_mouth) // 2
        mouth_base_radius = int(np.linalg.norm(left_mouth - right_mouth) * size_factor)

        # Calculate the scaled radii
        radius_x = int(mouth_base_radius * radius_factor_x)
        radius_y = int(mouth_base_radius * radius_factor_y)

        # Apply the x/y_offset to the mouth center
        mouth_center[0] += x_offset
        mouth_center[1] += y_offset

        # Calculate bounding box for mouth region
        ymin = max(0, mouth_center[1] - radius_y)
        ymax = min(img_orig.size(1), mouth_center[1] + radius_y)
        xmin = max(0, mouth_center[0] - radius_x)
        xmax = min(img_orig.size(2), mouth_center[0] + radius_x)

        mouth_region_orig = img_orig[:, ymin:ymax, xmin:xmax]
        mouth_mask = self.soft_oval_mask(ymax - ymin, xmax - xmin,
                                            (radius_x, radius_y),
                                            radius_x, radius_y,
                                            feather_radius).to(img_orig.device)

        target_ymin = ymin
        target_ymax = ymin + mouth_region_orig.size(1)
        target_xmin = xmin
        target_xmax = xmin + mouth_region_orig.size(2)

        img_swap_mouth = img_swap[:, target_ymin:target_ymax, target_xmin:target_xmax]
        blended_mouth = blend_alpha * img_swap_mouth + (1 - blend_alpha) * mouth_region_orig

        img_swap[:, target_ymin:target_ymax, target_xmin:target_xmax] = mouth_mask * blended_mouth + (1 - mouth_mask) * img_swap_mouth
        return img_swap

    def restore_eyes(self, img_orig, img_swap, kpss_orig, blend_alpha=0.5, feather_radius=10, size_factor=3.5, radius_factor_x=1.0, radius_factor_y=1.0, x_offset=0, y_offset=0, eye_spacing_offset=0):
        """
        Extract eyes from img_orig using the provided keypoints and place them in img_swap.

        Args:
            img_orig (torch.Tensor): The original image tensor of shape (C, H, W) from which eyes are extracted.
            img_swap (torch.Tensor): The target image tensor of shape (C, H, W) where eyes are placed.
            kpss_orig (list): List of keypoints arrays for detected faces. Each keypoints array contains coordinates for 5 keypoints.
            radius_factor_x (float): Factor to scale the horizontal radius. 1.0 means circular, >1.0 means wider oval, <1.0 means narrower.
            radius_factor_y (float): Factor to scale the vertical radius. 1.0 means circular, >1.0 means taller oval, <1.0 means shorter.
            x_offset (int): Horizontal offset for shifting the eyes left (negative value) or right (positive value).
            y_offset (int): Vertical offset for shifting the eyes up (negative value) or down (positive value).
            eye_spacing_offset (int): Horizontal offset to move eyes closer together (negative value) or farther apart (positive value).

        Returns:
            torch.Tensor: The resulting image tensor with eyes from img_orig placed on img_swap.
        """
        # Extract original keypoints for left and right eye
        left_eye = np.array([int(val) for val in kpss_orig[0]])
        right_eye = np.array([int(val) for val in kpss_orig[1]])

        # Apply horizontal offset (x-axis)
        left_eye[0] += x_offset
        right_eye[0] += x_offset

        # Apply vertical offset (y-axis)
        left_eye[1] += y_offset
        right_eye[1] += y_offset

        # Calculate eye distance and radii
        eye_distance = np.linalg.norm(left_eye - right_eye)
        base_eye_radius = int(eye_distance / size_factor)

        # Calculate the scaled radii
        radius_x = int(base_eye_radius * radius_factor_x)
        radius_y = int(base_eye_radius * radius_factor_y)

        # Adjust for eye spacing (horizontal movement)
        left_eye[0] += eye_spacing_offset
        right_eye[0] -= eye_spacing_offset

        def extract_and_blend_eye(eye_center, radius_x, radius_y, img_orig, img_swap, blend_alpha, feather_radius):
            ymin = max(0, eye_center[1] - radius_y)
            ymax = min(img_orig.size(1), eye_center[1] + radius_y)
            xmin = max(0, eye_center[0] - radius_x)
            xmax = min(img_orig.size(2), eye_center[0] + radius_x)

            eye_region_orig = img_orig[:, ymin:ymax, xmin:xmax]
            eye_mask = self.soft_oval_mask(ymax - ymin, xmax - xmin,
                                            (radius_x, radius_y),
                                            radius_x, radius_y,
                                            feather_radius).to(img_orig.device)

            target_ymin = ymin
            target_ymax = ymin + eye_region_orig.size(1)
            target_xmin = xmin
            target_xmax = xmin + eye_region_orig.size(2)

            img_swap_eye = img_swap[:, target_ymin:target_ymax, target_xmin:target_xmax]
            blended_eye = blend_alpha * img_swap_eye + (1 - blend_alpha) * eye_region_orig

            img_swap[:, target_ymin:target_ymax, target_xmin:target_xmax] = eye_mask * blended_eye + (1 - eye_mask) * img_swap_eye

        # Process both eyes with updated positions
        extract_and_blend_eye(left_eye, radius_x, radius_y, img_orig, img_swap, blend_alpha, feather_radius)
        extract_and_blend_eye(right_eye, radius_x, radius_y, img_orig, img_swap, blend_alpha, feather_radius)

        return img_swap

    def apply_fake_diff(self, swapped_face, original_face, DiffAmount):
        swapped_face = swapped_face.permute(1,2,0)
        original_face = original_face.permute(1,2,0)

        diff = swapped_face-original_face
        diff = torch.abs(diff)

        # Find the diffrence between the swap and original, per channel
        fthresh = DiffAmount*2.55

        # Bimodal
        diff[diff<fthresh] = 0
        diff[diff>=fthresh] = 1

        # If any of the channels exceeded the threshhold, them add them to the mask
        diff = torch.sum(diff, dim=2)
        diff = torch.unsqueeze(diff, 2)
        diff[diff>0] = 1

        diff = diff.permute(2,0,1)

        return diff
