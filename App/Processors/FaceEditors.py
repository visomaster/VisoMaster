import torch
import numpy as np
from torchvision.transforms import v2
from skimage import transform as trans
from App.Processors.ModelsData import models_dir
from App.Processors.Utils import FaceUtil as faceutil
from torchvision import transforms
import pickle
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from App.Processors.ModelsProcessor import ModelsProcessor

from torch.cuda import nvtx

    
class FaceEditors:
    def __init__(self, models_processor: 'ModelsProcessor'):
        self.models_processor = models_processor
        self.lp_mask_crop = faceutil.create_faded_inner_mask(size=(512, 512), border_thickness=5, fade_thickness=15, blur_radius=5, device=self.models_processor.device)
        self.lp_mask_crop = torch.unsqueeze(self.lp_mask_crop, 0)
        self.lp_lip_array = np.array(self.load_lip_array())

    def load_lip_array(self):
        with open(f'{models_dir}/liveportrait_onnx/lip_array.pkl', 'rb') as f:
            return pickle.load(f)
        
    def lp_motion_extractor(self, img, face_editor_type='Human-Face', **kwargs) -> dict:
        kp_info = {}
        with torch.no_grad():
            # We force to use TensorRT because it doesn't work well in trt
            #if self.models_processor.provider_name == "TensorRT-Engine":
            if self.models_processor.provider_name == "!TensorRT-Engine":
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models_trt['LivePortraitMotionExtractor']:
                        self.models_processor.models_trt['LivePortraitMotionExtractor'] = self.models_processor.load_model_trt('LivePortraitMotionExtractor', custom_plugin_path=None, precision="fp32")

                motion_extractor_model = self.models_processor.models_trt['LivePortraitMotionExtractor']
                input_spec = motion_extractor_model.input_spec()
                output_spec = motion_extractor_model.output_spec()

                # prepare_source
                I_s = torch.div(img.type(torch.float32), 255.)
                I_s = torch.clamp(I_s, 0, 1)  # clamp to 0~1
                I_s = torch.unsqueeze(I_s, 0).contiguous()

                nvtx.range_push("forward")

                feed_dict = {}
                feed_dict["img"] = I_s
                #stream = torch.cuda.Stream()
                #preds_dict = motion_extractor_model.predict_async(feed_dict, stream)
                preds_dict = motion_extractor_model.predict_async(feed_dict, torch.cuda.current_stream())
                #preds_dict = motion_extractor_model.predict(feed_dict)

                kp_info = {
                    'pitch': preds_dict["pitch"],
                    'yaw': preds_dict["yaw"],
                    'roll': preds_dict["roll"],
                    't': preds_dict["t"],
                    'exp': preds_dict["exp"],
                    'scale': preds_dict["scale"],
                    'kp': preds_dict["kp"]
                }

                nvtx.range_pop()

            else:
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models['LivePortraitMotionExtractor']:
                        self.models_processor.models['LivePortraitMotionExtractor'] = self.models_processor.load_model('LivePortraitMotionExtractor')

                motion_extractor_model = self.models_processor.models['LivePortraitMotionExtractor']

                # prepare_source
                I_s = torch.div(img.type(torch.float32), 255.)
                I_s = torch.clamp(I_s, 0, 1)  # clamp to 0~1
                I_s = torch.unsqueeze(I_s, 0).contiguous()

                pitch = torch.empty((1,66), dtype=torch.float32, device=self.models_processor.device).contiguous()
                yaw = torch.empty((1,66), dtype=torch.float32, device=self.models_processor.device).contiguous()
                roll = torch.empty((1,66), dtype=torch.float32, device=self.models_processor.device).contiguous()
                t = torch.empty((1,3), dtype=torch.float32, device=self.models_processor.device).contiguous()
                exp = torch.empty((1,63), dtype=torch.float32, device=self.models_processor.device).contiguous()
                scale = torch.empty((1,1), dtype=torch.float32, device=self.models_processor.device).contiguous()
                kp = torch.empty((1,63), dtype=torch.float32, device=self.models_processor.device).contiguous()

                io_binding = motion_extractor_model.io_binding()
                io_binding.bind_input(name='img', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=I_s.size(), buffer_ptr=I_s.data_ptr())
                io_binding.bind_output(name='pitch', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=pitch.size(), buffer_ptr=pitch.data_ptr())
                io_binding.bind_output(name='yaw', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=yaw.size(), buffer_ptr=yaw.data_ptr())
                io_binding.bind_output(name='roll', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=roll.size(), buffer_ptr=roll.data_ptr())
                io_binding.bind_output(name='t', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=t.size(), buffer_ptr=t.data_ptr())
                io_binding.bind_output(name='exp', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=exp.size(), buffer_ptr=exp.data_ptr())
                io_binding.bind_output(name='scale', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=scale.size(), buffer_ptr=scale.data_ptr())
                io_binding.bind_output(name='kp', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=kp.size(), buffer_ptr=kp.data_ptr())

                if self.models_processor.device == "cuda":
                    torch.cuda.synchronize()
                elif self.models_processor.device != "cpu":
                    self.models_processor.syncvec.cpu()
                motion_extractor_model.run_with_iobinding(io_binding)

                kp_info = {
                    'pitch': pitch,
                    'yaw': yaw,
                    'roll': roll,
                    't': t,
                    'exp': exp,
                    'scale': scale,
                    'kp': kp
                }

            flag_refine_info: bool = kwargs.get('flag_refine_info', True)
            if flag_refine_info:
                bs = kp_info['kp'].shape[0]
                kp_info['pitch'] = faceutil.headpose_pred_to_degree(kp_info['pitch'])[:, None]  # Bx1
                kp_info['yaw'] = faceutil.headpose_pred_to_degree(kp_info['yaw'])[:, None]  # Bx1
                kp_info['roll'] = faceutil.headpose_pred_to_degree(kp_info['roll'])[:, None]  # Bx1
                kp_info['kp'] = kp_info['kp'].reshape(bs, -1, 3)  # BxNx3
                kp_info['exp'] = kp_info['exp'].reshape(bs, -1, 3)  # BxNx3

        return kp_info

    def lp_appearance_feature_extractor(self, img, face_editor_type='Human-Face'):
        with torch.no_grad():
            # We force to use TensorRT. 
            #if self.models_processor.provider_name == "TensorRT-Engine":
            if self.models_processor.provider_name == "!TensorRT-Engine":
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models_trt['LivePortraitAppearanceFeatureExtractor']:
                        self.models_processor.models_trt['LivePortraitAppearanceFeatureExtractor'] = self.models_processor.load_model_trt('LivePortraitAppearanceFeatureExtractor', custom_plugin_path=None, precision="fp16")

                appearance_feature_extractor_model = self.models_processor.models_trt['LivePortraitAppearanceFeatureExtractor']

                # prepare_source
                I_s = torch.div(img.type(torch.float32), 255.)
                I_s = torch.clamp(I_s, 0, 1)  # clamp to 0~1
                I_s = torch.unsqueeze(I_s, 0).contiguous()

                nvtx.range_push("forward")

                feed_dict = {}
                feed_dict["img"] = I_s
                preds_dict = appearance_feature_extractor_model.predict_async(feed_dict, torch.cuda.current_stream())
                #preds_dict = appearance_feature_extractor_model.predict(feed_dict)

                output = preds_dict["output"]

                nvtx.range_pop()

            else:
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models['LivePortraitAppearanceFeatureExtractor']:
                        self.models_processor.models['LivePortraitAppearanceFeatureExtractor'] = self.models_processor.load_model('LivePortraitAppearanceFeatureExtractor')

                appearance_feature_extractor_model = self.models_processor.models['LivePortraitAppearanceFeatureExtractor']

                # prepare_source
                I_s = torch.div(img.type(torch.float32), 255.)
                I_s = torch.clamp(I_s, 0, 1)  # clamp to 0~1
                I_s = torch.unsqueeze(I_s, 0).contiguous()

                output = torch.empty((1,32,16,64,64), dtype=torch.float32, device=self.models_processor.device).contiguous()

                io_binding = appearance_feature_extractor_model.io_binding()
                io_binding.bind_input(name='img', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=I_s.size(), buffer_ptr=I_s.data_ptr())
                io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())

                if self.models_processor.device == "cuda":
                    torch.cuda.synchronize()
                elif self.models_processor.device != "cpu":
                    self.models_processor.syncvec.cpu()
                appearance_feature_extractor_model.run_with_iobinding(io_binding)

        return output

    def lp_retarget_eye(self, kp_source: torch.Tensor, eye_close_ratio: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        """
        kp_source: BxNx3
        eye_close_ratio: Bx3
        Return: Bx(3*num_kp)
        """
        with torch.no_grad():
            # We force to use TensorRT. 
            #if self.models_processor.provider_name == "TensorRT-Engine":
            if self.models_processor.provider_name == "!TensorRT-Engine":
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models_trt['LivePortraitStitchingEye']:
                        self.models_processor.models_trt['LivePortraitStitchingEye'] = self.models_processor.load_model_trt('LivePortraitStitchingEye', custom_plugin_path=None, precision="fp16")

                stitching_eye_model = self.models_processor.models_trt['LivePortraitStitchingEye']

                feat_eye = faceutil.concat_feat(kp_source, eye_close_ratio).contiguous()

                nvtx.range_push("forward")

                feed_dict = {}
                feed_dict["input"] = feat_eye
                preds_dict = stitching_eye_model.predict_async(feed_dict, torch.cuda.current_stream())
                #preds_dict = stitching_eye_model.predict(feed_dict)

                delta = preds_dict["output"]

                nvtx.range_pop()

            else:
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models['LivePortraitStitchingEye']:
                        self.models_processor.models['LivePortraitStitchingEye'] = self.models_processor.load_model('LivePortraitStitchingEye')

                stitching_eye_model = self.models_processor.models['LivePortraitStitchingEye']

                feat_eye = faceutil.concat_feat(kp_source, eye_close_ratio).contiguous()
                delta = torch.empty((1,63), dtype=torch.float32, device=self.models_processor.device).contiguous()

                io_binding = stitching_eye_model.io_binding()
                io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=feat_eye.size(), buffer_ptr=feat_eye.data_ptr())
                io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=delta.size(), buffer_ptr=delta.data_ptr())

                if self.models_processor.device == "cuda":
                    torch.cuda.synchronize()
                elif self.models_processor.device != "cpu":
                    self.models_processor.syncvec.cpu()
                stitching_eye_model.run_with_iobinding(io_binding)

        return delta.reshape(-1, kp_source.shape[1], 3)

    def lp_retarget_lip(self, kp_source: torch.Tensor, lip_close_ratio: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        """
        kp_source: BxNx3
        lip_close_ratio: Bx2
        Return: Bx(3*num_kp)
        """
        with torch.no_grad():
            # We force to use TensorRT. 
            #if self.models_processor.provider_name == "TensorRT-Engine":
            if self.models_processor.provider_name == "!TensorRT-Engine":
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models_trt['LivePortraitStitchingLip']:
                        self.models_processor.models_trt['LivePortraitStitchingLip'] = self.models_processor.load_model_trt('LivePortraitStitchingLip', custom_plugin_path=None, precision="fp16")

                stitching_lip_model = self.models_processor.models_trt['LivePortraitStitchingLip']

                feat_lip = faceutil.concat_feat(kp_source, lip_close_ratio).contiguous()

                nvtx.range_push("forward")

                feed_dict = {}
                feed_dict["input"] = feat_lip
                preds_dict = stitching_lip_model.predict_async(feed_dict, torch.cuda.current_stream())
                #preds_dict = stitching_lip_model.predict(feed_dict)

                delta = preds_dict["output"]

                nvtx.range_pop()

            else:
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models['LivePortraitStitchingLip']:
                        self.models_processor.models['LivePortraitStitchingLip'] = self.models_processor.load_model('LivePortraitStitchingLip')

                stitching_lip_model = self.models_processor.models['LivePortraitStitchingLip']

                feat_lip = faceutil.concat_feat(kp_source, lip_close_ratio).contiguous()
                delta = torch.empty((1,63), dtype=torch.float32, device=self.models_processor.device).contiguous()

                io_binding = stitching_lip_model.io_binding()
                io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=feat_lip.size(), buffer_ptr=feat_lip.data_ptr())
                io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=delta.size(), buffer_ptr=delta.data_ptr())

                if self.models_processor.device == "cuda":
                    torch.cuda.synchronize()
                elif self.models_processor.device != "cpu":
                    self.models_processor.syncvec.cpu()
                stitching_lip_model.run_with_iobinding(io_binding)

        return delta.reshape(-1, kp_source.shape[1], 3)
    
    def lp_stitch(self, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        """
        kp_source: BxNx3
        kp_driving: BxNx3
        Return: Bx(3*num_kp+2)
        """
        with torch.no_grad():
            # We force to use TensorRT. 
            #if self.models_processor.provider_name == "TensorRT-Engine":
            if self.models_processor.provider_name == "!TensorRT-Engine":
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models_trt['LivePortraitStitching']:
                        self.models_processor.models_trt['LivePortraitStitching'] = self.models_processor.load_model_trt('LivePortraitStitching', custom_plugin_path=None, precision="fp16")

                stitching_model = self.models_processor.models_trt['LivePortraitStitching']

                feat_stiching = faceutil.concat_feat(kp_source, kp_driving).contiguous()

                nvtx.range_push("forward")

                feed_dict = {}
                feed_dict["input"] = feat_stiching
                preds_dict = stitching_model.predict_async(feed_dict, torch.cuda.current_stream())
                #preds_dict = stitching_model.predict(feed_dict)

                delta = preds_dict["output"]

                nvtx.range_pop()

            else:
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models['LivePortraitStitching']:
                        self.models_processor.models['LivePortraitStitching'] = self.models_processor.load_model('LivePortraitStitching')

                stitching_model = self.models_processor.models['LivePortraitStitching']

                feat_stiching = faceutil.concat_feat(kp_source, kp_driving).contiguous()
                delta = torch.empty((1,65), dtype=torch.float32, device=self.models_processor.device).contiguous()

                io_binding = stitching_model.io_binding()
                io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=feat_stiching.size(), buffer_ptr=feat_stiching.data_ptr())
                io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=delta.size(), buffer_ptr=delta.data_ptr())

                if self.models_processor.device == "cuda":
                    torch.cuda.synchronize()
                elif self.models_processor.device != "cpu":
                    self.models_processor.syncvec.cpu()
                stitching_model.run_with_iobinding(io_binding)

        return delta

    def lp_stitching(self, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        """ conduct the stitching
        kp_source: Bxnum_kpx3
        kp_driving: Bxnum_kpx3
        """
        bs, num_kp = kp_source.shape[:2]

        # calculate default delta from kp_source (using kp_source as default)
        kp_driving_default = kp_source.clone()

        default_delta = self.models_processor.lp_stitch(kp_source, kp_driving_default, face_editor_type=face_editor_type)

        # Clone default delta values for expression and translation/rotation
        default_delta_exp = default_delta[..., :3*num_kp].reshape(bs, num_kp, 3).clone()  # 1x20x3
        default_delta_tx_ty = default_delta[..., 3*num_kp:3*num_kp+2].reshape(bs, 1, 2).clone()  # 1x1x2

        # Debug: Print default delta values (should be close to zero)
        #print("default_delta_exp:", default_delta_exp)
        #print("default_delta_tx_ty:", default_delta_tx_ty)

        kp_driving_new = kp_driving.clone()

        # calculate new delta based on kp_driving
        delta = self.models_processor.lp_stitch(kp_source, kp_driving_new, face_editor_type=face_editor_type)

        # Clone new delta values for expression and translation/rotation
        delta_exp = delta[..., :3*num_kp].reshape(bs, num_kp, 3).clone()  # 1x20x3
        delta_tx_ty = delta[..., 3*num_kp:3*num_kp+2].reshape(bs, 1, 2).clone()  # 1x1x2

        # Debug: Print new delta values
        #print("delta_exp:", delta_exp)
        #print("delta_tx_ty:", delta_tx_ty)

        # Calculate the difference between new and default delta
        delta_exp_diff = delta_exp - default_delta_exp
        delta_tx_ty_diff = delta_tx_ty - default_delta_tx_ty

        # Debug: Print the delta differences
        #print("delta_exp_diff:", delta_exp_diff)
        #print("delta_tx_ty_diff:", delta_tx_ty_diff)

        # Apply delta differences to the keypoints only if significant differences are found
        kp_driving_new += delta_exp_diff
        kp_driving_new[..., :2] += delta_tx_ty_diff

        return kp_driving_new

    def lp_warp_decode(self, feature_3d: torch.Tensor, kp_source: torch.Tensor, kp_driving: torch.Tensor, face_editor_type='Human-Face') -> torch.Tensor:
        """ get the image after the warping of the implicit keypoints
        feature_3d: Bx32x16x64x64, feature volume
        kp_source: BxNx3
        kp_driving: BxNx3
        """

        with torch.no_grad():
            if self.models_processor.provider_name == "TensorRT-Engine":
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models_trt['LivePortraitWarpingSpadeFix']:
                        self.models_processor.models_trt['LivePortraitWarpingSpadeFix'] = self.models_processor.load_model_trt('LivePortraitWarpingSpadeFix', custom_plugin_path=f'{models_dir}/grid_sample_3d_plugin.dll', precision="fp16")

                warping_spade_model = self.models_processor.models_trt['LivePortraitWarpingSpadeFix']

                feature_3d = feature_3d.contiguous()
                kp_source = kp_source.contiguous()
                kp_driving = kp_driving.contiguous()

                nvtx.range_push("forward")

                feed_dict = {}
                feed_dict["feature_3d"] = feature_3d
                feed_dict["kp_source"] = kp_source
                feed_dict["kp_driving"] = kp_driving
                stream = torch.cuda.Stream()
                preds_dict = warping_spade_model.predict_async(feed_dict, stream)
                #preds_dict = warping_spade_model.predict_async(feed_dict, torch.cuda.current_stream())
                #preds_dict = warping_spade_model.predict(feed_dict)

                out = preds_dict["out"]

                nvtx.range_pop()
            else:
                if face_editor_type == 'Human-Face':
                    if not self.models_processor.models['LivePortraitWarpingSpade']:
                        self.models_processor.models['LivePortraitWarpingSpade'] = self.models_processor.load_model('LivePortraitWarpingSpade')

                warping_spade_model = self.models_processor.models['LivePortraitWarpingSpade']

                feature_3d = feature_3d.contiguous()
                kp_source = kp_source.contiguous()
                kp_driving = kp_driving.contiguous()

                out = torch.empty((1,3,512,512), dtype=torch.float32, device=self.models_processor.device).contiguous()
                io_binding = warping_spade_model.io_binding()
                io_binding.bind_input(name='feature_3d', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=feature_3d.size(), buffer_ptr=feature_3d.data_ptr())
                io_binding.bind_input(name='kp_driving', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=kp_driving.size(), buffer_ptr=kp_driving.data_ptr())
                io_binding.bind_input(name='kp_source', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=kp_source.size(), buffer_ptr=kp_source.data_ptr())
                io_binding.bind_output(name='out', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=out.size(), buffer_ptr=out.data_ptr())

                if self.models_processor.device == "cuda":
                    torch.cuda.synchronize()
                elif self.models_processor.device != "cpu":
                    self.models_processor.syncvec.cpu()
                warping_spade_model.run_with_iobinding(io_binding)

        return out
    
    def face_parser_makeup_direct_rgb(self, img, parsing, part=(17,), color=[230, 50, 20], blend_factor=0.2):
        device = img.device  # Ensure we use the same device

        # Clamp blend factor to ensure it stays between 0 and 1
        blend_factor = min(max(blend_factor, 0.0), 1.0)

        # Normalize the target RGB color to [0, 1]
        r, g, b = [x / 255.0 for x in color]
        tar_color = torch.tensor([r, g, b], dtype=torch.float32).view(3, 1, 1).to(device)

        #print(f"Target RGB color: {tar_color}")

        # Create hair mask based on parsing for multiple parts
        if isinstance(part, tuple):
            hair_mask = torch.zeros_like(parsing, dtype=torch.bool)
            for p in part:
                hair_mask |= (parsing == p)  # Accumulate masks for all parts in the tuple
        else:
            hair_mask = (parsing == part)

        #print(f"Hair mask shape: {hair_mask.shape}, Non-zero elements in mask: {hair_mask.sum().item()}")

        # Expand mask to match the image dimensions
        mask = hair_mask.unsqueeze(0).expand_as(img)
        #print(f"Expanded mask shape: {mask.shape}, Non-zero elements: {mask.sum().item()}")

        # Ensure that the image is normalized to [0, 1]
        image_normalized = img.float() / 255.0

        # Perform the color blending for the target region
        # (1 - blend_factor) * original + blend_factor * target_color
        changed = torch.where(
            mask,
            (1 - blend_factor) * image_normalized + blend_factor * tar_color,
            image_normalized
        )

        # Scale back to [0, 255] for final output
        changed = torch.clamp(changed * 255, 0, 255).to(torch.uint8)

        return changed

    def apply_face_makeup(self, img, parameters):
        # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']

        # Normalize the image and perform parsing
        temp = torch.div(img, 255)
        temp = v2.functional.normalize(temp, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        temp = torch.reshape(temp, (1, 3, 512, 512))
        outpred = torch.empty((1, 19, 512, 512), dtype=torch.float32, device=self.models_processor.device).contiguous()

        self.models_processor.run_faceparser(temp, outpred)

        # Perform parsing prediction
        outpred = torch.squeeze(outpred)
        outpred = torch.argmax(outpred, 0)

        # Clone the image for modifications
        out = img.clone()

        # Apply makeup for each face part
        if parameters['FaceMakeupEnableToggle']:
            color = [parameters['FaceMakeupRedSlider'], parameters['FaceMakeupGreenSlider'], parameters['FaceMakeupBlueSlider']]
            out = self.face_parser_makeup_direct_rgb(img=out, parsing=outpred, part=(1, 7, 8, 10), color=color, blend_factor=parameters['FaceMakeupBlendAmountDecimalSlider'])

        if parameters['HairMakeupEnableToggle']:
            color = [parameters['HairMakeupRedSlider'], parameters['HairMakeupGreenSlider'], parameters['HairMakeupBlueSlider']]
            out = self.face_parser_makeup_direct_rgb(img=out, parsing=outpred, part=17, color=color, blend_factor=parameters['HairMakeupBlendAmountDecimalSlider'])

        if parameters['EyeBrowsMakeupEnableToggle']:
            color = [parameters['EyeBrowsMakeupRedSlider'], parameters['EyeBrowsMakeupGreenSlider'], parameters['EyeBrowsMakeupBlueSlider']]
            out = self.face_parser_makeup_direct_rgb(img=out, parsing=outpred, part=(2, 3), color=color, blend_factor=parameters['EyeBrowsMakeupBlendAmountDecimalSlider'])

        if parameters['LipsMakeupEnableToggle']:
            color = [parameters['LipsMakeupRedSlider'], parameters['LipsMakeupGreenSlider'], parameters['LipsMakeupBlueSlider']]
            out = self.face_parser_makeup_direct_rgb(img=out, parsing=outpred, part=(12, 13), color=color, blend_factor=parameters['LipsMakeupBlendAmountDecimalSlider'])

        # Define the different face attributes to apply makeup on
        face_attributes = {
            1: parameters['FaceMakeupEnableToggle'],  # Face
            2: parameters['EyeBrowsMakeupEnableToggle'],  # Left Eyebrow
            3: parameters['EyeBrowsMakeupEnableToggle'],  # Right Eyebrow
            7: parameters['FaceMakeupEnableToggle'],  # Left Ear
            8: parameters['FaceMakeupEnableToggle'],  # Right Ear
            10: parameters['FaceMakeupEnableToggle'],  # Nose
            12: parameters['LipsMakeupEnableToggle'],  # Upper Lip
            13: parameters['LipsMakeupEnableToggle'],  # Lower Lip
            17: parameters['HairMakeupEnableToggle'],  # Hair
        }

        # Pre-calculated kernel per dilatazione (kernel 3x3)
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=self.models_processor.device)

        # Apply blur if blur kernel size is greater than 1
        blur_kernel_size = parameters['FaceEditorBlurAmountSlider'] * 2 + 1
        if blur_kernel_size > 1:
            gauss = transforms.GaussianBlur(blur_kernel_size, (parameters['FaceEditorBlurAmountSlider'] + 1) * 0.2)

        # Generate masks for each face attribute
        face_parses = []
        for attribute in face_attributes.keys():
            if face_attributes[attribute]:  # Se l'attributo è abilitato
                attribute_idxs = torch.tensor([attribute], device=self.models_processor.device)

                # Create the mask: white for the part, black for the rest
                attribute_parse = torch.isin(outpred, attribute_idxs).float()
                attribute_parse = torch.clamp(attribute_parse, 0, 1)  # Manteniamo i valori tra 0 e 1
                attribute_parse = torch.reshape(attribute_parse, (1, 1, 512, 512))

                # Dilate the mask (if necessary)
                for i in range(1):  # One pass, modify if needed
                    attribute_parse = torch.nn.functional.conv2d(attribute_parse, kernel, padding=(1, 1))
                    attribute_parse = torch.clamp(attribute_parse, 0, 1)

                # Squeeze to restore dimensions
                attribute_parse = torch.squeeze(attribute_parse)

                # Apply blur if required
                if blur_kernel_size > 1:
                    attribute_parse = gauss(attribute_parse.unsqueeze(0)).squeeze(0)

            else:
                # If the attribute is not enabled, use a black mask
                attribute_parse = torch.zeros((512, 512), dtype=torch.float32, device=self.models_processor.device)
            
            # Add the mask to the list
            face_parses.append(attribute_parse)

        # Create a final mask to combine all the individual masks
        combined_mask = torch.zeros((512, 512), dtype=torch.float32, device=self.models_processor.device)
        for face_parse in face_parses:
            # Add batch and channel dimensions for interpolation
            face_parse = face_parse.unsqueeze(0).unsqueeze(0)  # From (512, 512) to (1, 1, 512, 512)
            
            # Apply bilinear interpolation for anti-aliasing
            face_parse = torch.nn.functional.interpolate(face_parse, size=(512, 512), mode='bilinear', align_corners=True)
            
            # Remove the batch and channel dimensions
            face_parse = face_parse.squeeze(0).squeeze(0)  # Back to (512, 512)
            combined_mask = torch.max(combined_mask, face_parse)  # Combine the masks

        # Final application of the makeup mask on the original image
        out = img * (1 - combined_mask.unsqueeze(0)) + out * combined_mask.unsqueeze(0)

        return out, combined_mask.unsqueeze(0)