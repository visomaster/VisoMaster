from itertools import product as product
from typing import TYPE_CHECKING
import pickle

import torch
import cv2
import numpy as np
from torchvision.transforms import v2

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor
from app.processors.models_data import models_dir
from app.processors.utils import faceutil

class FaceLandmarkDetectors:
    def __init__(self, models_processor: 'ModelsProcessor'):
        self.models_processor = models_processor

    def run_detect_landmark(self, img, bbox, det_kpss, detect_mode='203', score=0.5, from_points=False):
        kpss_5 = []
        kpss = []
        scores = []

        if detect_mode=='5':
            if not self.models_processor.models['FaceLandmark5']:
                self.models_processor.models['FaceLandmark5'] = self.models_processor.load_model('FaceLandmark5')

                feature_maps = [[64, 64], [32, 32], [16, 16]]
                min_sizes = [[16, 32], [64, 128], [256, 512]]
                steps = [8, 16, 32]
                image_size = 512
                # re-initialize self.models_processor.anchors due to clear_mem function
                self.models_processor.anchors  = []

                for k, f in enumerate(feature_maps):
                    min_size_array = min_sizes[k]
                    for i, j in product(range(f[0]), range(f[1])):
                        for min_size in min_size_array:
                            s_kx = min_size / image_size
                            s_ky = min_size / image_size
                            dense_cx = [x * steps[k] / image_size for x in [j + 0.5]]
                            dense_cy = [y * steps[k] / image_size for y in [i + 0.5]]
                            for cy, cx in product(dense_cy, dense_cx):
                                self.models_processor.anchors += [cx, cy, s_kx, s_ky]

            kpss_5, kpss, scores = self.detect_face_landmark_5(img, bbox=bbox, det_kpss=det_kpss, from_points=from_points)

        elif detect_mode=='68':
            if not self.models_processor.models['FaceLandmark68']:
                self.models_processor.models['FaceLandmark68'] = self.models_processor.load_model('FaceLandmark68')

            kpss_5, kpss, scores = self.detect_face_landmark_68(img, bbox=bbox, det_kpss=det_kpss, from_points=from_points)

        elif detect_mode=='3d68':
            if not self.models_processor.models['FaceLandmark3d68']:
                self.models_processor.models['FaceLandmark3d68'] = self.models_processor.load_model('FaceLandmark3d68')
                with open(f'{models_dir}/meanshape_68.pkl', 'rb') as f:
                    self.models_processor.mean_lmk = pickle.load(f)

            kpss_5, kpss, scores = self.detect_face_landmark_3d68(img, bbox=bbox, det_kpss=det_kpss, from_points=from_points)

            return kpss_5, kpss, scores

        elif detect_mode=='98':
            if not self.models_processor.models['FaceLandmark98']:
                self.models_processor.models['FaceLandmark98'] = self.models_processor.load_model('FaceLandmark98')

            kpss_5, kpss, scores = self.detect_face_landmark_98(img, bbox=bbox, det_kpss=det_kpss, from_points=from_points)

        elif detect_mode=='106':
            if not self.models_processor.models['FaceLandmark106']:
                self.models_processor.models['FaceLandmark106'] = self.models_processor.load_model('FaceLandmark106')

            kpss_5, kpss, scores = self.detect_face_landmark_106(img, bbox=bbox, det_kpss=det_kpss, from_points=from_points)

            return kpss_5, kpss, scores

        elif detect_mode=='203':
            if not self.models_processor.models['FaceLandmark203']:
                self.models_processor.models['FaceLandmark203'] = self.models_processor.load_model('FaceLandmark203')

            kpss_5, kpss, scores = self.detect_face_landmark_203(img, bbox=bbox, det_kpss=det_kpss, from_points=from_points)

            return kpss_5, kpss, scores

        elif detect_mode=='478':
            if not self.models_processor.models['FaceLandmark478']:
                self.models_processor.models['FaceLandmark478'] = self.models_processor.load_model('FaceLandmark478')

            if not self.models_processor.models['FaceBlendShapes']:
                self.models_processor.models['FaceBlendShapes'] = self.models_processor.load_model('FaceBlendShapes')

            kpss_5, kpss, scores = self.detect_face_landmark_478(img, bbox=bbox, det_kpss=det_kpss, from_points=from_points)

            return kpss_5, kpss, scores

        if len(kpss_5) > 0:
            if len(scores) > 0:
                if np.mean(scores) >= score:
                    return kpss_5, kpss, scores
            else:
                return kpss_5, kpss, scores

        return [], [], []


    def detect_face_landmark_5(self, img, bbox, det_kpss, from_points=False):
        if from_points == False:
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = 512.0  / (max(w, h)*1.5)
            image, M = faceutil.transform(img, center, 512, _scale, rotate)
        else:
            image, M = faceutil.warp_face_by_face_landmark_5(img, det_kpss, 512, mode='arcface128', interpolation=v2.InterpolationMode.BILINEAR)

        image = image.permute(1,2,0)

        mean = torch.tensor([104, 117, 123], dtype=torch.float32, device=self.models_processor.device)
        image = torch.sub(image, mean)

        image = image.permute(2,0,1)
        image = torch.reshape(image, (1, 3, 512, 512))

        height, width = (512, 512)
        tmp = [width, height, width, height, width, height, width, height, width, height]
        scale1 = torch.tensor(tmp, dtype=torch.float32, device=self.models_processor.device)

        conf = torch.empty((1,10752,2), dtype=torch.float32, device=self.models_processor.device).contiguous()
        landmarks = torch.empty((1,10752,10), dtype=torch.float32, device=self.models_processor.device).contiguous()

        io_binding = self.models_processor.models['FaceLandmark5'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='conf', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,10752,2), buffer_ptr=conf.data_ptr())
        io_binding.bind_output(name='landmarks', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,10752,10), buffer_ptr=landmarks.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['FaceLandmark5'].run_with_iobinding(io_binding)

        scores = torch.squeeze(conf)[:, 1]
        priors = torch.tensor(self.models_processor.anchors).view(-1, 4)
        priors = priors.to(self.models_processor.device)

        pre = torch.squeeze(landmarks, 0)

        tmp = (priors[:, :2] + pre[:, :2] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 2:4] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 4:6] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 6:8] * 0.1 * priors[:, 2:], priors[:, :2] + pre[:, 8:10] * 0.1 * priors[:, 2:])
        landmarks = torch.cat(tmp, dim=1)
        landmarks = torch.mul(landmarks, scale1)

        landmarks = landmarks.cpu().numpy()

        # ignore low scores
        score=.1
        inds = torch.where(scores>score)[0]
        inds = inds.cpu().numpy()
        scores = scores.cpu().numpy()

        landmarks, scores = landmarks[inds], scores[inds]

        # sort
        order = scores.argsort()[::-1]

        if len(order) > 0:
            landmarks = landmarks[order][0]
            scores = scores[order][0]

            landmarks = np.array([[landmarks[i], landmarks[i + 1]] for i in range(0,10,2)])

            IM = faceutil.invertAffineTransform(M)
            landmarks = faceutil.trans_points2d(landmarks, IM)
            scores = np.array([scores])

            return landmarks, landmarks, scores

        return [], [], []

    def detect_face_landmark_68(self, img, bbox, det_kpss, from_points=False):
        if from_points == False:
            crop_image, affine_matrix = faceutil.warp_face_by_bounding_box_for_landmark_68(img, bbox, (256, 256))
        else:
            crop_image, affine_matrix = faceutil.warp_face_by_face_landmark_5(img, det_kpss, 256, mode='arcface128', interpolation=v2.InterpolationMode.BILINEAR)

        crop_image = crop_image.to(dtype=torch.float32)
        crop_image = torch.div(crop_image, 255.0)
        crop_image = torch.unsqueeze(crop_image, 0).contiguous()

        io_binding = self.models_processor.models['FaceLandmark68'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32,  shape=crop_image.size(), buffer_ptr=crop_image.data_ptr())

        io_binding.bind_output('landmarks_xyscore', self.models_processor.device)
        io_binding.bind_output('heatmaps', self.models_processor.device)

        # Sync and run model
        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['FaceLandmark68'].run_with_iobinding(io_binding)
        net_outs = io_binding.copy_outputs_to_cpu()
        face_landmark_68 = net_outs[0]
        face_heatmap = net_outs[1]

        face_landmark_68 = face_landmark_68[:, :, :2][0] / 64.0
        face_landmark_68 = face_landmark_68.reshape(1, -1, 2) * 256.0
        face_landmark_68 = cv2.transform(face_landmark_68, cv2.invertAffineTransform(affine_matrix))

        face_landmark_68 = face_landmark_68.reshape(-1, 2)
        face_landmark_68_score = np.amax(face_heatmap, axis = (2, 3))
        face_landmark_68_score = face_landmark_68_score.reshape(-1, 1)

        face_landmark_68_5, face_landmark_68_score = faceutil.convert_face_landmark_68_to_5(face_landmark_68, face_landmark_68_score)

        return face_landmark_68_5, face_landmark_68, face_landmark_68_score

    def detect_face_landmark_3d68(self, img, bbox, det_kpss, from_points=False):
        if from_points == False:
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = 192  / (max(w, h)*1.5)
            aimg, M = faceutil.transform(img, center, 192, _scale, rotate)
        else:
            aimg, M = faceutil.warp_face_by_face_landmark_5(img, det_kpss, image_size=192, mode='arcface128', interpolation=v2.InterpolationMode.BILINEAR)

        aimg = torch.unsqueeze(aimg, 0).contiguous()
        aimg = aimg.to(dtype=torch.float32)
        aimg = self.models_processor.normalize(aimg)
        io_binding = self.models_processor.models['FaceLandmark3d68'].io_binding()
        io_binding.bind_input(name='data', device_type=self.models_processor.device, device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

        io_binding.bind_output('fc1', self.models_processor.device)

        # Sync and run model
        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['FaceLandmark3d68'].run_with_iobinding(io_binding)
        pred = io_binding.copy_outputs_to_cpu()[0][0]

        if pred.shape[0] >= 3000:
            pred = pred.reshape((-1, 3))
        else:
            pred = pred.reshape((-1, 2))
        if 68 < pred.shape[0]:
            pred = pred[68*-1:,:]
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (192 // 2)
        if pred.shape[1] == 3:
            pred[:, 2] *= (192 // 2)

        IM = faceutil.invertAffineTransform(M)
        pred = faceutil.trans_points3d(pred, IM)

        # at moment we don't use 3d points
        
        #'''
        #P = faceutil.estimate_affine_matrix_3d23d(self.models_processor.mean_lmk, pred)
        #s, R, t = faceutil.P2sRt(P)
        #rx, ry, rz = faceutil.matrix2angle(R)
        #pose = np.array( [rx, ry, rz], dtype=np.float32 ) #pitch, yaw, roll
        #'''

        # convert from 3d68 to 2d68 keypoints
        landmark2d68 = np.array(pred[:, [0, 1]])

        # convert from 68 to 5 keypoints
        landmark2d68_5, _ = faceutil.convert_face_landmark_68_to_5(landmark2d68, [])

        return landmark2d68_5, landmark2d68, []

    def detect_face_landmark_98(self, img, bbox, det_kpss, from_points=False):
        if from_points == False:
            crop_image, detail = faceutil.warp_face_by_bounding_box_for_landmark_98(img, bbox, (256, 256))
        else:
            crop_image, M = faceutil.warp_face_by_face_landmark_5(img, det_kpss, image_size=256, mode='arcface128', interpolation=v2.InterpolationMode.BILINEAR)
            h, w = (crop_image.size(dim=1), crop_image.size(dim=2))

        landmark = []
        landmark_5 = []
        landmark_score = []
        if crop_image is not None:
            crop_image = crop_image.to(dtype=torch.float32)
            crop_image = torch.div(crop_image, 255.0)
            crop_image = torch.unsqueeze(crop_image, 0).contiguous()

            io_binding = self.models_processor.models['FaceLandmark98'].io_binding()
            io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32,  shape=crop_image.size(), buffer_ptr=crop_image.data_ptr())

            io_binding.bind_output('landmarks_xyscore', self.models_processor.device)

            # Sync and run model
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()
            self.models_processor.models['FaceLandmark98'].run_with_iobinding(io_binding)
            landmarks_xyscore = io_binding.copy_outputs_to_cpu()[0]

            if len(landmarks_xyscore) > 0:
                for one_face_landmarks in landmarks_xyscore:
                    landmark_score = one_face_landmarks[:, [2]].reshape(-1)
                    landmark = one_face_landmarks[:, [0, 1]].reshape(-1,2)

                    ##recover, and grouped as [98,2]
                    if from_points == False:
                        landmark[:, 0] = landmark[:, 0] * detail[1] + detail[3] - detail[4]
                        landmark[:, 1] = landmark[:, 1] * detail[0] + detail[2] - detail[4]
                    else:
                        landmark[:, 0] = landmark[:, 0] * w
                        landmark[:, 1] = landmark[:, 1] * h

                        IM = faceutil.invertAffineTransform(M)
                        landmark = faceutil.trans_points2d(landmark, IM)

                    landmark_5, landmark_score = faceutil.convert_face_landmark_98_to_5(landmark, landmark_score)

        return landmark_5, landmark, landmark_score

    def detect_face_landmark_106(self, img, bbox, det_kpss, from_points=False):
        if from_points == False:
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = 192  / (max(w, h)*1.5)
            #print('param:', img.size(), bbox, center, (192, 192), _scale, rotate)
            aimg, M = faceutil.transform(img, center, 192, _scale, rotate)
        else:
            aimg, M = faceutil.warp_face_by_face_landmark_5(img, det_kpss, image_size=192, mode='arcface128', interpolation=v2.InterpolationMode.BILINEAR)

        aimg = torch.unsqueeze(aimg, 0).contiguous()
        aimg = aimg.to(dtype=torch.float32)
        aimg = self.models_processor.normalize(aimg)
        io_binding = self.models_processor.models['FaceLandmark106'].io_binding()
        io_binding.bind_input(name='data', device_type=self.models_processor.device, device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

        io_binding.bind_output('fc1', self.models_processor.device)

        # Sync and run model
        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['FaceLandmark106'].run_with_iobinding(io_binding)
        pred = io_binding.copy_outputs_to_cpu()[0][0]

        if pred.shape[0] >= 3000:
            pred = pred.reshape((-1, 3))
        else:
            pred = pred.reshape((-1, 2))

        if 106 < pred.shape[0]:
            pred = pred[106*-1:,:]

        pred[:, 0:2] += 1
        pred[:, 0:2] *= (192 // 2)
        if pred.shape[1] == 3:
            pred[:, 2] *= (192 // 2)

        IM = faceutil.invertAffineTransform(M)
        pred = faceutil.trans_points(pred, IM)

        pred_5 = []
        if pred is not None:
            # convert from 106 to 5 keypoints
            pred_5 = faceutil.convert_face_landmark_106_to_5(pred)

        return pred_5, pred, []

    def detect_face_landmark_203(self, img, bbox, det_kpss, from_points=False):
        if from_points == False:
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = 224  / (max(w, h)*1.5)

            aimg, M = faceutil.transform(img, center, 224, _scale, rotate)
        elif len(det_kpss) == 0:
            return [], [], []
        else:
            if det_kpss.shape[0] == 5:
                aimg, M = faceutil.warp_face_by_face_landmark_5(img, det_kpss, image_size=224, mode='arcface128', interpolation=v2.InterpolationMode.BILINEAR)
            else:
                aimg, M, IM = faceutil.warp_face_by_face_landmark_x(img, det_kpss, dsize=224, scale=1.5, vy_ratio=-0.1, interpolation=v2.InterpolationMode.BILINEAR)

        aimg = torch.unsqueeze(aimg, 0).contiguous()
        aimg = aimg.to(dtype=torch.float32)
        aimg = torch.div(aimg, 255.0)
        io_binding = self.models_processor.models['FaceLandmark203'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

        io_binding.bind_output('output', self.models_processor.device)
        io_binding.bind_output('853', self.models_processor.device)
        io_binding.bind_output('856', self.models_processor.device)

        # Sync and run model
        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['FaceLandmark203'].run_with_iobinding(io_binding)
        out_lst = io_binding.copy_outputs_to_cpu()
        out_pts = out_lst[2]

        out_pts = out_pts.reshape((-1, 2)) * 224.0

        IM=None
        if len(det_kpss) == 0 or det_kpss.shape[0] == 5:
            IM = faceutil.invertAffineTransform(M)

        out_pts = faceutil.trans_points(out_pts, IM)

        out_pts_5 = []
        if out_pts is not None:
            # convert from 203 to 5 keypoints
            out_pts_5 = faceutil.convert_face_landmark_203_to_5(out_pts)

        return out_pts_5, out_pts, []

    def detect_face_landmark_478(self, img, bbox, det_kpss, from_points=False):
        if from_points == False:
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = 256.0  / (max(w, h)*1.5)
            #print('param:', img.size(), bbox, center, (192, 192), _scale, rotate)
            aimg, M = faceutil.transform(img, center, 256, _scale, rotate)
        else:
            aimg, M = faceutil.warp_face_by_face_landmark_5(img, det_kpss, 256, mode='arcfacemap', interpolation=v2.InterpolationMode.BILINEAR)

        aimg = torch.unsqueeze(aimg, 0).contiguous()
        aimg = aimg.to(dtype=torch.float32)
        aimg = torch.div(aimg, 255.0)
        io_binding = self.models_processor.models['FaceLandmark478'].io_binding()
        io_binding.bind_input(name='input_12', device_type=self.models_processor.device, device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

        io_binding.bind_output('Identity', self.models_processor.device)
        io_binding.bind_output('Identity_1', self.models_processor.device)
        io_binding.bind_output('Identity_2', self.models_processor.device)

        # Sync and run model
        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['FaceLandmark478'].run_with_iobinding(io_binding)
        landmarks, faceflag, blendshapes = io_binding.copy_outputs_to_cpu() # pylint: disable=unused-variable
        landmarks = landmarks.reshape( (1,478,3))

        landmark = []
        landmark_5 = []
        landmark_score = [] # pylint: disable=unused-variable
        if len(landmarks) > 0:
            for one_face_landmarks in landmarks:
                landmark = one_face_landmarks
                IM = faceutil.invertAffineTransform(M)
                landmark = faceutil.trans_points3d(landmark, IM)

                #'''
                #P = faceutil.estimate_affine_matrix_3d23d(self.models_processor.mean_lmk, landmark)
                #s, R, t = faceutil.P2sRt(P)
                #rx, ry, rz = faceutil.matrix2angle(R)
                #pose = np.array( [rx, ry, rz], dtype=np.float32 ) #pitch, yaw, roll
                #'''
                landmark = landmark[:, [0, 1]].reshape(-1,2)

                #get scores
                landmark_for_score = landmark[self.models_processor.LandmarksSubsetIdxs]
                landmark_for_score = landmark_for_score[:, :2]
                landmark_for_score = np.expand_dims(landmark_for_score, axis=0)
                landmark_for_score = landmark_for_score.astype(np.float32)
                landmark_for_score = torch.from_numpy(landmark_for_score).to(self.models_processor.device)

                io_binding_bs = self.models_processor.models['FaceBlendShapes'].io_binding()
                io_binding_bs.bind_input(name='input_points', device_type=self.models_processor.device, device_id=0, element_type=np.float32,  shape=tuple(landmark_for_score.shape), buffer_ptr=landmark_for_score.data_ptr())
                io_binding_bs.bind_output('output', self.models_processor.device)

                # Sync and run model
                if self.models_processor.device == "cuda":
                    torch.cuda.synchronize()
                elif self.models_processor.device != "cpu":
                    self.models_processor.syncvec.cpu()
                self.models_processor.models['FaceBlendShapes'].run_with_iobinding(io_binding_bs)
                landmark_score = io_binding_bs.copy_outputs_to_cpu()[0] # pylint: disable=unused-variable

                # convert from 478 to 5 keypoints
                landmark_5 = faceutil.convert_face_landmark_478_to_5(landmark)

        #return landmark, landmark_score
        return landmark_5, landmark, []