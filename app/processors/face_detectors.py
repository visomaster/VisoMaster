from typing import TYPE_CHECKING

import torch
from torchvision.transforms import v2
import numpy as np

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor

from app.processors.utils import faceutil

class FaceDetectors:
    def __init__(self, models_processor: 'ModelsProcessor'):
        self.models_processor = models_processor

    def run_detect(self, img, detect_mode='RetinaFace', max_num=1, score=0.5, input_size=(512, 512), use_landmark_detection=False, landmark_detect_mode='203', landmark_score=0.5, from_points=False, rotation_angles=None):
        rotation_angles = rotation_angles or [0]
        bboxes = []
        kpss_5 = []
        kpss = []

        if detect_mode=='RetinaFace':
            if not self.models_processor.models['RetinaFace']:
                self.models_processor.models['RetinaFace'] = self.models_processor.load_model('RetinaFace')

            bboxes, kpss_5, kpss = self.detect_retinaface(img, max_num=max_num, score=score, input_size=input_size, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points, rotation_angles=rotation_angles)

        elif detect_mode=='SCRFD':
            if not self.models_processor.models['SCRFD2.5g']:
                self.models_processor.models['SCRFD2.5g'] = self.models_processor.load_model('SCRFD2.5g')

            bboxes, kpss_5, kpss = self.detect_scrdf(img, max_num=max_num, score=score, input_size=input_size, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points, rotation_angles=rotation_angles)

        elif detect_mode=='Yolov8':
            if not self.models_processor.models['YoloFace8n']:
                self.models_processor.models['YoloFace8n'] = self.models_processor.load_model('YoloFace8n')

            bboxes, kpss_5, kpss = self.detect_yoloface(img, max_num=max_num, score=score, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points, rotation_angles=rotation_angles)

        elif detect_mode=='Yunet':
            if not self.models_processor.models['YunetN']:
                self.models_processor.models['YunetN'] = self.models_processor.load_model('YunetN')

            bboxes, kpss_5, kpss = self.detect_yunet(img, max_num=max_num, score=score, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points, rotation_angles=rotation_angles)

        return bboxes, kpss_5, kpss

    def detect_retinaface(self, img, max_num, score, input_size, use_landmark_detection, landmark_detect_mode, landmark_score, from_points, rotation_angles=None):
        rotation_angles = rotation_angles or [0]
        img_landmark = None
        if use_landmark_detection:
            img_landmark = img.clone()

        # Resize image to fit within the input_size
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        img_height, img_width = (img.size()[1], img.size()[2])
        im_ratio = torch.div(img_height, img_width)

        model_ratio = 1.0
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = torch.div(new_height,  img.size()[1])

        resize = v2.Resize((new_height, new_width), antialias=True)
        img = resize(img)
        img = img.permute(1,2,0)

        det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.float32, device=self.models_processor.device)
        det_img[:new_height,:new_width,  :] = img

        # Switch to RGB and normalize
        #det_img = det_img[:, :, [2,1,0]]
        det_img = torch.sub(det_img, 127.5)
        det_img = torch.div(det_img, 128.0)
        det_img = det_img.permute(2, 0, 1) #3,128,128

        scores_list = []
        bboxes_list = []
        kpss_list = []

        cx = input_size[0] / 2  # image center x coordinate
        cy = input_size[1] / 2  # image center y coordinate

        if len(rotation_angles) > 1:
            do_rotation = True
        else:
            do_rotation = False
            
        for angle in rotation_angles:
            # Prepare data and find model parameters
            if angle != 0:
                aimg, M = faceutil.transform(det_img, (cx, cy), 640, 1.0, angle)
                IM = faceutil.invertAffineTransform(M)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
            else:
                IM = None
                aimg = torch.unsqueeze(det_img, 0).contiguous()

            io_binding = self.models_processor.models['RetinaFace'].io_binding()
            io_binding.bind_input(name='input.1', device_type=self.models_processor.device, device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

            io_binding.bind_output('448', self.models_processor.device)
            io_binding.bind_output('471', self.models_processor.device)
            io_binding.bind_output('494', self.models_processor.device)
            io_binding.bind_output('451', self.models_processor.device)
            io_binding.bind_output('474', self.models_processor.device)
            io_binding.bind_output('497', self.models_processor.device)
            io_binding.bind_output('454', self.models_processor.device)
            io_binding.bind_output('477', self.models_processor.device)
            io_binding.bind_output('500', self.models_processor.device)

            # Sync and run model
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()
            self.models_processor.models['RetinaFace'].run_with_iobinding(io_binding)

            net_outs = io_binding.copy_outputs_to_cpu()

            input_height = aimg.shape[2]
            input_width = aimg.shape[3]

            fmc = 3
            center_cache = {}
            for idx, stride in enumerate([8, 16, 32]):
                scores = net_outs[idx]
                bbox_preds = net_outs[idx+fmc]
                bbox_preds = bbox_preds * stride

                kps_preds = net_outs[idx+fmc*2] * stride # Corrected variable name here
                height = input_height // stride
                width = input_width // stride
                key = (height, width, stride)
                if key in center_cache:
                    anchor_centers = center_cache[key]
                else:
                    anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                    anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                    anchor_centers = np.stack([anchor_centers]*2, axis=1).reshape( (-1,2) )
                    if len(center_cache)<100:
                        center_cache[key] = anchor_centers

                pos_inds = np.where(scores>=score)[0]

                x1 = anchor_centers[:, 0] - bbox_preds[:, 0]
                y1 = anchor_centers[:, 1] - bbox_preds[:, 1]
                x2 = anchor_centers[:, 0] + bbox_preds[:, 2]
                y2 = anchor_centers[:, 1] + bbox_preds[:, 3]

                bboxes = np.stack([x1, y1, x2, y2], axis=-1)

                pos_scores = scores[pos_inds]
                pos_bboxes = bboxes[pos_inds]

                # bboxes
                if angle != 0:
                    if len(pos_bboxes) > 0:
                        # Split the points into coordinates (x1, y1) and (x2, y2)
                        points1 = pos_bboxes[:, :2]  # (x1, y1)
                        points2 = pos_bboxes[:, 2:]  # (x2, y2)

                        # Apply the inverse of the rotation matrix to points1 and points2
                        points1 = faceutil.trans_points2d(points1, IM)
                        points2 = faceutil.trans_points2d(points2, IM)

                        _x1 = points1[:, 0]
                        _y1 = points1[:, 1]
                        _x2 = points2[:, 0]
                        _y2 = points2[:, 1]

                        if angle in (-270, 90):
                            # x1, y2, x2, y1
                            points1 = np.stack((_x1, _y2), axis=1)
                            points2 = np.stack((_x2, _y1), axis=1)
                        elif angle in (-180, 180):
                            # x2, y2, x1, y1
                            points1 = np.stack((_x2, _y2), axis=1)
                            points2 = np.stack((_x1, _y1), axis=1)
                        elif angle in (-90, 270):
                            # x2, y1, x1, y2
                            points1 = np.stack((_x2, _y1), axis=1)
                            points2 = np.stack((_x1, _y2), axis=1)

                        # Reassemble the transformed points into the format [x1', y1', x2', y2']
                        pos_bboxes = np.hstack((points1, points2))

                # kpss
                preds = []
                for i in range(0, kps_preds.shape[1], 2):
                    px = anchor_centers[:, i%2] + kps_preds[:, i]
                    py = anchor_centers[:, i%2+1] + kps_preds[:, i+1]

                    preds.append(px)
                    preds.append(py)
                kpss = np.stack(preds, axis=-1)
                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]

                if do_rotation:
                    for i in range(len(pos_kpss)):
                        face_size = max(pos_bboxes[i][2] - pos_bboxes[i][0], pos_bboxes[i][3] - pos_bboxes[i][1])
                        angle_deg_to_front = faceutil.get_face_orientation(face_size, pos_kpss[i])
                        if angle_deg_to_front < -50.00 or angle_deg_to_front > 50.00:
                            pos_scores[i] = 0.0

                        if angle != 0:
                            pos_kpss[i] = faceutil.trans_points2d(pos_kpss[i], IM)

                    pos_inds = np.where(pos_scores>=score)[0]
                    pos_scores = pos_scores[pos_inds]
                    pos_bboxes = pos_bboxes[pos_inds]
                    pos_kpss = pos_kpss[pos_inds]

                kpss_list.append(pos_kpss)
                bboxes_list.append(pos_bboxes)
                scores_list.append(pos_scores)

        if len(bboxes_list) == 0:
            return [], [], []

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        det_scale = det_scale.numpy()###

        bboxes = np.vstack(bboxes_list) / det_scale

        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        dets = pre_det
        thresh = 0.4
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scoresb = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        orderb = scoresb.argsort()[::-1]

        keep = []
        
        # Add integer conversion for max_num here:
        max_num = int(max_num) # Ensure max_num is an integer before comparison

        while orderb.size > 0:
            i = orderb[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[orderb[1:]])
            yy1 = np.maximum(y1[i], y1[orderb[1:]])
            xx2 = np.minimum(x2[i], x2[orderb[1:]])
            yy2 = np.minimum(y2[i], y2[orderb[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (areas[i] + areas[orderb[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            orderb = orderb[inds + 1]

        det = pre_det[keep, :]

        kpss = kpss[order,:,:]
        kpss = kpss[keep,:,:]

        #if max_num > 0 and det.shape[0] > max_num: # OLD LINE - No longer needed
        if max_num > 0 and det.shape[0] > 1:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            det_img_center = img_height // 2, img_width // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - det_img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - det_img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]

            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]

        score_values = det[:, 4]
        # delete score column
        det = np.delete(det, 4, 1)

        kpss_5 = kpss.copy()
        if use_landmark_detection and len(kpss_5) > 0:
            kpss = []
            for i in range(kpss_5.shape[0]):
                landmark_kpss_5, landmark_kpss, landmark_scores = self.models_processor.run_detect_landmark(img_landmark, det[i], kpss_5[i], landmark_detect_mode, landmark_score, from_points)
                # Always add to kpss, regardless of the length of landmark_kpss.
                kpss.append(landmark_kpss if len(landmark_kpss) > 0 else kpss_5[i])
                if len(landmark_kpss_5) > 0:
                    if len(landmark_scores) > 0:
                        if np.mean(landmark_scores) > np.mean(score_values[i]):
                            kpss_5[i] = landmark_kpss_5
                    else:
                        kpss_5[i] = landmark_kpss_5
            kpss = np.array(kpss, dtype=object)

        return det, kpss_5, kpss  

    def detect_scrdf(self, img, max_num, score, input_size, use_landmark_detection, landmark_detect_mode, landmark_score, from_points, rotation_angles=None):
        rotation_angles = rotation_angles or [0]
        img_landmark = None
        if use_landmark_detection:
            img_landmark = img.clone()

        # Resize image to fit within the input_size
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        img_height, img_width = (img.size()[1], img.size()[2])
        im_ratio = torch.div(img_height, img_width)

        model_ratio = 1.0
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = torch.div(new_height,  img.size()[1])

        resize = v2.Resize((new_height, new_width), antialias=True)
        img = resize(img)
        img = img.permute(1,2,0)

        det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.float32, device=self.models_processor.device)
        det_img[:new_height,:new_width,  :] = img

        # Switch to RGB and normalize
        #det_img = det_img[:, :, [2,1,0]]
        det_img = torch.sub(det_img, 127.5)
        det_img = torch.div(det_img, 128.0)
        det_img = det_img.permute(2, 0, 1) #3,128,128

        scores_list = []
        bboxes_list = []
        kpss_list = []

        cx = input_size[0] / 2  # image center x coordinate
        cy = input_size[1] / 2  # image center y coordinate

        if len(rotation_angles) > 1:
            do_rotation = True
        else:
            do_rotation = False

        input_name = self.models_processor.models['SCRFD2.5g'].get_inputs()[0].name
        outputs = self.models_processor.models['SCRFD2.5g'].get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)

        for angle in rotation_angles:
            # Prepare data and find model parameters
            if angle != 0:
                aimg, M = faceutil.transform(det_img, (cx, cy), 640, 1.0, angle)
                IM = faceutil.invertAffineTransform(M)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
            else:
                IM = None
                aimg = torch.unsqueeze(det_img, 0).contiguous()

            io_binding = self.models_processor.models['SCRFD2.5g'].io_binding()
            io_binding.bind_input(name=input_name, device_type=self.models_processor.device, device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

            for i in range(len(output_names)):
                io_binding.bind_output(output_names[i], self.models_processor.device)

            # Sync and run model
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()
            self.models_processor.models['SCRFD2.5g'].run_with_iobinding(io_binding)

            net_outs = io_binding.copy_outputs_to_cpu()

            input_height = aimg.shape[2]
            input_width = aimg.shape[3]

            fmc = 3
            center_cache = {}
            for idx, stride in enumerate([8, 16, 32]):
                scores = net_outs[idx]
                bbox_preds = net_outs[idx+fmc]
                bbox_preds = bbox_preds * stride

                kps_preds = net_outs[idx+fmc*2] * stride
                height = input_height // stride
                width = input_width // stride
                key = (height, width, stride)
                if key in center_cache:
                    anchor_centers = center_cache[key]
                else:
                    anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                    anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                    anchor_centers = np.stack([anchor_centers]*2, axis=1).reshape( (-1,2) )
                    if len(center_cache)<100:
                        center_cache[key] = anchor_centers

                pos_inds = np.where(scores>=score)[0]

                x1 = anchor_centers[:, 0] - bbox_preds[:, 0]
                y1 = anchor_centers[:, 1] - bbox_preds[:, 1]
                x2 = anchor_centers[:, 0] + bbox_preds[:, 2]
                y2 = anchor_centers[:, 1] + bbox_preds[:, 3]

                bboxes = np.stack([x1, y1, x2, y2], axis=-1)

                pos_scores = scores[pos_inds]
                pos_bboxes = bboxes[pos_inds]

                # bboxes
                if angle != 0:
                    if len(pos_bboxes) > 0:
                        # Split the points into coordinates (x1, y1) and (x2, y2)
                        points1 = pos_bboxes[:, :2]  # (x1, y1)
                        points2 = pos_bboxes[:, 2:]  # (x2, y2)

                        # Apply the inverse of the rotation matrix to points1 and points2
                        points1 = faceutil.trans_points2d(points1, IM)
                        points2 = faceutil.trans_points2d(points2, IM)

                        _x1 = points1[:, 0]
                        _y1 = points1[:, 1]
                        _x2 = points2[:, 0]
                        _y2 = points2[:, 1]

                        if angle in (-270, 90):
                            # x1, y2, x2, y1
                            points1 = np.stack((_x1, _y2), axis=1)
                            points2 = np.stack((_x2, _y1), axis=1)
                        elif angle in (-180, 180):
                            # x2, y2, x1, y1
                            points1 = np.stack((_x2, _y2), axis=1)
                            points2 = np.stack((_x1, _y1), axis=1)
                        elif angle in (-90, 270):
                            # x2, y1, x1, y2
                            points1 = np.stack((_x2, _y1), axis=1)
                            points2 = np.stack((_x1, _y2), axis=1)

                        # Reassemble the transformed points into the format [x1', y1', x2', y2']
                        pos_bboxes = np.hstack((points1, points2))

                # kpss
                preds = []
                for i in range(0, kps_preds.shape[1], 2):
                    px = anchor_centers[:, i%2] + kps_preds[:, i]
                    py = anchor_centers[:, i%2+1] + kps_preds[:, i+1]

                    preds.append(px)
                    preds.append(py)
                kpss = np.stack(preds, axis=-1)
                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]

                if do_rotation:
                    for i in range(len(pos_kpss)):
                        face_size = max(pos_bboxes[i][2] - pos_bboxes[i][0], pos_bboxes[i][3] - pos_bboxes[i][1])
                        angle_deg_to_front = faceutil.get_face_orientation(face_size, pos_kpss[i])
                        if angle_deg_to_front < -50.00 or angle_deg_to_front > 50.00:
                            pos_scores[i] = 0.0

                        if angle != 0:
                            pos_kpss[i] = faceutil.trans_points2d(pos_kpss[i], IM)

                    pos_inds = np.where(pos_scores>=score)[0]
                    pos_scores = pos_scores[pos_inds]
                    pos_bboxes = pos_bboxes[pos_inds]
                    pos_kpss = pos_kpss[pos_inds]

                kpss_list.append(pos_kpss)
                bboxes_list.append(pos_bboxes)
                scores_list.append(pos_scores)

        if len(bboxes_list) == 0:
            return [], [], []

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        det_scale = det_scale.numpy()###

        bboxes = np.vstack(bboxes_list) / det_scale

        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        dets = pre_det
        thresh = 0.4
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scoresb = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        orderb = scoresb.argsort()[::-1]

        keep = []
        while orderb.size > 0:
            i = orderb[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[orderb[1:]])
            yy1 = np.maximum(y1[i], y1[orderb[1:]])
            xx2 = np.minimum(x2[i], x2[orderb[1:]])
            yy2 = np.minimum(y2[i], y2[orderb[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (areas[i] + areas[orderb[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            orderb = orderb[inds + 1]

        det = pre_det[keep, :]

        kpss = kpss[order,:,:]
        kpss = kpss[keep,:,:]

        #if max_num > 0 and det.shape[0] > max_num:
        if max_num > 0 and det.shape[0] > 1:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            det_img_center = img_height // 2, img_width // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - det_img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - det_img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]

            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]

        score_values = det[:, 4]
        # delete score column
        det = np.delete(det, 4, 1)

        kpss_5 = kpss.copy()
        if use_landmark_detection and len(kpss_5) > 0:
            kpss = []
            for i in range(kpss_5.shape[0]):
                landmark_kpss_5, landmark_kpss, landmark_scores = self.models_processor.run_detect_landmark(img_landmark, det[i], kpss_5[i], landmark_detect_mode, landmark_score, from_points)
                # Always add to kpss, regardless of the length of landmark_kpss.
                kpss.append(landmark_kpss if len(landmark_kpss) > 0 else kpss_5[i])
                if len(landmark_kpss_5) > 0:
                    if len(landmark_scores) > 0:
                        if np.mean(landmark_scores) > np.mean(score_values[i]):
                            kpss_5[i] = landmark_kpss_5
                    else:
                        kpss_5[i] = landmark_kpss_5
            kpss = np.array(kpss, dtype=object)

        return det, kpss_5, kpss

    def detect_yoloface(self, img, max_num, score, use_landmark_detection, landmark_detect_mode, landmark_score, from_points, rotation_angles=None):
        rotation_angles = rotation_angles or [0]
        img_landmark = None
        if use_landmark_detection:
            img_landmark = img.clone()

        # Resize image to fit within the input_size
        input_size = (640, 640)
        img_height, img_width = (img.size()[1], img.size()[2])
        im_ratio = torch.div(img_height, img_width)

        # model_ratio = float(input_size[1]) / input_size[0]
        model_ratio = 1.0
        if im_ratio>model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = torch.div(new_height,  img.size()[1])

        resize = v2.Resize((new_height, new_width), antialias=True)
        img = resize(img)
        img = img.permute(1,2,0)

        det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.uint8, device=self.models_processor.device)
        det_img[:new_height,:new_width,  :] = img

        det_img = det_img.permute(2, 0, 1)

        scores_list = []
        bboxes_list = []
        kpss_list = []

        cx = input_size[0] / 2  # image center x coordinate
        cy = input_size[1] / 2  # image center y coordinate

        if len(rotation_angles) > 1:
            do_rotation = True
        else:
            do_rotation = False

        for angle in rotation_angles:
            # Prepare data and find model parameters
            if angle != 0:
                aimg, M = faceutil.transform(det_img, (cx, cy), 640, 1.0, angle)
                IM = faceutil.invertAffineTransform(M)
                aimg = aimg.permute(1, 2, 0)
                aimg = torch.div(aimg, 255.0)
                aimg = aimg.permute(2, 0, 1)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
            else:
                aimg = det_img.permute(1, 2, 0)
                aimg = torch.div(aimg, 255.0)
                aimg = aimg.permute(2, 0, 1)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
                IM = None

            io_binding = self.models_processor.models['YoloFace8n'].io_binding()
            io_binding.bind_input(name='images', device_type=self.models_processor.device, device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())
            io_binding.bind_output('output0', self.models_processor.device)

            # Sync and run model
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()
            self.models_processor.models['YoloFace8n'].run_with_iobinding(io_binding)

            net_outs = io_binding.copy_outputs_to_cpu()

            outputs = np.squeeze(net_outs).T

            bbox_raw, score_raw, kps_raw, *_ = np.split(outputs, [4, 5], axis=1)

            keep_indices = np.where(score_raw > score)[0]

            if keep_indices.any():
                bbox_raw, kps_raw, score_raw = bbox_raw[keep_indices], kps_raw[keep_indices], score_raw[keep_indices]

                # Compute the transformed bounding box coordinates
                x1 = bbox_raw[:, 0] - bbox_raw[:, 2] / 2
                y1 = bbox_raw[:, 1] - bbox_raw[:, 3] / 2
                x2 = bbox_raw[:, 0] + bbox_raw[:, 2] / 2
                y2 = bbox_raw[:, 1] + bbox_raw[:, 3] / 2

                # Stack the results into a single array
                bboxes_raw = np.stack((x1, y1, x2, y2), axis=-1)

                # bboxes
                if angle != 0:
                    if len(bboxes_raw) > 0:
                        # Split the points into coordinates (x1, y1) and (x2, y2)
                        points1 = bboxes_raw[:, :2]  # (x1, y1)
                        points2 = bboxes_raw[:, 2:]  # (x2, y2)

                        # Apply the inverse of the rotation matrix to points1 and points2
                        points1 = faceutil.trans_points2d(points1, IM)
                        points2 = faceutil.trans_points2d(points2, IM)

                        _x1 = points1[:, 0]
                        _y1 = points1[:, 1]
                        _x2 = points2[:, 0]
                        _y2 = points2[:, 1]

                        if angle in (-270, 90):
                            # x1, y2, x2, y1
                            points1 = np.stack((_x1, _y2), axis=1)
                            points2 = np.stack((_x2, _y1), axis=1)
                        elif angle in (-180, 180):
                            # x2, y2, x1, y1
                            points1 = np.stack((_x2, _y2), axis=1)
                            points2 = np.stack((_x1, _y1), axis=1)
                        elif angle in (-90, 270):
                            # x2, y1, x1, y2
                            points1 = np.stack((_x2, _y1), axis=1)
                            points2 = np.stack((_x1, _y2), axis=1)

                        # Reassemble the transformed points into the format [x1', y1', x2', y2']
                        bboxes_raw = np.hstack((points1, points2))

                kps_list = []
                for kps in kps_raw:
                    indexes = np.arange(0, len(kps), 3)
                    temp_kps = []
                    for index in indexes:
                        temp_kps.append([kps[index], kps[index + 1]])
                    kps_list.append(np.array(temp_kps))

                kpss_raw = np.stack(kps_list)

                if do_rotation:
                    for i in range(len(kpss_raw)):
                        face_size = max(bboxes_raw[i][2] - bboxes_raw[i][0], bboxes_raw[i][3] - bboxes_raw[i][1])
                        angle_deg_to_front = faceutil.get_face_orientation(face_size, kpss_raw[i])
                        if angle_deg_to_front < -50.00 or angle_deg_to_front > 50.00:
                            score_raw[i] = 0.0

                        if angle != 0:
                            kpss_raw[i] = faceutil.trans_points2d(kpss_raw[i], IM)

                    keep_indices = np.where(score_raw>=score)[0]
                    score_raw = score_raw[keep_indices]
                    bboxes_raw = bboxes_raw[keep_indices]
                    kpss_raw = kpss_raw[keep_indices]

                kpss_list.append(kpss_raw)
                bboxes_list.append(bboxes_raw)
                scores_list.append(score_raw)

        if len(bboxes_list) == 0:
            return [], [], []

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        det_scale = det_scale.numpy()###

        bboxes = np.vstack(bboxes_list) / det_scale

        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        dets = pre_det
        thresh = 0.4
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scoresb = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        orderb = scoresb.argsort()[::-1]

        keep = []
        while orderb.size > 0:
            i = orderb[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[orderb[1:]])
            yy1 = np.maximum(y1[i], y1[orderb[1:]])
            xx2 = np.minimum(x2[i], x2[orderb[1:]])
            yy2 = np.minimum(y2[i], y2[orderb[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (areas[i] + areas[orderb[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            orderb = orderb[inds + 1]

        det = pre_det[keep, :]

        kpss = kpss[order,:,:]
        kpss = kpss[keep,:,:]

        #if max_num > 0 and det.shape[0] > max_num:
        if max_num > 0 and det.shape[0] > 1:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            det_img_center = img_height // 2, img_width // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - det_img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - det_img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]

            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]

        score_values = det[:, 4]
        # delete score column
        det = np.delete(det, 4, 1)

        kpss_5 = kpss.copy()
        if use_landmark_detection and len(kpss_5) > 0:
            kpss = []
            for i in range(kpss_5.shape[0]):
                landmark_kpss_5, landmark_kpss, landmark_scores = self.models_processor.run_detect_landmark(img_landmark, det[i], kpss_5[i], landmark_detect_mode, landmark_score, from_points)
                # Always add to kpss, regardless of the length of landmark_kpss.
                kpss.append(landmark_kpss if len(landmark_kpss) > 0 else kpss_5[i])
                if len(landmark_kpss_5) > 0:
                    if len(landmark_scores) > 0:
                        if np.mean(landmark_scores) > np.mean(score_values[i]):
                            kpss_5[i] = landmark_kpss_5
                    else:
                        kpss_5[i] = landmark_kpss_5
            kpss = np.array(kpss, dtype=object)

        return det, kpss_5, kpss

    def detect_yunet(self, img, max_num, score, use_landmark_detection, landmark_detect_mode, landmark_score, from_points, rotation_angles=None):
        rotation_angles = rotation_angles or [0]
        img_landmark = None
        if use_landmark_detection:
            img_landmark = img.clone()

        # Resize image to fit within the input_size
        input_size = (640, 640)
        img_height, img_width = (img.size()[1], img.size()[2])
        im_ratio = torch.div(img_height, img_width)

        model_ratio = 1.0
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = torch.div(new_height,  img.size()[1])

        resize = v2.Resize((new_height, new_width), antialias=False)
        img = resize(img)

        img = img.permute(1,2,0)

        det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.uint8, device=self.models_processor.device)
        det_img[:new_height,:new_width,  :] = img

        # Switch to BGR
        det_img = det_img[:, :, [2,1,0]]

        det_img = det_img.permute(2, 0, 1) #3,640,640

        scores_list = []
        bboxes_list = []
        kpss_list = []

        cx = input_size[0] / 2  # image center x coordinate
        cy = input_size[1] / 2  # image center y coordinate

        if len(rotation_angles) > 1:
            do_rotation = True
        else:
            do_rotation = False

        input_name = self.models_processor.models['YunetN'].get_inputs()[0].name
        outputs = self.models_processor.models['YunetN'].get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)

        for angle in rotation_angles:
            # Prepare data and find model parameters
            if angle != 0:
                aimg, M = faceutil.transform(det_img, (cx, cy), 640, 1.0, angle)
                IM = faceutil.invertAffineTransform(M)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
            else:
                IM = None
                aimg = torch.unsqueeze(det_img, 0).contiguous()
            aimg = aimg.to(dtype=torch.float32)

            io_binding = self.models_processor.models['YunetN'].io_binding()
            io_binding.bind_input(name=input_name, device_type=self.models_processor.device, device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

            for i in range(len(output_names)):
                io_binding.bind_output(output_names[i], self.models_processor.device)

            # Sync and run model
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()
            self.models_processor.models['YunetN'].run_with_iobinding(io_binding)
            net_outs = io_binding.copy_outputs_to_cpu()

            strides = [8, 16, 32]
            for idx, stride in enumerate(strides):
                cls_pred = net_outs[idx].reshape(-1, 1)
                obj_pred = net_outs[idx + len(strides)].reshape(-1, 1)
                reg_pred = net_outs[idx + len(strides) * 2].reshape(-1, 4)
                kps_pred = net_outs[idx + len(strides) * 3].reshape(
                    -1, 5 * 2)

                anchor_centers = np.stack(
                    np.mgrid[:(input_size[1] // stride), :(input_size[0] //
                                                            stride)][::-1],
                    axis=-1)
                anchor_centers = (anchor_centers * stride).astype(
                    np.float32).reshape(-1, 2)

                scores = (cls_pred * obj_pred)
                pos_inds = np.where(scores>=score)[0]

                bbox_cxy = reg_pred[:, :2] * stride + anchor_centers[:]
                bbox_wh = np.exp(reg_pred[:, 2:]) * stride
                tl_x = (bbox_cxy[:, 0] - bbox_wh[:, 0] / 2.)
                tl_y = (bbox_cxy[:, 1] - bbox_wh[:, 1] / 2.)
                br_x = (bbox_cxy[:, 0] + bbox_wh[:, 0] / 2.)
                br_y = (bbox_cxy[:, 1] + bbox_wh[:, 1] / 2.)

                bboxes = np.stack([tl_x, tl_y, br_x, br_y], axis=-1)

                pos_scores = scores[pos_inds]
                pos_bboxes = bboxes[pos_inds]

                # bboxes
                if angle != 0:
                    if len(pos_bboxes) > 0:
                        # Split the points into coordinates (x1, y1) and (x2, y2)
                        points1 = pos_bboxes[:, :2]  # (x1, y1)
                        points2 = pos_bboxes[:, 2:]  # (x2, y2)

                        # Apply the inverse of the rotation matrix to points1 and points2
                        points1 = faceutil.trans_points2d(points1, IM)
                        points2 = faceutil.trans_points2d(points2, IM)

                        _x1 = points1[:, 0]
                        _y1 = points1[:, 1]
                        _x2 = points2[:, 0]
                        _y2 = points2[:, 1]

                        if angle in (-270, 90):
                            # x1, y2, x2, y1
                            points1 = np.stack((_x1, _y2), axis=1)
                            points2 = np.stack((_x2, _y1), axis=1)
                        elif angle in (-180, 180):
                            # x2, y2, x1, y1
                            points1 = np.stack((_x2, _y2), axis=1)
                            points2 = np.stack((_x1, _y1), axis=1)
                        elif angle in (-90, 270):
                            # x2, y1, x1, y2
                            points1 = np.stack((_x2, _y1), axis=1)
                            points2 = np.stack((_x1, _y2), axis=1)

                        # Reassemble the transformed points into the format [x1', y1', x2', y2']
                        pos_bboxes = np.hstack((points1, points2))

                # kpss
                kpss = np.concatenate(
                    [((kps_pred[:, [2 * i, 2 * i + 1]] * stride) + anchor_centers)
                        for i in range(5)],
                    axis=-1)

                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]

                if do_rotation:
                    for i in range(len(pos_kpss)):
                        face_size = max(pos_bboxes[i][2] - pos_bboxes[i][0], pos_bboxes[i][3] - pos_bboxes[i][1])
                        angle_deg_to_front = faceutil.get_face_orientation(face_size, pos_kpss[i])
                        if angle_deg_to_front < -50.00 or angle_deg_to_front > 50.00:
                            pos_scores[i] = 0.0

                        if angle != 0:
                            pos_kpss[i] = faceutil.trans_points2d(pos_kpss[i], IM)

                    pos_inds = np.where(pos_scores>=score)[0]
                    pos_scores = pos_scores[pos_inds]
                    pos_bboxes = pos_bboxes[pos_inds]
                    pos_kpss = pos_kpss[pos_inds]

                kpss_list.append(pos_kpss)
                bboxes_list.append(pos_bboxes)
                scores_list.append(pos_scores)

        if len(bboxes_list) == 0:
            return [], [], []

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        det_scale = det_scale.numpy()###

        bboxes = np.vstack(bboxes_list) / det_scale

        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        dets = pre_det
        thresh = 0.4
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scoresb = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        orderb = scoresb.argsort()[::-1]

        keep = []
        while orderb.size > 0:
            i = orderb[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[orderb[1:]])
            yy1 = np.maximum(y1[i], y1[orderb[1:]])
            xx2 = np.minimum(x2[i], x2[orderb[1:]])
            yy2 = np.minimum(y2[i], y2[orderb[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (areas[i] + areas[orderb[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            orderb = orderb[inds + 1]

        det = pre_det[keep, :]

        kpss = kpss[order,:,:]
        kpss = kpss[keep,:,:]

        #if max_num > 0 and det.shape[0] > max_num:
        if max_num > 0 and det.shape[0] > 1:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            det_img_center = img_height // 2, img_width // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - det_img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - det_img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]

            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]

        score_values = det[:, 4]
        # delete score column
        det = np.delete(det, 4, 1)

        kpss_5 = kpss.copy()
        if use_landmark_detection and len(kpss_5) > 0:
            kpss = []
            for i in range(kpss_5.shape[0]):
                landmark_kpss_5, landmark_kpss, landmark_scores = self.models_processor.run_detect_landmark(img_landmark, det[i], kpss_5[i], landmark_detect_mode, landmark_score, from_points)
                # Always add to kpss, regardless of the length of landmark_kpss.
                kpss.append(landmark_kpss if len(landmark_kpss) > 0 else kpss_5[i])
                if len(landmark_kpss_5) > 0:
                    if len(landmark_scores) > 0:
                        if np.mean(landmark_scores) > np.mean(score_values[i]):
                            kpss_5[i] = landmark_kpss_5
                    else:
                        kpss_5[i] = landmark_kpss_5
            kpss = np.array(kpss, dtype=object)

        return det, kpss_5, kpss