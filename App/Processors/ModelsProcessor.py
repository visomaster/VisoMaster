from PySide6.QtCore import QRunnable, QThreadPool, Signal, QObject, QTimer, QCoreApplication, QThread
import PySide6.QtCore as qtc
from PySide6.QtWidgets import QProgressDialog
import onnxruntime
from functools import partial
from threading import Thread
from App.UI.Widgets import WidgetActions as widget_actions
from App.UI.Widgets.WidgetComponents import ProgressDialog
import time
import threading
lock = threading.Lock()
import torch
from App.Processors.Utils import FaceUtil as faceutil
from skimage import transform as trans
import torchvision
from torchvision.transforms import v2
import numpy as np

class Worker(QObject):
    finished = Signal()

    def __init__(self, model_name, models, providers):
        super().__init__()
        self.model_name = model_name
        self.models = models
        self.providers = providers

    def run(self):
        time.sleep(0.5)
        self.models[self.model_name]['model_instance'] = onnxruntime.InferenceSession(
            self.models[self.model_name]['model_path'], providers=self.providers,
        )
        self.finished.emit()  # Signal that loading is complete

class ModelsProcessor(QObject):
    processing_complete = Signal()
    model_loaded = Signal()  # Signal emitted with Onnx InferenceSession

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.providers = [
            ('TensorrtExecutionProvider', {
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': "tensorrt-engines",
                'trt_timing_cache_enable': True,
                'trt_timing_cache_path': "tensorrt-engines",
                'trt_dump_ep_context_model': True,
                'trt_ep_context_file_path': "tensorrt-engines",
                'trt_layer_norm_fp32_fallback': True,
                'trt_builder_optimization_level': 5,
            }),
            ('CUDAExecutionProvider'),
            ('CPUExecutionProvider')
        ]
        self.models = {
            'Inswapper128': {'model_path': './App/ONNXModels/inswapper_128.fp16.onnx', 'model_instance': None}
        }
        self.current_loading_model = False
        self.model_loaded.connect(self.hideModelLoadProgressBar)


    def test_run_model_function(self, model_name, *args):
        if not self.models[model_name]['model_instance']:
            self.load_model_and_exec(model_name, self.test_run_model_function, model_name, *args)
        else:
            print("Success", self.models['Inswapper128'])

    def load_model_and_exec(self, model_name, exec_func: callable, *args):
        self.showModelLoadingProgressBar()
        # Create worker and thread
        self.worker = Worker(model_name, self.models, self.providers)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        # Connect signals
        self.worker.finished.connect(self.on_model_load_finished)
        self.worker_thread.started.connect(self.worker.run)
        # Start the thread
        self.worker_thread.start()
        
        # Initialize and start the timer for checking the loading status
        self.model_load_timer = QTimer()
        self.model_load_timer.timeout.connect(partial(self.check_model_loaded, model_name, exec_func, *args))
        self.model_load_timer.start(100)  # Check every 100ms

    def on_model_load_finished(self):
        self.worker_thread.quit()
        self.worker_thread.wait()
        self.model_loaded.emit()
        self.model_load_timer.stop()  # Stop checking once the model is loaded

    def check_model_loaded(self, model_name, exec_func: callable, *args):
        if not self.models[model_name]['model_instance']:
            QCoreApplication.processEvents()
        else:
            self.model_load_timer.stop()  # Stop checking once the model is loaded
            self.model_loaded.emit()
            exec_func(*args)  # Execute the next function

    def showModelLoadingProgressBar(self):
        self.main_window.model_load_dialog = ProgressDialog("Loading Models...This is gonna take a while.", "Cancel", 0, 100, self.main_window)
        self.main_window.model_load_dialog.setWindowModality(qtc.Qt.ApplicationModal)
        self.main_window.model_load_dialog.setMinimumDuration(2000)
        self.main_window.model_load_dialog.setWindowTitle("Loading Models")
        self.main_window.model_load_dialog.setAutoClose(True)  # Close the dialog when finished
        self.main_window.model_load_dialog.setCancelButton(None)
        self.main_window.model_load_dialog.setWindowFlag(qtc.Qt.WindowCloseButtonHint, False)
        self.main_window.model_load_dialog.setValue(0)
        self.main_window.model_load_dialog.show()

    def hideModelLoadProgressBar(self):
        if self.main_window.model_load_dialog:
            self.main_window.model_load_dialog.close()

    def run_detect(self, img, detect_mode='Retinaface', max_num=1, score=0.5, use_landmark_detection=False, landmark_detect_mode='203', landmark_score=0.5, from_points=False, rotation_angles:list[int]=[0]):
        bboxes = []
        kpss_5 = []
        kpss = []

        if detect_mode=='Retinaface':
            if not self.retinaface_model:
                self.retinaface_model = onnxruntime.InferenceSession('./models/det_10g.onnx', providers=self.providers)

            bboxes, kpss_5, kpss = self.detect_retinaface(img, max_num=max_num, score=score, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points, rotation_angles=rotation_angles)

        elif detect_mode=='SCRDF':
            if not self.scrdf_model:
                self.scrdf_model = onnxruntime.InferenceSession('./models/scrfd_2.5g_bnkps.onnx', providers=self.providers)

            bboxes, kpss_5, kpss = self.detect_scrdf(img, max_num=max_num, score=score, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points, rotation_angles=rotation_angles)

        elif detect_mode=='Yolov8':
            if not self.yoloface_model:
                self.yoloface_model = onnxruntime.InferenceSession('./models/yoloface_8n.onnx', providers=self.providers)

            bboxes, kpss_5, kpss = self.detect_yoloface(img, max_num=max_num, score=score, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points, rotation_angles=rotation_angles)

        elif detect_mode=='Yunet':
            if not self.yunet_model:
                self.yunet_model = onnxruntime.InferenceSession('./models/yunet_n_640_640.onnx', providers=self.providers)

            bboxes, kpss_5, kpss = self.detect_yunet(img, max_num=max_num, score=score, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points, rotation_angles=rotation_angles)

        return bboxes, kpss_5, kpss
    
    def detect_retinaface(self, img, max_num, score, use_landmark_detection, landmark_detect_mode, landmark_score, from_points, rotation_angles:list[int]=[0]):
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

        resize = v2.Resize((new_height, new_width), antialias=True)
        img = resize(img)
        img = img.permute(1,2,0)

        det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.float32, device='cuda:0')
        det_img[:new_height,:new_width,  :] = img

        # Switch to RGB and normalize
        det_img = det_img[:, :, [2,1,0]]
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
            # Prepare data and find model 
            if angle != 0:
                aimg, M = faceutil.transform(det_img, (cx, cy), 640, 1.0, angle)
                IM = faceutil.invertAffineTransform(M)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
            else:
                IM = None
                aimg = torch.unsqueeze(det_img, 0).contiguous()

            io_binding = self.retinaface_model.io_binding()
            io_binding.bind_input(name='input.1', device_type='cuda', device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

            io_binding.bind_output('448', 'cuda')
            io_binding.bind_output('471', 'cuda')
            io_binding.bind_output('494', 'cuda')
            io_binding.bind_output('451', 'cuda')
            io_binding.bind_output('474', 'cuda')
            io_binding.bind_output('497', 'cuda')
            io_binding.bind_output('454', 'cuda')
            io_binding.bind_output('477', 'cuda')
            io_binding.bind_output('500', 'cuda')

            # Sync and run model
            syncvec = self.syncvec.cpu()
            self.retinaface_model.run_with_iobinding(io_binding)

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
                K = height * width
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
                landmark_kpss_5, landmark_kpss, landmark_scores = self.run_detect_landmark(img_landmark, det[i], kpss_5[i], landmark_detect_mode, landmark_score, from_points)
                kpss.append(landmark_kpss)
                if len(landmark_kpss_5) > 0:
                    if len(landmark_scores) > 0:
                        if np.mean(landmark_scores) > np.mean(score_values[i]):
                            kpss_5[i] = landmark_kpss_5
                    else:
                        kpss_5[i] = landmark_kpss_5

        return det, kpss_5, kpss

    def run_recognize(self, img, kps, similarity_type='Opal', face_swapper_model='Inswapper128'):
        if face_swapper_model == 'Inswapper128':
            if not self.recognition_model:
                self.recognition_model = onnxruntime.InferenceSession('./models/w600k_r50.onnx', providers=self.providers)

            embedding, cropped_image = self.recognize(self.recognition_model, img, kps, similarity_type=similarity_type)
        elif face_swapper_model == 'SimSwap512':
            if not self.recognition_simswap_model:
                self.recognition_simswap_model = onnxruntime.InferenceSession('./models/simswap_arcface_model.onnx', providers=self.providers)

            embedding, cropped_image = self.recognize(self.recognition_simswap_model, img, kps, similarity_type=similarity_type)

        elif face_swapper_model == 'GhostFace-v1' or face_swapper_model == 'GhostFace-v2' or face_swapper_model == 'GhostFace-v3':
            if not self.recognition_ghost_model:
                self.recognition_ghost_model = onnxruntime.InferenceSession('./models/ghost_arcface_backbone.onnx', providers=self.providers)

            embedding, cropped_image = self.recognize(self.recognition_ghost_model, img, kps, similarity_type=similarity_type)

        return embedding, cropped_image

    def recognize(self, recognition_model, img, face_kps, similarity_type):
        if similarity_type == 'Optimal':
            # Find transform & Transform
            img, _ = faceutil.warp_face_by_face_landmark_5(img, face_kps, mode='arcfacemap', interpolation=v2.InterpolationMode.BILINEAR)
        elif similarity_type == 'Pearl':
            # Find transform
            dst = self.arcface_dst.copy()
            dst[:, 0] += 8.0

            tform = trans.SimilarityTransform()
            tform.estimate(face_kps, dst)

            # Transform
            img = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) )
            img = v2.functional.crop(img, 0,0, 128, 128)
            img = v2.Resize((112, 112), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)(img)
        else:
            # Find transform
            tform = trans.SimilarityTransform()
            tform.estimate(face_kps, self.arcface_dst)

            # Transform
            img = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) )
            img = v2.functional.crop(img, 0,0, 112, 112)

        if recognition_model == self.recognition_model or recognition_model == self.recognition_simswap_model:
            # Switch to BGR and normalize
            img = img.permute(1,2,0) #112,112,3
            cropped_image = img
            img = img[:, :, [2,1,0]]
            img = torch.sub(img, 127.5)
            img = torch.div(img, 127.5)
            img = img.permute(2, 0, 1) #3,112,112
        else:
            cropped_image = img.permute(1,2,0) #112,112,3
            # Converti a float32 e normalizza
            img = torch.div(img.float(), 127.5)
            img = torch.sub(img, 1)

        # Prepare data and find model parameters
        img = torch.unsqueeze(img, 0).contiguous()
        input_name = recognition_model.get_inputs()[0].name

        outputs = recognition_model.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)

        io_binding = recognition_model.io_binding()
        io_binding.bind_input(name=input_name, device_type='cuda', device_id=0, element_type=np.float32,  shape=img.size(), buffer_ptr=img.data_ptr())

        for i in range(len(output_names)):
            io_binding.bind_output(output_names[i], 'cuda')

        # Sync and run model
        self.syncvec.cpu()
        recognition_model.run_with_iobinding(io_binding)

        # Return embedding
        return np.array(io_binding.copy_outputs_to_cpu()).flatten(), cropped_image