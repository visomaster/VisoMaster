import cv2
import math
from math import sin, cos, acos, degrees, floor, ceil
import numpy as np
from skimage import transform as trans
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from numpy.linalg import norm as l2norm
import kornia.geometry.transform as kgm

# <--left profile
src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)

# <--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

# ---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

# -->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

# -->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)

# Definisci i punti di riferimento come tensore PyTorch
arcface_src_cuda = torch.tensor(
    [[38.2946, 51.6963],
     [73.5318, 51.5014],
     [56.0252, 71.7366],
     [41.5493, 92.3655],
     [70.7299, 92.2041]],
    dtype=torch.float32,

) # Shape: (5, 2)
if torch.cuda.is_available():
    arcface_src_cuda = arcface_src_cuda.to('cuda')

def pad_image_by_size(img, image_size):
    # Se image_size non è una tupla, crea una tupla con altezza e larghezza uguali
    if not isinstance(image_size, tuple):
        image_size = (image_size, image_size)

    # Larghezza e altezza dell'immagine
    w, h = img.size(dim=2), img.size(dim=1)

    # Dimensioni target
    target_h, target_w = image_size

    # Verifica se la larghezza o l'altezza è inferiore alle dimensioni target
    if w < target_w or h < target_h:
        # Calcolo del padding necessario a destra e in basso
        pad_right = max(target_w - w, 0)  # Assicura che il padding sia non negativo
        pad_bottom = max(target_h - h, 0)  # Assicura che il padding sia non negativo

        # Aggiungi padding all'immagine (pad_left, pad_right, pad_top, pad_bottom)
        img = torch.nn.functional.pad(img, (0, pad_right, 0, pad_bottom), mode='constant', value=0)

    return img

def transform(img, center, output_size, scale, rotation):
    # pad image by image size
    img = pad_image_by_size(img, output_size)

    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2,
                                                output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]

    cropped = v2.functional.affine(img, np.rad2deg(t.rotation), (t.translation[0], t.translation[1]) , t.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )
    cropped = v2.functional.crop(cropped, 0,0, output_size, output_size)

    return cropped, M

def trans_points2d(pts, M):
    # Add a column of ones to the pts array to create homogeneous coordinates
    ones_column = np.ones((pts.shape[0], 1), dtype=np.float32)
    homogeneous_pts = np.hstack([pts, ones_column])

    # Perform the matrix multiplication for all points at once
    transformed_pts = np.dot(homogeneous_pts, M.T)

    # Return only the first two columns (x and y coordinates)
    return transformed_pts[:, :2]

def trans_points3d(pts, M):
    scale = np.sqrt(M[0, 0]**2 + M[0, 1]**2)

    # Add a column of ones to the pts array to create homogeneous coordinates for 2D transformation
    ones_column = np.ones((pts.shape[0], 1), dtype=np.float32)
    homogeneous_pts = np.hstack([pts[:, :2], ones_column])

    # Perform the matrix multiplication for all points at once
    transformed_2d = np.dot(homogeneous_pts, M.T)

    # Scale the z-coordinate
    scaled_z = pts[:, 2] * scale

    # Combine the transformed 2D points with the scaled z-coordinate
    transformed_pts = np.hstack([transformed_2d[:, :2], scaled_z.reshape(-1, 1)])

    return transformed_pts

def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)

def estimate_affine_matrix_3d23d(X, Y):
    ''' Using least-squares solution
    Args:
        X: [n, 3]. 3d points(fixed)
        Y: [n, 3]. corresponding 3d points(moving). Y = PX
    Returns:
        P_Affine: (3, 4). Affine camera matrix (the third row is [0, 0, 0, 1]).
    '''
    X_homo = np.hstack((X, np.ones([X.shape[0],1]))) #n x 4
    P = np.linalg.lstsq(X_homo, Y,rcond=None)[0].T # Affine matrix. 3 x 4
    return P

def P2sRt(P):
    ''' decompositing camera matrix P
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t: (3,). translation.
    '''
    t = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2.0
    r1 = R1/np.linalg.norm(R1)
    r2 = R2/np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t

def matrix2angle(R):
    ''' get three Euler angles from Rotation Matrix
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: pitch
        y: yaw
        z: roll
    '''
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    # rx, ry, rz = np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)
    rx, ry, rz = x*180/np.pi, y*180/np.pi, z*180/np.pi
    return rx, ry, rz

def warp_affine_torchvision(img, matrix, image_size, rotation_ratio=0.0, border_value=0.0, border_mode='replicate', interpolation_value=v2.functional.InterpolationMode.NEAREST, device='cpu'):
    # Ensure image_size is a tuple (width, height)
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    # Ensure the image tensor is on the correct device and of type float
    if isinstance(img, torch.Tensor):
        img_tensor = img.to(device).float()
        if img_tensor.dim() == 3:  # If no batch dimension, add one
            img_tensor = img_tensor.unsqueeze(0)
    else:
        img_tensor = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)

    # Extract the translation parameters from the affine matrix
    t = trans.SimilarityTransform()
    t.params[0:2] = matrix

    # Define default rotation
    rotation = t.rotation

    if rotation_ratio != 0:
        rotation *=rotation_ratio  # Rotation in degrees

    # Convert border mode
    if border_mode == 'replicate':
        fill = [border_value] * img_tensor.shape[1]  # Same value for all channels
    elif border_mode == 'constant':
        fill = [border_value] * img_tensor.shape[1]  # Same value for all channels
    else:
        raise ValueError("Unsupported border_mode. Use 'replicate' or 'constant'.")

    # Apply the affine transformation
    warped_img_tensor = v2.functional.affine(img_tensor, angle=rotation, translate=(t.translation[0], t.translation[1]), scale=t.scale, shear=(0.0, 0.0), interpolation=interpolation_value, center=(0, 0), fill=fill)

    # Crop the image to the desired size
    warped_img_tensor = v2.functional.crop(warped_img_tensor, 0,0, image_size[1], image_size[0])

    return warped_img_tensor.squeeze(0)

def umeyama(src, dst, estimate_scale):
    num = src.shape[0]
    dim = src.shape[1]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    A = np.dot(dst_demean.T, src_demean) / num
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    T = np.eye(dim + 1, dtype=np.double)
    U, S, V = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))
    if estimate_scale:
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0
    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale
    return T

def get_matrix(lmk, templates):
    if templates.shape[0] == 1:
        return umeyama(lmk, templates[0], True)[0:2, :]
    test_lmk = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_error, best_matrix = float("inf"), []
    for i in np.arange(templates.shape[0]):
        matrix = umeyama(lmk, templates[i], True)[0:2, :]
        error = np.sum(
            np.sqrt(np.sum((np.dot(matrix, test_lmk.T).T - templates[i]) ** 2, axis=1))
        )
        if error < min_error:
            min_error, best_matrix = error, matrix
    return best_matrix

def align_crop(img, lmk, image_size, mode='arcfacemap', interpolation=v2.InterpolationMode.NEAREST):
    if mode != 'arcfacemap':
        if mode == 'arcface112':
            templates = float(image_size) / 112.0 * arcface_src
        else:
            factor = float(image_size) / 128.0
            templates = arcface_src * factor
            templates[:, 0] += (factor * 8.0)
    else:
        templates = float(image_size) / 112.0 * src_map[112]

    matrix = get_matrix(lmk, templates)
    '''
    warped = cv2.warpAffine(
        img,
        matrix,
        (image_size, image_size),
        borderValue=0.0,
        borderMode=cv2.BORDER_REPLICATE,
    )
    '''
    warped = warp_affine_torchvision(img, matrix, (image_size, image_size), rotation_ratio=57.2958, border_value=0.0, border_mode='replicate', interpolation_value=v2.functional.InterpolationMode.NEAREST, device=img.device)

    return warped, matrix

def get_arcface_template(image_size=112, mode='arcface112'):
    if mode=='arcface112':
        template = float(image_size) / 112.0 * arcface_src
    elif mode=='arcface128':
        factor = float(image_size) / 128.0
        template = arcface_src * factor
        template[:, 0] += (factor * 8.0)
    else:
        template = float(image_size) / 112.0 * src_map[112]

    return template

# lmk is prediction; src is template
def estimate_norm_arcface_template(lmk, src=arcface_src):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')

    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #print((error, min_error))
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    #print(src[min_index])
    return min_M, min_index

# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112, mode='arcface112'):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')

    if mode != 'arcfacemap':
        if mode == 'arcface112':
            src = float(image_size) / 112.0 * arcface_src
        else:
            factor = float(image_size) / 128.0
            src = arcface_src * factor
            src[:, 0] += (factor * 8.0)
    else:
        src = float(image_size) / 112.0 * src_map[112]

    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #print((error, min_error))
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    #print(src[min_index])
    return min_M, min_index

def warp_face_by_bounding_box(img, bboxes, image_size=112):
    # pad image by image size
    img = pad_image_by_size(img, image_size)

    # Set source points from bounding boxes
    source_points = np.array([ [ bboxes[0], bboxes[1] ], [ bboxes[2], bboxes[1] ], [ bboxes[0], bboxes[3] ], [ bboxes[2], bboxes[3] ] ]).astype(np.float32)

    # Set target points from image size
    target_points = np.array([ [ 0, 0 ], [ image_size, 0 ], [ 0, image_size ], [ image_size, image_size ] ]).astype(np.float32)

    # Find transform
    tform = trans.SimilarityTransform()
    tform.estimate(source_points, target_points)

    # Transform
    img = v2.functional.affine(img, tform.rotation, (tform.translation[0], tform.translation[1]) , tform.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )
    img = v2.functional.crop(img, 0,0, image_size, image_size)
    M = tform.params[0:2]

    return img, M

def warp_face_by_face_landmark_5(img, kpss, image_size=112, mode='arcface112', interpolation=v2.InterpolationMode.NEAREST):
    # pad image by image size
    img = pad_image_by_size(img, image_size)

    M, pose_index = estimate_norm(kpss, image_size, mode=mode)
    t = trans.SimilarityTransform()
    t.params[0:2] = M
    img = v2.functional.affine(img, t.rotation*57.2958, (t.translation[0], t.translation[1]) , t.scale, 0, interpolation=interpolation, center = (0, 0) )
    img = v2.functional.crop(img, 0,0, image_size, image_size)

    return img, M

def getRotationMatrix2D(center, output_size, scale, rotation, is_clockwise = True):
    scale_ratio = scale
    if not is_clockwise:
        rotation = -rotation
    rot = float(rotation) * np.pi / 180.0
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2,
                                                output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]

    return M

def invertAffineTransform(M):
    '''
    t = trans.SimilarityTransform()
    t.params[0:2] = M
    IM = t.inverse.params[0:2, :]
    '''
    M_H = np.vstack([M, np.array([0, 0, 1])])
    IM = np.linalg.inv(M_H)

    return IM

def warp_face_by_bounding_box_for_landmark_68(img, bbox, input_size):
    """
    :param img: raw image
    :param bbox: the bbox for the face
    :param input_size: tuple input image size
    :return:
    """
    # pad image by image size
    img = pad_image_by_size(img, input_size[0])

    scale = 195 / np.subtract(bbox[2:], bbox[:2]).max()
    translation = (256 - np.add(bbox[2:], bbox[:2]) * scale) * 0.5
    rotation = 0

    t1 = trans.SimilarityTransform(scale=scale)
    t2 = trans.SimilarityTransform(rotation=rotation)
    t3 = trans.SimilarityTransform(translation=translation)

    t = t1 + t2 + t3
    affine_matrix = np.array([ [ scale, 0, translation[0] ], [ 0, scale, translation[1] ] ])

    crop_image = v2.functional.affine(img, t.rotation, (t.translation[0], t.translation[1]) , t.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )
    crop_image = v2.functional.crop(crop_image, 0,0, input_size[1], input_size[0])

    if torch.mean(crop_image.to(dtype=torch.float32)[0, :, :]) < 30:
        crop_image = cv2.cvtColor(crop_image.permute(1, 2, 0).to('cpu').numpy(), cv2.COLOR_RGB2Lab)
        crop_image[:, :, 0] = cv2.createCLAHE(clipLimit = 2).apply(crop_image[:, :, 0])
        crop_image = torch.from_numpy(cv2.cvtColor(crop_image, cv2.COLOR_Lab2RGB)).to(img.device).permute(2, 0, 1)

    return crop_image, affine_matrix

def warp_face_by_bounding_box_for_landmark_98(img, bbox_org, input_size):
    """
    :param img: raw image
    :param bbox: the bbox for the face
    :param input_size: tuple input image size
    :return:
    """
    # pad image by image size
    img = pad_image_by_size(img, input_size[0])

    ##preprocess
    bbox = bbox_org.copy()
    min_face = 20
    base_extend_range = [0.2, 0.3]
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    if bbox_width <= min_face or bbox_height <= min_face:
        return None, None
    add = int(max(bbox_width, bbox_height))

    bimg = torch.nn.functional.pad(img, (add, add, add, add), 'constant', 0)

    bbox += add

    face_width = (1 + 2 * base_extend_range[0]) * bbox_width
    center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]

    ### make the box as square
    bbox[0] = center[0] - face_width // 2
    bbox[1] = center[1] - face_width // 2
    bbox[2] = center[0] + face_width // 2
    bbox[3] = center[1] + face_width // 2

    # crop
    bbox = bbox.astype(np.int32)
    crop_image = bimg[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]

    h, w = (crop_image.size(dim=1), crop_image.size(dim=2))

    t_resize = v2.Resize((input_size[1], input_size[0]), antialias=False)
    crop_image = t_resize(crop_image)

    return crop_image, [h, w, bbox[1], bbox[0], add]

def create_bounding_box_from_face_landmark_106_98_68(face_landmark_106_98_68):
    min_x, min_y = np.min(face_landmark_106_98_68, axis = 0)
    max_x, max_y = np.max(face_landmark_106_98_68, axis = 0)
    bounding_box = np.array([ min_x, min_y, max_x, max_y ]).astype(np.int16)
    return bounding_box

def convert_face_landmark_68_to_5(face_landmark_68, face_landmark_68_score):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55], dtype=np.int32) - 1
    face_landmark_5 = np.stack([
        np.mean(face_landmark_68[lm_idx[[1, 2]], :], 0),  # left eye
        np.mean(face_landmark_68[lm_idx[[3, 4]], :], 0),  # right eye
        face_landmark_68[lm_idx[0], :],  # nose
        face_landmark_68[lm_idx[5], :],  # lip
        face_landmark_68[lm_idx[6], :]   # lip
    ], axis=0)

    if np.any(face_landmark_68_score):
        face_landmark_5_score = np.stack([
            np.mean(face_landmark_68_score[lm_idx[[1, 2]], :], 0),  # left eye
            np.mean(face_landmark_68_score[lm_idx[[3, 4]], :], 0),  # right eye
            face_landmark_68_score[lm_idx[0], :],  # nose
            face_landmark_68_score[lm_idx[5], :],  # lip
            face_landmark_68_score[lm_idx[6], :]   # lip
        ], axis=0)
    else:
        face_landmark_5_score = np.array([])

    return face_landmark_5, face_landmark_5_score

def convert_face_landmark_98_to_5(face_landmark_98, face_landmark_98_score):
    face_landmark_5 = np.array(
    [
        face_landmark_98[96], # eye left
        face_landmark_98[97], # eye-right
        face_landmark_98[54], # nose,
        face_landmark_98[76], # lip left
        face_landmark_98[82]  # lip right
    ])

    face_landmark_5_score = np.array(
    [
        face_landmark_98_score[96], # eye left
        face_landmark_98_score[97], # eye-right
        face_landmark_98_score[54], # nose,
        face_landmark_98_score[76], # lip left
        face_landmark_98_score[82]  # lip right
    ])

    return face_landmark_5, face_landmark_5_score

def convert_face_landmark_106_to_5(face_landmark_106):
    face_landmark_5 = np.array(
    [
        face_landmark_106[38], # eye left
        face_landmark_106[88], # eye-right
        face_landmark_106[86], # nose,
        face_landmark_106[52], # lip left
        face_landmark_106[61]  # lip right
    ])

    return face_landmark_5

def convert_face_landmark_203_to_5(face_landmark_203, use_mean_eyes=False):
    if use_mean_eyes:
        eye_left = np.mean(face_landmark_203[[0, 6, 12, 18]], axis=0)  # Average of left eye points
        eye_right = np.mean(face_landmark_203[[24, 30, 36, 42]], axis=0)  # Average of right eye points
    else:
        eye_left = face_landmark_203[197]  # Specific left eye point
        eye_right = face_landmark_203[198]  # Specific right eye point

    nose = face_landmark_203[201]  # Nose
    lip_left = face_landmark_203[48]  # Left lip corner
    lip_right = face_landmark_203[66]  # Right lip corner

    face_landmark_5 = np.array([eye_left, eye_right, nose, lip_left, lip_right])

    return face_landmark_5

def convert_face_landmark_478_to_5(face_landmark_478, use_mean_eyes=False):
    if use_mean_eyes:
        eye_left = np.mean(face_landmark_478[[472, 471, 470, 469]], axis=0)  # Average of left eye points
        eye_right = np.mean(face_landmark_478[[477, 476, 475, 474]], axis=0)  # Average of right eye points
    else:
        eye_left = face_landmark_478[468]  # Specific left eye point
        eye_right = face_landmark_478[473]  # Specific right eye point

    nose = face_landmark_478[4]  # Nose
    lip_left = face_landmark_478[61]  # Left lip corner
    lip_right = face_landmark_478[291]  # Right lip corner

    face_landmark_5 = np.array([eye_left, eye_right, nose, lip_left, lip_right])

    return face_landmark_5

def convert_face_landmark_x_to_5(pts, **kwargs):
    pts_score = kwargs.get('pts_score', [])
    use_mean_eyes = kwargs.get('use_mean_eyes', False)

    if pts.shape[0] == 5:
        return pts
    elif pts.shape[0] == 68:
        pt5 = convert_face_landmark_68_to_5(face_landmark_68=pts, face_landmark_68_score=pts_score)
    elif pts.shape[0] == 98:
        pt5 = convert_face_landmark_98_to_5(face_landmark_98=pts, face_landmark_98_score=pts_score)
    elif pts.shape[0] == 106:
        pt5 = convert_face_landmark_106_to_5(face_landmark_106=pts)
    elif pts.shape[0] == 203:
        pt5 = convert_face_landmark_203_to_5(face_landmark_203=pts, use_mean_eyes=use_mean_eyes)
    elif pts.shape[0] == 478:
        pt5 = convert_face_landmark_478_to_5(face_landmark_478=pts, use_mean_eyes=use_mean_eyes)
    else:
        raise Exception(f'Unknow shape: {pts.shape}')

    return pt5

def test_bbox_landmarks(img, bbox, kpss, caption='image', show_kpss_label=False):
        image = img.permute(1,2,0).to('cpu').numpy().copy()
        if len(bbox) > 0:
            box = bbox.astype(int)
            color = (255, 0, 0)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)

        if len(kpss) > 0:
            for i in range(kpss.shape[0]):
                kps = kpss[i].astype(int)
                color = (0, 0, 255)
                cv2.circle(image, (kps[0], kps[1]), 1, color,
                           2)
                if show_kpss_label:
                    if kpss.shape[0] == 5:
                        match i:
                            case 0:
                                text = "LE"
                            case 1:
                                text = "RE"
                            case 2:
                                text = "NO"
                            case 3:
                                text = "LM"
                            case 4:
                                text = "RM"
                    else:
                        text = str(i)

                    image = cv2.putText(image, text, (kps[0], kps[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA, False)

        cv2.imshow(caption, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def test_multi_bbox_landmarks(img, bboxes, kpss, caption='image', show_kpss_label=False):
    if len(bboxes) > 0 and len(kpss) > 0:
        for i in range(np.array(kpss).shape[0]):
            test_bbox_landmarks(img, bboxes[i], kpss[i], caption=caption, show_kpss_label=show_kpss_label)
    elif len(bboxes) > 0:
        for i in range(np.array(bboxes).shape[0]):
            test_bbox_landmarks(img, bboxes[i], [], caption=caption, show_kpss_label=show_kpss_label)
    elif len(kpss) > 0:
        for i in range(np.array(kpss).shape[0]):
            test_bbox_landmarks(img, [], kpss[i], caption=caption, show_kpss_label=show_kpss_label)

def detect_img_color(img):
    frame = img.permute(1,2,0)

    b = frame[:, :, :1]
    g = frame[:, :, 1:2]
    r = frame[:, :, 2:]

    # computing the mean
    b_mean = torch.mean(b.to(float))
    g_mean = torch.mean(g.to(float))
    r_mean = torch.mean(r.to(float))

    # displaying the most prominent color
    if (b_mean > g_mean and b_mean > r_mean):
        return 'BGR'
    elif (g_mean > r_mean and g_mean > b_mean):
        return 'GBR'

    return 'RGB'

def get_face_orientation(face_size, lmk):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    src = np.squeeze(arcface_src, axis=0)
    src = float(face_size) / 112.0 * src
    tform.estimate(lmk, src)

    angle_deg_to_front = np.rad2deg(tform.rotation)

    return angle_deg_to_front

def rgb_to_yuv(image, normalize=False):
    """
    Convert an RGB image to YUV.
    Args:
        image (torch.Tensor): The input image tensor in RGB format (C, H, W) with values in the range [0, 255].
    Returns:
        torch.Tensor: The image tensor in YUV format (C, H, W).
    """
    if normalize:
        # Ensure the image is in the range [0, 1]
        image = torch.div(image, 255.0)

    # Define the conversion matrix from RGB to YUV
    conversion_matrix = torch.tensor([[0.299, 0.587, 0.114],
                                      [-0.14713, -0.28886, 0.436],
                                      [0.615, -0.51499, -0.10001]], device=image.device, dtype=image.dtype)

    # Apply the conversion matrix
    yuv_image = torch.tensordot(image.permute(1, 2, 0), conversion_matrix, dims=1).permute(2, 0, 1)

    return yuv_image

def yuv_to_rgb(image, normalize=False):
    """
    Convert a YUV image to RGB.
    Args:
        image (torch.Tensor): The input image tensor in YUV format (C, H, W) with values in the range [0, 1].
    Returns:
        torch.Tensor: The image tensor in RGB format (C, H, W).
    """
    # Define the conversion matrix from YUV to RGB
    conversion_matrix = torch.tensor([[1, 0, 1.13983],
                                      [1, -0.39465, -0.58060],
                                      [1, 2.03211, 0]], device=image.device, dtype=image.dtype)

    # Apply the conversion matrix
    rgb_image = torch.tensordot(image.permute(1, 2, 0), conversion_matrix, dims=1).permute(2, 0, 1)

    # Ensure the image is in the range [0, 1]
    rgb_image = torch.clamp(rgb_image, 0, 1)

    if normalize:
        rgb_image = torch.mul(rgb_image, 255.0)

    return rgb_image

def rgb_to_lab(rgb, normalize=False):
    # Assume rgb is in (C, H, W) format and values are in [0, 1]
    if normalize:
        rgb = rgb / 255.0

    # Transpose to (H, W, C) for processing
    rgb = rgb.permute(1, 2, 0).contiguous()

    # Linearization (Gamma Correction)
    mask = rgb > 0.04045
    rgb_linear = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

    # Conversion from RGB to XYZ
    rgb_linear = rgb_linear.view(-1, 3)
    matrix_rgb_to_xyz = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=rgb.dtype, device=rgb.device)

    xyz = torch.matmul(rgb_linear, matrix_rgb_to_xyz.T)

    # Normalize by D65 white point
    white_point = torch.tensor([0.95047, 1.00000, 1.08883], dtype=xyz.dtype, device=xyz.device)
    xyz = xyz / white_point

    # Conversion from XYZ to LAB
    epsilon = 0.008856
    kappa = 903.3

    mask = xyz > epsilon
    f_xyz = torch.where(mask, xyz ** (1/3), (kappa * xyz + 16) / 116)

    L = (116 * f_xyz[:, 1]) - 16
    a = 500 * (f_xyz[:, 0] - f_xyz[:, 1])
    b = 200 * (f_xyz[:, 1] - f_xyz[:, 2])

    lab = torch.stack([L, a, b], dim=1)
    lab = lab.view(rgb.shape[0], rgb.shape[1], 3)  # (H, W, 3)
    lab = lab.permute(2, 0, 1)  # Back to (C, H, W)

    return lab

def lab_to_rgb(lab, normalize=False):
    # Assume lab is in (C, H, W) format
    if lab.dim() != 3 or lab.shape[0] != 3:
        raise ValueError("LAB tensor must have shape (3, H, W)")

    # Transpose to (H, W, C)
    lab = lab.permute(1, 2, 0).contiguous()

    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    # Conversion from LAB to XYZ
    epsilon = 0.008856
    kappa = 903.3

    fy = (L + 16) / 116
    fx = fy + (a / 500)
    fz = fy - (b / 200)

    fx3 = fx ** 3
    fz3 = fz ** 3

    x = torch.where(fx3 > epsilon, fx3, (116 * fx - 16) / kappa)
    y = torch.where(L > (kappa * epsilon), ((L + 16) / 116) ** 3, L / kappa)
    z = torch.where(fz3 > epsilon, fz3, (116 * fz - 16) / kappa)

    # Denormalize by D65 white point
    white_point = torch.tensor([0.95047, 1.00000, 1.08883], dtype=lab.dtype, device=lab.device)
    xyz = torch.stack([x, y, z], dim=2) * white_point

    # Conversion from XYZ to RGB
    xyz = xyz.view(-1, 3)
    matrix_xyz_to_rgb = torch.tensor([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ], dtype=lab.dtype, device=lab.device)

    rgb_linear = torch.matmul(xyz, matrix_xyz_to_rgb.T)

    # Apply gamma correction
    mask = rgb_linear > 0.0031308
    rgb = torch.where(mask, 1.055 * (rgb_linear ** (1 / 2.4)) - 0.055, 12.92 * rgb_linear)

    # Reshape back to image format
    rgb = rgb.view(lab.shape[0], lab.shape[1], 3)
    rgb = torch.clamp(rgb, 0.0, 1.0)
    rgb = rgb.permute(2, 0, 1)  # Back to (C, H, W)

    if normalize:
        rgb = rgb * 255.0

    return rgb

def rgb_to_hsv(image):
    device = image.device  # Ensure operations happen on the same device as the input image

    # Convert image to float if needed
    image = image.float() / 255.0 if image.dtype == torch.uint8 else image.float()

    r, g, b = image[0], image[1], image[2]  # Split the RGB channels

    max_val, _ = torch.max(image, dim=0)  # Max value per pixel across RGB channels, shape [512, 512]
    min_val, _ = torch.min(image, dim=0)  # Min value per pixel across RGB channels, shape [512, 512]
    delta = max_val - min_val  # Difference between max and min, shape [512, 512]

    # Initialize Hue, Saturation, and Value tensors as float32
    h = torch.zeros_like(max_val, dtype=torch.float32).to(device)
    s = torch.zeros_like(max_val, dtype=torch.float32).to(device)
    v = max_val  # Value is max_val (no need to change dtype)

    # Avoid division by zero: only compute where delta != 0
    mask = delta != 0

    # Hue calculation based on which color channel is the maximum
    r_mask = max_val == r
    g_mask = max_val == g
    b_mask = max_val == b

    h[mask & r_mask] = ((g - b) / delta % 6)[mask & r_mask]
    h[mask & g_mask] = (((b - r) / delta) + 2)[mask & g_mask]
    h[mask & b_mask] = (((r - g) / delta) + 4)[mask & b_mask]

    h = h * 60.0  # Scale hue to [0, 360] range
    h = h / 360.0  # Normalize hue to [0, 1]

    # Saturation calculation: only compute where max_val != 0
    s[max_val != 0] = (delta / max_val)[max_val != 0]

    # Stack the HSV channels together
    hsv_image = torch.stack([h, s, v], dim=0)

    return hsv_image

def hsv_to_rgb(hsv_image):
    device = hsv_image.device  # Ensure operations happen on the same device as the input image
    
    h, s, v = hsv_image[0], hsv_image[1], hsv_image[2]  # Split the HSV channels

    h = h * 360.0  # Convert hue back to [0, 360] range

    c = v * s  # Chroma
    x = c * (1 - torch.abs((h / 60.0) % 2 - 1))  # Second largest component of the color
    m = v - c  # Match value

    # Initialize r, g, b with zeros
    r = torch.zeros_like(h, device=device)
    g = torch.zeros_like(h, device=device)
    b = torch.zeros_like(h, device=device)

    # Conditions for different hue ranges
    h1 = (0 <= h) & (h < 60)
    h2 = (60 <= h) & (h < 120)
    h3 = (120 <= h) & (h < 180)
    h4 = (180 <= h) & (h < 240)
    h5 = (240 <= h) & (h < 300)
    h6 = (300 <= h) & (h < 360)

    # Apply the color transformation logic based on hue ranges
    r[h1] = c[h1]
    g[h1] = x[h1]
    b[h1] = 0

    r[h2] = x[h2]
    g[h2] = c[h2]
    b[h2] = 0

    r[h3] = 0
    g[h3] = c[h3]
    b[h3] = x[h3]

    r[h4] = 0
    g[h4] = x[h4]
    b[h4] = c[h4]

    r[h5] = x[h5]
    g[h5] = 0
    b[h5] = c[h5]

    r[h6] = c[h6]
    g[h6] = 0
    b[h6] = x[h6]

    # Add m to match the value and scale the RGB channels back to [0, 1]
    r = r + m
    g = g + m
    b = b + m

    # Stack the RGB channels together
    rgb_image = torch.stack([r, g, b], dim=0)

    return rgb_image

def sharpen(img):
    device = img.device  # Ensure we use the same device

    # Convert img to float and normalize it
    img = img.float() / 255.0

    # Gaussian smoothing using PyTorch's functional API (approximation of Gaussian blur)
    gauss_kernel = get_gaussian_kernel(5).to(device)  # Create a Gaussian kernel for blurring
    img = img.unsqueeze(0)  # Add batch dimension for convolution
    gauss_out = torch.nn.functional.conv2d(img, gauss_kernel, padding=2, groups=img.size(1))
    gauss_out = gauss_out.squeeze(0)  # Remove batch dimension

    alpha = 1.5
    img_out = (img.squeeze(0) - gauss_out) * alpha + img.squeeze(0)

    # Clamp values between 0 and 1, then scale back to [0, 255]
    img_out = torch.clamp(img_out, 0.0, 1.0) * 255.0

    return img_out.to(torch.uint8)

def get_gaussian_kernel(sigma, kernel_size=5):
    """Create a 2D Gaussian kernel for convolution."""
    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords -= (kernel_size - 1) / 2.0

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    g_kernel = torch.outer(g, g)
    g_kernel = g_kernel.unsqueeze(0).unsqueeze(0)  # Make it 4D for convolution
    return g_kernel.expand(3, 1, kernel_size, kernel_size)  # Apply to each channels

# Live Portrait
#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/crop.py
def parse_pt2_from_pt101(pt101, use_lip=True):
    """
    parsing the 2 points according to the 101 points, which cancels the roll
    """
    # the former version use the eye center, but it is not robust, now use interpolation
    pt_left_eye = np.mean(pt101[[39, 42, 45, 48]], axis=0)  # left eye center
    pt_right_eye = np.mean(pt101[[51, 54, 57, 60]], axis=0)  # right eye center

    if use_lip:
        # use lip
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt101[75] + pt101[81]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)

    return pt2

def parse_pt2_from_pt98(pt98, use_lip=True, use_mean_eyes=False):
    """
    parsing the 2 points according to the 98 points, which cancels the roll
    """
    if use_mean_eyes:
        pt_left_eye = np.mean(pt98[[66, 60, 62, 64]], axis=0)  # Average of left eye points
        pt_right_eye = np.mean(pt98[[74, 68, 70, 72]], axis=0)  # Average of right eye points
    else:
        pt_left_eye = pt98[96] # Specific left eye point
        pt_right_eye = pt98[97] # Specific right eye point

    if use_lip:
        # use lip
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt98[76] + pt98[82]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)

    return pt2

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/crop.py
def parse_pt2_from_pt106(pt106, use_lip=True, use_mean_eyes=False):
    """
    parsing the 2 points according to the 106 points, which cancels the roll
    """
    if use_mean_eyes:
        pt_left_eye = np.mean(pt106[[33, 35, 40, 39]], axis=0)  # Average of left eye points
        pt_right_eye = np.mean(pt106[[87, 89, 94, 93]], axis=0)  # Average of right eye points
    else:
        pt_left_eye = pt106[38] # Specific left eye point
        pt_right_eye = pt106[88] # Specific right eye point

    if use_lip:
        # use lip
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt106[52] + pt106[61]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)

    return pt2

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/crop.py
def parse_pt2_from_pt203(pt203, use_lip=True, use_mean_eyes=False):
    """
    parsing the 2 points according to the 203 points, which cancels the roll
    """
    if use_mean_eyes:
        pt_left_eye = np.mean(pt203[[0, 6, 12, 18]], axis=0)  # Average of left eye points
        pt_right_eye = np.mean(pt203[[24, 30, 36, 42]], axis=0)  # Average of right eye points
    else:
        pt_left_eye = pt203[197]  # Specific left eye point
        pt_right_eye = pt203[198]  # Specific right eye point

    if use_lip:
        # use lip
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt203[48] + pt203[66]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)

    return pt2

def parse_pt2_from_pt478(pt478, use_lip=True, use_mean_eyes=False):
    """
    parsing the 2 points according to the 203 points, which cancels the roll
    """
    if use_mean_eyes:
        pt_left_eye = np.mean(pt478[[472, 471, 470, 469]], axis=0)  # Average of left eye points
        pt_right_eye = np.mean(pt478[[477, 476, 475, 474]], axis=0)  # Average of right eye points
    else:
        pt_left_eye = pt478[468]  # Specific left eye point
        pt_right_eye = pt478[473]  # Specific right eye point

    if use_lip:
        # use lip
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt478[61] + pt478[291]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)

    return pt2

def parse_pt2_from_pt68(pt68, use_lip=True):
    """
    parsing the 2 points according to the 68 points, which cancels the roll
    """
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55], dtype=np.int32) - 1
    if use_lip:
        pt5 = np.stack([
            np.mean(pt68[lm_idx[[1, 2]], :], 0),  # left eye
            np.mean(pt68[lm_idx[[3, 4]], :], 0),  # right eye
            pt68[lm_idx[0], :],  # nose
            pt68[lm_idx[5], :],  # lip
            pt68[lm_idx[6], :]   # lip
        ], axis=0)

        pt2 = np.stack([
            (pt5[0] + pt5[1]) / 2,
            (pt5[3] + pt5[4]) / 2
        ], axis=0)
    else:
        pt2 = np.stack([
            np.mean(pt68[lm_idx[[1, 2]], :], 0),  # left eye
            np.mean(pt68[lm_idx[[3, 4]], :], 0),  # right eye
        ], axis=0)

    return pt2

def parse_pt2_from_pt5(pt5, use_lip=True):
    """
    parsing the 2 points according to the 5 points, which cancels the roll
    """
    pt_left_eye = pt5[0] # Specific left eye point
    pt_right_eye = pt5[1] # Specific right eye point

    if use_lip:
        # use lip
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt5[3] + pt5[4]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)

    return pt2

def parse_pt2_from_pt9(pt9, use_lip=True):
    '''
    parsing the 2 points according to the 9 points, which cancels the roll
    ['right eye right', 'right eye left', 'left eye right', 'left eye left', 'nose tip', 'lip right', 'lip left', 'upper lip', 'lower lip']
    '''
    if use_lip:
        pt9 = np.stack([
            (pt9[2] + pt9[3]) / 2, # left eye
            (pt9[0] + pt9[1]) / 2, # right eye
            pt9[4],
            (pt9[5] + pt9[6] ) / 2 # lip
        ], axis=0)
        pt2 = np.stack([
            (pt9[0] + pt9[1]) / 2, # eye
            pt9[3] # lip
        ], axis=0)
    else:
        pt2 = np.stack([
            (pt9[2] + pt9[3]) / 2,
            (pt9[0] + pt9[1]) / 2,
        ], axis=0)

    return pt2

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/crop.py
def parse_pt2_from_pt_x(pts, use_lip=True, use_mean_eyes=False):
    if pts.shape[0] == 101:
        pt2 = parse_pt2_from_pt101(pts, use_lip=use_lip)
    elif pts.shape[0] == 106:
        pt2 = parse_pt2_from_pt106(pts, use_lip=use_lip, use_mean_eyes=use_mean_eyes)
    elif pts.shape[0] == 68:
        pt2 = parse_pt2_from_pt68(pts, use_lip=use_lip)
    elif pts.shape[0] == 5:
        pt2 = parse_pt2_from_pt5(pts, use_lip=use_lip)
    elif pts.shape[0] == 203:
        pt2 = parse_pt2_from_pt203(pts, use_lip=use_lip, use_mean_eyes=use_mean_eyes)
    elif pts.shape[0] == 98:
        pt2 = parse_pt2_from_pt98(pts, use_lip=use_lip, use_mean_eyes=use_mean_eyes)
    elif pts.shape[0] == 478:
        pt2 = parse_pt2_from_pt478(pts, use_lip=use_lip, use_mean_eyes=use_mean_eyes)
    elif pts.shape[0] > 101:
        # take the first 101 points
        pt2 = parse_pt2_from_pt101(pts[:101], use_lip=use_lip)
    elif pts.shape[0] == 9:
        pt2 = parse_pt2_from_pt9(pts, use_lip=use_lip)
    else:
        raise Exception(f'Unknow shape: {pts.shape}')

    if not use_lip:
        # NOTE: to compile with the latter code, need to rotate the pt2 90 degrees clockwise manually
        v = pt2[1] - pt2[0]
        pt2[1, 0] = pt2[0, 0] - v[1]
        pt2[1, 1] = pt2[0, 1] + v[0]

    return pt2

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/crop.py
def parse_rect_from_landmark(
    pts,
    scale=1.5,
    need_square=True,
    vx_ratio=0,
    vy_ratio=0,
    use_deg_flag=False,
    **kwargs
):
    """parsing center, size, angle from 101/68/5/x landmarks
    vx_ratio: the offset ratio along the pupil axis x-axis, multiplied by size
    vy_ratio: the offset ratio along the pupil axis y-axis, multiplied by size, which is used to contain more forehead area

    judge with pts.shape
    """
    pt2 = parse_pt2_from_pt_x(pts, use_lip=kwargs.get('use_lip', True), use_mean_eyes=kwargs.get('use_mean_eyes', False))

    uy = pt2[1] - pt2[0]
    l = np.linalg.norm(uy)
    if l <= 1e-3:
        uy = np.array([0, 1], dtype=np.float32)
    else:
        uy /= l
    ux = np.array((uy[1], -uy[0]), dtype=np.float32)

    # the rotation degree of the x-axis, the clockwise is positive, the counterclockwise is negative (image coordinate system)
    # print(uy)
    # print(ux)
    angle = acos(ux[0])
    if ux[1] < 0:
        angle = -angle

    # rotation matrix
    M = np.array([ux, uy])

    # calculate the size which contains the angle degree of the bbox, and the center
    center0 = np.mean(pts, axis=0)
    rpts = (pts - center0) @ M.T  # (M @ P.T).T = P @ M.T
    lt_pt = np.min(rpts, axis=0)
    rb_pt = np.max(rpts, axis=0)
    center1 = (lt_pt + rb_pt) / 2

    size = rb_pt - lt_pt
    if need_square:
        m = max(size[0], size[1])
        size[0] = m
        size[1] = m

    size *= scale  # scale size
    center = center0 + ux * center1[0] + uy * center1[1]  # counterclockwise rotation, equivalent to M.T @ center1.T
    center = center + ux * (vx_ratio * size) + uy * \
        (vy_ratio * size)  # considering the offset in vx and vy direction

    if use_deg_flag:
        angle = degrees(angle)

    return center, size, angle

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/crop.py
def parse_bbox_from_landmark(pts, **kwargs):
    center, size, angle = parse_rect_from_landmark(pts, **kwargs)
    cx, cy = center
    w, h = size

    # calculate the vertex positions before rotation
    bbox = np.array([
        [cx-w/2, cy-h/2],  # left, top
        [cx+w/2, cy-h/2],
        [cx+w/2, cy+h/2],  # right, bottom
        [cx-w/2, cy+h/2]
    ], dtype=np.float32)

    # construct rotation matrix
    bbox_rot = bbox.copy()
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ], dtype=np.float32)

    # calculate the relative position of each vertex from the rotation center, then rotate these positions, and finally add the coordinates of the rotation center
    bbox_rot = (bbox_rot - center) @ R.T + center

    return {
        'center': center,  # 2x1
        'size': size,  # scalar
        'angle': angle,  # rad, counterclockwise
        'bbox': bbox,  # 4x2
        'bbox_rot': bbox_rot,  # 4x2
    }

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/crop.py
def _estimate_similar_transform_from_pts(
    pts,
    dsize,
    scale=1.5,
    vx_ratio=0,
    vy_ratio=-0.1,
    flag_do_rot=True,
    **kwargs
):
    """ calculate the affine matrix of the cropped image from sparse points, the original image to the cropped image, the inverse is the cropped image to the original image
    pts: landmark, 101 or 68 points or other points, Nx2
    scale: the larger scale factor, the smaller face ratio
    vx_ratio: x shift
    vy_ratio: y shift, the smaller the y shift, the lower the face region
    rot_flag: if it is true, conduct correction
    """
    center, size, angle = parse_rect_from_landmark(
        pts, scale=scale, vx_ratio=vx_ratio, vy_ratio=vy_ratio,
        use_lip=kwargs.get('use_lip', True),
        use_mean_eyes=kwargs.get('use_mean_eyes', False)
    )

    s = dsize / size[0]  # scale
    tgt_center = np.array([dsize / 2, dsize / 2], dtype=np.float32)  # center of dsize

    if flag_do_rot:
        costheta, sintheta = cos(angle), sin(angle)
        cx, cy = center[0], center[1]  # ori center
        tcx, tcy = tgt_center[0], tgt_center[1]  # target center
        # need to infer
        M_INV = np.array(
            [[s * costheta, s * sintheta, tcx - s * (costheta * cx + sintheta * cy)],
             [-s * sintheta, s * costheta, tcy - s * (-sintheta * cx + costheta * cy)]],
            dtype=np.float32
        )
    else:
        M_INV = np.array(
            [[s, 0, tgt_center[0] - s * center[0]],
             [0, s, tgt_center[1] - s * center[1]]],
            dtype=np.float32
        )

    M_INV_H = np.vstack([M_INV, np.array([0, 0, 1])])
    M = np.linalg.inv(M_INV_H)

    # M_INV is from the original image to the cropped image, M is from the cropped image to the original image
    return M_INV, M[:2, ...]

def warp_face_by_face_landmark_x(img, pts, **kwargs):
    dsize = kwargs.get('dsize', 224)  # 512
    scale = kwargs.get('scale', 1.5)  # 1.5 | 1.6 | 2.5
    vy_ratio = kwargs.get('vy_ratio', -0.1)  # -0.0625 | -0.1 | -0.125
    interpolation = kwargs.get('interpolation', v2.InterpolationMode.BILINEAR)

    # pad image by image size
    img = pad_image_by_size(img, dsize)
    #if pts.shape[0] == 5:
    #    scale *= 2.20
    #    vy_ratio += (-vy_ratio / 2.20)

    M_o2c, M_c2o = _estimate_similar_transform_from_pts(
        pts,
        dsize=dsize,
        scale=scale,
        vy_ratio=vy_ratio,
        flag_do_rot=kwargs.get('flag_do_rot', True),
    )

    t = trans.SimilarityTransform()
    t.params[0:2] = M_o2c
    img = v2.functional.affine(img, t.rotation*57.2958, translate=(t.translation[0], t.translation[1]), scale=t.scale, shear=(0.0, 0.0), interpolation=interpolation, center=(0, 0))
    img = v2.functional.crop(img, 0,0, dsize, dsize)

    return img, M_o2c, M_c2o

def create_faded_inner_mask(size, border_thickness, fade_thickness, blur_radius=3, device='cuda'):
    """
    Create a mask with a thick black border and a faded white center towards the border (optimized version).
    The white edges are smoothed using Gaussian blur.

    Parameters:
    - size: Tuple (height, width) for the mask size.
    - border_thickness: The thickness of the outer black border.
    - fade_thickness: The thickness over which the white center fades into the black border.
    - blur_radius: The radius for the Gaussian blur to smooth the white edges.
    - device: Device to perform the computation ('cuda' for GPU, 'cpu' for CPU).

    Returns:
    - mask: A PyTorch tensor containing the mask.
    """
    height, width = size
    mask = torch.zeros((height, width), dtype=torch.float32, device=device)  # Start with a black mask

    # Define the inner region
    inner_start = border_thickness
    inner_end_x = width - border_thickness
    inner_end_y = height - border_thickness

    # Create grid for distance calculations on the specified device
    y_indices, x_indices = torch.meshgrid(torch.arange(height, device=device),
                                          torch.arange(width, device=device), indexing='ij')

    # Calculate distances to the nearest border for each point
    dist_to_left = x_indices - inner_start
    dist_to_right = inner_end_x - x_indices - 1
    dist_to_top = y_indices - inner_start
    dist_to_bottom = inner_end_y - y_indices - 1

    # Calculate minimum distance to any border
    dist_to_border = torch.minimum(torch.minimum(dist_to_left, dist_to_right),
                                   torch.minimum(dist_to_top, dist_to_bottom))

    # Mask inside the fading region
    fade_region = (dist_to_border >= 0) & (dist_to_border < fade_thickness)
    mask[fade_region] = dist_to_border[fade_region] / fade_thickness

    # Mask in the full white region
    white_region = dist_to_border >= fade_thickness
    mask[white_region] = 1.0

    # Apply Gaussian blur to smooth the white edges
    mask = mask[None, None, :, :]  # Aggiungi batch e channel
    mask = torchvision.transforms.functional.gaussian_blur(mask, kernel_size=(blur_radius, blur_radius), sigma=(blur_radius / 2))
    mask = mask[0, 0, :, :]  # Rimuovi batch e channel

    return mask

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/crop.py
def prepare_paste_back(mask_crop, crop_M_c2o, dsize, interpolation=v2.InterpolationMode.BILINEAR):
    """prepare mask for later image paste back
    """
    t = trans.SimilarityTransform()
    t.params[0:2] = crop_M_c2o

    # pad image by image size
    mask_crop = pad_image_by_size(mask_crop, (dsize[0], dsize[1]))

    mask_ori = v2.functional.affine(mask_crop, t.rotation*57.2958, translate=(t.translation[0], t.translation[1]), scale=t.scale, shear=(0.0, 0.0), interpolation=interpolation, center=(0, 0))
    mask_ori = v2.functional.crop(mask_ori, 0,0, dsize[0], dsize[1]) # cols, rows

    return mask_ori

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/crop.py
def paste_back(img_crop, M_c2o, img_ori, mask_ori, interpolation=v2.InterpolationMode.BILINEAR):
    """paste back the image
    """
    dsize = (img_ori.shape[1], img_ori.shape[2])
    t = trans.SimilarityTransform()
    t.params[0:2] = M_c2o

    # pad image by image size
    img_crop = pad_image_by_size(img_crop, dsize)

    output = v2.functional.affine(img_crop, t.rotation*57.2958, translate=(t.translation[0], t.translation[1]), scale=t.scale, shear=(0.0, 0.0), interpolation=interpolation, center=(0, 0))
    output = v2.functional.crop(output, 0,0, dsize[0], dsize[1]) # cols, rows

    # Converti i tensor al tipo appropriato prima delle operazioni in-place
    output = output.float()  # Converte output in torch.float32
    img_ori = img_ori.float()  # Assicura che img_ori sia float

    # Ottimizzazione con operazioni in-place
    output.mul_(mask_ori)  # In-place multiplication
    output.add_(img_ori.mul_(1 - mask_ori))  # In-place addition and multiplication
    output.clamp_(0, 255)  # In-place clamping
    output = output.to(torch.uint8)

    return output

def paste_back_adv(img_crop, M_c2o, img, mask_crop, interpolation=v2.InterpolationMode.BILINEAR):
    """
    Paste back the transformed cropped image onto the original image with a mask.

    Parameters:
    - img_crop (torch.Tensor: float32): Cropped image tensor (C x H x W).
    - M_c2o (numpy array): Rotation/Translation matrix.
    - img (torch.Tensor: uint8): Original image tensor (C x H x W).
    - mask_crop (torch.Tensor: float32): Mask image tensor (1 x H x W) con bordi sfumati.
    - interpolation: InterpolationMode.

    Returns:
    - img (torch.Tensor: uint8): Modified image tensor.
    """
    
    tform = trans.SimilarityTransform()
    tform.params[0:2] = M_c2o
    corners = np.array([[0, 0], [0, 511], [511, 0], [511, 511]])

    # Calcola i nuovi limiti
    x = (M_c2o[0][0] * corners[:, 0] + M_c2o[0][1] * corners[:, 1] + M_c2o[0][2])
    y = (M_c2o[1][0] * corners[:, 0] + M_c2o[1][1] * corners[:, 1] + M_c2o[1][2])

    left = max(floor(np.min(x)), 0)
    top = max(floor(np.min(y)), 0)
    right = min(ceil(np.max(x)), img.shape[2])
    bottom = min(ceil(np.max(y)), img.shape[1])

    # Converti img in float32 [0, 1]
    img = torch.clamp(img.float() / 255.0, 0, 1)

    # Trasforma img_crop senza inverso
    img_crop = v2.functional.pad(img_crop, (0, 0, img.shape[2] - 512, img.shape[1] - 512))
    img_crop = v2.functional.affine(img_crop, tform.rotation * 57.2958, 
                                    (tform.translation[0], tform.translation[1]), 
                                    tform.scale, 0, interpolation=interpolation, center=(0, 0))
    img_crop = img_crop[:, top:bottom, left:right]  # Ritaglia l'area trasformata

    # Trasforma mask_crop nello stesso modo di img_crop
    mask_crop = v2.functional.pad(mask_crop, (0, 0, img.shape[2] - 512, img.shape[1] - 512))
    mask_crop = v2.functional.affine(mask_crop, tform.rotation * 57.2958, 
                                     (tform.translation[0], tform.translation[1]), 
                                     tform.scale, 0, interpolation=interpolation, center=(0, 0))
    mask_crop = mask_crop[:, top:bottom, left:right]
    
    # Clampa la maschera tra 0 e 1
    mask_crop = torch.clamp(mask_crop, 0, 1)

    # Crea il complemento della maschera per l'area dell'immagine originale
    mask_inv = 1 - mask_crop

    # Applica mask_crop a img_crop e mask_inv a img_diff
    img_diff = img[:, top:bottom, left:right]
    img_crop = torch.mul(mask_crop, img_crop)  # Applica la maschera sfumata al ritaglio
    img_diff = torch.mul(mask_inv, img_diff)   # Applica il complemento all'area originale

    # Somma img_crop (trasformato) all'area ritagliata dell'immagine originale
    img_crop = torch.add(img_crop, img_diff)

    # Inserisci l'area modificata nell'immagine originale (ancora in float)
    img[:, top:bottom, left:right] = img_crop

    # Alla fine converti tutto in uint8
    img = torch.clamp(img * 255.0, 0, 255).to(torch.uint8)

    return img

def paste_back_kgm(img_crop, M_c2o, img_ori, mask_ori):
    """paste back the image
    """
    dsize = (img_ori.shape[1], img_ori.shape[2])

    # pad image by image size
    img_crop = pad_image_by_size(img_crop, (img_ori.shape[1], img_ori.shape[2]))

    img_crop = img_crop.float()
    img_back = transform_img_kgm(img_crop, M_c2o, dsize=dsize)
    img_back = torch.clip(mask_ori * img_back + (1 - mask_ori) * img_ori, 0, 255)

    return img_back.to(torch.uint8)

def transform_img_kgm(img, M, dsize, mode='bilinear', padding_mode='zeros', align_corners=True, fill_value=(0, 0, 0)):
    """ Conduct similarity or affine transformation to the image using PyTorch CUDA.

    Args:
    img (torch.Tensor): Input image tensor (C x H x W)
    M (torch.Tensor): 2x3 or 3x3 transformation matrix
    dsize (tuple[int, int]): size of the output image (height, width).
    mode(str, optional): interpolation mode to calculate output values 'bilinear' | 'nearest'. Default: "bilinear"
    align_corners (bool, optional): mode for grid_generation. Default: True
    fill_value (Tensor, optional): tensor of shape that fills the padding area. Only supported for RGB. Default: zeros(3)

    Returns:
    torch.Tensor: Transformed image
    """
    if isinstance(dsize, tuple) or isinstance(dsize, list):
        _dsize = tuple(dsize)
    else:
        _dsize = (dsize, dsize)

    # Convert M to a torch tensor if it is a numpy.ndarray
    if isinstance(M, np.ndarray):
        M = torch.from_numpy(M).to(img.device, dtype=img.dtype)

        # Prepare the transformation matrix
    M = M[:2, :]  # Ensure it's a 2x3 matrix

    img_transformed = kgm.warp_affine(src=img.unsqueeze(0), M=M[None], dsize=(_dsize[0], _dsize[1]), mode=mode, padding_mode=padding_mode, align_corners=align_corners, fill_value=fill_value)
    img_transformed = img_transformed.squeeze(0)

    return img_transformed

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/live_portrait_wrapper.py
def calculate_distance_ratio(lmk: np.ndarray, idx1: int, idx2: int, idx3: int, idx4: int, eps: float = 1e-6) -> np.ndarray:
    return (np.linalg.norm(lmk[:, idx1] - lmk[:, idx2], axis=1, keepdims=True) /
            (np.linalg.norm(lmk[:, idx3] - lmk[:, idx4], axis=1, keepdims=True) + eps))

def calc_eye_close_ratio(lmk: np.ndarray, target_eye_ratio: np.ndarray = None) -> np.ndarray:
    lefteye_close_ratio = calculate_distance_ratio(lmk, 6, 18, 0, 12)
    righteye_close_ratio = calculate_distance_ratio(lmk, 30, 42, 24, 36)
    if target_eye_ratio is not None:
        return np.concatenate([lefteye_close_ratio, righteye_close_ratio, target_eye_ratio], axis=1)
    else:
        return np.concatenate([lefteye_close_ratio, righteye_close_ratio], axis=1)

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/live_portrait_wrapper.py
def calc_lip_close_ratio(lmk: np.ndarray) -> np.ndarray:
    return calculate_distance_ratio(lmk, 90, 102, 48, 66)

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/camera.py
def headpose_pred_to_degree(pred):
    """
    Converts a headpose prediction to degrees.

    Args:
        pred: (bs, 66) or (bs, 1) or other shapes.
              (bs, 66) indicates a classification task with 66 classes.

    Returns:
        degree: Converted headpose prediction in degrees if input shape is (bs, 66).
                Otherwise, returns the input as is.
    """
    # Check if pred is (bs, 66)
    if pred.ndim > 1 and pred.shape[1] == 66:
        # Get the device of the input tensor
        device = pred.device
        
        # Create an index tensor [0, 1, 2, ..., 65]
        idx_tensor = [idx for idx in range(0, 66)]
        idx_tensor = torch.FloatTensor(idx_tensor).to(device)
        
        # Apply softmax to get probabilities over the 66 classes
        pred = torch.nn.functional.softmax(pred, dim=1)
        
        # Calculate the weighted sum (degree estimation)
        # This step computes the sum of probabilities * indices and scales the result
        degree = torch.sum(pred * idx_tensor, axis=1) * 3 - 97.5
        
        return degree

    # If input is not (bs, 66), return it unchanged
    return pred

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/camera.py
def get_rotation_matrix(pitch_, yaw_, roll_):
    """ The input angles are in degrees """
    
    # If the inputs are scalar or lists, convert them to tensors
    if not isinstance(pitch_, torch.Tensor):
        pitch_ = torch.tensor(pitch_)
    if not isinstance(yaw_, torch.Tensor):
        yaw_ = torch.tensor(yaw_)
    if not isinstance(roll_, torch.Tensor):
        roll_ = torch.tensor(roll_)

    # Convert degrees to radians
    pitch = pitch_ / 180 * torch.pi
    yaw = yaw_ / 180 * torch.pi
    roll = roll_ / 180 * torch.pi

    # Get the device (either CPU or GPU)
    device = pitch.device

    # If the tensors are one-dimensional, add an extra dimension
    if pitch.ndim == 1:
        pitch = pitch.unsqueeze(1)
    if yaw.ndim == 1:
        yaw = yaw.unsqueeze(1)
    if roll.ndim == 1:
        roll = roll.unsqueeze(1)

    # Calculate rotation matrices for pitch, yaw, and roll
    bs = pitch.shape[0]  # Batch size
    ones = torch.ones([bs, 1]).to(device)
    zeros = torch.zeros([bs, 1]).to(device)

    # Rotation matrix around x-axis (pitch)
    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(pitch), -torch.sin(pitch),
        zeros, torch.sin(pitch), torch.cos(pitch)
    ], dim=1).reshape([bs, 3, 3])

    # Rotation matrix around y-axis (yaw)
    rot_y = torch.cat([
        torch.cos(yaw), zeros, torch.sin(yaw),
        zeros, ones, zeros,
        -torch.sin(yaw), zeros, torch.cos(yaw)
    ], dim=1).reshape([bs, 3, 3])

    # Rotation matrix around z-axis (roll)
    rot_z = torch.cat([
        torch.cos(roll), -torch.sin(roll), zeros,
        torch.sin(roll), torch.cos(roll), zeros,
        zeros, zeros, ones
    ], dim=1).reshape([bs, 3, 3])

    # Combine the rotations (z, y, x)
    rot = rot_z @ rot_y @ rot_x

    # Return the transposed rotation matrix
    return rot.permute(0, 2, 1)  # transpose

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/live_portrait_wrapper.py
def transform_keypoint(kp_info: dict):
    """
    Transforms the keypoints using the pose (pitch, yaw, roll), shift (translation), and expression deformation.
    
    Args:
        kp_info: A dictionary containing the following keys:
            - 'kp': Tensor of shape (bs, k, 3), the keypoints.
            - 'pitch', 'yaw', 'roll': Tensors representing head pose angles.
            - 't': Translation vector (bs, 3).
            - 'exp': Expression deformation vector (bs, k, 3).
            - 'scale': Scaling factor.
    
    Returns:
        kp_transformed: Transformed keypoints of shape (bs, k, 3).
    """
    kp = kp_info['kp']    # (bs, k, 3) keypoints
    pitch, yaw, roll = kp_info['pitch'], kp_info['yaw'], kp_info['roll']
    t, exp = kp_info['t'], kp_info['exp']
    scale = kp_info['scale']

    # Convert pose angles to degrees
    pitch = headpose_pred_to_degree(pitch)
    yaw = headpose_pred_to_degree(yaw)
    roll = headpose_pred_to_degree(roll)

    # Determine the batch size
    bs = kp.shape[0]
    
    # Determine the number of keypoints
    if kp.ndim == 2:
        num_kp = kp.shape[1] // 3  # For shape (bs, num_kpx3)
    else:
        num_kp = kp.shape[1]  # For shape (bs, num_kp, 3)

    # Get the rotation matrix based on pitch, yaw, and roll
    rot_mat = get_rotation_matrix(pitch, yaw, roll)  # (bs, 3, 3)

    # Apply the transformation: s * (R * x_c,s + exp) + t
    kp_transformed = kp.view(bs, num_kp, 3) @ rot_mat + exp.view(bs, num_kp, 3)
    kp_transformed *= scale[..., None]  # Apply scaling
    
    # Apply translation, only to x and y (ignore z)
    kp_transformed[:, :, 0:2] += t[:, None, 0:2]  

    return kp_transformed

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/gradio_pipeline.py
@torch.no_grad()
def update_delta_new_eyeball_direction(eyeball_direction_x, eyeball_direction_y, delta_new, **kwargs):
    if eyeball_direction_x > 0:
            delta_new[0, 11, 0] += eyeball_direction_x * 0.0007
            delta_new[0, 15, 0] += eyeball_direction_x * 0.001
    else:
        delta_new[0, 11, 0] += eyeball_direction_x * 0.001
        delta_new[0, 15, 0] += eyeball_direction_x * 0.0007

    delta_new[0, 11, 1] += eyeball_direction_y * -0.001
    delta_new[0, 15, 1] += eyeball_direction_y * -0.001
    blink = -eyeball_direction_y / 2.

    delta_new[0, 11, 1] += blink * -0.001
    delta_new[0, 13, 1] += blink * 0.0003
    delta_new[0, 15, 1] += blink * -0.001
    delta_new[0, 16, 1] += blink * 0.0003

    return delta_new

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/gradio_pipeline.py
@torch.no_grad()
def update_delta_new_smile(smile, delta_new, **kwargs):
    delta_new[0, 20, 1] += smile * -0.01
    delta_new[0, 14, 1] += smile * -0.02
    delta_new[0, 17, 1] += smile * 0.0065
    delta_new[0, 17, 2] += smile * 0.003
    delta_new[0, 13, 1] += smile * -0.00275
    delta_new[0, 16, 1] += smile * -0.00275
    delta_new[0, 3, 1] += smile * -0.0035
    delta_new[0, 7, 1] += smile * -0.0035

    return delta_new

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/gradio_pipeline.py
@torch.no_grad()
def update_delta_new_wink(wink, delta_new, **kwargs):
    delta_new[0, 11, 1] += wink * 0.001
    delta_new[0, 13, 1] += wink * -0.0003
    delta_new[0, 17, 0] += wink * 0.0003
    delta_new[0, 17, 1] += wink * 0.0003
    delta_new[0, 3, 1] += wink * -0.0003

    return delta_new

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/gradio_pipeline.py
@torch.no_grad()
def update_delta_new_eyebrow(eyebrow, delta_new, **kwargs):
    if eyebrow > 0:
        delta_new[0, 1, 1] += eyebrow * 0.001
        delta_new[0, 2, 1] += eyebrow * -0.001
    else:
        delta_new[0, 1, 0] += eyebrow * -0.001
        delta_new[0, 2, 0] += eyebrow * 0.001
        delta_new[0, 1, 1] += eyebrow * 0.0003
        delta_new[0, 2, 1] += eyebrow * -0.0003

    return delta_new

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/gradio_pipeline.py
@torch.no_grad()
def update_delta_new_lip_variation_zero(lip_variation_zero, delta_new, **kwargs):
    delta_new[0, 19, 0] += lip_variation_zero

    return delta_new

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/gradio_pipeline.py
@torch.no_grad()
def update_delta_new_lip_variation_one(lip_variation_one, delta_new, **kwargs):
    delta_new[0, 14, 1] += lip_variation_one * 0.001
    delta_new[0, 3, 1] += lip_variation_one * -0.0005
    delta_new[0, 7, 1] += lip_variation_one * -0.0005
    delta_new[0, 17, 2] += lip_variation_one * -0.0005

    return delta_new

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/gradio_pipeline.py
@torch.no_grad()
def update_delta_new_lip_variation_two(lip_variation_two, delta_new, **kwargs):
    delta_new[0, 20, 2] += lip_variation_two * -0.001
    delta_new[0, 20, 1] += lip_variation_two * -0.001
    delta_new[0, 14, 1] += lip_variation_two * -0.001

    return delta_new

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/gradio_pipeline.py
@torch.no_grad()
def update_delta_new_lip_variation_three(lip_variation_three, delta_new, **kwargs):
    delta_new[0, 19, 1] += lip_variation_three * 0.001
    delta_new[0, 19, 2] += lip_variation_three * 0.0001
    delta_new[0, 17, 1] += lip_variation_three * -0.0001

    return delta_new

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/gradio_pipeline.py
@torch.no_grad()
def update_delta_new_mov_x(mov_x, delta_new, **kwargs):
    delta_new[0, 5, 0] += mov_x

    return delta_new

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/gradio_pipeline.py
@torch.no_grad()
def update_delta_new_mov_y(mov_y, delta_new, **kwargs):
    delta_new[0, 5, 1] += mov_y

    return delta_new

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/live_portrait_wrapper.py
def calc_combined_eye_ratio(c_d_eyes_i, source_lmk, device='cuda'):
    c_s_eyes = calc_eye_close_ratio(source_lmk[None])
    c_s_eyes_tensor = torch.from_numpy(c_s_eyes).float().to(device)
    #c_d_eyes_i_tensor = torch.Tensor([c_d_eyes_i[0][0]]).reshape(1, 1).to(device)
    c_d_eyes_i_numpy = np.array([c_d_eyes_i[0][0]], dtype=np.float32)  # Assicurati che sia un array NumPy
    c_d_eyes_i_tensor = torch.from_numpy(c_d_eyes_i_numpy).reshape(1, 1).to(device)
    # [c_s,eyes, c_d,eyes,i]
    combined_eye_ratio_tensor = torch.cat([c_s_eyes_tensor, c_d_eyes_i_tensor], dim=1)

    return combined_eye_ratio_tensor

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/live_portrait_wrapper.py
def calc_combined_lip_ratio(c_d_lip_i, source_lmk, device='cuda'):
    c_s_lip = calc_lip_close_ratio(source_lmk[None])
    c_s_lip_tensor = torch.from_numpy(c_s_lip).float().to(device)
    #c_d_lip_i_tensor = torch.Tensor([c_d_lip_i[0]]).to(device).reshape(1, 1) # 1x1
    c_d_lip_i_numpy = np.array([c_d_lip_i[0]], dtype=np.float32)  # Assicurati che sia un array NumPy
    c_d_lip_i_tensor = torch.from_numpy(c_d_lip_i_numpy).to(device).reshape(1, 1)  # 1x1
    # [c_s,lip, c_d,lip,i]
    combined_lip_ratio_tensor = torch.cat([c_s_lip_tensor, c_d_lip_i_tensor], dim=1) # 1x2

    return combined_lip_ratio_tensor

#imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/helper.py
def concat_feat(kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
    """
    kp_source: (bs, k, 3)
    kp_driving: (bs, k, 3)
    Return: (bs, 2k*3)
    """
    bs_src = kp_source.shape[0]
    bs_dri = kp_driving.shape[0]
    assert bs_src == bs_dri, 'batch size must be equal'

    feat = torch.cat([kp_source.view(bs_src, -1), kp_driving.view(bs_dri, -1)], dim=1)
    return feat

def apply_laplace_filter(img):
    # Definiere den Laplace-Kernel
    laplace_kernel = torch.tensor([[0,  1, 0],
                                   [1, -4, 1],
                                   [0,  1, 0]], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)

    # Erweitere den Graustufen-Bild-Tensor für Faltung (Batches und Kanäle hinzufügen)
    img = img.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W) für die Faltung
    # Faltung mit dem Laplace-Kernel durchführen
    laplacian = torch.nn.functional.conv2d(img, laplace_kernel, padding=1)

    return laplacian.squeeze(0).squeeze(0)  # (H, W)

def jpegBlur(img, q):
    device = img.device  # Original device (CPU or GPU)

    # Ensure the image is in [C, H, W] format
    if img.dim() != 3 or img.size(0) != 3:
        raise ValueError("Image must have shape [3, H, W].")

    # Convert to uint8 if necessary
    if img.dtype == torch.float32:
        img_uint8 = img.type(torch.uint8).cpu()
    elif img.dtype == torch.uint8:
        img_uint8 = img.cpu()
    else:
        raise ValueError("Unsupported image data type.")

    # Encode JPEG (works on CPU)
    buffer = torchvision.io.encode_jpeg(img_uint8, quality=q)

    # Decode JPEG (input must be on CPU when using nvjpeg)
    img_blurred = torchvision.io.decode_jpeg(buffer)

    # Move back to the original device
    img_blurred = img_blurred.to(device).type(torch.float32)

    return img_blurred

def histogram_matching(source_image, target_image, diffslider):
    # Determine the device (CPU or GPU)
    device = source_image.device

    # Convert images to float tensors in range [0, 1], shape (C, H, W)
    source_image_t = source_image.float().to(device) / 255.0  # (C, H, W)
    target_image_t = target_image.float().to(device) / 255.0  # (C, H, W)

    matched_target_image_t = target_image_t.clone()

    bin_edges = torch.linspace(0.0, 1.0, steps=257, device=device)  # 257 edges for 256 bins

    for channel in range(3):
        source_channel = source_image_t[channel, :, :]  # Shape: (H, W)
        target_channel = target_image_t[channel, :, :]

        # Compute histograms
        source_hist = torch.histc(source_channel, bins=256, min=0.0, max=1.0)
        target_hist = torch.histc(target_channel, bins=256, min=0.0, max=1.0)

        # Compute probability mass functions (PMFs)
        source_pmf = source_hist / source_hist.sum()
        target_pmf = target_hist / target_hist.sum()

        # Compute cumulative distribution functions (CDFs)
        source_cdf = torch.cumsum(source_pmf, dim=0)
        target_cdf = torch.cumsum(target_pmf, dim=0)

        # Flatten the target channel for interpolation
        target_channel_flat = target_channel.flatten()

        # Interpolate target pixel values to get their CDF values
        interp_t_values = interp1d(
            target_channel_flat, bin_edges[:-1], target_cdf, device=device
        )

        # Invert the source CDF to get matched pixel values
        matched_channel_flat = interp1d_inverse(
            interp_t_values, source_cdf, bin_edges[:-1], device=device
        )

        # Reshape back to original image shape
        matched_channel = matched_channel_flat.reshape(target_channel.shape)

        # Update the matched image
        matched_target_image_t[channel, :, :] = matched_channel

    # Blend the images according to diffslider
    alpha = diffslider / 100.0
    final_image_t = (1 - alpha) * target_image_t + alpha * matched_target_image_t

    # Scale back to [0, 255] and clip
    final_image_t = torch.clamp(final_image_t * 255.0, 0.0, 255.0)

    # Ensure it's on the original device and has type float
    final_image_tensor = final_image_t.to(device).float()

    return final_image_tensor

def histogram_matching_withmask(source_image, target_image, mask, diffslider):
    # Determine the device (CPU or GPU)
    device = source_image.device

    # Convert images to float tensors in range [0, 1], shape (C, H, W)
    source_image_t = source_image.float().to(device) / 255.0  # (C, H, W)
    target_image_t = target_image.float().to(device) / 255.0  # (C, H, W)
    mask_t = mask.float().to(device)

    # Apply histogram matching only to the masked areas
    matched_target_image_t = target_image_t.clone()

    # Define the condition for the mask
    valid_mask = (mask_t > 0.2)  # Shape: (1, H, W) or (H, W)

    # Remove channel dimension from mask if present
    if valid_mask.dim() == 3 and valid_mask.size(0) == 1:
        valid_mask = valid_mask.squeeze(0)

    # Create bin edges for histograms
    bin_edges = torch.linspace(0.0, 1.0, steps=257, device=device)  # 257 edges for 256 bins

    for channel in range(3):
        source_channel = source_image_t[channel, :, :]  # Shape: (H, W)
        target_channel = target_image_t[channel, :, :]

        # Extract masked values
        masked_source_values = source_channel[valid_mask]
        masked_target_values = target_channel[valid_mask]

        # Remove NaNs and Infs
        masked_source_values = masked_source_values[~torch.isnan(masked_source_values)]
        masked_source_values = masked_source_values[~torch.isinf(masked_source_values)]
        masked_target_values = masked_target_values[~torch.isnan(masked_target_values)]
        masked_target_values = masked_target_values[~torch.isinf(masked_target_values)]

        # Check if masked values are empty
        if masked_source_values.numel() == 0 or masked_target_values.numel() == 0:
            print(f"No valid masked pixels for channel {channel}. Skipping histogram matching for this channel.")
            continue

        # Ensure values are within [0.0, 1.0]
        masked_source_values = torch.clamp(masked_source_values, 0.0, 1.0)
        masked_target_values = torch.clamp(masked_target_values, 0.0, 1.0)

        # Compute histograms
        source_hist = torch.histc(masked_source_values, bins=256, min=0.0, max=1.0)
        target_hist = torch.histc(masked_target_values, bins=256, min=0.0, max=1.0)

        # Add epsilon to histogram counts to prevent zeros
        source_hist += 1e-6
        target_hist += 1e-6

        # Compute probability mass functions (PMFs)
        source_hist_sum = source_hist.sum()
        target_hist_sum = target_hist.sum()
        if source_hist_sum == 0 or target_hist_sum == 0:
            print(f"Histogram sum is zero for channel {channel}. Skipping histogram matching for this channel.")
            continue

        source_pmf = source_hist / source_hist_sum
        target_pmf = target_hist / target_hist_sum

        # Compute cumulative distribution functions (CDFs)
        source_cdf = torch.cumsum(source_pmf, dim=0)
        target_cdf = torch.cumsum(target_pmf, dim=0)

        # Ensure CDFs are strictly increasing
        source_cdf = torch.maximum(source_cdf, torch.cummax(source_cdf, dim=0)[0])
        target_cdf = torch.maximum(target_cdf, torch.cummax(target_cdf, dim=0)[0])

        # Check for NaNs or Infs in CDFs
        if torch.isnan(source_cdf).any() or torch.isinf(source_cdf).any():
            print(f"Channel {channel}: source_cdf contains NaN or Inf values. Skipping histogram matching for this channel.")
            continue

        # Flatten the target channel for interpolation
        target_channel_flat = target_channel.flatten()

        # Interpolate target pixel values to get their CDF values
        interp_t_values = interp1d(
            target_channel_flat, bin_edges[:-1], target_cdf, device=device
        )

        # Invert the source CDF to get matched pixel values
        matched_channel_flat = interp1d_inverse(
            interp_t_values, source_cdf, bin_edges[:-1], device=device
        )

        # Reshape back to original image shape
        matched_channel = matched_channel_flat.reshape(target_channel.shape)

        # Apply the mapping only to the valid areas
        matched_target_image_t[channel, :, :][valid_mask] = matched_channel[valid_mask]

    # Blend the images according to diffslider
    alpha = diffslider / 100.0
    final_image_t = (1 - alpha) * target_image_t + alpha * matched_target_image_t

    # Scale back to [0, 255] and clip
    final_image_t = torch.clamp(final_image_t * 255.0, 0.0, 255.0)

    # Ensure it's on the original device and has type float
    final_image_tensor = final_image_t.to(device).float()

    return final_image_tensor

def interp1d(x, xp, fp, device='cpu'):
    # Ensure xp is increasing
    assert torch.all(xp[1:] >= xp[:-1]), "xp must be increasing"

    # Move tensors to the specified device and make them contiguous
    x = x.to(device).contiguous()
    xp = xp.to(device).contiguous()
    fp = fp.to(device)

    # Find indices in xp for each x
    indices = torch.searchsorted(xp, x, right=True) - 1
    indices = indices.clamp(0, len(xp) - 2)

    x0 = xp[indices]
    x1 = xp[indices + 1]
    y0 = fp[indices]
    y1 = fp[indices + 1]

    # Compute the slope
    slope = (y1 - y0) / (x1 - x0 + 1e-6)  # Add epsilon to prevent division by zero

    # Compute the interpolated values
    y = y0 + slope * (x - x0)

    # Handle edge cases
    y = torch.where(x < xp[0], fp[0], y)
    y = torch.where(x > xp[-1], fp[-1], y)

    return y

def interp1d_inverse(y, fp, xp, device='cpu'):
    # Ensure fp is increasing
    assert torch.all(fp[1:] >= fp[:-1]), "fp must be increasing"

    # Move tensors to the specified device and make them contiguous
    y = y.to(device).contiguous()
    fp = fp.to(device).contiguous()
    xp = xp.to(device)

    # Find indices in fp for each y
    indices = torch.searchsorted(fp, y, right=True) - 1
    indices = indices.clamp(0, len(fp) - 2)

    y0 = fp[indices]
    y1 = fp[indices + 1]
    x0 = xp[indices]
    x1 = xp[indices + 1]

    # Compute the slope
    slope = (x1 - x0) / (y1 - y0 + 1e-6)  # Add epsilon to prevent division by zero

    # Compute the interpolated values
    x = x0 + slope * (y - y0)

    # Handle edge cases
    x = torch.where(y < fp[0], xp[0], x)
    x = torch.where(y > fp[-1], xp[-1], x)

    return x

def histogram_matching_DFL_test(source_image, target_image, diffslider):
    # Converti i tensori Torch in array di tipo float32 e normalizza le immagini [0, 1]
    source_image = source_image.type(torch.float32) / 255.0  # Forma (C, H, W)
    target_image = target_image.type(torch.float32) / 255.0  # Forma (C, H, W)

    # Converti da RGB a LAB (le funzioni dovrebbero supportare direttamente (C, H, W))
    source = rgb_to_lab(source_image, False)  # Converti in LAB direttamente su (C, H, W)
    target = rgb_to_lab(target_image, False)  # Converti in LAB direttamente su (C, H, W)

    # Calcola media e deviazione standard per canali L, a, b direttamente su (C, H, W)
    target_l_mean, target_l_std = target[0].mean(), target[0].std()
    target_a_mean, target_a_std = target[1].mean(), target[1].std()
    target_b_mean, target_b_std = target[2].mean(), target[2].std()

    source_l_mean, source_l_std = source[0].mean(), source[0].std()
    source_a_mean, source_a_std = source[1].mean(), source[1].std()
    source_b_mean, source_b_std = source[2].mean(), source[2].std()

    # Scala con le deviazioni standard reciproche del fattore proposto dal paper
    target_l = (target[0] - target_l_mean) * (source_l_std / target_l_std) + source_l_mean
    target_a = (target[1] - target_a_mean) * (source_a_std / target_a_std) + source_a_mean
    target_b = (target[2] - target_b_mean) * (source_b_std / target_b_std) + source_b_mean

    # Clamping dei valori
    target_l = torch.clamp(target_l, 0, 100)
    target_a = torch.clamp(target_a, -127, 127)
    target_b = torch.clamp(target_b, -127, 127)

    matched_target_image = torch.stack([target_l, target_a, target_b], 0)  # Forma (C, H, W)

    # Converti da LAB a RGB direttamente su (C, H, W)
    matched_target_image = lab_to_rgb(matched_target_image, False)  # Converti in RGB direttamente

    # Calcolo dell'immagine finale
    final_image = (1 - diffslider / 100) * target_image + (diffslider / 100) * matched_target_image
    final_image = torch.clamp(final_image * 255, 0, 255)  # Converti in intervallo [0, 255]

    return final_image

def histogram_matching_DFL_Orig(source_image, target_image, mask, diffslider):
    # Converti i tensori Torch in array di tipo float32
    source_image = source_image.type(torch.float32) / 255.0  # Forma (C, H, W)
    target_image = target_image.type(torch.float32) / 255.0  # Forma (C, H, W)
    mask = mask.type(torch.float32).squeeze()  # Rimuove dimensioni inutili, Forma (H, W)
    mask_cutoff = 0.2

    # Aggiungi una dimensione per i canali
    mask = mask.unsqueeze(0)  # Forma (1, H, W)

    # Espandi la maschera per coprire tutti i canali
    source_mask = mask.expand(source_image.shape[0], -1, -1)  # Espande a (C, H, W)
    target_mask = mask.expand(target_image.shape[0], -1, -1)  # Espande a (C, H, W)

    # Converti da RGB a LAB (richiede un formato specifico)
    source = rgb_to_lab(source_image, False)  # Converti in LAB direttamente su (C, H, W)
    target = rgb_to_lab(target_image, False)  # Converti in LAB direttamente su (C, H, W)

    # Applica la maschera
    source_input = source.clone()
    if source_mask is not None:
        # Usa la maschera espansa per coprire tutte le dimensioni
        source_input[source_mask < mask_cutoff] = 0.0

    target_input = target.clone()
    if target_mask is not None:
        target_input[target_mask < mask_cutoff] = 0.0

    # Calcola media e deviazione standard per canali L, a, b direttamente su (C, H, W)
    target_l_mean, target_l_std = target_input[0].mean(), target_input[0].std()
    target_a_mean, target_a_std = target_input[1].mean(), target_input[1].std()
    target_b_mean, target_b_std = target_input[2].mean(), target_input[2].std()

    source_l_mean, source_l_std = source_input[0].mean(), source_input[0].std()
    source_a_mean, source_a_std = source_input[1].mean(), source_input[1].std()
    source_b_mean, source_b_std = source_input[2].mean(), source_input[2].std()

    # Scala con le deviazioni standard reciproche del fattore proposto dal paper
    target_l = (target[0] - target_l_mean) * (source_l_std / target_l_std) + source_l_mean
    target_a = (target[1] - target_a_mean) * (source_a_std / target_a_std) + source_a_mean
    target_b = (target[2] - target_b_mean) * (source_b_std / target_b_std) + source_b_mean

    # Clamping dei valori
    target_l = torch.clamp(target_l, 0, 100)
    target_a = torch.clamp(target_a, -127, 127)
    target_b = torch.clamp(target_b, -127, 127)

    matched_target_image = torch.stack([target_l, target_a, target_b], 0)  # Forma (C, H, W)

    # Converti da LAB a RGB direttamente su (C, H, W)
    matched_target_image = lab_to_rgb(matched_target_image, False)  # Converti in RGB direttamente

    # Calcolo dell'immagine finale
    final_image = (1 - diffslider / 100) * target_image + (diffslider / 100) * matched_target_image
    final_image = torch.clamp(final_image * 255, 0, 255)  # Converti in intervallo [0, 255]

    return final_image

def transform_t(img, center, output_size, scale, rotation):
    device = img.device
    dtype = img.dtype
    img = pad_image_by_size(img, output_size)

    scale_ratio = scale
    rot_rad = torch.tensor(rotation * torch.pi / 180.0, device=device, dtype=dtype)
    cos_theta = torch.cos(rot_rad) * scale_ratio
    sin_theta = torch.sin(rot_rad) * scale_ratio

    a = cos_theta
    b = sin_theta
    c = -sin_theta
    d = cos_theta

    cx, cy = center
    cx = cx * scale_ratio
    cy = cy * scale_ratio
    tx = -cx
    ty = -cy
    tx_final = output_size / 2
    ty_final = output_size / 2
    tx_total = tx_final + a * tx + b * ty
    ty_total = ty_final + c * tx + d * ty

    M = torch.tensor([[a, b, tx_total],
                      [c, d, ty_total]], dtype=dtype, device=device)
    img_batch = img.unsqueeze(0)
    grid = torch.nn.functional.affine_grid(M.unsqueeze(0), img_batch.size(), align_corners=False)
    cropped_batch = torch.nn.functional.grid_sample(img_batch, grid, align_corners=False, mode='bilinear')
    cropped = cropped_batch.squeeze(0)

    return cropped, M

def trans_points2d_t(pts, M):
    if pts.dim() != 2 or pts.size(1) != 2:
        raise ValueError("pts deve essere un tensore 2D con dimensione (N, 2)")
    ones_column = torch.ones((pts.size(0), 1), dtype=pts.dtype, device=pts.device)
    homogeneous_pts = torch.cat([pts, ones_column], dim=1)
    transformed_pts = homogeneous_pts @ M.T

    return transformed_pts[:, :2]

def invertAffineTransform_t(M):
    if M.dim() == 2 and M.size() == (2, 3):
        M_H = torch.cat([M, torch.tensor([[0, 0, 1]], device=M.device, dtype=M.dtype)], dim=0)
        IM_H = torch.inverse(M_H)
        IM = IM_H[:2, :]
    else:
        raise ValueError("M deve essere di dimensione (2, 3)")

    return IM

def get_face_orientation_t(face_size, lmk):
    assert lmk.shape == (5, 2), "lmk deve essere un tensore di forma (5, 2)"
    device = lmk.device

    # Aggiungiamo un controllo per portare arcface_src_cuda su CUDA se necessario
    if device != arcface_src_cuda.device:
        arcface_src_cuda = arcface_src_cuda.to(device)

    # Non è necessario ripetere per batch perché `lmk` ha già forma (5, 2)
    src_scaled = (face_size / 112.0) * arcface_src_cuda  # Shape: (5, 2)

    # Calcolo del centro dei landmark
    centroid_lmk = lmk.mean(dim=0, keepdim=True)  # Shape: (1, 2)
    centroid_src = src_scaled.mean(dim=0, keepdim=True)  # Shape: (1, 2)

    # Landmark centrati
    lmk_centered = lmk - centroid_lmk  # Shape: (5, 2)
    src_centered = src_scaled - centroid_src  # Shape: (5, 2)

    # Norme
    norm_lmk = torch.norm(lmk_centered, dim=1).pow(2).sum().unsqueeze(0)  # Shape: (1,)
    norm_src = torch.norm(src_centered, dim=1).pow(2).sum().unsqueeze(0)  # Shape: (1,)
    scale = torch.sqrt(norm_src / norm_lmk)  # Shape: (1,)

    # Scaling dei landmark
    lmk_scaled = lmk_centered * scale  # Shape: (5, 2)

    # Calcolo della matrice di covarianza
    covariance = torch.mm(src_centered.t(), lmk_scaled)  # Shape: (2, 2)
    U, S, V = torch.svd(covariance)

    # Calcolo della matrice di rotazione
    R = torch.mm(U, V.t())  # Shape: (2, 2)

    # Controllo del determinante per garantire una rotazione valida
    det = torch.det(R)
    if det < 0:
        U[:, -1] *= -1
        R = torch.mm(U, V.t())

    # Calcolo dell'angolo in radianti e conversione in gradi
    angle_rad = torch.atan2(R[1, 0], R[0, 0])  # Forma (1,)
    angle_deg = torch.rad2deg(angle_rad)

    return angle_deg

def calculate_lmk_rotation_translation(source_landmarks, target_landmarks):
    """
    Calcola la matrice di rotazione e traslazione tra due insiemi di punti di landmark.
    
    :param source_landmarks: numpy array di dimensione (203, 2) o (203, 3) - Landmark sorgente.
    :param target_landmarks: numpy array di dimensione (203, 2) o (203, 3) - Landmark target.
    :return: (R, t) - Matrice di rotazione e vettore di traslazione.
    """
    
    # Step 1: Calcola i centri di massa di ciascun insieme di punti
    source_center = np.mean(source_landmarks, axis=0)
    target_center = np.mean(target_landmarks, axis=0)

    # Step 2: Centra i punti rispetto al centro di massa
    centered_source = source_landmarks - source_center
    centered_target = target_landmarks - target_center

    # Step 3: Calcola la matrice di covarianza
    covariance_matrix = np.dot(centered_source.T, centered_target)

    # Step 4: Applica la decomposizione SVD
    U, S, Vt = np.linalg.svd(covariance_matrix)

    # Step 5: Calcola la matrice di rotazione
    R = np.dot(Vt.T, U.T)

    # Step 6: Correggi eventuali riflessioni (per mantenere la det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Step 7: Calcola la traslazione
    t = target_center - np.dot(source_center, R)

    return R, t

def rotation_matrix_to_angle(R):
    """
    Converti la matrice di rotazione 2x2 in un angolo (in gradi).
    """
    # Calcola l'angolo di rotazione in radianti
    angle_rad = np.arctan2(R[1, 0], R[0, 0])  # Usando la matrice di rotazione
    # Converti l'angolo in gradi
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def get_matrix_lmk_rotation_translation(R, t):
    """
    Combina la matrice di rotazione e il vettore di traslazione in un'istanza SimilarityTransform.
    
    :param R: Matrice di rotazione 2x2.
    :param t: Vettore di traslazione 2x1.
    :return: Istanza di SimilarityTransform con rotazione e traslazione.
    """
    # Estrai l'angolo di rotazione dalla matrice di rotazione
    rotation_angle = rotation_matrix_to_angle(R)

    # Crea un'istanza di SimilarityTransform usando l'angolo di rotazione e la traslazione
    t = trans.SimilarityTransform(rotation=np.radians(rotation_angle), translation=t)

    M = t.params[0:2]
    
    return M