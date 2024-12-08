import os
import cv2
image_extensions = ('.jpg', '.jpeg', '.jpe', '.png', '.webp', '.tif', '.tiff', '.jp2', '.exr', '.hdr', '.ras', '.pnm', '.ppm', '.pgm', '.pbm', '.pfm')
video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp')

DFM_MODELS_PATH = './App/ONNXModels/DFM_Models'

DFM_MODELS_DATA = {}
def absoluteFilePaths(directory: str, include_subfolders=False):
    if include_subfolders:
        for dirpath,_,filenames in os.walk(directory):
            for f in filenames:
                yield os.path.abspath(os.path.join(dirpath, f))
    else:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                yield file_path

def truncate_text(text):
    if len(text) >= 25:
        return f'{text[:22]}...'
    return text

def get_video_files(folder_name, include_subfolders=False):
    return [f for f in absoluteFilePaths(folder_name, include_subfolders) if f.lower().endswith(video_extensions)]

def get_image_files(folder_name, include_subfolders=False):
    return [f for f in absoluteFilePaths(folder_name, include_subfolders) if f.lower().endswith(image_extensions)]

def is_image_file(file_name: str):
    return file_name.lower().endswith(image_extensions)

def is_video_file(file_name: str):
    return file_name.lower().endswith(video_extensions)

def get_file_type(file_name):
    if is_image_file(file_name):
        return 'image'
    if is_video_file(file_name):
        return 'video'
    return None

def get_dfm_models_data():
    DFM_MODELS_DATA.clear()
    for dfm_file in os.listdir(DFM_MODELS_PATH):
        if dfm_file.endswith(('.dfm','.onnx')):
            DFM_MODELS_DATA[dfm_file] = f'{DFM_MODELS_PATH}/{dfm_file}'
    return DFM_MODELS_DATA
    
def get_dfm_models_selection_values():
    return list(get_dfm_models_data().keys())
def get_dfm_models_default_value():
    dfm_values = list(DFM_MODELS_DATA.keys())
    if dfm_values:
        return dfm_values[0]
    return ''

def get_scaled_resolution(media_capture: cv2.VideoCapture):
    max_height = 1080
    max_width = 1920

    media_width = media_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    media_height = media_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if media_width > max_width or media_height > max_height:
        width_scale = max_width/media_width
        height_scale = max_height/media_height
        scale = min(width_scale, height_scale)
        media_width,media_height = media_width* scale, media_height*scale
    return int(media_width), int(media_height)