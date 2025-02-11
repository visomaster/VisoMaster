import os
import shutil
import cv2
import time
import numpy as np
from functools import wraps
from datetime import datetime
from pathlib import Path
from torchvision.transforms import v2
import threading
lock = threading.Lock()

image_extensions = ('.jpg', '.jpeg', '.jpe', '.png', '.webp', '.tif', '.tiff', '.jp2', '.exr', '.hdr', '.ras', '.pnm', '.ppm', '.pgm', '.pbm', '.pfm')
video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.gif')

DFM_MODELS_PATH = './model_assets/dfm_models'

DFM_MODELS_DATA = {}

def get_scaling_transforms():
    t512 = v2.Resize((512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
    t384 = v2.Resize((384, 384), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
    t256 = v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
    t128 = v2.Resize((128, 128), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
    return t512, t384, t256, t128  

t512, t384, t256, t128 = get_scaling_transforms()

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
    if len(text) >= 50:
        return f'{text[:47]}...'
    return text

def get_video_files(folder_name, include_subfolders=False):
    return [f for f in absoluteFilePaths(folder_name, include_subfolders) if f.lower().endswith(video_extensions)]

def get_image_files(folder_name, include_subfolders=False):
    return [f for f in absoluteFilePaths(folder_name, include_subfolders) if f.lower().endswith(image_extensions)]

def is_image_file(file_name: str):
    return file_name.lower().endswith(image_extensions)

def is_video_file(file_name: str):
    return file_name.lower().endswith(video_extensions)

def is_file_exists(file_path: str) -> bool:
    if not file_path:
        return False
    return Path(file_path).is_file()

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

def benchmark(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Record the start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.perf_counter()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.6f} seconds.")
        return result  # Return the result of the original function
    return wrapper

def read_frame(capture_obj: cv2.VideoCapture, preview_mode=False):
    with lock:
        ret, frame = capture_obj.read()
    if ret and preview_mode:
        pass
        # width, height = get_scaled_resolution(capture_obj)
        # frame = cv2.resize(fr2ame, dsize=(width, height), interpolation=cv2.INTER_LANCZOS4)
    return ret, frame

def read_image_file(image_path):
    try:
        img = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_UNCHANGED)
    except:
        print("Failed To Load: ", image_path)
        return None
    if img is not None and len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)  # Remove alpha channel

    return img

def get_output_file_path(original_media_path, output_folder, media_type='video'):
    date_and_time = datetime.now().strftime(r'%Y_%m_%d_%H_%M_%S')
    input_filename = os.path.basename(original_media_path)
    # Create a temp Path object to split and merge the original filename to get the new output filename
    temp_path = Path(input_filename)
    # output_filename = "{0}_{2}{1}".format(temp_path.stem, temp_path.suffix, date_and_time)
    if media_type=='video':
        output_filename = f'{temp_path.stem}_{date_and_time}.mp4'
    elif media_type=='image':
        output_filename = f'{temp_path.stem}_{date_and_time}.png'
    output_file_path = os.path.join(output_folder, output_filename)
    return output_file_path

def is_ffmpeg_in_path():
    if not cmd_exist('ffmpeg'):
        print("FFMPEG Not found in your system!")
        return False
    return True

def cmd_exist(cmd):
    try:
        return shutil.which(cmd) is not None
    except ImportError:
        return any(
            os.access(os.path.join(path, cmd), os.X_OK)
            for path in os.environ["PATH"].split(os.pathsep)
        )

def get_dir_of_file(file_path):
    if file_path:
        return os.path.dirname(file_path)
    return os.path.curdir
    