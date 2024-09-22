import os
from PySide6 import QtWidgets
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
video_extensions = ('.mp4', '.avi', '.mov', '.mkv')

def absoluteFilePaths(directory: str):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

def truncate_text(text):
    if len(text) >= 25:
        return f'{text[:22]}...'
    return text

def get_video_files(folder_name):
    return [f for f in absoluteFilePaths(folder_name) if f.endswith(video_extensions)]

def get_image_files(folder_name):
    return [f for f in absoluteFilePaths(folder_name) if f.endswith(image_extensions)]

def is_image_file(file_name: str):
    return file_name.endswith(image_extensions)

def is_video_file(file_name: str):
    return file_name.endswith(video_extensions)

def get_file_type(file_name):
    if is_image_file(file_name):
        return 'image'
    if is_video_file(file_name):
        return 'video'
    return None
