import os

def absoluteFilePaths(directory: str):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

def truncate_text(text):
    if len(text) >= 25:
        return f'{text[:22]}...'
    return text

def get_video_files(folder_name):
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    return [f for f in absoluteFilePaths(folder_name) if f.endswith(video_extensions)]

def get_image_files(folder_name):
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    return [f for f in absoluteFilePaths(folder_name) if f.endswith(image_extensions)]