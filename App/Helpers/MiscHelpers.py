import os

def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

def truncate_text(text):
    if len(text) >= 25:
        return f'{text[:22]}...'
    return text