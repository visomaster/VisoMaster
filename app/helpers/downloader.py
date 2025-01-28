from tqdm import tqdm
import requests
from pathlib import Path
from integrity_checker import check_file_integrity

def download_file(model_name, file_path, hash_file_path, url):

    if Path(file_path).is_file():
        print(f"Skipping {file_path} as it is already downloaded ")
    else:   
        print(f"Downloading {model_name}")     
        response = requests.get(url, stream=True)

        # Sizes in bytes.
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024

        with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
            with open(file_path, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)

        if total_size != 0 and progress_bar.n != total_size:
            raise RuntimeError("Could not download file")
        
        if check_file_integrity(file_path, hash_file_path):
            print("File Integrity Verified Successfully!")
            print(f"File Saved at: {file_path}")