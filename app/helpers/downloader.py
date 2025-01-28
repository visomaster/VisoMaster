import requests
from pathlib import Path
import os

from tqdm import tqdm

from integrity_checker import check_file_integrity

def download_file(model_name: str, file_path: str, correct_hash: str, url: str) -> bool:
    """
    Downloads a file and verifies its integrity.

    Parameters:
    - model_name (str): Name of the model being downloaded.
    - file_path (str): Path where the file will be saved.
    - correct_hash (str): Expected hash value of the file for integrity check.
    - url (str): URL to download the file from.

    Returns:
    - bool: True if the file is downloaded and verified successfully, False otherwise.
    """
    # Remove the file if it already exists and restart download
    if Path(file_path).is_file():
        os.remove(file_path)
        print(f"{file_path} already exists. Re-downloading...")
    
    print(f"Downloading {model_name} from {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=5)
        response.raise_for_status()  # Raise an error for bad HTTP responses (e.g., 404, 500)
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {model_name}: {e}")
        return False

    total_size = int(response.headers.get("content-length", 0))  # File size in bytes
    block_size = 1024  # Size of chunks to download
    max_attempts = 3
    attempt = 1

    def download_and_save():
        """Handles the file download and saves it to disk."""
        with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
            with open(file_path, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
    
    while attempt <= max_attempts:
        try:
            download_and_save()
            
            # Verify file integrity
            if check_file_integrity(file_path, correct_hash):
                print("File integrity verified successfully!")
                print(f"File saved at: {file_path}")
                return True
            else:
                print(f"Integrity check failed for {file_path}. Retrying download (Attempt {attempt}/{max_attempts})...")
                os.remove(file_path)
                attempt += 1
        except requests.exceptions.Timeout:
            print("Connection timed out! Retrying download...")
            attempt += 1
        except Exception as e:
            print(f"An error occurred during download: {e}")
            attempt += 1

    print(f"Failed to download {model_name} after {max_attempts} attempts.")
    return False