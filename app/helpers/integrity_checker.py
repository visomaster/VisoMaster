import hashlib

BUF_SIZE = 131072  # read in 128kb chunks!

def get_file_hash(file_path: str) -> str:
    hash_sha256 = hashlib.sha256()

    with open(file_path, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            hash_sha256.update(data)  

    # print("SHA256: {0}".format(hash_sha256.hexdigest()))
    return hash_sha256.hexdigest()

def write_hash_to_file(hash: str, hash_file_path: str):
    with open(hash_file_path, 'w') as hash_file:
        hash_file.write(hash)

def get_hash_from_hash_file(hash_file_path: str) -> str:
    with open(hash_file_path, 'r') as hash_file:
        hash_sha256 = hash_file.read().strip()
        return hash_sha256
    
def check_file_integrity(file_path, correct_hash) -> bool:
    actual_hash = get_file_hash(file_path)
    return actual_hash==correct_hash