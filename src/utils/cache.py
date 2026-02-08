import hashlib
import os
from typing import Tuple


def compute_file_hash(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def index_paths(base_dir: str, key: str) -> Tuple[str, str]:
    os.makedirs(base_dir, exist_ok=True)
    index_path = os.path.join(base_dir, f"{key}_index.json")
    docs_path = os.path.join(base_dir, f"{key}_docs.json")
    return index_path, docs_path
