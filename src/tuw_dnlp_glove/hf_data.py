from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path


GLOVE_URL = "https://huggingface.co/datasets/SLU-CSCI4750/glove.6B.100d.txt/resolve/main/glove.6B.100d.txt.gz"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "tuw_dnlp_glove"
DEFAULT_FILENAME = "glove.6B.100d.txt.gz"


def resolve_glove_path(local_path: str | None, *, url: str = GLOVE_URL) -> Path:
    """
    Return a local path to the decompressed GloVe file.

    If the user provides a local file, use it directly.
    Otherwise download the compressed file from Hugging Face and unpack it into
    a local cache directory.
    """
    if local_path:
        path = Path(local_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"local GloVe file not found: {path}")
        return path

    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    gz_path = DEFAULT_CACHE_DIR / DEFAULT_FILENAME
    txt_path = DEFAULT_CACHE_DIR / "glove.6B.100d.txt"

    if not gz_path.exists():
        urllib.request.urlretrieve(url, gz_path)

    if not txt_path.exists():
        with gzip.open(gz_path, "rb") as src, txt_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)

    return txt_path
