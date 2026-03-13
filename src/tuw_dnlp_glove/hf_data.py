from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_HF_REPO_ID = "stanfordnlp/glove"
DEFAULT_HF_FILENAME = "glove.6B.100d.txt"
DEFAULT_HF_REPO_TYPE = "dataset"


def resolve_glove_path(
    local_path: str | None,
    *,
    repo_id: str = DEFAULT_HF_REPO_ID,
    filename: str = DEFAULT_HF_FILENAME,
    repo_type: str = DEFAULT_HF_REPO_TYPE,
) -> Path:
    """
    Return a local path to the GloVe file.

    If the user supplied a local file, use it directly.
    Otherwise download from Hugging Face and return the cached path.
    """
    if local_path:
        path = Path(local_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"local GloVe file not found: {path}")
        return path

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
    )
    return Path(downloaded)
