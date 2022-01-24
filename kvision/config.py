import os
from pathlib import Path


_dataset_path = os.environ.get("DATASET_PATH")

if not _dataset_path:
    raise ValueError("Environment variables are misconfigured.")


DATASET_PATH: Path = Path(_dataset_path).resolve()

ENV: str = os.environ.get("PYTHON_ENV", "production")

IMAGE_HEIGHT: int = 224

IMAGE_WIDTH: int = 224

IMAGE_CHANNELS: int = 3

BATCH_SIZE = 128
