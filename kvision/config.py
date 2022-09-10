import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def load_env(name: str) -> str:
    _env = os.environ.get(name)
    if not _env:
        raise ValueError(f"Missing env {name}")
    return _env


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_enable_xla_devices")

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = Path(load_env("DATASET_PATH")).resolve()

GDRIVE_DATASET_PATH = Path(load_env("DRIVE_DATASET_PATH")).resolve()

TENSORBOARD_LOG_DIR = Path(load_env("TENSORBOARD_LOG_DIR"))

ENV = os.environ.get("PYTHON_ENV", "production")

IMAGE_HEIGHT = os.environ.get("KV_IMAGE_HEIGHT", 299)

IMAGE_WIDTH = os.environ.get("KV_IMAGE_WIDTH", 299)

IMAGE_CHANNELS = os.environ.get("KV_IMAGE_CHANNELS", 3)

BATCH_SIZE = os.environ.get("KV_BATCH_SIZE", 32)

if BATCH_SIZE % 16 != 0:
    logger.warning(f"{BATCH_SIZE} is not divisible by 16; Tensor Cores will not be used")
