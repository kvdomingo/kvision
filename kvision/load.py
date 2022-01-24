from loguru import logger
from pathlib import Path
from .config import DATASET_PATH


def load() -> dict[str, list[Path]]:
    folders = [f for f in DATASET_PATH.iterdir() if f.is_dir() and f.name.lower() != "group"]
    members = {
        k: v
        for k, v in zip(
            map(lambda f: f.name, folders),
            [
                [g for g in folder.rglob("*") if g.suffix.lower() in {".jpg", ".png", ".webp", ".jpeg", ".jfif"}]
                for folder in folders
            ],
        )
    }
    logger.info("Loaded: " + str({k: len(v) for k, v in members.items()}))
    return members
