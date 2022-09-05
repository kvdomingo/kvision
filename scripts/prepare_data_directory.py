import os
import shutil
import sys
from argparse import ArgumentParser
from itertools import chain
from pathlib import Path

import cv2 as cv
from loguru import logger
from pandas import DataFrame
from tqdm import tqdm

from kvision.config import BASE_DIR, DATASET_PATH, GDRIVE_DATASET_PATH


def main(dry_run: bool = False):
    filenames: list[Path] = []
    categories: list[str] = []
    image_extensions = {"jpg", "jpeg", "jfif", "png", "bmp", "webp"}
    video_extensions = {"mp4"}
    video_frame_interval = 5
    valid_extensions = {*image_extensions, *video_extensions}
    for folder in chain(DATASET_PATH.iterdir(), GDRIVE_DATASET_PATH.iterdir()):
        if not folder.is_dir() or folder.name.lower() == "group":
            continue
        for file in folder.rglob("*"):
            if file.suffix.lower().lstrip(".") in valid_extensions:
                filenames.append(file)
                categories.append(folder.name)
    df = DataFrame(dict(filename=filenames, category=categories))
    class_names = set(categories)
    if dry_run:
        initial_count = len(df)
        for _, row in df.iterrows():
            cap = cv.VideoCapture(str(row["filename"]))
            frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
            cap.release()
            initial_count += int(frame_count // video_frame_interval)
        logger.info(f"Found {initial_count} potential files belonging to {len(class_names)} categories")
        sys.exit(0)
    for class_name in class_names:
        os.makedirs(BASE_DIR / "data" / class_name, exist_ok=True)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if row["filename"].suffix.lower().lstrip(".") in image_extensions:
            shutil.copy2(row["filename"], BASE_DIR / "data" / row["category"])
        else:
            cap = cv.VideoCapture(str(row["filename"]))
            frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened():
                logger.error(f"Could not read file {row['filename']}")
            for fc in tqdm(range(frame_count), leave=False):
                ret, frame = cap.read()
                if not ret:
                    break
                if fc % video_frame_interval == 0:
                    save_name = ".".join(row["filename"].name.split(".")[:-1])
                    save_name = f"{save_name}-{str(fc + 1).zfill(4)}.png"
                    cv.imwrite(str(BASE_DIR / "data" / row["category"] / save_name), frame)
            cap.release()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    main(dry_run=args.dry_run)
