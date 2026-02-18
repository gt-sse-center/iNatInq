"""
Script to upload iNat24 dataset to S3.

Training images: https://ml-inat-competition-datasets.s3.amazonaws.com/2024/train.tar.gz
Training images metadata: https://ml-inat-competition-datasets.s3.amazonaws.com/2024/train.json.tar.gz
"""

import os
import json
import logging
import sys
import tempfile
from pathlib import Path
import boto3  # pyright: ignore[reportMissingTypeStubs]
from botocore.config import Config  # pyright: ignore[reportMissingTypeStubs]
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(filename)s:%(lineno)d [%(levelname)s] %(message)s")


BUCKET = "inquire-train-data"
ENDPOINT = "https://localhost:9000"
SECRET_KEY_ID = ""
SECRET_ACCESS_KEY = ""
REGION = ""

IMAGES_DIR = ""
METADATA_FILE = "train.json"
SUCCESSFUL_FILE = "successful-uploads.json"

# Max number of images to process per script execution
MAX_IMAGE_UPLOADS = 1000


class S3Client:
    def __init__(self) -> None:
        self.client = boto3.client(  # pyright: ignore[reportUnknownMemberType]
            "s3",
            endpoint_url=ENDPOINT,
            aws_access_key_id=SECRET_KEY_ID,
            aws_secret_access_key=SECRET_ACCESS_KEY,
            region_name=REGION,
            config=Config(
                connect_timeout=30,
                read_timeout=30,
                retries={"max_attempts": 3},
            ),
        )

    def upload_image(self, image_path: Path, image_id: str):
        """Raises if upload fails."""
        abs_file_path = str(image_path.absolute())

        self.client.upload_file(abs_file_path, BUCKET, image_id)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


def build_image_id_mapping(path: Path) -> dict[str, str]:
    """Returns mapping of file names to image ids."""
    with open(path) as f:
        obj: dict[str, object] = json.load(f)
    images: list[dict[str, str]] = obj.get("images", None)  # pyright: ignore[reportAssignmentType]
    if not images:
        raise Exception("no images field in metadata json file")

    image_id_map: dict[str, str] = {}
    for image in images:
        file_name = image.get("file_name", None)
        if file_name is None:
            raise Exception("no file_name in image entry")
        assert file_name not in image_id_map

        image_id = image.get("id", None)
        if image_id is None:
            raise Exception("no id in image entry")

        image_id_map[file_name] = image_id

    return image_id_map


def load_successful_set() -> set[str]:
    successful_file = Path(SUCCESSFUL_FILE)
    assert not successful_file.is_dir()
    if not successful_file.exists():
        with open(successful_file, "w") as f:
            f.write("[]")
    with open(successful_file) as f:
        succ_list: list[str] = json.load(f)
        assert isinstance(succ_list, list)
    return set(succ_list)


def save_successful_set(successful: set[str]):
    """
    Save the current list of successful file uploads.
    Does not raise if writing to json file fails.
    """
    try:
        fd, tmp_path = tempfile.mkstemp(dir=".", suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            succ_list = list(successful)
            json.dump(succ_list, f)
        os.replace(tmp_path, SUCCESSFUL_FILE)
    except Exception as e:
        logging.error(f"Failed to save successful uploads list: {e}")


def main():
    if not IMAGES_DIR:
        raise ValueError("IMAGES_DIR must be set to a non-empty path")
    parent_images_dir = Path(IMAGES_DIR)
    assert parent_images_dir.is_dir()

    # Ensure we have permissions to read this directory
    try:
        _ = next(parent_images_dir.iterdir())
    except PermissionError as e:
        logging.error(f"Cannot open {parent_images_dir}: {e}")
        sys.exit(1)

    client = S3Client()

    successful_uploads = load_successful_set()

    image_id_map = build_image_id_mapping(Path(METADATA_FILE))
    logging.info(f"Read metadata for {len(image_id_map)} images...")

    fail_count = 0
    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            # Submit upload tasks to worker pool
            submitted_count = 0
            futures_with_file_name: dict[Future[None], str] = {}
            for f in parent_images_dir.rglob("*"):
                # Skip directories - rglob will return all of its internal contents
                if not f.is_file():
                    continue

                # Name with full path relative to current working directory
                file_name = str(f.relative_to(parent_images_dir.parent))
                try:
                    if file_name in successful_uploads:
                        continue

                    if submitted_count >= MAX_IMAGE_UPLOADS:
                        break

                    image_id = image_id_map.get(file_name, None)
                    if image_id is None:
                        raise Exception(f"no image_id found for file: {file_name}")

                    future = pool.submit(client.upload_image, f, image_id)
                    futures_with_file_name[future] = file_name
                    submitted_count += 1
                except Exception as e:
                    logging.error(f"Upload failed for {file_name}: {e}")
                    fail_count += 1

            # Process resolved tasks
            resolved_count = 0
            for future in as_completed(futures_with_file_name):
                resolved_count += 1
                file_name = "unknown_file_name"
                try:
                    mapped_file_name = futures_with_file_name.get(future, None)
                    assert mapped_file_name is not None
                    file_name = mapped_file_name
                    future.result()
                    successful_uploads.add(file_name)
                except Exception as e:
                    logging.error(f"Upload failed for {file_name}: {e}")
                    fail_count += 1
                finally:
                    if resolved_count % min(MAX_IMAGE_UPLOADS // 2, 1_000) == 0:
                        logging.info(
                            f"Uploading image #{resolved_count}",
                            extra={"successful (all time)": len(successful_uploads), "failed": fail_count},
                        )
                        save_successful_set(successful_uploads)

    except Exception as e:
        logging.error(e)

    finally:
        logging.info(f"Finished execution of {MAX_IMAGE_UPLOADS} file uploads...")
        logging.info(f"Success count: {len(successful_uploads)}")
        logging.info(f"Fail count: {fail_count}")
        save_successful_set(successful_uploads)


if __name__ == "__main__":
    main()
