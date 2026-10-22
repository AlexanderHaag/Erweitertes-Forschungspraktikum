"""Download and verify the BigSolDB v2.0 source dataset from Zenodo."""

import argparse
import hashlib
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "BigSolDBv2.0.csv"
DATASET_URL = (
    "https://zenodo.org/api/records/15094979/files/"
    "BigSolDBv2.0.csv/content"
)
EXPECTED_MD5 = "1e6d8a510387746e9faab5f39dc46317"
CHUNK_SIZE = 1024 * 1024


def file_md5(path):
    """Return the MD5 checksum of a file, reading it in bounded chunks."""
    digest = hashlib.md5()
    with path.open("rb") as dataset_file:
        for chunk in iter(lambda: dataset_file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(destination=DATASET_PATH, overwrite=False):
    """Download the pinned Zenodo dataset and verify its published checksum."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if file_md5(destination) == EXPECTED_MD5:
            print(f"Dataset already exists and checksum matches: {destination}")
            return destination
        if not overwrite:
            raise FileExistsError(
                f"{destination} already exists with a different checksum; "
                "pass --force to replace it."
            )

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".part",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            request = Request(
                DATASET_URL,
                headers={"User-Agent": "Erweitertes-Forschungspraktikum dataset downloader"},
            )
            digest = hashlib.md5()
            with urlopen(request, timeout=60) as response:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    temporary_file.write(chunk)
                    digest.update(chunk)

        actual_checksum = digest.hexdigest()
        if actual_checksum != EXPECTED_MD5:
            raise ValueError(
                f"Downloaded dataset checksum mismatch: expected {EXPECTED_MD5}, "
                f"received {actual_checksum}."
            )

        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    print(f"Downloaded and verified dataset: {destination}")
    return destination


def main():
    """Parse command-line options and download the source dataset."""
    parser = argparse.ArgumentParser(
        description="Download BigSolDBv2.0.csv from Zenodo."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing dataset if its checksum does not match",
    )
    args = parser.parse_args()
    download_dataset(overwrite=args.force)


if __name__ == "__main__":
    main()
