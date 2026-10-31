import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Dict

from PIL import Image

from .config import PROJECT_ROOT


VISDRONE_VAL_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    "VisDrone2019-DET-val.zip"
)
VISDRONE_ROOT = PROJECT_ROOT / "datasets" / "VisDrone"
VISDRONE_YAML_PATH = PROJECT_ROOT / "datasets" / "visdrone-val.yaml"
VISDRONE_ARCHIVE_ROOT = "VisDrone2019-DET-val"


def _safe_archive_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError("Unsafe path in VisDrone archive: {}".format(name))
    return member


def _convert_annotation(annotation: Path, image: Path, label: Path) -> None:
    with Image.open(image) as source_image:
        width, height = source_image.size
    labels = []
    for row in annotation.read_text(encoding="utf-8").splitlines():
        if not row.strip():
            continue
        fields = row.split(",")
        if len(fields) < 6:
            raise ValueError("Invalid VisDrone annotation row in {}".format(annotation))
        if fields[4] == "0":
            continue

        x, y, box_width, box_height = map(int, fields[:4])
        class_id = int(fields[5]) - 1
        x_center = (x + box_width / 2) / width
        y_center = (y + box_height / 2) / height
        labels.append(
            "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                class_id,
                x_center,
                y_center,
                box_width / width,
                box_height / height,
            )
        )
    label.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")


def prepare_visdrone_val(dataset_root: Path = VISDRONE_ROOT) -> Dict[str, object]:
    """Download and convert only the VisDrone2019-DET validation split."""
    dataset_root = Path(dataset_root).expanduser().resolve()
    images_dir = dataset_root / "images" / "val"
    labels_dir = dataset_root / "labels" / "val"

    if images_dir.is_dir() and labels_dir.is_dir():
        image_stems = {image.stem for image in images_dir.glob("*.jpg")}
        label_stems = {label.stem for label in labels_dir.glob("*.txt")}
        if image_stems and image_stems == label_stems:
            return {
                "dataset_root": str(dataset_root),
                "images": len(image_stems),
                "labels": len(label_stems),
                "downloaded": False,
            }

    if dataset_root.exists():
        raise FileExistsError(
            "VisDrone dataset directory exists but is incomplete: {}. "
            "Move or repair it before preparing the dataset again.".format(dataset_root)
        )

    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".visdrone-val-", dir=str(dataset_root.parent)
    ) as temporary_dir:
        temporary_root = Path(temporary_dir)
        archive_path = temporary_root / "VisDrone2019-DET-val.zip"
        request = urllib.request.Request(
            VISDRONE_VAL_URL,
            headers={"User-Agent": "VideoCV-model-benchmark"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            with archive_path.open("wb") as archive_file:
                while chunk := response.read(1024 * 1024):
                    archive_file.write(chunk)

        if not zipfile.is_zipfile(archive_path):
            raise ValueError("Downloaded VisDrone validation file is not a ZIP archive")

        extracted_images = temporary_root / "source-images"
        extracted_annotations = temporary_root / "source-annotations"
        extracted_images.mkdir()
        extracted_annotations.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            for entry in archive.infolist():
                member = _safe_archive_member(entry.filename)
                parts = member.parts
                if (
                    len(parts) != 3
                    or parts[0] != VISDRONE_ARCHIVE_ROOT
                    or parts[1] not in {"images", "annotations"}
                    or entry.is_dir()
                ):
                    continue
                if parts[1] == "images" and Path(parts[2]).suffix.lower() == ".jpg":
                    target = extracted_images / Path(parts[2]).name
                elif parts[1] == "annotations" and Path(parts[2]).suffix.lower() == ".txt":
                    target = extracted_annotations / Path(parts[2]).name
                else:
                    continue
                with archive.open(entry) as source, target.open("wb") as destination:
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)

        image_paths = sorted(extracted_images.glob("*.jpg"))
        annotation_paths = sorted(extracted_annotations.glob("*.txt"))
        if not image_paths or len(image_paths) != len(annotation_paths):
            raise ValueError(
                "VisDrone archive must contain matching image and annotation files"
            )

        prepared_root = temporary_root / "prepared"
        prepared_images = prepared_root / "images" / "val"
        prepared_labels = prepared_root / "labels" / "val"
        prepared_images.mkdir(parents=True)
        prepared_labels.mkdir(parents=True)
        for image in image_paths:
            annotation = extracted_annotations / image.with_suffix(".txt").name
            if not annotation.is_file():
                raise ValueError("Missing annotation file for {}".format(image.name))
            destination_image = prepared_images / image.name
            os.replace(image, destination_image)
            _convert_annotation(
                annotation, destination_image, prepared_labels / annotation.name
            )

        os.replace(prepared_root, dataset_root)

    return {
        "dataset_root": str(dataset_root),
        "images": len(image_paths),
        "labels": len(annotation_paths),
        "downloaded": True,
    }


def ensure_visdrone_dataset(dataset_yaml: str) -> Dict[str, object]:
    """Prepare VisDrone on demand when its tracked dataset YAML is selected."""
    if Path(dataset_yaml).expanduser().resolve() != VISDRONE_YAML_PATH.resolve():
        return {
            "downloaded": False,
            "skipped": True,
            "reason": "Not the project's VisDrone validation YAML",
        }
    return prepare_visdrone_val()
