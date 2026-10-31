"""Download and safely prepare the official VisDrone-MOT validation archive."""

import http.cookiejar
import html
import os
import re
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional
from urllib.parse import urlencode, urljoin

from .config import PROJECT_ROOT
from .mot import discover_mot_sequences


VISDRONE_MOT_VAL_URL = (
    "https://drive.google.com/uc?export=download"
    "&id=1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu"
)
VISDRONE_MOT_VAL_PAGE = (
    "https://drive.google.com/file/d/1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu/view"
)
VISDRONE_MOT_ROOT = PROJECT_ROOT / "datasets" / "VisDrone-MOT"
VISDRONE_MOT_ARCHIVE_NAME = "VisDrone2019-MOT-val.zip"
MAX_ARCHIVE_BYTES = 3 * 1024 * 1024 * 1024


def _safe_archive_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts or not member.parts:
        raise ValueError("Unsafe path in VisDrone-MOT archive: {}".format(name))
    return member


def _download_archive(destination: Path) -> None:
    request = urllib.request.Request(
        VISDRONE_MOT_VAL_URL,
        headers={"User-Agent": "VideoCV-model-benchmark"},
    )
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )
    try:
        response = opener.open(request, timeout=60)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not download the official VisDrone-MOT validation archive: {}. "
            "You can download it in a browser from {} and pass its path to "
            "prepare-visdrone-mot --archive.".format(exc, VISDRONE_MOT_VAL_PAGE)
        ) from exc

    with response:
        initial = response.read(4096)
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" in content_type or initial.lstrip().lower().startswith(
            (b"<!doctype html", b"<html")
        ):
            page = (initial + response.read(64 * 1024)).decode(
                "utf-8", errors="replace"
            )
            if "quota" in page.lower() or "too many users" in page.lower():
                raise RuntimeError(
                    "Google Drive download quota is currently exceeded. Download {} "
                    "in a browser and pass the ZIP file to prepare-visdrone-mot "
                    "--archive.".format(VISDRONE_MOT_VAL_PAGE)
                )
            form = re.search(
                r"<form[^>]*action=['\"]([^'\"]+)['\"][^>]*>(.*?)</form>",
                page,
                re.IGNORECASE | re.DOTALL,
            )
            if form is None:
                raise RuntimeError(
                    "Google Drive returned a confirmation page that could not be "
                    "processed. Download {} in a browser and pass the ZIP file to "
                    "prepare-visdrone-mot --archive.".format(VISDRONE_MOT_VAL_PAGE)
                )
            parameters = dict(
                re.findall(
                    r"<input[^>]*name=['\"]([^'\"]+)['\"][^>]*value=['\"]([^'\"]*)['\"][^>]*>",
                    form.group(2),
                    re.IGNORECASE,
                )
            )
            download_url = urljoin(response.geturl(), html.unescape(form.group(1)))
            request = urllib.request.Request(
                "{}?{}".format(download_url, urlencode(parameters)),
                headers={"User-Agent": "VideoCV-model-benchmark"},
            )
            try:
                response = opener.open(request, timeout=60)
            except urllib.error.URLError as exc:
                raise RuntimeError(
                    "Could not download the official VisDrone-MOT validation archive: {}"
                    .format(exc)
                ) from exc
            with response:
                _write_download_response(response, destination)
            return

        _write_download_response(response, destination, initial)


def _write_download_response(
    response: Any, destination: Path, initial: bytes = b""
) -> None:
    headers = response.headers
    content_type = headers.get("Content-Type", "").lower()
    if "text/html" in content_type:
        page = (initial + response.read(64 * 1024)).decode(
            "utf-8", errors="replace"
        )
        if "quota" in page.lower() or "too many users" in page.lower():
            reason = "Google Drive download quota is currently exceeded"
        else:
            reason = "Google Drive returned an error instead of the archive"
        raise RuntimeError(
            "{}. Download {} in a browser and pass the ZIP file to "
            "prepare-visdrone-mot --archive.".format(reason, VISDRONE_MOT_VAL_PAGE)
        )

    content_length = headers.get("Content-Length")
    if content_length and int(content_length) > MAX_ARCHIVE_BYTES:
        raise ValueError(
            "VisDrone-MOT archive exceeds the allowed size of {} bytes".format(
                MAX_ARCHIVE_BYTES
            )
        )
    bytes_written = len(initial)
    with destination.open("wb") as archive_file:
        archive_file.write(initial)
        while chunk := response.read(1024 * 1024):
            bytes_written += len(chunk)
            if bytes_written > MAX_ARCHIVE_BYTES:
                raise ValueError(
                    "VisDrone-MOT archive exceeds the allowed size of {} bytes"
                    .format(MAX_ARCHIVE_BYTES)
                )
            archive_file.write(chunk)


def prepare_visdrone_mot(
    archive_path: Optional[Path] = None,
    dataset_root: Path = VISDRONE_MOT_ROOT,
) -> Dict[str, object]:
    """Extract the official MOT validation archive into a canonical dataset layout."""
    dataset_root = Path(dataset_root).expanduser().resolve()
    if dataset_root.exists():
        try:
            sequences = discover_mot_sequences(dataset_root)
        except ValueError as exc:
            raise FileExistsError(
                "VisDrone-MOT dataset directory exists but is incomplete: {}. "
                "Move or repair it before preparing the dataset again.".format(
                    dataset_root
                )
            ) from exc
        return {
            "dataset_root": str(dataset_root),
            "sequences": len(sequences),
            "frames": sum(
                sum(
                    1
                    for path in sequence.frames_dir.iterdir()
                    if path.suffix.lower()
                    in {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
                )
                for sequence in sequences
            ),
            "prepared": False,
        }

    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".visdrone-mot-", dir=str(dataset_root.parent)
    ) as temporary_dir:
        temporary_root = Path(temporary_dir)
        if archive_path is None:
            archive = temporary_root / VISDRONE_MOT_ARCHIVE_NAME
            _download_archive(archive)
        else:
            archive = Path(archive_path).expanduser().resolve()
            if not archive.is_file():
                raise FileNotFoundError(
                    "VisDrone-MOT archive does not exist: {}".format(archive)
                )

        if not zipfile.is_zipfile(archive):
            raise ValueError("VisDrone-MOT archive is not a valid ZIP file")

        extracted_root = temporary_root / "extracted"
        extracted_root.mkdir()
        with zipfile.ZipFile(archive) as archive_file:
            for entry in archive_file.infolist():
                member = _safe_archive_member(entry.filename)
                mode = entry.external_attr >> 16
                if mode & 0o170000 == 0o120000:
                    raise ValueError(
                        "Symbolic links are not allowed in VisDrone-MOT archive"
                    )
                destination = extracted_root.joinpath(*member.parts)
                if entry.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive_file.open(entry) as source, destination.open("wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)

        sequences = discover_mot_sequences(extracted_root)
        layout_roots = {
            sequence.annotations.parent.parent
            for sequence in sequences
            if sequence.annotations.parent.name == "annotations"
            and sequence.frames_dir.parent.name == "sequences"
            and sequence.frames_dir.parent.parent
            == sequence.annotations.parent.parent
        }
        if len(layout_roots) != 1:
            raise ValueError(
                "VisDrone-MOT archive must contain matching sequences/ and "
                "annotations/ directories"
            )
        prepared_root = next(iter(layout_roots))
        if not (prepared_root / "sequences").is_dir():
            raise ValueError("VisDrone-MOT archive is missing its sequences directory")

        os.replace(prepared_root, dataset_root)
        sequences = discover_mot_sequences(dataset_root)

    return {
        "dataset_root": str(dataset_root),
        "sequences": len(sequences),
        "frames": sum(
            sum(
                1
                for path in sequence.frames_dir.iterdir()
                if path.suffix.lower()
                in {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
            )
            for sequence in sequences
        ),
        "prepared": True,
    }
