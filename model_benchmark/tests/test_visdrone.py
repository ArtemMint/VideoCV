import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from model_benchmark.visdrone import (
    ensure_visdrone_dataset,
    prepare_visdrone_val,
)


def make_archive(*, unsafe_member=False):
    buffer = io.BytesIO()
    image_buffer = io.BytesIO()
    Image.new("RGB", (100, 50)).save(image_buffer, format="JPEG")

    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "VisDrone2019-DET-val/images/000001.jpg", image_buffer.getvalue()
        )
        archive.writestr(
            "VisDrone2019-DET-val/annotations/000001.txt",
            "10,20,30,40,1,1,0,0\n0,0,10,10,0,2,0,0\n",
        )
        if unsafe_member:
            archive.writestr("../outside.txt", "unsafe")
    buffer.seek(0)
    return buffer


class PrepareVisDroneTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.dataset_root = Path(self.temp_dir.name) / "datasets" / "VisDrone"

    def test_downloads_converts_and_reuses_validation_dataset(self):
        archive = make_archive()
        with patch("model_benchmark.visdrone.urllib.request.urlopen", return_value=archive):
            result = prepare_visdrone_val(self.dataset_root)

        self.assertTrue(result["downloaded"])
        self.assertEqual(result["images"], 1)
        self.assertEqual(result["labels"], 1)
        self.assertTrue((self.dataset_root / "images/val/000001.jpg").is_file())
        labels = (self.dataset_root / "labels/val/000001.txt").read_text(
            encoding="utf-8"
        )
        self.assertEqual(labels, "0 0.250000 0.800000 0.300000 0.800000\n")

        with patch("model_benchmark.visdrone.urllib.request.urlopen") as download:
            cached = prepare_visdrone_val(self.dataset_root)

        download.assert_not_called()
        self.assertFalse(cached["downloaded"])
        self.assertEqual(cached["images"], 1)

    def test_rejects_unsafe_archive_members_and_preserves_no_partial_dataset(self):
        archive = make_archive(unsafe_member=True)
        with patch("model_benchmark.visdrone.urllib.request.urlopen", return_value=archive):
            with self.assertRaisesRegex(ValueError, "Unsafe path"):
                prepare_visdrone_val(self.dataset_root)

        self.assertFalse(self.dataset_root.exists())

    def test_does_not_overwrite_an_incomplete_existing_dataset(self):
        self.dataset_root.mkdir(parents=True)

        with patch("model_benchmark.visdrone.urllib.request.urlopen") as download:
            with self.assertRaisesRegex(FileExistsError, "incomplete"):
                prepare_visdrone_val(self.dataset_root)

        download.assert_not_called()

    def test_auto_prepare_only_runs_for_project_visdrone_yaml(self):
        config_path = Path(__file__).resolve().parents[2] / "datasets" / "visdrone-val.yaml"

        with patch(
            "model_benchmark.visdrone.prepare_visdrone_val",
            return_value={"downloaded": True, "images": 548, "labels": 548},
        ) as prepare:
            result = ensure_visdrone_dataset(str(config_path))

        prepare.assert_called_once_with()
        self.assertTrue(result["downloaded"])

        with patch("model_benchmark.visdrone.prepare_visdrone_val") as prepare:
            result = ensure_visdrone_dataset("datasets/custom.yaml")

        prepare.assert_not_called()
        self.assertTrue(result["skipped"])


if __name__ == "__main__":
    unittest.main()
