import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

from model_benchmark.cli import main


class PrepareDatasetCliTests(unittest.TestCase):
    def test_prepare_visdrone_uses_downloaded_status(self):
        output = io.StringIO()
        with patch(
            "model_benchmark.cli.prepare_visdrone_val",
            return_value={
                "downloaded": True,
                "images": 548,
                "labels": 548,
                "dataset_root": "/tmp/VisDrone",
            },
        ):
            with contextlib.redirect_stdout(output):
                result = main(["prepare-visdrone"])

        self.assertEqual(result, 0)
        self.assertIn("Prepared 548 images and labels", output.getvalue())

    def test_prepare_visdrone_mot_uses_prepared_status(self):
        output = io.StringIO()
        with patch(
            "model_benchmark.cli.prepare_visdrone_mot",
            return_value={
                "prepared": True,
                "sequences": 7,
                "frames": 2846,
                "dataset_root": "/tmp/VisDrone-MOT",
            },
        ):
            with contextlib.redirect_stdout(output):
                result = main(["prepare-visdrone-mot"])

        self.assertEqual(result, 0)
        self.assertIn("Prepared 7 sequences and 2846 frames", output.getvalue())

    def test_mot_example_config_points_to_prepared_dataset(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "config.visdrone-mot.example.json"
        )

        import json

        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        self.assertEqual(config["tracking_dataset"], "datasets/VisDrone-MOT")
        self.assertEqual(config["dataset"], "datasets/visdrone-val.yaml")
        self.assertEqual(config["tracker"], "bytetrack.yaml")


if __name__ == "__main__":
    unittest.main()
