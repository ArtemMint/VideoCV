import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from model_benchmark import ModelRunConfig
from model_benchmark.report import write_summary_report
from model_benchmark.runner import extract_metrics, run_one


class ModelRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def test_extracts_detection_latency_and_per_class_metrics(self):
        validation = SimpleNamespace(
            box=SimpleNamespace(
                mp=0.8,
                mr=0.7,
                map50=0.75,
                map=0.55,
                maps=[0.6, 0.5],
            ),
            speed={
                "preprocess": 1.0,
                "inference": 9.0,
                "postprocess": 2.0,
            },
        )

        metrics = extract_metrics(validation, {0: "person", 1: "car"})

        self.assertEqual(metrics["precision"], 0.8)
        self.assertEqual(metrics["recall"], 0.7)
        self.assertEqual(metrics["map50"], 0.75)
        self.assertEqual(metrics["map50_95"], 0.55)
        self.assertAlmostEqual(metrics["fps_from_validation_speed"], 1000 / 12)
        self.assertEqual(metrics["per_class_map50_95"], {"person": 0.6, "car": 0.5})

    def test_run_writes_individual_reports_and_passes_config_to_validator(self):
        model_path = self.root / "model.pt"
        model_path.touch()
        config = ModelRunConfig(model=model_path, name="unit-test")
        validation_result = SimpleNamespace(
            box=SimpleNamespace(mp=0.9, mr=0.8, map50=0.85, map=0.65, maps=[0.65]),
            speed={"inference": 10.0},
            save_dir=self.root / "ultralytics-results",
        )
        calls = {}

        class FakeYOLO:
            names = {0: "object"}

            def __init__(self, path, task):
                calls["model"] = (path, task)
                self.callbacks = {}

            def add_callback(self, event, callback):
                self.callbacks[event] = callback

            def val(self, **kwargs):
                calls["val"] = kwargs
                self.callbacks["on_val_batch_start"](None)
                self.callbacks["on_val_batch_end"](None)
                self.callbacks["on_val_batch_start"](None)
                self.callbacks["on_val_batch_end"](None)
                return validation_result

        class FakeSampler:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return None

            def as_dict(self):
                return {
                    "process_cpu_percent_avg": 20.0,
                    "rss_baseline_mb": 100.0,
                    "rss_peak_mb": 120.0,
                    "rss_peak_increase_mb": 20.0,
                }

        with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=FakeYOLO)}):
            with patch("model_benchmark.runner._resolve_device", return_value="cpu"):
                with patch("model_benchmark.runner._ResourceSampler", FakeSampler):
                    result = run_one(config, self.root / "reports", 1)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(calls["model"], (str(model_path), "detect"))
        self.assertEqual(calls["val"]["data"], "coco8.yaml")
        self.assertEqual(calls["val"]["device"], "cpu")
        self.assertTrue(calls["val"]["plots"])
        self.assertTrue(Path(result["report_json"]).is_file())
        self.assertTrue(Path(result["report_html"]).is_file())
        report_data = json.loads(Path(result["report_json"]).read_text(encoding="utf-8"))
        self.assertEqual(report_data["metrics"]["map50"], 0.85)
        self.assertEqual(
            report_data["metrics"]["validation_batch_latency"]["sample_count"], 2
        )
        self.assertIn(
            "p95_ms", report_data["metrics"]["validation_batch_latency"]
        )
        self.assertEqual(report_data["resources"]["rss_peak_mb"], 120.0)
        self.assertEqual(report_data["tracking_metrics"]["status"], "not_evaluated")

    def test_failed_model_run_is_reported_instead_of_appearing_successful(self):
        model_path = self.root / "model.pt"
        model_path.touch()
        config = ModelRunConfig(model=model_path)

        class BrokenYOLO:
            def __init__(self, path, task):
                raise RuntimeError("model load failed")

        with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=BrokenYOLO)}):
            with patch("model_benchmark.runner._resolve_device", return_value="cpu"):
                result = run_one(config, self.root / "reports", 1)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["message"], "model load failed")
        self.assertTrue(Path(result["report_json"]).is_file())

    def test_writes_comparison_summary(self):
        output = self.root / "reports"
        result = {
            "name": "first",
            "status": "completed",
            "configuration": {"model_format": "pt"},
            "metrics": {"precision": 0.9, "recall": 0.8, "map50": 0.85, "map50_95": 0.6},
            "resources": {"process_cpu_percent_avg": 30.0, "rss_peak_mb": 200.0},
        }

        write_summary_report(output, [result])

        self.assertTrue((output / "summary.html").is_file())
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["runs"][0]["name"], "first")
        self.assertIn("mAP50-95", (output / "summary.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
