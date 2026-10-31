import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from model_benchmark import ModelRunConfig
from model_benchmark.report import write_summary_report
from model_benchmark.runner import _assess_requirements, extract_metrics, run_one


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

    def test_run_measures_load_warmup_prediction_and_repeated_validation(self):
        model_path = self.root / "model.pt"
        model_path.touch()
        tracking_dataset = self.root / "mot"
        tracking_dataset.mkdir()
        config = ModelRunConfig(
            model=model_path,
            name="unit-test",
            tracking_dataset=str(tracking_dataset),
            repeats=2,
            warmup_iterations=1,
            inference_iterations=2,
            min_map50_95=0.5,
            max_latency_ms=1000,
            max_memory_mb=1000,
        )
        validation_result = SimpleNamespace(
            box=SimpleNamespace(mp=0.9, mr=0.8, map50=0.85, map=0.65, maps=[0.65]),
            speed={"inference": 10.0},
            save_dir=self.root / "ultralytics-results",
        )
        calls = {}

        class FakeYOLO:
            names = {0: "object"}
            overrides = {"quantize": None, "imgsz": 640, "dynamic": False}

            def __init__(self, path, task):
                calls["model"] = (path, task)
                self.callbacks = {}

            def predict(self, **kwargs):
                calls.setdefault("predict", []).append(kwargs)
                return [
                    SimpleNamespace(
                        speed={
                            "preprocess": 1.0,
                            "inference": 8.0,
                            "postprocess": 2.0,
                        }
                    )
                ]

            def add_callback(self, event, callback):
                self.callbacks[event] = callback

            def val(self, **kwargs):
                calls["val"] = kwargs
                for _ in range(2):
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
                    with patch(
                        "model_benchmark.runner.ensure_visdrone_dataset",
                        return_value={"downloaded": True, "images": 548},
                    ) as ensure_dataset:
                        with patch(
                            "model_benchmark.runner._representative_frame",
                            return_value=(object(), "test-frame.jpg"),
                        ):
                            with patch(
                                "model_benchmark.runner.evaluate_mot_sequences",
                                return_value={"status": "evaluated", "mota": 0.9},
                            ) as evaluate_tracking:
                                result = run_one(config, self.root / "reports", 1)

        self.assertEqual(result["status"], "completed")
        ensure_dataset.assert_called_once_with(config.dataset)
        self.assertEqual(calls["model"], (str(model_path), "detect"))
        self.assertEqual(calls["val"]["data"], config.dataset)
        self.assertEqual(calls["val"]["device"], "cpu")
        self.assertIn("quantize", calls["val"])
        self.assertNotIn("half", calls["val"])
        self.assertTrue(all("quantize" in call for call in calls["predict"]))
        self.assertTrue(all("half" not in call for call in calls["predict"]))
        self.assertTrue(calls["val"]["plots"])
        self.assertEqual(len(calls["predict"]), 5)
        self.assertTrue(Path(result["report_json"]).is_file())
        self.assertTrue(Path(result["report_html"]).is_file())
        report_data = json.loads(Path(result["report_json"]).read_text(encoding="utf-8"))
        self.assertEqual(report_data["metrics"]["map50"], 0.85)
        self.assertEqual(report_data["metrics"]["validation_batch_latency"]["sample_count"], 4)
        self.assertIn(
            "p95_ms", report_data["metrics"]["validation_batch_latency"]
        )
        self.assertEqual(
            report_data["timings"]["warmup"]["per_frame_latency"]["sample_count"], 1
        )
        self.assertEqual(
            report_data["metrics"]["inference"]["end_to_end_frame"]["sample_count"], 4
        )
        self.assertEqual(
            report_data["metrics"]["inference"]["stage_latency_ms"]["inference"]["p50_ms"],
            8.0,
        )
        self.assertIn("p50_fps", report_data["metrics"]["inference"]["fps"])
        self.assertEqual(report_data["resources"]["rss_peak_mb"], 120.0)
        self.assertEqual(report_data["tracking_metrics"]["status"], "evaluated")
        self.assertEqual(report_data["tracking_metrics"]["mota"], 0.9)
        self.assertEqual(
            evaluate_tracking.call_args.args[1], tracking_dataset.resolve()
        )
        self.assertEqual(
            evaluate_tracking.call_args.kwargs["tracker"], config.tracker
        )
        self.assertIn("dataset_setup_seconds", report_data["timings"])
        self.assertTrue(report_data["decision"]["meets_requirements"])
        self.assertEqual(
            report_data["decision"]["checks"][1]["observed"],
            report_data["metrics"]["inference"]["end_to_end_frame"]["p95_ms"],
        )
        self.assertTrue(any("downloaded and prepared" in note for note in report_data["notes"]))

    def test_failed_model_run_is_reported_instead_of_appearing_successful(self):
        model_path = self.root / "model.pt"
        model_path.touch()
        config = ModelRunConfig(model=model_path, dataset="coco8.yaml")

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
        result = self._summary_result(
            "first", 0.6, 20.0, 100.0, "same-conditions"
        )
        dominated = self._summary_result(
            "dominated", 0.5, 25.0, 120.0, "same-conditions"
        )
        different_device = self._summary_result(
            "different-device", 0.1, 200.0, 500.0, "other-conditions"
        )

        write_summary_report(output, [result, dominated, different_device])

        self.assertTrue((output / "summary.html").is_file())
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["runs"][0]["name"], "first")
        self.assertIn("mAP50-95", (output / "summary.html").read_text(encoding="utf-8"))
        self.assertTrue(summary["runs"][0]["comparison"]["pareto_optimal"])
        self.assertEqual(
            summary["runs"][1]["comparison"]["dominated_by"], ["first"]
        )
        self.assertTrue(summary["runs"][2]["comparison"]["pareto_optimal"])

    def test_decision_uses_p95_and_available_gpu_memory(self):
        model_path = self.root / "quality-gated.pt"
        model_path.touch()
        config = ModelRunConfig(
            model=model_path,
            max_latency_ms=50,
            max_gpu_memory_mb=100,
        )
        metrics = {
            "map50_95": 0.6,
            "inference": {"end_to_end_frame": {"p50_ms": 20, "p95_ms": 55}},
        }
        resources = {
            "rss_peak_mb": 200,
            "gpu_process_memory": {"available": False},
            "gpu": {"available": True, "memory_used_peak_mb": 80},
        }

        result = _assess_requirements(config, metrics, resources)

        self.assertFalse(result["meets_requirements"])
        self.assertEqual(result["checks"][0]["observed"], 55)
        self.assertTrue(result["checks"][1]["passed"])

    def _summary_result(self, name, map50_95, latency, memory, group):
        return {
            "name": name,
            "status": "completed",
            "comparison_group": group,
            "configuration": {"model_format": "pt"},
            "metrics": {
                "precision": 0.9,
                "recall": 0.8,
                "map50": 0.85,
                "map50_95": map50_95,
                "inference": {
                    "end_to_end_frame": {"p50_ms": latency, "p95_ms": latency + 5},
                    "fps": {"p50_fps": 1000 / latency},
                },
            },
            "resources": {
                "process_cpu_percent_avg": 30.0,
                "rss_peak_mb": memory,
                "gpu_process_memory": {"available": False},
            },
            "decision": {"meets_requirements": None, "checks": []},
        }


if __name__ == "__main__":
    unittest.main()
