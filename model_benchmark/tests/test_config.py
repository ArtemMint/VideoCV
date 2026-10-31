import json
import tempfile
import unittest
from pathlib import Path

from model_benchmark import ModelRunConfig
from model_benchmark.config import detect_model_format


class ModelRunConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def make_model(self, filename):
        model = self.root / filename
        model.touch()
        return model

    def test_defaults_use_local_visdrone_and_bytetrack(self):
        config = ModelRunConfig(model=self.make_model("model.pt"))

        self.assertEqual(
            config.dataset,
            str(Path(__file__).resolve().parents[2] / "datasets" / "visdrone-val.yaml"),
        )
        self.assertEqual(config.tracker, "bytetrack.yaml")
        self.assertEqual(config.imgsz, 640)
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.batch, 1)
        self.assertEqual(config.conf, 0.001)

    def test_custom_dataset_path_is_resolved_and_must_exist(self):
        model = self.make_model("model.pt")
        dataset = self.root / "custom.yaml"
        dataset.write_text("path: .\ntrain: train\nval: val\nnames: [object]\n", encoding="utf-8")

        config = ModelRunConfig(model=model, dataset=str(dataset))

        self.assertEqual(config.dataset, str(dataset.resolve()))
        with self.assertRaisesRegex(ValueError, "Dataset path does not exist"):
            ModelRunConfig(model=model, dataset=str(self.root / "missing.yaml"))

    def test_detects_pt_and_onnx_formats(self):
        self.assertEqual(detect_model_format(self.make_model("model.pt")), "pt")
        self.assertEqual(detect_model_format(self.make_model("model.onnx")), "onnx")

    def test_detects_openvino_xml_file_and_directory(self):
        xml_file = self.make_model("model.xml")
        openvino_dir = self.root / "openvino"
        openvino_dir.mkdir()
        (openvino_dir / "model.xml").touch()

        self.assertEqual(detect_model_format(xml_file), "openvino")
        self.assertEqual(detect_model_format(openvino_dir), "openvino")

    def test_rejects_missing_unsupported_and_ambiguous_model_paths(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            ModelRunConfig(model=self.root / "missing.pt")

        with self.assertRaisesRegex(ValueError, "Unsupported model format"):
            ModelRunConfig(model=self.make_model("model.engine"))

        openvino_dir = self.root / "ambiguous"
        openvino_dir.mkdir()
        (openvino_dir / "first.xml").touch()
        (openvino_dir / "second.xml").touch()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            ModelRunConfig(model=openvino_dir)

    def test_rejects_invalid_evaluation_parameters(self):
        model = self.make_model("model.pt")
        invalid_configs = (
            lambda: ModelRunConfig(model=model, imgsz=0),
            lambda: ModelRunConfig(model=model, batch=0),
            lambda: ModelRunConfig(model=model, repeats=0),
            lambda: ModelRunConfig(model=model, repeats=1.5),
            lambda: ModelRunConfig(model=model, warmup_iterations=-1),
            lambda: ModelRunConfig(model=model, inference_iterations=0),
            lambda: ModelRunConfig(model=model, conf=-0.1),
            lambda: ModelRunConfig(model=model, iou=1.1),
            lambda: ModelRunConfig(model=model, min_map50_95=1.1),
            lambda: ModelRunConfig(model=model, max_latency_ms=0),
            lambda: ModelRunConfig(model=model, max_memory_mb="100"),
            lambda: ModelRunConfig(model=model, max_gpu_memory_mb=-1),
            lambda: ModelRunConfig(model=model, dataset=""),
            lambda: ModelRunConfig(model=model, tracker=""),
            lambda: ModelRunConfig(model=model, device=""),
        )
        for invalid_config in invalid_configs:
            with self.subTest(invalid_config=invalid_config):
                with self.assertRaises(ValueError):
                    invalid_config()

    def test_serializes_measurement_and_acceptance_parameters(self):
        config = ModelRunConfig(
            model=self.make_model("model.pt"),
            repeats=4,
            warmup_iterations=7,
            inference_iterations=25,
            half=True,
            export_parameters={"dynamic": True, "opset": 17},
            min_map50_95=0.4,
            max_latency_ms=40,
            max_memory_mb=1200,
            max_gpu_memory_mb=900,
        )

        serialized = config.as_dict()

        self.assertEqual(serialized["repeats"], 4)
        self.assertEqual(serialized["warmup_iterations"], 7)
        self.assertEqual(serialized["inference_iterations"], 25)
        self.assertTrue(serialized["half"])
        self.assertEqual(serialized["export_parameters"]["opset"], 17)
        self.assertEqual(serialized["max_latency_ms"], 40)
        self.assertEqual(serialized["max_gpu_memory_mb"], 900)

    def test_loads_json_and_exports_format_and_host_environment(self):
        model = self.make_model("model.onnx")
        config_file = self.root / "run.json"
        config_file.write_text(
            json.dumps({"model": str(model), "dataset": "coco8.yaml"}),
            encoding="utf-8",
        )

        config = ModelRunConfig.from_json(config_file)
        serialized = config.as_dict()

        self.assertEqual(config.model_format, "onnx")
        self.assertEqual(serialized["model_format"], "onnx")
        self.assertEqual(serialized["dataset"], "coco8.yaml")
        self.assertIn("os", serialized["environment"])
        self.assertIn("architecture", serialized["environment"])

    def test_json_rejects_non_object_and_unknown_fields(self):
        config_file = self.root / "run.json"
        config_file.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must contain an object"):
            ModelRunConfig.from_json(config_file)

        config_file.write_text(
            json.dumps({"model": str(self.make_model("model.pt")), "unknown": True}),
            encoding="utf-8",
        )
        with self.assertRaises(TypeError):
            ModelRunConfig.from_json(config_file)

    def test_loads_multiple_run_configurations_with_shared_defaults(self):
        first_model = self.make_model("first.pt")
        second_model = self.make_model("second.onnx")
        config_file = self.root / "multi.json"
        config_file.write_text(
            json.dumps(
                {
                    "runs": [
                        {"model": str(first_model), "name": "first"},
                        {"model": str(second_model), "imgsz": 320},
                    ]
                }
            ),
            encoding="utf-8",
        )

        configs = ModelRunConfig.many_from_json(config_file)

        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[0].name, "first")
        self.assertEqual(configs[0].dataset, str(
            Path(__file__).resolve().parents[2] / "datasets" / "visdrone-val.yaml"
        ))
        self.assertEqual(configs[0].tracker, "bytetrack.yaml")
        self.assertEqual(configs[1].imgsz, 320)
        self.assertEqual(configs[1].model_format, "onnx")

    def test_default_model_comparison_points_to_local_weights(self):
        project_root = Path(__file__).resolve().parents[2]
        configs = ModelRunConfig.many_from_json(
            project_root / "model_benchmark" / "config.multi.example.json"
        )

        self.assertEqual(len(configs), 6)
        self.assertEqual(
            [config.model_format for config in configs],
            ["pt", "onnx", "openvino"] * 2,
        )
        self.assertTrue(all(config.device == "cpu" for config in configs))
        self.assertTrue(
            all(config.model.is_file() or config.model.is_dir() for config in configs)
        )


if __name__ == "__main__":
    unittest.main()
