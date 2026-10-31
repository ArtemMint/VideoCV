import json
import os
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


ModelPath = Union[str, Path]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def detect_model_format(model_path: ModelPath) -> str:
    """Return the supported YOLO model format inferred from its path."""
    path = Path(model_path).expanduser()

    if path.is_dir():
        xml_models = list(path.glob("*.xml"))
        if len(xml_models) != 1:
            raise ValueError(
                "An OpenVINO model directory must contain exactly one .xml file"
            )
        return "openvino"

    if not path.is_file():
        raise ValueError("Model path does not exist or is not a regular file: {}".format(path))

    suffix = path.suffix.lower()
    if suffix == ".pt":
        return "pt"
    if suffix == ".onnx":
        return "onnx"
    if suffix == ".xml":
        return "openvino"

    raise ValueError(
        "Unsupported model format '{}'; use .pt, .onnx, or OpenVINO .xml/directory".format(
            suffix or "<no extension>"
        )
    )


@dataclass(frozen=True)
class ModelRunConfig:
    """Parameters shared by comparable YOLO model evaluation runs."""

    model: ModelPath
    name: str = ""
    dataset: str = "datasets/visdrone-val.yaml"
    tracking_dataset: Optional[str] = None
    tracker: str = "bytetrack.yaml"
    imgsz: int = 640
    device: str = "auto"
    batch: int = 1
    conf: float = 0.001
    iou: float = 0.7
    repeats: int = 3
    warmup_iterations: int = 5
    inference_iterations: int = 20
    quantize: Optional[int] = None
    export_parameters: Dict[str, Any] = field(default_factory=dict)
    min_map50_95: Optional[float] = None
    max_latency_ms: Optional[float] = None
    max_memory_mb: Optional[float] = None
    max_gpu_memory_mb: Optional[float] = None

    def __post_init__(self) -> None:
        model_path = Path(self.model).expanduser()
        detect_model_format(model_path)
        object.__setattr__(self, "model", model_path)

        if not isinstance(self.name, str):
            raise ValueError("name must be a string")
        if not isinstance(self.dataset, str) or not self.dataset.strip():
            raise ValueError("Dataset must be a non-empty name or path")
        dataset_path = Path(self.dataset).expanduser()
        if dataset_path.is_absolute() or "/" in self.dataset or "\\" in self.dataset:
            if not dataset_path.is_absolute():
                dataset_path = PROJECT_ROOT / dataset_path
            dataset_path = dataset_path.resolve()
            if not dataset_path.exists():
                raise ValueError("Dataset path does not exist: {}".format(dataset_path))
            object.__setattr__(self, "dataset", str(dataset_path))
        if self.tracking_dataset is not None:
            if (
                not isinstance(self.tracking_dataset, str)
                or not self.tracking_dataset.strip()
            ):
                raise ValueError("tracking_dataset must be a non-empty directory path")
            tracking_path = Path(self.tracking_dataset).expanduser()
            if not tracking_path.is_absolute():
                tracking_path = PROJECT_ROOT / tracking_path
            tracking_path = tracking_path.resolve()
            if not tracking_path.is_dir():
                raise ValueError(
                    "Tracking dataset directory does not exist: {}".format(tracking_path)
                )
            object.__setattr__(self, "tracking_dataset", str(tracking_path))
        if not self.tracker.strip():
            raise ValueError("Tracker must be a non-empty Ultralytics tracker config")
        if not self.device.strip():
            raise ValueError("Device must be a non-empty device name or 'auto'")
        if self.imgsz <= 0:
            raise ValueError("imgsz must be greater than zero")
        if self.batch <= 0:
            raise ValueError("batch must be greater than zero")
        for field_name in ("repeats", "warmup_iterations", "inference_iterations"):
            field_value = getattr(self, field_name)
            if not isinstance(field_value, int) or isinstance(field_value, bool):
                raise ValueError("{} must be an integer".format(field_name))
        if self.repeats <= 0:
            raise ValueError("repeats must be greater than zero")
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations cannot be negative")
        if self.inference_iterations <= 0:
            raise ValueError("inference_iterations must be greater than zero")
        if self.quantize not in (None, 16, 32):
            raise ValueError("quantize must be 16 (FP16), 32 (FP32), or null")
        if not isinstance(self.export_parameters, dict):
            raise ValueError("export_parameters must be an object")
        if not 0.0 <= self.conf <= 1.0:
            raise ValueError("conf must be between 0 and 1")
        if not 0.0 <= self.iou <= 1.0:
            raise ValueError("iou must be between 0 and 1")
        if self.min_map50_95 is not None:
            if not isinstance(self.min_map50_95, (int, float)) or isinstance(
                self.min_map50_95, bool
            ):
                raise ValueError("min_map50_95 must be a number")
            if not 0.0 <= self.min_map50_95 <= 1.0:
                raise ValueError("min_map50_95 must be between 0 and 1")
        for field_name in ("max_latency_ms", "max_memory_mb", "max_gpu_memory_mb"):
            field_value = getattr(self, field_name)
            if field_value is not None:
                if not isinstance(field_value, (int, float)) or isinstance(
                    field_value, bool
                ):
                    raise ValueError("{} must be a number".format(field_name))
                if field_value <= 0:
                    raise ValueError("{} must be greater than zero".format(field_name))

    @property
    def model_format(self) -> str:
        return detect_model_format(self.model)

    @classmethod
    def from_json(cls, config_path: ModelPath) -> "ModelRunConfig":
        with Path(config_path).open("r", encoding="utf-8") as config_file:
            values = json.load(config_file)
        if not isinstance(values, dict):
            raise ValueError("Run configuration JSON must contain an object")
        return cls(**values)

    @classmethod
    def many_from_json(cls, config_path: ModelPath) -> List["ModelRunConfig"]:
        with Path(config_path).open("r", encoding="utf-8") as config_file:
            values = json.load(config_file)
        if isinstance(values, dict) and "runs" in values:
            runs = values["runs"]
            if not isinstance(runs, list) or not runs:
                raise ValueError("'runs' must be a non-empty list")
            return [cls(**run) for run in runs]
        if isinstance(values, dict):
            return [cls(**values)]
        raise ValueError("Run configuration JSON must contain an object")

    def as_dict(self) -> Dict[str, Any]:
        values = asdict(self)
        values["model"] = str(self.model)
        values["model_format"] = self.model_format
        values["environment"] = {
            "os": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        }
        return values
