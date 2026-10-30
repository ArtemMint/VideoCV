import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Union


ModelPath = Union[str, Path]


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
    dataset: str = "coco8.yaml"
    tracker: str = "bytetrack.yaml"
    imgsz: int = 640
    device: str = "auto"
    batch: int = 1
    conf: float = 0.001
    iou: float = 0.7

    def __post_init__(self) -> None:
        model_path = Path(self.model).expanduser()
        detect_model_format(model_path)
        object.__setattr__(self, "model", model_path)

        if not isinstance(self.name, str):
            raise ValueError("name must be a string")
        if not self.dataset.strip():
            raise ValueError("Dataset must be a non-empty name or path")
        if not self.tracker.strip():
            raise ValueError("Tracker must be a non-empty Ultralytics tracker config")
        if not self.device.strip():
            raise ValueError("Device must be a non-empty device name or 'auto'")
        if self.imgsz <= 0:
            raise ValueError("imgsz must be greater than zero")
        if self.batch <= 0:
            raise ValueError("batch must be greater than zero")
        if not 0.0 <= self.conf <= 1.0:
            raise ValueError("conf must be between 0 and 1")
        if not 0.0 <= self.iou <= 1.0:
            raise ValueError("iou must be between 0 and 1")

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
