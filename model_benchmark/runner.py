import math
import re
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ModelRunConfig
from .visdrone import ensure_visdrone_dataset


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return value or "model"


def _resolve_device(config: ModelRunConfig) -> str:
    if config.device.lower() != "auto":
        return config.device
    if config.model_format != "pt":
        return "cpu"

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to resolve device='auto' for .pt models"
        ) from exc
    return "0" if torch.cuda.is_available() else "cpu"


class _ResourceSampler:
    def __init__(self) -> None:
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError(
                "psutil is required for CPU and memory measurements"
            ) from exc

        self._process = psutil.Process()
        self._stop = threading.Event()
        self._rss_samples = []
        self._cpu_start = 0.0
        self._started_at = 0.0
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "_ResourceSampler":
        self._rss_start = self._process.memory_info().rss
        self._cpu_start = sum(self._process.cpu_times()[:2])
        self._started_at = time.perf_counter()
        self._sample()
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)
        self._thread.start()
        return self

    def _sample(self) -> None:
        self._rss_samples.append(self._process.memory_info().rss)

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(0.1):
            self._sample()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        self.elapsed_seconds = time.perf_counter() - self._started_at
        self.cpu_seconds = sum(self._process.cpu_times()[:2]) - self._cpu_start
        self.rss_peak = max(self._rss_samples)

    def as_dict(self) -> Dict[str, float]:
        peak_mb = self.rss_peak / (1024 * 1024)
        baseline_mb = self._rss_start / (1024 * 1024)
        return {
            "process_cpu_percent_avg": (
                self.cpu_seconds / self.elapsed_seconds * 100.0
                if self.elapsed_seconds
                else 0.0
            ),
            "process_cpu_seconds": self.cpu_seconds,
            "rss_baseline_mb": baseline_mb,
            "rss_peak_mb": peak_mb,
            "rss_peak_increase_mb": max(0.0, peak_mb - baseline_mb),
        }


def _float_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def extract_metrics(
    validation_results: Any, class_names: Optional[Any] = None
) -> Dict[str, Any]:
    """Extract stable detection metrics from Ultralytics validation results."""
    box_metrics = getattr(validation_results, "box", None)
    metric_names = {
        "precision": "mp",
        "recall": "mr",
        "map50": "map50",
        "map50_95": "map",
    }
    metrics: Dict[str, Any] = {}

    for output_name, source_name in metric_names.items():
        value = _float_or_none(getattr(box_metrics, source_name, None))
        if value is not None:
            metrics[output_name] = value

    speed = getattr(validation_results, "speed", None)
    if isinstance(speed, dict):
        metrics["latency_ms_per_image"] = {
            key: _float_or_none(value) for key, value in speed.items()
        }
        measured_stages = [
            value for value in metrics["latency_ms_per_image"].values()
            if value is not None
        ]
        if measured_stages:
            total_ms = sum(measured_stages)
            metrics["fps_from_validation_speed"] = (
                1000.0 / total_ms if total_ms > 0 else None
            )

    maps = getattr(box_metrics, "maps", None)
    names = class_names or getattr(validation_results, "names", None)
    if maps is not None and names is not None:
        per_class = {}
        for class_id, value in enumerate(maps):
            class_name = (
                names.get(class_id, str(class_id))
                if isinstance(names, dict)
                else names[class_id]
                if isinstance(names, (list, tuple)) and class_id < len(names)
                else str(class_id)
            )
            class_map = _float_or_none(value)
            if class_map is not None:
                per_class[class_name] = class_map
        metrics["per_class_map50_95"] = per_class

    return metrics


def _latency_summary(samples_seconds: List[float]) -> Dict[str, Any]:
    samples_ms = sorted(sample * 1000.0 for sample in samples_seconds)
    if not samples_ms:
        return {}
    p95_index = max(0, math.ceil(0.95 * len(samples_ms)) - 1)
    return {
        "p50_ms": statistics.median(samples_ms),
        "p95_ms": samples_ms[p95_index],
        "mean_ms": statistics.fmean(samples_ms),
        "sample_count": len(samples_ms),
    }


def run_one(
    config: ModelRunConfig, output_dir: Path, index: int
) -> Dict[str, Any]:
    """Evaluate one model and persist its result, including explicit failures."""
    run_name = config.name or "{}-{}".format(Path(config.model).stem, config.model_format)
    run_name = "{:02d}-{}".format(index, _safe_name(run_name))
    report_dir = output_dir / run_name
    report_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "name": config.name or run_name,
        "status": "failed",
        "configuration": config.as_dict(),
        "resolved_device": None,
        "metrics": {},
        "resources": {},
        "timings": {},
        "artifacts": {},
        "notes": [],
        "tracking_metrics": {
            "status": "not_evaluated",
            "reason": "The selected detection dataset has no ground-truth track IDs",
        },
    }
    if "coco8" in config.dataset.lower():
        result["notes"].append(
            "COCO8 is a tiny smoke-test dataset; metrics are not statistically "
            "reliable for production model selection."
        )

    try:
        dataset_setup_started = time.perf_counter()
        dataset_setup = ensure_visdrone_dataset(config.dataset)
        result["timings"]["dataset_setup_seconds"] = (
            time.perf_counter() - dataset_setup_started
        )
        if dataset_setup.get("downloaded"):
            result["notes"].append(
                "VisDrone validation data was missing and was downloaded and prepared "
                "automatically before evaluation."
            )

        from ultralytics import YOLO

        device = _resolve_device(config)
        result["resolved_device"] = device
        with _ResourceSampler() as sampler:
            load_started = time.perf_counter()
            model = YOLO(str(config.model), task="detect")
            result["timings"]["model_load_seconds"] = (
                time.perf_counter() - load_started
            )

            batch_started = []
            batch_latencies = []

            def on_val_batch_start(_validator: Any) -> None:
                batch_started.append(time.perf_counter())

            def on_val_batch_end(_validator: Any) -> None:
                if batch_started:
                    batch_latencies.append(
                        time.perf_counter() - batch_started.pop()
                    )

            model.add_callback("on_val_batch_start", on_val_batch_start)
            model.add_callback("on_val_batch_end", on_val_batch_end)
            validation_started = time.perf_counter()
            validation_results = model.val(
                data=config.dataset,
                imgsz=config.imgsz,
                batch=config.batch,
                device=device,
                conf=config.conf,
                iou=config.iou,
                plots=True,
                verbose=False,
                project=str(output_dir / "ultralytics"),
                name=run_name,
                exist_ok=True,
            )
            result["timings"]["validation_wall_seconds"] = (
                time.perf_counter() - validation_started
            )
            result["metrics"] = extract_metrics(
                validation_results, getattr(model, "names", None)
            )
            result["metrics"]["validation_batch_latency"] = _latency_summary(
                batch_latencies
            )

        result["resources"] = sampler.as_dict()
        save_dir = getattr(validation_results, "save_dir", None)
        if save_dir is not None:
            result["artifacts"]["ultralytics_validation_dir"] = str(save_dir)
        result["status"] = "completed"
    except Exception as exc:
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    from .report import write_model_report

    write_model_report(report_dir, result)
    result["report_json"] = str(report_dir / "report.json")
    result["report_html"] = str(report_dir / "report.html")
    return result
