import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import ModelRunConfig
from .mot import evaluate_mot_sequences
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


def _cuda_device_index(device: str) -> Optional[int]:
    if device.lower() == "cuda":
        return 0
    if device.lower().startswith("cuda:"):
        suffix = device.split(":", 1)[1]
        return int(suffix) if suffix.isdigit() else 0
    if re.fullmatch(r"\d+(,\d+)*", device):
        return int(device.split(",", 1)[0])
    return None


class _ResourceSampler:
    def __init__(self) -> None:
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError(
                "psutil is required for CPU and memory measurements"
            ) from exc

        self._psutil = psutil
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._rss_samples = []
        self._tree_cpu_seconds = 0.0
        self._last_cpu_by_pid: Dict[int, float] = {}
        self._system_cpu_samples = []
        self._temperature_samples: List[Tuple[str, float]] = []
        self._temperature_error: Optional[str] = None
        self._started_at = 0.0
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "_ResourceSampler":
        self._started_at = time.perf_counter()
        try:
            baseline_processes = [self._process] + self._process.children(recursive=True)
        except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
            baseline_processes = [self._process]
        for process in baseline_processes:
            try:
                self._last_cpu_by_pid[process.pid] = sum(process.cpu_times()[:2])
            except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                continue
        self._sample()
        self._rss_start = self._rss_samples[-1]
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)
        self._thread.start()
        return self

    def _sample(self) -> None:
        try:
            processes = [self._process] + self._process.children(recursive=True)
        except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
            processes = [self._process]

        tree_rss = 0
        for process in processes:
            try:
                cpu_seconds = sum(process.cpu_times()[:2])
                previous = self._last_cpu_by_pid.get(
                    process.pid,
                    0.0 if process.pid != self._process.pid else cpu_seconds,
                )
                self._tree_cpu_seconds += max(0.0, cpu_seconds - previous)
                self._last_cpu_by_pid[process.pid] = cpu_seconds
                tree_rss += process.memory_info().rss
            except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                continue
        self._rss_samples.append(tree_rss)
        self._system_cpu_samples.append(self._psutil.cpu_percent(interval=None))
        try:
            sensors = self._psutil.sensors_temperatures()
            self._temperature_samples.extend(
                (label, item.current)
                for label, entries in sensors.items()
                for item in entries
                if item.current is not None
            )
        except (AttributeError, NotImplementedError, OSError) as exc:
            self._temperature_error = "{}: {}".format(type(exc).__name__, exc)

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(0.1):
            self._sample()

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        self.elapsed_seconds = time.perf_counter() - self._started_at
        self.cpu_seconds = self._tree_cpu_seconds
        self.rss_peak = max(self._rss_samples)

    def as_dict(self) -> Dict[str, Any]:
        peak_mb = self.rss_peak / (1024 * 1024)
        baseline_mb = self._rss_start / (1024 * 1024)
        metrics: Dict[str, Any] = {
            "process_cpu_percent_avg": (
                self.cpu_seconds / self.elapsed_seconds * 100.0
                if self.elapsed_seconds
                else 0.0
            ),
            "process_cpu_seconds": self.cpu_seconds,
            "system_cpu_percent_avg": (
                statistics.fmean(self._system_cpu_samples)
                if self._system_cpu_samples
                else 0.0
            ),
            "rss_baseline_mb": baseline_mb,
            "rss_peak_mb": peak_mb,
            "rss_peak_increase_mb": max(0.0, peak_mb - baseline_mb),
            "rss_scope": "benchmark process and observed child processes",
        }
        hottest_sensor = (
            max(self._temperature_samples, key=lambda reading: reading[1])
            if self._temperature_samples
            else None
        )
        metrics["system_temperature_peak_c"] = (
            hottest_sensor[1] if hottest_sensor else None
        )
        metrics["system_temperature_sensor"] = hottest_sensor[0] if hottest_sensor else None
        metrics["system_temperature_available"] = hottest_sensor is not None
        if self._temperature_error:
            metrics["cpu_temperature_unavailable_reason"] = self._temperature_error
        return metrics


class _GpuSampler:
    """Collect optional NVIDIA GPU telemetry through NVML when available."""

    def __init__(self, device: str, runtime: Dict[str, Any]) -> None:
        self._device = device
        self._runtime = runtime
        self._nvml = None
        self._handle = None
        self._samples: List[Dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.unavailable_reason: Optional[str] = None
        self._energy_joules = 0.0
        self._last_power_watts: Optional[float] = None
        self._last_sample_at: Optional[float] = None

    def __enter__(self) -> "_GpuSampler":
        if self._device.lower() == "cpu":
            self.unavailable_reason = "CPU device selected"
            return self
        cuda_index = _cuda_device_index(self._device)
        accelerator = self._runtime.get("accelerator", {})
        device_model = accelerator.get("model", "")
        if cuda_index is None and "NVIDIA" not in device_model.upper():
            self.unavailable_reason = "NVML telemetry is available only for NVIDIA devices"
            return self
        try:
            import pynvml

            self._nvml = pynvml
            pynvml.nvmlInit()
            match = re.search(r"\.(\d+)$", self._device)
            index = (
                cuda_index
                if cuda_index is not None
                else int(match.group(1))
                if match
                else 0
            )
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self._sample()
            self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)
            self._thread.start()
        except Exception as exc:
            self.unavailable_reason = "{}: {}".format(type(exc).__name__, exc)
            if self._nvml is not None:
                try:
                    self._nvml.nvmlShutdown()
                except Exception as shutdown_exc:
                    self.unavailable_reason += "; NVML shutdown failed: {}".format(
                        shutdown_exc
                    )
                self._nvml = None
        return self

    def _sample(self) -> None:
        if self._nvml is None or self._handle is None:
            return
        try:
            now = time.perf_counter()
            power = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            if self._last_sample_at is not None and self._last_power_watts is not None:
                self._energy_joules += self._last_power_watts * (
                    now - self._last_sample_at
                )
            utilization = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
            memory = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            temperature = self._nvml.nvmlDeviceGetTemperature(
                self._handle, self._nvml.NVML_TEMPERATURE_GPU
            )
            self._samples.append(
                {
                    "utilization_percent": float(utilization.gpu),
                    "memory_used_mb": float(memory.used) / (1024 * 1024),
                    "memory_total_mb": float(memory.total) / (1024 * 1024),
                    "temperature_c": float(temperature),
                    "power_watts": float(power),
                }
            )
            self._last_sample_at = now
            self._last_power_watts = power
        except Exception as exc:
            self.unavailable_reason = "{}: {}".format(type(exc).__name__, exc)

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(0.2):
            self._sample()

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception as exc:
                self.unavailable_reason = "{}: {}".format(type(exc).__name__, exc)

    def as_dict(self) -> Dict[str, Any]:
        if not self._samples:
            return {"available": False, "reason": self.unavailable_reason}
        metrics = {
            "available": True,
            "utilization_percent_avg": statistics.fmean(
                item["utilization_percent"] for item in self._samples
            ),
            "memory_used_peak_mb": max(item["memory_used_mb"] for item in self._samples),
            "memory_total_mb": self._samples[-1]["memory_total_mb"],
            "temperature_peak_c": max(item["temperature_c"] for item in self._samples),
            "power_watts_avg": statistics.fmean(
                item["power_watts"] for item in self._samples
            ),
            "estimated_energy_joules": self._energy_joules,
            "sample_count": len(self._samples),
        }
        if self.unavailable_reason:
            metrics["sampling_warning"] = self.unavailable_reason
        return metrics


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


def _distribution_summary_ms(samples_ms: List[float]) -> Dict[str, Any]:
    samples = sorted(samples_ms)
    if not samples:
        return {}
    p95_index = max(0, math.ceil(0.95 * len(samples)) - 1)
    return {
        "p50_ms": statistics.median(samples),
        "p95_ms": samples[p95_index],
        "mean_ms": statistics.fmean(samples),
        "stddev_ms": statistics.pstdev(samples),
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "sample_count": len(samples),
    }


def _latency_summary(samples_seconds: List[float]) -> Dict[str, Any]:
    return _distribution_summary_ms([sample * 1000.0 for sample in samples_seconds])


def _fps_summary(latencies_ms: List[float]) -> Dict[str, Any]:
    fps = [1000.0 / latency for latency in latencies_ms if latency > 0.0]
    if not fps:
        return {}
    summary = _distribution_summary_ms(fps)
    return {
        key.replace("_ms", "_fps"): value for key, value in summary.items()
    }


def _repeat_value_summary(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {}
    return {
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "stddev": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
        "sample_count": len(values),
    }


def _runtime_profile(device: str) -> Dict[str, Any]:
    cpu_model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith(("model name", "hardware")) and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break

    versions = {}
    for package in ("ultralytics", "torch", "onnxruntime", "openvino", "psutil"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    device_model = cpu_model or platform.machine()
    accelerator: Dict[str, Any] = {"type": "cpu"}
    if device.lower() != "cpu":
        cuda_index = _cuda_device_index(device)
        if cuda_index is not None:
            try:
                import torch

                if torch.cuda.is_available():
                    device_model = torch.cuda.get_device_name(cuda_index)
                    accelerator = {
                        "type": "cuda",
                        "index": cuda_index,
                        "model": device_model,
                        "cuda_runtime_version": torch.version.cuda,
                    }
                else:
                    accelerator = {
                        "type": "requested_accelerator",
                        "device": device,
                    }
            except ImportError:
                accelerator = {"type": "requested_accelerator", "device": device}
        else:
            accelerator = {"type": "requested_accelerator", "device": device}
        if accelerator["type"] == "requested_accelerator" and device.upper().startswith(
            ("GPU", "NPU")
        ):
            try:
                import openvino

                core = openvino.Core()
                device_name = device.upper()
                device_model = core.get_property(device_name, "FULL_DEVICE_NAME")
                accelerator = {
                    "type": "openvino",
                    "device": device_name,
                    "model": device_model,
                }
            except (ImportError, RuntimeError, ValueError) as exc:
                accelerator["resolution_error"] = "{}: {}".format(
                    type(exc).__name__, exc
                )

    return {
        "host": platform.node(),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "cpu_model": cpu_model or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "device": device,
        "device_model": device_model,
        "accelerator": accelerator,
        "runtime_versions": versions,
        "python_version": platform.python_version(),
    }


def _comparison_group(config: ModelRunConfig, runtime: Dict[str, Any]) -> str:
    signature = {
        "host": runtime["host"],
        "device": runtime["device"],
        "device_model": runtime["device_model"],
        "dataset": config.dataset,
        "tracking_dataset": config.tracking_dataset,
        "tracker": config.tracker if config.tracking_dataset else None,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "conf": config.conf,
        "iou": config.iou,
        "quantize": config.quantize,
        "export_parameters": config.export_parameters,
        "model_export_parameters": runtime.get("model_export_parameters", {}),
    }
    digest = hashlib.sha256(
        json.dumps(signature, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:10]
    return "conditions-{}".format(digest)


def _assess_requirements(
    config: ModelRunConfig, metrics: Dict[str, Any], resources: Dict[str, Any]
) -> Dict[str, Any]:
    requirements = {
        "min_map50_95": config.min_map50_95,
        "max_latency_ms": config.max_latency_ms,
        "max_memory_mb": config.max_memory_mb,
        "max_gpu_memory_mb": config.max_gpu_memory_mb,
    }
    checks = []
    map_value = metrics.get("map50_95")
    latency = metrics.get("inference", {}).get("end_to_end_frame", {}).get("p95_ms")
    memory = resources.get("rss_peak_mb")
    gpu_process_memory = resources.get("gpu_process_memory", {})
    gpu_device_memory = resources.get("gpu", {})
    gpu_memory = (
        gpu_process_memory.get("peak_allocated_mb")
        if gpu_process_memory.get("available")
        else gpu_device_memory.get("memory_used_peak_mb")
        if gpu_device_memory.get("available")
        else None
    )
    observed = {
        "min_map50_95": map_value,
        "max_latency_ms": latency,
        "max_memory_mb": memory,
        "max_gpu_memory_mb": gpu_memory,
    }
    for key, limit in requirements.items():
        if limit is None:
            continue
        value = observed[key]
        passed = value is not None and (
            value >= limit if key == "min_map50_95" else value <= limit
        )
        checks.append(
            {
                "criterion": key,
                "limit": limit,
                "observed": value,
                "passed": passed,
            }
        )
    return {
        "meets_requirements": all(item["passed"] for item in checks) if checks else None,
        "checks": checks,
    }


def _aggregate_validation_metrics(repeats: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    scalar_keys = {
        key
        for run in repeats
        for key, value in run.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    repeat_statistics = {}
    for key in sorted(scalar_keys):
        samples = [float(run[key]) for run in repeats if key in run]
        metrics[key] = statistics.median(samples)
        repeat_statistics[key] = _repeat_value_summary(samples)

    class_names = {
        name
        for run in repeats
        for name in run.get("per_class_map50_95", {})
    }
    if class_names:
        metrics["per_class_map50_95"] = {
            name: statistics.median(
                run["per_class_map50_95"][name]
                for run in repeats
                if name in run.get("per_class_map50_95", {})
            )
            for name in sorted(class_names)
        }

    stage_names = {
        name
        for run in repeats
        for name, value in run.get("latency_ms_per_image", {}).items()
        if value is not None
    }
    if stage_names:
        metrics["validation_stage_latency_ms"] = {
            name: _distribution_summary_ms(
                [
                    float(run["latency_ms_per_image"][name])
                    for run in repeats
                    if run.get("latency_ms_per_image", {}).get(name) is not None
                ]
            )
            for name in sorted(stage_names)
        }
        metrics["latency_ms_per_image"] = {
            name: summary["p50_ms"]
            for name, summary in metrics["validation_stage_latency_ms"].items()
        }
    metrics["validation_repeat_statistics"] = repeat_statistics
    return metrics


def _synchronize_device(device: str) -> None:
    index = _cuda_device_index(device)
    if index is None:
        return
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize(index)


def _predict_once(
    model: Any, frame: Any, config: ModelRunConfig, device: str
) -> Tuple[float, Dict[str, float]]:
    _synchronize_device(device)
    started = time.perf_counter()
    predictions = model.predict(
        source=frame,
        imgsz=config.imgsz,
        device=device,
        conf=config.conf,
        iou=config.iou,
        quantize=config.quantize,
        verbose=False,
        save=False,
    )
    _synchronize_device(device)
    elapsed = time.perf_counter() - started
    speed = getattr(predictions[0], "speed", {}) if predictions else {}
    stages = {}
    if isinstance(speed, dict):
        for stage in ("preprocess", "inference", "postprocess"):
            value = _float_or_none(speed.get(stage))
            if value is not None:
                stages[stage] = value
    return elapsed, stages


def _representative_frame(dataset: str) -> Tuple[Any, str]:
    from ultralytics.data.utils import check_det_dataset
    import cv2

    builtin_coco8 = dataset.strip().lower() == "coco8.yaml"
    dataset_info = check_det_dataset(dataset, autodownload=builtin_coco8)
    validation_sources = dataset_info["val"]
    if not isinstance(validation_sources, (list, tuple)):
        validation_sources = [validation_sources]
    image_suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

    def candidate_paths(source: Any):
        source_path = Path(source).expanduser()
        if source_path.is_dir():
            for image_path in source_path.rglob("*"):
                if image_path.suffix.lower() in image_suffixes:
                    yield image_path
        elif source_path.is_file() and source_path.suffix.lower() == ".txt":
            for line in source_path.read_text(encoding="utf-8").splitlines():
                listed_path = Path(line.strip()).expanduser()
                if not listed_path.is_absolute():
                    listed_path = source_path.parent / listed_path
                if listed_path.suffix.lower() in image_suffixes:
                    yield listed_path
        elif source_path.suffix.lower() in image_suffixes:
            yield source_path

    for source in validation_sources:
        for image_path in candidate_paths(source):
            frame = cv2.imread(str(image_path))
            if frame is not None:
                return frame, str(image_path)
    raise ValueError("No readable validation image found for inference measurements")


def _reset_gpu_peak(device: str) -> None:
    index = _cuda_device_index(device)
    if index is None:
        return
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(index)


def _torch_gpu_memory(device: str) -> Dict[str, Any]:
    index = _cuda_device_index(device)
    if index is None:
        return {"available": False, "reason": "Selected device is not CUDA"}
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "PyTorch CUDA telemetry unavailable"}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "CUDA is not available"}
    return {
        "available": True,
        "scope": "benchmark process allocations",
        "peak_allocated_mb": torch.cuda.max_memory_allocated(index) / (1024 * 1024),
        "peak_reserved_mb": torch.cuda.max_memory_reserved(index) / (1024 * 1024),
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
        "runtime": {},
        "artifacts": {},
        "notes": [],
        "tracking_metrics": {
            "status": "not_evaluated",
            "reason": (
                "No tracking_dataset was configured; the detection dataset has no "
                "ground-truth track IDs"
            ),
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
        result["runtime"] = _runtime_profile(device)
        result["runtime"]["requested_precision"] = (
            "fp16" if config.quantize == 16 else "fp32/default"
        )
        result["comparison_group"] = _comparison_group(config, result["runtime"])

        with _ResourceSampler() as sampler, _GpuSampler(
            device, result["runtime"]
        ) as gpu_sampler:
            load_started = time.perf_counter()
            model = YOLO(str(config.model), task="detect")
            result["timings"]["model_load_seconds"] = (
                time.perf_counter() - load_started
            )

            export_keys = {
                "format", "imgsz", "batch", "quantize", "int8", "dynamic",
                "simplify", "opset", "workspace", "nms",
            }
            model_overrides = getattr(model, "overrides", {}) or {}
            result["runtime"]["model_export_parameters"] = {
                key: model_overrides[key]
                for key in sorted(export_keys)
                if isinstance(model_overrides, dict) and key in model_overrides
            }
            result["runtime"]["configured_export_parameters"] = config.export_parameters
            result["comparison_group"] = _comparison_group(config, result["runtime"])

            _reset_gpu_peak(device)
            frame, probe_source = _representative_frame(config.dataset)
            result["runtime"]["inference_probe_source"] = probe_source
            result["notes"].append(
                "Per-frame inference timings use one real image from the configured "
                "validation split; detection metrics use the full validation split."
            )
            warmup_samples = []
            for _ in range(config.warmup_iterations):
                elapsed, _ = _predict_once(model, frame, config, device)
                warmup_samples.append(elapsed)
            result["timings"]["warmup"] = {
                "iterations": len(warmup_samples),
                "total_seconds": sum(warmup_samples),
                "per_frame_latency": _latency_summary(warmup_samples),
            }

            inference_samples = []
            inference_stages: Dict[str, List[float]] = {
                "preprocess": [],
                "inference": [],
                "postprocess": [],
                "total_stages": [],
            }
            for _ in range(config.repeats):
                for _ in range(config.inference_iterations):
                    elapsed, stages = _predict_once(model, frame, config, device)
                    inference_samples.append(elapsed)
                    for stage in ("preprocess", "inference", "postprocess"):
                        if stage in stages:
                            inference_stages[stage].append(stages[stage])
                    if stages:
                        inference_stages["total_stages"].append(sum(stages.values()))
            end_to_end_summary = _latency_summary(inference_samples)
            result["metrics"]["inference"] = {
                "end_to_end_frame": end_to_end_summary,
                "fps": _fps_summary(
                    [sample * 1000.0 for sample in inference_samples]
                ),
                "stage_latency_ms": {
                    stage: _distribution_summary_ms(samples)
                    for stage, samples in inference_stages.items()
                    if samples
                },
            }

            batch_started = []
            batch_latencies = []

            def on_val_batch_start(*_: Any) -> None:
                batch_started.append(time.perf_counter())

            def on_val_batch_end(*_: Any) -> None:
                if batch_started:
                    batch_latencies.append(
                        time.perf_counter() - batch_started.pop()
                    )

            model.add_callback("on_val_batch_start", on_val_batch_start)
            model.add_callback("on_val_batch_end", on_val_batch_end)
            validation_runs = []
            validation_walls = []
            validation_results = None
            for repeat in range(config.repeats):
                validation_started = time.perf_counter()
                validation_results = model.val(
                    data=config.dataset,
                    imgsz=config.imgsz,
                    batch=config.batch,
                    device=device,
                    conf=config.conf,
                    iou=config.iou,
                    quantize=config.quantize,
                    plots=True,
                    verbose=False,
                    project=str(output_dir / "ultralytics"),
                    name="{}_repeat_{}".format(run_name, repeat + 1),
                    exist_ok=True,
                )
                validation_walls.append(time.perf_counter() - validation_started)
                validation_runs.append(
                    extract_metrics(validation_results, getattr(model, "names", None))
                )
            result["timings"]["validation_wall_seconds"] = sum(validation_walls)
            result["timings"]["validation_repeat_wall_latency"] = _latency_summary(
                validation_walls
            )
            result["timings"]["validation_repeats"] = config.repeats
            result["metrics"].update(_aggregate_validation_metrics(validation_runs))
            result["metrics"]["validation_batch_latency"] = _latency_summary(
                batch_latencies
            )

            if config.tracking_dataset is not None:
                tracking_started = time.perf_counter()
                result["tracking_metrics"] = evaluate_mot_sequences(
                    model,
                    Path(config.tracking_dataset),
                    tracker=config.tracker,
                    imgsz=config.imgsz,
                    device=device,
                    conf=config.conf,
                    iou=config.iou,
                    quantize=config.quantize,
                )
                result["timings"]["tracking_wall_seconds"] = (
                    time.perf_counter() - tracking_started
                )

        result["resources"] = sampler.as_dict()
        result["resources"]["gpu"] = gpu_sampler.as_dict()
        result["resources"]["gpu_process_memory"] = _torch_gpu_memory(device)
        save_dir = getattr(validation_results, "save_dir", None)
        if save_dir is not None:
            result["artifacts"]["ultralytics_validation_dir"] = str(save_dir)
        result["decision"] = _assess_requirements(
            config, result["metrics"], result["resources"]
        )
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
