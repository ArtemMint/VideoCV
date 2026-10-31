# VideoCV

## Project layout

- `cpp/` contains the C++ implementation, public headers, and pybind11 bindings:
  `cpp/src/`, `cpp/include/`, and `cpp/bindings/`.
- Python applications and packages remain in their current top-level locations
  (`Tracker/`, `MavLinkCTRL/`, `VideoRecorder/`, `model_benchmark/`, and the
  root-level entry-point scripts).
- `models/` contains example YOLO weights and exported model formats.

The C++ extension is built from the repository root with:

```bash
cmake -S . -B build
cmake --build build
```

## YOLO model benchmark

The `model_benchmark` package evaluates YOLO detection models on a common
dataset, measures process CPU and memory use, and writes an individual report
for every model plus a comparison summary.

Supported model formats are detected from the model path:

- PyTorch: `.pt` file
- ONNX: `.onnx` file
- OpenVINO: `.xml` file or a directory containing exactly one `.xml` file

The default dataset is the VisDrone2019-DET validation split, described by
[datasets/visdrone-val.yaml](datasets/visdrone-val.yaml). Tracking configuration
is recorded as `bytetrack.yaml` by default, but tracking metrics are not
evaluated: the detection dataset has no ground-truth track IDs. The other
defaults are 640-pixel input size, batch size 1, automatic device selection,
confidence threshold 0.001, and IoU threshold 0.7. Automatic device selection
uses CUDA for `.pt` models when available, CPU otherwise; ONNX and OpenVINO
currently use CPU when set to `auto`. An explicit `device` value is passed to
Ultralytics.

The VisDrone validation split is the default for models trained with the
VisDrone 10-class label map. The included VisDrone-trained YOLO26n OpenVINO
model can be evaluated directly. Do not validate COCO-pretrained models against
VisDrone and compare those metrics: their class label maps do not match.
The multi-model example therefore explicitly keeps COCO8 for the COCO-pretrained
YOLO models. For a fair hardware comparison, set the same explicit `device` for
each backend (for example, `cpu`) instead of mixing automatic CUDA for PyTorch
and CPU for ONNX/OpenVINO.

### Dataset setup

The default validation config downloads and prepares the VisDrone validation
split automatically before inference if the local files are missing. You can
also prepare it explicitly:

```bash
python -m model_benchmark.cli prepare-visdrone
```

This downloads only the VisDrone2019-DET validation archive (about 82 MB),
converts annotations to YOLO format, and writes the result to
`datasets/VisDrone/`. The large dataset files are ignored by Git; the small
dataset YAML is tracked. The full VisDrone train/test archives are not
downloaded. If the dataset directory exists but is incomplete, setup stops with
an error instead of overwriting local files; remove or repair that directory
before retrying.

To use your own prepared detection dataset, put its images/labels under
`datasets/` and create a YOLO dataset YAML. Set `dataset` in the run
configuration to that YAML path, for example `"dataset":
"datasets/my-dataset.yaml"`. The YAML must define `train`, `val`, and `names`
in Ultralytics format. The path is validated before inference. Dataset paths
with relative components are resolved from the repository root.

### Run one or more models

The comparison config includes local COCO-pretrained Nano weights and their
ONNX and OpenVINO exports: YOLOv8n and YOLO26n, for six runs total.
All runs explicitly use CPU so backend comparisons do not silently compare GPU
PyTorch with CPU ONNX/OpenVINO. Override `device` consistently if you want to
compare another supported device. YOLO27 has no downloadable weights yet, so
YOLO26n is the third model family. The original weights are official Ultralytics
release assets; Ultralytics models are distributed under AGPL-3.0 or a
commercial license.

The comparison config uses COCO8 explicitly because these weights use COCO
class IDs. To validate a VisDrone-trained model with the default dataset, run
the single-model config:

```bash
python -m model_benchmark.cli run model_benchmark/config.example.json \
  --output model_benchmark_results
```

Run the COCO model/format comparison config:

```bash
python -m model_benchmark.cli run model_benchmark/config.multi.example.json \
  --output model_benchmark_results
```

The first COCO comparison run may download Ultralytics' COCO8 dataset. The
project dependencies are listed in [requirements.txt](requirements.txt).
Install the optional inference backends needed by your model formats with:

```bash
python -m pip install -r model_benchmark/requirements-backends.txt
```

This installs CPU runtimes and export tooling for ONNX and OpenVINO. PyTorch
dependencies for `.pt` models are supplied by the existing environment.
If an optional backend is missing, that model's report will show the loading
error rather than silently substituting another runtime.

The local ONNX/OpenVINO files are exported from their matching `.pt` weights at
640x640. To recreate an export, use for example:

```bash
yolo export model=models/yolov8n.pt format=onnx imgsz=640 device=cpu
yolo export model=models/yolov8n.pt format=openvino imgsz=640 device=cpu
```

Repeat with `yolo11n.pt` or `yolo26n.pt` for the other model families.

The command validates each model in turn, continues after an individual model
fails, and returns a non-zero exit code if any run fails.

Install the project's Python packages with:

```bash
python -m pip install -r requirements.txt
```

The camera/video streamer and C++ extension also need Linux system packages,
which pip cannot provide:

```bash
sudo apt install build-essential cmake libopencv-dev python3-dev \
  python3-gi python3-gst-1.0 gir1.2-gstreamer-1.0 \
  gstreamer1.0-tools gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-ugly gstreamer1.0-libav
```

Use the matching Python development package for the Python version in the
virtual environment. Select a PyTorch build suitable for the operating system
and CUDA version when GPU inference is needed.

A single-run JSON file is also supported, for example
[config.example.json](model_benchmark/config.example.json). For multiple
models, use a top-level `runs` list. Each entry can override defaults:

```json
{
  "runs": [
    {"name": "yolo-v1-pt", "model": "models/yolo-v1.pt"},
    {
      "name": "yolo-v1-openvino",
      "model": "models/yolo-v1_openvino_model",
      "imgsz": 512,
      "device": "CPU"
    }
  ]
}
```

The following options can be set per run: `name`, `model`, `dataset`, `tracker`,
`imgsz`, `device`, `batch`, `conf`, `iou`, `repeats`, `warmup_iterations`,
`inference_iterations`, `half`, `export_parameters`, `min_map50_95`,
`max_latency_ms`, `max_memory_mb`, and `max_gpu_memory_mb`. Dataset defaults to
`datasets/visdrone-val.yaml`; COCO-pretrained models should explicitly use
`coco8.yaml` or another COCO-labeled dataset. Dataset paths are resolved from
the repository root. Only the project's known VisDrone validation YAML is
automatically prepared; custom dataset paths are never downloaded implicitly.
`export_parameters` records exporter settings for traceability; it does not
re-export or modify a model. `half` is passed as the requested FP16 mode.
Defaults are three validation repeats, five warm-up calls, and 20 timed
prediction calls per repeat.

CLI overrides can set common repeat counts and acceptance limits across all
runs in a config:

```bash
python -m model_benchmark.cli run model_benchmark/config.multi.example.json \
  --repeats 5 --warmup-iterations 10 --inference-iterations 30 \
  --min-map50-95 0.25 --max-latency-ms 40 --max-memory-mb 1500 \
  --max-gpu-memory-mb 1000
```

### Reports and measurements

Each run gets its own directory with `report.json` and `report.html`. Reports
include model format and parameters, host/device model, OS and runtime versions,
requested precision, exporter metadata, Precision, Recall, mAP50, mAP50-95,
per-class mAP, load time, warm-up time, and repeated validation wall times.
After a separate warm-up, the runner repeatedly predicts one real image from
the configured validation split and reports preprocess/inference/postprocess
stage timings, end-to-end per-frame p50/p95/mean/stddev/range, and FPS
distribution. Detection metrics are collected in repeated validation passes.

Resource reports include benchmark-process-tree CPU and RAM, whole-system CPU
utilization, optional system temperature sensors, and optional GPU utilization,
memory, temperature, power, and estimated energy. GPU board telemetry is
collected through NVML when available; install it with
`python -m pip install nvidia-ml-py`. CUDA process allocation peaks are reported
when PyTorch CUDA telemetry is available. If a sensor cannot be read, the
report records that it is unavailable rather than treating it as zero. CPU
utilization is relative to one logical CPU and can exceed 100% for
multi-threaded work.

The output directory also contains `summary.json` and `summary.html` for
comparing all runs. Pareto-optimal runs are identified using mAP50-95, p95
end-to-end latency, host RAM, and GPU memory (process allocation when available,
otherwise device-wide NVML usage). Dominance is checked
only among runs with the same comparison group: same host and device model,
dataset, image size, batch, thresholds, precision, and configured exporter
parameters. Different devices or CPU/GPU backends are deliberately not ranked
together. `min_map50_95`, `max_latency_ms`, `max_memory_mb`, and
`max_gpu_memory_mb` produce explicit per-run acceptance checks
(`max_latency_ms` is checked against p95) and a
`meets_requirements` result. Failed runs receive
an explicit failed report and error message; metrics are not filled with
success-like defaults. When acceptance limits are configured, the CLI exits
successfully only if at least one completed model meets all its limits.

### Current implementation status and next work

| Planned area | Status |
| --- | --- |
| Reproducible per-model parameters and multi-run config | Implemented |
| YOLOv8/26 `.pt`, ONNX, and OpenVINO CPU evaluation | Implemented and smoke-tested |
| Detection Precision, Recall, mAP, per-class results, and validation plots | Implemented |
| Separate load/warm-up/inference-stage/frame timings and repeated summaries | Implemented; see measurement limits below |
| Process-tree/system CPU, RAM, optional GPU and temperature/energy telemetry | Implemented with documented sensor availability |
| Acceptance thresholds and quality/latency/memory Pareto comparison | Implemented within equivalent hardware and run conditions |
| Individual and comparison JSON/HTML reports | Implemented |
| Unit tests and end-to-end validation | Unit tests cover the benchmark flow; hardware-specific measurements need validation on the target device |
| Ground-truth tracker stability metrics | Not implemented: COCO8 has no track IDs |
| Automatic error categorization and AI-assisted development workflow | Not implemented |
| Windows/macOS/Linux comparison in CI | Not implemented |

COCO8 currently supplies only four validation images in this workspace. It is
useful as a pipeline smoke test, not for statistically reliable model ranking.
The initial CPU comparison also showed visible mAP differences between the
PyTorch and exported variants for some model families. Treat these numbers as
a parity warning, not a conclusion that one format is more accurate. Before
using exports in production, compare per-image detections and validation
metrics on a representative, larger dataset with matching preprocessing,
confidence, IoU, and precision settings.

The timed per-frame probe repeats the same first validation image to keep image
content fixed; report latency separately from validation metrics. It measures
the `predict` API path (preprocess, model, and postprocess), not camera capture,
video decoding, rendering, or full application throughput. Use a representative
dataset and target hardware before making deployment decisions. Model loading is
measured once per model object; validation and inference probes are repeated.

NVML utilization, temperature, and power describe the selected NVIDIA device,
not only this process; energy is an estimate integrated from sampled board
power. CPU energy is not measured because there is no portable sensor interface.
Process-tree figures cover child processes observed during sampling. The
benchmark records runtime versions so results can be interpreted if runtime
versions change; keep software, ambient conditions, and power modes fixed for
high-confidence device comparisons.

Recommended next steps:

1. Add a representative, labeled validation dataset and an export parity test
   comparing `.pt` with ONNX/OpenVINO per image.
2. Validate the measurements against an external camera/stream timing harness
   and collect onboard power telemetry on the target platform if available.
3. Add a MOT-format dataset and tracker metrics such as IDF1/HOTA before
   reporting tracker stability.
4. Add CI smoke tests for supported OS/runtime combinations and document the
   AI-assisted change/review workflow.

### Tests

```bash
python -m unittest discover -s model_benchmark/tests -v
```
