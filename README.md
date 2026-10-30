# VideoCV

## YOLO model benchmark

The `model_benchmark` package evaluates YOLO detection models on a common
dataset, measures process CPU and memory use, and writes an individual report
for every model plus a comparison summary.

Supported model formats are detected from the model path:

- PyTorch: `.pt` file
- ONNX: `.onnx` file
- OpenVINO: `.xml` file or a directory containing exactly one `.xml` file

The default dataset is Ultralytics' `coco8.yaml`. Tracking configuration is
recorded as `bytetrack.yaml` by default, but tracking metrics are not evaluated:
COCO8 has no ground-truth track IDs. The other defaults are 640-pixel input
size, batch size 1, automatic device selection, confidence threshold 0.001, and
IoU threshold 0.7. Automatic device selection uses CUDA for `.pt` models when
available, CPU otherwise; ONNX and OpenVINO currently use CPU when set to
`auto`. An explicit `device` value is passed to Ultralytics.

COCO8 is intentionally kept as the default smoke-test dataset, but it contains
only a handful of validation images. Its metrics verify that evaluation works;
they are too noisy to establish a reliable production ranking. For a fair
hardware comparison, set the same explicit `device` for each backend (for
example, `cpu`) instead of mixing automatic CUDA for PyTorch and CPU for
ONNX/OpenVINO.

### Run one or more models

The comparison config includes local COCO-pretrained Nano weights and their
ONNX and OpenVINO exports: YOLOv8n, YOLO11n, and YOLO26n, for nine runs total.
All runs explicitly use CPU so backend comparisons do not silently compare GPU
PyTorch with CPU ONNX/OpenVINO. Override `device` consistently if you want to
compare another supported device. YOLO27 has no downloadable weights yet, so
YOLO26n is the third model family. The original weights are official Ultralytics
release assets; Ultralytics models are distributed under AGPL-3.0 or a
commercial license.

Run the included comparison config:

```bash
python -m model_benchmark.cli run model_benchmark/config.multi.example.json \
  --output model_benchmark_results
```

The first run may download Ultralytics' COCO8 dataset. The project dependencies
are listed in [requirements.txt](requirements.txt).
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
`imgsz`, `device`, `batch`, `conf`, and `iou`. Dataset defaults to `coco8.yaml`;
change it only when deliberately comparing on another dataset. The model path
is resolved from the current working directory.

### Reports and measurements

Each run gets its own directory with `report.json` and `report.html`. Reports
include model format, run parameters, inferred device, OS, architecture, and
CPU metadata, Precision, Recall, mAP50, mAP50-95, per-class mAP when available,
validation average latency/FPS, p50/p95 validation batch latency, model load
and validation wall time, process CPU utilization, and baseline/peak process
RSS memory. CPU percentage is expressed relative to one
logical CPU, so it can exceed 100% for multi-threaded work. Ultralytics
validation plots are retained in the run's `ultralytics/` artifact directory
for visual error analysis.

The output directory also contains `summary.json` and `summary.html` for
comparing all runs. Failed models get an explicit failed report and error
message; metrics are not filled with success-like defaults.

### Current implementation status and next work

| Planned area | Status |
| --- | --- |
| Reproducible per-model parameters and multi-run config | Implemented |
| YOLOv8/11/26 `.pt`, ONNX, and OpenVINO CPU evaluation | Implemented and smoke-tested |
| Detection Precision, Recall, mAP, per-class results, and validation plots | Implemented |
| Latency/FPS, process CPU, and process RSS reporting | Implemented; see measurement limits below |
| Individual and comparison JSON/HTML reports | Implemented |
| Unit tests and end-to-end COCO8 run | Implemented; 13 unit tests pass and all nine configured runs completed |
| Ground-truth tracker stability metrics | Not implemented: COCO8 has no track IDs |
| Automatic error categorization and AI-assisted development workflow | Not implemented |
| Windows/macOS/Linux comparison in CI | Not implemented |

COCO8 currently supplies only four validation images in this workspace. It is
useful as a pipeline smoke test, not for statistically reliable model ranking.
The first nine-run CPU check also showed visible mAP differences between the
PyTorch and exported variants for some model families. Treat these numbers as
a parity warning, not a conclusion that one format is more accurate. Before
using exports in production, compare per-image detections and validation
metrics on a representative, larger dataset with matching preprocessing,
confidence, IoU, and precision settings.

CPU and memory values are for the benchmark Python process, not the whole
process tree or system. GPU memory is not measured. Validation batch p50/p95
includes the validation loop rather than an isolated, warmed-up model call;
use the exported format's validation speed fields for a rough inference
comparison only. Repeated runs, warm-up separation, child-process accounting,
and optional GPU metrics are good next improvements.

Recommended next steps:

1. Add a representative, labeled validation dataset and an export parity test
   comparing `.pt` with ONNX/OpenVINO per image.
2. Measure warm-up separately, repeat each benchmark, and summarize latency
   percentiles and variance; add process-tree and optional GPU memory metrics.
3. Add a MOT-format dataset and tracker metrics such as IDF1/HOTA before
   reporting tracker stability.
4. Add CI smoke tests for supported OS/runtime combinations and document the
   AI-assisted change/review workflow.

### Tests

```bash
python -m unittest discover -s model_benchmark/tests -v
```
