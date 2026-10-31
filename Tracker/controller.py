"""
Camera and Interactive Tracking Controllers for VideoCV.

Provides:
- CameraTracker: Threaded GStreamer capture & UDP H.264 streaming with YOLO ByteTrack inference.
- InteractiveTracker: Interactive mouse selection and ROI zooming for targets.
- PID: Standard PID controller with anti-windup clamping for flight control.
- render_roi_zoom: Helper to render picture-in-picture (PiP) zoomed views of tracked targets.
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
import time
from typing import Any, Callable, Sequence

# 1. Resolve libunwind conflict: GStreamer links libunwind.so.8, which overrides
# _Unwind_Resume with an unwinder that aborts on OpenVINO/C++ exceptions during device probing.
# Preloading libgcc_s with RTLD_GLOBAL preserves the GCC standard unwinder across shared libraries.
try:
    ctypes.CDLL("libgcc_s.so.1", mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass

# 2. Disable OpenVINO telemetry background fork process which causes aborts/crashes in multithreaded apps
os.environ["OPENVINO_TELEMETRY_OPT_IN"] = "0"
os.environ["OV_DISABLE_TELEMETRY"] = "1"
os.environ["CI"] = "true"

try:
    import openvino_telemetry

    openvino_telemetry.Telemetry.init = lambda self, *args, **kwargs: None
    openvino_telemetry.Telemetry.send = lambda self, *args, **kwargs: None
    openvino_telemetry.Telemetry.send_event = lambda self, *args, **kwargs: None
except Exception:
    pass

# 3. Initialize OpenVINO Core early so hardware device plugins are probed with native unwinding
try:
    import openvino as ov

    _ = ov.Core().available_devices
except Exception:
    pass

import cv2
import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

try:
    import tracker_ops
except ImportError:
    tracker_ops = None

logger = logging.getLogger(__name__)

# Ensure GStreamer is initialized once
if not Gst.is_initialized():
    Gst.init(None)

DEFAULT_FRAME_SIZE: tuple[int, int] = (640, 640)
DEFAULT_FPS: int = 30
DEFAULT_IMGSZ: int = 640
DEFAULT_MODEL_PATH: str = "models/yolo26n_visdrone_openvino_model"
DEFAULT_VIDEO_PATH: str = "image_processing_examples/files/drone-video.mp4"
FALLBACK_VIDEO_PATH: str = "media/drone.mp4"


class PID:
    """Proportional-Integral-Derivative (PID) controller with anti-windup clamping."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        limit: float = 1.0,
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.limit = abs(float(limit))
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error: float, dt: float) -> float:
        """Compute PID output given error and time delta."""
        if dt > 0:
            self.integral += error * dt
            # Anti-windup clamping on integral term
            if self.ki != 0:
                max_integral = self.limit / abs(self.ki)
                self.integral = max(min(self.integral, max_integral), -max_integral)
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0.0

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return max(min(output, self.limit), -self.limit)

    def reset(self) -> None:
        """Reset integral accumulator and previous error."""
        self.integral = 0.0
        self.prev_error = 0.0

    def __repr__(self) -> str:
        return (
            f"PID(kp={self.kp}, ki={self.ki}, kd={self.kd}, limit={self.limit}, "
            f"integral={self.integral:.4f}, prev_error={self.prev_error:.4f})"
        )


def render_roi_zoom(
    frame: np.ndarray,
    tracks: Sequence[tuple[int, int, int, int, str, float, int]],
    selected_track_id: int,
    zoom_size: int = 180,
) -> np.ndarray:
    """
    Render picture-in-picture (PiP) zoomed-in view of the selected tracked object.

    Args:
        frame: Input BGR image array.
        tracks: List of track tuples: (x1, y1, x2, y2, cls, conf, tid).
        selected_track_id: ID of the selected track to zoom into (< 0 for none).
        zoom_size: Size (width and height) of the PiP inset window in pixels.

    Returns:
        New image array with the PiP zoom overlay if active, or a copy of the frame.
    """
    # Use C++ acceleration if available
    if tracker_ops is not None and hasattr(tracker_ops, "render_roi_zoom"):
        try:
            return tracker_ops.render_roi_zoom(frame, list(tracks), selected_track_id, zoom_size)
        except Exception:
            pass

    output = frame.copy()
    if selected_track_id < 0 or not tracks:
        return output

    crop_coords = None
    for track in tracks:
        if len(track) >= 7 and track[6] == selected_track_id:
            crop_coords = (track[0], track[1], track[2], track[3])
            break

    if crop_coords is None:
        return output

    x1, y1, x2, y2 = crop_coords
    h, w = frame.shape[:2]

    # Clip coordinates within frame bounds
    cx1 = max(0, min(w, x1))
    cy1 = max(0, min(h, y1))
    cx2 = max(0, min(w, x2))
    cy2 = max(0, min(h, y2))

    if cx2 <= cx1 or cy2 <= cy1:
        return output

    cropped = frame[cy1:cy2, cx1:cx2]
    if cropped.size == 0:
        return output

    # Scale to zoom_size x zoom_size and draw border
    resized_crop = cv2.resize(cropped, (zoom_size, zoom_size), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(resized_crop, (0, 0), (zoom_size - 1, zoom_size - 1), (0, 255, 255), 3)

    # Position at bottom-right corner with 10px margin
    px = w - zoom_size - 10
    py = h - zoom_size - 10

    if px >= 0 and py >= 0:
        output[py : py + zoom_size, px : px + zoom_size] = resized_crop
        label = f"ZOOM #{selected_track_id}"
        cv2.putText(
            output,
            label,
            (px + 5, max(15, py - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output


class InteractiveTracker:
    """Interactive visual tracker with mouse selection and PiP ROI zoom overlay."""

    def __init__(self, window_name: str | None = "Video Playback") -> None:
        self.selected_track_id: int = -1
        self.latest_tracks: list[tuple[int, int, int, int, str, float, int]] = []
        self.window_name = window_name

        if self.window_name:
            try:
                cv2.namedWindow(self.window_name)
                cv2.setMouseCallback(self.window_name, self.on_mouse_click)
            except cv2.error as err:
                logger.warning("Could not initialize OpenCV window '%s': %s", self.window_name, err)

    def on_mouse_click(self, event: int, x: int, y: int, flags: int, param: Any = None) -> None:
        """Handle mouse click: left click selects target bounding box; right click deselects."""
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_id = -1
            for track in self.latest_tracks:
                if len(track) >= 7:
                    x1, y1, x2, y2, _cls, _conf, tid = track[:7]
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        clicked_id = tid
                        break

            self.selected_track_id = clicked_id
            if clicked_id != -1:
                logger.info("[Target Locked] Selected Track ID: #%d", clicked_id)
            else:
                logger.info("[Target Unlocked] Zoom reset")

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.selected_track_id = -1
            logger.info("[Target Unlocked] Zoom reset")

    def process_frame(self, frame: np.ndarray, zoom_size: int = 200) -> np.ndarray:
        """Render ROI zoom overlay and bounding boxes on the frame."""
        output_frame = render_roi_zoom(frame, self.latest_tracks, self.selected_track_id, zoom_size)

        for track in self.latest_tracks:
            if len(track) >= 7:
                x1, y1, x2, y2, cls, conf, tid = track[:7]
                is_selected = tid == self.selected_track_id
                color = (0, 255, 255) if is_selected else (0, 255, 0)
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cls} #{tid} {conf:.2f}"
                cv2.putText(
                    output_frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        return output_frame


class CameraTracker:
    """
    GStreamer camera capture & UDP H.264 streamer with YOLO tracking and drone telemetry.
    """

    def __init__(
        self,
        host: str,
        port: int = 5000,
        device: str | None = "/dev/video0",
        frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE,
        fps: int = DEFAULT_FPS,
        model: Any = None,
        tracker_config: str = "bytetrack.yaml",
        conf: float = 0.4,
        iou: float = 0.5,
        imgsz: int = DEFAULT_IMGSZ,
        infer_device: str = "CPU",
        frame_processor: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.device = device
        self.frame_size = frame_size
        self.fps = fps
        self.tracker_config = tracker_config
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.infer_device = infer_device
        self.frame_processor = frame_processor

        # Model can be passed directly as YOLO instance/mock or string path
        self._model_instance = model if not isinstance(model, str) and model is not None else None
        self._model_path = model if isinstance(model, str) else DEFAULT_MODEL_PATH

        self._latest_frame: np.ndarray | None = None
        self._latest_tracks: list[tuple[int, int, int, int, str, float, int]] = []
        self._telemetry: dict[str, float] = {}
        self.running: bool = False

        # Synchronization locks
        self.telemetry_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.tracks_lock = threading.Lock()

        # Build pipelines
        self.capture_pipeline_desc = self._build_capture_pipeline_desc(self.device, self.frame_size)
        self.send_pipeline_desc = self._build_send_pipeline_desc(self.host, self.port, self.frame_size, self.fps)

        self.capture_pipeline: Gst.Element = Gst.parse_launch(self.capture_pipeline_desc)
        self.send_pipeline: Gst.Element = Gst.parse_launch(self.send_pipeline_desc)
        self.sink: Gst.Element | None = None
        self.src: Gst.Element | None = None

    @staticmethod
    def _build_capture_pipeline_desc(device: str | None, frame_size: tuple[int, int]) -> str:
        width, height = frame_size

        # If device is not set or default /dev/video0 doesn't exist, fall back to sample video
        if not device or (device == "/dev/video0" and not os.path.exists("/dev/video0")):
            if os.path.exists(DEFAULT_VIDEO_PATH):
                device = DEFAULT_VIDEO_PATH
            elif os.path.exists(FALLBACK_VIDEO_PATH):
                device = FALLBACK_VIDEO_PATH
            else:
                device = "/dev/video0"

        # Direct custom GStreamer pipeline string
        if "!" in device:
            return device

        # V4L2 device (webcam or video capture card)
        if device.startswith("/dev/video"):
            return (
                f"v4l2src device={device} "
                f"! videoconvert "
                f"! videoscale "
                f"! video/x-raw,format=BGR,width={width},height={height} "
                f"! appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
            )

        # File source
        location = device
        if not os.path.exists(location):
            if os.path.exists(DEFAULT_VIDEO_PATH):
                location = DEFAULT_VIDEO_PATH
            elif os.path.exists(FALLBACK_VIDEO_PATH):
                location = FALLBACK_VIDEO_PATH

        return (
            f"filesrc location=\"{location}\" "
            f"! decodebin "
            f"! videoconvert "
            f"! videoscale "
            f"! video/x-raw,format=BGR,width={width},height={height} "
            f"! appsink name=sink emit-signals=true sync=true max-buffers=1 drop=true"
        )

    @staticmethod
    def _build_send_pipeline_desc(host: str, port: int, frame_size: tuple[int, int], fps: int) -> str:
        width, height = frame_size
        return (
            f"appsrc name=src format=time is-live=true max-bytes=0 block=false "
            f"caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 "
            f"! videoconvert "
            f"! x264enc tune=zerolatency bitrate=5000 speed-preset=ultrafast "
            f"! rtph264pay config-interval=1 pt=96 "
            f"! udpsink host={host} port={port} sync=false"
        )

    def get_model(self) -> Any:
        """Lazily load the YOLO model if not already provided."""
        if self._model_instance is None:
            from ultralytics import YOLO

            logger.info("Loading YOLO model from: %s", self._model_path)
            self._model_instance = YOLO(self._model_path)
        return self._model_instance

    @property
    def latest_tracks(self) -> list[tuple[int, int, int, int, str, float, int]]:
        """Return a thread-safe copy of the latest tracked targets."""
        with self.tracks_lock:
            return list(self._latest_tracks)

    @latest_tracks.setter
    def latest_tracks(self, tracks: Sequence[tuple[int, int, int, int, str, float, int]]) -> None:
        with self.tracks_lock:
            self._latest_tracks = list(tracks)

    @property
    def latest_frame(self) -> np.ndarray | None:
        """Return a thread-safe copy of the latest ingested video frame."""
        with self.frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    @latest_frame.setter
    def latest_frame(self, frame: np.ndarray | None) -> None:
        with self.frame_lock:
            self._latest_frame = frame.copy() if frame is not None else None

    @property
    def telemetry(self) -> dict[str, float]:
        """Return a thread-safe copy of the current drone telemetry."""
        with self.telemetry_lock:
            return dict(self._telemetry)

    @telemetry.setter
    def telemetry(self, data: dict[str, float]) -> None:
        with self.telemetry_lock:
            self._telemetry = dict(data)

    def update_telemetry(self, telemetry: dict[str, float]) -> None:
        """Update drone telemetry dictionary in a thread-safe manner."""
        with self.telemetry_lock:
            self._telemetry = telemetry.copy()

    def get_telemetry_snapshot(self) -> dict[str, float]:
        """Return an isolated copy of current telemetry."""
        with self.telemetry_lock:
            return self._telemetry.copy()

    def start(self) -> "CameraTracker":
        """Start GStreamer pipelines and hook the frame callback."""
        self.running = True
        self.sink = self.capture_pipeline.get_by_name("sink")
        self.src = self.send_pipeline.get_by_name("src")

        if self.sink is not None:
            self.sink.connect("new-sample", self.on_new_sample)

        self.capture_pipeline.set_state(Gst.State.PLAYING)
        self.send_pipeline.set_state(Gst.State.PLAYING)
        return self

    def stop(self) -> None:
        """Stop GStreamer pipelines and release resources."""
        self.running = False
        if hasattr(self, "capture_pipeline") and self.capture_pipeline is not None:
            self.capture_pipeline.set_state(Gst.State.NULL)
        if hasattr(self, "send_pipeline") and self.send_pipeline is not None:
            self.send_pipeline.set_state(Gst.State.NULL)

    def __enter__(self) -> "CameraTracker":
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    @staticmethod
    def _annotate_tracks(
        frame: np.ndarray,
        tracks: Sequence[tuple[int, int, int, int, str, float, int]],
    ) -> np.ndarray:
        """Draw detected tracks into the frame sent to the video stream."""
        if not tracks:
            return frame

        output_frame = frame.copy()
        for track in tracks:
            if len(track) < 7:
                continue
            x1, y1, x2, y2, cls, conf, tid = track
            color = (0, 255, 0)
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                output_frame,
                f"{cls} #{tid} {conf:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
        return output_frame

    def yolo_worker(self) -> None:
        """Worker thread executing YOLO ByteTrack inference on latest available frames."""
        logger.info("YOLO worker thread started")
        try:
            model = self.get_model()
        except Exception as exc:
            logger.exception("Failed to initialize YOLO model: %s", exc)
            return

        while self.running:
            frame_to_process = None
            with self.frame_lock:
                if self._latest_frame is not None:
                    frame_to_process = self._latest_frame.copy()

            if frame_to_process is None:
                time.sleep(0.01)
                continue

            try:
                results = model.track(
                    frame_to_process,
                    persist=True,
                    verbose=True,
                    tracker=self.tracker_config,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    device=self.infer_device,
                )[0]

                tracks: list[tuple[int, int, int, int, str, float, int]] = []
                if results.boxes is not None and results.boxes.id is not None:
                    for box, track_id in zip(results.boxes, results.boxes.id):
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_idx = int(box.cls[0])
                        if isinstance(results.names, dict):
                            cls = results.names.get(cls_idx, str(cls_idx))
                        elif isinstance(results.names, (list, tuple)):
                            cls = results.names[cls_idx]
                        else:
                            cls = str(cls_idx)
                        tid = int(track_id)
                        tracks.append((x1, y1, x2, y2, cls, conf, tid))

                with self.tracks_lock:
                    self._latest_tracks = tracks

            except Exception as err:
                logger.error("Error during YOLO tracking: %s", err)

            time.sleep(0.001)

        logger.info("YOLO worker thread finished")

    def on_new_sample(self, sink: Gst.Element) -> Gst.FlowReturn:
        """Callback invoked by GStreamer appsink when a new frame is decoded."""
        sample: Gst.Sample | None = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buf: Gst.Buffer = sample.get_buffer()
        caps: Gst.Caps = sample.get_caps()
        structure = caps.get_structure(0)
        width: int = structure.get_value("width")
        height: int = structure.get_value("height")
        expected_size = width * height * 3

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            logger.error("Failed to map GStreamer input buffer")
            return Gst.FlowReturn.ERROR

        try:
            if map_info.size < expected_size:
                logger.error(
                    "Invalid input buffer: got %d bytes, expected at least %d for %dx%d BGR",
                    map_info.size,
                    expected_size,
                    width,
                    height,
                )
                return Gst.FlowReturn.ERROR

            raw_frame = np.frombuffer(map_info.data[:expected_size], dtype=np.uint8)
            frame = raw_frame.reshape((height, width, 3)).copy()
        finally:
            buf.unmap(map_info)

        with self.frame_lock:
            self._latest_frame = frame

        stream_frame = frame
        if self.frame_processor is not None:
            try:
                stream_frame = self.frame_processor(frame)
            except Exception as exc:
                logger.error("frame_processor failed: %s", exc)
                stream_frame = frame

        stream_frame = self._annotate_tracks(stream_frame, self.latest_tracks)

        if self.src is not None:
            out_buf = Gst.Buffer.new_wrapped(stream_frame.tobytes())
            out_buf.pts = buf.pts
            out_buf.dts = buf.dts
            out_buf.duration = buf.duration
            flow_return = self.src.emit("push-buffer", out_buf)
            if flow_return != Gst.FlowReturn.OK:
                logger.warning("Output appsrc push failed: %s", getattr(flow_return, "value_nick", flow_return))
                return flow_return

        return Gst.FlowReturn.OK


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    with CameraTracker(host="127.0.0.1", port=5000) as tracker:
        yolo_thread = threading.Thread(target=tracker.yolo_worker, daemon=True)
        yolo_thread.start()

        loop = GLib.MainLoop()
        try:
            loop.run()
        except KeyboardInterrupt:
            print("Stopping stream...")
            tracker.stop()
            loop.quit()
