"""
Unit tests for Tracker.controller module.

Verifies functionality and consistency of:
- PID controller
- render_roi_zoom helper
- InteractiveTracker
- CameraTracker
"""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock

import cv2
import numpy as np

from Tracker.controller import (
    Gst,
    PID,
    CameraTracker,
    InteractiveTracker,
    render_roi_zoom,
)


class PIDControllerTests(unittest.TestCase):
    """Tests for PID controller."""

    def test_init_defaults_and_limits(self) -> None:
        pid = PID(kp=1.0, ki=0.1, kd=0.05, limit=2.5)
        self.assertEqual(pid.kp, 1.0)
        self.assertEqual(pid.ki, 0.1)
        self.assertEqual(pid.kd, 0.05)
        self.assertEqual(pid.limit, 2.5)
        self.assertEqual(pid.integral, 0.0)
        self.assertEqual(pid.prev_error, 0.0)

        # Negative limit should be forced positive
        pid_neg = PID(kp=1.0, ki=0.1, kd=0.05, limit=-3.0)
        self.assertEqual(pid_neg.limit, 3.0)

    def test_proportional_response(self) -> None:
        pid = PID(kp=0.5, ki=0.0, kd=0.0, limit=10.0)
        out = pid.compute(error=4.0, dt=0.1)
        self.assertAlmostEqual(out, 2.0)
        self.assertEqual(pid.prev_error, 4.0)

    def test_integral_accumulation_and_anti_windup(self) -> None:
        pid = PID(kp=0.0, ki=1.0, kd=0.0, limit=5.0)
        # 1st step: error 2.0, dt 1.0 -> integral = 2.0, out = 2.0
        out1 = pid.compute(error=2.0, dt=1.0)
        self.assertAlmostEqual(out1, 2.0)
        self.assertAlmostEqual(pid.integral, 2.0)

        # 2nd step: error 2.0, dt 1.0 -> integral = 4.0, out = 4.0
        out2 = pid.compute(error=2.0, dt=1.0)
        self.assertAlmostEqual(out2, 4.0)

        # Accumulate large error -> anti-windup should clamp integral to limit / ki = 5.0
        for _ in range(10):
            pid.compute(error=10.0, dt=1.0)
        self.assertLessEqual(pid.integral, 5.0)
        self.assertGreaterEqual(pid.integral, -5.0)

    def test_derivative_response(self) -> None:
        pid = PID(kp=0.0, ki=0.0, kd=0.2, limit=10.0)
        # 1st step: error 2.0, prev_error was 0.0, dt 0.1 -> derivative = (2.0 - 0) / 0.1 = 20.0
        out = pid.compute(error=2.0, dt=0.1)
        self.assertAlmostEqual(out, 0.2 * 20.0)

    def test_zero_dt_avoids_division_by_zero(self) -> None:
        pid = PID(kp=1.0, ki=1.0, kd=1.0, limit=10.0)
        out = pid.compute(error=5.0, dt=0.0)
        self.assertAlmostEqual(out, 5.0)  # only kp * error, derivative = 0, integral unchanged

    def test_output_limit_clamping(self) -> None:
        pid = PID(kp=10.0, ki=0.0, kd=0.0, limit=2.0)
        self.assertEqual(pid.compute(error=100.0, dt=0.1), 2.0)
        self.assertEqual(pid.compute(error=-100.0, dt=0.1), -2.0)

    def test_reset(self) -> None:
        pid = PID(kp=1.0, ki=1.0, kd=1.0, limit=10.0)
        pid.compute(error=5.0, dt=0.5)
        self.assertNotEqual(pid.integral, 0.0)
        self.assertNotEqual(pid.prev_error, 0.0)

        pid.reset()
        self.assertEqual(pid.integral, 0.0)
        self.assertEqual(pid.prev_error, 0.0)

    def test_repr(self) -> None:
        pid = PID(kp=1.0, ki=0.5, kd=0.1, limit=2.0)
        rep = repr(pid)
        self.assertIn("PID", rep)
        self.assertIn("kp=1.0", rep)
        self.assertIn("limit=2.0", rep)


class RenderRoiZoomTests(unittest.TestCase):
    """Tests for render_roi_zoom helper function."""

    def setUp(self) -> None:
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_empty_tracks_returns_copy(self) -> None:
        result = render_roi_zoom(self.frame, [], selected_track_id=1, zoom_size=100)
        self.assertEqual(result.shape, self.frame.shape)
        np.testing.assert_array_equal(result, self.frame)

    def test_negative_id_returns_copy(self) -> None:
        tracks = [(10, 10, 50, 50, "car", 0.9, 1)]
        result = render_roi_zoom(self.frame, tracks, selected_track_id=-1, zoom_size=100)
        np.testing.assert_array_equal(result, self.frame)

    def test_unmatched_id_returns_copy(self) -> None:
        tracks = [(10, 10, 50, 50, "car", 0.9, 1)]
        result = render_roi_zoom(self.frame, tracks, selected_track_id=42, zoom_size=100)
        np.testing.assert_array_equal(result, self.frame)

    def test_valid_track_renders_zoom(self) -> None:
        # Fill a bright patch for the target
        self.frame[20:80, 30:90] = 200
        tracks = [(30, 20, 90, 80, "drone", 0.95, 7)]
        result = render_roi_zoom(self.frame, tracks, selected_track_id=7, zoom_size=100)
        self.assertEqual(result.shape, self.frame.shape)

        # PiP is placed at bottom-right: w - zoom_size - 10, h - zoom_size - 10
        px = 640 - 100 - 10
        py = 480 - 100 - 10
        pip_region = result[py : py + 100, px : px + 100]
        # Inset contains the scaled bright patch and yellow border (0, 255, 255)
        self.assertTrue(np.any(pip_region > 0))

    def test_boundary_clipping(self) -> None:
        # Bounding box coordinates partially outside the frame bounds
        tracks = [(-20, -30, 100, 150, "truck", 0.88, 3)]
        result = render_roi_zoom(self.frame, tracks, selected_track_id=3, zoom_size=80)
        self.assertEqual(result.shape, self.frame.shape)

    def test_degenerate_bbox_coordinates(self) -> None:
        # x2 <= x1 or y2 <= y1
        tracks = [(100, 100, 50, 50, "truck", 0.88, 3)]
        result = render_roi_zoom(self.frame, tracks, selected_track_id=3, zoom_size=80)
        np.testing.assert_array_equal(result, self.frame)


class InteractiveTrackerTests(unittest.TestCase):
    """Tests for InteractiveTracker."""

    def test_headless_init(self) -> None:
        tracker = InteractiveTracker(window_name=None)
        self.assertEqual(tracker.selected_track_id, -1)
        self.assertEqual(tracker.latest_tracks, [])

    def test_mouse_click_selection(self) -> None:
        tracker = InteractiveTracker(window_name=None)
        tracker.latest_tracks = [
            (10, 10, 50, 50, "car", 0.9, 101),
            (100, 100, 200, 200, "truck", 0.85, 202),
        ]

        # Click inside truck bounding box
        tracker.on_mouse_click(event=cv2.EVENT_LBUTTONDOWN, x=150, y=150, flags=0)
        self.assertEqual(tracker.selected_track_id, 202)

        # Click outside all boxes -> reset to -1
        tracker.on_mouse_click(event=cv2.EVENT_LBUTTONDOWN, x=500, y=400, flags=0)
        self.assertEqual(tracker.selected_track_id, -1)

    def test_mouse_right_click_resets(self) -> None:
        tracker = InteractiveTracker(window_name=None)
        tracker.selected_track_id = 99
        tracker.on_mouse_click(event=cv2.EVENT_RBUTTONDOWN, x=50, y=50, flags=0)
        self.assertEqual(tracker.selected_track_id, -1)

    def test_process_frame(self) -> None:
        tracker = InteractiveTracker(window_name=None)
        tracker.latest_tracks = [(20, 20, 80, 80, "person", 0.92, 5)]
        tracker.selected_track_id = 5

        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        processed = tracker.process_frame(frame, zoom_size=100)
        self.assertEqual(processed.shape, frame.shape)
        # Should contain drawn annotations
        self.assertTrue(np.any(processed > 0))


class CameraTrackerTests(unittest.TestCase):
    """Tests for CameraTracker."""

    def test_capture_pipeline_descriptions(self) -> None:
        # V4L2 device
        desc_v4l2 = CameraTracker._build_capture_pipeline_desc("/dev/video0", (640, 480))
        self.assertIn("v4l2src device=/dev/video0", desc_v4l2)
        self.assertIn("width=640,height=480", desc_v4l2)
        self.assertIn("appsink name=sink", desc_v4l2)

        # File path
        desc_file = CameraTracker._build_capture_pipeline_desc(
            "image_processing_examples/files/drone-video.mp4", (1280, 720)
        )
        self.assertIn("filesrc", desc_file)
        self.assertIn("decodebin", desc_file)
        self.assertIn("width=1280,height=720", desc_file)

        # Custom pipeline with '!'
        custom = "videotestsrc ! appsink name=sink"
        desc_custom = CameraTracker._build_capture_pipeline_desc(custom, (640, 480))
        self.assertEqual(desc_custom, custom)

    def test_send_pipeline_description(self) -> None:
        desc_send = CameraTracker._build_send_pipeline_desc("127.0.0.1", 5005, (640, 480), 30)
        self.assertIn("appsrc name=src", desc_send)
        self.assertIn("width=640,height=480,framerate=30/1", desc_send)
        self.assertIn("x264enc", desc_send)
        self.assertIn("rtph264pay", desc_send)
        self.assertIn("udpsink host=127.0.0.1 port=5005", desc_send)

    def test_telemetry_isolation(self) -> None:
        tracker = CameraTracker(
            host="127.0.0.1",
            port=5000,
            device="videotestsrc num-buffers=1 ! appsink name=sink",
            model=MagicMock(),
        )
        data = {"altitude": 10.5, "battery": 92.0}
        tracker.update_telemetry(data)

        # Modify original dict to verify thread-safe copy
        data["altitude"] = 999.0
        snapshot = tracker.get_telemetry_snapshot()
        self.assertEqual(snapshot["altitude"], 10.5)

        # Property getter isolation
        prop_val = tracker.telemetry
        prop_val["battery"] = 0.0
        self.assertEqual(tracker.telemetry["battery"], 92.0)

        tracker.stop()

    def test_tracks_and_frame_properties_isolation(self) -> None:
        tracker = CameraTracker(
            host="127.0.0.1",
            port=5000,
            device="videotestsrc num-buffers=1 ! appsink name=sink",
            model=MagicMock(),
        )

        sample_frame = np.ones((100, 100, 3), dtype=np.uint8) * 50
        tracker.latest_frame = sample_frame
        retrieved = tracker.latest_frame
        self.assertIsNotNone(retrieved)
        # Modify retrieved array
        retrieved[0, 0, 0] = 255
        self.assertEqual(tracker.latest_frame[0, 0, 0], 50)

        # Tracks property isolation
        tracks = [(1, 2, 3, 4, "car", 0.9, 10)]
        tracker.latest_tracks = tracks
        retrieved_tracks = tracker.latest_tracks
        retrieved_tracks.append((5, 6, 7, 8, "truck", 0.8, 20))
        self.assertEqual(len(tracker.latest_tracks), 1)

        tracker.stop()

    def test_yolo_worker_with_mock_model(self) -> None:
        # Create mock YOLO model and track result
        mock_model = MagicMock()
        mock_result = MagicMock()

        # Mock boxes with coordinates, conf, cls, id
        box1 = MagicMock()
        box1.conf = [0.88]
        box1.xyxy = [[10.0, 20.0, 100.0, 150.0]]
        box1.cls = [0]

        mock_result.boxes = MagicMock()
        mock_result.boxes.id = [42]
        mock_result.boxes.__iter__.return_value = [box1]
        mock_result.names = {0: "truck"}

        mock_model.track.return_value = [mock_result]

        tracker = CameraTracker(
            host="127.0.0.1",
            port=5000,
            device="videotestsrc num-buffers=1 ! appsink name=sink",
            model=mock_model,
        )
        tracker.running = True
        tracker.latest_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        worker_thread = threading.Thread(target=tracker.yolo_worker, daemon=True)
        worker_thread.start()

        # Wait for tracks to update
        for _ in range(50):
            if len(tracker.latest_tracks) > 0:
                break
            time.sleep(0.02)

        tracker.running = False
        worker_thread.join(timeout=1.0)

        self.assertEqual(len(tracker.latest_tracks), 1)
        x1, y1, x2, y2, cls, conf, tid = tracker.latest_tracks[0]
        self.assertEqual((x1, y1, x2, y2), (10, 20, 100, 150))
        self.assertEqual(cls, "truck")
        self.assertAlmostEqual(conf, 0.88)
        self.assertEqual(tid, 42)

        tracker.stop()

    def test_frame_processor_hook_and_sample_ingestion(self) -> None:
        processed_flags = []

        def custom_processor(frame: np.ndarray) -> np.ndarray:
            processed_flags.append(True)
            return frame

        # Use videotestsrc for real GStreamer capture
        pipe_str = "videotestsrc num-buffers=2 ! video/x-raw,format=BGR,width=160,height=120 ! appsink name=sink emit-signals=true max-buffers=1 drop=true"
        tracker = CameraTracker(
            host="127.0.0.1",
            port=5000,
            device=pipe_str,
            frame_size=(160, 120),
            model=MagicMock(),
            frame_processor=custom_processor,
        )

        with tracker:
            # Allow pipeline to pull samples
            for _ in range(50):
                if tracker.latest_frame is not None and processed_flags:
                    break
                time.sleep(0.02)

        self.assertIsNotNone(tracker.latest_frame)
        self.assertEqual(tracker.latest_frame.shape, (120, 160, 3))
        self.assertTrue(len(processed_flags) > 0)

    def test_on_new_sample_streams_frames_with_detections(self) -> None:
        tracker = CameraTracker(
            host="127.0.0.1",
            port=5000,
            device="videotestsrc num-buffers=1 ! appsink name=sink",
            frame_size=(100, 100),
            model=MagicMock(),
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        input_buffer = Gst.Buffer.new_wrapped(frame.tobytes())
        sample = MagicMock()
        sample.get_buffer.return_value = input_buffer
        sample.get_caps.return_value = Gst.Caps.from_string(
            "video/x-raw,format=BGR,width=100,height=100"
        )
        sink = MagicMock()
        sink.emit.return_value = sample
        tracker.src = MagicMock()
        tracker.src.emit.return_value = Gst.FlowReturn.OK
        tracker.latest_tracks = [(10, 10, 80, 80, "truck", 0.9, 1)]

        flow_return = tracker.on_new_sample(sink)

        self.assertEqual(flow_return, Gst.FlowReturn.OK)
        pushed_buffer = tracker.src.emit.call_args.args[1]
        pushed_frame = np.frombuffer(
            pushed_buffer.extract_dup(0, pushed_buffer.get_size()), dtype=np.uint8
        ).reshape((100, 100, 3))
        np.testing.assert_array_equal(pushed_frame[10, 10], (0, 255, 0))
        tracker.stop()


if __name__ == "__main__":
    unittest.main()
