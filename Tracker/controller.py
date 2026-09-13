import os
import sys
import time
import threading
import gi
import cv2
import numpy as np

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from VideoRecorder import MotionRecorder
import tracker_ops


Gst.init(None)
Gst.debug_set_default_threshold(Gst.DebugLevel.INFO)

IMGSZ = 640
model = YOLO('models/vis_epoch50_openvino_model', task='detect')


class PID:
    def __init__(
            self, 
            kp: float, 
            ki: float, 
            kd: float, 
            limit: float = 1.0,
    ) -> None:
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limit = limit
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error: float, dt: float) -> float:
        if dt > 0:
            self.integral += error * dt
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0.0
            
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return max(min(output, self.limit), -self.limit)

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


class CameraTracker:
    def __init__(
            self, 
            host: str, 
            port: int = 5000, 
            device: str = '/dev/video0', 
            frame_size: tuple[int, int] = (640, 640), 
            fps: int = 30
    ):
        self.fps = fps
        self.frame_size: tuple[int, int] = frame_size
        
        self.latest_frame = None
        self.latest_tracks = []
        self.telemetry = {}
        self.running = True

        # Спеціальні locks для потокобезпечності
        self.telemetry_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.tracks_lock = threading.Lock()

        self._capture_pipeline(device)
        self._send_pipeline(host, port)
        self.motion_recorder = MotionRecorder(fps=self.fps)

    def __enter__(self) -> 'CameraTracker':
        self.sink: Gst.Element = self.capture_pipeline.get_by_name('sink')
        self.src: Gst.Element = self.send_pipeline.get_by_name('src')
        self.sink.connect('new-sample', self.on_new_sample)
        
        self.capture_pipeline.set_state(Gst.State.PLAYING)
        self.send_pipeline.set_state(Gst.State.PLAYING)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.running = False
        self.capture_pipeline.set_state(Gst.State.NULL)
        self.send_pipeline.set_state(Gst.State.NULL)

    def _capture_pipeline(self, device: str | None = None) -> None:
        # Пайплайн для відеофайлу з декодуванням і приведенням до формату BGR
        self.capture_pipeline = Gst.parse_launch(f"""
            filesrc location="image_processing/files/drone-video.mp4"
            ! decodebin
            ! videoconvert
            ! videoscale
            ! video/x-raw,format=BGR,width={self.frame_size[0]},height={self.frame_size[1]}
            ! appsink name=sink emit-signals=true sync=true max-buffers=1 drop=true
        """)

    def _send_pipeline(self, host: str, port: int) -> None:
        # Увага: ширина у 2 рази більша (frame_size[0] * 2) через hstack
        self.send_pipeline: Gst.Element = Gst.parse_launch(f"""
            appsrc name=src format=time is-live=true max-bytes=0 block=false
                caps=video/x-raw,format=BGR,width={self.frame_size[0] * 2},height={self.frame_size[1]},framerate={self.fps}/1
            ! videoconvert
            ! x264enc tune=zerolatency bitrate=5000 speed-preset=ultrafast
            ! rtph264pay config-interval=1 pt=96
            ! udpsink host={host} port={port} sync=false
        """)

    def update_telemetry(self, telemetry: dict[str, float]) -> None:
        with self.telemetry_lock:
            self.telemetry = telemetry.copy()

    def get_telemetry_snapshot(self) -> dict[str, float]:
        with self.telemetry_lock:
            return self.telemetry.copy()

    def yolo_worker(self):
        """Воркер для виконання трекінгу YOLO у декілька FPS без блокування основного потоку."""
        while self.running:
            frame_to_process = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()

            if frame_to_process is None:
                time.sleep(0.01)
                continue

            results = model.track(
                frame_to_process,
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml",
                conf=0.4,
                iou=0.5,
                imgsz=IMGSZ,
                device='CPU'
            )[0]

            tracks: list[tuple[int, int, int, int, str, float, int]] = []
            if results.boxes is not None and results.boxes.id is not None:
                for box, track_id in zip(results.boxes, results.boxes.id):
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = results.names[int(box.cls[0])]
                    tid = int(track_id)
                    tracks.append((x1, y1, x2, y2, cls, conf, tid))

            with self.tracks_lock:
                self.latest_tracks = tracks

            # Запобігає 100% завантаженню CPU
            time.sleep(0.005)

    def on_new_sample(self, sink: Gst.Element) -> Gst.FlowReturn:
        sample: Gst.Sample = sink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.OK

        buf: Gst.Buffer = sample.get_buffer()
        caps: Gst.Caps = sample.get_caps()
        
        structure = caps.get_structure(0)
        width = structure.get_value('width')
        height = structure.get_value('height')

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        try:
            # 1. Створюємо numpy array поверх пам'яті GStreamer
            raw_frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape(height, width, 3)

            # 2. КРИТИЧНО: Робимо копію в оперативну пам'ять Python, 
            # щоб C++ модуль працював зі власною безпечною ділянкою пам'яті!
            frame = raw_frame.copy()

            if (width, height) != self.frame_size:
                frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_LINEAR)

        finally:
            # Unmap робимо тільки після того, як створили безпечну копію `frame`
            buf.unmap(map_info)

        # 3. Безпечно передаємо `frame` (який вже є власником власної пам'яті) в C++
        frame_processed = frame.copy()
        
        with self.frame_lock:
            prev_frame = self.latest_frame.copy() if self.latest_frame is not None else None
            self.latest_frame = frame.copy()

        if prev_frame is not None:
            mean_motion, frame_processed = tracker_ops.process_motion_and_blur(
                frame, prev_frame, 20.0
            )

        with self.tracks_lock:
            current_tracks = list(self.latest_tracks)

        if current_tracks:
            tracker_ops.draw_tracks(frame_processed, current_tracks)

        telemetry = self.get_telemetry_snapshot()
        final_output = tracker_ops.combine_and_annotate(
            frame, 
            frame_processed,
            telemetry.get('latitude', 0.0),
            telemetry.get('longitude', 0.0),
            telemetry.get('altitude', 0.0),
            telemetry.get('battery', 0)
        )

        # 4. Відправка результату
        out_buf = Gst.Buffer.new_wrapped(final_output.tobytes())
        out_buf.pts = buf.pts
        out_buf.dts = buf.dts
        out_buf.duration = buf.duration
        self.src.emit('push-buffer', out_buf)

        return Gst.FlowReturn.OK


if __name__ == '__main__':
    with CameraTracker(host='127.0.0.1', port=5000) as tracker:
        yolo_thread = threading.Thread(target=tracker.yolo_worker, daemon=True)
        yolo_thread.start()
        
        loop = GLib.MainLoop()
        try:
            loop.run()
        except KeyboardInterrupt:
            print("Stop streaming...")
            tracker.running = False
            loop.quit()
