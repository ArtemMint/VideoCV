import cv2
import threading
import time
import queue
import numpy as np

class MotionRecorder:
    def __init__(
            self,
            fps: float = 30.0, 
            duration: int = 5
    ) -> None:
        self.fps = fps
        self.recording_duration = duration
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self.is_recording = False

    def _record_worker(self):
        filename = f"video_artifacts/motion_{int(time.time())}.mp4"
        fourcc: int = cv2.VideoWriter_fourcc(*'mp4v')

        first_frame = self.frame_queue.get()
        h, w = first_frame.shape[:2]

        out = cv2.VideoWriter(filename, fourcc, self.fps, (w, h))
        out.write(first_frame)

        frame_interval = 1.0 / self.fps  # 33ms between frames
        start_time = time.time()

        while time.time() - start_time < self.recording_duration:
            frame_start = time.time()

            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                out.write(frame)

            # wait to maintain the target frame rate
            elapsed = time.time() - frame_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        out.release()
        self.is_recording = False
        print(f"Finished recording {filename}")

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            print("Started recording motion event.")

            while not self.frame_queue.empty():
                self.frame_queue.get()

            thread = threading.Thread(target=self._record_worker)
            thread.start()
            print("Created new recording thread.")

    def process_frame(
            self, 
            frame: np.ndarray
    ) -> np.ndarray:
        if not self.is_recording:
            self.start_recording()
        if self.is_recording:
            # Use a separate thread to put the frame into the queue to avoid blocking the main thread
            threading.Thread(
                target=lambda f: self.frame_queue.put_nowait(f),
                args=(frame.copy(),),
                daemon=True
            ).start()
        return frame
