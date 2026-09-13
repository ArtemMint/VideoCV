import cv2
import time
import numpy as np
import tracker_ops

class InteractiveTracker:
    def __init__(self):
        self.selected_track_id = -1
        self.latest_tracks = [] # [(x1, y1, x2, y2, cls, conf, tid), ...]
        
        # Створюємо вікно та реєструємо mouse callback
        cv2.namedWindow("Video Playback")
        cv2.setMouseCallback("Video Playback", self.on_mouse_click)

    def on_mouse_click(self, event, x, y, flags, param):
        """Обробник кліку миші для вибору об'єкта"""
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_id = -1
            for (x1, y1, x2, y2, cls, conf, tid) in self.latest_tracks:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    clicked_id = tid
                    break
            
            self.selected_track_id = clicked_id
            if clicked_id != -1:
                print(f"[Target Locked] Selected Track ID: #{clicked_id}")
            else:
                print("[Target Unlocked] Zoom reset")

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.selected_track_id = -1
            print("[Target Unlocked] Zoom reset")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # 1. Застосування ROI Zoom у C++
        output_frame = tracker_ops.render_roi_zoom(
            frame, 
            self.latest_tracks, 
            self.selected_track_id,
            200
        )

        # 2. Draw the same boxes used by the mouse hit-test.
        for x1, y1, x2, y2, cls, conf, tid in self.latest_tracks:
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

DEFAULT_FPS = 30.0

ring_buffer = tracker_ops.FrameRingBuffer(capacity=30)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, DEFAULT_FPS)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

timer = cv2.TickMeter()
prev_frame = None
fps = DEFAULT_FPS

tracker = InteractiveTracker()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret or frame is None or frame.size == 0:
        print("Stream ended or failed to read a non-empty frame.")
        break

    tracker.latest_tracks = [
            (100, 150, 220, 280, "car", 0.85, 1),
            (350, 200, 420, 310, "drone", 0.92, 2)
        ]

    ring_buffer.push(frame)  # Push the current frame into the ring buffer
    time.sleep(0.33)  # Simulate processing delay

    timer.start()  # Start timing

    if prev_frame is not None:
        # mean_motion, annotated_frame = tracker_ops.process_motion_and_blur(frame, prev_frame, 20.0)
        # display_frame = tracker.process_frame(annotated_frame)
        # comb_frame = tracker_ops.combine_and_annotate(display_frame, prev_frame, fps)
        timer.stop()  # Stop timing
        fps = timer.getFPS()
        if fps > DEFAULT_FPS:
            fps = DEFAULT_FPS
        cv2.imshow('Video Playback', frame)

    prev_frame = frame.copy()  # Store the current frame for the next iteration

    processing_time = timer.getTimeSec()
    print(f"Frame processed in {processing_time:.6f} seconds.")
    timer.reset()  # Reset the timer for the next frame


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

batch_frames = ring_buffer.get_as_batch()

print(f"Отримано батч з {len(batch_frames)} кадрів із кількістю кадрів у буфері: {ring_buffer.get_size()} / {ring_buffer.get_capacity()}")


if batch_frames:
    cv2.imshow("Oldest Frame in Buffer", batch_frames[0])
    cv2.imshow("Newest Frame in Buffer", batch_frames[-1])
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
