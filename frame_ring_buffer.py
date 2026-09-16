import cv2
import time
import tracker_ops

from Tracker.controller import InteractiveTracker

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
fps = DEFAULT_FPS

while cap.isOpened():
    ret, frame = cap.read()

    if not ret or frame is None or frame.size == 0:
        print("Stream ended or failed to read a non-empty frame.")
        break

    ring_buffer.push(frame)  # Push the current frame into the ring buffer

    timer.start()  # Start timing

    tracker_ops.process_pipeline(frame)
    timer.stop()  # Stop timing
    fps = timer.getFPS()
    if fps > DEFAULT_FPS:
        fps = DEFAULT_FPS
    cv2.imshow('Video Playback', frame)

    processing_time = timer.getTimeSec()
    print(f"Frame processed in {processing_time:.6f} seconds.")
    timer.reset()  # Reset the timer for the next frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

batch_frames = ring_buffer.get_as_batch()


if batch_frames:
    cv2.imshow("Oldest Frame in Buffer", batch_frames[0])
    cv2.imshow("Newest Frame in Buffer", batch_frames[-1])
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
