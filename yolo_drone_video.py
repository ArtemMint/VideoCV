import cv2

from ultralytics import solutions, YOLO

# model_path = "models/vis_epoch50.pt"
# model_object = YOLO(model_path)
# model_object.export(
#     format="openvino",
#     imgsz=640,
#     dynamic=True,
#     half=True,
# )

model_openvino = YOLO("models/yolo26n_visdrone_openvino_model")

video = "https://www.pexels.com/download/video/29268886/"
cap = cv2.VideoCapture(video)
assert cap.isOpened(), "Error reading video file"


# Process video
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = model_openvino.track(
        frame,
        persist=True,
        verbose=True,
        tracker="bytetrack.yaml",
        conf=0.4,
        iou=0.5,
        imgsz=640,
        device='CPU'
    )[0]

cap.release()
cv2.destroyAllWindows()
