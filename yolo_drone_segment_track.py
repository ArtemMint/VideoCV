import cv2

from ultralytics import solutions, YOLO

model_path = "models/vis_epoch50.pt"
model_object = YOLO(model_path)
model_object.export(
    format="openvino",
    imgsz=640,
    dynamic=True,
    half=True,
)

model_onnx = YOLO("best.onnx")
model_openvino = YOLO("best_openvino_model")

model_onnx.benchmark(task="segment", device="CPU", imgsz=640, half=True, verbose=True)
model_openvino.benchmark(task="segment", device="CPU", imgsz=640, half=True, verbose=True)

video = "https://www.pexels.com/download/video/29268886/"
cap = cv2.VideoCapture(video)
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("instance-segmentation.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init InstanceSegmentation
isegment = solutions.InstanceSegmentation(
    show=True,  # display the output
    model=model_object,  # model="yolo26n-seg.pt" for object segmentation using YOLO26.
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break

    im0 = cv2.resize(im0, (int(w/3), int(h/3)))
    results = isegment(im0)
    video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
