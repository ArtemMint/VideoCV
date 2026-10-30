import cv2

cap = cv2.VideoCapture(0)
i = 0
path_to_save = "image_processing_examples/frame_"
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video frame is empty or processing is complete.")
        break

    if cv2.waitKey(30) & 0xFF == ord('s'):
        cv2.imwrite(path_to_save + str(i) + ".jpg", frame)
        i += 1

    cv2.imshow("Video Playback", frame)

cap.release()
cv2.destroyAllWindows()
