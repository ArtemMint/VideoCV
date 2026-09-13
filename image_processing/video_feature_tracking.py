import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)


class ORBLKTracker:
    def __init__(self, max_features=None):
        self.orb = cv.ORB.create(nfeatures=max_features)
        self.points = []
        self.old_gray = None
        self.frame_count = 0
        self.min_points = 300
        self.redetection_interval = 10
        self.lk_params = dict(winSize=(21, 21), maxLevel=4,
                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    def detect(self, gray):
        kps = self.orb.detect(gray, None)
        if not kps:
            return None
        pts = np.array([kp.pt for kp in kps], dtype=np.float32)
        return pts.reshape(-1, 2)

    def update(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # gray = cv.GaussianBlur(gray, (5, 5), 0)
        # gray = cv.Canny(gray, 100, 200)

        if self.frame_count % self.redetection_interval == 0 or len(self.points) < self.min_points:
            self.points = self.detect(gray)
        else:
            self.points, status, _ = cv.calcOpticalFlowPyrLK(
                self.old_gray,
                gray,
                self.points,
                None,
                **self.lk_params)
            self.points = self.points[status.flatten() == 1].reshape(-1, 2)
        self.old_gray = gray.copy()
        self.frame_count += 1
        return self.points

if __name__ == "__main__":
    tracker = ORBLKTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        points = tracker.update(frame)
        points = [cv.KeyPoint(x=p[0], y=p[1], size=1) for p in points]

        # Draw the points on the original frame
        img_keypoints = cv.drawKeypoints(
            frame,
            points,
            None,
            color=(0, 255, 0),
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        concat = np.hstack((frame, img_keypoints))
        cv.imshow('ORB Keypoints', concat)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()
