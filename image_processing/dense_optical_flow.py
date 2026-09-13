import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[:, :, 1] = 255  # set saturation to maximum (darkness of the color)


def flow_vectors(frame, flow, step=16):
    for y in range(0, frame.shape[0], step):
        for x in range(0, frame.shape[1], step):
            fx, fy = flow[y, x]
            cv.arrowedLine(frame,
                           (x, y),
                           (int(x + fx), int(y + fy)),
                           (0, 255, 0), 2, tipLength=0.3)


while True:
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, cv.OPTFLOW_FARNEBACK_GAUSSIAN)
    mag, ang = cv.cartToPolar(flow[:, :, 0], flow[:, :, 1])
    hsv[:, :, 0] = ang * 180 / np.pi / 2
    hsv[:, :, 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # flow_vectors(frame2, flow)

    combined = np.hstack((frame2, bgr))
    cv.imshow('frame', combined)

    # overlay = cv.addWeighted(frame2, 0.6, bgr, 0.4, 0)
    # cv.imshow('Flow Overlay', overlay)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('files/img_1.png', frame2)
        cv.imwrite('opticalhsv.png', bgr)
    prvs = next

cv.destroyAllWindows()
