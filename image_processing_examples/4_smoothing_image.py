import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

while (1):
    # Take each frame
    _, frame = cap.read()

    kernel = np.ones((5, 5), np.float32) / 25
    gausian_kernel = cv.getGaussianKernel(5, 0)

    # dst = cv.filter2D(frame, -1, gausian_kernel)
    dst = cv.bilateralFilter(frame, 9, 75, 75)

    cv.imshow('frame', frame)
    cv.imshow('avarage_blur', dst)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()