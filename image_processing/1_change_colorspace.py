import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while (1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_orange = np.array([0, 50, 50])
    upper_orange = np.array([10, 255, 255])
    lower_green = np.array([50, 50, 50])
    upper_green = np.array([70, 255, 255])


    # Threshold the HSV image to get only blue colors
    mask_orange = cv.inRange(hsv, lower_orange, upper_orange)
    mask_green = cv.inRange(hsv, lower_green, upper_green)

    mask = cv.bitwise_or(mask_orange, mask_green)

    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('res', res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()