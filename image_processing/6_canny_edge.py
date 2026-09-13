from cgitb import grey

import cv2 as cv

cap = cv.VideoCapture(0)

cv.namedWindow('Canny Edge Detection')

cv.createTrackbar('minVal', 'Canny Edge Detection', 0, 255, lambda x: None)
cv.createTrackbar('maxVal', 'Canny Edge Detection', 0, 255, lambda x: None)

while (1):
    # Take each frame
    _, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    # blurred = cv.bilateralFilter(gray, 9, 75, 75)


    t1 = cv.getTrackbarPos('minVal', 'Canny Edge Detection')
    t2 = cv.getTrackbarPos('maxVal', 'Canny Edge Detection')

    res = cv.Canny(blurred, t1, t2)

    cv.imshow('frame', frame)
    cv.imshow('Canny Edge Detection', res)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
