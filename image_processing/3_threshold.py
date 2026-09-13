import cv2 as cv

cap = cv.VideoCapture(0)

while (1):

    # Take each frame
    _, frame = cap.read()

    frame = cv.GaussianBlur(frame, (5, 5), 0)
    # frame = cv.medianBlur(frame, 5)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    ret, th1 = cv.threshold(gray, 200, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    th2 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 3)

    cv.imshow('threashold', th1)
    cv.imshow('adaptive_threshold', th2)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()