import cv2 as cv

cap = cv.VideoCapture(0)

while (1):
    # Take each frame
    _, frame = cap.read()
    frame = cv.GaussianBlur(frame, (5,5), 0)

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    laplacian = cv.Laplacian(frame, cv.CV_64F)
    sobelx = cv.Sobel(frame, cv.CV_64F, 1, 0, ksize=5)
    sobely = cv.Sobel(frame, cv.CV_64F, 0, 1, ksize=5)

    cv.imshow('frame', frame)
    cv.imshow('laplacian', laplacian)
    cv.imshow('sobelx', sobelx)
    cv.imshow('sobely', sobely)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
