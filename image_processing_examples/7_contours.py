import cv2 as cv
import numpy as np


def main():
    cap = cv.VideoCapture(0)

    while (1):
        # Take each frame
        _, frame = cap.read()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            perimeter = cv.arcLength(cnt, True)
            epsilon = 0.1 * perimeter
            approx = cv.approxPolyDP(cnt, epsilon, True)
            res = cv.drawContours(frame.copy(), [approx], -1 , (0, 255, 0), 3)

        cv.imshow('frame', frame)
        cv.imshow('Draw Contours', res)

        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break

    cv.destroyAllWindows()


def procees_picture():
    img = cv.imread('../result.png')

    # Use CLAHE for better contrast
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = 255 - gray

    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        perimeter = cv.arcLength(cnt, True)
        epsilon = 0.04 * perimeter
        approx = cv.approxPolyDP(cnt, epsilon, True)
        if 5000 > cv.contourArea(approx) > 300:
            # x, y, w, h = cv.boundingRect(approx)
            # cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            #  Get the minimum area rectangle for the contour
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.int32(box)
            cv.drawContours(img, [box], 0, (255, 0, 0), 2)

            # Get aspect ratio
            # aspect_ratio = w / h
            # cv.putText(img, f'AR: {aspect_ratio:.2f}', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)



    cv.imshow('Draw Contours', img)
    cv.imshow('Thresh', thresh)

    k = cv.waitKey(0) & 0xFF
    if k == 27:
        cv.destroyAllWindows()


if __name__ == '__main__':
    # procees_picture()
    main()
