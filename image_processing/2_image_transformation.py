import numpy as np
import cv2 as cv

img = cv.imread('../result.png')
assert img is not None, "file could not be read, check with os.path.exists()"

# res = cv.resize(img,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)
#OR
# height, width = img.shape[:2]
# res = cv.resize(img,(2*width, 2*height), interpolation = cv.INTER_CUBIC)

rows, cols = img.shape[:2]

# M = np.float32([
#     [.2, 0, 100],
#     [0, 1, 50]
# ])  # 2x3 transformation matrix to shift the image 100 pixels to the right and 50 pixels down


# M = cv.getRotationMatrix2D(
#     ((cols-1)/2.0,(rows-1)/2.0),
#     90,
#     1)


pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv.getAffineTransform(pts1,pts2)

print(M)

res = cv.warpAffine(img, M, (cols, rows))

cv.imshow('res', res)
cv.waitKey(0)
cv.destroyAllWindows()