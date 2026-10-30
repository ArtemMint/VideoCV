import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image_path = "image_processing_examples/files/stereo_depth/"


img1 = cv.imread(image_path + 'img_1.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread(image_path + 'img_2.jpg', cv.IMREAD_GRAYSCALE)
if img1 is None or img2 is None:
    raise ValueError("Could not open or find the image!")

# Use this only for photos taken with stereo fixed camera

# stereo = cv.StereoBM.create(numDisparities=16, blockSize=15)
stereo = cv.StereoSGBM.create(minDisparity=0, numDisparities=14, blockSize=15)

disparity = stereo.compute(img1, img2)

plt.imshow(disparity,'gray')
plt.show()
