from __future__ import print_function

import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

gray1 = cv.imread('files/img1.jpeg', cv.IMREAD_GRAYSCALE)
gray2 = cv.imread('files/img2.jpeg', cv.IMREAD_GRAYSCALE)

# Resize the images by half to speed up the processing
gray1 = cv.resize(gray1, None, fx=0.5, fy=0.5)
gray2 = cv.resize(gray2, None, fx=0.5, fy=0.5)

if gray1 is None and gray2 is None:
    print('Could not open or find the image!')
    exit(0)

#  Create an ORB detector with default values
orb = cv.ORB.create(nfeatures=500)

#  Detect points and compute descriptors
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# Match descriptors using the Brute-Force matcher
matcher = cv.DescriptorMatcher.create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)


img_matches = cv.drawMatches(
    gray1,
    keypoints1,
    gray2,
    keypoints2,
    good_matches,
    None,
    flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)


# -- Localize the object
obj = np.empty((len(good_matches), 2), dtype=np.float32)
scene = np.empty((len(good_matches), 2), dtype=np.float32)
for i in range(len(good_matches)):
    # -- Get the points from the good matches
    obj[i, 0] = keypoints1[good_matches[i].queryIdx].pt[0]
    obj[i, 1] = keypoints1[good_matches[i].queryIdx].pt[1]
    scene[i, 0] = keypoints2[good_matches[i].trainIdx].pt[0]
    scene[i, 1] = keypoints2[good_matches[i].trainIdx].pt[1]


print('Object points:')
print(obj)
print('Scene points:')
print(scene)

H, _ = cv.findHomography(obj, scene, cv.RANSAC)

print('Homography matrix:')
print(H)

# -- Get the corners from the image_1 ( the object to be "detected" )
obj_corners = np.empty((4, 1, 2), dtype=np.float32)
obj_corners[0, 0, 0] = 0
obj_corners[0, 0, 1] = 0
obj_corners[1, 0, 0] = gray1.shape[1]
obj_corners[1, 0, 1] = 0
obj_corners[2, 0, 0] = gray1.shape[1]
obj_corners[2, 0, 1] = gray1.shape[0]
obj_corners[3, 0, 0] = 0
obj_corners[3, 0, 1] = gray1.shape[0]

scene_corners = cv.perspectiveTransform(obj_corners, H)

cv.polylines(img_matches, [np.int32(scene_corners) + (gray1.shape[1], 0)], True, (0, 255, 0), 1, cv.LINE_AA)

cv.imshow('Good Matches & Object detection', img_matches)

cv.waitKey()
