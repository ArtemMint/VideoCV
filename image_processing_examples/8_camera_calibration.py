import numpy as np
import cv2 as cv
import glob

CHESSBOARD = (7,3)
 
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CHESSBOARD[0]*CHESSBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARD[0],0:CHESSBOARD[1]].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Use chessboard images from the camera_calibration folder
images = glob.glob('image_processing_examples/files/camera_calibration/*.jpg')

print(images)

gray = None
 
for fname in images:
    img = cv.imread(fname)
    if img is None:
        continue
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, CHESSBOARD, None, cv.CALIB_CB_ADAPTIVE_THRESH)
 
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
 
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
 
        # Draw and display the corners
        cv.drawChessboardCorners(img, CHESSBOARD, corners2, ret)
        cv.imshow('Chessboard Detection', img)
        cv.waitKey(500)
 
cv.destroyAllWindows()

if gray is None or not objpoints or not imgpoints:
    raise ValueError("No valid chessboard images were found for calibration.")


h, w = gray.shape[:2]

# Perform camera calibration by
# passing the value of above found out 3D points (threedpoints)
# and its corresponding pixel coordinates of the
# detected corners (twodpoints)
ret, matrix, distortion, r_vecs, t_vecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n Matrix:")
print(matrix)

print("\n Distortion coefficient:")
print(distortion)

print("\n Rotation Vectors:")
print(r_vecs)

print("\n Translation Vectors:")
print(t_vecs)

# Load the test image
test_image_path = "image_processing_examples/files/camera_calibration/frame_2.jpg"
img_test = cv.imread(test_image_path)

if img_test is None:
    raise ValueError("Could not open or find the image!")

h, w = img_test.shape[:2]

# Matrix optimization
newcameramtx, roi = cv.getOptimalNewCameraMatrix(matrix, distortion, (w, h), 1, (w, h))

# Undistort the image
dst = cv.undistort(img_test, matrix, distortion, None, newcameramtx)

# Crop the image if needed
# x, y, w_box, h_box = roi
# dst = dst[y:y+h_box, x:x+w_box]
cv.imwrite('image_processing_examples/files/camera_calibration/calibrated_result.png', dst)
