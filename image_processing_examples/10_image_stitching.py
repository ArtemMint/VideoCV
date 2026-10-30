import cv2 as cv
import numpy as np


#  Method 1: Use the built-in stitcher

def stitch_images(img1, img2):
    """
    Stitch two images together using OpenCV's built-in stitcher.
    :param img1: First image (numpy array)
    :param img2: Second image (numpy array)
    :return: Stitched panorama image (numpy array)
    """
    imgs: list[np.ndarray] = []
    for img_name in ["image_processing_examples/files/image_stiching/img_1.jpg", "image_processing_examples/files/image_stiching/img_2.jpg"]:
        img = cv.imread(cv.samples.findFile(img_name))
        if img is None:
            raise ValueError("Could not open or find the image!")
        imgs.append(img)
        cv.imshow('Image Stitching', img)
        cv.waitKey(1000)

    stitcher = cv.Stitcher.create(cv.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(imgs)

    if status != cv.Stitcher_OK:
        raise ValueError("Stitching failed with status " + str(status))

    cv.imshow('Image Stitching', pano)
    cv.waitKey()

def custom_stitching(img1, img2, result_cropped=True, save_to=None):
    """
    Stitch two images together using OpenCV's custom stitcher.
    :param img1: First image (numpy array)
    :param img2: Second image (numpy array)
    :return: Stitched panorama image (numpy array)
    """
    if img1 is None or img2 is None:
        raise ValueError("Could not open or find the image!")

    # Detect keypoints and compute descriptors using SIFT
    sift = cv.SIFT.create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        raise ValueError("Could not find SIFT descriptors in both images")

    bf = cv.BFMatcher(cv.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        raise ValueError("Not enough good matches found between the two images.")

    pts_1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # RANSAC to filter out outliers
    H, _ = cv.findHomography(pts_2, pts_1, cv.RANSAC, 5.0)

    # Calculater width and height of the stitched image
    width = img1.shape[1] + img2.shape[1]
    height = img2.shape[0]

    # Трансформуємо праве зображення у систему координат лівого
    result = cv.warpPerspective(img2, H, (width, height))

    # Add the left image to the result
    result[0:img1.shape[0], 0:img1.shape[1]] = img1

    # Cropping the result to remove black areas
    if result_cropped:
        gray_result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray_result, 1, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv.boundingRect(contours[0])
        result = result[0:height, 0:x+w]

    if save_to:
        cv.imwrite(save_to, result)
        print("Custom panorama saved as " + save_to)

    cv.imshow("Custom Stitching Result", result)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    img1 = cv.imread("image_processing_examples/files/image_stiching/img_1.jpg")
    img2 = cv.imread("image_processing_examples/files/image_stiching/img_2.jpg")

    custom_stitching(img1, img2, save_to="custom_panorama.jpg")
