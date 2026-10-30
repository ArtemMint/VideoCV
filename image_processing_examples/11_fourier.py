import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image_processing_examples/files/fourier/img_1.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Could not open or find the image!")

# 2. Optimizing the size of the image for DFT (Discrete Fourier Transform)
rows, cols = img.shape
nrows = cv2.getOptimalDFTSize(rows)
ncols = cv2.getOptimalDFTSize(cols)
right = ncols - cols
bottom = nrows - rows
nimg = cv2.copyMakeBorder(img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=0)

# 3. DFT calculation
dft = cv2.dft(np.float32(nimg), flags=cv2.DFT_COMPLEX_OUTPUT)

# 4. Shift the zero-frequency component to the center of the spectrum
dft_shift = np.fft.fftshift(dft)

# 5. Calculate the magnitude spectrum
magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

# 6. Scale the magnitude spectrum for better visualization
magnitude_spectrum = 20 * np.log(magnitude_spectrum + 1)

# 7. Display the original image and the magnitude spectrum
plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

# 8. Display the phase spectrum
mask = np.zeros((nrows, ncols, 2), np.uint8)
offset = 100
mask[rows//2 - offset:rows//2 + offset, cols//2 - offset:cols//2 + offset] = 1
dft_shift_mask = dft_shift * mask

# 9. Calculate the magnitude spectrum
magnitude_spectrum = cv2.magnitude(dft_shift_mask[:, :, 0], dft_shift_mask[:, :, 1])

# 10. Scale the magnitude spectrum for better visualization
magnitude_spectrum = 20 * np.log(magnitude_spectrum + 1)

plt.subplot(224), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Low Pass Filtered Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

dft_ishift = np.fft.ifftshift(dft_shift_mask)
img_back_lp = cv2.idft(dft_ishift)
img_back_lp = cv2.magnitude(img_back_lp[:, :, 0], img_back_lp[:, :, 1])

plt.subplot(223), plt.imshow(img_back_lp, cmap='gray')
plt.title('Low Pass Filtered Image'), plt.xticks([]), plt.yticks([])

plt.show()
