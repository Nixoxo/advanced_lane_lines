import cv2
import matplotlib.pyplot as plt
import numpy as np
from .region_of_interest import region_of_interest as rog


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobel = None
    if orient == 'x':
        sobel = cv2.Sobel(hls, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(hls, cv2.CV_64F, 0, 1)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    binary_output = binary
    return binary_output


def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    magnitude = np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * magnitude / np.max(magnitude))
    # 5) Create a binary mask where mag thresholds are met
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    mag_binary = binary
    return mag_binary


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Calculate gradient direction
    # Apply threshold
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    gradient = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary = np.zeros_like(gradient)
    binary[(gradient >= thresh[0]) & (gradient <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    dir_binary = binary
    return dir_binary


def hue(H, thresh=(15, 100)):
    hue_binary = np.zeros_like(H)
    hue_binary[(H > thresh[0]) & (H < thresh[1])] = 1
    return hue_binary


def saturation(S, thresh=(90, 255)):
    saturation_binary = np.zeros_like(S)
    saturation_binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    return saturation_binary


def gradients(image):
    ksize = 3
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(60, 130))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(60, 130))

    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(50, 100))

    dir_binary = dir_threshold(image, sobel_kernel=27, thresh=(0.7, 1.3))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1) & (dir_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined


def extract(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    hue_image = hue(h_channel)
    saturation_image = saturation(s_channel)
    gradients_combined = gradients(img)
    combined = np.zeros_like(saturation_image)
    combined[(hue_image == 1) & (saturation_image ==1) | (gradients_combined==1)] = 1

    return combined
