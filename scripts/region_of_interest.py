import matplotlib.pyplot as plt
import numpy as np
import cv2
file = '../test_images/test1.jpg'

def ignore_area(img):
    imshape = img.shape
    empty_rgb = np.empty((720, 1280, 3), dtype=np.uint8)
    ignore_vertices_1 = np.array([[(imshape[1] * 3 / 10 - 50, imshape[0]),
                                   (imshape[1] * 1 / 2 , imshape[0] * 8 / 10),
                                   (imshape[1] * 1 / 2 , imshape[0] * 8 / 10),
                                   (imshape[1] * 7 / 10 + 50, imshape[0])]], dtype=np.int32)

    ignore_vertices_2 = np.array([[(imshape[1] * 3 / 10, imshape[0]),
                                   (imshape[1] * 1 / 2 - 50, imshape[0] * 6 / 10 + 50),
                                   (imshape[1] * 1 / 2 + 50, imshape[0] * 6 / 10 + 50),
                                   (imshape[1] * 7 / 10, imshape[0])]], dtype=np.int32)

    low_triangle = np.empty((720, 1280, 3), dtype=np.uint8)
    low_triangle = cv2.fillPoly(low_triangle, ignore_vertices_1, [255, 255, 255])
    high_triangle = np.empty((720, 1280, 3), dtype=np.uint8)
    high_triangle = cv2.fillPoly(high_triangle, ignore_vertices_2, [255,255,255])
    triangle = cv2.bitwise_or(low_triangle, high_triangle)
    some_max = np.max(triangle)
    return triangle


def region_of_interest(img):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    region_image_rgb = np.empty((720, 1280, 3), dtype=np.uint8)
    region_image_rgb[:, :, 0] = img[:, :] * 255
    region_image_rgb[:, :, 1] = img[:, :] * 255
    region_image_rgb[:, :, 2] = img[:, :] * 255
    #
    imshape = img.shape
    y_low = imshape[0]
    left_low = (imshape[1] * 1 / 10 - 50, y_low)
    left_high = (imshape[1] * 1 / 2 - 110, imshape[0] - imshape[0] * 1 / 2 + 75)
    right_high = (imshape[1] * 1 / 2 + 110, imshape[0] - imshape[0] * 1 / 2 + 75)
    right_low = (imshape[1] * 9 / 10 + 50, y_low)
    vertices = np.array([[left_low,
                          left_high,
                          right_high,
                          right_low]], dtype=np.int32)


    mask = np.empty((720, 1280, 3), dtype=np.uint8)

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, (255,255,255))
    mask2 = ignore_area(img)
    mask3 = cv2.bitwise_xor(mask, mask2)
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(region_image_rgb, mask3)
    return masked_image