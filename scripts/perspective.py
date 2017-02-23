import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

mid = 640
offset_distant = 130
offset_near = 200
# 500
# 680
src = np.float32([
    [mid - offset_distant, 450],
    [mid + offset_distant, 450],
    [mid - offset_near, 680],
    [mid + offset_near, 680]])

offset_test = 400
offset_delta = 300

dst = np.float32([
    [0, 100],
    [1200, 100],
    [400, 700],
    [800, 700]
])

def birds_view(img):

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    return warped

def get_matrix_for_brids_view_to_3d():
    return cv2.getPerspectiveTransform(dst, src)