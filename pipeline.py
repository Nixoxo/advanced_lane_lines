import matplotlib.pyplot as plt
import scripts.region_of_interest as rog
import scripts.extraction as ext
import scripts.lines as lines
import scripts.perspective as per
import scripts.sliding as sli
import scripts.curvature as cur
import scripts.distortion as dis
import numpy as np
import cv2
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import glob
import PIL.Image

mtx, dist = dis.calibrate()
def pipeline(img):
    orig_img = cv2.undistort(img, mtx, dist)

    gradients_image = ext.extract(orig_img)
    region_image = rog.region_of_interest(gradients_image)

    transformed = region_image[:,:,0].flatten().reshape(-1,1280)

    birds_view_image = per.birds_view(transformed)

    #img = birds_view_image

    Minv = per.get_matrix_for_brids_view_to_3d()
    result, birds_view_image, ploty, left_fit, right_fit, leftx, rightx, lefty, righty = sli.slide(birds_view_image, Minv, undist=orig_img)

    left, right, center = cur.curvature(ploty, left_fit, right_fit, leftx, rightx, lefty, righty)

    final = result
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(final, 'Left curvature: ' + str(int(left)) + 'm', (55,55), font, 1, (255, 255, 255),2)
    cv2.putText(final, 'Right curvature: ' + str(int(right)) + 'm', (55, 90), font, 1, (255, 255, 255), 2)
    cv2.putText(final, 'Center offset: ' + str(round(center, 2)) + 'm', (55, 125), font, 1, (255, 255, 255), 2)

    xsize = 1260
    ysize = 1000
    xsplit = 420
    ysplit = 300

    # Final image
    img_out = np.zeros((ysize, xsize, 3), dtype=np.uint8)
    img_out[ysplit:ysize, 0:xsize, 0:3] = cv2.resize(final,(xsize, 700))

    # Left
    gradients_rgb = np.empty((720, 1280, 3), dtype=np.uint8)
    gradients_rgb[:,:,0] = gradients_image[:,:] * 255
    gradients_rgb[:,:,1] = gradients_image[:,:] * 255
    gradients_rgb[:,:,2] = gradients_image[:,:] * 255
    gradients_resized = cv2.resize(gradients_rgb, (xsplit, ysplit))
    img_out[0:ysplit, 0:xsplit, 0:3] = gradients_resized

    # Mid
    region_image_rbg_resized = cv2.resize(region_image, (xsplit, ysplit))
    img_out[0:ysplit, xsplit:xsplit * 2, 0:3] = region_image_rbg_resized

    # Right
    birds_view_resized = cv2.resize(birds_view_image, (xsplit, ysplit))
    img_out[0:ysplit, 2 * xsplit:xsize, 0:3] = birds_view_resized

    # Horizontal Split
    cv2.line(img_out, (0, ysplit), (xsize, ysplit), [255,0,0], 2)
    # Vertical Split 1
    cv2.line(img_out, (xsplit, 0), (xsplit, ysplit), [255, 0, 0], 2)
    # Vertical Split 2
    cv2.line(img_out, (xsplit*2, 0), (xsplit*2, ysplit), [255, 0, 0], 2)
    return img_out

last_image = None

import scipy.misc

index = 0
def process_image(img):
    try:
        last_image = pipeline(img)
        return last_image
    except Exception as e:
        global index
        scipy.misc.imsave('test_images/broken/broken'+str(index)+'.jpg', img, 'JPEG')
        index = index + 1
        #return img

def video():
    project_output = 'project_video_output.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(project_output, audio=False)

if __name__ == '__main__':
    #util()
    video()
    #test_file = 'test_images/straight_lines1.jpg'
    test_file = 'test_images/test4.jpg'
    test_file = 'test_images/broken/broken10.jpg'
    #reading in an image
    """
    for img in glob.glob("test_images/*.jpg"):

        plt.figure()
        plt.title(img)
        plt.imshow(pipeline(mpimg.imread(img)))
    plt.show()
    """
    image = mpimg.imread(test_file)
    plt.imshow(image)
    fin = pipeline(image)
    plt.imshow(fin)
    plt.show()
