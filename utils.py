import cv2
import numpy as np
import matplotlib.image as mpimg

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def calibrate_cam(calibration_imgs):
    img_pts = []
    obj_pts = []

    for img_name in calibration_imgs:
        img = mpimg.imread(img_name)
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        for board_size in [(9,6),(9,5),(8,6),(8,5)]:
            #print("---try:", board_size)
            ret, corners = cv2.findChessboardCorners(gray_img, board_size, None)
            #print(corners)
            if ret:
                #print("board_size:", board_size)
                obj_p = np.zeros((board_size[0]*board_size[1], 3),np.float32)
                obj_p[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2)
                img_pts.append(corners)
                obj_pts.append(obj_p)
                img = cv2.drawChessboardCorners(img, board_size, corners, ret)
                cv2.imwrite("./camera_cal_output/output_" + img_name[11:] , cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                #print(img_name, " completed")
                break
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, gray_img.shape[::-1], None, None)
    return (ret, mtx, dist)

def warp(img, img_name):
    imshape = img.shape
    bot_x = 0.13*imshape[1] # offset from bottom corner
    top_x = 0.04*imshape[1] # offset from centre of image
    top_y = 0.63*imshape[0]
    bot_y = imshape[0]
    vertices = np.array([[(bot_x,bot_y),((imshape[1]/2) - top_x, top_y), ((imshape[1]/2) + top_x, top_y), (imshape[1] - bot_x,bot_y)]], dtype=np.int32)

    x = [vertices[0][0][0], vertices[0][1][0], vertices[0][2][0], vertices[0][3][0]]
    y = [vertices[0][0][1], vertices[0][1][1], vertices[0][2][1], vertices[0][3][1]]
    #roi_lines = np.copy(img)*0
    #for i in range(0, len(x)-1):
    #    cv2.line(roi_lines,(x[i],y[i]),(x[i+1],y[i+1]),(0,0,255),3)
    #roi_img = weighted_img(img, roi_lines, α=0.8, β=1., γ=0.)
    #cv2.imwrite("./test_images_output/" + img_name[:-4] +"/02_" + img_name[:-4] + "_roi.jpg" , cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
    #print("x:\n", x)
    #print("________\ny:\n", y)
    src = np.float32([[x[0],y[0]], [x[1],y[1]], [x[2],y[2]], [x[3],y[3]]])
    #dst = np.float32([[x[0],y[0]], [x[0],y[1]], [x[3],y[2]], [x[3],y[3]]])
    dst = np.float32([[x[0],y[0]], [x[0],0], [x[3],0], [x[3],y[3]]])
    #print("________\nsrc:\n", src)
    #print("________\ndst:\n", dst)

    #roi_mask = np.zeros_like(img)
    #ignore_mask_color = (255,255,255)
    #cv2.fillPoly(roi_mask, vertices, ignore_mask_color)
    #masked_img = cv2.bitwise_and(img, roi_mask)
    #cv2.imwrite("./test_images_output/" + img_name[:-4] +"/03_" + img_name[:-4] + "_masked.jpg" , cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))

    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(img, M, (imshape[1],imshape[0]))
    cv2.imwrite("./test_images_output/" + img_name[:-4] +"/04_" + img_name[:-4] + "_warped.jpg" , cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))
    return warped_img
