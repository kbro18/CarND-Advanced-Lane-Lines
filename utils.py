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
