import cv2
import numpy as np
import matplotlib.image as mpimg

def threshold_process(img, img_name, ksize=7, s_thresh=(180, 255), sx_thresh=(25, 255), sy_thresh=(15, 255), mag_thresh=(80, 255), dir_thresh=(0.8, 1.3)):
    img = np.copy(img)
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    # individual color channels:
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    l_channel = hls_img[:,:,1]
    s_channel = hls_img[:,:,2]
    
    gradx = abs_sobel_thresh(l_channel,'x', ksize, sx_thresh)
    grady = abs_sobel_thresh(l_channel,'y', ksize, sy_thresh)
    mag_binary = mag_threshold(gray_img, ksize, mag_thresh)
    dir_binary = dir_threshold(gray_img, ksize, dir_thresh)
    
    c_chan_binary = color_threshold(s_channel, s_thresh)
    
    # combine sobel masks
    gradxy = combine_masks(gradx, grady, "and")
    
    # combine sobel and color masks
    combined = combine_masks(gradxy, c_chan_binary, "or")
    
    return combined

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, sobel_thresh=(0, 255)):
    # Calculate directional gradient
    if orient == "x":
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)) # Take the derivative in x
    else:
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)) # Take the derivative in x
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Apply threshold
    gradient_binary = np.zeros_like(scaled_sobel)
    gradient_binary[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1
    return gradient_binary

def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Apply threshold
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(img, sobel_kernel=3, dir_thresh=(0, np.pi/2)):
    # Calculate gradient direction
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Apply threshold
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 1
    return dir_binary

def color_threshold(col_channel, color_thresh=(0,255)):
    col_binary = np.zeros_like(col_channel)
    col_binary[(col_channel >= color_thresh[0]) & (col_channel <= color_thresh[1])] = 1
    return col_binary

def combine_masks(m1, m2, logic):
    combined = np.zeros_like(m1)
    if logic == "and":
        combined[(m1 == 1) & (m2 == 1)] = 1
        return combined
    if logic == "or":
        combined[(m1 == 1) | (m2 == 1)] = 1
        return combined
    return None

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
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped_img = cv2.warpPerspective(img, M, (imshape[1],imshape[0]))
    cv2.imwrite("./test_images_output/" + img_name[:-4] +"/04_" + img_name[:-4] + "_warped.jpg" , cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))
    return warped_img, M, Minv
