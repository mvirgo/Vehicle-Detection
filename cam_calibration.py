import numpy as np
import cv2
import matplotlib.image as mpimg
import glob
import pickle

def camera_calibration():
    """ This function uses Chessboard images calibration images
    in order to correct the distortion of the given camera.
    Note that the images used for the pickle file in my repository
    come from Udacity's Advanced Lane Line detection project repository here:
    https://github.com/udacity/CarND-Advanced-Lane-Lines
    """

    # Load in the chessboard calibration images to a list
    cal_image_loc = glob.glob('camera_cal/*.jpg')
    calibration_images = []

    for fname in cal_image_loc:
        img = mpimg.imread(fname)
        calibration_images.append(img)

    # Prepare object points
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays for later storing object points and image points
    objpoints = []
    imgpoints = []

    # Iterate through images for their points
    for image in calibration_images:
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(image, (9, 6), corners, ret)

    # Returns camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    cam_cal_info = {'ret': ret, 'mtx': mtx, 'dist': dist, 'rvecs': rvecs, 'tvecs': tvecs}

    pickle.dump(cam_cal_info,open('cam_cal_info.p', "wb"))


camera_calibration()
