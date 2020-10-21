# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 23:37:44 2020

@author: nguye
*Author : nguyenrobot
Copyright : nguyenrobot
https://github.com/nguyenrobot
https://www.nguyenrobot.com
"""

""" documentation & tutorials
https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

"""

"""
Load chessboards photos and find camera matrix and distorsion coefficient
Save camera calibration data into S7_cam_calibration.p for further usage
"""

import numpy
import cv2
import glob
import pickle
import os

# termination criteria
criteria    = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp        = numpy.zeros((9*6,3), numpy.float32)
objp[:,:2]  = numpy.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('images_calibration/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        print('true')
        print(fname)
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
        cv2.imshow('img', img)
        file_name = os.path.splitext(os.path.basename(fname))[0] + '_out.jpg'
        cv2.imwrite('images_calibration/output_chessboard/' + file_name, img)
        cv2.waitKey(500)
cv2.destroyAllWindows()

# Cam matrix and distortion coeff 
ret, camera_matrix, distortion_coefficient, rvecs, tvecs    = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
S7_cam_calibration                                          = {"camera_matrix" : camera_matrix,
                                                               "distortion_coefficient" : distortion_coefficient}

# Save cam matrix and distortion coeff into S7_cam_calibration.p for further usages
pickle.dump(S7_cam_calibration, open( "S7_cam_calibration.p", "wb" ))