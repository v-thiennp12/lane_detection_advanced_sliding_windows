# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:56:34 2020

@author: nguye
"""
import pickle
import cv2

class pipeline_camcalibration:
    def __init__(self, cam_calibration_file):
        # cam_calibration_file : S7_cam_calibration.p
        calibration_data                     = pickle.load(open(cam_calibration_file, "rb" ))
        self.camera_matrix                   = calibration_data['camera_matrix']
        self.distortion_coefficient          = calibration_data['distortion_coefficient']
        
    def apply_undistort(self, frame):
        H,  W                   = frame.shape[:2]
        new_camera_matrix, ZoI  = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_coefficient, (W,H), 1, (W,H))
        
        # undistort
        frame_undistort         = cv2.undistort(frame, self.camera_matrix, self.distortion_coefficient, None, new_camera_matrix)
        # crop the image
        xi, yi, wi, hi          = ZoI
        self.frame_undistort    = frame_undistort[yi : yi+hi, xi : xi+wi]
        return self.frame_undistort