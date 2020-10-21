# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:58:05 2020

@author: 
Author : nguyenrobot
Copyright : nguyenrobot
https://github.com/nguyenrobot
https://www.nguyenrobot.com
"""
import cv2
import numpy

class birdseye:
    def __init__(self, points):
        self.source_points         = points['source_points']
        self.destination_points    = points['destination_points']

        self.birdeye_matrix        = cv2.getPerspectiveTransform(self.source_points, self.destination_points)
        self.inv_birdeye_matrix    = cv2.getPerspectiveTransform(self.destination_points, self.source_points)
    
    def apply_skyview(self, frame_camview):
        """apply birdseye view transform"""
        shape           = (frame_camview.shape[1], frame_camview.shape[0])
        frame_skyview   = cv2.warpPerspective(frame_camview, self.birdeye_matrix, (shape)) #, flags = cv2.INTER_LINEAR
        self.frame_skyview = frame_skyview
        return frame_skyview

    def apply_vehicleview(self, frame_skyview):
        """apply reversed birdseye view transform to get back to camera view"""
        shape           = (frame_skyview.shape[1], frame_skyview.shape[0])
        frame_camview   = cv2.warpPerspective(frame_skyview, self.inv_birdeye_matrix, shape)
        self.frame_camview = frame_camview
        return frame_camview