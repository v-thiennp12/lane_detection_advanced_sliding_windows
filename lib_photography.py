# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 22:49:10 2020

@author: nguye

Author : nguyenrobot
Copyright : nguyenrobot
https://github.com/nguyenrobot
https://www.nguyenrobot.com
"""
import numpy
import math
import cv2

def frame_HLS_balance_exposure(frame_HLS):
    """Applies the exposure correction"""
    exposure_correction_ratio = 1
    frame_L                 = frame_HLS[:,:,1]
    mean_L                  = numpy.mean(frame_L)
    max_L                   = numpy.max(frame_L)
    min_L                   = numpy.min(frame_L)
    if ~(max_L == min_L):
        frame_L             = frame_L - numpy.abs((frame_L - mean_L)/(max_L - min_L))*(frame_L - mean_L)*exposure_correction_ratio
    frame_HLS[:,:,1]        = numpy.uint8(frame_L)
    return frame_HLS

def frame_HLS_balance_white(frame_HLS):
    ratio_S = 1
    ratio_L = 1
    """apply white balance correction based on a reference pixel""" 
    frame_S                 = frame_HLS[:,:,2]
    frame_L_negative        = 255 - frame_HLS[:,:,1]    
    frame_S_Ln              = ratio_S*frame_S + ratio_L*frame_L_negative
    
    mask_min_S_Ln           = numpy.zeros_like(frame_HLS[:,:,2], dtype=bool)
    mask_max_Ln             = numpy.zeros_like(frame_HLS[:,:,1], dtype=bool)
    
    mask_min_S_Ln[(frame_S_Ln == numpy.min(frame_S_Ln))]                                = True
    mask_max_Ln[frame_L_negative == numpy.min(frame_L_negative[mask_min_S_Ln])]         = True

    min_S                   = frame_S[mask_min_S_Ln & mask_max_Ln][0]
    
    frame_S                 = frame_S - min_S
    frame_S[frame_S < 0]    = 0
    
    frame_HLS[:,:,2]        = frame_S
    return frame_HLS

def frame_H_keep_range(frame_HLS, thd_H):
    """keep pixel inside a H-range"""
    frame_H             = frame_HLS[:,:,0]
    color_selection_ind = numpy.zeros_like(frame_H, dtype=bool)  
    result = numpy.copy(frame_HLS)
    
    # Define selection by color / below the threshold
    for thd_H_i in thd_H:
        color_selection_ind = ((frame_H >= thd_H_i[0]) & (frame_H <= thd_H_i[1])) | color_selection_ind

    result[~color_selection_ind] = [0, 0, 0]
    return result