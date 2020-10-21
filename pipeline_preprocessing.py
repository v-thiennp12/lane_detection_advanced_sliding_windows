# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 16:19:31 2020

@author: nguye
Author : nguyenrobot
Copyright : nguyenrobot
https://github.com/nguyenrobot
https://www.nguyenrobot.com
"""
# import libraries
from lib_frame import *
from lib_photography import *

class pipeline_preprocessing:
    def __init__(self, frame_RGB, param):
        self.thd_S_mag          = param['thd_S_mag']
        self.thd_S_arg          = param['thd_S_arg']
        self.thd_S_x            = param['thd_S_x']
        
        self.thd_L_mag          = param['thd_L_mag']
        self.thd_L_arg          = param['thd_L_arg']
        self.thd_L_y            = param['thd_L_y']
        
        self.thd_L_shadow       = param['thd_shadow_L']
        self.thd_S_shadow       = param['thd_shadow_S']
        self.thd_L_highlight    = param['thd_highlight_L']
        self.thd_S_highlight    = param['thd_highlight_S']
        
        self.frame_RGB          = frame_RGB
        self.frame_HLS          = frameRGB2HLS(frame_RGB)
        
    def apply_white_balance(self):
        # exposure and white balance
        self.frame_HLS      = frame_HLS_balance_white(self.frame_HLS)

    def apply_exposure_balance(self):
        # exposure and white balance
        self.frame_HLS      = frame_HLS_balance_exposure(self.frame_HLS )
    
    def apply_highlight_remove(self):
        # Highlight removing
        L_cond = self.frame_HLS[:,:,1] > self.thd_L_highlight
        S_cond = self.frame_HLS[:,:,2] < self.thd_S_shadow        
        """we wont keep pixel with low saturation and high lightness"""
        LS_cond                        = (L_cond & S_cond)       
        self.frame_HLS[LS_cond,:]      = [0, 0, 0]
    
    def apply_shadow_remove(self):
        # Shadows removing        
        L_cond = self.frame_HLS[:,:,1]  < self.thd_L_shadow
        S_cond = self.frame_HLS[:,:,2]  > self.thd_S_shadow        
        """we wont keep pixel with high saturation and low lightness"""
        LS_cond                        = (L_cond & S_cond)       
        self.frame_HLS[LS_cond,:]      = [0, 0, 0]
        
    def apply_sobel(self):
        # 1st degree sobel operations
        # self.sobel_H_x, self.sobel_H_y, self.sobel_H_mag, self.sobel_H_arg = frame_sobel(self.frame_HLS[:,:,0])
        self.sobel_L_x, self.sobel_L_y, self.sobel_L_mag, self.sobel_L_arg = frame_sobel(self.frame_HLS[:,:,1])
        self.sobel_S_x, self.sobel_S_y, self.sobel_S_mag, self.sobel_S_arg = frame_sobel(self.frame_HLS[:,:,2])    
    
    def apply_sobel2(self):
        # 2nd degree sobel operations
        # self.sobel2_H_x, self.sobel2_H_y, self.sobel2_H_mag, self.sobel2_H_arg = frame_sobel(self.sobel_H_mag)
        self.sobel2_L_x, self.sobel2_L_y, self.sobel2_L_mag, self.sobel2_L_arg = frame_sobel(self.sobel_L_x)
        self.sobel2_S_x, self.sobel2_S_y, self.sobel2_S_mag, self.sobel2_S_arg = frame_sobel(self.sobel_S_x)
               
    def apply_sobel_mask(self):
        """apply filter on sobel magnitude of L and S chanels"""
        L_mag_cond                              = numpy.abs(127 - self.sobel_L_mag.astype(numpy.int)) > self.thd_L_mag
        S_mag_cond                              = numpy.abs(127 - self.sobel_S_mag.astype(numpy.int)) > self.thd_S_mag        
        LS_mag_cond                             = L_mag_cond | S_mag_cond
        self.mask_LS_mag                        = numpy.zeros_like(LS_mag_cond, dtype=numpy.uint8)
        self.mask_LS_mag[LS_mag_cond]           = 1
        
        """we keep pixel with sobel_S_arg far from scaled-127 = unscaled-90deg"""       
        Larg_cond                               = numpy.abs(127 - self.sobel_L_arg.astype(numpy.int)) > self.thd_L_arg
        Sarg_cond                               = numpy.abs(127 - self.sobel_L_arg.astype(numpy.int)) > self.thd_S_arg
        LS_arg_cond                             = Larg_cond & Sarg_cond
        self.mask_LS_Larg                       = numpy.zeros_like(LS_arg_cond, dtype=numpy.uint8)
        self.mask_LS_Larg[LS_arg_cond]          = 1
        
        """we remove pixels with high Ly """       
        Ly_cond                                 = numpy.abs(127 - self.sobel_L_y.astype(numpy.int)) < self.thd_L_y
        self.mask_Ly                            = numpy.zeros_like(LS_arg_cond, dtype=numpy.uint8)
        self.mask_Ly[Ly_cond]                   = 1            
        
    def apply_pipeline(self):
        
        self.apply_exposure_balance()
        self.apply_white_balance()
        self.apply_sobel()
        self.apply_sobel_mask()
        
        # create filter mask
        mask_note = self.mask_LS_mag + self.mask_LS_Larg + self.mask_Ly
        self.frame_HLS[~(mask_note >= 3)] = [0, 0, 0]

        # remove highlights and shadows
        self.apply_highlight_remove()
        self.apply_shadow_remove()