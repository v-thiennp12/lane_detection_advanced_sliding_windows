# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 00:03:16 2020

@author: nguye
Author : nguyenrobot
Copyright : nguyenrobot
https://github.com/nguyenrobot
https://www.nguyenrobot.com
"""

import os
from moviepy.editor import  VideoFileClip
from IPython.display import HTML
import math

# import libraries
from lib_frame import *
from lib_photography import *
from lib_perspective import birdseye
from lib_curve_slider import curve_slider
from pipeline_preprocessing import *
from pipeline_lanefinding import *

param_preprocessing     = { 'thd_highlight': 250,
                            'thd_shadow': 15,
                            'ratio_S': 1,
                            'ratio_L': 1,
                            'thd_sobel_S': 25, #40
                            'thd_sobel2_L': 75, #130
                            'thd_L_mag': 5,
                            'thd_L_x': 25,
                            'thd_L_arg_min': 75,
                            'thd_L_arg_max': 200}
points_ZoI              = numpy.float32([(440, 390), (0,400), (0, 720), (1280, 720), (1280,400), (840, 390)])   
points_1st_tranform     = {'source_points': numpy.float32([(550, 420), (0, 720), (1050, 720), (703, 420)]),
                       'destination_points': numpy.float32([(450, 0), (450, 720), (830, 720), (830, 0)])} #
param_curve_class       = {'num_of_windows': 15,
                            'histogram_width': 150,# 200
                            'histogram_vertical_ratio': 1, #0.6 by defaut
                            'margin_x': 100,
                            'min_pixel_inside': 75,
                            'min_pixel_confindex': 150,
                            'xm_by_pixel': 3.7/700,
                            'ym_by_pixel': 30/720,
                            'thd_confindex': 30,
                            'min_pixel_bold': 1200}
thd_gray                = 5

cam_calibration_file    = 'S7_cam_calibration.p'

# single frame processing
frame_filepath                          = 'images_test/frame_highway_A5_normal_12x370.jpg'
frame_filename                          = 'frame_highway_A5_normal_12x370.jpg'
frame_RGB_origin                        = frame_RGB_read(frame_filepath)            
lanefinding                             = pipeline_lanefinding(param_preprocessing, points_ZoI, thd_gray, points_1st_tranform, param_curve_class, cam_calibration_file)
frame_mix_RGB_origin_Lane_verbose       = lanefinding.apply_pipeline(frame_RGB_origin)

frame_show(frame_mix_RGB_origin_Lane_verbose, title = 'frame_mix_RGB_origin_Lane_verbose')
frame_RGB_save(frame_mix_RGB_origin_Lane_verbose, 'images_output/frame_highway_A5_normal_12x370_output.jpg')

# # video processing
os.listdir("videos_test/")
video_load              = VideoFileClip("videos_test/highway_A5_normal.mov")

#Process loaded video
video_processed         = video_load.fl_image(lanefinding.apply_pipeline) #NOTE: this function expects color images!!

#Write output video
video_output            = 'videos_output/highway_A5_normal_output.mp4'
# %time
video_processed.write_videofile(video_output, audio=True, logger='bar')

# # Display processed video
# HTML("""
# <video width="1280" height="720" controls>
#   <source src="{0}">
# </video>
# """.format(video_processed))