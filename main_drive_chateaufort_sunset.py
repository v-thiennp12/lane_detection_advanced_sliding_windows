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

param_preprocessing     = { 'thd_highlight': 245,
                            'thd_shadow': 10,
                            'ratio_S': 1,
                            'ratio_L': 1,
                            'thd_sobel_S': 25,
                            'thd_sobel2_L': 100,
                            'thd_L_mag': 0,
                            'thd_L_x': 15,
                            'thd_L_arg_min': 75,
                            'thd_L_arg_max': 255}
points_ZoI              = numpy.float32([(200, 350), (-380, 720), (1380, 580), (1080, 350)]) # cause of windshield-wiper, ZoI is not sysmetric
points_1st_tranform     = {'source_points':         numpy.float32([(580, 360), (100, 720), (1280, 720), (630, 360)]),\
                           'destination_points':    numpy.float32([(600, 0), (600, 720), (900, 720), (900, 0)])}
param_curve_class       = {'num_of_windows': 15,
                            'histogram_width': 150,
                            'histogram_vertical_ratio': 1, #10.6 by defaut
                            'margin_x': 100,
                            'min_pixel_inside': 25,
                            'min_pixel_confindex': 150,
                            'xm_by_pixel': 3.7/700,
                            'ym_by_pixel': 30/720,
                            'thd_confindex': 30,
                            'min_pixel_bold': 250}
thd_gray                = 10
   
# single frame processing
frame_filepath          = 'images_test/frame_sunsetdrive_1_838.jpg'
frame_filename          = 'frame_sunsetdrive_1_838.jpg'
frame_RGB_origin        = frame_RGB_read(frame_filepath)    
            
lanefinding                             = pipeline_lanefinding(param_preprocessing, points_ZoI, thd_gray, points_1st_tranform, param_curve_class)
frame_mix_RGB_origin_Lane_verbose       = lanefinding.apply_pipeline(frame_RGB_origin)

frame_show(frame_mix_RGB_origin_Lane_verbose, title = 'frame_mix_RGB_origin_Lane_verbose')
frame_RGB_save(frame_mix_RGB_origin_Lane_verbose, 'images_output/frame_mix_RGB_origin_frame_sunsetdrive_1_838.jpg')

# video processing
os.listdir("videos_test/")
video_load              = VideoFileClip("videos_test/video_chateaufort_sunset_drive.mov")

#Process loaded video
video_processed         = video_load.fl_image(lanefinding.apply_pipeline) #NOTE: this function expects color images!!

#Write output video
video_output            = 'videos_output/video_output_chateaufort_sunset_drive_new__.mp4'
# %time
video_processed.write_videofile(video_output, audio=True, logger='bar')

# Display processed video
HTML("""
<video width="1280" height="720" controls>
  <source src="{0}">
</video>
""".format(video_processed))