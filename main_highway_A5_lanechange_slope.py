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

cam_calibration_file    = 'S7_cam_calibration.p'                                    # camera calibration file
param_preprocessing_1st = { 'thd_highlight_L':                  255,                # mid L         # low-pass filter on H-chanel
                            'thd_highlight_S':                  0,                  # low S         # low-pass filter on H-chanel
                            'thd_shadow_L':                     30,                 # low L
                            'thd_shadow_S':                     50,                 # high S       
                            'thd_S_mag':                        25,                 # high S_mag 25
                            'thd_S_arg':                        0,                  # high S_arg 25
                            'thd_S_x':                          0,                  # high-pass filter on sobel of L-chanel in direction x, by defaut 35
                            'thd_L_mag':                        20,                 # high L_mag 25   # high-pass filter on magnitude of sobel of L-chanel, by defaut 5
                            'thd_L_arg':                        0,                      
                            'thd_L_y':                          75}
param_preprocessing_2nd = { 'thd_highlight_L':                  255,                #  55mid L     # low-pass filter on H-chanel
                            'thd_highlight_S':                  0,                  #  25low S      # low-pass filter on H-chanel
                            'thd_shadow_L':                     30,                 #  25low L
                            'thd_shadow_S':                     50,                 #  high S
                            'thd_S_mag':                        25,                 # high S_mag 25
                            'thd_S_arg':                        0,                  # high S_arg 25
                            'thd_S_x':                          0,                  # high-pass filter on sobel of L-chanel in direction x, by defaut 35
                            'thd_L_mag':                        20,                 # high-pass filter on magnitude of sobel of L-chanel, by defaut 5
                            'thd_L_arg':                        75,                 # 127          
                            'thd_L_y':                          100}                # low-pass filter on scaled argument of sobel vector of L-chanel, by defaut 200 == 70 degree
points_ZoI              = numpy.float32([(480, 390),         # top-right of 
                                         (0,400),           
                                         (0, 720),           # bottom-left of ZoI
                                         (1280, 720),        # bottom-right of ZoI
                                         (1280,400),
                                         (840, 390)])       # top-left of ZoI
points_birdeye_transform = {'source_points':         numpy.float32([(600, 420),        # source-points top-right
                                                                   (0, 720),           # bottom-left
                                                                   (1050, 720),        # bottom-right,
                                                                   (695, 420)]),       # top-left
                           'destination_points':    numpy.float32([(400, 0),           # destination-points top-right,
                                                                   (400, 720),         # bottom-left, 
                                                                   (720, 720),         # bottom-right, 
                                                                   (720, 0)])}         # top-left
param_curve_class       = {'num_of_windows':                    15,                 # number of windows used for curve-sliding
                            'histogram_width':                  75,                 # width of histogram for a curve, by defaut 200
                            'histogram_seed' :                  64,                 # histogram seed used for vertical slices of histogram
                            'histogram_vertical_ratio_end':     1,                  # height ratio used for histogram, 0.6 by defaut to end in ~2/3 from the top
                            'histogram_vertical_ratio_start':   0,                  # height ratio used for histogram, 0 by defaut to start from the top
                            'histogram_ratio_localmax':         1,                  # ratio of second max comparing to 1st, to be get picked
                            'offset_cam' :                      125,                # offset from vehicle's centerline to camera's centerline (positive if cam on vehicle's left)
                            'm_vehicle_width':                  1.8,                # [m] body width of ford focus 1999
                            'm_look_ahead':                     10,                 # [m] look ahead distance, used for curvature and space-left-in-lane estimation
                            'margin_x':                         50,                 # width of windows used for curve-sliding, by dafaut 100
                            'min_pixel_inside':                 50,                 # min pixel numnber inside a window
                            'max_pixel_inside':                 4500,               # max pixel numnber inside a window to be considered as a line
                            'max_width_not_a_line':             115,                # max width of a line, top eliminate much dispersed detectedpixels
                            'min_pixel_confindex':              50,                 # min pixel numnber inside a window, used for confindex evaluation
                            'xm_by_pixel':                      3.5/376,            # meter by pixel in x direction (horizontal)
                            'ym_by_pixel':                      2/35,               # meter by pixel in y direction (vertical)
                            'thd_confindex':                    33,                 # min confindex of a line
                            'min_pixel_bold':                   150,                # min pixel inside a window of bold-type line (bold : barrier or road-edge)
                            'min_pixel_doubleline':             150,                # min pixel inside a window of double-line-type line           
                            'doubleline_width_px':              50,                 # min width of a double-line, in pixel unit
                            'bold_width_px':                    75}                 # min width of a bold line, in pixel unit

# single frame processing
frame_filepath                          = 'images_test/frame_highway_A5_lanechange_slope_2_0x339.jpg'
frame_filename                          = 'frame_highway_A5_lanechange_slope_2_0x339.jpg'
frame_RGB_origin                        = frame_RGB_read(frame_filepath)            
lanefinding                             = pipeline_lanefinding(param_preprocessing_1st, param_preprocessing_2nd, points_ZoI, points_birdeye_transform, param_curve_class, cam_calibration_file)
frame_mix_RGB_origin_Lane_verbose       = lanefinding.apply_pipeline(frame_RGB_origin)

frame_show(frame_mix_RGB_origin_Lane_verbose, title = 'frame_mix_RGB_origin_Lane_verbose')
frame_RGB_save(frame_mix_RGB_origin_Lane_verbose, 'images_output/frame_highway_A5_lanechange_slope_2_0x339_output.jpg')



# # single frame processing
# frame_filepath                          = 'images_test/frame_highway_A5_lanechange_slope_3_0x628.jpg'
# frame_filename                          = 'frame_highway_A5_lanechange_slope_3_0x628.jpg'
# frame_RGB_origin                        = frame_RGB_read(frame_filepath)            
# lanefinding                             = pipeline_lanefinding(param_preprocessing_1st, param_preprocessing_2nd, points_ZoI, points_birdeye_transform, param_curve_class, cam_calibration_file)
# frame_mix_RGB_origin_Lane_verbose       = lanefinding.apply_pipeline(frame_RGB_origin)

# frame_show(frame_mix_RGB_origin_Lane_verbose, title = 'frame_mix_RGB_origin_Lane_verbose')
# frame_RGB_save(frame_mix_RGB_origin_Lane_verbose, 'images_output/frame_highway_A5_lanechange_slope_3_0x628_output.jpg')

# # single frame processing
# frame_filepath                          = 'images_test/frame_highway_A5_lanechange_slope_1_7x595.jpg'
# frame_filename                          = 'frame_highway_A5_lanechange_slope_1_7x595.jpg'
# frame_RGB_origin                        = frame_RGB_read(frame_filepath)            
# lanefinding                             = pipeline_lanefinding(param_preprocessing_1st, param_preprocessing_2nd, points_ZoI, points_birdeye_transform, param_curve_class, cam_calibration_file)
# frame_mix_RGB_origin_Lane_verbose       = lanefinding.apply_pipeline(frame_RGB_origin)

# frame_show(frame_mix_RGB_origin_Lane_verbose, title = 'frame_mix_RGB_origin_Lane_verbose')
# frame_RGB_save(frame_mix_RGB_origin_Lane_verbose, 'images_output/frame_highway_A5_lanechange_slope_1_7x595_output.jpg')

# # single frame processing
# frame_filepath                          = 'images_test/frame_highway_A5_lanechange_slope_1_10x651.jpg'
# frame_filename                          = 'frame_highway_A5_lanechange_slope_1_10x651.jpg'
# frame_RGB_origin                        = frame_RGB_read(frame_filepath)            
# lanefinding                             = pipeline_lanefinding(param_preprocessing_1st, param_preprocessing_2nd, points_ZoI, points_birdeye_transform, param_curve_class, cam_calibration_file)
# frame_mix_RGB_origin_Lane_verbose       = lanefinding.apply_pipeline(frame_RGB_origin)

# frame_show(frame_mix_RGB_origin_Lane_verbose, title = 'frame_mix_RGB_origin_Lane_verbose')
# frame_RGB_save(frame_mix_RGB_origin_Lane_verbose, 'images_output/frame_highway_A5_lanechange_slope_1_10x651_output.jpg')



# # # video processing
# os.listdir("videos_test/")
# video_load              = VideoFileClip("videos_test/highway_A5_lanechange.mov")
# #Process loaded video
# lanefinding             = pipeline_lanefinding(param_preprocessing_1st, param_preprocessing_2nd, points_ZoI, points_birdeye_transform, param_curve_class, cam_calibration_file)
# video_processed         = video_load.fl_image(lanefinding.apply_pipeline) #NOTE: this function expects color images!!

# #Write output video
# video_output            = 'videos_output/highway_A5_lanechange_new_output.mp4'
# # %time
# video_processed.write_videofile(video_output, audio=True, logger='bar', verbose=True)

# # Display processed video
# HTML("""
# <video width="1280" height="720" controls> <source src="{0}"> </video>""".format(video_processed))




# ## another video processing
# os.listdir("videos_test/")
# video_load              = VideoFileClip("videos_test/highway_A5_lanechange_slope.mov")
# #Process loaded video
# lanefinding             = pipeline_lanefinding(param_preprocessing_1st, param_preprocessing_2nd, points_ZoI, points_birdeye_transform, param_curve_class, cam_calibration_file)
# video_processed         = video_load.fl_image(lanefinding.apply_pipeline) #NOTE: this function expects color images!!

# #Write output video
# video_output            = 'videos_output/highway_A5_lanechange_slope_output.mp4'
# # %time
# video_processed.write_videofile(video_output, audio=True, logger='bar', verbose=True)

# # Display processed video
# HTML("""<video width="1280" height="720" controls> <source src="{0}"> </video>""".format(video_processed))


# ## another video processing
# os.listdir("videos_test/")
# video_load              = VideoFileClip("videos_test/drive_torvillier_80kmph.mov")
# #Process loaded video
# lanefinding             = pipeline_lanefinding(param_preprocessing_1st, param_preprocessing_2nd, points_ZoI, points_birdeye_transform, param_curve_class, cam_calibration_file)
# video_processed         = video_load.fl_image(lanefinding.apply_pipeline) #NOTE: this function expects color images!!

# #Write output video
# video_output            = 'videos_output/drive_torvillier_80kmph_output.mp4'
# # %time
# video_processed.write_videofile(video_output, audio=True, logger='bar', verbose=True)

# # Display processed video
# HTML("""<video width="1280" height="720" controls> <source src="{0}"> </video>""".format(video_processed))



# ## another video processing
# os.listdir("videos_test/")
# video_load              = VideoFileClip("videos_test/highway_A5_normal.mov")
# #Process loaded video
# lanefinding             = pipeline_lanefinding(param_preprocessing_1st, param_preprocessing_2nd, points_ZoI, points_birdeye_transform, param_curve_class, cam_calibration_file)
# video_processed         = video_load.fl_image(lanefinding.apply_pipeline) #NOTE: this function expects color images!!

# #Write output video
# video_output            = 'videos_output/highway_A5_normal_output.mp4'
# # %time
# video_processed.write_videofile(video_output, audio=True, logger='bar', verbose=True)

# # Display processed video
# HTML("""<video width="1280" height="720" controls> <source src="{0}"> </video>""".format(video_processed))

# # another video processing
# os.listdir("videos_test/")
# video_load              = VideoFileClip("videos_test/highway_A5_curve_slope.mov")
# #Process loaded video
# lanefinding             = pipeline_lanefinding(param_preprocessing_1st, param_preprocessing_2nd, points_ZoI, points_birdeye_transform, param_curve_class, cam_calibration_file)
# video_processed         = video_load.fl_image(lanefinding.apply_pipeline) #NOTE: this function expects color images!!

# #Write output video
# video_output            = 'videos_output/highway_A5_curve_slope_output.mp4'
# # %time
# video_processed.write_videofile(video_output, audio=True, logger='bar', verbose=True)

# # Display processed video
# HTML("""<video width="1280" height="720" controls> <source src="{0}"> </video>""".format(video_processed))

# # another video processing
# os.listdir("videos_test/")
# video_load              = VideoFileClip("videos_test/drive_barrier_torvillier.mov")
# #Process loaded video
# lanefinding             = pipeline_lanefinding(param_preprocessing_1st, param_preprocessing_2nd, points_ZoI, points_birdeye_transform, param_curve_class, cam_calibration_file)
# video_processed         = video_load.fl_image(lanefinding.apply_pipeline) #NOTE: this function expects color images!!

# #Write output video
# video_output            = 'videos_output/drive_barrier_torvillier_output.mp4'
# # %time
# video_processed.write_videofile(video_output, audio=True, logger='bar', verbose=True)

# # Display processed video
# HTML("""<video width="1280" height="720" controls> <source src="{0}"> </video>""".format(video_processed))

# ## another video processing
# os.listdir("videos_test/")
# video_load              = VideoFileClip("videos_test/highway_A5_multilane_multitarget.mov")
# #Process loaded video
# lanefinding             = pipeline_lanefinding(param_preprocessing_1st, param_preprocessing_2nd, points_ZoI, points_birdeye_transform, param_curve_class, cam_calibration_file)
# video_processed         = video_load.fl_image(lanefinding.apply_pipeline) #NOTE: this function expects color images!!

# #Write output video
# video_output            = 'videos_output/highway_A5_multilane_multitarget_output.mp4'
# # %time
# video_processed.write_videofile(video_output, audio=True, logger='bar', verbose=True)

# # Display processed video
# HTML("""<video width="1280" height="720" controls> <source src="{0}"> </video>""".format(video_processed))