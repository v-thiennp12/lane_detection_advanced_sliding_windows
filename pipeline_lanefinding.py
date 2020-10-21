# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 12:10:35 2020

@author: nguye
Author : nguyenrobot
Copyright : nguyenrobot
https://github.com/nguyenrobot
https://www.nguyenrobot.com
"""
# import libraries
from lib_frame import *
from lib_photography import *
from pipeline_preprocessing import pipeline_preprocessing
from lib_perspective import birdseye
from lib_curve_slider import curve_slider
from pipeline_camcalibration import *

class pipeline_lanefinding:
    def __init__(self, param_preprocessing, param_preprocessing_2nd, points_ZoI, points_1st_tranform, param_curve_class, cam_calibration_file):
        self.param_preprocessing        = param_preprocessing
        self.param_preprocessing_2nd    = param_preprocessing_2nd
        self.points_ZoI                 = points_ZoI
        self.points_1st_tranform        = points_1st_tranform
        self.param_curve_class          = param_curve_class     
        self.pipeline_camcalibration    = pipeline_camcalibration(cam_calibration_file)
        self.class_birdseye             = birdseye(self.points_1st_tranform)
        self.curve_class                = curve_slider(self.param_curve_class)
        
    def apply_pipeline(self, frame_RGB_origin):
        # Undistort frame
        frame_RGB_origin                = self.pipeline_camcalibration.apply_undistort(frame_RGB_origin)

        # Photography pre-rocessing
        preprocessing                   = pipeline_preprocessing(frame_RGB_origin, self.param_preprocessing)
        preprocessing.apply_pipeline()
        frame_HLS                       = preprocessing.frame_HLS

        # Birdeye
        # 1st birdseye transform        
        frame_RGB_skyview               = self.class_birdseye.apply_skyview(frame_RGB_origin)
        
        # 2nd pre-processing on birdseye view
        # ZoI Zone of Interest filtering
        frame_HLS[~(frame_mask_ZoI(frame_HLS, self.points_ZoI) > 0)] = [0, 0, 0]  
        frame_RGB_skyview_preprocessed  = self.class_birdseye.apply_skyview(frameHLS2RGB(frame_HLS))
        preprocessing_2nd               = pipeline_preprocessing(frame_RGB_skyview_preprocessed, self.param_preprocessing_2nd)
        preprocessing_2nd.apply_pipeline()
        frame_HLS_2nd                   = preprocessing_2nd.frame_HLS               
        frame_gray_2nd                  = frameHSL2gray(frame_HLS_2nd)
        
        frame_binary_2nd                = numpy.zeros_like(frame_gray_2nd, dtype = numpy.uint8)
        frame_binary_2nd[frame_gray_2nd > 0]= 1
        frame_binary_skyview            = frame_binary_2nd
        
        ## Curve slider        
        self.curve_class.windows_slide(frame_binary_skyview)            
        
        ## Display ZoI zone and Birdeye points
        frame_ZoI_source_points     = numpy.copy(frame_RGB_origin)
        cv2.drawContours(frame_ZoI_source_points, [numpy.int_(self.points_ZoI)], contourIdx=-1, color=(255,0,0), thickness=5)
        frame_mix_ZoI_points        = cv2.addWeighted(frame_ZoI_source_points, 0.5, frame_RGB_origin, 0.5, 0)    
        
        frame_source_points         = numpy.copy(frameHLS2RGB(frame_HLS))
        frame_dest_points           = numpy.copy(frameHLS2RGB(frame_HLS))
        cv2.drawContours(frame_source_points, [numpy.int_(self.points_1st_tranform['source_points'])], contourIdx=-1, color=(255,0,0), thickness=5)
        cv2.drawContours(frame_dest_points, [numpy.int_(self.points_1st_tranform['destination_points'])], contourIdx=-1, color=(0,255,0), thickness=5)
        frame_birdeye_points        = cv2.addWeighted(frame_dest_points, 0.5, frame_source_points, 0.5, 0)
        
        # ## display preprocessing for analyzes
        # frame_show(preprocessing.frame_HLS[:,:,1], title = 'preprocessing_1st_L')        
        # frame_show(preprocessing.sobel_L_x, title = 'preprocessing_1st_sobel_L_x')
        # frame_show(preprocessing.sobel_L_y, title = 'preprocessing_1st_sobel_L_y')
        # frame_show(preprocessing.sobel_L_mag, title = 'preprocessing_1st_sobel_L_mag')
        # frame_show(preprocessing.sobel_L_arg , title = 'preprocessing_1st_sobel_L_arg')         
        
        # frame_show(preprocessing.frame_HLS[:,:,2], title = 'preprocessing_1st_S')        
        # frame_show(preprocessing.sobel_S_x, title = 'preprocessing_1st_sobel_S_x')
        # frame_show(preprocessing.sobel_S_y, title = 'preprocessing_1st_sobel_S_y')
        # frame_show(preprocessing.sobel_S_mag, title = 'preprocessing_1st_sobel_S_mag')
        # frame_show(preprocessing.sobel_S_arg , title = 'preprocessing_1st_sobel_S_arg') 
        # ##
        
        # ## display 2nd preprocessing for analyzes
        # frame_show(preprocessing_2nd.frame_HLS[:,:,1], title = 'preprocessing_2nd_L')        
        # frame_show(preprocessing_2nd.sobel_L_x, title = 'preprocessing_2nd_sobel_L_x')
        # frame_show(preprocessing_2nd.sobel_L_y, title = 'preprocessing_2nd_sobel_L_y')
        # frame_show(preprocessing_2nd.sobel_L_mag, title = 'preprocessing_2nd_sobel_L_mag')
        # frame_show(preprocessing_2nd.sobel_L_arg , title = 'preprocessing_2nd_sobel_L_arg')              
        
        # frame_show(preprocessing_2nd.frame_HLS[:,:,2], title = 'preprocessing_2nd_S')        
        # frame_show(preprocessing_2nd.sobel_S_x, title = 'preprocessing_2nd_sobel_S_x')
        # frame_show(preprocessing_2nd.sobel_S_y, title = 'preprocessing_2nd_sobel_S_y')
        # frame_show(preprocessing_2nd.sobel_S_mag, title = 'preprocessing_2nd_sobel_S_mag')
        # frame_show(preprocessing_2nd.sobel_S_arg , title = 'preprocessing_2nd_sobel_S_arg')                 
        # #        
        
        # ### Display frame for further analyze
        # frame_show(preprocessing.frame_HLS[:,:,0] , title = 'frame_H_1st')
        # frame_show(preprocessing.frame_HLS[:,:,1] , title = 'frame_L_1st')
        # frame_show(preprocessing.frame_HLS[:,:,2] , title = 'frame_S_1st')
        
        # frame_show(frame_HLS_2nd[:,:,0] , title = 'frame_H_2nd')
        # frame_show(frame_HLS_2nd[:,:,1] , title = 'frame_L_2nd')
        # frame_show(frame_HLS_2nd[:,:,2] , title = 'frame_S_2nd')        
        
        # # frame_show(frame_gray_2nd, title = 'frame_gray_2nd')
        # frame_show(frame_mix_ZoI_points, title = 'frame_mix_ZoI_points')
        # frame_show(frame_birdeye_points, title = 'frame_birdeye_points')
        # frame_show(frame_RGB_skyview, title = 'frame_RGB_skyview')
        # frame_RGB_save(frame_RGB_skyview, 'images_analyze/frame_RGB_skyview.jpg')
        # frame_show(frame_binary_skyview, title = 'frame_binary_skyview')
        # frame_RGB_save(frame_binary_skyview, 'images_analyze/frame_binary_skyview.jpg')
        # ## Display ZoI zone and Birdeye points
        
        # Project detected lines in origin frame
        frame_RGB_skyview_CurrentLane       = numpy.zeros_like(frame_RGB_origin)
        frame_RGB_skyview_CurrentLane       = frame_RGB_draw_curve(frame_RGB_skyview_CurrentLane, self.curve_class.coeff_L, self.curve_class.xy_L_start, self.curve_class.xy_L_end, [255, 0, 50])
        frame_RGB_skyview_CurrentLane       = frame_RGB_draw_curve(frame_RGB_skyview_CurrentLane, self.curve_class.coeff_R, self.curve_class.xy_R_start, self.curve_class.xy_R_end, [0, 0, 255])
        
        frame_RGB_skyview_NextLane          = numpy.zeros_like(frame_RGB_origin)
        frame_RGB_skyview_NextLane          = frame_RGB_draw_curve(frame_RGB_skyview_NextLane, self.curve_class.coeff_next_L, self.curve_class.xy_next_L_start, self.curve_class.xy_next_L_end, [255, 255, 0])
        frame_RGB_skyview_NextLane          = frame_RGB_draw_curve(frame_RGB_skyview_NextLane, self.curve_class.coeff_next_R, self.curve_class.xy_next_R_start, self.curve_class.xy_next_R_end, [0, 255, 255])
        
        # Reversed birdseye view transform
        frame_RGB_camview_CurrentLane       = self.class_birdseye.apply_vehicleview(frame_RGB_skyview_CurrentLane)
        frame_RGB_camview_NextLane          = self.class_birdseye.apply_vehicleview(frame_RGB_skyview_NextLane)
        
        frame_mix_RGB_origin_Lane           = cv2.addWeighted(frame_RGB_origin, 0.9, frame_RGB_camview_NextLane, 1, 0)
        frame_mix_RGB_origin_Lane           = cv2.addWeighted(frame_mix_RGB_origin_Lane, 0.9, frame_RGB_camview_CurrentLane, 1, 0)
        
        """make verbose frame"""
        # frame_mix_RGB_origin_Lane_verbose   = self.make_frame_verbose(preprocessing, self.curve_class, frame_mix_RGB_origin_Lane, frame_RGB_skyview, frame_binary_skyview)
        frame_mix_RGB_origin_Lane_verbose   = self.make_frame_verbose(preprocessing, frame_mix_RGB_origin_Lane, frame_RGB_skyview, frame_binary_skyview)
        
        return frame_mix_RGB_origin_Lane_verbose
    
    def make_frame_verbose(self, preprocessing, frame_mix_RGB_origin_Lane, frame_RGB_skyview, frame_binary_skyview): # self.curve_class
        """make verbose frame"""
        frame_mix_RGB_origin_Lane_verbose   = numpy.copy(frame_mix_RGB_origin_Lane)    
        W                                   = int(frame_mix_RGB_origin_Lane_verbose.shape[1]/4)
        H                                   = int(frame_mix_RGB_origin_Lane_verbose.shape[0]/4)
        # print(H, W)
        frame_RGB_skyview_resized           = cv2.resize(frame_RGB_skyview, (W,H), interpolation = cv2.INTER_AREA)
        frame_skyview_resized               = frame_scale_uint8(cv2.resize(frame_binary_skyview, (W,H), interpolation = cv2.INTER_AREA))
        frame_RBG_skyview_resized           = numpy.dstack((frame_skyview_resized, frame_skyview_resized, frame_skyview_resized))
        frame_HLS_resized                   = cv2.resize(preprocessing.frame_HLS, (W,H), interpolation = cv2.INTER_AREA)
        frame_RGB_draw_windows_resized      = cv2.resize(self.curve_class.frame_RGB_draw_windows, (W,H), interpolation = cv2.INTER_AREA)
        frame_RGB_draw_windows_resized      = cv2.addWeighted(frame_RGB_draw_windows_resized, 0.9, frame_RBG_skyview_resized, 1, 0)
        
        # frame_RGB_draw_windows_unsized      = cv2.addWeighted(self.curve_class.frame_RGB_draw_windows, 0.9, numpy.dstack((frame_scale_uint8(frame_binary_skyview), frame_scale_uint8(frame_binary_skyview), frame_scale_uint8(frame_binary_skyview))), 0.2, 0)
        # frame_show(frame_RGB_draw_windows_unsized , title = 'frame_RGB_draw_windows_unsized')
        # frame_mix_RGB_origin_Lane            
        frame_mix_RGB_origin_Lane_verbose[:H,:W]        = frame_RGB_skyview_resized
        frame_mix_RGB_origin_Lane_verbose[:H,W:2*W]     = frame_RBG_skyview_resized
        frame_mix_RGB_origin_Lane_verbose[:H,2*W:3*W]   = frame_HLS_resized
        frame_mix_RGB_origin_Lane_verbose[:H,3*W:4*W]   = frame_RGB_draw_windows_resized
        
        
        """add text to verbose frame"""       
        # coeff in ego vehicle's coordination system, in pixel dimension
        coeff_L_from_vehicle_birdview_px       = self.curve_class.coeff_from_vehicle_birdview(self.curve_class.coeff_L)
        coeff_R_from_vehicle_birdview_px       = self.curve_class.coeff_from_vehicle_birdview(self.curve_class.coeff_R)
        coeff_next_L_from_vehicle_birdview_px  = self.curve_class.coeff_from_vehicle_birdview(self.curve_class.coeff_next_L)
        coeff_next_R_from_vehicle_birdview_px  = self.curve_class.coeff_from_vehicle_birdview(self.curve_class.coeff_next_R)
        
        # coeff in ego vehicle's coordination system, in meter dimension
        xm_by_pixel                            = self.curve_class.xm_by_pixel
        ym_by_pixel                            = self.curve_class.ym_by_pixel        
        coeff_L_from_vehicle_birdview          = self.curve_class.coeff_pixel_to_meter(coeff_L_from_vehicle_birdview_px, xm_by_pixel, ym_by_pixel)
        coeff_R_from_vehicle_birdview          = self.curve_class.coeff_pixel_to_meter(coeff_R_from_vehicle_birdview_px, xm_by_pixel, ym_by_pixel)
        coeff_next_L_from_vehicle_birdview     = self.curve_class.coeff_pixel_to_meter(coeff_next_L_from_vehicle_birdview_px, xm_by_pixel, ym_by_pixel)
        coeff_next_R_from_vehicle_birdview     = self.curve_class.coeff_pixel_to_meter(coeff_next_R_from_vehicle_birdview_px, xm_by_pixel, ym_by_pixel)
        
        # calculate heading_angle, curvature and space-left-in-lane in meter dimension
        m_look_ahead                           = self.curve_class.m_look_ahead
        heading_angle_L, curvature_L, space_left_in_lane_L = self.curve_class.coeff_SLAM(coeff_L_from_vehicle_birdview, m_look_ahead)
        heading_angle_R, curvature_R, space_left_in_lane_R = self.curve_class.coeff_SLAM(coeff_R_from_vehicle_birdview, m_look_ahead)
        
        # add text to verbose frame, curve polynomial      
        coeff_L_text                                                = [str("{:.2e}".format(x)) for x in coeff_L_from_vehicle_birdview]
        text_L = ' | '.join(coeff_L_text)
        coeff_R_text                                                = [str("{:.2e}".format(x)) for x in coeff_R_from_vehicle_birdview]
        text_R = ' | '.join(coeff_R_text)       
        coeff_next_L_text                                           = [str("{:.2e}".format(x)) for x in coeff_next_L_from_vehicle_birdview]
        text_next_L = ' | '.join(coeff_next_L_text)
        coeff_next_R_text                                           = [str("{:.2e}".format(x)) for x in coeff_next_R_from_vehicle_birdview]
        text_next_R = ' | '.join(coeff_next_R_text)        
        
        lanechange_color = (255, 255, 255)
        lanechange_text  = self.curve_class.lanechange
        if (self.curve_class.lanechange == 'Left to Right'):
            lanechange_color = (0, 0, 255)
            lanechange_text  = lanechange_text + '>>>>>>>>>>>>>>'
        if (self.curve_class.lanechange == 'Right to Left'):
            lanechange_color = (255, 0, 50)
            lanechange_text  = '<<<<<<<<<<<<<<<<<<' + lanechange_text
            
        cv2.putText(frame_mix_RGB_origin_Lane_verbose, 'Lane Change || ' + lanechange_text, (400, 350), cv2.FONT_HERSHEY_COMPLEX, 0.75, lanechange_color, 1, cv2.LINE_AA)
        
        cv2.putText(frame_mix_RGB_origin_Lane_verbose, 'Left Poly | ' + text_L, (100, 270), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 50), 1, cv2.LINE_AA)
        cv2.putText(frame_mix_RGB_origin_Lane_verbose, 'Left Type | ' + str(self.curve_class.linetype_L), (100, 290), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 50), 1, cv2.LINE_AA)
        cv2.putText(frame_mix_RGB_origin_Lane_verbose, 'Left Confindex | ' + str(self.curve_class.confindex_L), (100, 310), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 50), 1, cv2.LINE_AA)
        
        cv2.putText(frame_mix_RGB_origin_Lane_verbose, 'Right Poly | ' + text_R, (720, 270), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_mix_RGB_origin_Lane_verbose, 'Right Type | ' + str(self.curve_class.linetype_R), (720, 290), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_mix_RGB_origin_Lane_verbose, 'Right Confindex | ' + str(self.curve_class.confindex_R), (720, 310), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        cv2.putText(frame_mix_RGB_origin_Lane_verbose, text_next_L, (20, 200), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame_mix_RGB_origin_Lane_verbose, 'Next Left Type | ' + str(self.curve_class.linetype_next_L), (20, 220), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame_mix_RGB_origin_Lane_verbose, 'Next Left Confindex | ' + str(self.curve_class.confindex_next_L), (20, 240), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        
        cv2.putText(frame_mix_RGB_origin_Lane_verbose, text_next_R, (820, 200), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_mix_RGB_origin_Lane_verbose, 'Next Right Type | ' + str(self.curve_class.linetype_next_R), (820, 220), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_mix_RGB_origin_Lane_verbose, 'Next Right Confindex | ' + str(self.curve_class.confindex_next_R), (820, 240), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # add text to verbose frame, heading_angle, curvature and space-left-in-lane in meter dimension
        heading_angle_L_text        = str("{:.2e}".format(heading_angle_L))
        heading_angle_R_text        = str("{:.2e}".format(heading_angle_R))        
        curvature_L_text            = str("{:.2e}".format(curvature_L))
        curvature_R_text            = str("{:.2e}".format(curvature_R))
        space_left_in_lane_L_text   = str("{:.2e}".format(space_left_in_lane_L))
        space_left_in_lane_R_text   = str("{:.2e}".format(space_left_in_lane_R))        
        
        cv2.putText(frame_mix_RGB_origin_Lane_verbose, '_____________________', (20, 350), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_mix_RGB_origin_Lane_verbose, 'Look ahead distance | ' + str(m_look_ahead) + ' (m)', (20, 370), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_mix_RGB_origin_Lane_verbose, 'Heading angle Left | ' + heading_angle_L_text + ' (deg)', (20, 390), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_mix_RGB_origin_Lane_verbose, 'Curvature Left | ' + curvature_L_text + ' (1/m)', (20, 410), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_mix_RGB_origin_Lane_verbose, 'Space-left-in-lane Left | ' + space_left_in_lane_L_text + ' (m)', (20, 430), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.putText(frame_mix_RGB_origin_Lane_verbose, '_____________________', (820, 350), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_mix_RGB_origin_Lane_verbose, 'Look ahead distance | ' + str(m_look_ahead) + ' (m)', (820, 370), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)        
        cv2.putText(frame_mix_RGB_origin_Lane_verbose, 'Heading angle Right | ' + heading_angle_R_text + ' (deg)', (820, 390), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_mix_RGB_origin_Lane_verbose, 'Curvature Right | ' + curvature_R_text + ' (1/m)', (820, 410), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)       
        cv2.putText(frame_mix_RGB_origin_Lane_verbose, 'Space-left-in-lane Right | ' + space_left_in_lane_R_text + ' (m)', (820, 430), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)       
        
        return frame_mix_RGB_origin_Lane_verbose