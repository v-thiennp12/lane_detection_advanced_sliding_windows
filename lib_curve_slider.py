# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:14:52 2020

@author: nguye
Author : nguyenrobot
Copyright : nguyenrobot
https://github.com/nguyenrobot
https://www.nguyenrobot.com
"""

import cv2
import numpy
from lib_frame import *
import matplotlib.pyplot as plot
import matplotlib.image as image
# import matplotlib.pyplot as plot

# Apply window sliding method to find pixel inside a curve then use polyfiot to find a fitted 4-th degree polynomial
# We also classify detected curves into diffferent types and with a confiance level
class curve_slider:
    def __init__(self, param_curve_class):
        self.num_of_windows                     = param_curve_class['num_of_windows']
        self.margin_x                           = param_curve_class['margin_x']
        self.min_pixel_inside                   = param_curve_class['min_pixel_inside']
        self.max_pixel_inside                   = param_curve_class['max_pixel_inside']
        self.max_width_not_a_line               = param_curve_class['max_width_not_a_line']
        self.min_pixel_confindex                = param_curve_class['min_pixel_confindex']
        self.thd_confindex                      = param_curve_class['thd_confindex']
        self.min_pixel_bold                     = param_curve_class['min_pixel_bold']
        self.bold_width_px                      = param_curve_class['bold_width_px']
        self.min_pixel_doubleline               = param_curve_class['min_pixel_doubleline']
        self.doubleline_width_px                = param_curve_class['doubleline_width_px']        
        self.histogram_vertical_ratio_start     = param_curve_class['histogram_vertical_ratio_start']
        self.histogram_vertical_ratio_end       = param_curve_class['histogram_vertical_ratio_end']
        self.histogram_width                    = param_curve_class['histogram_width']
        self.histogram_seed                     = param_curve_class['histogram_seed']
        self.histogram_ratio_localmax           = param_curve_class['histogram_ratio_localmax']
        self.offset_cam                         = param_curve_class['offset_cam']
        self.m_look_ahead                       = param_curve_class['m_look_ahead']
        self.xm_by_pixel                        = param_curve_class['xm_by_pixel']
        self.ym_by_pixel                        = param_curve_class['ym_by_pixel']
        self.m_vehicle_width                    = param_curve_class['m_vehicle_width']
        
        self.lanechange                             = None
        
        self.coeff_L                                = [0, 0, 0, 0]
        self.coeff_R                                = [0, 0, 0, 0]     
        self.confindex_L                            = None
        self.confindex_R                            = None
        self.linetype_L                             = None
        self.linetype_R                             = None                
        self.coeff_next_L                           = [0, 0, 0, 0]
        self.coeff_next_R                           = [0, 0, 0, 0]           
        self.confindex_next_L                       = None
        self.confindex_next_R                       = None  
        self.linetype_next_L                        = None
        self.linetype_next_R                        = None

        self.previous_coeff_L                       = [0, 0, 0, 0]
        self.previous_coeff_R                       = [0, 0, 0, 0]
        self.previous_confindex_L                   = None
        self.previous_confindex_R                   = None
        self.previous_linetype_L                    = None
        self.previous_linetype_R                    = None
        self.previous_coeff_next_L                  = [0, 0, 0, 0]
        self.previous_coeff_next_R                  = [0, 0, 0, 0]
        self.previous_confindex_next_L              = None
        self.previous_confindex_next_R              = None
        self.previous_linetype_next_L               = None
        self.previous_linetype_next_R               = None      
    
    def windows_start(self, frame_binary):
        self.frame_width                = frame_binary.shape[1]
        self.frame_height               = frame_binary.shape[0]
        self.window_height              = numpy.int(self.frame_height/self.num_of_windows)
        self.width_center               = numpy.int(self.frame_width/2 - self.offset_cam)

        # find non zero pixel from grayscale frame
        self.nonzeropixel_x             = numpy.array(numpy.nonzero(frame_binary)[1])
        self.nonzeropixel_y             = numpy.array(numpy.nonzero(frame_binary)[0])
        self.frame_binary               = frame_binary
        self.thd_local_maximum          = (self.thd_confindex/100*self.num_of_windows)*self.min_pixel_inside*self.histogram_ratio_localmax
        self.histogram                  = self.histogram_seeded(frame_binary)        

    def windows_slide(self, frame_binary):
        self.windows_start(frame_binary)
        histogram                       = self.histogram
        # plot.figure()
        # plot.plot(histogram)
        # plot.savefig('images_analyze/histogram.png')
        width_center                    = self.width_center
        px_ind_on_any_line              = numpy.array([], dtype = numpy.int64)
        # frame_RGB_draw_windows          = numpy.dstack((frame_binary*150, frame_binary*150, frame_binary*150))
        frame_RGB_draw_windows          = numpy.dstack((frame_binary, frame_binary, frame_binary))
    
        """find curve on the left side"""
        curve                           = []
        for ij in range(1, 15):
            # if histogram[numpy.argmax(histogram)] < self.thd_local_maximum :
            #     break
            
            # Sliding            
            max_x                           = self.corrector_windows_start(numpy.argmax(histogram))
            px_ind_on_line, px_ind_on_windows, px_ind_on_any_line, drawn_windows \
                                            = self.window_slide_a_curve(max_x, px_ind_on_any_line)
            x_found                         = self.nonzeropixel_x[px_ind_on_line]
            # Evaluate detected pixels
            confindex, linetype             = self.curve_type(px_ind_on_windows)
            
            # plot.figure()
            # plot.plot(histogram)
            # print(max_x)
            
            # Erase a bandwidth to do confuse not other sliding windows
            if len(x_found) > 0:
                left_boundary                               = numpy.int(max([min([min(x_found), max_x - self.margin_x]), 0]))
                right_boundary                              = numpy.int(min([max([max(x_found), max_x + self.margin_x]), self.frame_width - 1]))
                histogram[left_boundary : right_boundary]   = 0
            else:
                left_boundary                               = numpy.int(max([max_x - self.histogram_width, 0]))
                right_boundary                              = numpy.int(min([max_x + self.histogram_width, self.frame_width - 1]))
                histogram[left_boundary : right_boundary]   = 0
            
            # Gather information for a curve
            if confindex >= self.thd_confindex:
                """extrapolation of detected pixels"""
                x_found_ext, y_found_ext, xy_start, xy_end \
                                            = self.curve_extrapolation(px_ind_on_line, px_ind_on_windows)
                                                
                """ fit detected pixels with polynomial """
                coeff, frame_binary_line    = self.polyfit_3rd(x_found_ext, y_found_ext, frame_binary)
                
                """estimate space-left-in-lane at y = 0 from [y] = coeff*[x]"""
                x_window_center_from_vehicle_birdview   = [self.width_center -  (topleft_rightbottom[0][0] + topleft_rightbottom[1][0])/2 for topleft_rightbottom in drawn_windows]
                dist_at_0                               = numpy.mean(x_window_center_from_vehicle_birdview)
                
                """put all information into an instant a_curve"""
                coeff_from_vehicle_birdview = self.coeff_from_vehicle_birdview(coeff)
                a_curve = {'dist_at_0'                      : dist_at_0,
                           'abs_dist_at_0'                  : numpy.abs(dist_at_0),
                           'coeff'                          : coeff,
                           'coeff_from_vehicle_birdview'    : coeff_from_vehicle_birdview,                           
                           'xy_start'                       : xy_start,
                           'xy_end'                         : xy_end,
                           'confindex'                      : confindex,
                           'linetype'                       : linetype,
                           'drawn_windows'                  : drawn_windows}
                            #'frame_binary_line'              : frame_binary_line,
                curve.append(a_curve)

        """choose best two candidates for each side"""
        # a good candidate stays near to the frame center
        curve_L = []
        curve_R = []
        
        if len(curve) > 0:
            curve_L = [d for d in curve if d['dist_at_0'] >= 0] 
            curve_R = [d for d in curve if d['dist_at_0'] < 0] 
        
        # a good candidate stays near to the frame center
        if len(curve_L) > 0:
            curve_L = sorted(curve_L, key = lambda i: i['abs_dist_at_0'], reverse = False)
            del curve_L[2:]
        if len(curve_R) > 0:
            curve_R = sorted(curve_R, key = lambda i: i['abs_dist_at_0'], reverse = False)
            del curve_R[2:]            
            
        # left line
        """fit curve with 3rd polynomial"""
        if len(curve_L) > 0:
            confindex_L             = curve_L[0]['confindex']
            linetype_L              = curve_L[0]['linetype']
            coeff_L                 = curve_L[0]['coeff']
            xy_L_start              = curve_L[0]['xy_start']
            xy_L_end                = curve_L[0]['xy_end']
                        
            """draw windows"""
            frame_RGB_draw_windows          = self.draw_windows(curve_L[0]['drawn_windows'], frame_RGB_draw_windows, [255, 0, 0])
        else:
            confindex_L = 0
            linetype_L  = 0
            xy_L_start  = 0
            xy_L_end    = 0
            coeff_L, __    = self.polyfit_3rd([], [], frame_binary)

        # next-left line
        """fit curve with 3rd polynomial"""
        if len(curve_L) > 1:
            confindex_next_L             = curve_L[1]['confindex']
            linetype_next_L              = curve_L[1]['linetype']
            coeff_next_L                 = curve_L[1]['coeff']
            xy_next_L_start              = curve_L[1]['xy_start']
            xy_next_L_end                = curve_L[1]['xy_end']
            
            """draw windows"""
            frame_RGB_draw_windows                      = self.draw_windows(curve_L[1]['drawn_windows'], frame_RGB_draw_windows, [255, 255, 0])            
        else:
            confindex_next_L = 0
            linetype_next_L  = 0
            xy_next_L_start  = 0
            xy_next_L_end    = 0            
            coeff_next_L, __      = self.polyfit_3rd([], [], frame_binary)
            
        # right line
        """fit curve with 3rd polynomial"""
        if len(curve_R) > 0:
            confindex_R             = curve_R[0]['confindex']
            linetype_R              = curve_R[0]['linetype']
            coeff_R                 = curve_R[0]['coeff']
            xy_R_start              = curve_R[0]['xy_start']
            xy_R_end                = curve_R[0]['xy_end']
            
            """draw windows"""
            frame_RGB_draw_windows          = self.draw_windows(curve_R[0]['drawn_windows'], frame_RGB_draw_windows, [0, 0, 255])
        else:
            confindex_R = 0
            linetype_R  = 0
            xy_R_start  = 0
            xy_R_end    = 0            
            coeff_R, __    = self.polyfit_3rd([], [], frame_binary)

        # next-right line
        """fit curve with 3rd polynomial"""
        if len(curve_R) > 1:
            confindex_next_R             = curve_R[1]['confindex']
            linetype_next_R              = curve_R[1]['linetype']
            coeff_next_R                 = curve_R[1]['coeff']
            xy_next_R_start              = curve_R[1]['xy_start']
            xy_next_R_end                = curve_R[1]['xy_end']
            
            """draw windows"""
            frame_RGB_draw_windows                      = self.draw_windows(curve_R[1]['drawn_windows'], frame_RGB_draw_windows, [0, 255, 255])
        else:
            confindex_next_R = 0
            linetype_next_R  = 0
            xy_next_R_start  = 0
            xy_next_R_end    = 0                
            coeff_next_R, __      = self.polyfit_3rd([], [], frame_binary)
        
        ## Update current coeff_            
        self.coeff_L                                = coeff_L
        self.coeff_R                                = coeff_R
        self.confindex_L                            = confindex_L
        self.confindex_R                            = confindex_R
        self.linetype_L                             = linetype_L
        self.linetype_R                             = linetype_R
        self.coeff_next_L                           = coeff_next_L
        self.coeff_next_R                           = coeff_next_R
        self.confindex_next_L                       = confindex_next_L
        self.confindex_next_R                       = confindex_next_R
        self.linetype_next_L                        = linetype_next_L
        self.linetype_next_R                        = linetype_next_R
        
        self.xy_L_start                             = xy_L_start
        self.xy_L_end                               = xy_L_end    
        self.xy_next_L_start                        = xy_next_L_start
        self.xy_next_L_end                          = xy_next_L_end
        self.xy_R_start                             = xy_R_start
        self.xy_R_end                               = xy_R_end    
        self.xy_next_R_start                        = xy_next_R_start
        self.xy_next_R_end                          = xy_next_R_end   
        
        self.frame_RGB_draw_windows                 = frame_RGB_draw_windows
               
        """check lane change"""
        lane_change                                 = self.lanechange_check()
        
        """# Update previous data"""    
        self.previous_data()
        
    def window_slide_a_curve(self, window_x_mid, mask_found_indice_on_any_line):       
        pixel_indice_on_line                = numpy.array([], dtype = numpy.int64)   
        pixel_indice_on_windows             = []
        drawn_windows                       = []
        
        for window_i in range(self.num_of_windows):
            y_top, y_bottom                 = self.next_y(window_i)
            x_left, x_right                 = self.next_x(window_x_mid)
            
            # find pixel on current lane's curves            
            pixel_indice_on_window_i        = self.find_pixel_indice_inside_rectangle(y_top, y_bottom, x_left, x_right, mask_found_indice_on_any_line)        
            mask_found_indice_on_any_line   = numpy.append(mask_found_indice_on_any_line, pixel_indice_on_window_i)         
            # append found indice for each windows i inside a serie
            pixel_indice_on_line            = numpy.append(pixel_indice_on_line, pixel_indice_on_window_i)
            pixel_indice_on_windows.append(pixel_indice_on_window_i)
         
            # # draw windows
            if len(self.nonzeropixel_x[pixel_indice_on_window_i]) > 0:
                drawn_windows.append(((x_left,y_top), (x_right,y_bottom)))
            # found next-window's bottom center
            window_x_mid = self.next_window_x_mid_(window_x_mid, pixel_indice_on_window_i)
        
        return pixel_indice_on_line, pixel_indice_on_windows, mask_found_indice_on_any_line, drawn_windows

    def draw_windows(self, drawn_windows, frame_RGB_draw_windows, window_color):       
        for lefttop_rightbottom in drawn_windows:    
            # draw windows
            cv2.rectangle(frame_RGB_draw_windows, lefttop_rightbottom[0], lefttop_rightbottom[1], window_color, 2)
        return frame_RGB_draw_windows    
        
    def histogram_seeded(self, frame_binary):
        histogram_slice_num     = numpy.int(self.frame_width/self.histogram_seed)
        
        vertical_start          = numpy.int(self.frame_height*self.histogram_vertical_ratio_start)
        vertical_end            = numpy.int(self.frame_height*self.histogram_vertical_ratio_end)
        
        histogram               = numpy.sum(frame_binary[vertical_start : vertical_end, : ], axis = 0)
        histogram_seeded        = numpy.zeros_like(histogram)
        
        last_seed               = self.frame_width%(self.histogram_seed*histogram_slice_num)
        for ij in range(histogram_slice_num + 1):
            left                = self.histogram_seed*ij
            right               = self.histogram_seed*(ij + 1) - 1        
            histogram_seeded[numpy.int(left + self.histogram_seed/2)]\
                                =  numpy.sum(histogram[left : right])        
        
        if last_seed > 0:
            histogram_seeded[self.histogram_seed*histogram_slice_num + numpy.int(last_seed/2)]\
                                =  numpy.sum(histogram[self.histogram_seed*histogram_slice_num : self.frame_width - 1])
                                
        return histogram_seeded

    def corrector_windows_start(self, max_x_):
        """improving of start position for better accuracy, we can disable this section"""
        max_x_at_start  = max_x_
        max_x_at_end    = max_x_
        histogram_width = self.histogram_width
        
        # Find a better starting position than max_x_, for start a new window in window_slider and for histogram bandwidth cleaning
        for ij in range(self.num_of_windows):
            mask_empty                              = numpy.array([], dtype = numpy.int64)
            x_left_                                 = max(0,max_x_ - histogram_width)
            x_right_                                = min(max_x_ + histogram_width, self.frame_width)
            y_top, y_bottom                         = self.next_y(ij)
            pixel_indice_for_histogram_windows      = self.find_pixel_indice_inside_rectangle(y_top, y_bottom, x_left_, x_right_, mask_empty)
            if len(pixel_indice_for_histogram_windows) >= self.min_pixel_inside:
                max_x_at_start = numpy.int(numpy.mean(self.nonzeropixel_x[pixel_indice_for_histogram_windows]))
                break
        
        # # Find a better ending position than max_x_, for histogram bandwidth cleaning
        # for ij in range(self.num_of_windows - 1, -1, -1):
        #     mask_empty                              = numpy.array([], dtype = numpy.int64)
        #     x_left_                                 = max(0,max_x_ - histogram_width)
        #     x_right_                                = min(max_x_ + histogram_width, self.frame_width)
        #     y_top, y_bottom                         = self.next_y(ij)
        #     pixel_indice_for_histogram_windows      = self.find_pixel_indice_inside_rectangle(y_top, y_bottom, x_left_, x_right_, mask_empty)
        #     if len(pixel_indice_for_histogram_windows) >= self.min_pixel_inside:
        #         max_x_at_end = numpy.int(numpy.mean(self.nonzeropixel_x[pixel_indice_for_histogram_windows]))
        #         break
        # return max_x_at_start, max_x_at_end    
        return max_x_at_start
    
    def next_y(self, i_window):
        """go from bottom to top""" 
        y_top                   = self.frame_height - (i_window + 1)*self.window_height
        y_bottom                = self.frame_height - (i_window + 0)*self.window_height
        
        """go from top down to bottom""" 
        # y_top                   = (i_window + 0)*self.window_height
        # y_bottom                = (i_window + 1)*self.window_height
        
        return y_top, y_bottom
    
    def next_x(self, mid_x_):
        x_left_                 = max(0, mid_x_ - self.margin_x)
        x_right_                = min(self.frame_width - 1, mid_x_ + self.margin_x)
        return x_left_, x_right_    
    
    def next_window_x_mid_(self, current_window_x_mid_, pixel_indice_inside):
        next_window_x_mid_      = current_window_x_mid_
        if len(pixel_indice_inside) >= self.min_pixel_inside:
             next_window_x_mid_ = numpy.int(numpy.mean(self.nonzeropixel_x[pixel_indice_inside]))
        return next_window_x_mid_
    
    def find_pixel_indice_inside_rectangle(self, top, bottom, left, right, mask_found_indice_on_any_line):
        mask_vertical           = (self.nonzeropixel_y <= bottom) & (self.nonzeropixel_y > top) # attention, we climb from the bottom of the frame to the top
        mask_horizontal         = (self.nonzeropixel_x >= left) & (self.nonzeropixel_x < right)
        
        mask_inside             = numpy.zeros_like(mask_vertical, dtype=numpy.uint8)     
        mask_inside[(mask_vertical & mask_horizontal)] = 1
        
        " exclude found indice on other lines"
        mask_inside[mask_found_indice_on_any_line]     = 0
        
        " exclude pixels from a window if they are so populated or so dispersed"
        x_inside                = self.nonzeropixel_x[numpy.nonzero(mask_inside)[0][:]]
        if len(x_inside) > 0:
            if (len(numpy.nonzero(mask_inside)[0][:]) > self.max_pixel_inside) \
                | (abs(max(x_inside) - min(x_inside)) > self.max_width_not_a_line) :
                    
                mask_inside[:]                         = 0
        
        """ return numpy.nonzero(mask_inside)[0][:] as indice of pixels found inside"""
        return numpy.nonzero(mask_inside)[0][:]

    def polyfit_3rd(self, x_found_, y_found_, frame_binary):
        """ fit detected pixels with polynomial """
        coeff_                              = [0, 0, 0, 0]
        frame_binary_line_                  = numpy.zeros_like(frame_binary, dtype = numpy.uint8)    
       
        # fit curve with 3rd degree polynomial        
        if len(y_found_) > 0:
            """ fit detected pixels with 3rd degree polynomial """
            coeff_                          = numpy.polyfit(y_found_, x_found_, 3)
            
            """"we removed out-of-frame pixels"""
            mask_outofframe                 =   (x_found_ > self.frame_width - 1) | (x_found_ < 0) \
                                              | (y_found_ > self.frame_height - 1) | (y_found_ < 0)
            x_found_                        = x_found_[~mask_outofframe]
            y_found_                        = y_found_[~mask_outofframe]
            
        if len(y_found_) > 0:
            # draw binary frame of detected pixels of a curve            
            frame_binary_line_[y_found_, x_found_] = 1        
        
        return coeff_, frame_binary_line_
        
    def curve_type(self, pixel_indice_on_windows_):                
        windows_size                = numpy.zeros(len(pixel_indice_on_windows_))
        mask_bold                   = numpy.zeros(len(pixel_indice_on_windows_))
        mask_doubleline             = numpy.zeros(len(pixel_indice_on_windows_))
        
        for ij in range(len(pixel_indice_on_windows_)):
            windows_size[ij]        = len(pixel_indice_on_windows_[ij])            
            x_in_window             = self.nonzeropixel_x[pixel_indice_on_windows_[ij]]
            
            if len(x_in_window) > 0:
                is_bold                 = (abs(max(x_in_window) - min(x_in_window)) > self.bold_width_px) \
                                        & (windows_size[ij] > self.min_pixel_bold)
                                        
                is_doubleline           = (abs(max(x_in_window) - min(x_in_window)) > self.doubleline_width_px) \
                                        & (windows_size[ij] > self.min_pixel_doubleline) \
                                        & ~is_bold
                if is_bold:
                    mask_bold[ij]       = 1
                if is_doubleline:
                    mask_doubleline[ij] = 1
            
        # Detect confiance level
        confindex           = int(len(windows_size[windows_size > self.min_pixel_confindex])/len(windows_size)*100)        
        bold_index          = int(len(mask_bold[mask_bold > 0])/len(mask_bold)*100)
        doubleline_index    = int(len(mask_doubleline[mask_doubleline > 0])/len(mask_doubleline)*100)
                
        linetype    = 'No Line'
        # 30
        if self.thd_confindex < confindex < 85:
            linetype = 'Dashed'
        elif 85 <= confindex:
            linetype = 'Solid'
            
        if doubleline_index > 50:
            linetype = 'Double-line'
            
        if bold_index > 50:
            linetype = 'Bold'
       
        return confindex, linetype
    
    def curve_extrapolation(self, pixel_indice_on_line_, pixel_indice_on_windows_):
        # detected pixels of a cruve
        x_found_                    = self.nonzeropixel_x[pixel_indice_on_line_]
        y_found_                    = self.nonzeropixel_y[pixel_indice_on_line_]        
                        
        """start and end point of detected curve, before extrapolation"""
        xy_start                            = (0,0)
        xy_end                              = (0,0)
        if len(y_found_) > 0:
            y_max_ind                       = numpy.argmax(y_found_)
            y_min_ind                       = numpy.argmin(y_found_)
            xy_start                        = (x_found_[y_max_ind], y_found_[y_max_ind]) # start is near to the frame's bottom
            xy_end                          = (x_found_[y_min_ind], y_found_[y_min_ind]) # end is near to the frame's top     
                
        """linear extrapolation based on 3 next (previous) windows that are not empty"""
        windows_size                = numpy.zeros(len(pixel_indice_on_windows_))
        for ij in range(len(pixel_indice_on_windows_)):
            windows_size[ij]        = len(pixel_indice_on_windows_[ij])       
        
        num_window_notempty         = len(windows_size[windows_size > self.min_pixel_confindex])
        if num_window_notempty >= int(self.thd_confindex/100*len(windows_size)):
                        
            # Fill extrapolated pixels for first window
            x_base = numpy.array([], dtype = numpy.int64)
            y_base = numpy.array([], dtype = numpy.int64)
            count      = 0            
            if len(pixel_indice_on_windows_[0]) == 0:
                # pick 3 next windows
                for ki in range(1, len(pixel_indice_on_windows_)):
                    if len(pixel_indice_on_windows_[ki]) > 0:
                        count += 1
                        x_base     = numpy.append(x_base, numpy.mean(self.nonzeropixel_x[pixel_indice_on_windows_[ki]]))
                        y_base     = numpy.append(y_base, numpy.mean(self.nonzeropixel_y[pixel_indice_on_windows_[ki]]))
                        
                    if count == 3:                      
                        # fit curve with a line
                        coeff_extrapolated                = numpy.polyfit(y_base, x_base, 1)
                        
                        y_extrapolated                   = numpy.int64(numpy.linspace(self.frame_height - 1, self.frame_height - self.min_pixel_inside, self.min_pixel_inside))
                        x_extrapolated                   = numpy.int64(numpy.polyval(coeff_extrapolated, y_extrapolated))
                        
                        # eliminate out-of-frame pixels >> to be done in polyfit_3rd
                        """extrapolated pixel could be out of frame"""                        
                        x_found_                          = numpy.append(x_found_, x_extrapolated)
                        y_found_                          = numpy.append(y_found_, y_extrapolated)                        
                        break
                
            # Fill extrapolated pixels for final window
            x_base = numpy.array([], dtype = numpy.int64)
            y_base = numpy.array([], dtype = numpy.int64)
            count      = 0
            final_ind  = len(pixel_indice_on_windows_) - 1
            
            if len(pixel_indice_on_windows_[final_ind]) == 0:
                # pick 3 previous windows
                for ki in range(final_ind - 1, 0, -1):
                    if len(pixel_indice_on_windows_[ki]) > 0:
                        count += 1
                        x_base     = numpy.append(x_base, numpy.mean(self.nonzeropixel_x[pixel_indice_on_windows_[ki]]))
                        y_base     = numpy.append(y_base, numpy.mean(self.nonzeropixel_y[pixel_indice_on_windows_[ki]]))
                                            
                    if count == 3:                     
                        # fit curve with a line
                        coeff_extrapolated                = numpy.polyfit(y_base, x_base, 1)
                        
                        y_extrapolated                   = numpy.int64(numpy.linspace(0, self.min_pixel_inside - 1, self.min_pixel_inside))
                        x_extrapolated                   = numpy.int64(numpy.polyval(coeff_extrapolated, y_extrapolated))
                        
                        # eliminate out-of-frame pixels >> to be done in polyfit_3rd
                        """extrapolated pixel could be out of frame"""                        
                        x_found_                          = numpy.append(x_found_, x_extrapolated)
                        y_found_                          = numpy.append(y_found_, y_extrapolated)                        
                        break                    
            
        return x_found_, y_found_, xy_start, xy_end
            
    def coeff_from_vehicle_birdview(self, coeff_from_topright_birdview):
        # + a translation of frame_width/2 in lateral direction
        # + a translation of frame_height in logitudinal
        # + a rotation of 180° around origin point
        # New Origin : Camera's center
        # New view : birdseye view
        # New x direction : from new Origin to the left of ego vehicle
        # New y direction : from new Origin toward ahead
        
        # coeff from the view at bottom-center of frame (cam origin)
        coeff_from_vehicle_birdview = [0, 0, 0, 0]
        
        is_all_zero                 = numpy.all((numpy.array(coeff_from_topright_birdview) == 0))
        if ~is_all_zero:        
            H                               = self.frame_height            
            d                               = coeff_from_topright_birdview[0]
            c                               = coeff_from_topright_birdview[1]
            b                               = coeff_from_topright_birdview[2]
            a                               = coeff_from_topright_birdview[3]           
                        
            coeff_from_vehicle_birdview       = numpy.zeros_like(coeff_from_topright_birdview)
            coeff_from_vehicle_birdview[0]    = d
            coeff_from_vehicle_birdview[1]    = -(3*d*H + c)
            coeff_from_vehicle_birdview[2]    = 3*d*H**2 + 2*c*H + b
            coeff_from_vehicle_birdview[3]    = -(d*H**3 + c*H**2 + b*H + a) + self.width_center
            
        return coeff_from_vehicle_birdview
    
    def coeff_pixel_to_meter(self, coeff_in_pixel, xm_by_pixel, ym_by_pixel):
        """convert 3rd polynomial's coefficient from pixel dimension to meter dimension""" 
        # input : coeff_in_pixel | [y] = coeff_in_pixel*[x] in pixel dimension
        # 
        # outpt : coeff_in_meter | [Y] = coeff_in_pixel*[X] in meter dimension
        #
        d = coeff_in_pixel[0]*xm_by_pixel/(ym_by_pixel**3)
        c = coeff_in_pixel[1]*xm_by_pixel/(ym_by_pixel**2)
        b = coeff_in_pixel[2]*xm_by_pixel/ym_by_pixel
        a = coeff_in_pixel[3]*xm_by_pixel
        
        coeff_in_meter = [d, c, b, a]
        
        return coeff_in_meter
    
    def coeff_SLAM(self, coeff, look_ahead):
        # look ahead            : look ahead distance
        # curvature             : curve's curvature at  Y = look ahead point
        # space-left-in-lane    : distance from vehicle wheel to line, at Y = 0
        # Y : direction from rear to front of ego vehicle
        # X : direction from right to left of ego vehicle
        # [X] = [Y]*coeff
        # coeff must be in ego vehicle's coordination system
        """coeff in H-translated coordination system """
        # translation H 
        H                               = look_ahead            
        # original coeff        
        d                               = coeff[0]
        c                               = coeff[1]
        b                               = coeff[2]
        a                               = coeff[3]           
        # coeff in H-translated coordination system
        d_H    = d
        c_H    = 3*d*H + c
        b_H    = 3*d*H**2 + 2*c*H + b
        a_H    = d*H**3 + c*H**2 + b*H + a
        
        # heading angle  : atan(b)
        # space-left-in-lane : a*sqrt(1+b^2)/sqrt(1+b^2+2ac)
        # curvature : K = 1/R = 2c/(sqrt(1+b^2))^3
        # curvature derivative in Y direction :6d/(sqrt(1+b^2))^3
        heading_angle       = numpy.arctan(b_H)*180/numpy.pi
        curvature           = 2*c_H/(numpy.sqrt(1 + b_H**2))**3
        space_left_in_lane  = a_H*numpy.sqrt(1 + b_H**2)/numpy.sqrt(1 + b_H**2 + 2*a_H*c_H) + self.m_vehicle_width/2
        
        return heading_angle, curvature, space_left_in_lane
            
    def lanechange_check(self): #, coeff_L, coeff_R):
        lane_change                                 = False
        self.lanechange                             = 'None'
        
        # Change to vehicle origin
        coeff_L     = self.coeff_from_vehicle_birdview(self.coeff_L)
        coeff_R     = self.coeff_from_vehicle_birdview(self.coeff_R)        
        pre_coeff_L = self.coeff_from_vehicle_birdview(self.previous_coeff_L)
        pre_coeff_R = self.coeff_from_vehicle_birdview(self.previous_coeff_R)        
        
        is_all_zero_L                               = numpy.all((numpy.array(coeff_L) == 0))
        is_all_zero_R                               = numpy.all((numpy.array(coeff_R) == 0))
        is_all_zero_pre_L                           = numpy.all((numpy.array(pre_coeff_L) == 0))
        is_all_zero_pre_R                           = numpy.all((numpy.array(pre_coeff_R) == 0))
                
        if ~is_all_zero_pre_L & ~is_all_zero_R:
            coeff_grad = numpy.abs(pre_coeff_L - coeff_R)
            d_cond = coeff_grad[0] < 5E-6
            c_cond = coeff_grad[1] < 5E-3
            b_cond = coeff_grad[2] < 5E-1
            a_cond = coeff_grad[3] < 1E1
                            
            if a_cond & b_cond & c_cond & d_cond:
                lane_change                                 = True
                self.lanechange                             = 'Right to Left'
                
        if ~is_all_zero_pre_R & ~is_all_zero_L:
            coeff_grad = numpy.abs(pre_coeff_R - coeff_L)
            d_cond = coeff_grad[0] < 5E-6
            c_cond = coeff_grad[1] < 5E-3
            b_cond = coeff_grad[2] < 5E-1
            a_cond = coeff_grad[3] < 1E1
            
            if a_cond & b_cond & c_cond & d_cond:
                lane_change                                 = True
                self.lanechange                             = 'Left to Right'                      
        
        return lane_change
        
    def previous_data(self):        
        self.previous_coeff_L                                = self.coeff_L        
        self.previous_coeff_R                                = self.coeff_R
        self.previous_confindex_L                            = self.confindex_L
        self.previous_confindex_R                            = self.confindex_R
        self.previous_linetype_L                             = self.linetype_L
        self.previous_linetype_R                             = self.linetype_R
        self.previous_coeff_next_L                           = self.coeff_next_L
        self.previous_coeff_next_R                           = self.coeff_next_R
        self.previous_confindex_next_L                       = self.confindex_next_L
        self.previous_confindex_next_R                       = self.confindex_next_R
        self.previous_linetype_next_L                        = self.linetype_next_L
        self.previous_linetype_next_R                        = self.linetype_next_R