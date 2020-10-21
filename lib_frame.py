# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 22:50:08 2020

@author: nguye
Author : nguyenrobot
Copyright : nguyenrobot
https://github.com/nguyenrobot
https://www.nguyenrobot.com
"""
import numpy
import math
import cv2
import matplotlib.pyplot as plot
import matplotlib.image as image
#%matplotlib inline

def frame_RGB_read(path):
    """"read a frame RGB from filepath"""
    return image.imread(path)

def frame_show(frame, title = '_'):
    plot.figure()
    # print(type(frame), frame.shape)
    plot.imshow(frame)
    plot.title(title)
    plot.colorbar()
    plot.savefig('images_analyze/' + title + '.jpg', dpi=500)

def frame_RGB_save(frame, path):
    """"read a frame RGB to filepath"""
    plot.imsave(path, frame)
    
def frameRGB2gray(frame_RGB):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plot.imshow(gray, cmap='gray')"""
    # Or use BGR2GRAY if you read an image with cv2.imread()    
    return cv2.cvtColor(frame_RGB, cv2.COLOR_RGB2GRAY)

def frameHSL2gray(frame_HLS):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plot.imshow(gray, cmap='gray')"""
    frame_RGB   = cv2.cvtColor(frame_HLS, cv2.COLOR_HLS2RGB)
    frame_gray  = cv2.cvtColor(frame_RGB, cv2.COLOR_RGB2GRAY)
    
    return frame_gray

def frameRGB2HLS(frame_RGB):
    """Applies the HLS transform to a RGB frame"""
    return cv2.cvtColor(frame_RGB, cv2.COLOR_RGB2HLS)

def frameHLS2RGB(frame_HLS):
    """Applies the RGB transform to a HLS frame"""
    return cv2.cvtColor(frame_HLS, cv2.COLOR_HLS2RGB)

def frame_subplot_HLS(frame_HLS, title=''):
    """plot a frame HLS"""
    frame_H         = frame_HLS[:,:,0]
    frame_L         = frame_HLS[:,:,1]    
    frame_S         = frame_HLS[:,:,2]
    
    plot.figure()    
    plot.subplot(221)
    plot.title(title)
    #plot.title('frame_RGB')
    plot.imshow(frameHLS2RGB(frame_HLS))
    
    plot.subplot(222)
    plot.title('frame_H')
    plot.imshow(frame_H)
    plot.colorbar()
    
    plot.subplot(223)
    plot.title('frame_L')
    plot.imshow(frame_L)
    plot.colorbar()    
    
    plot.subplot(224)
    plot.title('frame_S')
    plot.imshow(frame_S)
    plot.colorbar()   
   
def frame1_apply_thd(frame, thd_low, thd_high):
    """apply threshold filtering for a single chanel frame"""
    frame[(frame < thd_low) | (frame > thd_high)]      = 0
    return frame

def frame_scale_uint8(frame):
    """scale a single chanel frame to uint8 0..255, centered at 127"""
    # input     : .. -a .. 0   .. b .., real
    # output    : .. a' .. 127 .. b'.., unit8

    length      = numpy.max(numpy.abs(frame))    
    if length > 0:
        result  = numpy.uint8(127*(1 + frame/length))
    else:
        result  = numpy.zeros_like(frame, dtype = numpy.uint8)
    return result

def frame3_scale_unit8(frame3):
    """scale a 3-chanels frame to uint8"""
    result          = numpy.zeros_like(frame3, dtype = numpy.uint8)
    size3           = frame3.shape[2]
    for ind in range(0, size3):
        result[:,:,ind] = frame_scale_uint8(frame3[:,:,ind])
    return result
   
def frame_sobelX(frame_gray, ksize = 9):
    """apply sobel operator to lateral  X-direction"""
    return cv2.Sobel(frame_gray, cv2.CV_64F, 1, 0, ksize)

def frame_sobelY(frame_gray, ksize = 9):
    """apply sobel operator to logitudinal  Y-direction"""
    return cv2.Sobel(frame_gray, cv2.CV_64F, 0, 1, ksize)

def frame_sobel(frame_gray):
    """find magnitude of gradient-vector (caculated from sobelX and sobelY) for a single chanel frame"""
    sobel_x     = frame_sobelX(frame_gray)
    sobel_y     = frame_sobelY(frame_gray)
    sobel_mag   = numpy.sqrt(sobel_x**2 + sobel_y**2)
    sobel_arg   = (numpy.abs(numpy.arctan2(sobel_y, sobel_x)) - numpy.pi/2)*180/numpy.pi
        # how far the sobel-vector(y,x) from the vertical
   
    return frame_scale_uint8(sobel_x), frame_scale_uint8(sobel_y), frame_scale_uint8(sobel_mag), frame_scale_uint8(sobel_arg)

def frame_canny_edge(frame_gray, low_thd = 150, high_thd = 200):
    """Applies a Canny edge detection"""
    return cv2.Canny(frame_gray, low_thd, high_thd)

def gaussian_blur(frame_gray, kernel_size = 15):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(frame_gray, (kernel_size, kernel_size), 0)

def frame_mask_ZoI(frame_, points_ZoI):
    """ apply ZoI (zone-of-interest) filtering for a single chanel frame or a 3-chanels frame """
    if len(frame_.shape)    == 2:
        mask_ = numpy.zeros_like(frame_[:,:], dtype=numpy.uint8)
    elif len(frame_.shape)  == 3:
        mask_ = numpy.zeros_like(frame_[:,:,0], dtype=numpy.uint8)
        
    """ create polygon mask """
    return cv2.fillPoly(mask_, [numpy.int_(points_ZoI)], color=(255))
    
def frame_RGB_draw_curve(frame_RGB, coeff_, xy_start, xy_end, line_color_ = [255, 0, 0]):
    is_all_zero_        = numpy.all((numpy.array(coeff_) == 0))
    if ~is_all_zero_:
        """draw A 3rd degree polynomial curves on a RGB-frame"""
        frame_width     = frame_RGB[:,:,0].shape[1]
        frame_height    = frame_RGB[:,:,0].shape[0]
    
        y_linespace     = numpy.uint64(numpy.linspace(0, frame_height - 1, numpy.int(frame_height/3))) 
        x_linespace     = numpy.uint64(coeff_[0]*y_linespace**3 + coeff_[1]*y_linespace**2 + coeff_[2]*y_linespace + coeff_[3]*y_linespace**0)
        
        # remove pixel outside of frame
        # also remove extrapolated pixels
        mask_out_of_frame_  = (x_linespace < 0) | (x_linespace > frame_width - 1) \
                                | (y_linespace < xy_end[1]) | (y_linespace > xy_start[1])
                                # start near the frame's bottom
                                # end near the frame's top
        y_linespace         = y_linespace[~mask_out_of_frame_]
        x_linespace         = x_linespace[~mask_out_of_frame_]
            
        """draw curve"""
        t = 8
        for (x, y) in zip(x_linespace, y_linespace):
            cv2.line(frame_RGB, (int(x - t), y), (int(x + t), y), line_color_, int(t / 2))
    return frame_RGB

def frame_RGB_draw_zone(frame_RGB, coeff_L, coeff_R, fill_color = [255, 0, 255]):
    """draw 2 polynomial curves (left and right) on a RGB-frame"""
    frame_width     = frame_RGB[:,:,0].shape[1]
    frame_height    = frame_RGB[:,:,0].shape[0]
    
    y_linespace     = numpy.uint64(numpy.linspace(0, frame_height - 1, numpy.int(frame_height/3)))
    
    x_L_linespace = numpy.uint64(coeff_L[0]*y_linespace**3 + coeff_L[1]*y_linespace**2 + coeff_L[2]*y_linespace + coeff_L[3]*y_linespace**0)
    x_R_linespace = numpy.uint64(coeff_R[0]*y_linespace**3 + coeff_R[1]*y_linespace**2 + coeff_R[2]*y_linespace + coeff_R[3]*y_linespace**0)
    
    # remove pixel outside of frame
    mask_out_of_frame_L = (x_L_linespace < 0) | (x_L_linespace > frame_width - 1)
    y_linespace_L = y_linespace[~mask_out_of_frame_L]
    x_L_linespace = x_L_linespace[~mask_out_of_frame_L]
    
    mask_out_of_frame_R = (x_R_linespace < 0) | (x_R_linespace > frame_width - 1)
    y_linespace_R = y_linespace[~mask_out_of_frame_R]
    x_R_linespace = x_R_linespace[~mask_out_of_frame_R]

    # draw zone betwween curves
    points_ =  [numpy.int_(list(zip(numpy.append(x_L_linespace, list(reversed(x_R_linespace))), numpy.append(y_linespace_L, list(reversed(y_linespace_R))))))]
    cv2.fillPoly(frame_RGB, points_, fill_color)
    
    return frame_RGB