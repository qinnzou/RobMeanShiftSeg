#!/usr/local/bin/python
# -*- coding: utf-8 -*-
'''
    **********************************************************************************
    Author:   Ilker GURCAN / Sidharth SADANI
    Date:     2/10/17
    File:     utils
    Comments: Helper functions go here.
    **********************************************************************************
'''

import cv2
import numpy as np
import math


def rgb_2_luv(img_rgb):
    '''
    In case of 8 bit images, R, G, and B are converted to the floating-point
     format and scaled to fit 0 and 1. This outputs 0≤L≤100, −134≤u≤220, −140≤v≤122.
     Then, 8-bit images are converted to L←255/100L,u←255/354(u+134),v←255/262(v+140).

    :param img_rgb: RGB image (numpy array) either 8-bit unsigned-int or 32-bit floating-point
    :return: Luv image with the type input image has
    '''

    print("Image type: %s.\n"
          "Creating Luv Space Representation..."
          % img_rgb.dtype.name)
    print()
    # According to OpenCV documentation, images with type of floating-point is not scaled to the
    # range of [0, 1]. It must be done manually, before calculating its representation
    # in Luv space.
    if img_rgb.dtype == np.float32:
        min_val, max_val, _, _ = cv2.minMaxLoc(img_rgb.ravel())
        if min_val < 0 or max_val > 1.0:
            cv2.normalize(img_rgb, img_rgb, 0.0, 1,0, cv2.NORM_MINMAX, cv2.CV_32FC3)
    luv_img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2Luv)
    return luv_img


def comp_image_pow(img_luv):
    '''
    Calculates the square root of covariance matrix's trace.
     It is related to the power of the signal (image).
    :param img_luv: Representation of the image in Luv space.
    :return: Power of the image
    '''

    # Each Luv vector (i.e. pixel) is our data point,
    # and covariance between these 3 channels is calculated.
    l = img_luv[:, :, 0].ravel()
    u = img_luv[:, :, 1].ravel()
    v = img_luv[:, :, 2].ravel()
    luv = np.column_stack((l, u, v)).transpose()
    cov_m = np.cov(luv)
    return math.sqrt(np.trace(cov_m))
