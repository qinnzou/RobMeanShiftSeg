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

def comp_mean_shift(start_loc, img_luv_hist, r, stop_crit):
    '''
    Computes location of nearest mode or feature in the distribution.
    Using the mean shift based approach.
    :param start_loc: initial location for search window 1x3 tuple
    :param img_luv_hist: histogram of masked image
    :param r: search window radius
    :param stop_crit: criterion for stopping mean shift iteration
    :return: final center location obtained from convergence of mean shift.
    '''
    
    print "Beginning Mean Shift"
    Rad = np.int32(r) # Rounding the radius to allow for indexing

    init_loc = start_loc
    # Square Matrix of size R containing On pixels for circles of Radius R
    # So that distances don't need to be computed for each iteration of mean shift
    Idx_Mat = np.zeros([2*Rad+1, 2*Rad+1, 2*Rad+1], dtype=np.uint8)
    for i in range(0, Rad+1):
        for j in range(0, Rad+1):
            for k in range(0, Rad+1):
                dist = (Rad-i)^2 + (Rad-j)^2 + (Rad-k)^2
                dist_rt = np.sqrt(dist)
                if(dist_rt<=Rad):
                    Idx_Mat[i][j][k] = 1
                    Idx_Mat[i][j][2*Rad-k] = 1
                    Idx_Mat[i][2*Rad-j][k] = 1
                    Idx_Mat[i][2*Rad-j][2*Rad-k]=1
                    Idx_Mat[2*Rad-i][j][k] = 1
                    Idx_Mat[2*Rad-i][j][2*Rad-k] = 1
                    Idx_Mat[2*Rad-i][2*Rad-j][k] = 1
                    Idx_Mat[2*Rad-i][2*Rad-j][2*Rad-k]=1

    # Begin iteration
    n_iter = 0
    while(True):
        n_iter = n_iter + 1
        l = init_loc[0]
        u = init_loc[1]
        v = init_loc[2]
        
        l_l = 0 if l-Rad<=0 else l-Rad
        u_l = 255 if l+Rad>=255 else l+Rad
        l_u = 0 if u-Rad<=0 else u-Rad
        u_u = 255 if u+Rad>=255 else u+Rad
        l_v = 0 if v-Rad<=0 else v-Rad
        u_v = 255 if v++Rad>=255 else v+Rad
        
        pmf_sum = 0
        pmf_wt_sum = np.array((0,0,0))
        for p in range(l_l, u_l+1):
            for q in range(l_u, u_u+1):
                for r in range(l_v, u_v+1):
                    if Idx_Mat[p-l+Rad][q-u+Rad][r-v+Rad] == 1:
                        pmf_sum = pmf_sum + img_luv_hist[p][q][r]
                        pmf_wt_sum = pmf_wt_sum + np.array((p,q,r))*img_luv_hist[p][q][r]
                    else:
                        continue
    
        new_loc = pmf_wt_sum/pmf_sum
        change_vec = init_loc - new_loc
        change = np.linalg.norm(change_vec)

        print "Predicted New Loc, Change Vec and Mag Change: ", new_loc, change_vec, change
        
        # Stopping Criterion
        # NOTE: Stopping Criterion in paper makes no sense without histogram bin size as quantization error
        # might exceed the given 0.1
        if(change > 10):
            init_loc = np.int32(new_loc)
        else:
            break

    final_loc = init_loc
    print "Num Iterations: ", n_iter, "Final Mode/Feature: ", final_loc
    return final_loc
