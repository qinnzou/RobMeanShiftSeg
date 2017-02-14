'''
    **********************************************************************************
    Author:   Ilker GURCAN / Sidharth SADANI
    Date:     2/10/17
    File:     display_res
    Comments: This will have methods to display the output of each major step in the
     algorithm.
    **********************************************************************************
'''

import cv2
import numpy as np


def disp_res(original, mode_alloc, num_modes):

    cv2.imshow("Original Image", original)

    R = np.zeros([mode_alloc.shape[0], mode_alloc.shape[1]], dtype = np.uint8)
    G = np.zeros([mode_alloc.shape[0], mode_alloc.shape[1]], dtype = np.uint8)
    B = np.zeros([mode_alloc.shape[0], mode_alloc.shape[1]], dtype = np.uint8)

    col_dict = dict()

    for i in range(mode_alloc.shape[0]):
        for j in range(mode_alloc.shape[1]):
            idx = mode_alloc[i][j]
            try:
                R[i][j] = col_dict[idx][0]
                G[i][j] = col_dict[idx][1]
                B[i][j] = col_dict[idx][2]
            except:
                col_dict[idx] = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
                R[i][j] = col_dict[idx][0]
                G[i][j] = col_dict[idx][1]
                B[i][j] = col_dict[idx][2]

#    mode_disp = ((mode_alloc+1)*255)/num_modes
#    mode_disp = np.uint8(mode_disp)
    new_mode_disp = cv2.merge((B,G,R))
    # cv2.imshow("Mask", discarded_px*255)
    # cv2.imshow("Segmentation Result", mode_disp)
    cv2.imshow("Segmentation Result", new_mode_disp)
