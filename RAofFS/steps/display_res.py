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
    mode_disp = ((mode_alloc+1)*255)/num_modes
    mode_disp = np.uint8(mode_disp)
    # cv2.imshow("Mask", discarded_px*255)
    cv2.imshow("Segmentation Result", mode_disp)