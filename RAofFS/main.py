'''
    **********************************************************************************
    Author:   Ilker GURCAN / Sidharth SADANI
    Date:     2/10/17
    File:     main
    Comments: This file will have the main loop for
     "Robust Analysis of Feature Spaces: Color Image Segmentation". It is also entry
     point for the application.
    **********************************************************************************
'''


import argparse

from steps.display_res import *
from steps.alg_steps import *


# Configuration options
CONFIG = None


def main():

    print("Welcome to Robust Analysis of Feature Spaces Demonstration!")
    print("OpenCV Version: %s\n" % cv2.__version__)

    # Read the input image
    print("Input Image File: %s\n" % CONFIG.file)
    img_rgb = cv2.imread(CONFIG.file)
    no_pixels = img_rgb.shape[0]*img_rgb.shape[1]

    ## Demonstration Purpose Display
    cv2.imshow("Original Image: ", img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convert the image to Luv space, which is our "Feature Space"
    img_luv = rgb_2_luv(img_rgb)

    # Compute the power of the image
    pow_im = comp_image_pow(img_luv)
    print("Power of the Image: %.4f\n" % pow_im)

    # Choose length of the search window's radius
    r, n_min, n_con = pick_radius(pow_im, CONFIG.radiusOp)
    print("The search window's radius: %.3f\n" % r)

    # Main Loop begins
    print("Main Loop begins...")
    print("***************************************\n")
    cur_mode = -1
    discarded_px = np.ones([img_luv.shape[0], img_luv.shape[1]], dtype=np.uint8)
    mode_alloc = np.ones([img_luv.shape[0], img_luv.shape[1]], dtype=np.int32) * -1

    init_feat_pal = list()

    # for initialization of the while loop
    num_feat_final = 0
    no_free_pixels = no_pixels
    while cur_mode == -1 or num_feat_final > n_min:
        cur_mode += 1
        print("Running segmentation to construct mode: %d" % cur_mode)
        # Run mean shift algorithm from OpenCV
        
        # Compute LUV Space Histogram
        img_luv_hist = cv2.calcHist([img_luv], [0,1,2], discarded_px,[256,256,256],[0,256,0,256,0,256])
        
        # Initial search window
        sw_cand = pick_rand_locs(discarded_px)
        cand_in_fs = extract_centroids(sw_cand, img_rgb)
        sw, num_feat = find_sw(cand_in_fs, img_luv, img_luv_hist, r, discarded_px)

         # Initial Mean Shift Iteration
        feat_ctr, Idx_Mat = comp_mean_shift(sw, img_luv_hist, r, 0.1)
        feats_covered = None

        # Remove detected features from both spaces (image+feature)
        discarded_px, mode_alloc, num_feat_final = remove_det_feat(feat_ctr,
                                                                   cur_mode,
                                                                   discarded_px,
                                                                   mode_alloc,
                                                                   img_luv,
                                                                   Idx_Mat,
                                                                   r)

        init_feat_pal.append(feat_ctr)
        no_free_pixels = no_free_pixels - num_feat_final

        ## Stepwise Image Display
        mode_disp = ((mode_alloc+1)*255)/(cur_mode+1)
        mode_disp = np.uint8(mode_disp)
        # cv2.imshow("Original Image", img_rgb)
        cv2.imshow("Current Segmentation Result", mode_disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if no_free_pixels <= 40:
            break
        # if cur_mode == 4:
        #    break
    print("Main Loop ends...")
    print("***************************************")
    # End of while-Loop

    # The list of initial feature centers
    print("Initial Feature Palette: %s \nNum Modes: %d" % (str(list(init_feat_pal)), len(init_feat_pal)))
    
    # Defining Feature-Palette
    palette = det_init_palette(mode_alloc, n_min, cur_mode+1)
    init_pal_alloc = np.copy(mode_alloc)
    for p in range(cur_mode+1):
        if not (p in palette):
            not_mode_idx = mode_alloc == p
            init_pal_alloc[not_mode_idx] = -1
    disp_res(img_rgb, init_pal_alloc, len(palette))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    mode_alloc = det_fin_palette(r,
                                 init_feat_pal,
                                 img_luv,
                                 mode_alloc,
                                 cur_mode+1)
    disp_res(img_rgb, mode_alloc, cur_mode+1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Post-processing


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Add all configuration options here.
    parser.add_argument(
        '--file',
        type=str,
        default='../test1.jpg',
        help='Image file on which RAofFS will be applied'
    )
    parser.add_argument(
        '--radiusOp',
        type=str,
        default=OVER_SEG,
        help='Option for choosing search window\'s size. Options are:\n'
             + UNDER_SEG + '\n'
             + OVER_SEG + '\n'
             + QUANT + '\n'
             'Default is ' + OVER_SEG
    )
    CONFIG, _ = parser.parse_known_args()
    main()
