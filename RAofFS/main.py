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


from steps.alg_steps import *

# Configuration options
CONFIG = None


def main():

    print("Welcome to Robust Analysis of Feature Spaces Demonstration!")
    print("OpenCV Version: %s\n" % cv2.__version__)

    # Read the input image
    print("Input Image File: %s\n" % CONFIG.file)
    img_rgb = cv2.imread(CONFIG.file)

    # Convert the image to Luv space, which is our "Feature Space"
    img_luv = rgb_2_luv(img_rgb)

    # Compute the power of the image
    pow_im = comp_image_pow(img_luv)
    print("Power of the Image: %.4f\n" % pow_im)

    # Choose length of the search window's radius
    r, n_min, n_con = pick_radius(pow_im, CONFIG.radiusOp)
    print("The length of the search window's radius: %.3f\n" % r)

    # Main Loop begins
    print("Main Loop begins...")
    print("***************************************\n")
    cur_mode = 0
    discarded_px = np.ones([img_luv.shape[0], img_luv.shape[1]], dtype=np.uint8)
    mode_alloc = np.ones([img_luv.shape[0], img_luv.shape[1]], dtype=np.int32) * -1
    # Feature Space Histogram Compute the image histogram in Luv Space
    # img_luv_hist = cv2.calcHist([img_luv], [0,1,2], discarded_px,[101,201,201],[0,101,-100,101,-100,101])
    img_luv_hist = cv2.calcHist([img_luv], [0,1,2], discarded_px,[256,256,256],[0,256,0,256,0,256])
    # Initial search window
    sw_cand = pick_rand_locs(discarded_px)
    cand_in_fs = extract_centroids(sw_cand, img_rgb)
    sw, num_feat = find_sw(cand_in_fs, img_luv, img_luv_hist, r, discarded_px)
    ##### TEST MEAN SHIFT ########
    feat_ctr, Idx_Mat = comp_mean_shift(sw, img_luv_hist, r, 0.1)
    #############################

    #### TEST REMPVE DET FEATURE ########
    discarded_px, mode_alloc, num_feat_final = remove_det_feat(feat_ctr, cur_mode, discarded_px, mode_alloc, img_luv, Idx_Mat, r)
    #####################################
    cv2.imshow("Original", img_rgb)
    cv2.imshow("Mask", discarded_px*255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print num_feat, n_min, num_feat_final
'''
    while num_feat > n_min:
        print("Running segmentation to construct mode %d..." % cur_mode)
        # Run mean shift algorithm from OpenCV
        feats_covered = None
        # Remove detected features from both spaces (image+feature)
        # discarded_px, mode_alloc = remove_det_feat(feat_ctr,
                                                   cur_mode,
                                                   discarded_px,
                                                   mode_alloc, img_luv, Idx_Mat, r)
        # Determine next search window
        sw_cand = pick_rand_locs(discarded_px)
        cand_in_fs = extract_centroids(sw_cand, img_rgb)
        sw, num_feat = find_sw(cand_in_fs, img_luv, r)
        cur_mode += 1
    print("Main Loop ends...")
    print("***************************************")
    # End of while-Loop
    '''
    # Defining Feature-Palette

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
