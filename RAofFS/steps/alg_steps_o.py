'''
    **********************************************************************************
    Author:   Ilker GURCAN / Sidharth SADANI
    Date:     2/10/17
    File:     alg_steps
    Comments: Each step specified in the paper which implementation is based on, is
     coded in this file.
    **********************************************************************************
'''


UNDER_SEG = 'underSegmentation'
OVER_SEG = 'overSegmentation'
QUANT = 'quantization'
SEG_TABLE = {UNDER_SEG: [0.4, 400, 10],
             OVER_SEG: [0.3, 100, 10],
             QUANT: [0.2, 50, 0]}


def pick_radius(pow_im, user_op):
    '''
    Picks up the radius for search window based on user's preference and image's power

    :param pow_im: Power of the image as calculated in utils.utils module
    :param user_op: One of the user option for choosing radius of the search window
    :return: Search window's radius, # of minimum samples, Minimum # of connected components
    '''
    if user_op is None:
        raise 'User option cannot be empty'
    return pow_im*SEG_TABLE[user_op][0], SEG_TABLE[user_op][1], SEG_TABLE[user_op][2]


def pick_rand_locs(discarded_px):
    '''

    :param discarded_px: A 2D mask storing non-discarded pixels as 1.
     It is a numpy array
    :return: A python Tuple of 25 random locations picked from
     among non-discarded pixels
    '''

    return None


def extract_centroids(locations, img_domain):
    '''
    Extracts 9x9 patches centered around points given by :parameter locations
     and then averages all pixels in each patch to compute every mean.
     Finally, it maps all these mean values into Luv space

    :param locations: (x,y) pair for each location within the feature space
    :param img_domain: 3D image domain
    :return: python Tuple of size 25x3 whose each row is a vector in Luv space
    '''

    return None


def find_sw(center_locs, img_luv, r):
    '''
    Finds the search window containing the highest density of feature vectors.

    :param center_locs: Center location for each candidate search window in feature space
    :param img_luv: 3D feature space
    :param r: The length of search window's radius
    :return: a python Tuple of size 1x3 representing the vector for search window,
     # of features covered by this search window
    '''

    return None, None


def remove_det_feat(feats_covered, cur_mode, discarded_px, mode_alloc):
    '''
    Removes features that belong to current mode; then updates discarded
     pixels and mode allocation table accordingly.

    :param feats_covered: Features covered by the search window.
     Each feature is a pair of (x,y) location
    :param cur_mode: Id for current mode to be used with mode_alloc table
    :param discarded_px: A 2D mask storing non-discarded pixels as 1
    :param mode_mask: A 2D table storing allocated pixels for each mode
    :return: Updated discarded_px, mode_alloc
    '''

    return None, None

