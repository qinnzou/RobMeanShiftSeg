'''
    **********************************************************************************
    Author:   Ilker GURCAN / Sidharth SADANI
    Date:     2/10/17
    File:     alg_steps
    Comments: Each step specified in the paper which implementation is based on, is
     coded in this file.
    **********************************************************************************
'''

#NOTE: Move to global import, this is just for localized testing
import numpy as np
from utils.utils import *

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
    :return: A list of python Tuple of 25 random locations picked from
     among non-discarded pixels
    '''
    row_size, col_size  = discarded_px.shape
    #NOTE: Change the req_num_loc = 25
    req_num_loc = 25
    loc_ct = 0
    loc_list = list()
    while(loc_ct<req_num_loc):
        rand_row = np.random.randint(0,row_size)
        rand_col = np.random.randint(0,col_size)
        exists = False
       # Neglecting Masked Out Locations
        if(discarded_px[rand_row][rand_col] == 0):
            continue

        # Neglecting Already Read Locations
        for r,c in loc_list:
            if rand_row == r and rand_col == c:
                exists = True
                break
        if exists:
            continue

        loc_list.append((rand_row,rand_col))

        loc_ct = loc_ct + 1
    # return None
    # return row_size, col_size
    # print loc_list
    return loc_list


def extract_centroids(locations, img_domain):
    '''
    Extracts 3x3 patches centered around points given by :parameter locations
     and then averages all pixels in each patch to compute every mean.
     Finally, it maps all these mean values into Luv space

    :param locations: (x,y) pair for each location within the feature space
    :param img_domain: 3D image domain
    :return: python Tuple of size 25x3 whose each row is a vector in Luv space
    '''
    row_size, col_size, _ = img_domain.shape
#    print row_size, col_size, "THIS"
    rgb_means = np.empty([0,3], dtype = float)
    for i,j in locations:
#        print "(", i , "," ,j, ")", img_domain[i][j]
        # Computing the mean of the 3x3 windows centered around each of these locations
        pixel_ct = 0
        patch_sum = np.array((0,0,0), dtype = float);
        img_dom32 = np.int32(img_domain)
        for dx in range(-1,2):
            ne_i = i + dx   # Neighbor index, row_coord
            if ne_i < 0 or ne_i >= row_size:
                continue
            for dy in range(-1,2):
                ne_j = j + dy   # Neighbor index col_coord
                if ne_j < 0 or ne_j >= col_size:
                    continue
                pixel_ct = pixel_ct + 1
                patch_sum = patch_sum + img_dom32[ne_i][ne_j]
        patch_mean = patch_sum/pixel_ct
        rgb_means = np.vstack((rgb_means,patch_mean))
            
#        print "(", i , "," ,j, ")", patch_sum
#        print "(", i , "," ,j, ")", patch_mean, "RGB Mean"
        
    rgb_means = np.uint8(rgb_means)
    # print rgb_means
    rgb_means3D = rgb_means[np.newaxis,:,:]
    # print rgb_means3D, rgb_means3D.shape
    luv_means3D = rgb_2_luv(rgb_means3D)
    # print luv_means3D
    L = luv_means3D[0,:,0]
    U = luv_means3D[0,:,1]
    V = luv_means3D[0,:,2]

    # print L, U, V
    LUV_list = zip(L,U,V)
    # print LUV_list

    # return None
    return LUV_list

# def find_sw(center_locs, img_luv, luv_hist, r, discarded_px)
def find_sw(center_locs, img_luv, img_luv_hist, r, discarded_px):
    '''
    Finds the search window containing the highest density of feature vectors.

    :param center_locs: Center location for each candidate search window in feature space
    :param img_luv: 3D feature space
    :param r: The length of search window's radius
    :return: a python Tuple of size 1x3 representing the vector for search window,
     # of features covered by this search window
    '''
    n_pts = len(center_locs)
    inWindow = np.zeros([n_pts, n_pts], dtype=np.int32)
    Rad = np.int32(r)
    px_ct = list()
 #   print img_luv.shape, img_luv_hist.shape, Rad
    print "Radius: ", Rad
    for i in range(len(center_locs)):
        l = center_locs[i][0]
        u = center_locs[i][1]
        v = center_locs[i][2]
        l_l = 0 if l-Rad<=0 else l-Rad
        u_l = 255 if l+Rad>=255 else l+Rad
        l_u = 0 if u-Rad<=0 else u-Rad
        u_u = 255 if u+Rad>=255 else u+Rad
        l_v = 0 if v-Rad<=0 else v-Rad
        u_v = 255 if v++Rad>=255 else v+Rad
     #   print l_l, u_l, l_u, u_u, l_v, u_v
        '''
        # Uncomment this if you want to enforce circular window, much slower
        pixel_ct = 0
        for p in range(l_l,u_l+1):
            for q in range(l_u,u_u+1):
                for r in range(l_v, u_v+1):
                    d_vec = np.array((p-l,q-u,r-v))
                    dist = np.linalg.norm(d_vec)
                    if(dist<=Rad):
                        pixel_ct = pixel_ct + img_luv_hist[p,q,r]
        '''
    #NOTE: Using a square window approximation leads to a much faster result
    # if you want to use a circular window approx uncomment the above part of the code
    # but do note, that the results don't change very much
        pixel_ct = np.sum(img_luv_hist[l_l:u_l+1,l_u:u_u+1,l_v:u_v+1])
        px_ct.append(pixel_ct)
    #    print center_locs[i], pixel_ct
    # print px_ct
    idx = np.argmax(px_ct)
    print "Initial Search Location: ", center_locs[idx], idx, px_ct[idx]

    return center_locs[idx], np.int32(px_ct[idx])


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
