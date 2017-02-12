'''
    **********************************************************************************
    Author:   Ilker GURCAN / Sidharth SADANI
    Date:     2/10/17
    File:     alg_steps
    Comments: Each step specified in the paper which implementation is based on, is
     coded in this file.
    **********************************************************************************
'''

# NOTE: Move to global import, this is just for localized testing
import cv2
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
    row_size, col_size = discarded_px.shape
    # NOTE: Change the req_num_loc = 25
    req_num_loc = 25
    loc_ct = 0
    loc_list = list()
    while loc_ct<req_num_loc:
        rand_row = np.random.randint(0,row_size)
        rand_col = np.random.randint(0,col_size)
        exists = False
        # Neglecting Masked Out Locations
        if discarded_px[rand_row][rand_col] == 0:
            continue
        # Neglecting Already Read Locations
        for r,c in loc_list:
            if rand_row == r and rand_col == c:
                exists = True
                break
        if exists:
            continue

        loc_list.append((rand_row, rand_col))
        loc_ct += 1
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
    # print row_size, col_size, "THIS"
    rgb_means = np.empty([0,3], dtype = float)
    for i,j in locations:
        # print "(", i , "," ,j, ")", img_domain[i][j]
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
                pixel_ct += 1
                patch_sum = patch_sum + img_dom32[ne_i][ne_j]
        patch_mean = patch_sum/pixel_ct
        rgb_means = np.vstack((rgb_means,patch_mean))
            
#        print "(", i , "," ,j, ")", patch_sum
#        print "(", i , "," ,j, ")", patch_mean, "RGB Mean"
        
    rgb_means = np.uint8(rgb_means)
    # print rgb_means
    rgb_means3D = rgb_means[np.newaxis, :, :]
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
    in_window = np.zeros([n_pts, n_pts], dtype=np.int32)
    rad = np.int32(r)
    px_ct = list()
    # print img_luv.shape, img_luv_hist.shape, rad
    print("Radius: "+str(rad))
    for i in range(len(center_locs)):
        l = center_locs[i][0]
        u = center_locs[i][1]
        v = center_locs[i][2]
        l_l = 0 if l-rad<=0 else l-rad
        u_l = 255 if l+rad>=255 else l+rad
        l_u = 0 if u-rad<=0 else u-rad
        u_u = 255 if u+rad>=255 else u+rad
        l_v = 0 if v-rad<=0 else v-rad
        u_v = 255 if v+rad>=255 else v+rad
        # print l_l, u_l, l_u, u_u, l_v, u_v
        '''
        # Uncomment this if you want to enforce circular window, much slower
        pixel_ct = 0
        for p in range(l_l,u_l+1):
            for q in range(l_u,u_u+1):
                for r in range(l_v, u_v+1):
                    d_vec = np.array((p-l,q-u,r-v))
                    dist = np.linalg.norm(d_vec)
                    if(dist<=rad):
                        pixel_ct = pixel_ct + img_luv_hist[p,q,r]
        '''
    # NOTE: Using a square window approximation leads to a much faster result
    # if you want to use a circular window approx uncomment the above part of the code
    # but do note, that the results don't change very much
        pixel_ct = np.sum(img_luv_hist[l_l:u_l+1,l_u:u_u+1,l_v:u_v+1])
        px_ct.append(pixel_ct)
    #    print center_locs[i], pixel_ct
    # print px_ct
    idx = np.argmax(px_ct)
    print("Initial Search Location: ", center_locs[idx], idx, px_ct[idx])

    return center_locs[idx], np.int32(px_ct[idx])


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

    print("Beginning Mean Shift")
    rad = np.int32(r)  # Rounding the radius to allow for indexing

    init_loc = start_loc
    # Square Matrix of size R containing On pixels for circles of Radius R
    # So that distances don't need to be computed for each iteration of mean shift
    Idx_Mat = np.zeros([2 * rad + 1, 2 * rad + 1, 2 * rad + 1], dtype=np.uint8)
    for i in range(0, rad + 1):
        for j in range(0, rad + 1):
            for k in range(0, rad + 1):
                dist = (rad - i) ^ 2 + (rad - j) ^ 2 + (rad - k) ^ 2
                dist_rt = np.sqrt(dist)
                if dist_rt <= rad:
                    Idx_Mat[i][j][k] = 1
                    Idx_Mat[i][j][2 * rad - k] = 1
                    Idx_Mat[i][2 * rad - j][k] = 1
                    Idx_Mat[i][2 * rad - j][2 * rad - k] = 1
                    Idx_Mat[2 * rad - i][j][k] = 1
                    Idx_Mat[2 * rad - i][j][2 * rad - k] = 1
                    Idx_Mat[2 * rad - i][2 * rad - j][k] = 1
                    Idx_Mat[2 * rad - i][2 * rad - j][2 * rad - k] = 1

    # Begin iteration
    n_iter = 0
    while True:
        n_iter += 1
        l = init_loc[0]
        u = init_loc[1]
        v = init_loc[2]

        l_l = 0 if l - rad <= 0 else l - rad
        u_l = 255 if l + rad >= 255 else l + rad
        l_u = 0 if u - rad <= 0 else u - rad
        u_u = 255 if u + rad >= 255 else u + rad
        l_v = 0 if v - rad <= 0 else v - rad
        u_v = 255 if v + +rad >= 255 else v + rad

        pmf_sum = 0
        pmf_wt_sum = np.array((0, 0, 0))
        for p in range(l_l, u_l + 1):
            for q in range(l_u, u_u + 1):
                for r in range(l_v, u_v + 1):
                    if Idx_Mat[p - l + rad][q - u + rad][r - v + rad] == 1:
                        pmf_sum = pmf_sum + img_luv_hist[p][q][r]
                        pmf_wt_sum += np.array((p, q, r)) * img_luv_hist[p][q][r]
                    else:
                        continue

        new_loc = pmf_wt_sum / pmf_sum
        change_vec = init_loc - new_loc
        change = np.linalg.norm(change_vec)

        print("Predicted New Loc, Change Vec and Mag Change: ", new_loc, change_vec, change)

        # Stopping Criterion
        # NOTE: Stopping Criterion in paper makes no sense without histogram bin size as quantization error
        # might exceed the given 0.1
        if change > 10:
            init_loc = np.int32(new_loc)
        else:
            break

    final_loc = init_loc
    print("Num Iterations: ", n_iter, "Final Mode/Feature: ", final_loc)
    return final_loc


def remove_det_feat(feats_covered, cur_mode, discarded_px, mode_alloc):
    '''
    Removes features that belong to current mode; then updates discarded
     pixels and mode allocation table accordingly.

    :param feats_covered: Features covered by the search window.
     Each feature is a pair of (x,y) location
    :param cur_mode: Id for current mode to be used with mode_alloc table
    :param discarded_px: A 2D mask storing non-discarded pixels as 1
    :param mode_alloc: A 2D table storing allocated pixels for each mode
    :return: Updated discarded_px, mode_alloc
    '''

    return None, None


def det_init_palette(mode_alloc, n_min, num_modes):
    '''
    Defines initial palette based on: min number of elements in feature space and
     minimum number of elements in an area formed by 8-connected components

    :param mode_alloc: Allocation table that defines mapping between pixels and modes
    :param n_min: Minimum number of elements (pixels) required to be a mode in
     feature space
    :param num_modes: Total number of modes
    :return: Significant color palette, Modes which fail to become an actual mode
    '''
    palette = {}
    failed = []
    modes, counts = np.unique(mode_alloc, return_counts=True)
    mode_counts = dict(zip(modes, counts))
    for cur_mode in range(num_modes):
        if mode_counts[cur_mode] < n_min:
            failed.append(cur_mode)
            continue
        idx_out = mode_alloc != cur_mode  # Pixels not belong to the current mode
        img_bin = np.ones(mode_alloc.shape, dtype=np.uint8)
        img_bin[idx_out] = 0
        res = cv2.connectedComponentsWithStats(img_bin, 8, cv2.CV_32S)
        stats = res[2]  # num_labels, labels, stats, centroids--How res is organized
        # Check whether any of the connected components in this mode has exceeded n_min
        if np.any(stats[:, cv2.CC_STAT_AREA] > n_min):
            palette[cur_mode] = np.sum(stats[:, cv2.CC_STAT_AREA])
        else:
            failed.append(cur_mode)

    return palette, failed

