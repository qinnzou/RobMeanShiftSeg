from __future__ import absolute_import
import cv2
import numpy as np
from steps.alg_steps import *

# This is just a tester file to try and use indvidual modules
# Run this from the main directory : as follows python -m steps.Sandbox.py

from utils.utils import *

#print "Hello World"

cx = cv2.imread('steps/test2.png')
print("Image Dimensions: ", cx.shape)
dcx = np.ones([cx.shape[0], cx.shape[1]], dtype = np.bool)

a1 = np.array((1,2,3,16,17))
b1 = np.array((5,6,7,18,0))
a2 = np.array((0,0,0,19,20))
b2 = np.array((8,9,0,21,0))
a3 = np.array((10,11,12,0,22))
b3 = np.array((13,14,15,23,24))
c = np.vstack((a1,b1,a2,b2,a3,b3))

print(c)
d = c[3:5,1:3]
print(d, np.sum(d))

#loclist = pick_rand_locs(dcx)

#LUV_Cx = extract_centroids(loclist,cx)

#print LUV_Cx

#print "Size, of the matrix", d, e

##rn = np.random.random_integers(1,d,1)

#rn = np.random.randint(0,d)

#print rn
