import numpy as py
import cv2

img = cv2.imread('test1.jpg')

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
