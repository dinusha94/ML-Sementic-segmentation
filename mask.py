import cv2
import numpy as np
import os
import sys

im =  cv2.imread("data_mask/0.jpg", cv2.IMREAD_COLOR)
print im.shape
Conv = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imshow('imag1',Conv)
print Conv.shape
ret, mask = cv2.threshold(Conv, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
print mask.shape
cv2.imshow('image',mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
frame = 0
for i in range(len(list1)):
    im =  cv2.imread("data_mask/"+str(frame)+".jpg", cv2.IMREAD_COLOR)
    Conv = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)

    indices = np.where(mask==255)
    indices1 = np.where(mask==0)

    im[indices[0], indices[1], :] = [0, 0, 255]
    im[indices1[0], indices1[1], :] = [255, 0, 0]

    cv2.imwrite("data_mask_1/"+str(frame)+".jpg", im)
    frame += 1

"""
