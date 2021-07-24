# coding=utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt

_origin_img = plt.imread('cannyTest1.jpg')
cv2.imshow('_origin', _origin_img)
gray_img = _origin_img
img = cv2.GaussianBlur(gray_img, (5, 5), 0)
_cv_res_img = cv2.Canny(img, 18, 54)

cv2.imshow('opencv_Canny', _cv_res_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
