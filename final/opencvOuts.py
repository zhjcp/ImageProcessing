import cv2
import numpy as np

# 函数：灰度化
def make_gray(image):
    """
    RGB转灰度计算公式：
        Gray(i,j) = 0.299 * R(i,j) + 0.587 * G(i,j) + 0.114 * B(i,j)
    """
    b = image[:, :, 0].copy()
    g = image[:, :, 1].copy()
    r = image[:, :, 2].copy()
    # RGB转灰度
    gray_image = 0.299 * r + 0.587 * g + 0.114 * b
    gray_image = gray_image.astype(np.uint8)
    # 返回灰度图像
    return gray_image


origin_img = cv2.imread('otusTest1.jpg')
cv2.imshow('origin_img', origin_img)
gray_img = make_gray(origin_img)
ret, otsu_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
print(ret)
cv2.imshow('opencv_otsu_img', otsu_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
