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


# 函数：大津算法
def otsu(gray_image):
    """
        h：图像的宽度
        w：图像的高度
        （h*w 得到图像的像素数量）
        threshold_t ：灰度阈值（我们要求的值，大于这个值的像素我们将它的灰度设置为255，小于的设置为0）
        n0：小于阈值的像素数量，前景
        n1：大于等于阈值的像素数量，背景
        n0 + n1 == h * w
        w0：前景像素数量占总像素数量的比例
        w0 = n0 / (h * w)
        w1：背景像素数量占总像素数量的比例
        w1 = n1 / (h * w)
        w0 + w1 == 1
        u0：前景平均灰度
        u0 = 前景灰度累加和 / n0
        u1：背景平均灰度
        u1 = 背景灰度累加和 / n1
        u：平均灰度
        u = (前景灰度累加和 + 背景灰度累加和) / (h * w)
        u = w0 * u0 + w1 * u1
        g：类间方差（那个灰度的g最大，哪个灰度就是需要的阈值threshold_t）
        g = w0 * (u0 - u)^2 + w1 * (u1 - u)^2
        根据上面的关系，可以推出：
        g = w0 * w1 * (u0 - u1) ^ 2
    """
    h = gray_image.shape[0]
    w = gray_image.shape[1]
    threshold_t = 0
    temp_t = 0

    # 遍历每一个灰度值
    for t in range(255):
        # 使用numpy直接对数组进行运算（注：其中包含了一些直方图的操作）
        n0 = gray_image[np.where(gray_image < t)]   # np.where：利用条件筛选前景和背景像素
        n1 = gray_image[np.where(gray_image >= t)]
        w0 = len(n0) / (h * w)  # len：返回数组长度，即：像素数量
        w1 = len(n1) / (h * w)
        u0 = np.mean(n0) if len(n0) > 0 else 0.  # np.mean：求平均值
        u1 = np.mean(n1) if len(n0) > 0 else 0.
        g = w0 * w1 * (u0 - u1) * (u0 - u1)  # 求类间方差
        # 判断此次遍历得到的方差是否足够大
        if g > temp_t:
            temp_t = g
            threshold_t = t
    print('类间方差最大时对应阈值：', threshold_t)
    # 返回otus的结果阈值
    return threshold_t


origin_img = cv2.imread('otusTest1.jpg')
cv2.imshow('origin_img', origin_img)

# 1. 将数据转换成float32
img = origin_img.astype(np.float32)
# 2. 灰度化
gray_image = make_gray(image=img)
# 3. 执行outs算法并进行阈值处理
best_threshold_t = otsu(gray_image=gray_image)
# 4. 使用outs的结果进行阈值处理
otsu_img = gray_image
otsu_img[otsu_img < best_threshold_t] = 0  # 较小
otsu_img[otsu_img >= best_threshold_t] = 255  # 较大
cv2.imshow('otsu_img ', otsu_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
