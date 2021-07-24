import math
import cv2
import numpy as np
from matplotlib import pyplot as plt


# 函数：灰度化
def make_gray(img_url):
    """
    RGB转灰度计算公式：
        Gray(i,j) = 0.299 * R(i,j) + 0.587 * G(i,j) + 0.114 * B(i,j)
    """
    # 读取图片资源gv
    img = plt.imread(img_url)
    # BGR 转换成 RGB 格式
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 灰度化
    gray_img = np.dot(img_rgb[..., :3], [0.299, 0.587, 0.114])
    # 返回灰度化后的图像
    return gray_img


# 函数：去除噪音 - 使用 5x5 的高斯滤波器
def smooth(gray_img):
    """
    要生成一个 (2k+1)x(2k+1) 的高斯滤波器：
    滤波器的各个元素计算公式如下：
    H[i, j] = (1/(2*pi*sigma**2))*exp(-1/2*sigma**2((i-k-1)**2 + (j-k-1)**2))
    """
    sigma1 = sigma2 = 1.4
    gau_sum = 0
    gaussian = np.zeros([5, 5])
    for i in range(5):
        for j in range(5):
            gaussian[i, j] = math.exp((-1 / (2 * sigma1 * sigma2)) * (np.square(i - 3)
                                                                      + np.square(j - 3))) / (
                                     2 * math.pi * sigma1 * sigma2)
            gau_sum = gau_sum + gaussian[i, j]

    # 归一化处理
    gaussian = gaussian / gau_sum

    # 高斯滤波
    width, height = gray_img.shape
    gray_after_guasss = np.zeros([width - 5, height - 5])
    for i in range(width - 5):
        for j in range(height - 5):
            gray_after_guasss[i, j] = np.sum(gray_img[i:i + 5, j:j + 5] * gaussian)

    # 返回高斯滤波之后的图像
    return gray_after_guasss


# 函数：计算梯度幅值
def gradients(gray_smooth):
    width, height = gray_smooth.shape
    dx = np.zeros([width - 1, height - 1])
    dy = np.zeros([width - 1, height - 1])
    m = np.zeros([width - 1, height - 1])  # 梯度幅度
    theta = np.zeros([width - 1, height - 1])

    for i in range(width - 1):
        for j in range(height - 1):
            dx[i, j] = gray_smooth[i + 1, j] - gray_smooth[i, j]
            dy[i, j] = gray_smooth[i, j + 1] - gray_smooth[i, j]
            # 图像梯度幅作为图像强度
            m[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))
            # 计算梯度方向 artan(dx/dy)
            theta[i, j] = math.atan(dx[i, j] / (dy[i, j] + 0.000000001))

    #  返回梯度值
    return dx, dy, m, theta


# 函数：非极大抑制
def do_nms(m, dx, dy):
    d = np.copy(m)
    width, height = m.shape
    nms = np.copy(d)
    nms[0, :] = nms[width - 1, :] = nms[:, 0] = nms[:, height - 1] = 0

    for i in range(1, width - 1):
        for j in range(1, height - 1):

            # 如果当前梯度为0，该点就不是边缘点
            if m[i, j] == 0:
                nms[i, j] = 0

            else:
                grad_x = dx[i, j]  # 当前点 x 偏导
                grad_y = dy[i, j]  # 当前点 y 偏导
                grad_temp = d[i, j]  # 当前点 梯度

                # 导数方向趋向于 y 分量（如果 y 方向梯度值比较大，显然偏向y分量）
                if np.abs(grad_y) > np.abs(grad_x):
                    weight = np.abs(grad_x) / np.abs(grad_y)  # 权重
                    grad2 = d[i - 1, j]
                    grad4 = d[i + 1, j]

                    # 如果 x, y 方向导数符号一致
                    # 像素点位置关系
                    # g1 g2
                    #    c
                    #    g4 g3
                    if grad_x * grad_y > 0:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]

                    # 如果 x，y 方向导数符号相反
                    # 像素点位置关系
                    #    g2 g1
                    #    c
                    # g3 g4
                    else:
                        grad1 = d[i - 1, j + 1]
                        grad3 = d[i + 1, j - 1]

                # 如果 x 方向梯度值比较大
                else:
                    weight = np.abs(grad_y) / np.abs(grad_x)
                    grad2 = d[i, j - 1]
                    grad4 = d[i, j + 1]

                    # 如果 x, y 方向导数符号一致
                    # 像素点位置关系
                    #      g3
                    # g2 c g4
                    # g1
                    if grad_x * grad_y > 0:

                        grad1 = d[i + 1, j - 1]
                        grad3 = d[i - 1, j + 1]

                    # 如果 x，y 方向导数符号相反
                    # 像素点位置关系
                    # g1
                    # g2 c g4
                    #      g3
                    else:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]

                # 利用 grad1-grad4 对梯度进行插值
                grad_temp1 = weight * grad1 + (1 - weight) * grad2
                grad_temp2 = weight * grad3 + (1 - weight) * grad4

                # 当前像素的梯度幅度是局部最大，则认为是边缘点
                if grad_temp >= grad_temp1 and grad_temp >= grad_temp2:
                    nms[i, j] = grad_temp

                else:
                    # 否则，认为不可能是边缘点，置0（经过后面的阈值处理，这一部分相当于被舍弃了）
                    nms[i, j] = 0
    #  返回非极大抑制后的结果
    return nms


#  函数：双阈值选取
def double_threshold(nms_res):
    weight, height = nms_res.shape
    res = np.zeros([weight, height])

    # 定义低高阈值（tl和th一般1:2或1:3）
    tl = 0.13 * np.max(nms_res)
    th = 0.39 * np.max(nms_res)
    print(tl)
    print(th)

    # 利用双阈值进行处理
    for i in range(1, weight - 1):
        for j in range(1, height - 1):
            # 双阈值选取
            # 过小舍弃
            if nms_res[i, j] < tl:
                res[i, j] = 0
            # 过大保留
            elif nms_res[i, j] > th:
                res[i, j] = 1

            # 处理中间：连接到可靠边缘 ，则认为属于边缘
            elif (nms_res[i - 1, j - 1:j + 1] < th).any() \
                    or (nms_res[i + 1, j - 1:j + 1].any()
                        or (nms_res[i, [j - 1, j + 1]] < th).any()):
                res[i, j] = 1

    # 返回双阈值处理之后的图像
    return res


# 测试代码：测试自定义canny算法
# 1. 灰度化
_gray = make_gray(img_url='cannyTest1.jpg')
# 2. 高斯滤波
_smooth = smooth(gray_img=_gray)
# 3. 计算梯度
_dx, _dy, _M, _theta = gradients(gray_smooth=_smooth)
# 4. 非极大抑制
_NMS = do_nms(m=_M, dx=_dx, dy=_dy)
# 5. 双阈值处理
_double_threshold = double_threshold(nms_res=_NMS)
# 双阈值处理的结果即是最终结果
_res = _double_threshold

_origin_img = plt.imread('cannyTest1.jpg')
cv2.imshow('_origin', _origin_img)
cv2.imshow('_res', _res)


cv2.waitKey(0)
cv2.destroyAllWindows()
