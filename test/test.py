# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 16:23:05 2020

@author: 朱华
"""

# 实验目的：使用Matplotlib 中的绘图函数绘制灰度直方图
import cv2
from matplotlib import pyplot as plt

# 实现方式2：只使用matplotlib 的绘图功能，这在同时绘制多通道（BGR）的直方图
# 直方图的统计仍然由opencv或者numpy实现
img = cv2.imread('flower3.jpg')
color = ('b', 'g', 'r')
# 对一个列表或数组既要遍历索引又要遍历元素时
# 使用内置enumerate 函数会有更加直接，优美的做法
# enumerate 会将数组或列表组成一个索引序列。
# 使我们再获取索引和索引内容的时候更加方便
# i 索引 col索引项（颜色）
for i, col in enumerate(color):  # rgb三个通道
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
 
plt.xlim([0, 256])
plt.show()
'''
函数plt.plot的参数和作用:
    这里传入的是 一维数组（直方图的统计结果）+颜色
    
'''

'''
函数plt.xlim的参数和作用：
    参数同上
    设置x轴的数值显示范围
'''