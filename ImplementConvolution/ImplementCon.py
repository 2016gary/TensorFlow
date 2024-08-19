"""
Created on Sun Jun 11 11:06:12 2017

@author: Gary
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 实现Convolutional Neural Network卷积操作
# 卷积核 使用轮廓滤波器FIND_EDGES作为示例
kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
# filter EDGE_ENHANCE_MORE
# kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
# 将RGBA彩色图片转为灰度图片
img = Image.open(
    "C:\\Users\\Gary\\Desktop\\Machine-Learning-Deep-Learning\\卷积神经网络（Convolutional Neural Network, CNN）\\ImplementConvolution\\testCon.png").convert(
    "L")
# 将图片转为矩阵
img_size = img.size
img_data = img.getdata()
img_mat = np.matrix(img_data)
img_mat = np.reshape(img_mat, (img_size[0], img_size[1]))
# 步长 strides=1
# 填充 0
padding = 1
# 先在矩阵左右添加两列
add_column = np.zeros(img_size[0])
img_mat = np.column_stack((add_column, img_mat))
img_mat = np.column_stack((img_mat, add_column))
# 再在矩阵上下添加两行
add_line = np.zeros(img_size[0] + padding * 2)
img_mat = np.row_stack((add_line, img_mat))
img_mat = np.row_stack((img_mat, add_line))
# 图片像素值做卷积运算
result_arr = []
for i in range(1, img_mat.shape[0] - 1):
    for j in range(1, img_mat.shape[1] - 1):
        arr = []
        for a in range(i - 1, i + 2):
            for b in range(j - 1, j + 2):
                arr.append(img_mat[a, b])
        data = np.matrix(arr)
        data = np.reshape(data, (kernel.shape[0], kernel.shape[1]))
        val = np.dot(data, kernel)
        result_arr.append(val.sum())

# 得到卷积操作后的矩阵
result_mat = np.reshape(result_arr, (img_mat.shape[0] - padding * 2, img_mat.shape[1] - padding * 2))
# Relu激活函数（The Rectified Linear Unit）f(x)=max(0,x)
result_mat = np.matrix(np.maximum(result_mat, 0))
# 矩阵转为图片
new_img = Image.fromarray(result_mat)
# 展示图片
plt.figure("testResult")
# 第一张图展示原图片
plt.subplot(2, 2, 1)
plt.imshow(Image.open(
    "C:\\Users\\Gary\\Desktop\\Machine-Learning-Deep-Learning\\卷积神经网络（Convolutional Neural Network, CNN）\\ImplementConvolution\\testCon.png"))
plt.axis('off')
plt.title('原图片')
# 第二张图展示真实结果
plt.subplot(2, 2, 2)
plt.imshow(Image.open(
    "C:\\Users\\Gary\\Desktop\\Machine-Learning-Deep-Learning\\卷积神经网络（Convolutional Neural Network, CNN）\\ImplementConvolution\\trueResult.png"))
plt.axis('off')
plt.title('真实结果')
# 第三张图展示测试结果
plt.subplot(2, 2, 3)
plt.imshow(new_img)
plt.axis('off')
plt.title('测试结果')
plt.show()
