import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  ##PIL包读取图像数据

##读取图像 转化为灰度图像 转化为numpy数组
myim = Image.open("l_000493.jpg")
myimgray = np.array(myim.convert("L"),dtype = np.float32)

##可视化图像
plt.figure(figsize = (6,6))
plt.imshow(myimgray,cmap = plt.cm.gray)
plt.axis("off")
plt.show()
##lena图像，省略

##转化为1*1*512*512的张量
imh,imw = myimgray.shape
myimgray_t = torch.from_numpy(myimgray.reshape((1,1,imh,imw)))
print(myimgray_t.shape)
#torch.Size([1, 1, 416, 416])

##对灰度图像进行卷积提取图像轮廓
kersize = 3
ker = torch.ones(kersize,kersize,dtype = torch.float32)*-1
ker[2,2] = 24
ker = ker.reshape((1,1,kersize,kersize))
print(ker)
# tensor([[[[-1., -1., -1., -1., -1.],
#           [-1., -1., -1., -1., -1.],
#           [-1., -1., 24., -1., -1.],
#           [-1., -1., -1., -1., -1.],
#           [-1., -1., -1., -1., -1.]]]])

##进行卷积操作
conv2d = nn.Conv2d(1,2,(kersize,kersize), padding=1, bias = False)
conv2d.weight.data[0] = ker ##设置卷积时使用的核，第一个核使用边缘检测核

##对灰度图像进行卷积操作
imconv2out = conv2d(myimgray_t)

##对卷积后的结果进行最大值池化
maxpool2 = nn.MaxPool2d(2,stride = 2, padding=0) ##窗口大小为2 步长为2
pool2_out = maxpool2(imconv2out)
pool2_out_im = pool2_out.squeeze()
print(pool2_out.shape)
# torch.Size([1, 2, 254, 254])

##最大值池化可视化图像
plt.figure(figsize = (12,6))
plt.subplot(1,2,1)
plt.imshow(pool2_out_im[0].data,cmap = plt.cm.gray)
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(pool2_out_im[1].data, cmap= plt.cm.gray)
plt.axis("off")
plt.show()

##对卷积后的结果进行平均值池化
avgpool2 = nn.AvgPool2d(2,stride = 2) ##窗口大小为2 步长为2
pool2_out = avgpool2(imconv2out)
pool2_out_im = pool2_out.squeeze()
print(pool2_out.shape)
# torch.Size([1, 2, 254, 254])
print(pool2_out_im.size())

##平均值池化可视化图像
plt.figure(figsize = (12,6))
plt.subplot(1,2,1)
plt.imshow(pool2_out_im[0].data,cmap = plt.cm.gray)
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(pool2_out_im[1].data, cmap= plt.cm.gray)
plt.axis("off")
plt.show()


