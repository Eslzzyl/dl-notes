本文档根据《动手学深度学习 PyTorch版》整理而成。

# 基础

CNN 中的卷积操作实际上是互相关操作。卷积核按照从左到右、从上到下的顺序在图像中滑动，卷积核的每个像素和自己对应的像素相乘并累加，最后得到特征图中对应像素的值。

padding：在图像周围补0

stride：即卷积核移动的步长。

CNN 训练实际上是训练卷积核数组的值

# 多个输出通道

卷积核是二维的。

如果输入图像有多个通道，如 RGB 图像有 3 个通道，那么卷积核也要有对应的通道数量。所有通道的互相关运算结果相加得到特征图对应位置的像素。此时特征图是单通道的。如果希望特征图也有多个通道（卷积操作输出多个通道），那么要使用多个卷积核堆叠起来（称为 filter，是三维的）。假设输入通道数是 $c_i$，输出通道数是 $c_o$，那么 filter 的 shape 是 $H \times W \times c_i$。$c_o$ 个 filter 堆叠起来，形成一个 $H \times W \times c_i \times c_o$ 的张量。

关于 filter 和卷积核的区别：https://blog.csdn.net/weixin_38481963/article/details/109906338

# $1\times 1$ 卷积

这里的 $1 \times 1$ 指的是 $H \times W$ 的尺寸。$1 \times 1$ 卷积核的第三个维度（通道维度）一般不是 1，而是和输入图像的第三个维度匹配。

1 个 $1 \times 1$ 卷积核可以保持输入图像的高和宽不变，将通道数减少到 1。使用合适数量的 $1 \times 1$ 卷积核，然后将卷积结果堆叠，就可以保持宽高不变，同时将通道数量调整到想要的值。

$1 \times 1$ 卷积多用于降维（从而降低计算复杂度），它可以当作是一个全连接层。

# 池化

池化的作用是缓解卷积对位置的过度敏感性。

# Depthwise Separable Convolution

https://zhuanlan.zhihu.com/p/80041030

分为两个步骤：

## Depthwise Convolution

和普通卷积的唯一不同是不同通道的特征图不再相加，有几个通道，就形成几个特征图。

设输入图像为 $H \times W \times C$，则有 $C$ 个卷积核，产生 $C$ 张特征图。

## Pointwise Convolution

通过 $1 \times 1 \times C$ 的卷积核来对 $C$ 张特征图进行卷积，实际上是在通道方向上的加权平均操作。

这种 Depthwise Separable Convolution 相比传统卷积可以有效减少参数数量，理论上计算量减少，但对 GPU IO 性能的要求更高，实际计算速度可能不会显著增加。

FLOPs与模型推理速度：https://zhuanlan.zhihu.com/p/122943688