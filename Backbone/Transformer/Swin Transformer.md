ICCV 2021 最佳论文

参考：
- https://www.bilibili.com/video/BV13L4y1475U

# Introduction

对 CV 的密集预测任务，如检测，分隔等，多尺寸的特征是至关重要的。

ViT 无法提取多尺寸特征，且主要用于低分辨率图像（即使使用了 patch，但分辨率增加后 token 数量仍然平方增长）。

ViT 是在整张图上算 self-attention 的（将图像分块后），而 Swin Transformer 是在一个小窗口内算 self-attention 的。正因为如此，Swin Transformer 才有线性时间复杂度。

Swin Transformer 合并 patch 的操作类似 CNN 中的池化。

# Conclusion

略

# Related Work

略

# Method

## Overall Architecture

patch merging: 很像 pixel shuffle 的逆操作。将输入图像的长和高减半，通道数量翻倍。下图显示的最后一步是一个 $1\times 1$ 卷积，将通道数量从 $4C$ 降到 $2C$。

![patch merging 说明](https://img.eslzzyl.eu.org/6058b70fd79ebae96ea52ddb129b3da9.jpg)

