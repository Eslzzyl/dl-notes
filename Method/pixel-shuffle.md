参考：
- https://nico-curti.github.io/NumPyNet/NumPyNet/layers/pixelshuffle_layer.html
- https://zhuanlan.zhihu.com/p/562932795

low-level vision 常用技术。起初用于图像超分辨率（[原论文](https://arxiv.org/abs/1609.05158)）。这是一种将图像通道数减少，将高和宽扩大的方法。

![说明图](https://nico-curti.github.io/NumPyNet/NumPyNet/images/pixelshuffle.svg)

这种操作将一个 shape 为 $(B,C*r^2,H,W)$ 的张量 reshape 成 $(B,C,H*r,W*r)$ 的张量，其中 $r$ 称为 upscale factor。

在卷积网络中，图像的高和宽要减少，而通道数常常迅速增加。pixel shuffle 是一种类似反卷积的操作。