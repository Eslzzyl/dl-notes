# Abstract

在 CV 任务中使用 Transformer 必须合理处理计算复杂度的问题。高分辨率图像在图像复原任务中是十分常见的。Restomer 通过在 building blocks（多头注意力和前馈网络）中采用几个关键设计，从而得到了一种能够捕获远程像素交互的同时仍然高效的 Tansformer。

Restomer 在以下图像复原任务中都取得了 SOTA 成果：
- 去雨
- 单图像去运动模糊
- 去失焦模糊
- 去噪
  - 彩色和灰度图的高斯去噪
  - 真实图像去噪

# Introduction

图像复原需要强大的图像先验（priors）知识。CNN 能够从大规模训练数据中学到良好的先验知识，因此十分适合图像复原任务。

先前的一些将 Transformer 用于图像复原的工作大多将图片分割成小块来降低计算复杂度，但这种操作和 Transformer 捕获长距离依赖的特性是天然矛盾的。

Restomer 采用了 multi-Dconv head 'transposed' attention(MDTA) block 来替换原始 Transformer 中的 Multi-Head Attention 块。这个 MDTA 有线性复杂度。