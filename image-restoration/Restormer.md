# Abstract

在 CV 任务中使用 Transformer 必须合理处理计算复杂度的问题。高分辨率图像在图像复原任务中是十分常见的。Restormer 通过在 building blocks（多头注意力和前馈网络）中采用几个关键设计，从而得到了一种能够捕获远程像素交互的同时仍然高效的 Tansformer。

Restormer 在以下图像复原任务中都取得了 SOTA 成果：
- 去雨
- 单图像去运动模糊
- 去失焦模糊
- 去噪
  - 彩色和灰度图的高斯去噪
  - 真实图像去噪

# Introduction

图像复原需要强大的图像先验（priors）知识。CNN 能够从大规模训练数据中学到良好的先验知识，因此十分适合图像复原任务。

先前的一些将 Transformer 用于图像复原的工作大多将图片分割成小块来降低计算复杂度，但这种操作和 Transformer 捕获长距离依赖的特性是天然矛盾的。

Restormer 采用以下方法来改进原始 Transformer 结构：
- 使用 multi-Dconv head 'transposed' attention(MDTA) block 来替换原始 Transformer 中的 Multi-Head Attention 块。这个 MDTA 有线性复杂度。
- 为全连接层加入门控机制：gated-Dconv FN(GDFN)，过滤出有用的信息

Restormer 还使用了一种渐进式的训练方法，一开始使用小 patch size、大 batch size，随着训练的推进，逐渐增大 patch size，减少 batch size，从而允许模型逐渐学习到图片的全局特征。

# Background

## 图像复原

CNN 方法是图像复原方法中的绝对主流。而在这其中，基于 encoder-decoder 的 U 型网络得到了格外多的关注。U 型网络能够在保持较低计算复杂度的同时表征多尺度的层级特征。作者还提及了 skip connection、attention 方面的工作。作者提请读者注意 NTIRE challenge 中的一些相关报告和综述。

## 视觉 Transformer

视觉 Transformer 已经有了充足的发展。目前的一些基于 Transformer 的 low-level 图像复原方法主要采用了 Swin Transformer 式的、在图像局部应用 self-attention 的处理方式，这种方式有悖于 Transformer 相比 CNN 的优势，即捕获长距离依赖。

# Method

