参考：


标题 “AN IMAGE IS WORTH 16X16 WORDS” 指的是将图片划分成若干个 16 $\times$ 16 的小块

Vision Transformer 证明，将原始的 Transformer 直接用于视觉任务，同样可以获得不错的结果；如果在大规模数据集上进行预训练，然后迁移到小规模任务上进行微调（这种想法是 BERT 提出来的），那么 Vision Transformer 可以取得 SOTA 性能。

现有硬件能够接受的 Transformer 序列长度也就几百到上千，如果要用于视觉任务，那么序列长度就大大增加。以经典分类任务为例，输入图像的典型 size 是224 $\times$ 224 = 50176。因此，必须想办法减少序列长度。

# Introduction

ViT 之前的一些工作：
1. Local Network (CVPR 2018)，将尺寸相对小的特征图作为 Transformer 的输入
2. 在局部的窗口上做 Self-Attention
3. 在 H 和 W 方向上分别做 Self-Attention

ViT 的大致思路是，将图片分割成若干个 16 $\times$ 16 的小块，用某种 embedding 方法将这些小块转换成一个个 token 送入 Transformer。

在 NLP 中，大部分 Transformer 均采用无监督方式进行训练。但大多视觉的基线网络都会采用有监督方式进行训练，因此 ViT 同样采用有监督方式进行训练。

Transformer 相比 CNN 缺少一些必要的归纳偏置（inductive biases），使得本身并不像 CNN 一样天生地适合视觉任务。
1. locality：图片上相邻的区域具有相似的特征
2. translation equivariance（平移等变性），即 $f(g(x))=g(f(x))$，目标物体无论在图片中如何平移，通过 CNN 得到的结果总是一样的。

# Conclusion

作者认为既然 ViT 在分类任务上取得了良好的结果，那么应当在分割和检测任务上同样取得良好的结果。
- 检测：2020.12 ViT-FRCNN
- 分割：2020.12 SETR

把 ViT 变得更大：同作者团队 Scaling Vision Transformer

# Related Work

