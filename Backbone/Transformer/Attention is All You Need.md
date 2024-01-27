# 参考链接

李沐的解析视频：[https://www.bilibili.com/video/BV1pu411o7BE](https://www.bilibili.com/video/BV1pu411o7BE)

The Illustrated Transformer: [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
# Architecture

![arch|400](https://img.eslzzyl.eu.org/b9a7cc6937aba5382fda76310a1949e1.png)

## 输入

输入的词序列通过某种 embedding 算法转换为向量。embedding 可以是预训练出来的，比如为每个单词分配一个序列数字。

由于 Transformer 的输入是一次性吃进去的，不包含时间序列信息，因此必须设计某种机制，让输入带上时间序列信息，这就是 Positional Encoding 的目的。Positional Encoding 是通过正弦/余弦函数算出来的，公式并不复杂。

上述两个向量加起来，就是 Transformer 的输入。这输入同时作为 Q、K、V 进入 Multi-Head Attention。

## 输出

是一个概率。具体怎么运作的不太清楚。

## Encoder

由 6 个完全相同的层组成，即上图的 N$\times$ 部分

又可分两个子层：
1. 多头注意力子层
2. point-wise 前馈网络层

每个子层带有一个 residual connection，以及在子层后进行一次 Layer Norm。

有研究表明，在子层前面进行 Layer Norm 效果更好。

## Decoder

同样由 6 个完全相同的层组成。

Masked Multi-Head Attention：和时序有关系

## Attention 部分

### Scaled Dot-Product Attention

就一个核心公式：

$$
\text{Attention}(Q, K, V)=\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

这里 Q 和 K 要做内积来衡量两个向量的相似度。输入是矩阵，所有的 Q 和 所有的 K 都做内积，于是 Transformer 能够捕获全局的相关性。

Q 和 K 的内积最后除以 $\sqrt{d_k}$，是为 Scaled。

### Multi-Head Attention

感觉就是把多个 Attention 的结果拼接在一起，输入和输出接上线性层。Attention 本身没太多可以学习的东西，线性层的权重可以学习。

