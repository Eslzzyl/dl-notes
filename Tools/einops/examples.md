# 关于 `einops` 的更多花哨展示

来自官网 https://einops.rocks/1-einops-basics/#fancy-examples-in-random-order

将 6 张图片的像素打散后堆叠起来。所有字母都同时可见。

```python
rearrange(ims, '(b1 b2) h w c -> (h b1) (w b2) c ', b1=2)
```

![img](https://img2023.cnblogs.com/blog/2554727/202402/2554727-20240203213211027-1429958752.png)

在纵向上打散像素后堆叠：

```python
rearrange(ims, '(b1 b2) h w c -> (h b1) (b2 w) c', b1=2)
```

![img](https://img2023.cnblogs.com/blog/2554727/202402/2554727-20240203213258010-446890947.png)


```python
reduce(ims, '(b1 b2) h w c -> h (b2 w) c', 'max', b1=2)
```

![img](https://img2023.cnblogs.com/blog/2554727/202402/2554727-20240203223211160-179061676.png)

```python
reduce(ims, 'b (h 2) (w 2) c -> (c h) (b w)', 'mean')
```

![img](https://img2023.cnblogs.com/blog/2554727/202402/2554727-20240203223254722-224213888.png)

```python
reduce(ims, 'b (h 4) (w 3) c -> (h) (b w)', 'mean')
```

![img](https://img2023.cnblogs.com/blog/2554727/202402/2554727-20240203223327658-806459468.png)

将图片分成两半，计算二者的平均值：

```python
reduce(ims, 'b (h1 h2) w c -> h2 (b w)', 'mean', h1=2)
```

![img](https://img2023.cnblogs.com/blog/2554727/202402/2554727-20240203223420218-1612075499.png)

将图片分割成小块，然后转置每个小块：

```python
rearrange(ims, 'b (h1 h2) (w1 w2) c -> (h1 w2) (b w1 h2) c', h2=8, w2=8)
```

![img](https://img2023.cnblogs.com/blog/2554727/202402/2554727-20240203224101295-912961940.png)

上面代码的更细致的版本：

```python
rearrange(ims, 'b (h1 h2 h3) (w1 w2 w3) c -> (h1 w2 h3) (b w1 h2 w3) c', h2=2, w2=2, w3=2, h3=2)
```

![img](https://img2023.cnblogs.com/blog/2554727/202402/2554727-20240203224213120-826392045.png)

```python
rearrange(ims, '(b1 b2) (h1 h2) (w1 w2) c -> (h1 b1 h2) (w1 b2 w2) c', h1=3, w1=3, b2=3)
```

![img](https://img2023.cnblogs.com/blog/2554727/202402/2554727-20240203224247518-2105160941.png)

相当复杂的模式：

```python
reduce(ims, '(b1 b2) (h1 h2 h3) (w1 w2 w3) c -> (h1 w1 h3) (b1 w2 h2 w3 b2) c', 'mean', h2=2, w1=2, w3=2, h3=2, b2=2)
```

![img](https://img2023.cnblogs.com/blog/2554727/202402/2554727-20240203230518916-1152632447.png)

从每张图片中独立地减除背景并执行 normalize：

```python
im2 = reduce(ims, 'b h w c -> b () () c', 'max') - ims
im2 /= reduce(im2, 'b h w c -> b () () c', 'max')
rearrange(im2, 'b h w c -> h (b w) c')
```

![img](https://img2023.cnblogs.com/blog/2554727/202402/2554727-20240203230944776-330291106.png)

像素化：先通过平均池化下采样，然后再上采样：

```python
averaged = reduce(ims, 'b (h h2) (w w2) c -> b h w c', 'mean', h2=6, w2=8)
repeat(averaged, 'b h w c -> (h h2) (b w w2) c', h2=6, w2=8)
```

![img](https://img2023.cnblogs.com/blog/2554727/202402/2554727-20240203231055740-1371548440.png)

```python
rearrange(ims, 'b h w c -> w (b h) c')
```

![img](https://img2023.cnblogs.com/blog/2554727/202402/2554727-20240203231117322-482303776.png)

把颜色轴加入水平轴，并对水平轴下采样：

```python
reduce(ims, 'b (h h2) (w w2) c -> (h w2) (b w c)', 'mean', h2=3, w2=3)
```

![img](https://img2023.cnblogs.com/blog/2554727/202402/2554727-20240203231303631-1207690069.png)