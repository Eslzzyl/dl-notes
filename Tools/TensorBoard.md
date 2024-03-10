## 启动

如果系统默认启动了 TensorBoard，可以使用以下命令来关闭

```bash
ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9
```

然后启动 TensorBoard

```bash
tensorboard --port 6007 --logdir /path/to/your/tf-logs/direction
```

port 可以改，在 autodl 上，6007 是网页面板的唯一可用端口。

## 最佳实践

### 关于记录间隔

记录间隔不应该太大，否则数据点太少，可能会丢失一些关键的信息。TensorBoard 提供了平滑功能，即使数据点很多，也可以看到变化的趋势。

## 问题排查

使用官方提供的问题排查工具：

https://github.com/tensorflow/tensorboard/blob/master/tensorboard/tools/diagnose_tensorboard.py

保存到本地，然后执行即可。