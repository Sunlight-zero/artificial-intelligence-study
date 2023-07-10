利用 U-Net 和 VOC2012 数据集进行语义分割。

主要是为了尝试语义分割项目，获得一定的经验。

## 数据集

数据集需要储存在当前工作环境根目录的 `datasets` 文件夹中，命名为 `VOC2012`

训练数据的类别已经被统计出来作为 `num-classes.json` 存储在

## 知识点

### 交叉熵

`CrossEntropyLoss` 类导出的交叉熵函数支持两个参数：预测值和标签。其中，允许标签为一个 `[0, n)` 的整数，表示一个类别，计算时会被自动转换为独热码。

### 准确率

由于是分类问题，计算准确率只需要取 `argmax` 沿通道维取最大值即可。因为输出图像一般为 $(B, C, H, W)$ 维张量，沿第 1 维（从 0 开始计数）即可压缩通道维。

因此，准确数量可以表达为：
```python
torch.sum(y_hat.argmax(dim=1), y)
```

### 保存参数

使用 `model.state_dict()` 和 `optimizer.state_dict()` 可以保存模型和优化器当前的参数。

使用 `torch.save` 函数即可保存一个字典结构，将这些参数都拼合到一个字典内即可。

```python
checkpoint = {
    'model': unet.state_dict(),
    'optimizer': optimizer.state_dict(),
    'accuracy': num_accurate / num_total,
    'epoch': epoch + 1
}
torch.save(checkpoint, f"./parameter_epoch_{epoch + 1}.tar")
```

读取时，使用 `torch.load` 就可以重新加载一个字典结构。

```python
checkpoint = torch.load(path)
```
