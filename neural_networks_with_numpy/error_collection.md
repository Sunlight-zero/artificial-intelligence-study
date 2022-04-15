# 手写神经网络错误总结

### 2022/4/15

报错：`b`的`shape`变为`(30, 10)`了，但理论上应该是一个列向量。

原因：忘记加`zip`了，导致`b`引入了`self.weights`的第二个向量。

```python
    def feedforward(self, a):
        for w, b in self.weights, self.biases:
            a = sigmoid(w @ a + b)
        return a
```

此时`self.weights`的长度又恰好等于2，没有对`for`报错。

在多列表引用的时候必须加上`zip`。
