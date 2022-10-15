import numpy as np
import inspect
import collections
from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt
from IPython import display

def add_to_class(Class): # 作为一个修饰器使用
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper

class HyperParameters:
    def save_hyperparameters(self, ignore=[]):
        # currentframe 获得栈帧
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame) # 获得所有本地变量
        self.hparams = {
            k: v for k, v in local_vars.items()
            if k not in set(ignore + ['self']) and not k.startswith('_')
        } # 过滤 self 变量、下划线开头变量
        for k, v in self.hparams.items():
            setattr(self, k, v)

class ProgressBoard(HyperParameters):
    def __init__(
        self, xlabel=None, ylabel=None, xlim=None, ylim=None,
        xscale='linear', yscale='linear',
        ls=['-', '--', '-.', ':'],
        colors=['C0', 'C1', 'C2', 'C3'],
        fig=None, axes=None, figsize=(3.5, 2.5), display=True
    ):
        self.save_hyperparameters()
    
    def draw(self, x, y, label, every_n=1):
        Point = collections.namedtuple('Point', ['x', 'y']) # 创建命名元组
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict() # 有序映射
            self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = [] # 空列表
            self.data[label] = []
        points: list = self.raw_points[label]
        line: list = self.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return
        mean = lambda x: sum(x) / len(x)
        line.append(Point(
            mean([p.x for p in points]),
            mean([p.y for p in points])
        ))
        points.clear()
        if not self.display:
            return
        backend_inline.set_matplotlib_formats('svg') # 设置矢量图格式
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize) # 创建画布
        plt_lines, labels = [], []
        for (k, v), ls ,color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(plt.plot(
                [p.x for p in v], [p.y for p in v],
                linestyle=ls, color=color
            )[0])
            labels.append(k)
        axes = self.axes if self.axes else plt.gca() # get current axes 获得当前子图
        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        if not self.xlabel: self.xlabel = self.x
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels) # 设置图例
        display.display(self.fig) # 显示图像
        display.clear_output(wait=True) # 让新显示的图覆盖原图
