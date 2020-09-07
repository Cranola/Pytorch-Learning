import torch
from torch.nn import Module, Linear, MSELoss
from torch.optim import SGD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 定义一个线性函数并可视化
x = np.linspace(0, 20, 500)
y = 5 * x + 7
plt.plot(x, y)
# plt.show()

# 生成一些随机的点，来作为训练数据
x = np.random.rand(256)
noise = np.random.randn(256) / 4
y_1 = x * 5 + 7 + noise
y_2 = x * 5 + 7
plt.plot(x, y_1, label='data_1', alpha=0.3)
plt.plot(x, y_2, label='data_2', alpha=1)
plt.legend()
plt.show()


