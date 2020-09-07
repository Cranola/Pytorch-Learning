import os
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 对于微调来说，一般语音领域微调前几层，图像识别领域微调后几层
# 比如一个微调技巧，利用预训练模型的卷积层输出作为一个特征提取，把之后的全连接层当作一个简单的网络，而我们只需要训练最后全连接层的参数即可
# 对于不同的层可以设置不同的学习率，一般情况下建议，对于使用的原始数据做初始化的层设置的学习率要小于（一般可设置小于10倍）初始化的学习率
# 这样保证对于已经初始化的数据不会扭曲的过快，而使用初始化学习率的新层可以快速的收敛
breeds = ['a', 'b', 'c', 'd']
breed2idx = dict((breed, idx) for idx, breed in enumerate(breeds))
idx2breed = dict((idx, breed) for idx, breed in enumerate(breeds))
print(breed2idx)
print(idx2breed)
print(idx2breed[2])
