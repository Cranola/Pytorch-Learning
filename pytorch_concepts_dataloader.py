import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchvision.datasets as dataset
import torchvision.models as models
from torchvision import transforms as transforms

# Dataset是一个抽象类，为了方便的读取，需要将要待使用的数据包装成dataset类。自定义的该类需要继承Dataset
# 并且实现实现两个成员方法


class MyDataset(Dataset):

    def __init__(self, csv_file):
        self.f = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.f)

    def __getitem__(self, idx):
        return self.f.iloc[idx].cloumn


# 实例化一个对象访问
data_demo = MyDataset('')

# DataLoader实现了对Dataset的读取, 返回一个可迭代对象
training_data = DataLoader(data_demo, batch_size=10, num_workers=0, shuffle=True)
####
data = iter(training_data)
print(next(data))
####
# 常用的方法是用for循环对其遍历
for i, data in enumerate(training_data):
    print(i, data)

# 预加载数据集 torchvision.datasets, 已经提前处理好的数据集，可以直接加载使用
train_set = dataset.MNIST(
    root="",
    train=False,
    download=False,
    transform=None
)

# 预加载模型
resnet50 = models.resnet50(pretrained=True)

# transforms模块提供了一般的图像转换操作, 用于数据增强
# 最后一步参数是根据ImageNet训练的归一化参数，可直接使用，可以认定为固定值
# 详情参考https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/21
transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，再随机裁剪为32*32
    transforms.RandomHorizontalFlip(),  # 有一半的几率做水平翻转
    transforms.RandomRotation((-45, 45)),  # 随机翻转
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.2023, 0.1994, 0.2010))  # RGB每层用到的归一化和方差
])
