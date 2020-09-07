import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.fc = nn.Linear(1350, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


net = Net()
print(net)

# 网络的可学习参数可以通过net.parameters()返回
for parameter in net.parameters():
    print(parameter)

# net.named_parameters()可以同时返回参数的名称及参数
for name, parameter in net.named_parameters():
    print(name, ':', parameter.size())

input_tensor = torch.randn(1, 1, 32, 32)
out = net(input_tensor)

# 在反向传播时，要先将所有参数的梯度清零
# torch.nn只支持mini_batches, 不支持一次只输入一个样本，必须是一个batch
net.zero_grad()
out.backward(torch.ones(1, 10))

# 利用nn中定义的损失函数
y = torch.arange(0, 10).view(1, 10).float()
criterion = nn.MSELoss()
loss = criterion(y, out)
print(loss.item())

# 优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
# 梯度清零，与net.zero_grad()效果一样
optimizer.zero_grad()
loss.backward()
# 更新参数
optimizer.step()

