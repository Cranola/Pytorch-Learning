import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from collections import Counter

df = pd.read_csv('./data/banking.csv', usecols=['age','job','education','marital','housing','loan','y'])
df = df[:1000]
print(len(df))

result_var = 'y'
cat_var = ['job', 'education', 'marital', 'housing', 'loan']
cont_var = ['age']

for col in df.columns:
    if col in cat_var:
        ccol = Counter(df[col])
        print(col, len(ccol), ccol)
        print("\r\n")

for col in df.columns:
    if col in cat_var:
        df[col].fillna('---')
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    if col in cont_var:
        df[col] = df[col].fillna(0)

print(df.head())

Y = df['y']
X = df.drop(columns=['y'])


# 要使用pytorch处理数据，要用Dataset进行数据集的定义
class BankingDataset(Dataset):
    def __init__(self, X, Y):
        self.x = X.values
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


training_data = BankingDataset(X, Y)
print(training_data[0])


# 定义模型
class BankingModel(nn.Module):
    def __init__(self):
        super(BankingModel, self).__init__()
        self.lin1 = nn.Linear(6, 500)
        self.lin2 = nn.Linear(500, 100)
        self.lin3 = nn.Linear(100, 2)
        self.bn_in = nn.BatchNorm1d(6)
        self.bn1 = nn.BatchNorm1d(500)
        self.bn2 = nn.BatchNorm1d(100)

    def forward(self, x_in):
        x = self.bn_in(x_in)
        x = F.relu(self.lin1(x))
        x = self.bn1(x)

        x = F.relu(self.lin2(x))
        x = self.bn2(x)

        x = self.lin3(x)
        x = torch.sigmoid(x)
        return x


# 模型训练
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

criterion = nn.CrossEntropyLoss()
model = BankingModel().to(DEVICE)

# 检测模型是否有问题
rn = torch.rand(3, 6).to(DEVICE)
output = model(rn)
print(output)

# 超参数设置
LEARNING_RATE = 0.01
batch_size = 500
EPOCHS = 10

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
training_dl = DataLoader(training_data, batch_size=batch_size, shuffle=True)

# 开始训练
model.train()   # 这个的作用是什么？？？
losses = []
for epoch in range(EPOCHS):
    for i, (x, y) in enumerate(training_dl):
        x = x.float().to(DEVICE)
        y = y.long().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(x)

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().data.item())
    print('Epoch : %d/%d, Loss: %.4f' % (epoch+1, EPOCHS, np.mean(losses)))

# 检测一下模型准确率
model.eval()
correct = 0
total = 0
for i, (x, y) in enumerate(training_dl):
    x = x.float().to(DEVICE)
    y = y.long()
    output_model = model(x).cpu()
    _, predicted = torch.max(output_model.data, 1)
    total += y.size(0)
    correct += (predicted == y).sum()
print('准确率：%.4f %%' % (100 * correct / total))