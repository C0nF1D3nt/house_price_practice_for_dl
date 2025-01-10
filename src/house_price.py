import torch
import pandas as pd
from torch import nn
import torchvision
from torch.utils import data

# 读取数据
train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')

# print(train_data.shape)
# print(test_data.shape)

# 数据预处理
all_features = pd.concat((train_data.iloc[:, 1: -1], test_data.iloc[:, 1:]))
numeric_idx = [col for col in all_features.dtypes[all_features.dtypes != 'object'].index]
all_features[numeric_idx] = all_features[numeric_idx].apply(
    lambda x: ((x - x.mean()) / x.std())
)
all_features[numeric_idx] = all_features[numeric_idx].fillna(0)
all_features = pd.get_dummies(all_features, dummy_na=True)
# print(all_features.shape)

num_epochs = 100
batch_size = 256
lr = 0.1
weight_decay = 35
n_train = train_data.shape[0]

train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.iloc[:, -1].values, dtype=torch.float32)
# print(train_features.shape)
# print(test_features.shape)
# print(train_labels.shape)

train_dataset = data.TensorDataset(train_features, train_labels)
train_iter = data.DataLoader(train_dataset, batch_size, shuffle=True)

in_feature = train_features.shape[1]
net = nn.Sequential(
    nn.Linear(in_feature, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1),
)

loss = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

def log_rmse(net, train_features, train_labels):
    preds = net(train_features).reshape(train_labels.shape)
    preds = torch.clamp(preds, 1, float('inf'))
    return torch.sqrt(loss(torch.log(preds), torch.log(train_labels)))

for epoch in range(num_epochs):
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y.reshape(y_hat.shape))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print(f'epoch {epoch + 1}: rmse: {log_rmse(net, train_features, train_labels)}')

preds = net(test_features).detach().numpy()
test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
result = pd.concat((test_data['Id'], test_data['SalePrice']), axis=1)
result.to_csv('../result/result.csv', index=False)

