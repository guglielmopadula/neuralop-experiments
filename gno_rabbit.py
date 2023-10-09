import torch
import matplotlib.pyplot as plt
import sys
from utilities import DenseNet
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from nn_conv import NNConv_old
import torch.nn.functional as F
from tqdm import trange
from neuralop.models import TFNO1d,FNO,TFNO
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_params
from neuralop import LpLoss, H1Loss
import numpy as np
import torch.nn as nn
from neuralop.models.spectral_convolution import FactorizedSpectralConv
device = 'cpu'


x=np.load("rabbit_x.npy").reshape(300,-1,1)
y=np.load("rabbit_u.npy").reshape(300,-1,1)
points_mesh=np.load("points_mesh.npy")
n=x.shape[0]
m=x.shape[1]
test_subset=np.random.choice(n, size=100, replace=False, p=None)
A=np.arange(n)
train_subset=A[~np.in1d(A,test_subset)]


class KernelNN(torch.nn.Module):
    def __init__(self, width, ker_width, depth, ker_in, in_width=1, out_width=1):
        super(KernelNN, self).__init__()
        self.depth = depth
        self.fc1 = torch.nn.Linear(in_width, width)
        kernel = DenseNet([ker_in, ker_width, ker_width, width**2], torch.nn.ReLU)
        self.conv1 = NNConv_old(width, width, kernel, aggr='mean')
        self.fc2 = torch.nn.Linear(width, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        for k in range(self.depth):
            x = F.relu(self.conv1(x, edge_index, edge_attr))

        x = self.fc2(x)
        return x



r=0.01
t=0
print(np.max(points_mesh))
print(np.min(points_mesh))
for i in trange(len(points_mesh)):
    for l in np.arange(len(points_mesh[np.linalg.norm(points_mesh-points_mesh[i],axis=1)<r])):
        t=t+1
print(t)
edge_attributes=np.zeros((n,t,8))
index=np.zeros((2,t),dtype=np.int64)
t=0
for i in trange(len(points_mesh)):
    for l in np.arange(len(points_mesh[np.linalg.norm(points_mesh-points_mesh[i],axis=1)<r])):
        index[0,t]=i
        index[1,t]=l
        edge_attributes[:,t,:3]=points_mesh[i]
        edge_attributes[:,t,3:6]=points_mesh[l]
        edge_attributes[:,t,6]=x[:,i,0]
        edge_attributes[:,t,7]=x[:,l,0]
        t=t+1

x=torch.tensor(x,dtype=torch.float32)
y=torch.tensor(y,dtype=torch.float32)
index=torch.tensor(index,dtype=torch.int64)
edge_attributes=torch.tensor(edge_attributes,dtype=torch.float32)



x_train=x[train_subset]
y_train=y[train_subset]
edge_attributes_train=edge_attributes[train_subset]
edge_attributes_test=edge_attributes[test_subset]
x_test=x[test_subset]
y_test=y[test_subset]

train_dataset=[]
test_dataset=[]
for i in range(len(train_subset)):
    train_dataset.append(Data(x=x_train[i].reshape(-1,1), y=y_train[i].reshape(-1,1), edge_index=index, edge_attr=edge_attributes_train[i].reshape(-1,8)))
for i in range(len(test_subset)):  
    test_dataset.append(Data(x=x_test[i], y=y_test[i], edge_index=index, edge_attr=edge_attributes_test[i]))

train_dataloader=DataLoader(train_dataset, batch_size=1, shuffle=False)
test_dataloader=DataLoader(test_dataset, batch_size=1, shuffle=False)
model=KernelNN(5,5,5,8)

num_epochs=10
learning_rate=0.01
loss = H1Loss(d=2)
l2loss = LpLoss(d=2, p=2)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in trange(num_epochs):
    tot_loss=0
    for data in train_dataloader:
        data=data.to(device)
        y_pred = model(data).reshape(data.y.shape)
        l = torch.linalg.norm(y_pred-data.y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        tot_loss=tot_loss+(torch.norm(y_pred-data.y)/torch.linalg.norm(data.y))/len(train_dataloader)
    print("Train loss is", tot_loss.item())

model=model.eval()
tot_loss=0
for data in test_dataloader:
    data=data.to(device)
    x = x.to(device)
    y = y.to(device)
    y_pred = model(data).reshape(data.y.shape)
    tot_loss=tot_loss+(torch.norm(y_pred-y)/torch.linalg.norm(y))/len(test_dataloader)

print("Test loss is", tot_loss.item())




'''
fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y']
    # Model prediction
    out = model(x.unsqueeze(0))

    ax = fig.add_subplot(3, 3, index*3 + 1)
    ax.imshow(x[0], cmap='gray')
    if index == 0:
        ax.set_title('Input x')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 2)
    ax.imshow(y.squeeze())
    if index == 0:
        ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 3)
    ax.imshow(out.squeeze().detach().numpy())
    if index == 0:
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
fig.savefig("fig.png")

'''