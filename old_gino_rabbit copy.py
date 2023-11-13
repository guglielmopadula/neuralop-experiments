import torch
import matplotlib.pyplot as plt
import sys
from utilities import DenseNet
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from nn_conv import NNConv_old
import torch.nn.functional as F
from tqdm import trange
from neuralop.models import TFNO3d
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
print(len(points_mesh))
num_samples=x.shape[0]
m=x.shape[1]
test_subset=np.random.choice(num_samples, size=100, replace=False, p=None)
A=np.arange(num_samples)
train_subset=A[~np.in1d(A,test_subset)]

q=16#int(np.ceil(points_mesh.shape[0]**(1/3)))
grid_points=np.zeros((q**3,3))
bary=np.load("bary.npy")
def from1dto3d(i):
    return np.array([i % q,(i/q)%q,i//(q**2)])

def from3dto1d(i,j,k):
    return i+j*q+k*q*q

for i in trange(q**3):
    grid_points[i]=from1dto3d(i)
grid_points=(grid_points-np.min(grid_points))/(np.max(grid_points)-np.min(grid_points))


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


def fun(mu,x):
    return np.exp(-mu*((x[:,0]-bary[0])**2 + (x[:,1]-bary[1])**2+(x[:,2]-bary[2])**2)**0.5)


mu=1+3*np.arange(num_samples)/(num_samples-1)
mu=mu.reshape(300,1)
grid_values=fun(mu,grid_points)

r=0.04
t=0
for i in trange(len(points_mesh)):
    for l in np.arange(len(grid_points[np.linalg.norm(grid_points-points_mesh[i],axis=1)<r])):
        t=t+1
assert t>0
print(2*t)
edge_attributes=np.zeros((num_samples,2*t,8))
index=np.zeros((2,2*t),dtype=np.int64)
t=0
for i in trange(len(points_mesh)):
    for l in np.arange(len(grid_points[np.linalg.norm(grid_points-points_mesh[i],axis=1)<r])):
        index[0,2*t]=i
        index[1,2*t]=l+len(points_mesh)
        index[0,2*t+1]=l+len(points_mesh)
        index[1,2*t+1]=i
        edge_attributes[:,2*t,:3]=points_mesh[i]
        edge_attributes[:,2*t,3:6]=grid_points[l]
        edge_attributes[:,2*t,6]=x[:,i,0]
        edge_attributes[:,2*t,7]=grid_values[:,l]
        edge_attributes[:,2*t+1,:3]=grid_points[l]
        edge_attributes[:,2*t+1,3:6]=points_mesh[i]
        edge_attributes[:,2*t,6]=grid_values[:,l]
        edge_attributes[:,2*t,7]=x[:,i,0]
        t=t+1

x=torch.tensor(x,dtype=torch.float32)
y=torch.tensor(y,dtype=torch.float32)
index=torch.tensor(index,dtype=torch.int64)
edge_attributes=torch.tensor(edge_attributes,dtype=torch.float32)

grid_values=grid_values.reshape(300,-1,1)
x=torch.cat((x,torch.tensor(grid_values, dtype=torch.float32)),axis=1)
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

class GINO(nn.Module):
    def __init__(self):
        super(GINO, self).__init__()
        self.kernel1=KernelNN(3,3,3,8)
        self.tfn=TFNO3d(16,16,16,32,1,1)
        self.kernel2=KernelNN(3,3,3,8)
    
    def forward(self,data):
        x_hat=self.kernel1(data)
        x_new=x_hat[len(points_mesh):].reshape(1,1,q,q,q)
        x_new=self.tfn(x_new)
        x_hat=torch.cat((x_hat[:len(points_mesh)].reshape(1,-1,1),x_new.reshape(1,-1,1)),axis=1)
        new_data=Data(x=x_hat.reshape(data.x.shape), y=data.y, edge_index=data.edge_index, edge_attr=data.edge_attr)
        x_hat=self.kernel2(new_data)
        return x_hat[:len(points_mesh)]



model=GINO()
num_epochs=500
learning_rate=0.001
loss = H1Loss(d=2)
l2loss = LpLoss(d=2, p=2)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in trange(num_epochs):
    tot_loss=0
    for data in train_dataloader:
        data=data.to(device)
        tmp=model(data)
        l = torch.linalg.norm(tmp.reshape(data.y.shape)-data.y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        tot_loss=tot_loss+(torch.norm(tmp.reshape(data.y.shape)-data.y)/torch.linalg.norm(data.y))/len(train_dataloader)
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



