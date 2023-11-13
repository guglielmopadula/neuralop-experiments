import torch
import matplotlib.pyplot as plt
import sys
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import trange
import numpy as np
import torch.nn as nn
from torch_geometric.nn import NNConv
device = 'cpu'
import scipy


x=np.load("rabbit_x.npy").reshape(300,-1,1)
y=np.load("rabbit_u.npy").reshape(300,-1,1)
points_mesh=np.load("points_mesh.npy")
n=x.shape[0]
m=x.shape[1]
test_subset=np.random.choice(n, size=100, replace=False, p=None)
A=np.arange(n)
train_subset=A[~np.in1d(A,test_subset)]

def compute_list(positions,r=0.015):
    tot=0
    tree=scipy.spatial.cKDTree(positions)
    for i in range(len(positions)):
        l=tree.query_ball_point(positions[i],r=r)
        tot=tot+len(l)
    list=np.zeros((2,tot),dtype=np.int64)
    acc=0
    for i in range(len(positions)):
        l=tree.query_ball_point(positions[i],r=r)
        for j in range(len(l)):
            list[0,acc]=i
            list[1,acc]=l[j]
            acc=acc+1
    return list



class GNO(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(GNO, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.nn=nn.Sequential(nn.Linear(8,100),nn.ReLU(),nn.Linear(100,100),nn.ReLU(),nn.Linear(100,100),nn.ReLU(),nn.Linear(100,in_channels*out_channels))
        self.nn_conv=NNConv(in_channels,out_channels,self.nn,aggr="mean")
    
    def forward(self,batch,radius=0.015):
        x=self.nn_conv(batch.x,batch.edge_index,batch.edge_attr)
        return x
    



r=0.015
t=0
print(np.max(points_mesh))

x=torch.tensor(x,dtype=torch.float32)
y=torch.tensor(y,dtype=torch.float32)
points_mesh=torch.tensor(points_mesh,dtype=torch.float32)
mylist=torch.tensor(compute_list(points_mesh,r=r))

x_train=x[train_subset]
y_train=y[train_subset]
x_test=x[test_subset]
y_test=y[test_subset]

train_dataset=[]
test_dataset=[]
for i in range(len(train_subset)):
    tmp=x_train[i].reshape(-1,1)
    train_dataset.append(Data(x=tmp, y=y_train[i].reshape(-1,1), edge_index=mylist, edge_attr=torch.cat((points_mesh[mylist[0]],points_mesh[mylist[1]],tmp[mylist[0]],tmp[mylist[1]]),dim=1)))
for i in range(len(test_subset)):  
    tmp=x_test[i].reshape(-1,1)
    test_dataset.append(Data(x=x_test[i], y=y_test[i], edge_index=mylist, edge_attr=torch.cat((points_mesh[mylist[0]],points_mesh[mylist[1]],tmp[mylist[0]],tmp[mylist[1]]),dim=1)))

train_dataloader=DataLoader(train_dataset, batch_size=1, shuffle=False)
test_dataloader=DataLoader(test_dataset, batch_size=1, shuffle=False)
model=GNO(1,1)

num_epochs=10
learning_rate=0.01
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
        tot_loss=tot_loss+((torch.norm(y_pred-data.y)/torch.linalg.norm(data.y))/len(train_dataloader)).item()
    print("Train loss is", tot_loss)

model=model.eval()
tot_loss=0
for data in test_dataloader:
    data=data.to(device)
    x = x.to(device)
    y = y.to(device)
    y_pred = model(data).reshape(data.y.shape)
    tot_loss=tot_loss+((torch.norm(y_pred-y)/torch.linalg.norm(y))/len(test_dataloader)).item()

print("Test loss is", tot_loss)




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