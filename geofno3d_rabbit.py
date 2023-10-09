import torch
import matplotlib.pyplot as plt
import sys
from tqdm import trange
from neuralop.models import TFNO3d,FNO,TFNO
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
n=x.shape[0]
m=x.shape[1]
test_subset=np.random.choice(n, size=100, replace=False, p=None)
A=np.arange(n)
train_subset=A[~np.in1d(A,test_subset)]
x=torch.tensor(x,dtype=torch.float32)
y=torch.tensor(y,dtype=torch.float32)

eigs=torch.tensor(np.load("lapl_spectrum_matrix.npy"),dtype=torch.float32)
x_train=x[train_subset]
y_train=x[train_subset]
x_test=x[test_subset]
y_test=x[test_subset]
train_dataloader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train,y_train),batch_size=200)
test_dataloader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test,y_test),batch_size=100)

class MyGeoTFNO(nn.Module):
    def __init__(self, n_modes, in_channels, hidden_channels,eigs):
        super().__init__()
        self.matrix=eigs
        self.q=n_modes
        self.matrix_inv=torch.pinverse(self.matrix)
        self.tfno=TFNO3d(n_modes_height=self.q, n_modes_width=self.q, n_modes_depth=self.q,in_channels=in_channels ,hidden_channels=32)
        self.lin0=nn.Linear(m,self.q*self.q*self.q)
        self.lin1=nn.Linear(self.q*self.q*self.q,m)

    def forward(self, x):
        x=x.reshape(-1,m)
        x = self.lin0(x).reshape(-1,1,self.q,self.q,self.q)
        x = self.tfno(x)
        x= self.lin1(x.reshape(-1,self.q*self.q*self.q)) 
        return x
model=MyGeoTFNO(n_modes=16,in_channels=1 ,hidden_channels=32,eigs=eigs)
'''
model=nn.Sequential(FactorizedSpectralConv(1,10,(101,101)),nn.ReLU(),
                    FactorizedSpectralConv(10,10,(101,101)),nn.ReLU(),
                    FactorizedSpectralConv(10,10,(101,101)),nn.ReLU(),
                    FactorizedSpectralConv(10,1,(101,101)),nn.ReLU())
'''
num_epochs=500
learning_rate=0.0001
loss = H1Loss(d=2)
l2loss = LpLoss(d=2, p=2)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in trange(num_epochs):
    tot_loss=0
    for data in train_dataloader:
        x, y = data
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x).reshape(y.shape)
        l = torch.linalg.norm(y_pred-y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        tot_loss=tot_loss+(torch.norm(y_pred-y)/torch.linalg.norm(y))/len(train_dataloader)
    #print("Train loss is", tot_loss.item())

model=model.eval()
tot_loss=0
for data in test_dataloader:
    x, y = data
    x = x.to(device)
    y = y.to(device)
    y_pred = model(x).reshape(y.shape)
    l = loss(y_pred, y)
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