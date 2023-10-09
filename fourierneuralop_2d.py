import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import TFNO,FNO
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_params
from neuralop import LpLoss, H1Loss
import numpy as np
import torch.nn as nn
from neuralop.models.spectral_convolution import FactorizedSpectralConv
device = 'cpu'


x=np.load("fem_sol_x.npy").reshape(-1,1,101,101)
y=np.load("fem_sol_y.npy").reshape(-1,1,101,101)
n=x.shape[0]
test_subset=np.random.choice(n, size=20, replace=False, p=None)
A=np.arange(n)
train_subset=A[~np.in1d(A,test_subset)]
x=torch.tensor(x,dtype=torch.float32)
y=torch.tensor(y,dtype=torch.float32)
x_train=x[train_subset]
y_train=x[train_subset]
x_test=x[test_subset]
y_test=x[test_subset]
train_dataloader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train,y_train))
test_dataloader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test,y_test))

model = TFNO(n_modes=(16, 16),in_channels=1 ,hidden_channels=32)
'''
model=nn.Sequential(FactorizedSpectralConv(1,10,(101,101)),nn.ReLU(),
                    FactorizedSpectralConv(10,10,(101,101)),nn.ReLU(),
                    FactorizedSpectralConv(10,10,(101,101)),nn.ReLU(),
                    FactorizedSpectralConv(10,1,(101,101)),nn.ReLU())
'''
num_epochs=50
learning_rate=0.0001
loss = H1Loss(d=2)
l2loss = LpLoss(d=2, p=2)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    tot_loss=0
    for data in train_dataloader:
        x, y = data
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        l = loss(y_pred, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        tot_loss=tot_loss+(torch.norm(y_pred-y)/torch.linalg.norm(y))/len(train_dataloader)
    print("Train loss is", tot_loss.item())

model=model.eval()
tot_loss=0
for data in test_dataloader:
    x, y = data
    x = x.to(device)
    y = y.to(device)
    y_pred = model(x)
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