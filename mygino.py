from meshtransport import generate_uniform_box_points,find_minimuum_bounding_box
from meshtransport import KNeighBallChanger,KIWDBallChanger
import torch
import time
from torch import nn
import numpy as np
from neuralop.models import TFNO3d
irregular_grid=np.load("points_mesh.npy")
irregular_grid=(irregular_grid-np.min(irregular_grid))/(np.max(irregular_grid)-np.min(irregular_grid))



inp_positions=irregular_grid


rabbit_x=np.load("rabbit_x.npy")
rabbit_u=np.load("rabbit_u.npy")
test_set=np.random.choice(np.arange(len(rabbit_u)),size=100,replace=False)
train_set=np.array([i for i in np.arange(len(rabbit_u)) if i not in test_set])

rabbit_u_train=rabbit_u[train_set]
rabbit_u_test=rabbit_u[test_set]
rabbit_x_train=rabbit_x[train_set]
rabbit_x_test=rabbit_x[test_set]

rabbit_u_train=torch.tensor(rabbit_u_train,dtype=torch.float32)
rabbit_u_test=torch.tensor(rabbit_u_test,dtype=torch.float32)
rabbit_x_train=torch.tensor(rabbit_x_train,dtype=torch.float32)
rabbit_x_test=torch.tensor(rabbit_x_test,dtype=torch.float32)


train_data=torch.utils.data.TensorDataset(rabbit_x_train,rabbit_u_train)
train_data_loader=torch.utils.data.DataLoader(train_data,batch_size=2,shuffle=True)
test_data=torch.utils.data.TensorDataset(rabbit_x_test,rabbit_u_test)
test_data_loader=torch.utils.data.DataLoader(test_data,batch_size=2,shuffle=True)
radius=0.015

class MyGeoTFNO(nn.Module):
    def __init__(self, n_modes, in_channels, hidden_channels,out_channels,points,num_train_points):
        super().__init__()
        self.points=torch.tensor(points,dtype=torch.float32)
        self.num_train_points=num_train_points
        self.regular_points=torch.tensor(generate_uniform_box_points(np.max(points,axis=0),np.min(points,axis=0),num_train_points),dtype=torch.float32)
        self.tfno=TFNO3d(n_modes_height=n_modes, n_modes_width=n_modes, n_modes_depth=n_modes,in_channels=in_channels ,hidden_channels=32)
        self.encoder=KIWDBallChanger()
        self.decoder=KIWDBallChanger()
    def forward(self, x):
        x=self.encoder(x,self.points,self.regular_points).reshape(-1,1,self.num_train_points,self.num_train_points,self.num_train_points)
        x = self.tfno(x)
        x=x.reshape(-1,self.num_train_points**3)
        x= self.decoder(x,self.regular_points,self.points) 
        return x

model=MyGeoTFNO(n_modes=16,in_channels=1 ,hidden_channels=500,out_channels=1,points=inp_positions,num_train_points=100)
num_epochs=500
learning_rate=0.01
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs=1000

for epoch in range(epochs):
    for data in train_data_loader:
        start=time.time()
        rabbit_x_train,rabbit_u_train=data
        optimizer.zero_grad()
        y_pred = model(rabbit_x_train)
        l = loss(y_pred, rabbit_u_train)
        l.backward()
        optimizer.step()
        print(l.item())
        print(time.time()-start)

model.eval()
for data in test_data_loader:
    rabbit_x_train,rabbit_u_train=data
    y_pred = model(rabbit_x_train)
    l = loss(y_pred, rabbit_u_train)
    l.backward()
    optimizer.step()
    print(l.item())

