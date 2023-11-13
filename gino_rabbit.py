from meshtransport import generate_uniform_box_points,find_minimuum_bounding_box
from meshtransport import KNeighBallChanger,KIWDBallChanger
import torch
import time
from torch import nn
from torch.nn import functional as F
import numpy as np
from neuralop import TFNO3d
from neuralop import IntegralTransform
from neuralop import NeighborSearch
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
train_data_loader=torch.utils.data.DataLoader(train_data,batch_size=1,shuffle=True)
test_data=torch.utils.data.TensorDataset(rabbit_x_test,rabbit_u_test)
test_data_loader=torch.utils.data.DataLoader(test_data,batch_size=1,shuffle=True)
radius=0.015

class MyGeoTFNO(nn.Module):
    def __init__(self, n_modes, in_channels, hidden_channels,out_channels,points,num_train_points):
        super().__init__()
        self.points=torch.tensor(points,dtype=torch.float32)
        self.num_train_points=num_train_points
        self.regular_points=torch.tensor(generate_uniform_box_points(np.max(points,axis=0),np.min(points,axis=0),num_train_points),dtype=torch.float32)
        self.tfno=TFNO3d(n_modes_height=n_modes, n_modes_width=n_modes, n_modes_depth=n_modes,in_channels=in_channels ,hidden_channels=32)
        self.nb_search_out = NeighborSearch(use_open3d=False)
        self.gno = IntegralTransform(
                    mlp_layers=[6,50,50,1],
                    mlp_non_linearity=F.gelu,
                    transform_type='linear' 
        )

    def forward(self, x):
        in_to_out_nb = self.nb_search_out(self.points, self.regular_points,0.05)
        x=x.reshape(-1,1)
        x=torch.nan_to_num(x)
        x = self.gno(y=self.points, neighbors=in_to_out_nb, x=self.regular_points, f_y=x)
        x=x.reshape(-1,1,self.num_train_points,self.num_train_points,self.num_train_points)
        x = self.tfno(x)
        x=x.reshape(self.num_train_points**3,1)
        in_to_out_nb = self.nb_search_out(self.regular_points, self.points,0.05)
        x = self.gno(y=self.regular_points, neighbors=in_to_out_nb, x=self.points, f_y=x)
        x=torch.nan_to_num(x)
        return x

model=MyGeoTFNO(n_modes=16,in_channels=1 ,hidden_channels=500,out_channels=1,points=inp_positions,num_train_points=25)
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
        l = loss(y_pred.reshape(-1), rabbit_u_train.reshape(-1))
        l.backward()
        optimizer.step()
        print(l.item())


model.eval()
for data in test_data_loader:
    rabbit_x_train,rabbit_u_train=data
    y_pred = model(rabbit_x_train)
    l = loss(y_pred.reshape(-1), rabbit_u_train.reshape(-1))
    l.backward()
    optimizer.step()

