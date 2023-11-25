import torch
from tqdm import trange
import numpy as np
import torch.nn as nn
device = 'cpu'
import faiss


x=np.load("rabbit_x.npy").reshape(300,-1,1)
y=np.load("rabbit_u.npy").reshape(300,-1,1)
points_mesh=np.load("points_mesh.npy")
n=x.shape[0]
m=x.shape[1]
test_subset=np.random.choice(n, size=100, replace=False, p=None)
A=np.arange(n)
train_subset=A[~np.in1d(A,test_subset)]



class GNO(nn.Module):
    def __init__(self,in_channels,out_channels,points,radius=0.0015):
        super(GNO, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.radius=radius
        self.points=points
        self.linear=nn.Linear(self.in_channels,self.out_channels,bias=False)    
        self.nn=nn.Sequential(nn.Linear(8,100),nn.ReLU(),nn.Linear(100,100),nn.ReLU(),nn.Linear(100,100),nn.ReLU(),nn.Linear(100,out_channels))
        self.list=self.construct_neigh(self.points,self.radius)

    def construct_neigh(self,points,r):
        index=faiss.IndexFlatL2(3)
        n=points.shape[0]
        index.add(points.cpu().numpy())
        lims,D,I=index.range_search(points.cpu().numpy(),r)
        lims=torch.tensor(lims.astype(np.int64)).to(points.device)
        I=torch.tensor(I.astype(np.int64)).to(points.device)
        diff_lims=torch.diff(lims)
        rep=torch.repeat_interleave(torch.arange(n),diff_lims)
        return torch.concatenate((rep.unsqueeze(0),I.unsqueeze(0)),dim=0).to(points.device)


    def forward(self,batch):
        points_x=self.points[self.list[1]].unsqueeze(0).repeat(batch.shape[0],1,1)
        points_y=self.points[self.list[0]].unsqueeze(0).repeat(batch.shape[0],1,1)
        batch_x=batch[:,self.list[1],:]
        batch_y=batch[:,self.list[0],:]
        new_batch=torch.concatenate((points_x,points_y,batch_x,batch_y),dim=2)
        tmp_list=self.list[0].unsqueeze(0).unsqueeze(-1).repeat(batch.shape[0],1,batch.shape[2])
        tmp_array=torch.zeros(batch.shape,requires_grad=True)
        new_batch=self.linear(batch)+torch.scatter_reduce(tmp_array,1,tmp_list,self.nn(new_batch),reduce='mean')
        return new_batch
    



t=0
print(np.max(points_mesh))

x=torch.tensor(x,dtype=torch.float32)
y=torch.tensor(y,dtype=torch.float32)
points_mesh=torch.tensor(points_mesh,dtype=torch.float32)
gno=GNO(1,1,points_mesh)
gno=gno.to(device)


x_train=x[train_subset]
y_train=y[train_subset]
x_test=x[test_subset]
y_test=y[test_subset]

train_dataset=torch.utils.data.TensorDataset(x_train,y_train)
test_dataset=torch.utils.data.TensorDataset(x_test,y_test)
train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
test_dataloader=torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


model=GNO(1,1,points_mesh)

num_epochs=10
learning_rate=0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in trange(num_epochs):
    tot_loss=0
    for data in train_dataloader:
        data=data
        x,y=data
        x=x.to(device)
        y=y.to(device)
        y_pred = model(x).reshape(y.shape)
        l = torch.linalg.norm(y_pred-y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        tot_loss=tot_loss+((torch.norm(y_pred-y)/torch.linalg.norm(y))/len(train_dataloader)).item()
    print("Train loss is", tot_loss)

model=model.eval()
tot_loss=0
for data in test_dataloader:
    data=data
    x,y=data
    x=x.to(device)
    y=y.to(device)
    y_pred = model(x).reshape(y.shape)
    tot_loss=tot_loss+((torch.norm(y_pred-y)/torch.linalg.norm(y))/len(test_dataloader)).item()

print("Test loss is", tot_loss)




