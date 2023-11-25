import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
from scipy.sparse.linalg import spsolve
from tqdm import trange
from scipy.sparse.linalg import eigsh,lobpcg
from scipy.sparse import coo_matrix,csr_matrix
import scipy.sparse
import itertools
import torch
import scipy.io
'''
def circumcenter(a, b, c):
    l = np.array([
        np.linalg.norm(b - c)**2,
        np.linalg.norm(a - c)**2,
        np.linalg.norm(a - b)**2
    ])
    
    ba = np.array([
        l[0] * (l[1] + l[2] - l[0]),
        l[1] * (l[2] + l[0] - l[1]),
        l[2] * (l[0] + l[1] - l[2])
    ])
    
    sum_ba = np.sum(ba)
    
    cc = (ba[0] / sum_ba) * a + (ba[1] / sum_ba) * b + (ba[2] / sum_ba) * c
    return cc

def volume(a, b, c, d):
    A = np.column_stack((b - a, c - a, d - a))
    return np.linalg.det(A) / 6.0

def dual_laplace(V, T):
    nt = T.shape[0]
    nv = V.shape[0]

    turn = np.array([
        [-1, 2, 3, 1],
        [3, -1, 0, 2],
        [1, 3, -1, 0],
        [2, 0, 1, -1]
    ])

    def get_tet(i):
        return V[T[i]]

    tripL = []
    tripM = []

    for k in trange(nt):
        t = get_tet(k)
        cc = circumcenter(t[0], t[1], t[2])

        for i in range(4):
            for j in range(4):
                if i != j:
                    cf = circumcenter(t[i], t[j], t[turn[i][j]])
                    ce = 0.5 * (t[i] + t[j])
                    vol = volume(t[i], ce, cf, cc)
                    wij = 6.0 * vol / np.linalg.norm(t[i] - t[j])**2

                    tripL.append((T[k, i], T[k, j], wij))
                    tripL.append((T[k, j], T[k, i], wij))

                    tripL.append((T[k, i], T[k, i], -wij))
                    tripL.append((T[k, j], T[k, j], -wij))

                    tripM.append((T[k, i], T[k, i], vol))
                    tripM.append((T[k, j], T[k, j], vol))

    L = lil_matrix((nv, nv))
    M = lil_matrix((nv, nv))

    for (i, j, val) in tripL:
        L[i, j] = val
    for (i, j, val) in tripM:
        M[i, j] = val

    L = csr_matrix(L)
    M = csr_matrix(M)

    return L, M



def compute_edges_undirected(n_faces):
    h=n_faces.shape[1]
    comb=np.array(list(itertools.combinations(range(h),2)))
    comb_bak=comb.copy()
    comb[:,1]=comb_bak[:,0]
    comb[:,0]=comb_bak[:,1]
    comb=np.vstack((comb,comb_bak))
    faces_comb=n_faces[:,comb].reshape(-1,2)
    faces_comb=np.sort(faces_comb,axis=1)
    faces_comb=np.unique(faces_comb,axis=0)
    return faces_comb



T=np.load("triangles_irr_2d_dense.npy")

edges=compute_edges_undirected(T)


def laplacian(edges):
    """
    Computes the laplacian matrix.
    The definition of the laplacian is
    L[i, j] = deg(i)       , if i == j
    L[i, j] = -1  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph.

    Args:
        verts: tensor of shape (V, 3) containing the vertices of the graph
        edges: tensor of shape (E, 2) containing the vertex indices of each edge
    Returns:
        L: Sparse FloatTensor of shape (V, V)
    """
    V = torch.max(edges) + 1

    e0, e1 = edges.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=edges.device)
    A = torch.sparse_coo_tensor(idx, ones, (V, V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()
    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = -1 if (i, j) is an edge
    deg0 = deg[e0]
    deg0 = torch.where(deg0 > 0.0, -1.0, deg0)
    deg1 = deg[e1]
    deg1 = torch.where(deg1 > 0.0, -1.0, deg1)
    val = torch.cat([deg0, deg1])
    L = torch.sparse_coo_tensor(idx, val, (V, V))
    # Then we add the diagonal values L[i, i] = deg(i).
    idx = torch.arange(V, device=edges.device)
    idx = torch.stack([idx, idx], dim=0)
    L += torch.sparse_coo_tensor(idx, deg, (V, V))

    return L


T=np.load("triangles_irr_2d_dense.npy")
edges=compute_edges_undirected(T)
edges=torch.from_numpy(edges)
L=laplacian(edges)
w,v=torch.lobpcg(L,k=512,largest=False)
v=v.numpy()
np.save("eigenbasis_irr_2d_dense.npy",v)


'''

import torch
from torch import nn
v=np.load("eigenbasis_irr_2d_dense.npy")
z=np.load("fun_irr_2d_dense.npy")
latent_dense=z@v
print(np.linalg.norm(z-z@v@v.T)/np.linalg.norm(z))
m=z.shape[1]
theta=np.load("x_irr_2d_dense.npy")
theta=theta.reshape(-1,1,2)
theta=np.tile(theta,(1,m,1))

def compute_reduction(quantity,v):
    num_points=v.shape[0]
    num_red=v.shape[1]
    quantity_=quantity.reshape(quantity.shape[0],num_points,-1).clone()
    out_shape=quantity_.shape[2]
    quantity_=np.transpose(quantity_,(0,2,1))
    quantity_=quantity_@v
    return quantity_.reshape(-1,out_shape*num_red)

def revert_reduction(quantity,v):
    num_points=v.shape[0]
    num_red=v.shape[1]
    quantity_=quantity.reshape(quantity.shape[0],-1,num_red).clone()
    out_shape=quantity_.shape[1]
    quantity_=(quantity_@v.T)
    quantity_=np.transpose(quantity_,(0,2,1))
    return quantity_.reshape(quantity.shape[0],num_points,out_shape)


z=torch.tensor(z,dtype=torch.float32).reshape(-1,v.shape[0],1)
theta=torch.tensor(theta,dtype=torch.float32)
v=torch.tensor(v,dtype=torch.float32)

z_red=compute_reduction(z,v)
theta_red=compute_reduction(theta,v)


class LaplaceNeuralOperator(torch.nn.Module):
    def __init__(self,eigenvectors,input_size,output_size):
        super(LaplaceNeuralOperator, self).__init__()
        self.eigenvectors=eigenvectors
        self.num_points=eigenvectors.shape[0]
        num_red=eigenvectors.shape[1]
        self.input_size=input_size
        self.output_size=output_size
        self.nn=nn.Sequential(nn.Linear(input_size*num_red,500),nn.BatchNorm1d(500),nn.ReLU(),
                              nn.Linear(500,500),nn.BatchNorm1d(500),nn.ReLU(),
                              nn.Linear(500,500),nn.BatchNorm1d(500),nn.ReLU(),
                              nn.Linear(500,500),nn.BatchNorm1d(500),nn.ReLU(),
                              nn.Linear(500,output_size*num_red))


    def forward(self, x):
        return self.nn(x)
    
    def train(self,data_x,data_y):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(1000):
            optimizer.zero_grad()
            data_x_red=compute_reduction(data_x,self.eigenvectors)
            data_y_red=compute_reduction(data_y,self.eigenvectors)
            data_y_pred_red = self(data_x_red)
            l = torch.linalg.norm(data_y_pred_red-data_y_red)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                data_y_pred=revert_reduction(data_y_pred_red,self.eigenvectors)
                print(torch.linalg.norm(data_y_pred-data_y)/torch.linalg.norm(data_y))
        return l
    

model=LaplaceNeuralOperator(v,2,1)

model.train(theta,z)