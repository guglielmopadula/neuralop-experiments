import numpy as np
import scipy.spatial
radius=np.random.rand(10000,1)
angle=np.random.rand(10000,1)*2*np.pi
x=radius*np.cos(angle)
y=radius*np.sin(angle)
data=np.concatenate((x,y),axis=1)
np.save("data_irr_2d_dense.npy",data)
triangles=scipy.spatial.Delaunay(data).simplices
np.save("triangles_irr_2d_dense.npy",triangles)



theta=np.random.rand(500,1,2)
z=np.sin(theta*data.reshape(1,-1,2))
z=z.sum(axis=2)
np.save("fun_irr_2d_dense.npy",z)


import numpy as np
import scipy.spatial
radius=np.random.rand(5000,1)
angle=np.random.rand(5000,1)*2*np.pi
x=radius*np.cos(angle)
y=radius*np.sin(angle)
data=np.concatenate((x,y),axis=1)
np.save("data_irr_2d.npy",data)
triangles=scipy.spatial.Delaunay(data).simplices
np.save("triangles_irr_2d.npy",triangles)



theta=np.random.rand(500,1,2)
z=np.sin(theta*data.reshape(1,-1,2))
z=z.sum(axis=2)
np.save("fun_irr_2d.npy",z)

