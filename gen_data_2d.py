import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import numpy as np
from tqdm import trange
def compute(mu):
    msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (1.0, 1.0)), n=(100, 100),
                            cell_type=mesh.CellType.triangle,)
    V = fem.FunctionSpace(msh, ("Lagrange", 1))
    facets = mesh.locate_entities_boundary(msh, dim=1,
                                       marker=lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 0.0),
                                                                      np.isclose(x[0], mu)),np.logical_or(np.isclose(x[1], 0.0),
                                                                      np.isclose(x[1], mu))))
    dofs = fem.locate_dofs_topological(V=V, entity_dim=1,entities=facets)
    bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = inner(grad(u), grad(v)) * dx + mu*inner(u, v) * dx
    x = ufl.SpatialCoordinate(msh)
    u_n = fem.Function(V)
    f =ufl.sin((x[0] - mu/2) ** 2 + (x[1] - mu/2) ** 2)   
    def fun_f(x):
       return np.sin((x[0] - mu/2) ** 2 + (x[1] - mu/2) ** 2)
    L = inner(f, v) * dx 
    problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    u_n.interpolate(fun_f)
    points = msh.geometry.x
    return np.array(u_n.vector),np.array(uh.vector),points 

test_y,test_x,points=compute(1.0)
points=np.array(points*100,dtype=np.int64)
y_vec=np.zeros((100,101,101))
x_vec=np.zeros((100,101,101))
for i in trange(100):
  y,x,pts=compute(1.0+4.0*i/100)
  for j in range(len(points)):
      y_vec[i,points[j,0],points[j,1]]=y[j]
      x_vec[i,points[j,0],points[j,1]]=x[j]
np.save("fem_sol_y.npy",y_vec)
np.save("fem_sol_x.npy",x_vec)
