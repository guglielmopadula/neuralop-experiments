import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import time
import dolfinx.fem.petsc
from dolfinx.fem import FunctionSpace
import pyvista
from mpi4py import MPI
from dolfinx import mesh
import tetgen
import meshio
from tqdm import trange

def volume_2_x(mesh):
    shape=mesh.shape
    mesh=mesh.reshape(-1,mesh.shape[-3],mesh.shape[-2],mesh.shape[-1])
    tmp=np.sum(np.sum(mesh[:,:,:,0],axis=2)*(np.linalg.det(mesh[:,:,1:,1:]-np.expand_dims(mesh[:,:,0,1:],2))/6),axis=1)
    return tmp.reshape(shape[:-3])


def calculate_simulation(mu,nodes,elem,bary):
    start=time.time()
    nodes=nodes-np.min(nodes,axis=0)
    gdim = 3
    shape = "tetrahedron"
    degree = 1
    cell = ufl.Cell(shape, geometric_dimension=gdim)
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))
    domain = mesh.create_mesh(MPI.COMM_WORLD, elem, nodes, domain)
    V = FunctionSpace(domain, ("CG", 1))
    uD = fem.Function(V)
    uD.interpolate(lambda x: np.exp(-mu*((x[0]-bary[0])**2 + (x[1]-bary[1])**2+(x[2]-bary[2])**2)**0.5))
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V) 
    f = fem.Function(V)
    f.interpolate(lambda x: np.exp(-((x[0]-bary[0])**2 + (x[1]-bary[1])**2+(x[2]-bary[2])**2)))
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    energy=fem.form(u* ufl.dx)
    points = domain.geometry.x
    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()    
    value=fem.assemble.assemble_scalar(energy)
    u_val=uh.x.array
    ud_val=uD.x.array
    end=time.time()
    print(end-start)
    return value,u_val,ud_val,points

if __name__=="__main__":
    np.random.seed(0)
    NUM_SAMPLES=300
    points=np.load("points.npy").astype(np.float64)
    tets=np.load("tetras.npy").astype(np.float64)
    bary=np.mean(points,axis=0)
    value,uh,udh,points_mesh=calculate_simulation(1,points,tets,bary)
    energy_data=np.zeros(NUM_SAMPLES)
    u_data=np.zeros((NUM_SAMPLES,len(uh)))
    x=np.zeros((NUM_SAMPLES,len(uh)))
    energy_data[0]=value
    u_data[0]=uh
    x[0]=udh


    for i in trange(1,NUM_SAMPLES):
        value,uh,udh,_=calculate_simulation(1+3*i/(NUM_SAMPLES-1),points,tets,bary)
        energy_data[i]=value
        u_data[i]=uh
        x[i]=udh

    np.save("rabbit_energy.npy",energy_data)
    np.save("rabbit_u.npy",u_data)
    np.save("rabbit_x.npy",x)
    np.save("bary.npy",bary)
    np.save("points_mesh.npy",points_mesh)