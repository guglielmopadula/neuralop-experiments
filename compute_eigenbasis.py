import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
from scipy.sparse.linalg import spsolve
from tqdm import trange
from scipy.sparse.linalg import eigsh
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

P=np.load("points.npy")
T=np.load("tetras.npy")
L,M=dual_laplace(P,T)
eigs,vecs=eigsh(-L,k=64,M=M,which='SM',mode="buckling",sigma=0)
lapl_spectrum_matrix=vecs
np.save("lapl_spectrum_matrix.npy",lapl_spectrum_matrix)

#eigvals, eigvecs = eigh(L.toarray(), M.toarray())