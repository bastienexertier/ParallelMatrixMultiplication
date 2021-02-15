""" 3D Matrix Multiplication """

import sys

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

try:
	M = N = K = int(sys.argv[1])
	p1 = int(sys.argv[2])
	p2 = int(sys.argv[3])
	p3 = int(sys.argv[4])
except IndexError:
	print('python 3D.py N p1 p2 p3')
	sys.exit(0)
except ValueError:
	print('N, p1, p2, p3 must be integers')
	sys.exit(0)

m, n, k = M//p1, N//p2, K//p3
k2, n1, n3 = k//p2, n//p1, n//p3

assert size == p1 * p2 * p3, 'p1p2p3 != nb of mpi processes'

i = rank%p1
j = (rank//p1)%p2
l = (((rank-i)//p1)-j)//p2

assert 0 <= i < p1
assert 0 <= j < p2
assert 0 <= l < p3

A_il_j = np.random.randint(-1000, 1000, (m, k2))
B_lj_i = np.random.randint(-1000, 1000, (k, n1))

# Load matrices from files for testing

# with open(f'data\\data-A-{rank}.dat', 'rb') as f:
# 	A_il_j = np.load(f)
# with open(f'data\\data-B-{rank}.dat', 'rb') as f:
# 	B_lj_i = np.load(f)

assert A_il_j.shape == (m, k2)
assert B_lj_i.shape == (k, n1)

start_time = MPI.Wtime()

# Creation of groups (1)

G_il = comm.Split(hash((i, l))%(2<<16), j)
G_lj = comm.Split(hash((l, j))%(2<<16), i)
G_ij = comm.Split(hash((i, j))%(2<<16), l)

assert G_il.Get_size() == p2
assert G_lj.Get_size() == p1
assert G_ij.Get_size() == p3

# All gather (2,3)

A_il = np.concatenate(G_il.allgather(A_il_j), axis=1)
B_lj = np.concatenate(G_lj.allgather(B_lj_i), axis=1)

assert A_il.shape == (m, k)
assert B_lj.shape == (k, n)

# Local mat mul (4)

D_ij_l = np.matmul(A_il, B_lj)

assert D_ij_l.shape == (m, n)

# all to all (5)

D_ij_r_l = G_ij.alltoall([D_ij_l[:,r*n3:(r+1)*n3] for r in range(p3)])

assert all([d.shape == (m, n3) for d in D_ij_r_l])
assert len(D_ij_r_l) == p3

# sum (6)

C_ij_l = sum(D_ij_r_l)

assert C_ij_l.shape == D_ij_r_l[0].shape

end_time = MPI.Wtime()

# Saving results for testing

# with open(f'res\\res-{rank}.dat', 'wb') as f:
# 	np.save(f, C_ij_l)

if rank == 0:
	print(round(max(end_times) - min(start_times), 4))
