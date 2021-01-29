""" 2D double broadcas matrix multiplication """

from sys import argv

from math import sqrt
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

Q = int(sqrt(size))
assert int(sqrt(size)) == sqrt(size), 'number of core is not a perfect square'

myrow,   mycol =   int(rank//Q),             int(rank%Q)
rowComm, colComm = comm.Split(myrow, mycol), comm.Split(mycol, myrow)

N = int(argv[1])//Q
A = np.random.randint(-1000, 1000, (N, N))
B = np.random.randint(-1000, 1000, (N, N))

# with open(f'data\\data-A-{rank}.dat', 'rb') as f:
# 	A = np.load(f)
# with open(f'data\\data-B-{rank}.dat', 'rb') as f:
# 	B = np.load(f)

BuffA = np.copy(A)
BuffB = np.copy(B)
C = np.zeros(A.shape)

start_time = MPI.Wtime()

for k in range(Q):
	tmpA = A if mycol == k else BuffA
	tmpB = B if myrow == k else BuffB

	rowComm.Bcast(tmpA, root=k)
	colComm.Bcast(tmpB, root=k)

	C += np.matmul(tmpA, tmpB)

end_time = MPI.Wtime()

start_times = comm.gather(start_time, root=0)
end_times = comm.gather(end_time, root=0)

if rank == 0:
	print(round(max(end_times) - min(start_times), 4))

# with open(f'res\\res-{rank}.dat', 'wb') as f:
# 	np.save(f, C)
