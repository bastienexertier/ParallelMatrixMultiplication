
import numpy as np

M = 8
N = 16
K = 16
p1, p2, p3 = 4, 4, 2
m, n, k = M//p1, N//p2, K//p3
k2, n1, n3 = k//p2, n//p1, n//p3

A = np.random.randint(-10, 10, (M, K))
B = np.random.randint(-10, 10, (K, N))
Q = p1*p2*p3


for rank in range(Q):
	i, j, l = rank//(p2*p3), (rank//p2)%p3, rank%p1
	i, j, l = rank%p1, (rank//p3)%p2, rank//(p2*p3)
	i = rank%p1
	j = (rank//p1)%p2
	l = (((rank-i)//p1)-j)//p2

	with open(f'data\\data-A-{rank}.dat', 'bw') as f:
		A_il = A[i*m:(i+1)*m, l*k:(l+1)*k]
		a = A_il[:, j*k2:(j+1)*k2]
		np.save(f, a)
	with open(f'data\\data-B-{rank}.dat', 'bw') as f:
		B_lj = B[l*k:(l+1)*k, j*n:(j+1)*n]
		b = B_lj[:, i*n1:(i+1)*n1]
		np.save(f, b)

while input('result ready?') != 'y':
	print('press y')

res = [[] for _ in range(Q)]
for rank in range(Q):
	with open(f'res\\res-{rank}.dat', 'rb') as f:
		res[rank].append(np.load(f))

C = np.matmul(A, B)
assert C.shape == (M, N)
print(C)
print(res)
