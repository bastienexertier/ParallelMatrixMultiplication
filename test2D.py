""" test module for 2D matri multiply """
# run this code to generate input
# run 2D to compute the result
# press y to compare

import numpy as np

N = 1500
Q = 3
D = N//Q
A = np.random.randint(-10, 10, (N, N))
B = np.random.randint(-10, 10, (N, N))

for rank in range(Q**2):
	myrow, mycol = int(rank//Q), int(rank%Q)

	with open(f'data\\data-A-{rank}.dat', 'bw') as f:
		a = A[D*myrow: D*(myrow+1), D*mycol:D*(mycol+1)]
		np.save(f, a)
	with open(f'data\\data-B-{rank}.dat', 'bw') as f:
		b = B[D*myrow: D*(myrow+1), D*mycol:D*(mycol+1)]
		np.save(f, b)

while input('result ready?') != 'y':
	print('press y')

res = [[] for _ in range(Q)]
for i in range(Q):
	for j in range(Q):
		with open(f'res\\res-{i*Q + j}.dat', 'rb') as f:
			res[i].append(np.load(f))

for i, sub in enumerate(res):
	res[i] = np.concatenate(sub, axis=1)
res = np.concatenate(res, axis=0)

print((res == np.matmul(A, B)).all())
