# MPI Project Comparison of 2 matrix multiplication algorithm

## Introduction

### Goals

My goal is to compare two matrix multiplication alorithm :
* 2D double broadcast matrix multiplication
* 3D matrix multiplication
The goal is to find which algorithm is better for "small" matrices, which one is better for bigger matrices and around where the transition is.

### Environment

For this project I worked with **python** and **mpi4py**.  
For the data I used the library **numpy**.  
Every local matrix multiplication was done by `numpy.matmul` as it is probably the best implementation in python.

## 2D Double Broadcast

### Concept

This algorithm divides the matrices into `q*q = p` parts.
Each process starts with a part of A and B.
Then, at each step i between 0 and q, every process in row i broadcasts its part A to every process in the same column.
Every process in column i broadcast its part of B to every other process in the same row.
Then every matrix compute a local matrix multiplication on the two parts of A and B it as.

### Code

#### Communicators

```python
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

Q = int(sqrt(size))

myrow, mycol = int(rank//Q), int(rank%Q)
rowComm, colComm = comm.Split(myrow, mycol), comm.Split(mycol, myrow)
```

Each process finds the row and the column it belongs to. Then, it joins a communicator with every other process in the same row, and a communicator with every process in the same column.

#### Buffers

We initialize two buffer for storing the receive A and B parts, and a buffer for the local part of the result C.

```python
BuffA = np.empty(A.shape, dtype='i')
BuffB = np.empty(B.shape, dtype='i')
C = np.zeros(A.shape)
```

#### Main Loop

```python
for k in range(Q):
	tmpA = A if mycol == k else BuffA
	tmpB = B if myrow == k else BuffB

	rowComm.Bcast(tmpA, root=k)
	colComm.Bcast(tmpB, root=k)

	C += np.matmul(tmpA, tmpB)
```

For each loop, each process sets the variables `tmpA` and `tmpB`. Those variables either hold `A` and `B` if its the time for the process to broadcast (send) its part, and `buffA` and `buffB` if it needs to broadcast (receive) the parts of another process. 

Then the process calls the `broadcast` function with `root=k`.

Finally, we do a local `C += matmul(A, B)` with the local parts.

## 3D Algorithm

### Source

I implemented the 3D matrix multiplication algorithm following this [paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.120.4575&rep=rep1&type=pdf). All the sections of the code will show the number of the step it corresponds to in the paper.

### Code

#### Useful variables

```python
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

m, n, k = M//p1, N//p2, K//p3
k2, n1, n3 = k//p2, n//p1, n//p3

i = rank%p1
j = (rank//p1)%p2
l = (((rank-i)//p1)-j)//p2
```

Each process compute its set of values `i`, `j` and `l`.

#### Communicators (1)

We set the communicators `G_il` for `A` between every processor having the same `(i, l)`.  
We set the communicators `G_lj` for `B` between every processor having the same `(l, j)`.  
We set the communicators `G_ij` for `C` between every processor having the same `(i, j)`.  
However we can't use a tuple as the `color` of the communicator. So, we will take advantage of python's `hash` function that will return a unique integer for every couple (a, b).  
The new rank is equal to the letter not used.

```python
G_il = comm.Split(hash((i, l)), j)
G_lj = comm.Split(hash((l, j)), i)
G_ij = comm.Split(hash((i, j)), l)
```

#### All Gather (2, 3)

The first step is to do an **All Gather** of each part of A and B that processes hold, over their designated communicators.  
We then concatenate the list of arrays back into a matrix.

```python
A_il = np.concatenate(G_il.allgather(A_il_j), axis=1)
B_lj = np.concatenate(G_lj.allgather(B_lj_i), axis=1)
```

#### Local Matmul (4)

Each process then compute its local `matmul(A_il, B_lj)`.

```python
D_ij_l = np.matmul(A_il, B_lj)
```

#### AlltoAll and sum (5, 6)

We then slice the result and perform an `alltoall` to share the result over the `G_ij` (C dedicated communicators).
Finally, each process `sums` matrices received.

```python
D_ij_r_l = G_ij.alltoall([D_ij_l[:,r*n3:(r+1)*n3] for r in range(p3)])
C_ij_l = sum(D_ij_r_l)
```

## Performances

For the comparison we need to choose a number or processes which is the same for the 2D and 3D algorithms. So, we need to finds number that are power of two (as most machines hold a power of two number of cores), which is also a square (for 2D algorithm) : `16, 64, 256, ..`. We will only study the performances for `p=16` and `p=64`.  

### P = 16

### P = 64
