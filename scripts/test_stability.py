#!/usr/bin/python3
'''
Linear algebra analysis for different matrices
'''

import numpy as np

A1 = np.array([[-1, -10],
               [1, -1]])


eigs = np.linalg.eig(A1)

print('Eigenvlaue of Matrix A1', eigs[0])

A1_sum = A1 + A1.T
eigs = np.linalg.eig(A1_sum)

print('Eig value of A1 + A1^T', eigs[0])

A2 = np.array([[-1, 1],
               [-10, -1]])

eigs = np.linalg.eig(A2)

print('Eigenvlaue of Matrix A2', eigs[0])



