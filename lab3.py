import math
import numpy as np
import numpy.linalg as linalg
import scipy import optimize
from numpy.linalg import inv

def matrix_norm(A):
	norm_A = linalg.norm(np.array(A), 1)
	return norm_A

def norm_vect(x):
	x_norm = vector_norm(x)
	for i in range(len(x)):
		x[i] /= x_norm
	return x

def vector_norm(x):
	res = 0
	for i in range(len(x)):
		res += x[i] * x[i]
	return math.sqrt(res)

def vect_vect_mult(x, y):
	res = 0
	for i in range(len(x)):
		res += x[i] * y[i]
	return res

def matrix_vect_mult(A, x):
	res = []
	for i in range(len(A)):
		res_i = 0
		for j in range(len(x)):
			res_i += A[i][j] * x[j]
		res.append(res_i)

	return res

def scalar_vect_mult(a, x):
	for i in range(len(x)):
		x[i] *= a
	return x

def vect_vect_sub(x, y):
	for i in range(len(x)):
		x[i] -= y[i]
	return x

def vector_iter(A, x):
	y_k = matrix_vect_mult(A, x)
	lambda1 = vect_vect_mult(x, y)
	#print (lambda1)
	res = 0
	x_k = x
	i = 0
	tolerance = 1e-6
	while vector_norm(vect_vect_sub(y_k, scalar_vect_mult(lambda1, x))) > tolerance:
		#print (vector_norm(vect_vect_sub(y_k, scalar_vect_mult(lambda1, x))))
		x_k = norm_vect(y_k) # x = y / ||y||
		#print (i)
		#print (lambda1)
		y_k = matrix_vect_mult(A, x_k) # y = A*x
		lambda1 = vect_vect_mult(x_k, y_k) #eig = (A * x) * x where ||x|| = 1
		res = lambda1
		print(lambda1)
	return res

#eig_min = 


A = [ [7, 2, 1, 1],  [2, 8, 3, 1], [1, 3, 9, 1], [1, 1, 1, 10] ]

x = [-2.52, 3.13, -1.91, 1]
y = [1, 0, 0, 0]

x = norm_vect(x)
print (vector_norm(x))
#print (matrix_vect_mult(A, [1, 1, 1, 1]))

print (vector_iter(A, x))

def fun(x):
	return [(x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0) / 2.0, (0.5 * (x[1] - x[0])**3 + x[1]) / 2.0]

def jac(x):
	return [[(1 + 1.5 * (x[0] - x[1])**2) / 2.0, (-1.5 * (x[0] - x[1]) ** 2) / 2.0], [(-1.5 * (x[1] - x[0])**2) / 2.0, (1.5(x[1] - x[0])**2 + 1) / 2.0]]

solution =  optimize.root(fun, [0, 0], jac=jac, method='hybr')

print(solution.x)