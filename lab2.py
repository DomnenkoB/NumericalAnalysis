import math

def converges(M):
	for i in range(0, len(M)):
		row_sum = 0
		for j in range(0, len(M)):
			if (i != j):
				row_sum += math.fabs(M[i][j])
		if row_sum > math.fabs(M[i][i]):
			return False
	return True

def norm(x):
	res = 0
	for i in range(0, len(x)):
		res += math.fabs(x[i])
	return res

def tridiag(M, d):
	if converges(M) == False:
		print('Method diverges')
		return
	A = []
	B = []
	C = []
	x = []
	n = len(M)

	A.append(0)
	for i in range (0, n):
		B.append(M[i][i])
		x.append(0)
	for i in range(0, n - 1):
		A.append(M[i + 1][i])
		C.append(M[i][i + 1])
	C.append(0)

	print(A)
	print(B)
	print(C)

	C[0] /= B[0]
	d[0] /= B[0]

	for i in range(1, n):
		div = 1 / (B[i] - C[i - 1] * A[i])
		C[i] *= div
		d[i] *= (d[i] - d[i - 1] * A[i]) * div

	x[len(M) - 1] = d[n - 1]
	for i in range(n - 2, -1, -1):
		x[i] = d[i] - C[i] * x[i + 1]

	return x

def jacobi(A, b):
	if converges(M) == False:
		print('Method diverges')
		return

	n = len(A)
	#print(n)
	D = [[0 for i in range(0, n)] for j in range(0, n)]
	R = [[0 for i in range(0, n)] for j in range(0, n)]
	
	for i in range(0, n):
		for j in range(0, n):
			if i != j:
				R[i][j] = A[i][j]
			else:
				D[i][i] = A[i][i]

	#print(D)
	#print(R)

	eps = 1e-5

	x_prev = [0 for i in range(0, n)]
	x_cur = [0 for i in range(0, n)]
	dif = [1 for i in range(0, n)]


	while math.fabs(norm(dif)) > eps:
		#it += 1
		for i in range(0, n):
			x_prev[i] = x_cur[i]
			
		for i in range(0, n):
			r = 0
			for j in range(0, n):
				if i != j:
					r += R[i][j] * x_prev[j]
			x_cur[i] = (1 / D[i][i]) * (b[i] - r)

		for i in range(0, n):
			dif[i] = x_cur[i] - x_prev[i]

	return(x_cur)




M = [
	[4, 1, 0, 0],
	[1, 3, 1, 0],
	[0, 1, 7, -3],
	[0, 0, -5, 8]
]

d = [1, 1, 1, 1]

print(tridiag(M, d))

M = [
	[4, 1, 0, 0],
	[1, 3, 1, 0],
	[0, 1, 7, -3],
	[0, 0, -5, 8]
]

d = [1, 1, 1, 1]

print(jacobi(M, d))