import math
import random
import numpy as np
import matplotlib.pyplot as plt

def f(x):
	return np.power(2, x) + x * x - 1.15

def d_f(x):
	return np.log(2) * np.power(2, x) + 2 * x

def d2_f(x):
	return np.log(2) * np.log(2) * np.power(2, x) + 2

def alpha(x):
	return (-1.0) / d_f(x)

def newton(a, b, eps):
	x = random.uniform(a, b)

	#convergence_crt

	if (math.fabs(1.0 + alpha(x) * d_f(x))):
		print('Method diverges, x value:')
		print(x)
	else:
		# in order to iterate we decrease the value of x_n
		x_n = x
		x_n -= 2 * eps
		iter_cnt = 0

			#'iteration' process

			while math.fabs(x_n - x) > eps:
				x = x_n
				x_n = x + f(x) * alpha(x)
				iter_cnt += 1

				print('Iteration: ' + str(iter_cnt) + ', x value:')
				print(x_n)

			print('Answer:')
			print((x_n))

	def modified_newton(a, b, eps):
		x = random.uniform(a, b)

		#convergence_crt

		print(math.fabs(1.0 + alpha(x) * d_f(x)))
		if (math.fabs(1.0 + alpha(x) * d_f(x)) > 1):
			print('Method diverges, x value:')
			print(x)
		else:
			# in order to iterate we decrease the value of x_n
			x_n = x
			x_n -= 2 * eps
		iter_cnt = 0

		#'iteration' process

		while math.fabs(x_n - x) > eps:
			x = x_n
			x_n = x - (f(x) * eps) / (f(x + eps) - f(x))
			iter_cnt += 1
			print('Iteration: ' + str(iter_cnt) + ', x value:')
			print(x_n)

		print('Answer:')
		print((x_n))


newton(-1, 1, 0.001)
modified_newton(-1, 1, 0.001)


x = np.arange(-4, 4, 0.01)

plt.title('Function 2^x + x*x - 1.15')
plt.grid(True)
plt.ylim(-8, 8)
plt.plot(x, f(x))
plt.show()

plt.title('Function log(2)*2^x + 2x')
plt.grid(True)
plt.ylim(-8, 8)
plt.plot(x, d_f(x))
plt.show()

plt.title('Function log^2(2)*2^x + 2')
plt.grid(True)
plt.ylim(-8, 8)
plt.plot(x, d2_f(x))
plt.show()

