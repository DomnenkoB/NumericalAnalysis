import numpy as np
import math
import matplotlib.pyplot as plt

def f(x):
	return x * np.log(x)

def d_f(x):
	return np.log(x) + 1

def find_tau(x):
	return math.exp(x - 1)

def get_coefficients(a, b):
	c_1 = (f(a) - f(b)) / (a - b)

	tau = find_tau(f(a) - f(b) / (a - b))
	c_0 = - (c_1 * a + c_1 * tau - f(a) - f(tau)) / 2

	return [c_0, c_1]

def Q_0(M, m):
	return (m + M) / 2

def Q_1(coef, x):
	c_0 = coef[0]
	c_1 = coef[1]
	return c_0 + c_1 * x

def OLS(samples):
	X = np.matrix([[1, (x / 100), ((x / 100) * (x / 100))] for x in range (100, 700, 6)])
	X_transpose = X.getT()
	Z = (X_transpose * X)

	b = ((Z.getI() * X_transpose) * samples)

	return b

def f_OLS(b, x):
	res = 0
	for i in range(len(b)):
		res += b.item(i, 0) * np.power(x, i)
	return res

a = 1
b = 7
tau = find_tau(f(a) - f(b) / (a - b))
print (tau)
coef = get_coefficients(1, 7)

x = np.arange(-4, 10, 0.01)

y_samples = [[(x / 100) * np.log((x / 100))] for x in range(100, 700, 6)]

m = min([(x / 100)*np.log(x/100) for x in range(100, 700, 6)])
M = max([(x / 100)*np.log(x/100) for x in range(100, 700, 6)])
print(m, M)


c = OLS(y_samples)

print(len(c))

plt.title('x * log (x)')
plt.grid(True)
plt.ylim(-4, 15)
plt.plot(x, f(x), label = "x*log(x)")
plt.plot(x, f_OLS(c, x), label = "OLS")
plt.plot(x, Q_1(coef, x), label = "Q_1")
plt.plot([a, b], [Q_0(M, m), Q_0(M, m)], label = "Q_0")
plt.legend()
plt.show()