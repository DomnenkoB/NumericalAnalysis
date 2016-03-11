import numpy as np
import math
import matplotlib.pyplot as plt

def f(x):
	return x * np.log(x)

def d_f(x):
	return np.log(x) + 1

def find_tau(x):
	# log (x) = const - 1
	return np.exp(x - 1)

def get_coefficients(a, b):
	c_1 = (f(a) - f(b)) / (a - b)

	#tau = find_tau(f(a) - f(b) / (a - b))
	#print (tau)
	tau = 3.561 #THIS KOSTYL` WAS WRITTEN BECAUSE BLOODY NUMPY CHANGES FUNCTION SIGNATURE

	c_0 = - (c_1 * a + c_1 * tau - f(a) - f(tau)) / 2

	#print(c_0, c_1)

	return [c_0, c_1]

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

def f_OLS(x):
	return - 1.53055319 + 1.20961922 * x + 0.1380164 * x * x

#d_f (eps) = f(a) - f(b) / a - b find eps
y = -f(7)
print (y)
a = 1
b = 7
tau = find_tau(f(a) - f(b) / (a - b))
print (tau)
coef = get_coefficients(1, 7)

x = np.arange(-4, 10, 0.01)

y_samples = [[(x / 100) * np.log((x / 100))] for x in range(100, 700, 6)]


OLS(y_samples)
plt.title('x * log (x)')
plt.grid(True)
plt.ylim(-4, 15)
plt.plot(x, f(x), label = "x*log(x)")
plt.plot(x, f_OLS(x), label = "OLS")
plt.plot(x, Q_1(coef, x), label = "Q_1")
plt.legend()
plt.show()
#telecsope
#Разложить в ряд полиномом Чебишева

#Закодить регрессию для приличия

#f(x) = sum {0, inf} (d_j * T_j(x)), [a, b]

#[-1, 1]: f(x) = sum {0, inf} (a_j * pow(x, j)) = Q_n(x)

#Q_n(x) ~ Q_[n-1] (x) = |T_n(x)| <= pow(2, (1 - n))

