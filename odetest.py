import numpy as np
from math import exp
from matplotlib import pyplot as plt
from odeivp import *

def f(t, x):
	return x - t**2 + 1

def y(t):
	return (t + 1.0)**2.0 - 0.5 * np.exp(t)
	
t, x = rkf(f, 0.0, 2.0, 0.5, 1.e-5, 0.01, 0.25)

plt.plot(t, x, 'b.', label='numerical')
plt.plot(t, y(t), label='analytical')
plt.title("Runge Kutta Fehlberg")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend(loc="best")
plt.show()
