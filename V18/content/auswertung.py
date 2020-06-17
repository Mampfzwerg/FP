import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

A0 = ufloat(4130, 60)
t05 = ufloat(4943, 5)
t = 7177

A = A0 * unp.exp(- np.log(2) / t05 * t)

#print(A)

m = ufloat(0.403, 0)
d = ufloat(-2.683, 0.051)
mu = 1647.72

#E = m * mu2 + d
E = m * mu + d

#print(E)

dE = 2.35 * unp.sqrt(0.1 * ufloat(661.35, 0.05) * 2.9e-3)
#print(dE)


eps = E / 511

EK = E * 2 * eps / (1 + 2 * eps)
EP = E / (1 + 2 * eps)

#print (EK, EP)

y = np.genfromtxt('mess2.txt', unpack=True)
x = np.arange(0, 8192)

#print(np.sum(y[0:1180]))
#print(np.sum(y[1640:1655]))

m = np.array([0.006, 0.85, 0.45])
D = 4.5
W = (1 - np.exp(- m * D)) * 100

print(W)