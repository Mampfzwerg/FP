import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

#11111111111111111111111111111111111111111111111111111

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

#222222222222222222222222222222222222222222222222222

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

#print(W)

#333333333333333333333333333333333333333333333

#mu = np.array([ufloat(207.667, 0.023), ufloat(692.477, 0.026), ufloat(758.120, 0.016), ufloat(889.976, 0.008), ufloat(959.058, 0.035)]) 
#a = np.array([ufloat(2.40e3, 0.04e3), ufloat(301, 5), ufloat(685, 6), ufloat(1769, 7), ufloat(231, 4)])
#b = np.array([ufloat(3.06, 0.11), ufloat(4.08, 0.15), ufloat(4.53, 0.10), ufloat(5.46, 0.05), ufloat(5.37, 0.23)])
#c = np.array([ufloat(99, 7), ufloat(20.0, 0.9), ufloat(17.4, 1.3), ufloat(8.9, 1.6), ufloat(6.3, 0.9)])
#
#m = ufloat(0.403, 0)
#d = ufloat(-2.683, 0.051)
#E = m * mu + d
#
#e = ufloat(-0.742, 0.221)
#n = ufloat(-0.934, 0.035)
#Q = unp.exp(e) * E ** n
#
#Z = a * unp.sqrt(b * np.pi)
#
#W = np.array([34.1, 0.5, 18.3, 62.1, 8.9])
#A = 4 * Z / (0.0538 * Q * W * 3771)

#for i in range(5):
#    print(E[i], W[i], mu[i], a[i], b[i], c[i], Z[i], Q[i], A[i])

#print(np.mean(A[2:4]))

#44444444444444444444444444444444444444444

mu = np.array([197.82, 237.03, 468.05, 607.09, 739.04, 879.60, 1517.94])
a = np.array([-2054373.96, -4951747.22, -12252338.86, -22934205.13, -76310651.18, -27221198.74, -8988277.11])   
b = np.array([-8490.82, -65164.91, -90236.30, -180267.05, -294905.26, -143724.37, -103001.94]) 
c = np.array([2056192.47, 4953165.50, 12254032.57, 22935710.52, 76313418.70, 27224703.53, 8990390.62])

m = ufloat(0.403, 0)
d = ufloat(-2.683, 0.051)
E = m * mu + d
#print(E)

e = ufloat(-0.742, 0.221)
n = ufloat(-0.934, 0.035)
Q = unp.exp(e) * E ** n

Z = a * np.sqrt(-b * np.pi)

W = np.array([0.1, 4, 4, 4, 19, 36, 47])
A = 4 * Z / (0.8906 * Q * W * 4797)

for i in range(7):
    print(W[i], Z[i], Q[i], A[i])

    #(E[i], W[i], mu[i], a[i], b[i], c[i], Z[i], Q[i], A[i])