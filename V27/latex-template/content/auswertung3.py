import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp
from numpy.linalg import inv

x = np.genfromtxt('mess4.txt', unpack=True)

s = np.array([x[1]-x[0], x[2]-x[1], x[3]-x[2], x[4]-x[3], x[5]-x[4], x[6]-x[5], x[7]-x[6]])
ms = ufloat(np.mean(s), np.std(s))

print(np.mean(s),np.std(s))

y = np.genfromtxt('mess5.txt', unpack=True)

s2 = np.array([y[1]-y[0], y[3]-y[2], y[5]-y[4], y[7]-y[6], y[9]-y[8], y[11]-y[10], y[13]-y[12], y[15]-y[14]])

ms2 = ufloat(np.mean(s2), np.std(s2))
print(np.mean(s2), np.std(s2))

lb = 26.95 

l = (ms2/ms)*(lb/2)

print(l)
l1 = l*10**-12 
h = 6.626*10**(-34)
c = 2.99*10**8
lr = 480*10**-9
mu = 9.274*10**-24
B = 318*10**-3

print((h*c*l1)/(lr**2*mu*B))