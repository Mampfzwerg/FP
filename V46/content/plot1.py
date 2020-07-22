import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

z, B = np.genfromtxt('mess1.txt', unpack=True) #mm, mT

def Bfit(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

def Bfit2(x, a, b, c, d):
    return 4*a * x**3 + 3*b * x**2 + 2*c * x + d

paramsB, covB = curve_fit(Bfit, z[10:32], B[10:32])
errB = np.sqrt(np.diag(covB))

a = ufloat(paramsB[0], errB[0])
b = ufloat(paramsB[1], errB[1])
c = ufloat(paramsB[2], errB[2])
d = ufloat(paramsB[3], errB[3])
e = ufloat(paramsB[4], errB[4])

print(a, b, c, d, e)

x = np.linspace(z[10], z[32], 500)

print(Bfit(1.25, *paramsB))

plt.plot(z, B, 'x', color='#1891fc', label='Messwerte')
plt.plot(x, Bfit(x, *paramsB), 'b-', label='Approximation')
plt.plot(x, Bfit2(x, paramsB[0], paramsB[1], paramsB[2], paramsB[3]), '-', color='#18c8fc', label='1. Abl. der Approximation')
plt.grid(linestyle='dotted', which="both")
plt.xlabel(r'$z$/mm')
plt.ylabel(r'$B$/mT')
#plt.xlim(0, 2)
#plt.ylim(-5, 5)
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('plot1.pdf')