import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

T, I = np.genfromtxt('mess1.txt', unpack=True)
T = T + 273.15
ln = np.log(abs(I))

Tx = T[9:30]
Ix = I[9:30]

def f(x, a, b):
    return np.exp(a/x) * b

params, cov = curve_fit(f, Tx, Ix, p0=[-6000, 1])
err = np.sqrt(np.diag(cov))

# Parameter
a = ufloat(params[0], err[0])
b = ufloat(params[1], err[1])
print(a, b)

z = np.linspace(T[9], T[30], 500)


plt.plot(z, f(z, *params), 'b-', label='Exponentielle Approximation')
plt.plot(T, ln, 'cx', label='Messwerte')
#plt.xlim(0, 4000)
plt.xlabel(r'$T/K$')
plt.ylabel(r'ln($I$)')
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('plot1.pdf')