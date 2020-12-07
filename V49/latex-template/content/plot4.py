import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

t, U = np.genfromtxt('mess4.txt', unpack=True) 

T2 = 4*10**3
g = 2.67*10**8
G2 = 0.079 

def fit(t, a, b, c):
    return a*np.exp(-(2*t)/T2)*np.exp(-t**3/b)+c

#p0 = [1, 0, 0]

params, cov = curve_fit(fit, t, U)
err = np.sqrt(np.diag(abs(cov)))

a = ufloat(params[0], err[0])
b = ufloat(params[1], err[1])
c = ufloat(params[2], err[2])

print(a, b, c)

plt.plot(t, U, 'rx', label=r'Messwerte')
plt.plot(t, fit(t, *params), 'b-', label='Ausgleichsrechnung')
plt.xlabel(r'$\tau \, / \, ms$')
plt.ylabel(r'$U \, / \, V$')
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('plot4.pdf')

D = 3/(2*b*10**-9*g**2*G2**2)
print(D)
