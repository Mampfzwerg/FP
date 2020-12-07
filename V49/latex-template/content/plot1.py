import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

t, U = np.genfromtxt('mess1.txt', unpack=True) 

def fit(t, a, b, c):
    return a*(1-2*np.exp(-t/b))+c

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
plt.xscale('log')
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('plot1.pdf')