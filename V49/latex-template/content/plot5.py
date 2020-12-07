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
U2 = np.log(U)-(2*t)/(T2)

params, covariance_matrix = np.polyfit(t**3, U2, deg=1, cov=True)
errors = np.sqrt(np.diag(covariance_matrix))

print('a = {:.8f} ± {:.8f}'.format(params[0], errors[0]))
print('b = {:.8f} ± {:.8f}'.format(params[1], errors[1]))

def gerade (x, m, b):
    return m*x+b

z = np.linspace(0, 8000)

plt.plot(t**3, U2, 'rx', label=r'Messwerte')
plt.plot(z, gerade (z, *params), 'b-', label='Lineare Regression')
plt.xlabel(r'$\tau^3\, / \, ms^3$')
plt.ylabel(r'$ln(U(\tau))-\frac{2\tau}{T_2}$')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('plot5.pdf')

D = 3/(2*b*10**-9*g**2*G2**2)
print(D)

kb = 1.38*10**-23
Te = 295.35
n = 890.2*10**-6

r = (kb*Te)/(6*np.pi*n*D)

print(r)