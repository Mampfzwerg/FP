import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

T, I = np.genfromtxt('mess1.txt', unpack=True)
T = T + 273.15
I = np.abs(I)

Tx = T[9:30]
Ix = I[9:30]
#print(T[9], T[30])

def f(x, a, b):
    return np.exp(a/x) * b

params, cov = curve_fit(f, Tx, Ix) #, p0=[-6000, 1])
err = np.sqrt(np.diag(cov))

# Parameter
a = ufloat(params[0], err[0])
b = ufloat(params[1], err[1])
#print(a, b)

kB = 8.617333262e-5
W = -a * kB
#print(W)

Tmax = [T[27], T[28]]
Tmax = ufloat(np.mean(Tmax), np.std(Tmax))
#print(Tmax)

diff = np.zeros(T.size - 1)
for i in range(T.size - 1):
    diff[i] = abs(T[i+1] - T[i])
    
H = ufloat(np.mean(diff), np.std(diff))
tau = H * a / (60 * Tmax)
tau0 = tau * unp.exp(-W/(kB * Tmax))
print(tau, tau0)


z = np.linspace(T[9], T[30], 500)

plt.plot(T, I, 'x', color='#18c8fc', label='Messwerte')
plt.plot(Tx, Ix, '.', color='#1891fc', label=' Approximierte Messwerte')
plt.plot(z, f(z, *params), 'b-', label='Exponentielle Approximation')
plt.ylim(1e-2, 10)
plt.yscale('log')
plt.grid(linestyle='dotted', which="both")
plt.xlabel(r'$T$/K')
plt.ylabel(r'ln($I \: /$ pA)')
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('plot1.pdf')