import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

T, I = np.genfromtxt('mess1.txt', unpack=True)
T = T + 273.15
I = np.abs(I)

Tb,Ib = np.ones(7), np.ones(7)
Tb[0:6], Ib[0:6] = T[1:7], I[1:7]
Tb[6], Ib[6] = T[42], I[42]

def b(y, c, d):
    return c * np.exp(d*y)

params, cov = curve_fit(b, Tb, Ib, p0=[1e-3, 1e-3])
err = np.sqrt(np.diag(cov))

z = np.linspace(T[0], T[45], 500)

plt.plot(T, I, 'x', color='#18c8fc', label='Messwerte')
plt.plot(Tb, Ib, '.', color='b', label='Untergrund-Stützwerte')
plt.plot(z, b(z, *params), 'b-', label='Untergrund')

Tx = T[20:33]
Ix = I[20:33] - b(T[20:33], *params) 
print(T[20], T[33])

def f(x, a, b):
    return np.exp(a/x) * b

params, cov = curve_fit(f, Tx, Ix) #, p0=[-6000, 1])
err = np.sqrt(np.diag(cov))

# Parameter
a = ufloat(params[0], err[0])
b = ufloat(params[1], err[1])
print(a, b)

kB = 8.617333262e-5
W = -a * kB
print(W)

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


z = np.linspace(np.min(Tx), np.max(Tx), 500)

plt.plot(Tx, Ix, 'x', color='#1891fc', label='Bereinigte und approximierte Messwerte')
plt.plot(z, f(z, *params), '#1891fc', label='Exponentielle Approximation')
#plt.ylim(1e-2, 10)
#plt.yscale('log')
plt.grid(linestyle='dotted', which="both")
plt.xlabel(r'$T$/K')
plt.ylabel(r'$I$/pA')
plt.legend(loc='lower right')
plt.ylim(0, 1.5)

plt.tight_layout()
plt.savefig('plot1.pdf')