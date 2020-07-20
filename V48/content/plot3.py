import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy import stats , integrate
from scipy.integrate import quad, trapz, simps

T, I = np.genfromtxt('mess1.txt', unpack=True)
T = T + 273.15
I = np.abs(I)

Tb,Ib = np.ones(7), np.ones(7)
Tb[0:6], Ib[0:6] = T[1:7], I[1:7]
Tb[6], Ib[6] = T[42], I[42]


def f(x, c, d):
    return c * np.exp(d*x)

params, cov = curve_fit(f, Tb, Ib, p0=[1e-3, 1e-3])
err = np.sqrt(np.diag(cov))

# Parameter
c = ufloat(params[0], err[0])
d = ufloat(params[1], err[1])
print(c, d)

z = np.linspace(T[0], T[45], 500)

plt.plot(T, I, 'x', color='#18c8fc', label='Messwerte')
plt.plot(Tb, Ib, '.', color='b', label='Untergrund-Stützwerte')
plt.plot(z, f(z, *params), 'b-', label='Untergrund')


#bereinigt
Tx = T[16:43]
Ix = I[16:43]
Ix = Ix - f(Tx, 1.63e-6, 0.0462)
Ix = np.abs(Ix)

diff = np.zeros(T.size - 1)
for i in range(T.size - 1):
    diff[i] = abs(T[i+1] - T[i])
    
H = ufloat(np.mean(diff), np.std(diff))
A = -integrate.simps(Ix,Tx) / (Ix * unp.nominal_values(H))
#A = np.abs(A)

def g(T, e, f):
    return e * np.exp(f/T)

par, cov2 = curve_fit(g, Tx, A)#, p0=[1e2, 1e2])
err2 = np.sqrt(np.diag(cov2))

e = ufloat(par[0], err2[0])
f = ufloat(par[1], err2[1])
print(e, f)
#f =np.abs(f)

kB = 8.617333262e-5
W = -f * kB
print(W)

Tmax = [T[27], T[28]]
Tmax = ufloat(np.mean(Tmax), np.std(Tmax))
#print(Tmax)

tau = H * f / (60 * Tmax)
tau0 = tau * unp.exp(-W/(kB * Tmax))
print(tau, tau0)

def f(x, c, d):
    return c * np.exp(d*x)

z = np.linspace(T[0], T[45], 500)
plt.plot(Tx, Ix, 'x', color='#1891fc', label=' Bereinigte Messwerte')
plt.fill_between(T[16:43], I[16:43], f(T[16:43], 1.63e-6, 0.0462), color="b", alpha=0.3)
#plt.plot(Tx[17:21], np.abs(A[17:21]), 'rx', label='Fitwerte')
#plt.plot(z, g(z, *par), 'r-', label='Fit')

plt.grid(linestyle='dotted', which="both")
plt.xlabel(r'$T$/K')
plt.ylabel(r'$I$/pA')
plt.legend(loc='best')
plt.ylim(0, 1.5)
#plt.ylim(0,50)

plt.tight_layout()
plt.savefig('plot3.pdf')
