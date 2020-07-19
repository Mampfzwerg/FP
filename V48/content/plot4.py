import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy import stats , integrate
from scipy.integrate import quad, trapz, simps

T, I = np.genfromtxt('mess2.txt', unpack=True)
T = T + 273.15
I = np.abs(I)

Tb,Ib = np.ones(7), np.ones(7)
Tb[0:6], Ib[0:6] = T[1:7], I[1:7]
Tb[6], Ib[6] = T[43], I[43]


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
Tx = T[16:44]
Ix = I[16:44]
Ix = Ix - f(Tx, 6e-6, 0.0411)
Ix = np.abs(Ix)

diff = np.zeros(T.size - 1)
for i in range(T.size - 1):
    diff[i] = abs(T[i+1] - T[i])
    
H = ufloat(np.mean(diff), np.std(diff))
A = -integrate.simps(Ix,Tx) / (Ix * unp.nominal_values(H))
#A = np.abs(A)

def g(T, e, f):
    return e * np.exp(f/T)

par, cov2 = curve_fit(g, Tx, A)
err2 = np.sqrt(np.diag(cov2))

e = ufloat(par[0], err2[0])
f = ufloat(par[1], err2[1])
print(e, f)
#f =np.abs(f)

kB = 8.617333262e-5
W = -f * kB
print(W)

Tmax = [T[33], T[34]]
Tmax = ufloat(np.mean(Tmax), np.std(Tmax))
#print(Tmax)

tau = H * f / (60 * Tmax)
tau0 = tau * unp.exp(-W/(kB * Tmax))
print(tau, tau0)

def f(x, c, d):
    return c * np.exp(d*x)

z = np.linspace(T[0], T[45], 500)
plt.plot(Tx, Ix, 'x', color='#1891fc', label=' Bereinigte Messwerte')
plt.fill_between(T[16:44], I[16:44], f(T[16:44], 6e-6, 0.0411), color="b", alpha=0.3)
#plt.plot(Tx, np.abs(A), 'rx', label='Fitwerte')
#plt.plot(z, g(z, *par), 'r-', label='Fit')

plt.grid(linestyle='dotted', which="both")
plt.xlabel(r'$T$/K')
plt.ylabel(r'$I$/pA')
plt.legend(loc='best')
plt.ylim(0, 1.5)

plt.tight_layout()
plt.savefig('plot4.pdf')