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



#bereinigt
Tx = T[16:43]
Ix = I[16:43]
Ix = Ix - f(Tx, 1.63e-6, 0.0462)
Ix = np.abs(Ix)

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

diff = np.zeros(T.size - 1)
for i in range(T.size - 1):
    diff[i] = abs(T[i+1] - T[i])
    
H = ufloat(np.mean(diff), np.std(diff))
A = -integrate.simps(Ix,Tx) / (Ix * unp.nominal_values(H))
A = np.log(np.abs(A))


#kB = 8.617333262e-5
#W = -f * kB
#print(W)
#
#Tmax = [T[27], T[28]]
#Tmax = ufloat(np.mean(Tmax), np.std(Tmax))
##print(Tmax)
#
#tau = H * f / (60 * Tmax)
#tau0 = tau * unp.exp(-W/(kB * Tmax))
#print(tau, tau0)

def h(T, a, b):
    return a * T + b

pa, co = np.polyfit(1/Tx[0:17], np.abs(A[0:17]), deg=1, cov=True)
err3 = np.sqrt(np.diag(co))

print('a = {:.3f} ± {:.4f}'.format(pa[0], err3[0]))
print('b = {:.3f} ± {:.4f}'.format(pa[1], err3[1]))


z = np.linspace(1/Tx[17], 1/Tx[0], 500)

plt.plot(1/Tx[0:17], np.abs(A[0:17]), 'rx', label='Integration')
#plt.plot(z, h(z, 5045736.925, -20223.508), 'r-', label='Fit')

plt.grid(linestyle='dotted', which="both")
plt.xlabel(r'$T$/K')
plt.ylabel(r'A')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('plot6.pdf')


