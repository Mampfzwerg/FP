import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy import stats , integrate
from scipy.integrate import quad, trapz, simps

T1, I1 = np.genfromtxt('mess1.txt', unpack=True)
T1 = T1 + 273.15
I1 = np.abs(I1)

T2, I2 = np.genfromtxt('mess2.txt', unpack=True)
T2 = T2 + 273.15
I2 = np.abs(I2)

def f(x, t0, W):
    kB = 8.617333262e-5
    return t0 * np.exp(W/(kB*x)) 

#approx

W1a = 0.606
W2a = 0.562

t01a = -0.46e-12 
t02a = -8.00e-12

#int

W1i = 1.563
W2i = 1.617
t01i = 6.64e-32
t02i = 7.55e-32

#Werte

t1a = f(T1, t01a, W1a)
t2a = f(T2, t02a, W2a)
t1a = np.abs(t1a)
t2a = np.abs(t2a)
t1i = f(T1, t01i, W1i)
t2i = f(T2, t02i, W2i)

print(t1a)

plt.plot(T1, t1a, 'x', color='#18c8fc', label='Näherungsmethode, H1')
plt.plot(T1, t1i, '.', color='#18c8fc', label='Integrationsmethode, H1')
plt.plot(T2, t2a, 'x', color='#1891fc', label='Näherungsmethode, H2')
plt.plot(T2, t2i, '.', color='#1891fc', label='Integrationsmethode, H2')

plt.grid(linestyle='dotted', which="both")
plt.yscale('log')
plt.xlabel(r'$T$/K')
plt.ylabel(r'ln($\tau \: /$ s)')
plt.legend(loc='best')
#plt.ylim(0, 1.5)
#plt.ylim(0,50)

plt.tight_layout()
plt.savefig('plot5.pdf')