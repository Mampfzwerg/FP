import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp

f, Ih1, Is1, Ih2, Is2 = np.genfromtxt('mess.txt', unpack=True)

#Konstanten
mu0 = 1.2566*10**-6

#Radien der Spulen
Rh = 0.1579
Rs = 0.1639
Rv = 0.11735

#Windungszahlen der Spulen
Nh = 154
Ns = 11
Nv = 20

#Korrektur des vertikalen Erdmagnetfeldes

Iv = 0.229
Bv = mu0 * (8*Iv*Nv)/(np.sqrt(125)*Rv)
print('Korrektur des vertikalen Erdmagnetfeldes: ',Bv)
Bh1 = mu0 * (8*Ih1*Ns)/(np.sqrt(125)*Rs)
Bh2 = mu0 * (8*Ih2*Ns)/(np.sqrt(125)*Rs)
Bs1 = mu0 * (8*Is1*Nh)/(np.sqrt(125)*Rh) + Bh1
Bs2 = mu0 * (8*Is2*Nh)/(np.sqrt(125)*Rh) + Bh2
print('Horizontales Feld 1: ',Bs1)
print('Horizontales Feld 2: ',Bs2)

params, covariance_matrix = np.polyfit(f, Bs2, deg=1, cov=True)

errors = np.sqrt(np.diag(covariance_matrix))

print('a = {:.8f} ± {:.8f}'.format(params[0], errors[0]))
print('b = {:.8f} ± {:.8f}'.format(params[1], errors[1]))

def gerade (x, m, b):
    return m*x+b

z = np.linspace(0, 1000)

plt.plot(f, Bs2, 'rx', label = 'Messwerte')
plt.plot(z, gerade (z, *params), 'b-', label='Lineare Regression')
plt.xlabel(r'$f$')
plt.xlabel(r'$B$')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('plot2.pdf')