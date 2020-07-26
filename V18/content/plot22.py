import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy.integrate import quad

y = np.genfromtxt('mess2.txt', unpack=True)
x = np.arange(y.size)
plt.plot(x, y, 'bx', label='137Cs')

def gauss(x, m, a, b, c):
    return a * np.exp(-(x-m)**2 / b) + c


x1 = np.arange(1640, 1655)
y1 = y[1640:1655]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[1650, 10000, 100, 1])

m = ufloat(par[0], np.sqrt(cov[0][0]))
a = ufloat(par[1], np.sqrt(cov[1][1]))
b = ufloat(par[2], np.sqrt(cov[2][2]))
c = ufloat(par[3], np.sqrt(cov[3][3]))
#print(m, a, b, c)

m = par[0]
a = par[1]
b = par[2]
c = par[3]
I = quad(gauss, 1640, 1655, args=(m,a,b,c))
print(I)

z = np.linspace(1640, 1655, 200)

plt.plot(z, gauss(z, *par), 'r-', label='Gaussian Fit')
plt.xlim(1635, 1660)
plt.xlabel(r'Channel')
plt.ylabel(r'Counts')
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('plot22.pdf')

#print(gauss(1647.72, *par))