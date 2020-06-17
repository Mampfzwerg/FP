import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

y = np.genfromtxt('mess3.txt', unpack=True)
x = np.arange(y.size)
plt.plot(x, y, 'b-', label='125Sb or 133Ba')

def gauss(x, m, a, b, c):
    return a * np.exp(-(x-m)**2 / b) + c


x1 = np.arange(175, 225)
y1 = y[175:225]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[200, 10000, 100, 1])

m = ufloat(par[0], np.sqrt(cov[0][0]))
a = ufloat(par[1], np.sqrt(cov[1][1]))
b = ufloat(par[2], np.sqrt(cov[2][2]))
c = ufloat(par[3], np.sqrt(cov[3][3]))
print(m, a, b, c)
plt.plot(x1, gauss(x1, *par), 'r-', label='Gaussian Fits')



x1 = np.arange(675, 725)
y1 = y[675:725]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[700, 10000, 100, 1])

m = ufloat(par[0], np.sqrt(cov[0][0]))
a = ufloat(par[1], np.sqrt(cov[1][1]))
b = ufloat(par[2], np.sqrt(cov[2][2]))
c = ufloat(par[3], np.sqrt(cov[3][3]))
print(m, a, b, c)
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(725, 775)
y1 = y[725:775]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[750, 10000, 100, 1])

m = ufloat(par[0], np.sqrt(cov[0][0]))
a = ufloat(par[1], np.sqrt(cov[1][1]))
b = ufloat(par[2], np.sqrt(cov[2][2]))
c = ufloat(par[3], np.sqrt(cov[3][3]))
print(m, a, b, c)
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(875, 925)
y1 = y[875:925]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[900, 10000, 100, 1])

m = ufloat(par[0], np.sqrt(cov[0][0]))
a = ufloat(par[1], np.sqrt(cov[1][1]))
b = ufloat(par[2], np.sqrt(cov[2][2]))
c = ufloat(par[3], np.sqrt(cov[3][3]))
print(m, a, b, c)
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(925, 975)
y1 = y[925:975]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[950, 10000, 100, 1])

m = ufloat(par[0], np.sqrt(cov[0][0]))
a = ufloat(par[1], np.sqrt(cov[1][1]))
b = ufloat(par[2], np.sqrt(cov[2][2]))
c = ufloat(par[3], np.sqrt(cov[3][3]))
print(m, a, b, c)
plt.plot(x1, gauss(x1, *par), 'r-')


plt.xlim(0, 1200)
plt.xlabel(r'Channel')
plt.ylabel(r'Counts')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('plot3.pdf')