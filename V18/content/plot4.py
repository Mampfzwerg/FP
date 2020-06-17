import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

y = np.genfromtxt('mess4.txt', unpack=True)
x = np.arange(y.size)
plt.plot(x, y, 'b-', label='Unknown radiation source')

def gauss(x, m, a, b, c):
    return a * np.exp(-(x-m)**2 / b) + c


x1 = np.arange(196, 200)
y1 = y[196:200]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[198, 10000, 100, 1])

m = par[0]
a = par[1]
b = par[2]
c = par[3]
print(m, a, b, c)
plt.plot(x1, gauss(x1, *par), 'r-', label='Gaussian Fits')



x1 = np.arange(234, 238)
y1 = y[234:238]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[236, 10000, 100, 1])

m = par[0]
a = par[1]
b = par[2]
c = par[3]
print(m, a, b, c)
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(465, 471)
y1 = y[465:471]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[468, 10000, 100, 1])

m = par[0]
a = par[1]
b = par[2]
c = par[3]
print(m, a, b, c)
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(604, 611)
y1 = y[604:611]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[607, 10000, 100, 1])

m = par[0]
a = par[1]
b = par[2]
c = par[3]
print(m, a, b, c)
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(736, 743)
y1 = y[736:743]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[739, 10000, 100, 1])

m = par[0]
a = par[1]
b = par[2]
c = par[3]
print(m, a, b, c)
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(875, 885)
y1 = y[875:885]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[880, 10000, 100, 1])

m = par[0]
a = par[1]
b = par[2]
c = par[3]
print(m, a, b, c)
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(1513, 1524)
y1 = y[1513:1524]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[1518, 10000, 100, 1])

m = par[0]
a = par[1]
b = par[2]
c = par[3]
print(m, a, b, c)
plt.plot(x1, gauss(x1, *par), 'r-')

plt.xlim(0, 1700)
plt.xlabel(r'Channel')
plt.ylabel(r'Counts')
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('plot4.pdf')