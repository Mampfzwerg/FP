
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

y = np.genfromtxt('mess1.txt', unpack=True)
x = np.arange(y.size)
plt.plot(x, y, 'b-', label='152Eu')

def gauss(x, m, a, b, c):
    return a * np.exp(-(x-m)**2 / b) + c


x1 = np.arange(200, 400)
y1 = y[200:400]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[320, 10000, 100, 1])

m = ufloat(par[0], np.sqrt(cov[0][0]))
a = ufloat(par[1], np.sqrt(cov[1][1]))
b = ufloat(par[2], np.sqrt(cov[2][2]))
c = ufloat(par[3], np.sqrt(cov[3][3]))
print(m, a, b, c)
plt.plot(x1, gauss(x1, *par), 'r-', label='Gaussian Fits')



x1 = np.arange(550, 670)
y1 = y[550:670]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[610, 10000, 100, 1])

m = ufloat(par[0], np.sqrt(cov[0][0]))
a = ufloat(par[1], np.sqrt(cov[1][1]))
b = ufloat(par[2], np.sqrt(cov[2][2]))
c = ufloat(par[3], np.sqrt(cov[3][3]))
print(m, a, b, c)
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(750, 950)
y1 = y[750:950]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[850, 10000, 100, 1])

m = ufloat(par[0], np.sqrt(cov[0][0]))
a = ufloat(par[1], np.sqrt(cov[1][1]))
b = ufloat(par[2], np.sqrt(cov[2][2]))
c = ufloat(par[3], np.sqrt(cov[3][3]))
print(m, a, b, c)
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(1000, 1050)
y1 = y[1000:1050]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[1025, 10000, 100, 1])

m = ufloat(par[0], np.sqrt(cov[0][0]))
a = ufloat(par[1], np.sqrt(cov[1][1]))
b = ufloat(par[2], np.sqrt(cov[2][2]))
c = ufloat(par[3], np.sqrt(cov[3][3]))
print(m, a, b, c)
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(1050, 1150)
y1 = y[1050:1150]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[1100, 10000, 100, 1])

m = ufloat(par[0], np.sqrt(cov[0][0]))
a = ufloat(par[1], np.sqrt(cov[1][1]))
b = ufloat(par[2], np.sqrt(cov[2][2]))
c = ufloat(par[3], np.sqrt(cov[3][3]))
print(m, a, b, c)
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(1850, 2050)
y1 = y[1850:2050]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[1950, 10000, 100, 1])

m = ufloat(par[0], np.sqrt(cov[0][0]))
a = ufloat(par[1], np.sqrt(cov[1][1]))
b = ufloat(par[2], np.sqrt(cov[2][2]))
c = ufloat(par[3], np.sqrt(cov[3][3]))
print(m, a, b, c)
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(2100, 2200)
y1 = y[2100:2200]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[2150, 10000, 100, 1])

m = ufloat(par[0], np.sqrt(cov[0][0]))
a = ufloat(par[1], np.sqrt(cov[1][1]))
b = ufloat(par[2], np.sqrt(cov[2][2]))
c = ufloat(par[3], np.sqrt(cov[3][3]))
print(m, a, b, c)
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(2350, 2450)
y1 = y[2350:2450]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[2400, 10000, 100, 1])

m = ufloat(par[0], np.sqrt(cov[0][0]))
a = ufloat(par[1], np.sqrt(cov[1][1]))
b = ufloat(par[2], np.sqrt(cov[2][2]))
c = ufloat(par[3], np.sqrt(cov[3][3]))
print(m, a, b, c)
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(2650, 2750)
y1 = y[2650:2750]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[2700, 10000, 100, 1])

m = ufloat(par[0], np.sqrt(cov[0][0]))
a = ufloat(par[1], np.sqrt(cov[1][1]))
b = ufloat(par[2], np.sqrt(cov[2][2]))
c = ufloat(par[3], np.sqrt(cov[3][3]))
print(m, a, b, c)
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(2750, 2800)
y1 = y[2750:2800]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[2775, 10000, 100, 1])

m = ufloat(par[0], np.sqrt(cov[0][0]))
a = ufloat(par[1], np.sqrt(cov[1][1]))
b = ufloat(par[2], np.sqrt(cov[2][2]))
c = ufloat(par[3], np.sqrt(cov[3][3]))
print(m, a, b, c)
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(3450, 3550)
y1 = y[3450:3550]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[3500, 10000, 100, 1])

m = ufloat(par[0], np.sqrt(cov[0][0]))
a = ufloat(par[1], np.sqrt(cov[1][1]))
b = ufloat(par[2], np.sqrt(cov[2][2]))
c = ufloat(par[3], np.sqrt(cov[3][3]))
print(m, a, b, c)
plt.plot(x1, gauss(x1, *par), 'r-')


plt.xlim(0, 4000)
plt.xlabel(r'Channel')
plt.ylabel(r'Counts')
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('plot1.pdf')