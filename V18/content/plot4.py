import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

y = np.genfromtxt('mess4.txt', unpack=True)
x = np.arange(y.size)
plt.plot(x, y, 'b-', label='Unknown radiation source')

m = ufloat(0.403, 0)
d = ufloat(-2.683, 0.051)

def E(x):
    return ufloat(0.403, 0) * x + ufloat(-2.683, 0.051)

def gauss(x, m, a, b, c):
    return a * np.exp(-(x-m)**2 / b) + c


x1 = np.arange(196, 200)
y1 = y[196:200]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[198, 10000, 100, 1])

m = par[0]
a = par[1]
b = par[2]
c = par[3]
print(m, E(m))
plt.plot(x1, gauss(x1, *par), 'r-', label='Gaussian Fits')



x1 = np.arange(234, 238)
y1 = y[234:238]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[236, 10000, 100, 1])

m = par[0]
a = par[1]
b = par[2]
c = par[3]
print(m, E(m))
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(465, 471)
y1 = y[465:471]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[468, 10000, 100, 1])

m = par[0]
a = par[1]
b = par[2]
c = par[3]
print(m, E(m))
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(604, 611)
y1 = y[604:611]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[607, 10000, 100, 1])

m = par[0]
a = par[1]
b = par[2]
c = par[3]
print(m, E(m))
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(736, 743)
y1 = y[736:743]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[739, 10000, 100, 1])

m = par[0]
a = par[1]
b = par[2]
c = par[3]
print(m, E(m))
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(875, 885)
y1 = y[875:885]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[880, 10000, 100, 1])

m = par[0]
a = par[1]
b = par[2]
c = par[3]
print(m, E(m))
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(1513, 1524)
y1 = y[1513:1524]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[1518, 10000, 100, 1])

m = par[0]
a = par[1]
b = par[2]
c = par[3]
print(m, E(m))
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(1650, 1670)
y1 = y[1650:1670]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[1660, 10000, 100, 1])

m = par[0]
a = par[1]
b = par[2]
c = par[3]
print(m, E(m))
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(1900, 1925)
y1 = y[1900:1925]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[1910, 10000, 100, 1])

m = par[0]
a = par[1]
b = par[2]
c = par[3]
print(m, E(m))
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(2310, 2340)
y1 = y[2310:2340]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[2325, 10000, 100, 1])

m = par[0]
a = par[1]
b = par[2]
c = par[3]
print(m, E(m))
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(2765, 2795)
y1 = y[2765:2795]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[2780, 10000, 100, 1])

m = par[0]
a = par[1]
b = par[2]
c = par[3]
print(m, E(m))
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(3070, 3100)
y1 = y[3070:3100]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[3085, 10000, 100, 1])

m = par[0]
a = par[1]
b = par[2]
c = par[3]
print(m, E(m))
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(3405, 3435)
y1 = y[3405:3435]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[3420, 10000, 100, 1])

m = par[0]
a = par[1]
b = par[2]
c = par[3]
print(m, E(m))
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(4370, 4400)
y1 = y[4370:4400]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[4385, 10000, 100, 1])

m = par[0]
a = par[1]
b = par[2]
c = par[3]
print(m, E(m))
plt.plot(x1, gauss(x1, *par), 'r-')



x1 = np.arange(5460, 5490)
y1 = y[5460:5490]
par, cov = optimize.curve_fit(gauss, x1, y1, p0=[5475, 10000, 100, 1])

m = par[0]
a = par[1]
b = par[2]
c = par[3]
print(m, E(m))
plt.plot(x1, gauss(x1, *par), 'r-')


plt.xlim(0, 6000)
#plt.ylim(0, 200)
plt.xlabel(r'Channel')
plt.ylabel(r'Counts')
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('plot4.pdf')