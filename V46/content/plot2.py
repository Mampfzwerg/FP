import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

l1, t11, t12 = np.genfromtxt('mess2.txt', unpack=True) #microm, rad, rad
l2, t21, t22 = np.genfromtxt('mess3.txt', unpack=True) #microm, rad, rad
l3, t31, t32 = np.genfromtxt('mess4.txt', unpack=True) #microm, rad, rad

d1 = 5.11e-3
d2 = 1.296e-3
d3 = 1.36e-3

t1 = np.abs(t11 - t12) * 0.0174533
t2 = np.abs(t21 - t22) * 0.0174533
t3 = np.abs(t31 - t32) * 0.0174533

t1d = t1 / d1
t2d = t2 / d2
t3d = t3 / d3

#print(t1, t1d, t2, t2d, t3, t3d)

plt.plot(l1**2, t1d, 'x', color='#01DF01', label=r'Hochreines GaAs')
plt.plot(l2**2, t2d, 'x', color='r', label=r'n-dotiertes GaAs mit $N = 2,8 \cdot 10^{18} \frac{1}{cm^3}$')
plt.plot(l3**2, t3d, 'x', color='b', label=r'n-dotiertes GaAs mit $N = 1,2 \cdot 10^{18} \frac{1}{cm^3}$')
plt.grid(linestyle='dotted', which="both")
plt.xlabel(r'$\lambda^2$/$\mu$m$^2$')
plt.ylabel(r'$\frac{\Theta}{d}$/radm$^{-1}$')
#plt.xlim(0, 2)
#plt.ylim(-5, 5)
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('plot2.pdf')
plt.close()

######################################################################################

diff1 = t2d - t1d
#diff1, lx = np.zeros(8), np.zeros(8)
#diff1[0:4], diff1[4:] = diff0[0:4], diff0[5:]
#lx[0:4], lx[4:] = l2[0:4], l2[5:]
diff2 = t3d - t1d
#print(l2**2, lx**2)

z = np.linspace(np.min(l1**2), np.max(l1**2), 500)

plt.plot(l2**2, diff1, 'x', color='r', label=r'n-dotiertes GaAs mit $N = 2,8 \cdot 10^{18} \frac{1}{cm^3}$')
plt.plot(l3**2, diff2, 'x', color='b', label=r'n-dotiertes GaAs mit $N = 1,2 \cdot 10^{18} \frac{1}{cm^3}$')

def gerade(x, m):
    return m*x + 0

params1, cov1 = curve_fit(gerade, l2**2, diff1)
errors1 = np.sqrt(np.diag(cov1))

print('a1 = {:.3f} ± {:.4f}'.format(params1[0], errors1[0]))
#print('b1 = {:.3f} ± {:.4f}'.format(params1[1], errors1[1]))

plt.plot(z, gerade(z, params1[0]), 'r-', label='Fit $f(\lambda^2) = a_1 \lambda^2$')

params2, cov2 = curve_fit(gerade, l3**2, diff2)
errors2 = np.sqrt(np.diag(cov2))

print('a2 = {:.3f} ± {:.4f}'.format(params2[0], errors2[0]))
#print('b2 = {:.3f} ± {:.4f}'.format(params2[1], errors2[1]))

plt.plot(z, gerade(z, params2[0]), 'b-', label='Fit $f(\lambda^2) = a_2 \lambda^2$')
plt.grid(linestyle='dotted', which="both")
plt.xlabel(r'$\lambda^2$/$\mu$m$^2$')
plt.ylabel(r'$\frac{\Theta}{d}$/radm$^{-1}$')
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('plot3.pdf')

a1 = ufloat(params1[0], errors1[0]) *1e6
a2 = ufloat(params2[0], errors2[0]) *1e6

e0 = 1.602e-19
eps = 8.854187817e-12
N1 = 2.8e18
N2 = 1.2e18
c = 299792458
B = 444.84e-3
me = 9.1093837015e-31
n = 3.3543

m1 = unp.sqrt(e0**3 * B * N1 / (8 * np.pi**2 * eps * c**3 * a1 * n)) /me
m2 = unp.sqrt(e0**3 * B * N2 / (8 * np.pi**2 * eps * c**3 * a2 * n)) /me
mtheo = 0.067

print(m1, m2)

abw1 = np.abs(m1 - mtheo)/mtheo * 100
abw2 = np.abs(m2 - mtheo)/mtheo * 100

print(unp.nominal_values(abw1), unp.nominal_values(abw2))


