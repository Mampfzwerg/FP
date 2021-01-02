import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

fig, ax1 = plt.subplots()

phi, I = np.genfromtxt('data/detektorscan1.UXD', unpack=True)
ax1.plot(phi, I, 'x', color='#00BFFF', label='1. Detektorscan')

def gauss(phi, mu, si, a, b):
    return a / np.sqrt(2 * np.pi * si**2)* np.exp(-(phi - mu)**2 / (2 * si**2)) + b

par, cov = optimize.curve_fit(gauss, phi, I, p0=[4e-4, 4e-2, 1e5, 1e4])
mu = ufloat(par[0], np.sqrt(cov[0][0]))
si = ufloat(par[1], np.sqrt(cov[1][1]))
a = ufloat(par[2], np.sqrt(cov[2][2]))
b = ufloat(par[3], np.sqrt(cov[3][3]))
print(mu, si, a, b)

phiz = np.linspace(np.min(phi), np.max(phi), 1000)
ax1.plot(phiz, gauss(phiz, *par), '-', color='#2E64FE', label='Gaußverteilung')

#Halbwertsbreite
Imax = np.max(gauss(phiz, *par)) 
Ix = np.zeros(200) + 0.5 * Imax
phix = np.linspace(unp.nominal_values(mu)-0.051, unp.nominal_values(mu)+0.051, 200)
ax1.plot(phix, Ix, '-', color='#A901DB', label='Halbwertsbreite')

print(Imax, np.max(I))

ax2 = ax1.twinx()
ax2.grid(linestyle='dotted', which="both")

ax1.set_xlim(np.min(phi), np.max(phi))
ax1.set_xlabel(r'Winkel $\alpha_i$ (°)')
ax1.set_ylabel(r'Intensität')
ax1.set_ylim(-5e4, Imax+5e4)
ax2.set_ylabel(r'Reflektivität')
ax2.set_ylim(-5e4/Imax, (Imax+5e4)/Imax)
ax1.legend(loc='center right')

plt.tight_layout()
plt.savefig('plot1.pdf')
plt.close()
###################################################################################################################

fig, ax1 = plt.subplots()

phi, I = np.genfromtxt('data/zscan1.UXD', unpack=True)
ax1.plot(phi, I, 'x', color='#00BFFF', label='1. Z-Scan')

Imax = np.max(I)
print(Imax)
Ix = np.zeros(200) + 0.5 * Imax
phix = np.linspace(phi[16], phi[23], 200)
print(phi[23] - phi[16])
ax1.plot(phix, Ix, '-', color='#A901DB', label='Strahlbreite')


ax2 = ax1.twinx()
ax2.grid(linestyle='dotted', which="both")

ax1.set_xlim(np.min(phi), np.max(phi))
ax1.set_xlabel(r'Höhe $z$ (mm)')
ax1.set_ylabel(r'Intensität')
ax1.set_ylim(-5e4, Imax+5e4)
ax2.set_ylabel(r'Reflektivität')
ax2.set_ylim(-5e4/Imax, (Imax+5e4)/Imax)
ax1.legend(loc='center right')

plt.tight_layout()
plt.savefig('plot2.pdf')
plt.close()
##################################################################################################################

fig, ax1 = plt.subplots()

phi, I = np.genfromtxt('data/rockingscan1.UXD', unpack=True)
ax1.plot(phi, I, 'x', color='#00BFFF', label='1. Rockingscan')

ax1.plot(phi[4], I[4], 'x', color='#A901DB', label='Geometriewinkel')
ax1.plot(phi[41], I[41], 'x', color='#A901DB')
print(phi[4], phi[41])

ImaxG = 1636650
Imax = np.max(I)

ax2 = ax1.twinx()
ax2.grid(linestyle='dotted', which="both")

ax1.set_xlim(np.min(phi), np.max(phi))
ax1.set_xlabel(r'Winkel $\alpha_i$ (°)')
ax1.set_ylabel(r'Intensität')
ax1.set_ylim(-5e4, Imax+5e4)
ax2.set_ylabel(r'Reflektivität')
ax2.set_ylim(-5e4/ImaxG, (Imax+5e4)/ImaxG)
ax1.legend(loc='center right')

plt.tight_layout()
plt.savefig('plot3.pdf')
plt.close()
##################################################################################################################

