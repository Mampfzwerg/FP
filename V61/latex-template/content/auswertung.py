import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp


phi, I = np.genfromtxt('polarisation.txt', unpack=True)
plt.plot(phi, I, 'x', color='#00BFFF', label='Daten')

def cos(phi, I0, phi2):
    return I0 * (np.cos((phi + phi2)*(2*np.pi/360)))**2 

par, cov = optimize.curve_fit(cos, phi, I, p0=[0.9, 90])
I0 = ufloat(par[0], np.sqrt(cov[0][0]))
phi2 = ufloat(par[1], np.sqrt(cov[1][1]))
print(I0, phi2)

phix = np.linspace(np.min(phi), np.max(phi), 1000)
plt.plot(phix, cos(phix, *par), '-', color='#2E64FE', label='Ausgleichsrichtung')

plt.xlim(np.min(phi)-10, np.max(phi)+10)
plt.xlabel(r'Polarisationswinkel $\phi$ (°)')
plt.ylabel(r'Intensität $I(\phi)$ (mW)')
#plt.ylim(-5e4, Imax+5e4)
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('pol.pdf')
plt.close()

###################################################################################################################

r, I = np.genfromtxt('TEM00.txt', unpack=True)
I = I * 1e3
plt.plot(r, I, 'x', color='#00BFFF', label='Daten TEM$_{00}$-Mode')

def exp1(r, I0, r0, w):
    return I0 * np.exp(-(r - r0)**2/(2*w**2)) 

par, cov = optimize.curve_fit(exp1, r, I)#, p0=[0.9, 90])
I0 = ufloat(par[0], np.sqrt(cov[0][0]))
r0 = ufloat(par[1], np.sqrt(cov[1][1]))
w = ufloat(par[2], np.sqrt(cov[2][2]))
print(I0, r0, w)

rx = np.linspace(np.min(r), np.max(r), 1000)
plt.plot(rx, exp1(rx, *par), '-', color='#2E64FE', label='Ausgleichsrichtung')

plt.xlim(np.min(r)-3, np.max(r)+3)
plt.xlabel(r'Abstand $r$ (mm)')
plt.ylabel(r'Intensität $I(r)$ ($\mu$W)')
#plt.ylim(-5e4, Imax+5e4)
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('TEM00.pdf')
plt.close()

###################################################################################################################

r, I = np.genfromtxt('TEM01.txt', unpack=True)
print(np.min(I))
I = I - np.min(I)
plt.plot(r, I, 'x', color='#00BFFF', label='Daten TEM$_{01}$-Mode')

def exp2(r, I0, r0, w):
    return I0 * 8*(r-r0)**2/(w**2) * np.exp(-2*(r - r0)**2/(2*w**2))

par, cov = optimize.curve_fit(exp2, r, I)#, p0=[0.151, 0.5, 8])
I0 = ufloat(par[0], np.sqrt(cov[0][0]))
r0 = ufloat(par[1], np.sqrt(cov[1][1]))
w = ufloat(par[2], np.sqrt(cov[2][2]))

print(I0, r0, w)

rx = np.linspace(np.min(r)-3, np.max(r)+3, 1000)
plt.plot(rx, exp2(rx, *par), '-', color='#2E64FE', label='Ausgleichsrichtung')

plt.xlim(np.min(r)-3, np.max(r)+3)
plt.xlabel(r'Abstand $r$ (mm)')
plt.ylabel(r'Intensität $I(r)$ ($\mu$W)')
#plt.ylim(-5e4, Imax+5e4)
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('TEM01.pdf')
plt.close()

#####################################################################################################

L, I = np.genfromtxt('gg_plan.txt', unpack=True)
L2, I2 = np.genfromtxt('gg_kon.txt', unpack=True)
plt.plot(L, I, 'x', color='#00BFFF', label='plan + konkav')
plt.plot(L2, I2, 'x', color='#2E64FE', label='konkav + konkav')
plt.xlim(np.min(L)-3, np.max(L2)+3)
plt.xlabel(r'Resonatorlänge $L$ (cm)')
plt.ylabel(r'Intensität $I(L)$ (mW)')
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('gg.pdf')
plt.close()

#####################################################################################################

n1, d1 = np.genfromtxt(  '80.txt', unpack=True)
n2, d2 = np.genfromtxt( '100.txt', unpack=True)
n3, d3 = np.genfromtxt( '600.txt', unpack=True)
n4, d4 = np.genfromtxt('1200.txt', unpack=True)

d1 = unp.uarray(d1, 0.1)
d2 = unp.uarray(d2, 0.1)
d3 = unp.uarray(d3, 0.1)
d4 = unp.uarray(d4, 0.1)


l123 = ufloat(76, 2)
l4 = ufloat(25, 2)

g1 = 80   * 1e3
g2 = 100  * 1e3
g3 = 600  * 1e3
g4 = 1200 * 1e3

def lam(n, d, l, g):
    #return d / (g * n * l)
    return unp.sin(unp.tan(d/l)) / (g * n)

lam1 = lam(n1, d1, l123, g1)
lam2 = lam(n2, d2, l123, g2)
lam3 = lam(n3, d3, l123, g3)
lam4 = lam(n4, d4, l4,   g4)

lam = np.concatenate((lam1, lam2, lam3, lam4))
#print(lam)
lam = np.mean(lam)

#print(np.mean(lam1) * 1e9)
#print(np.mean(lam2) * 1e9)
#print(np.mean(lam3) * 1e9)
#print(np.mean(lam4) * 1e9)
print(lam * 1e9)