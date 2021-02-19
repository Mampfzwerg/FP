import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

def gg(L, r1, r2):
    fkt = 1 - L/r1 - L/r2 + L**2/(r1 * r2)
    return fkt

def gg2(L, r):
    fkt = 1 - L/r
    return fkt

L = np.linspace(0, 10, 500)

plt.plot(L, 0*L + 1, linewidth=1.4, color='#F72209', label='plan + plan')
#plt.plot(L, gg(L, 1, 1), linewidth=1.4, color='#FB9D02', label='konkav 1')
plt.plot(L, gg(L, 1.4, 1.4), linewidth=1.4, color='#4DD30A', label='konkav + konkav')
#plt.plot(L, gg(L, 1, 1.4), linewidth=1.4, color='#AD0EBD', label='konkav 3')
#plt.plot(L, gg2(L, 1), linewidth=1.4, color='#045DF9', label='kombi 1')
plt.plot(L, gg2(L, 1.4), linewidth=1.4, color='#045DF9', label='plan + konkav')

plt.xlabel(r'$L$ (m)')
plt.ylabel(r'$g_1 g_2$')
plt.grid(linestyle='dashed', color='#7A7980')
plt.xlim(0, 3)
plt.ylim(-0.2, 1.2)
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('g1g2.pdf')
