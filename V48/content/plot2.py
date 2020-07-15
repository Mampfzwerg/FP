import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

T, I = np.genfromtxt('mess2.txt', unpack=True)
ln = np.log(abs(I))


plt.plot(T, ln, 'cx', label='Messwerte')
#plt.xlim(0, 4000)
plt.xlabel(r'$T/K$')
plt.ylabel(r'ln($I$)')
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('plot2.pdf')