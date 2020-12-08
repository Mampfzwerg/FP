import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

l, B = np.genfromtxt('daten/PLMOKE/2020-07-10 PLMOKE 110819A (2)/int/S110819A_T1.5K_IPS-0.500000T_HS-0.489730T_int.txt', unpack=True)
plt.plot(l, B, 'b-', label=r'$B = -0.5T$')

plt.xlabel(r'Channel')
plt.ylabel(r'Counts')
plt.legend(loc='best')
#plt.xlim(0, 2000)
plt.tight_layout()
plt.savefig('plot2.pdf')