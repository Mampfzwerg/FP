import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

y = np.genfromtxt('mess4.txt', unpack=True)
x = np.arange(y.size)
plt.plot(x, y, 'b-', label='unknown radiation source')

plt.xlabel(r'Channel')
plt.ylabel(r'Counts')
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('plot41.pdf')