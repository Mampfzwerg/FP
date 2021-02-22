import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp
from numpy.linalg import inv

I, B = np.genfromtxt('mess.txt', unpack=True)

params, covariance_matrix = np.polyfit(I, B, deg=1, cov=True)

errors = np.sqrt(np.diag(covariance_matrix))

print('a = {:.8f} ± {:.8f}'.format(params[0], errors[0]))
print('b = {:.8f} ± {:.8f}'.format(params[1], errors[1]))

def gerade (x, m, b):
    return m*x+b

z = np.linspace(0, 5)

plt.plot(I, B, 'rx', label = 'Messwerte')
plt.plot(z, gerade (z, *params), 'b-', label='Lineare Regression')
plt.xlabel(r'$I / A$')
plt.ylabel(r'$B / mT$')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('plot1.pdf')