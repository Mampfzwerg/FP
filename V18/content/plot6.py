import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

E = np.array([121.78, 244.70, 344.30, 411.12, 443.96, 778.90, 867.37, 964.08, 1085.90, 1112.10, 1408.00])
x = np.array([ 308.80, 613.80, 860.70, 1026.51, 1107.93, 1938.62, 2158.39, 2398.18, 2700.41, 2765.02, 3499.69])
W = np.array([28.6, 7.6, 26.5, 2.2, 3.1, 12.9, 4.2, 14.6, 10.2, 13.6, 21.0])
a = np.array([ ufloat(3493.0, 15.0), ufloat(535.0, 9.0), ufloat(1140.0, 6.0), ufloat(74.7, 3.5), ufloat(94.2, 3.0),
    ufloat(143.3, 2.4), ufloat(42.5, 2.1), ufloat(120.3, 2.2), ufloat(67.5, 2.2), ufloat(88.5, 2.6), ufloat(90.7, 1.4)])
b = np.array([ufloat(3.5, 0.0), ufloat(4.2, 0.2), ufloat(5.5, 0.1), ufloat(5.7, 0.6), ufloat(6.4, 0.5), ufloat(14.3, 0.5),
    ufloat(17.1, 2.0), ufloat(19.3, 0.8), ufloat(30.2, 2.3), ufloat(23.2, 1.7), ufloat(28.9, 1.1)])

t = 4111
A = ufloat(1574, 23)
Z = a * unp.sqrt(b * np.pi)
Q = 4 * Z / (0.0538 * A * (W/100)  * t)

#print(Z, unp.sqrt(Z), Q)

#E = np.log(E)
Q = unp.nominal_values(Q)
#print(E)

def exp(x, a, b):
    return a*x**b

params, covariance_matrix = optimize.curve_fit(exp, E, Q)
errors = np.sqrt(np.diag(covariance_matrix))

print('a = {:.3f} ± {:.4f}'.format(params[0], errors[0]))
print('b = {:.3f} ± {:.4f}'.format(params[1], errors[1]))

z = np.linspace(np.min(E), np.max(E))

plt.plot(E, Q, 'bx', label='152Eu')
plt.plot(z, exp(z, *params), 'r-', label='Exponential Fit')
plt.xlabel(r'$E \;/\; $keV')
plt.ylabel(r'$Q$')
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('plot6.pdf')