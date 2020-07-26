import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
import sympy
from scipy import optimize
from scipy.integrate import quad


def klein(E, a):
    c= 3e8
    m0= 9e-31
    m = 0.403 #Fitparameter
    d = -2.683
    E_gamma = 661.35 *10**3 * 1.6 * 10**(-19)
    E = (E* m*10**3 +d*10**3 ) *1.6 * 10**(-19)
    return a *(3/8)*(m0*c**2 / E_gamma**2) * (2+ (E/(E_gamma-E))**2 *  ((m0*c**2 / E_gamma)**2 + (E_gamma -E)/E_gamma - (2*m0*c**2 / E_gamma) * ((E_gamma -E)/E_gamma)) )


y1 = np.genfromtxt('mess2.txt', unpack=True)
y = y1[750:1180]
x = np.linspace(750,1180, 430)
x2 = np.linspace(0,1180, 1180)
#x3 = x = np.arange(y1.size)

params, params_covariance = curve_fit(klein, x, y)
errors = np.sqrt(np.diag(params_covariance)) #Sigma Formel 
print('a:', params[0], '+-', errors[0])
  
plt.plot(x,y,'b-', label=r'137Cs')
plt.plot(x2, klein(x2, *params), 'r-', label=r'Fit with Klein-Nishina')
plt.xlabel(r'Channel')
plt.ylabel(r'Counts')
plt.grid(True)
plt.legend(loc='best') 
plt.savefig('kleinnishina.pdf')

a = params[0]
print(quad(klein, 0, 1180, args=(a,)))