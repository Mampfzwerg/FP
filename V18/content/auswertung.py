import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

A0 = ufloat(4130, 60)
t05 = ufloat(4943, 5)
t = 7177

A = A0 * unp.exp(- np.log(2) / t05 * t)

print(A)