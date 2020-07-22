import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

n11 = np.array([16, 48, 4, 18, 4, 5, 21, 2, 53])
n12 = np.array([49, 47, 4, 49, 18, 24, 16, 13, 29])

n21 = np.array([47, 55, 12, 2, 3, 26, 31, 25, 13])
n22 = np.array([39, 14, 11, 10, 16, 16, 27, 59, 20])

n31 = np.array([51, 14, 21, 15, 18, 57, 17, 31, 59])
n32 = np.array([4, 10, 35, 27, 2, 34, 51, 57, 14])

x = 5/3

print(n32, n32 * x)