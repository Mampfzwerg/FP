import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp

T, I = np.genfromtxt('mess1.txt', unpack=True)

diff = np.zeros(T.size - 1)

for i in range(T.size - 1):
    diff[i] = abs(T[i+1] - T[i])

H = ufloat(np.mean(diff), np.std(diff))
#print(diff)
print(H)

########################################################################

T, I = np.genfromtxt('mess2.txt', unpack=True)

diff = np.zeros(T.size - 1)

for i in range(T.size - 1):
    diff[i] = abs(T[i+1] - T[i])

H = ufloat(np.mean(diff), np.std(diff))
#print(diff)
print(H)





