import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy.signal import find_peaks
import csv

filename = 'scope_5.csv'

with open(filename) as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    header_row = next(reader)

    #for index, column_header in enumerate(header_row):
    #    print(index, column_header)

    second, volt = [],[]
    for row in reader:  
        sec1 = float(row[0])
        second.append(sec1)
        volt1 = float(row[1])
        volt.append(volt1)

peaks,_ = find_peaks(volt, height=0, distance = 2)
peaks2, _ = find_peaks(volt, height=0.1, distance=60)

pek = []
for i in range(0,len(peaks)):
    pek.append(volt[peaks[i]])

pek2 = []
for i in range(0,len(peaks)):
    pek2.append(second[peaks[i]])

pek3 = []
for i in range(0,len(peaks2)):
    pek3.append(volt[peaks2[i]])

pek4 = []
for i in range(0,len(peaks2)):
    pek4.append(second[peaks2[i]])

def fit(t, a, b, c):
    return a*np.exp(-(t)/b)+c

params, cov = curve_fit(fit, pek4, pek3)
err = np.sqrt(np.diag(abs(cov)))

a = ufloat(params[0], err[0])
b = ufloat(params[1], err[1])
c = ufloat(params[2], err[2])

print(a, b, c)

plt.plot(pek4, pek3, 'rx', label=r'Peaks')
pek5 = np.array(pek4)
plt.plot(pek4, fit(pek5, *params), 'b-', label='Ausgleichsrechnung')
plt.xlabel(r'$t \, / \, s$')
plt.ylabel(r'$U \, / \, V$')
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('plot2.pdf')
