import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp

R1, R2, V, I, t = np.genfromtxt('mess1.txt', unpack=True) 

U = unp.uarray(V,0.5)
I2 = unp.uarray(I,0.5) 
t2 = unp.uarray(t,2)
Mol = 63.5
m = 342
kappa = 137.8*10**9
V = 7.11*10**(-6)

T1 = 0.00134*(R1*10**3)**2+2.296*(R1*10**3)-243.02
T2 = 0.00134*(R2*10**3)**2+2.296*(R2*10**3)-243.02

T1K = T1 + 273.15
T2K = T2 + 273.15

Cp = (U*(I2*10**(-3))*t2*Mol)/((T1K-T2K)*m)

alpha = np.array([8.5,8.5,9.75,10.70,10.70,11.50,12.10,12.65,12.65,13.15,13.60,13.90,14.25,14.25,14.50,14.75,14.95,15.20,15.40,15.60,15.75,15.90,16.10,16.10,16.25,16.35,16.50,16.65])

Tbar = (T1K+T2K)/2
CV = Cp - 9*(alpha*10**(-6))**2*kappa*V*Tbar
print("Konstantes Volumen: ", np.absolute(CV))

plt.errorbar(Tbar, unp.nominal_values(np.absolute(CV)), yerr = unp.std_devs(CV), fmt='rx')
plt.xlabel(r'$\bar{T} \: / \: K$')
plt.ylabel(r'$C_V \: / \: Jmol^-1K^-1$')

plt.tight_layout()
plt.savefig('plot1.pdf')

#Debye Temp
Tbar2 = np.array([80.9373495,84.9464095,90.972763,97.0164963,103.6696927,111.1817038,118.0025967,125.4442087,131.8300216,140.4075379,147.9288263,156.0834432,165.7338883])
print(Tbar2)
Tab = np.array([15.3,10.6,13.6,11.0,11.5,13.3,12.2,16.2,13.3,15.2,15.7,12.9,14.8]) 
debye = Tbar2*Tab
print("Debye Temperaturen: ", debye)
dm = np.mean(debye)
print("Mittelwert: ", dm, np.std(debye))