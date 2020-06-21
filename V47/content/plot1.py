import matplotlib.pyplot as plt
import numpy as np

U, I, t, R1, R2 = np.genfromtxt('mess1.txt', unpack=True) 
Mol = 63.5
m = 342
kappa = 137.8*10**9
V = 7.11*10**(-6)

T1 = 0.00134*R1^2+2.296*R1-243.02
T2 = 0.00134*R2^2+2.296*R2-243.02

Cp = (U*I*t*M)/((T2-T1)*m)
print("Konstanter Druck: ", Cp)

alpha = np.array([0,1])

Tbar = (T2-T1)/2

CV = Cp - 9*alpha**2*kappa*V*Tbar
print("Konstantes Volumen: ", CV)

plt.plot(Tbar, CV, 'bx', label='CV')
plt.xlabel(r'$\bar{T} \: / \: K$')
plt.ylabel(r'$C_V \: / \: Jmol^-1K^-1$')
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('plot1.pdf')

#Debye Temp

Tab = np.array([0,0])
debye = Tbar*Tab
print("Debye Temperaturen: ", debye)
dm = np.mean(debye)
print("Mittelwert: ", dm)