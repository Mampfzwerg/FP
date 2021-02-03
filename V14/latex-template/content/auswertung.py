import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp
from numpy.linalg import inv

# Leermessung

t = 300
C = ufloat(6743, 82)

print(C/t)

# Würfel 1

C11 = ufloat(6665, 82)
I11 = C11/t
C12 = ufloat(6503, 81)
I12 = C12/t
C13 = ufloat(6375, 80)
I13 = C13/t

print("Für Würfel 1: ", C11/t, C12/t, C13/t)

# Würfel 2 

C21 = ufloat(1063, 33)
I21 = C21/t
C22 = ufloat(1090, 33)
I22 = C22/t
C23 = ufloat(1547, 39)
I23 = C23/t
C24 = ufloat(674, 26)
I24 = C24/t



print("Für Würfel 2: ", I21, I22, I23, I24)

l_diag = np.sqrt(3**2+3**2)
l_diagkurz = np.sqrt(2**2+2**2)
l_gerade = 3

mu21 = (1/l_gerade)*unp.log(I11/I21)
mu22 = (1/l_gerade)*unp.log(I11/I22)
mu23 = (1/l_diagkurz)*unp.log(I12/I23)
mu24 = (1/l_diag)*unp.log(I12/I24)

print("Absorptionskoeffizienten: ", mu21, mu22, mu23, mu24)

mu2 = np.array([mu21.n, mu22.n, mu23.n, mu24.n])
barmu2 = np.mean(mu2)

print("Mittelwert: ", barmu2, "+-", np.std(mu2))

# Würfel 3 

C31 = ufloat(4849, 70)
I31 = C31/t
C32 = ufloat(4860, 70)
I32 = C32/t
C33 = ufloat(4738, 69)
I33 = C33/t
C34 = ufloat(4338, 66)
I34 = C34/t



print("Für Würfel 3: ", I31, I32, I33, I34)

l_diag = np.sqrt(3**2+3**2)
l_diagkurz = np.sqrt(2**2+2**2)
l_gerade = 3

mu31 = (1/l_gerade)*unp.log(I11/I31)
mu32 = (1/l_gerade)*unp.log(I11/I32)
mu33 = (1/l_diagkurz)*unp.log(I12/I33)
mu34 = (1/l_diag)*unp.log(I12/I34)

print("Absorptionskoeffizienten: ", mu31, mu32, mu33, mu34)

mu3 = np.array([mu31.n, mu32.n, mu33.n, mu34.n])
barmu3 = np.mean(mu3)

print("Mittelwert: ", barmu3, "+-", np.std(mu3))

#Würfel 4

C41 = ufloat(3949,63)
C42 = ufloat(1653,41)
C43 = ufloat(3641,60)
C44 = ufloat(2835,53)
C45 = ufloat(2799,53)
C46 = ufloat(2735,53)
C47 = ufloat(3505,59)
C48 = ufloat(791,28)
C49 = ufloat(2190,47)
C410 = ufloat(1493,39)
C411 = ufloat(1554,39)
C412 = ufloat(1678,41)


I41 = C41/t
I42 = C42/t
I43 = C43/t
I44 = C44/t
I45 = C45/t
I46 = C46/t
I47 = C47/t
I48 = C48/t
I49 = C49/t
I410 = C410/t
I411 = C411/t
I412 = C412/t

print("Würfel 4: ", I41, I42, I43, I44, I45, I46, I47, I48, I49, I410, I411, I412)

A = np.array([[0, np.sqrt(2), 0, np.sqrt(2),0,0,0,0,0],[0,0,np.sqrt(2),0, np.sqrt(2),0,np.sqrt(2),0,0],[0,0,0,0,0,np.sqrt(2),0,np.sqrt(2),0],[1,1,1,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,1,1,1],[0,np.sqrt(2),0,0,0,np.sqrt(2),0,0,0],[np.sqrt(2),0,0,0,np.sqrt(2),0,0,0,np.sqrt(2)],[0,0,0,np.sqrt(2),0,0,0,np.sqrt(2),0],[0,0,1,0,0,1,0,0,1],[0,1,0,0,1,0,0,1,0],[1,0,0,1,0,0,1,0,0]])
AT = A.transpose()

B = AT.dot(A)

Binv = inv(B)

C = Binv.dot(AT)

IV = np.array([unp.log(I12/I41), unp.log(I12/I42), unp.log(I12/I43), unp.log(I11/I44), unp.log(I11/I45), unp.log(I11/I46), unp.log(I12/I47), unp.log(I12/I48), unp.log(I12/I49), unp.log(I11/I410), unp.log(I11/I411), unp.log(I11/I412)])

print("Absorption: ", C.dot(IV))