import matplotlib.pyplot as plt
import numpy as np

E = np.array([121.78, 244.70, 344.30, 411.12, 443.96, 778.90, 867.37, 964.08, 1085.90, 1112.10, 1408.00])
x = np.array([ 308.80, 613.80, 860.70, 1026.51, 1107.93, 1938.62, 2158.39, 2398.18, 2700.41, 2765.02, 3499.69])

params, covariance_matrix = np.polyfit(x, E, deg=1, cov=True)

errors = np.sqrt(np.diag(covariance_matrix))

print('a = {:.3f} ± {:.4f}'.format(params[0], errors[0]))
print('b = {:.3f} ± {:.4f}'.format(params[1], errors[1]))

def gerade (x, m, b):
    return m*x+b

z = np.linspace(np.min(x) - 100, np.max(x) + 100)

plt.plot(x, E, 'bx', label='152Eu')
plt.plot(z, gerade (z, *params), 'r-', label='Linear Regression')
plt.xlabel(r'$\mu_0\: / \:$ channel')
plt.ylabel(r'$E \: / \:$ keV')
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('plot5.pdf')