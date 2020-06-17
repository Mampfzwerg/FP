import matplotlib.pyplot as plt
import numpy as np

y = np.genfromtxt('mess2.txt', unpack=True)
x = np.arange(0, 8192)

plt.plot(x, y, 'b-', label='137Cs')
plt.xlabel(r'Channel')
plt.ylabel(r'Counts')
plt.legend(loc='best')
plt.xlim(0, 2000)
plt.ylim(0, 200)

plt.tight_layout()
plt.savefig('plot21.pdf')