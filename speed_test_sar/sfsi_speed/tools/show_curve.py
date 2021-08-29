import numpy as np
from matplotlib import pyplot as plt
step = 0.1
x = np.arange(0, 1 + step, step)
a = 3
y = 1/a * (np.log(1 + np.exp(-a * x)) + np.log(1 + np.exp(a * x)))

plt.figure()
plt.plot(x, y)
plt.show()
pass