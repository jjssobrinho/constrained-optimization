import numpy as np
import matplotlib.pyplot as plt
data_t = np.float64

def quad(x):
  # Quad function of problem 16.9 from Nocedal
  G = np.array([[2.0, -2.0], [-2.0, 4.0]])
  d = np.array([-2.0, -6.0])

  return 0.5*np.dot(x, np.matmul(G, x)) + np.dot(x, d)

size = 1000
x1 = np.linspace(-2.0, 2.0, size)
x2 = np.linspace(-2.0, 2.0, size)
x = np.zeros(2)
f = np.zeros([size, size])
for i in range(size):
  for j in range(size):
    x[0], x[1] = x1[j], x2[i]
    f[i, j] = quad(x)

plt.figure()
plt.contour(x2, x1, f, 38)
plt.plot(x1, 2.-x1, 'k')
plt.plot(x1, 1+0.5*x1, 'k')
plt.plot(x1, 0*x1, 'k')
plt.plot(0*x2, x2, 'k')
plt.ylim([-2.0, 2.0])
plt.fill_between([0, 2./3, 2], [1., 2.-2./3, 0], y2=0)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()