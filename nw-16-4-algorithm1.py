import numpy as np
import matplotlib.pyplot as plt
from optmimal_constrained import *

# definning the problem:
n = 2
G = np.array([[2., 0],
              [0, 2.]])
d = np.array([-2.0, -5.0])
a = np.array([[1.0,-2.0],
              [-1.0, -2.0],
              [-1., 2.0],
              [1.0, 0.0],
              [0.0, 1.0]])
b = np.array([-2.0, -6.0, -2.0, 0.0, 0.0])

# instace object:
opt = OptimalConstrained(n, G, d, a, b)

# initial point
x = np.array([2.0, 0.0])
opt.setX(x)
opt.wk = [2, 4]

alpha = 0.0
kmax = 10
for k in range(kmax):
  opt.evaluateP()
  print(f"k = {k}, x = {opt.x}, p = {opt.p}, w = {opt.wk}")
  pnorm = opt.getPNorm()
  if pnorm < tol:
    lagrange = opt.evaluateMultipliers()
    # checking to see if all multipliers are nonnegative:
    if all(lagrange > tol):
      print(f"x* = {opt.x}")
      break  #xstar = x -> optimal first order condition
    else:
      #get most negative multiplier:
      tmp = 100
      for j in range(len(opt.wk)):
        if lagrange[j] < tmp:
          tmp = lagrange[j]
          index = j
      # updating wk -> working set:
      opt.wk.pop(index) #removing index of most negative multiplier

  else: #(* p_k != 0 *)
    alpha = opt.evaluateAlpha()
    opt.x = opt.x + alpha*opt.p
    #check alpha value to see if there are block constraints:
    if alpha < 1.0:
      index = opt.getBlockConstraints()
      print(f"index = {index}")
      opt.wk.append(index)
  print(f"alpha = {alpha}")
