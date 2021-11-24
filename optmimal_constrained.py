import numpy as np
import matplotlib.pyplot as plt

tol = 0.000000001

class OptimalConstrained:
  def __init__(self, n, G, d, a_i, b_i):
    self.n = n
    self.G = G
    self.d = d
    self.a_i = a_i  #gradients of inequalities constains
    self.b_i = b_i
    self.nc = a_i.shape[0]  #number of constrains
    self.wk = []
    self.m = 0
    self.x = np.array(n)
    self.p = np.array(n)
    self.debug = False

  def setX(self, vec):
    if vec.size > self.n:
      raise ValueError("Wrong size of input vec!")
    else:
      self.x = vec

  def quad(self, x):
    return 0.5*np.dot(x, np.matmul(self.G, x)) + np.dot(x, self.d)

  def solveSystem(self):
    self.m = len(self.wk)
    m = self.m
    n = self.n
    g = np.matmul(self.G, self.x) + self.d

    if self.m>0:
      A = np.zeros([m, n])
      A[0:m,:] = self.a_i[self.wk]

      kkt = np.zeros([n+m,n+m])
      kkt[0:n,0:n] = self.G
      kkt[n:n+m,0:n] = A
      kkt[0:n,n:n+m] = -A.reshape(-1,m) #only works m=1
      if m>1:
        kkt[0:n,n:n+m] = -A.transpose()
    else:
      kkt = np.zeros([n,n])
      kkt[0:n,0:n] = self.G

    rhs = np.zeros([n+m])
    rhs[0:n] = -g
    kkt_inv = np.linalg.inv(kkt)
    sol = np.matmul(kkt_inv, rhs)
    
    if self.debug:
      print(f"kkt = \n {kkt}") 
      print(f"rhs = \n {rhs}") 
      print(f"g = \n {g}")

    return sol 

  def evaluateP(self):
    sol = self.solveSystem()
    self.p = sol[0:self.n]

  def getPNorm(self):
    return np.sqrt(np.dot(self.p, self.p))
  
  def evaluateMultipliers(self):
    sol = self.solveSystem()
    return sol[self.n:self.n+self.m]

  def evaluateAlpha(self):
    tmp = 10000000
    for i in range(self.nc):
      if i not in self.wk:
        if np.dot(self.a_i[i,:], self.p) < 0:
          minimal=(self.b_i[i]-np.dot(self.a_i[i,:],self.x))/np.dot(self.a_i[i,:],self.p)
          if minimal < tmp:
            tmp = minimal
    return min(1.0, tmp)

  def getBlockConstraints(self):
    tmp = 10000000
    for i in range(self.nc):
      if i not in self.wk:
        if np.dot(self.a_i[i,:], self.p) < 0:
          minimal=(self.b_i[i]-np.dot(self.a_i[i,:],self.x))/np.dot(self.a_i[i,:],self.p)
          if minimal < tmp:
            tmp = minimal
            index = i
    return index

    
