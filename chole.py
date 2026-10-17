
import numpy as np
import math

def chole(a):
    n = len(a)
    for k in range(n):
        try:
            a[k,k] = math.sqrt(a[k,k] - np.dot(a[k,0:k],a[k,0:k]))
        except ValueError:
            print('La matriz no es definida positiva!!!!')
        for i in range(k+1,n):
            a[i,k] = (a[i,k] - np.dot(a[i,0:k],a[k,0:k]))/a[k,k]
    for k in range(1,n): a[0:k,k] = 0.0
    return a

def cholsol(L,b):
    n = len(b)
  # Solucion de [L]{y} = {b}  
    for k in range(n):
        b[k] = (b[k] - np.dot(L[k,0:k],b[0:k]))/L[k,k]
  # Solucion de [L^T]{x} = {y}      
    for k in range(n-1,-1,-1):
        b[k] = (b[k] - np.dot(L[k+1:n,k],b[k+1:n]))/L[k,k]
    return b

A = np.array([[3.0, 2.0],[2.0, 6.0]])
b = np.array([5.0,8.0])

R=chole(np.copy(A))
x = cholsol(R,b)

print('Solucion: ',x)
