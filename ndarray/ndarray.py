import numpy as np


a = np.array([[1, 2, 3], [4, 5, 6]])
b = a[0]
print(b)  # Output: [1 2 3]
s = a.sum(axis=0)
print(s) # Output: [5 7 9]
v = a.sum(axis=1)
print(v) # Output: [ 6 15]
c = a[1]
print(c)  # Output: [4 5 6]
d = a[:, 1]
print(d)  # Output: [2 5]
e = a[:, 2]
print(e)  # Output: [3 6]
f = a[0, 1]
print(f)  # Output: 2
g = a[1, :]
print(g)  # Output: [4 5 6]

a2 = np.array([[7, 8, 9], [10, 11, 12]])
h = a + a2
print(h)  # Output: [[ 8 10 12]
          #          [14 16 18]]
i = a * a2
print(i)  # Output: [[ 7 16 27]
          #          [40 55 72]]    
j = a.dot(a2.T)
print(j)  # Output: [[  50   68]
          #          [ 122  167]]
print(a2.T) # Output: [[ 7 10]
          #          [ 8 11]
          #          [ 9 12]]