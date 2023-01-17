import numpy as np
import math
a = np.array([])
b = np.random.normal(0, 0.5, 5)
#print(math.floor(math.log(8/3)/math.log(3/2)))\
l = np.array([1,4,3,0.2,6,-9, 67])
i = 3
l1 = l[:i]
l2 = l[i+1:]
s1 = np.argsort(l1)
s1 = np.append(s1, i)
s2 = np.argsort(-l2)
s2 += i+1
s = np.concatenate((s1,s2))
print(s1)
print(s2)
print(s)