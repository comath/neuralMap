import numpy as np

import matrixIntake

a = np.arange(12, dtype=np.float64).reshape((3,4))

print a

matrixIntake.printMatrix(a)

print a

b = np.arange(5, dtype=np.float64)
c = np.arange(3,8, dtype=np.float64)
print np.dot(b,c)

print matrixIntake.customDot(b,c)