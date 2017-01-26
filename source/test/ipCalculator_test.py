# The full test suite for the ipCalulator. 
# It's mostly graphical for now, there should be some 
# higher dimensional tests using the connectivity of the data.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ipCalculatorWrap import ipCalculator

numData = 500
dim = 3
threshhold = 2

idMat = 2*np.identity(dim,dtype=np.float32)
originVec = np.zeros([dim],dtype=np.float32)
originVec[0] = 1
originVec[1] = 1
ipCalc = ipCalculator(idMat,originVec,threshhold)




mean = originVec
cov = idMat
data = np.random.multivariate_normal(mean,cov,numData).astype(dtype=np.float32)


chromaSig = ipCalc.batchChromaCalculate(data)
print chromaSig[0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for i in range(numData):
    xs = data[i][0]
    ys = data[i][1]
    zs = data[i][2]
    
    ax.scatter(xs, ys, zs, c=chromaSig[i], marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()