import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ipCalculatorWrap import ipCalculator



idMat = np.identity(3,dtype=np.float32)
zeroVec = np.zeros([3],dtype=np.float32)
ipCalc = ipCalculator(idMat,zeroVec,2)
print ipCalc.calculate(np.zeros([3],dtype=np.float32))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs, ys, zs, c=c, marker=m)