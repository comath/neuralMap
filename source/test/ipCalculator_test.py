# The full test suite for the ipCalulator. 
# It's mostly graphical for now, there should be some 
# higher dimensional tests using the connectivity of the data.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ipCalculatorWrap import ipCalculator



idMat = np.identity(2,dtype=np.float32)
zeroVec = np.zeros([2],dtype=np.float32)
ipCalc = ipCalculator(idMat,zeroVec,2)
print ipCalc.calculate(np.zeros([2],dtype=np.float32)) #Should be 1,1,1


mean = [0,0]
cov = [[1,0],[0,1]]
data = np.random.multivariate_normal(mean,cov,3).astype(dtype=np.float32)

print ipCalc.batchCalculate(data)

