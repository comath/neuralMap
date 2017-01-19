import numpy as np
import os

print os.getcwd()

from ipCalculatorWrap import ipCalculator
idMat = np.identity(3,dtype=np.float32)
zeroVec = np.zeros([3],dtype=np.float32)


ipCalc = ipCalculator(idMat,zeroVec,2)


print ipCalc.calculate(np.zeros([3],dtype=np.float32))