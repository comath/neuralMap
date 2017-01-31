# The full test suite for the ipCalulator. 
# It's mostly graphical for now, there should be some 
# higher dimensional tests using the connectivity of the data.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D
from ipCalculatorWrap import ipCalculator
from ipCalculatorWrap import convertToRGB

numData = 50
dim = 3
threshhold = 2

idMat = 2*np.identity(dim,dtype=np.float32)
originVec = np.zeros([dim],dtype=np.float32)
originVec[0] = 1
originVec[1] = 1
ipCalc = ipCalculator(idMat,originVec,threshhold)


possibleSigs = np.zeros([9,3],dtype=np.int32)
possibleSigs[0] = [0,0,0]
possibleSigs[1] = [1,0,0]
possibleSigs[2] = [0,1,0]
possibleSigs[3] = [1,1,0]
possibleSigs[4] = [0,0,1]
possibleSigs[5] = [1,0,1]
possibleSigs[6] = [0,1,1]
possibleSigs[7] = [1,1,1]

mean = originVec
cov = idMat
data = np.random.multivariate_normal(mean,cov,numData).astype(dtype=np.float32)


signatures = ipCalc.batchCalculate(data)

chromaSig = ipCalc.batchChromaCalculate(data)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


#HORIFIC HACK, like my spelling
def getDataBySig(data,signatures,sig):
	subdata = []
	for i in range(numData):
		if(np.linalg.norm(signatures[i,:]-sig) == 0):
			subdata.append(data[i])
	retdata = np.zeros([len(subdata),data[i].size])
	for i,datum in enumerate(subdata):
		retdata[i] = datum
	return retdata

print getDataBySig(data,signatures,possibleSigs[7])


ax.scatter(data[:,0],data[:,1],data[:,2], c=chromaSig, marker='o')





ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')



plt.show()

