# The full test suite for the ipCalulator. 
# It's mostly graphical for now, there should be some 
# higher dimensional tests using the connectivity of the data.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from mpl_toolkits.mplot3d import Axes3D
from ipCalculatorWrap import ipCalculator
from ipCalculatorWrap import convertToRGB

numData = 10000
dim = 3
threshhold = 2



idMat = np.identity(dim,dtype=np.float32)
originVec = np.zeros([dim],dtype=np.float32)
originVec[0] = 0
originVec[1] = 0
originVec[2] = 0
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
data = np.random.uniform(low=-2.0, high=2.0, size=[numData,3])
data = data.astype(np.float32, copy=False)

signatures = ipCalc.batchCalculate(data,1)

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




subPlots = []
labels = []
bools = []

for possibleSig in possibleSigs:
	dataBySig = getDataBySig(data,signatures,possibleSig)
	l = ax.scatter(dataBySig[:,0],dataBySig[:,1],dataBySig[:,2], c=convertToRGB(possibleSig), marker='o', visible=False)
	subPlots.append(l)
	labels.append(np.array_str(possibleSig))
	bools.append(True)



rax = plt.axes([0.05, 0.4, 0.1, 0.35])


check = CheckButtons(rax, labels, bools)

def func(label):
	index = labels.index(label)
	subPlots[index].set_visible(not subPlots[index].get_visible())
	plt.draw()
check.on_clicked(func)


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')



plt.show()

