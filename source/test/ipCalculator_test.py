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


sig = ipCalc.batchCalculate(data)

chromaSig = ipCalc.batchChromaCalculate(data)

print sig
print sig.shape
print chromaSig
print chromaSig.shape

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(numData):
	xs = data[i][0]
	ys = data[i][1]
	zs = data[i][2]
	ax.scatter(xs, ys, zs, c=convertToRGB(sig[i]), marker='o')


def drawSig(signatures,sig):
	for i in range(numData):
		if(np.linalg.norm(signatures[i,:]-sig) == 0):
			xs = data[i][0]
			ys = data[i][1]
			zs = data[i][2]
			ax.scatter(xs, ys, zs, c=convertToRGB(signatures[i]), marker='o')
	plt.draw()

axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnext, '[0,0,0]')
bnext.on_clicked(drawSig(sig,possibleSigs[0,:]))
bprev = Button(axprev, '[1,0,0]')
bnext.on_clicked(drawSig(sig,possibleSigs[1,:]))



n = 100
print sig[0,:]
print possibleSigs[0,:]
print np.linalg.norm(sig[0,:]-possibleSigs[0,:])
# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].



ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')



plt.show()

