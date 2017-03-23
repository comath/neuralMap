import fnmatch
import os
import re
import numpy as np
from scipy.misc import comb
import io
import sqlite3
from ipCalculatorWrap import ipCalculator, neuralLayer,readKey, pyCalcKeyLen

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def sizeCalculator(inDim,outDim,depth):
	size = 0
	for i in range(depth):
		size += comb(outDim,i)
	size *= (inDim*inDim+inDim)
	return size

def depthRestrictionCalc(memorySize,inDim,outDim):
	l = 0
	while(sizeCalculator(inDim,outDim,l) < memorySize):
		l +=1
	return l+3

# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

class nnMapper:
	"""docstring for ClassName"""
	def __init__(self,matrix,offset,filename,tablename):
		self.conn = sqlite3.connect(filename)
		self.curs = self.conn.cursor()
		depthRestriction = 7 #depthRestrictionCalc(64*1000*1000*1000,matrix.shape[0],matrix.shape[1])
		self.ipCalc = ipCalculator(matrix,offset,2,depthRestriction)
		self.tablename = tablename
		self.regCalc = neuralLayer(matrix,offset)
		self.keyLength = pyCalcKeyLen(offset.shape[0])
		self.curs.execute('''
			CREATE TABLE IF NOT EXISTS sig_%(tablename)s
				(sigIndex INTEGER PRIMARY KEY ASC, 
				ipSig array, 
				regSig array, 
				CONSTRAINT uniqueLocation UNIQUE (ipSig,regSig))''' 
			% {'tablename':tablename})
		self.curs.execute('''
			CREATE TABLE IF NOT EXISTS join_%(tablename)s
				(dataIndex INTEGER PRIMARY KEY ASC, 
				sigIndex INTEGER NOT NULL,
				errorVal float,
				FOREIGN KEY (sigIndex) REFERENCES sig_%(tablename)s(sigIndex))''' % {'tablename':tablename})

	def addPoints(self,indicies,points,errors = 0):
		ipSigs = self.ipCalc.batchCalculateUncompressed(points)
		regSigs = self.regCalc.batchCalculateUncompressed(points)

		for i,j in enumerate(indicies):
			self.curs.execute('SELECT sigIndex FROM sig_%(tablename)s WHERE ipSig=(?) AND regSig=(?)' 
				% {'tablename':self.tablename},
				(ipSigs[i],regSigs[i]))
			row = self.curs.fetchone()
			if row == None:
				self.curs.execute("INSERT INTO sig_%(tablename)s(ipSig,regSig) VALUES ((?),(?)) "
				% {'tablename':self.tablename},
					(ipSigs[i],regSigs[i]))
				self.curs.execute('SELECT sigIndex FROM sig_%(tablename)s WHERE ipSig=(?) AND regSig=(?)' 
					% {'tablename':self.tablename},
					(ipSigs[i],regSigs[i]))
				row = self.curs.fetchone()

			sigIndex = row[0]
			
			
			if errors != 0:
				self.curs.execute("INSERT INTO join_%(tablename)s(dataIndex,sigIndex,errorVal) VALUES ((?),(?),(?)) "
					% {'tablename':self.tablename},
					(j,sigIndex,errors[i]))
			else:
				self.curs.execute("INSERT INTO join_%(tablename)s(dataIndex,sigIndex,errorVal) VALUES ((?),(?),(?)) "
					% {'tablename':self.tablename},			(j,sigIndex,0))
		self.conn.commit()
	
	def checkPoint(self,point):
		ipSigs = self.ipCalc.calculateUncompressed(point)
		regSigs = self.regCalc.calculateUncompressed(point)
		self.curs.execute('SELECT sigIndex FROM sig_%(tablename)s WHERE ipSig=(?) AND regSig=(?)' 
				% {'tablename':self.tablename},
				(ipSigs[i],regSigs[i]))
		row = fetchone()
		if(row):
			return True
		else:
			return False
	
	def getNeighboors(self,point):
		ipSig = self.ipCalc.calculateUncompressed(point)
		regSig = self.regCalc.calculateUncompressed(point)
		self.curs.execute('SELECT sigIndex FROM sig_%(tablename)s WHERE ipSig=(?) AND regSig=(?)' 
				% {'tablename':self.tablename},
				(ipSig,regSig))
		row = fetchone()
		if(row):
			sigIndex = row[0]
			dataIndices = []
			for row in self.curs.execute('SELECT dataIndex, FROM join_%(tablename)s WHERE sigIndex=(?)' 
				% {'tablename':self.tablename},
				(sigIndex)):
				dataIndices.append(row[0])
			return dataIndices
		else:
			return []

	def getLocationsData():
		print "noting"