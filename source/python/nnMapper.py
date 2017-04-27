import fnmatch
import os
import re
import numpy as np
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

# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)



class nnMapper:
	"""docstring for ClassName"""
	def __init__(self,matrix,offset,filename,tablename):
		self.conn = sqlite3.connect(filename)
		self.curs = self.conn.cursor()
		self.ipCalc = ipCalculator(matrix,offset,2)
		self.tablename = tablename
		self.regCalc = neuralLayer(matrix,offset)
		self.keyLength = pyCalcKeyLen(offset.shape[0])
		self.curs.execute('''
			CREATE TABLE IF NOT EXISTS sig_%(tablename)s
				(sigIndex INTEGER PRIMARY KEY ASC, 
				sig array NOT NULL)''' 
			% {'tablename':tablename})
		self.curs.execute('''
			CREATE TABLE IF NOT EXISTS locJoin_%(tablename)s
				(locIndex INTEGER PRIMARY KEY ASC, 
				ipSigIndex INTEGER, 
				regSigIndex INTEGER, 
				CONSTRAINT uniqueLocation UNIQUE (ipSigIndex,regSigIndex),
				FOREIGN KEY (ipSigIndex) REFERENCES sig_%(tablename)s(sig),
				FOREIGN KEY (regSigIndex) REFERENCES sig_%(tablename)s(sig))''' 
			% {'tablename':tablename})
		self.curs.execute('''
			CREATE TABLE IF NOT EXISTS dataMap_%(tablename)s
				(dataIndex INTEGER PRIMARY KEY ASC, 
				locIndex INTEGER NOT NULL,
				errorVal float,
				FOREIGN KEY (locIndex) REFERENCES locJoin_%(tablename)s(locIndex))''' % {'tablename':tablename})

	def insertOrGetSig(self,sig):
		self.curs.execute('SELECT sigIndex FROM sig_%(tablename)s WHERE sig=(?)' 
			% {'tablename':self.tablename},	(sig,))
		row = self.curs.fetchone()
		if row == None:
			self.curs.execute("INSERT INTO sig_%(tablename)s(sig) VALUES ((?)) "
			% {'tablename':self.tablename},
				(sig,))
			self.curs.execute('SELECT sigIndex FROM sig_%(tablename)s WHERE sig=(?)' 
				% {'tablename':self.tablename},
				(sig,))
			row = self.curs.fetchone()
		return row[0]

	def getSigIndex(self,sig):
		self.curs.execute('SELECT sigIndex FROM sig_%(tablename)s WHERE sig=(?)' 
			% {'tablename':self.tablename},	(sig,))
		row = self.curs.fetchone()
		if row == None:
			return None
		else:
			return row[0]
	
	def insertOrGetPointLocationIndex(self,ipSigIndex,regSigIndex):
		self.curs.execute('SELECT locIndex FROM locJoin_%(tablename)s WHERE ipSigIndex=(?) AND regSigIndex=(?)' 
			% {'tablename':self.tablename},
			(ipSigIndex,regSigIndex))
		row = self.curs.fetchone()
		if row == None:
			self.curs.execute("INSERT INTO locJoin_%(tablename)s(ipSigIndex,regSigIndex) VALUES ((?),(?)) "
			% {'tablename':self.tablename},
				(ipSigIndex,regSigIndex))
			self.curs.execute('SELECT locIndex FROM locJoin_%(tablename)s WHERE ipSigIndex=(?) AND regSigIndex=(?)' 
				% {'tablename':self.tablename},
				(ipSigIndex,regSigIndex))
			row = self.curs.fetchone()
		return row[0]

	def addPoints(self,indicies,points,errors = 0):
		ipSigs = self.ipCalc.batchCalculateUncompressed(points)
		regSigs = self.regCalc.batchCalculateUncompressed(points)
		for i,j in enumerate(indicies):

			ipSigIndex = self.insertOrGetSig(ipSigs[i])
			regSigIndex = self.insertOrGetSig(regSigs[i])
			jointSigIndex = self.insertOrGetPointLocationIndex(ipSigIndex,regSigIndex)
			
			if errors != 0:
				self.curs.execute("INSERT INTO dataMap_%(tablename)s(dataIndex,locIndex,errorVal) VALUES ((?),(?),(?)) "
					% {'tablename':self.tablename},
					(j,jointSigIndex,errors[i]))
			else:
				self.curs.execute("INSERT INTO dataMap_%(tablename)s(dataIndex,locIndex,errorVal) VALUES ((?),(?),(?)) "
					% {'tablename':self.tablename},			
					(j,jointSigIndex,0))
		self.conn.commit()
	
	def getPointLocationIndex(self,point):
		ipSig = self.ipCalc.calculateUncompressed(point)
		regSig = self.regCalc.calculateUncompressed(point)
		ipSigIndex = self.getSigIndex(ipSig)
		regSigIndex = self.getSigIndex(regSig)
		
		if(ipSigIndex != None and regSigIndex != None):
			self.curs.execute('SELECT locIndex FROM locJoin_%(tablename)s WHERE ipSigIndex=(?) AND regSigIndex=(?)' 
					% {'tablename':self.tablename},
					(ipSigIndex,regSigIndex))
			row = self.curs.fetchone()
			if(row):
				return row[0]
		return None

	def getPointsLocationIndices(self,indices,points):
		ipSigs = self.ipCalc.batchCalculateUncompressed(points)
		regSigs = self.regCalc.batchCalculateUncompressed(points)
		locIndices = []
		for i,j in enumerate(indices):
			ipSigIndex = self.getSigIndex(ipSigs[i])
			regSigIndex = self.getSigIndex(regSigs[i])
			if(ipSigIndex != None and regSigIndex != None):
				self.curs.execute('SELECT locIndex FROM locJoin_%(tablename)s WHERE ipSigIndex=(?) AND regSigIndex=(?)' 
						% {'tablename':self.tablename},
						(ipSigIndex,regSigIndex))
				row = self.curs.fetchone()
				if(row):
					locIndices.append(row[0])
				else:
					locIndices.append(None)
			else:
				locIndices.append(None)
		return locIndices

	
	def getNeighboors(self,point):
		locIndex =self.getPointLocationIndex(point)
		if(locIndex):
			dataIndices = []
			for row in self.curs.execute('SELECT dataIndex FROM dataMap_%(tablename)s WHERE locIndex=(?)' 
				% {'tablename':self.tablename},
				(locIndex,)):
				dataIndices.append(row[0])
			return dataIndices
		else:
			return []

	def checkPoint(self, point):
		locIndex =self.getPointLocationIndex(point)
		if(locIndex):
			return True
		else:
			return False

	def checkPoints(self,indices,points):
		locIndices = self.getPointsLocationIndices(indices,points)
		boolLoc = []
		for i,j in enumerate(indices):
			if(locIndices[i]):
				boolLoc.append((j,True))
			else:
				boolLoc.append((j,False))
		return boolLoc

	def checkRegion(self, indices, points):
		regSigs = self.regCalc.batchCalculateUncompressed(points)
		locIndices = []
		for i,j in enumerate(indices):
			regSigIndex = self.getSigIndex(regSigs[i])
			if(regSigIndex != None):
				self.curs.execute('SELECT locIndex FROM locJoin_%(tablename)s WHERE regSigIndex=(?)' 
						% {'tablename':self.tablename},
						(regSigIndex,))
				row = self.curs.fetchone()
				if(row):
					locIndices.append((j,True))
				else:
					locIndices.append((j,False))
			else:
				locIndices.append((j,False))
		return locIndices