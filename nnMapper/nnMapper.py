import fnmatch
import os
import re
import numpy as np
import io
import sqlite3
from ipCalculatorWrap import ipCalculator, neuralLayer,readKey

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
	def __init__(self,matrix,offset,c,tablename):
		self.conn = sqlite3.connect(filename)
		self.c = self.conn.cursor()
		self.ipCalc = ipCalculator(matrix,offset,2)
		self.tablename = tablename
		self.regCalc = neuralLayer(matrix,offset)
		self.keyLength = ipCalc.keyLength
		self.conn.execute('''
			IF (EXISTS (SELECT name 
				FROM sqlite_master 
				WHERE type='table' 
					AND name='sig_%(tablename)s';))
			BEGIN
				CREATE TABLE sig_%(tablename)s
				(sigIndex int NOT NULL UNIQUE, 
				ipSig array, 
				regSig array, 
				CONSTRAINT uniqueLocation UNIQUE (ipSig,regSig),
				PRIMARY KEY (sigIndex))''' 
			% {'tablename':tablename})
		self.conn.execute('''
			IF (EXISTS (SELECT name 
				FROM sqlite_master 
				WHERE type='table' 
					AND name='sig_%(tablename)s';))
			CREATE TABLE join_%(tablename)s
				(dataIndex int NOT NULL UNIQUE, 
				sigIndex int,
				classificationError float,
				PRIMARY KEY (dataIndex),
				FOREIGN KEY (sigIndex) REFERENCES sig_%(tablename)s(sigIndex))''' % {'tablename':tablename})

	def addPoints(indicies,points,errors):
		ipSigs = ipCalc.batchCalculateUncompressed(points)
		regSigs = regCalc.batchCalculateUncompressed(points)

		for i,j in enumerate(indicies):
			self.conn.execute('SELECT sigIndex FROM sig_%(tablename)s WHERE ipSig=(?),regSig=(?)' 
				% {'tablename':self.tablename},
				(ipSigs(i),regSigs(i)))
			row = fetchone()
			if row:
				self.conn.execute('SELECT MAX(sigIndex) FROM sig_%(tablename)s')
				row = fetchone()
				sigIndex = row[0]+1
			else:
				sigIndex = row[0]
			self.conn.execute("INSERT INTO sig_%(tablename)s VALUES (?),(?),(?) "
				% {'tablename':self.tablename},
					(sigIndex,ipSigs[i],regSigs[i]))
			self.conn.execute("INSERT INTO join_%(tablename)s VALUES (?),(?),(?) "
					% {'tablename':self.tablename},
					(j,sigIndex,errors[i]))
		self.conn.commit()
	
	def checkPoint(point):
		ipSigs = ipCalc.calculateUncompressed(point)
		regSigs = regCalc.calculateUncompressed(point)
		self.conn.execute('SELECT sigIndex FROM sig_%(tablename)s WHERE ipSig=(?),regSig=(?)' 
				% {'tablename':self.tablename},
				(ipSigs(i),regSigs(i)))
		row = fetchone()
		if(row):
			return True
		else:
			return False
	
	def getNeighboors(point):
		ipSigs = ipCalc.calculateUncompressed(point)
		regSigs = regCalc.calculateUncompressed(point)
		self.conn.execute('SELECT sigIndex FROM sig_%(tablename)s WHERE ipSig=(?),regSig=(?)' 
				% {'tablename':self.tablename},
				(ipSigs(i),regSigs(i)))
		row = fetchone()
		if(row):
			sigIndex = row[0]
			dataIndices = []
			for row in self.conn.execute('SELECT dataIndex, FROM join_%(tablename)s WHERE sigIndex=(?)' 
				% {'tablename':self.tablename},
				(sigIndex)):
				dataIndices.append(row[0])
			return dataIndices
		else:
			return []