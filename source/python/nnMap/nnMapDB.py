import fnmatch
import os
import re
import numpy as np
import io
import sqlite3
from mapperWrap import readKey, pyCalcKeyLen


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


class nnMapDB:
	def __init__(self,filename,tablename):
		self.conn = sqlite3.connect(filename)
		self.curs = self.conn.cursor()
		self.tablename = tablename
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
		self.curs.execute('''
			CREATE TABLE IF NOT EXISTS fullTrace_%(tablename)s
				(dataIndex INTEGER NOT NULL, 
				rank INTEGER NOT NULL,
				ipSigIndex INTEGER NOT NULL,
				dist float NOT NULL,
				FOREIGN KEY (dataIndex) REFERENCES dataMap_%(tablename)s(dataIndex),
				FOREIGN KEY (ipSigIndex) REFERENCES sig_%(tablename)s(sigIndex))''' % {'tablename':tablename})

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

	def bulkLoadSigs(self,sigs):
		sigs = zip(sigs)
		self.curs.executemany("INSERT INTO sig_%(tablename)s(sig) VALUES ((?)) "% {'tablename':self.tablename},
				sigs)
		self.conn.commit()
	def bulkLoadJoint(self,joint):
		self.curs.executemany("INSERT INTO locJoin_%(tablename)s (ipSigIndex,regSigIndex) VALUES ((?),(?)) "
			% {'tablename':self.tablename},
				joint)
		self.conn.commit()
	def bulkLoadPoints(self,bulkPoints):

		self.curs.executemany("INSERT INTO dataMap_%(tablename)s (dataIndex,locIndex,errorVal) VALUES ((?),(?),(?)) "
				% {'tablename':self.tablename}, bulkPoints)
		self.conn.commit()

	def getSigIndex(self,sig):
		self.curs.execute('SELECT sigIndex FROM sig_%(tablename)s WHERE sig=(?)' 
			% {'tablename':self.tablename},	(sig,))
		row = self.curs.fetchone()
		if row == None:
			return None
		else:
			return row[0]

	def checkReg(self,regSig):
		regSigIndex = self.getSigIndex(regSig)
		if regSigIndex == None:
			return None
		else:
			self.curs.execute('SELECT locIndex FROM locJoin_%(tablename)s WHERE regSigIndex=(?)' 
				% {'tablename':self.tablename},	(regSigIndex,))
			row = self.curs.fetchone()
			if row == None:
				return None
			return row[0]

	def getPointLocationIndex(self,ipSig,regSig,precomputedIndex=False):
		if precomputedIndex:
			ipSigIndex = ipSig
			regSigIndex = regSig
		else:
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
	
	def insertOrGetPointLocationIndex(self,ipSigIndex,regSigIndex):
		self.curs.execute('SELECT locIndex FROM locJoin_%(tablename)s WHERE ipSigIndex=(?) AND regSigIndex=(?)' 
			% {'tablename':self.tablename},
			(ipSigIndex,regSigIndex))
		row = self.curs.fetchone()
		if row == None:
			self.curs.execute("INSERT INTO locJoin_%(tablename)s (ipSigIndex,regSigIndex) VALUES ((?),(?)) "
			% {'tablename':self.tablename},
				(ipSigIndex,regSigIndex))
			self.curs.execute('SELECT locIndex FROM locJoin_%(tablename)s WHERE ipSigIndex=(?) AND regSigIndex=(?)' 
				% {'tablename':self.tablename},
				(ipSigIndex,regSigIndex))
			row = self.curs.fetchone()
		return row[0]

	def addPointsToLocation(self,indexes,ipSig,regSig,errors):
		ipSigIndex = self.insertOrGetSig(ipSig)
		regSigIndex = self.insertOrGetSig(regSig)
		jointSigIndex = self.insertOrGetPointLocationIndex(ipSigIndex,regSigIndex)
		things = []
		for i in range(indexes.shape[0]):
			things.append((indexes[i].item(),jointSigIndex,errors[i].item(),))
		self.curs.executemany("INSERT INTO dataMap_%(tablename)s (dataIndex,locIndex,errorVal) VALUES ((?),(?),(?)) "
				% {'tablename':self.tablename}, things)
		self.conn.commit()

	def getLocationList(self):
		locList = []
		for row in curs.execute("""SELECT ipSig, sig AS regSig 
												FROM (SELECT regSigIndex, sig 
													AS ipSig 
													FROM locJoin_hidden%(hd)03ditte%(i)d 
														INNER JOIN sig_hidden%(hd)03ditte%(i)d 
															ON sigIndex = ipSigIndex) 
												INNER JOIN sig_hidden%(hd)03ditte%(i)d ON sigIndex = regSigIndex;"""
												%{'hd':hd,'i':i}):
			locList.append((row[0],row[1]))
		return locList

	def getRegionList(self):
		regList = []
		for row in curs.execute("""SELECT sig FROM sig_hidden%(hd)03ditte%(i)d
										WHERE sig_hidden%(hd)03ditte%(i)d IN 
											(SELECT regSigIndex FROM locJoin_hidden%(hd)03ditte%(i)d);"""%{'hd':hd,'i':i}):
			regList.append(row[0])
		return regList
