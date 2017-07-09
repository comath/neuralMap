import fnmatch
import os
import re
import numpy as np
import io
import sqlite3
import mapperWrap

class NoErrorLocation(ValueError):
	'''This is thrown when there is no areas with a sufficient error population'''

class nnMap():
	"""docstring for ClassName"""
	def __init__(self,matrices,offsets, threshold=2):
		self.threshold = threshold
		self.layers = zip(matrices,offsets)
		self.internalMaps = []
		self.ipCalcs = []
		self.regCalcs = []
		if(len(matrices) == 1):
			usedLayers = self.layers
		else:
			usedLayers = self.layers[:-1]
		for matrix, offset in usedLayers:
			matrix = np.ascontiguousarray(matrix.T, dtype=np.float32)
			offset = np.ascontiguousarray(offset, dtype=np.float32)
			self.internalMaps.append(mapperWrap.cy_nnMap(matrix, offset, threshold))
			self.ipCalcs.append(mapperWrap.traceCalc(matrix, offset))
			self.regCalcs.append(mapperWrap.neuralLayer(matrix, offset))

	def add(self,points,pointIndexes,**kwargs):
		errorClasses = kwargs.get('errorClasses')
		if(len(points.shape) == 1):
			if not 'errorClasses' in kwargs:
				errorClasses = 0
			self.internalMaps[0].add(points, pointIndexes, errorClass)
		elif(len(points.shape) == 2):
			if not 'errorClasses' in kwargs:
				errorClasses = np.zeros_like(pointIndexes)
			self.internalMaps[0].batchAdd(points, pointIndexes, errorClasses)
		else:
			raise ValueError('Must provide either a rank 1 or rank 2 numpy ndarray')

	def save(self, filename, **kwargs):
		tablename = kwargs.get('table_name')
		if not tablename:
			tablename = filename+"Table"
		self.internalDB = nnMap_db(filename,tablename)
		numLoc = self.internalMaps[0].numLocations(self)
		for i in range(numLoc):
			ipSig, regSig, normalPointIndex, errorPointIndexes = self.internalMaps[0].location(i).all()
			normalErrorClasses = np.zeros_like(normalPointIndex)
			errorClasses = np.ones_like(errorPointIndexes)
			indexes = np.concatonate(normalPointIndex,errorPointIndexes)
			errorClasses = np.concatonate(normalErrorClasses,errorClasses)
			self.internalDB.addPointsToLocation(indexes,ipSig,regSig,errorClasses)

	def adaptiveStep(self, data):
		for i, internalMap in enumerate(self.internalMaps):
			selectionMatrix =  np.ascontiguousarray(self.layers[i+1][0].T,dtype=np.float32)
			selectionBias = np.ascontiguousarray(self.layers[i+1][1],dtype=np.float32)
			newHiddenDim, npNewHPVec, npNewHPbias, newSelectionWeight, newSelectionBias = internalMap.adaptiveStep(data,selectionMatrix,selectionBias)
			if(newHiddenDim == self.layers[i][1].shape[0] + 1):
				newWeights = np.concatenate([self.layers[i][0].T,npNewHPVec], axis=0)
				newBias = np.concatenate([self.layers[i][1],npNewHPbias])
				return newWeights.T, newBias, newSelectionWeight.T, newSelectionBias
			else:
				NoErrorLocation("Cannot create a hyperplane")

	def load(self, filename, **kwarg):
		tablename = kwargs.get('tableName')
		if not 'tableName' in kwargs:
			tablename = filename+"Table"
		self.internalDB = nnMap_db(filename,tablename)

	def check(self,points):
		if not self.internalDB:
			raise ValueError("Must have a DB first")
		if(len(points.shape) == 1):
			points = np.reshape(points,(1,-1))
		numPoints = points.shape[0]
		ipSigs = self.ipCalc[0].getIntersections(points,self.threshold)
		regSigs = self.regCalc[0].getRegions(points)
		if(len(points.shape) > 1):
			retVals = []
			for i in range(numPoints):
				jointIndex = self.internalMaps[0].getPointLocationIndex(ipSig[i],regSigs[i])
				if(jointIndex != None):
					retBools.append(True)
				else:
					retBools.append(False)
		if(len(points.shape) == 1):
			jointIndex = self.internalMaps[0].getPointLocationIndex(ipSig[i],regSigs[i])
			if(jointIndex != None):
				return True
			else:
				return False