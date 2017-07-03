# nnMapper

This is based on my paper: https://arxiv.org/abs/1612.02522
The objective of this project is to make a fast neural network mapper to use in algorithms to adaptively adjust the neural network topology to the data, harden the network against misclassifying data (see adverserial examples) and several other applications. 

# Python Interface

To use this in python with either tensorflow or some other machine learning framework run `make mapperWrap` and copy the resulting binary (mapperWrap.so) to whichever folder you are working in. 

For now the python interface consists of one class, `nnMap`. 

#### Initializer 
```python
nnMap.__init__(A,b)
```
Parameters:
* A must be a rank 2 32 bit float numpy ndarray with C packing
* b must be a rank 1 32 bit float numpy ndarray with C packing
A must be a weight matrix, b must be the associated offsets. These must be for the hyperplane layer that you want to map out, directly below the selection layer that you may use later.

#### Adding points:
```python
nnMap.add(point,pointIndex,pointErrorClass)
```
Parameters:
* point must be a rank 1 32 bit float numpy ndarray with C packing
* pointIndex must be an integer
* pointErrorClass must be either 0 for no error, or 1 for error.
This adds the point to the map. You must provide the point's vector, a unique index for that point and whether or not you consider this point to be misclassified.

#### Adding points:
```python
nnMap.addBatch(points,pointsIndexes,pointsErrorClasses)
```
Parameters:
* points must be a rank 2 32 bit float numpy ndarray with C packing
* pointsIndexes must be a rank 1 32 bit integer numpy array. 
* pointsErrorClasses must be rank 1 32 bit integer numpy array consisting of either 0 for no error, or 1 for error.
This adds a batch of points in parallel to the internal map. The arrays `pointsErrorClasses` and `pointsIndexes` must be of the same length as the number of points you have. See `add` for the values these should take.

#### Adaptive Step:
```python
nnMap.adaptiveStep(points,A1,b1)
```
Parameters:
* points must be a rank 2 32 bit float numpy ndarray with C packing. This currently has to contain the complete dataset. 
* A1 must be a rank 2 32 bit float numpy ndarray with C packing
* b1 must be a rank 1 32 bit float numpy ndarray with C packing

Returns (in order):
* New dimesion, this will not change in the event of a failure. 
* new hyperplane vector, a rank 1 32 bit float numpy ndarray 
* New bias, a single 32 bit float in a numpy ndarray
* New selection matrix, a rank 2 32 bit float numpy ndarray of shape [outDim,inDim + 1], where A1 has shape [outDim,inDim]
* New selection bias, a rank 1 32 bit float numpy ndarray of shape [outDim]

A1 and b1 have to be the weight matrix and bias for the selection layer directly above the provided hyperplane layer. 

#### Location Count:
```python
nnMap.numLocations()
```
Returns the number of locations currently in the map.


#### Intersection signature of a location:
```python
nnMap.location(index).ipSig()
```
Parameters:
* index must be an integer less than the one returned by numLocations
This returns the intersection signature for the ith location (ordered lexographically).

#### Region signature of a location:
```python
nnMap.location(index).regSig()
```
Parameters:
* index must be an integer less than the one returned by numLocations
This returns the region signature for the ith location (ordered lexographically).

#### Point indexes of a location:
```python
nnMap.location(index).pointIndexes()
```
Parameters:
* index must be an integer less than the one returned by numLocations
Returns the indexes of the points stored in this location. You can then use these indexes to reference the actual points.



