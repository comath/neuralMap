# Purpose 

This is based on my paper: https://arxiv.org/abs/1612.02522
The objective of this project is to make a fast neural network mapper to use in algorithms to adaptively adjust the neural network topology to the data, harden the network against misclassifying data (adversarial examples) and several other applications. 

# Installing

To install this first make a python wheel with:
```bash
python setup.py bdist_wheel -d ./
```
Then install that wheel with pip, typically:
```bash
sudo pip install nnMap-0.1-cp27-cp27mu-linux_x86_64.whl
```

# Python Interface

To use this in python with either tensorflow or some other machine learning framework run `make mapperWrap` and copy the resulting binary (mapperWrap.so) to whichever folder you are working in. 

For now the python interface consists of one class, `nnMap`. 

#### nnMap Initializer 
```python
nnMap.__init__((A0,A1,...),(b0,b1,...),threshold=2)
```
Parameters:
* (A0,A1,...) must be a sequence of rank 2 32 bit float numpy ndarray with C packing
* (b0,b1,...) must be a sequence of rank 1 32 bit float numpy ndarray with C packing

The A sequence must be a weight matrix, and the bs must be the associated offsets for the fully connected layer's we want to map. The threshold is for the intersection calculation. See the paper for what difference values do to the map.
Currently it only maps the first layer as that is the most important for a sigmoidal neural network. Note, this expects matrices where you multiply on the left.

#### Adding points:
```python
nnMap.add(point,pointIndex,errorClasses=[0 array])
```
Parameters:
* points must be a rank 1 (for a single point) or a rank 2 32 bit float numpy ndarray with C packing
* pointIndexes must be an integer, or a rank 1 32 int numpy ndarray
* errorClasses must be an integer, or a rank 1 32 int numpy ndarray

This adds the point to the map. You must provide the point's vector, and a unique index for that point. Optionally provide whether or not you consider this point to be misclassified. must be either 0 for no error, or 1 for error, optional.

#### Adaptive Step:
```python
nnMap.adaptiveStep(points)
```
Parameters:
* points must be a rank 2 32 bit float numpy ndarray with C packing. This currently has to contain the complete dataset. 

Returns (in order):
* New hyperplane weights, a rank 2 32 bit float numpy ndarray 
* New hyperplane biases, a rank 1 32 bit float in a numpy ndarray
* New selection matrix, a rank 2 32 bit float numpy ndarray
* New selection bias, a rank 1 32 bit float numpy ndarray

Creates a new weight matrix and bias vector with an additional weight vector and bias value representing a new hyperplane designed to cut off the corner with the most error. The new selection matrix and bias are designed to ensure this new hyperplane is properly applied. This throws a `NoErrorLocation` error if there is no corner of sufficient error. 

#### Check Points
```python
nnMap.check(points)
```
Parameters:
* points must be either a rank 1 or rank 2 32 bit float numpy ndarray 

Returns
* a True or False for a single point or an array of boolean values

Computes the region and intersection of each point provided and checks if those occur in the database. You must save your map before using this as it operates off of the database.

#### Save/Load
```python
nnMap.save(filename, tablename=filename+"Table")
```
```python
nnMap.load(filename, tablename=filename+"Table")
```
Parameters:
* filename

Saves to an sqlite database. Provide a tablename if you want to save multiple maps to the same file.


