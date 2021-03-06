# C Interface

The tool is mainly in `mapper`, but the `adaptiveTools` and `selectionTrainer` files contain the necessary functions to perform adaptive backprop. The `key` file contains functions to handle the bit packed signatures, and the `location` file contains the struct that holds the data produced by the algorithm. `ipTrace` does all the heavy lifting to compute the trace. `mapperTree` contains the binary tree used by `mapper`, and `vector` contains an implementation of C++'s std::vector and is used heavily in 'adaptiveTools'. Finally, `ipCalculator` contains a much slower, but general purpose, means of computing the associated intersection to a point and `parallelTree` contains the data structure that `ipCalculator` uses. `ipTrace` is replacing 'ipCalculator', but needs a few more features before it is complete.


When you add a point with it's index to a map struct an intersection trace and the index is stored in a location (see location.h) under a binary tree with the point's intersection signature (ipSig/ipKey) and region signature (regSig/regKey) as the key pair. If you provide some points with an error class of 1 then they are stored in the same location, though slightly separately to aid in adaptive backprop. This is too much information, but storage is cheap and the full trace may be used separately in the anomaly detection schema later. 


You can access these locations by either turning them into an array using `getLocations` and iterating through them or by using them in adaptive backprop with the handful of functions in the `adaptiveTools/selectionTrainer` files. 	
