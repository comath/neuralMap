// From http://eddmann.com/posts/implementing-a-dynamic-vector-array-in-c/
#ifndef LOCATION_H
#define LOCATION_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LOCATION_INIT_CAPACITY 4

/*
The struct that needs to be supplied to the mapperTree. 

Contains the trace (in compressed form, the raw contains the ordering of hyperplanes and the dists constains the trace distances)
Also, the point's index and the error boolean.
*/
typedef struct pointInfo {
	int * traceRaw;
	float * traceDists;
	long int index;
	int errorClass;
} pointInfo;

/*
Specilized allocation structure similar to std::vector
It's a convienent (makes memory management easier) means of storing arrays of pointInfo. 

The error class is either 1 for an error, 0 for not an error. 
*/

typedef struct location {
    
    int m;

    float *traceDists;
    int *traceRaws; 
    int *pointIndexes;
    int capacity;
    int total;

    float *traceDists_error;
    int *traceRaws_error;
    int *pointIndexes_error;
    int capacity_error;
    int total_error;
} location;


pointInfo * allocPointInfo(int m);
void freePointInfo(pointInfo *pi);

//
void location_init(location * l,int m);
int location_total(location * l, int errorClass);
int location_complete_total(location * l);
// Both the add and set operations copy the data in the point info to free up the point info buffer
void location_add(location * l, pointInfo * pi);
void location_set(location * l, int index, pointInfo * pi);
// Returns a pointInfo with references to the correct part of the location's arrays
pointInfo location_get(location * l, int index, int errorClass);
void location_free(location * l);
// Fills a given array with the indexes of either error class.
void location_get_indexes(location * l, int* indexHolder,int errorClass);


#endif